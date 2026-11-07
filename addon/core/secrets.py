"""Secure API key storage using macOS Keychain (/usr/bin/security) with .env and config fallback."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import subprocess
from typing import Callable


KEYCHAIN_SERVICE = "anki-ai-vocab-importer"
SECRET_KEYS = ("llm_api_key", "pexels_api_key", "pixabay_api_key")

_ENV_VAR_ALIASES: dict[str, tuple[str, ...]] = {
    "llm_api_key": ("GOOGLE_API_KEY", "GEMINI_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"),
    "pexels_api_key": ("PEXELS_API_KEY",),
    "pixabay_api_key": ("PIXABAY_API_KEY", "PIXABABY_API_KEY"),
}


def load_dotenv_secrets(dotenv_path: Path | None = None) -> dict[str, str]:
    """Read key=value pairs from .env file in project root (if present) without logging values."""
    candidates: list[Path] = []
    if dotenv_path is not None:
        candidates.append(dotenv_path)
    # addon/core/secrets.py -> addon/ -> repo root
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidates.append(repo_root / ".env")
    candidates.append(Path(__file__).resolve().parent.parent / ".env")

    parsed: dict[str, str] = {}
    for path in candidates:
        try:
            if not path.is_file():
                continue
            for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    clean_k = k.strip()
                    clean_v = v.strip().strip("\"'").strip()
                    if clean_k and clean_v:
                        parsed[clean_k] = clean_v
        except Exception:
            continue
    return parsed


class SecretStore:
    """Manages API credentials in macOS Keychain with .env and config fallback."""

    def __init__(
        self,
        service_name: str = KEYCHAIN_SERVICE,
        use_keychain: bool | None = None,
        fallback_getter: Callable[[], dict[str, object]] | None = None,
        fallback_writer: Callable[[dict[str, object]], None] | None = None,
        dotenv_path: Path | None = None,
    ) -> None:
        self.service_name = service_name
        if use_keychain is None:
            self.use_keychain = platform.system() == "Darwin"
        else:
            self.use_keychain = use_keychain
        self._fallback_getter = fallback_getter
        self._fallback_writer = fallback_writer
        self._dotenv_path = dotenv_path
        self._memory_cache: dict[str, str] = {}

    def get_secret(self, account: str) -> str:
        """Retrieve a secret by account key (checks memory -> .env/env -> Keychain -> config)."""
        if account in self._memory_cache and self._memory_cache[account]:
            return self._memory_cache[account]

        # 1. Check .env file and process environment variables first so updating .env works immediately
        dotenv_vals = load_dotenv_secrets(self._dotenv_path)
        for env_name in _ENV_VAR_ALIASES.get(account, (account.upper(),)):
            val = dotenv_vals.get(env_name) or os.environ.get(env_name) or ""
            if val.strip():
                cleaned = val.strip()
                self._memory_cache[account] = cleaned
                return cleaned

        # 2. Check macOS Keychain
        if self.use_keychain:
            val = self._read_macos_keychain(account)
            if val:
                self._memory_cache[account] = val
                return val

        # 3. Check Anki add-on config fallback
        if self._fallback_getter is not None:
            cfg = self._fallback_getter() or {}
            raw = cfg.get(account, "")
            if isinstance(raw, str) and raw.strip():
                self._memory_cache[account] = raw.strip()
                return raw.strip()

        return ""

    def set_secret(self, account: str, secret_value: str) -> bool:
        """Store a secret in macOS Keychain (or fallback config if keychain fails)."""
        cleaned = (secret_value or "").strip()
        self._memory_cache[account] = cleaned

        if self.use_keychain:
            if not cleaned:
                self._delete_macos_keychain(account)
                return True
            if self._write_macos_keychain(account, cleaned):
                if self._fallback_getter and self._fallback_writer:
                    cfg = dict(self._fallback_getter() or {})
                    if cfg.get(account):
                        cfg[account] = ""
                        self._fallback_writer(cfg)
                return True

        if self._fallback_getter and self._fallback_writer:
            cfg = dict(self._fallback_getter() or {})
            cfg[account] = cleaned
            self._fallback_writer(cfg)
            return True

        return False

    def _read_macos_keychain(self, account: str) -> str:
        try:
            proc = subprocess.run(
                [
                    "/usr/bin/security",
                    "find-generic-password",
                    "-s",
                    self.service_name,
                    "-a",
                    account,
                    "-w",
                ],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
            if proc.returncode == 0:
                return proc.stdout.strip()
        except Exception:
            pass
        return ""

    def _write_macos_keychain(self, account: str, secret_value: str) -> bool:
        try:
            proc = subprocess.run(
                [
                    "/usr/bin/security",
                    "add-generic-password",
                    "-U",
                    "-s",
                    self.service_name,
                    "-a",
                    account,
                    "-w",
                    secret_value,
                ],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
            return proc.returncode == 0
        except Exception:
            return False

    def _delete_macos_keychain(self, account: str) -> None:
        try:
            subprocess.run(
                [
                    "/usr/bin/security",
                    "delete-generic-password",
                    "-s",
                    self.service_name,
                    "-a",
                    account,
                ],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
        except Exception:
            pass


def redact_secret(value: str) -> str:
    """Return a masked representation safe for UI status display."""
    if not value:
        return "(not set)"
    if len(value) <= 8:
        return "••••••••"
    return f"{value[:4]}••••{value[-4:]}"
