"""Dictionary API client for verified IPA lookup."""

from __future__ import annotations

from typing import Any, Protocol
from urllib.parse import quote
import requests


class HttpGetProtocol(Protocol):
    def get(self, url: str, *, headers: dict[str, str] | None = None, params: dict[str, Any] | None = None, timeout: float = 10.0) -> Any: ...


class FreeDictionaryClient:
    """Queries api.dictionaryapi.dev for verified IPA transcriptions of single English words."""

    BASE_URL = "https://api.dictionaryapi.dev/api/v2/entries/en"

    def __init__(self, timeout: float = 8.0, http_client: HttpGetProtocol | None = None) -> None:
        self.timeout = timeout
        self._http = http_client or requests.Session()

    def lookup_ipa(self, expression: str, locale: str = "en_US") -> tuple[str, str]:
        """Return (ipa, source_label) if verified IPA is found, else ("", "").

        Only queries single words (no spaces) to avoid pointless 404s on idioms/phrasal verbs.
        Never fabricates IPA.
        """
        word = (expression or "").strip()
        if not word or " " in word:
            return "", ""

        url = f"{self.BASE_URL}/{quote(word.lower())}"
        try:
            resp = self._http.get(url, timeout=self.timeout)
        except Exception:
            return "", ""

        if getattr(resp, "status_code", 500) != 200:
            return "", ""

        try:
            entries = resp.json()
        except Exception:
            return "", ""

        if not isinstance(entries, list) or not entries:
            return "", ""

        prefer_tag = "-uk" if "GB" in (locale or "").upper() else "-us"
        fallback_ipa = ""

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            top_phonetic = entry.get("phonetic")
            if isinstance(top_phonetic, str) and top_phonetic.strip() and not fallback_ipa:
                fallback_ipa = top_phonetic.strip()

            phonetics = entry.get("phonetics")
            if isinstance(phonetics, list):
                for ph in phonetics:
                    if not isinstance(ph, dict):
                        continue
                    text = ph.get("text")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    text = text.strip()
                    if not fallback_ipa:
                        fallback_ipa = text
                    audio_url = str(ph.get("audio") or "").lower()
                    if prefer_tag in audio_url:
                        return text, "dictionaryapi.dev (verified)"

        if fallback_ipa:
            return fallback_ipa, "dictionaryapi.dev (verified)"
        return "", ""

