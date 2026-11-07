"""Safe image downloading, validation, and Anki media persistence."""

from __future__ import annotations

import re
from typing import Any
import requests

from ..core.schemas import ImageCandidate
from ..providers.dictionary import HttpGetProtocol


MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MB safety limit

_ALLOWED_MIME_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

_SAFE_TOKEN_RE = re.compile(r"[^a-zA-Z0-9_-]+")


class MediaValidationError(ValueError):
    """Raised when a downloaded image fails content-type, magic-bytes, or size checks."""


def detect_image_extension(data: bytes, content_type: str = "") -> str:
    """Validate image magic bytes and return a safe extension (.jpg, .png, or .webp)."""
    if not data:
        raise MediaValidationError("Empty image payload")
    if len(data) > MAX_IMAGE_BYTES:
        raise MediaValidationError(f"Image exceeds maximum allowed size ({MAX_IMAGE_BYTES} bytes)")

    # Check magic bytes first (prevents disguised HTML/scripts)
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"

    mime = (content_type or "").split(";")[0].strip().lower()
    if mime in _ALLOWED_MIME_TO_EXT:
        raise MediaValidationError(f"Header claimed '{mime}' but file magic bytes are not a valid JPEG/PNG/WebP image")
    raise MediaValidationError(f"Unsupported or unsafe image content type: '{mime or 'unknown'}'")


def build_safe_media_filename(sense_key: str, candidate: ImageCandidate, ext: str) -> str:
    """Build a deterministic, collision-resistant filename for Anki media."""
    safe_sense = _SAFE_TOKEN_RE.sub("_", sense_key or "vocab").strip("_")[:36]
    safe_prov = _SAFE_TOKEN_RE.sub("_", candidate.provider or "img").strip("_")[:12]
    safe_id = _SAFE_TOKEN_RE.sub("_", candidate.photo_id or "0").strip("_")[:20]
    clean_ext = ext if ext.startswith(".") else f".{ext}"
    return f"ai_vocab_{safe_sense}_{safe_prov}_{safe_id}{clean_ext}"


class MediaPersistenceService:
    """Downloads verified image candidates and saves them into Anki's collection.media."""

    def __init__(self, timeout: float = 12.0, http_client: HttpGetProtocol | None = None) -> None:
        self.timeout = timeout
        if http_client is not None:
            self._http = http_client
        else:
            session = requests.Session()
            session.headers.update({"User-Agent": "AnkiAIVocabImporter/1.0 (https://ankiweb.net; macOS Anki Add-on)"})
            self._http = session

    def download_image_bytes(self, url: str) -> tuple[bytes, str]:
        """Download image bytes from HTTPS/HTTP URL and validate size and magic bytes."""
        cleaned_url = (url or "").strip()
        if not cleaned_url.startswith(("https://", "http://")):
            raise MediaValidationError("Image URL must use HTTPS or HTTP")

        try:
            resp = self._http.get(cleaned_url, timeout=self.timeout)
        except Exception as exc:
            raise MediaValidationError(f"Failed to download image: {exc}") from exc

        status = getattr(resp, "status_code", 500)
        if status != 200:
            raise MediaValidationError(f"Image download returned HTTP {status}")

        content = getattr(resp, "content", b"")
        headers = getattr(resp, "headers", {}) or {}
        content_type = str(headers.get("Content-Type") or headers.get("content-type") or "")

        ext = detect_image_extension(content, content_type)
        return content, ext

    def prefetch_candidate_image(self, candidate: ImageCandidate) -> tuple[bytes, str]:
        """Download and validate candidate photo off the UI thread, caching bytes on `candidate`."""
        if candidate.downloaded_bytes and candidate.downloaded_ext:
            return candidate.downloaded_bytes, candidate.downloaded_ext
        data, ext = self.download_image_bytes(candidate.download_url)
        candidate.downloaded_bytes = data
        candidate.downloaded_ext = ext
        return data, ext

    def save_candidate_to_anki_media(
        self,
        col: Any,
        sense_key: str,
        candidate: ImageCandidate,
    ) -> str:
        """Persist candidate photo (using pre-downloaded bytes if available) to `col.media`."""
        if candidate.downloaded_bytes and candidate.downloaded_ext:
            data = candidate.downloaded_bytes
            ext = detect_image_extension(data)
        else:
            data, ext = self.download_image_bytes(candidate.download_url)
        desired_name = build_safe_media_filename(sense_key, candidate, ext)
        stored_name = col.media.write_data(desired_name, data)
        return str(stored_name or desired_name)

    def fetch_thumbnail(self, candidate: ImageCandidate) -> bytes | None:
        """Fetch thumbnail bytes for UI preview; returns None on any error."""
        if candidate.thumbnail_bytes:
            return candidate.thumbnail_bytes
        url = candidate.thumbnail_url or candidate.download_url
        if not url:
            return None
        try:
            data, _ = self.download_image_bytes(url)
            candidate.thumbnail_bytes = data
            return data
        except Exception:
            return None

