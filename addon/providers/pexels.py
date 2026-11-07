"""Official Pexels API client for retrieving licensed illustrative photos."""

from __future__ import annotations

from typing import Any
import requests

from ..core.schemas import ImageCandidate
from .dictionary import HttpGetProtocol


class ImageProviderError(RuntimeError):
    """Raised when an image provider fails or hits rate limits."""


class PexelsClient:
    """Queries Pexels photo search API (https://api.pexels.com/v1/search)."""

    SEARCH_URL = "https://api.pexels.com/v1/search"

    def __init__(
        self,
        api_key: str,
        timeout: float = 10.0,
        http_client: HttpGetProtocol | None = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.timeout = timeout
        self._http = http_client or requests.Session()

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def search_photos(self, query: str, per_page: int = 4) -> list[ImageCandidate]:
        """Return up to `per_page` Pexels photo candidates with attribution metadata."""
        cleaned_query = (query or "").strip()
        if not cleaned_query or not self.api_key:
            return []

        headers = {"Authorization": self.api_key}
        params: dict[str, Any] = {
            "query": cleaned_query,
            "per_page": max(1, min(per_page, 10)),
            "orientation": "landscape",
        }
        try:
            resp = self._http.get(self.SEARCH_URL, headers=headers, params=params, timeout=self.timeout)
        except Exception as exc:
            raise ImageProviderError(f"Pexels network error: {exc}") from exc

        status = getattr(resp, "status_code", 500)
        if status == 401 or status == 403:
            raise ImageProviderError("Pexels API authentication failed (check Pexels API key).")
        if status == 429:
            raise ImageProviderError("Pexels API rate limit reached (HTTP 429).")
        if status >= 400:
            raise ImageProviderError(f"Pexels API returned HTTP {status}.")

        try:
            data = resp.json()
        except Exception as exc:
            raise ImageProviderError(f"Invalid JSON from Pexels API: {exc}") from exc

        photos = data.get("photos", [])
        if not isinstance(photos, list):
            return []

        candidates: list[ImageCandidate] = []
        for photo in photos:
            if not isinstance(photo, dict):
                continue
            src = photo.get("src") or {}
            if not isinstance(src, dict):
                continue
            thumb_url = str(src.get("tiny") or src.get("small") or src.get("medium") or "").strip()
            download_url = str(src.get("medium") or src.get("large") or src.get("original") or "").strip()
            if not download_url:
                continue
            photo_id = str(photo.get("id") or "").strip()
            candidates.append(
                ImageCandidate(
                    provider="pexels",
                    photo_id=photo_id,
                    thumbnail_url=thumb_url or download_url,
                    download_url=download_url,
                    page_url=str(photo.get("url") or "").strip(),
                    photographer=str(photo.get("photographer") or "Unknown").strip(),
                    photographer_url=str(photo.get("photographer_url") or "").strip(),
                    license_label="Pexels License",
                    alt_text=str(photo.get("alt") or cleaned_query).strip(),
                    width=int(photo.get("width") or 0),
                    height=int(photo.get("height") or 0),
                )
            )
        return candidates

