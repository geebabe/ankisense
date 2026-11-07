"""Official Pixabay API fallback client for retrieving licensed illustrative photos."""

from __future__ import annotations

from typing import Any
import requests

from ..core.schemas import ImageCandidate
from .dictionary import HttpGetProtocol
from .pexels import ImageProviderError


class PixabayClient:
    """Queries Pixabay photo search API (https://pixabay.com/api/)."""

    SEARCH_URL = "https://pixabay.com/api/"

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
        """Return up to `per_page` Pixabay photo candidates with attribution metadata."""
        cleaned_query = (query or "").strip()
        if not cleaned_query or not self.api_key:
            return []

        # Pixabay requires per_page between 3 and 200
        safe_per_page = max(3, min(per_page, 20))
        params: dict[str, Any] = {
            "key": self.api_key,
            "q": cleaned_query[:100],
            "image_type": "photo",
            "safesearch": "true",
            "per_page": safe_per_page,
        }

        try:
            resp = self._http.get(self.SEARCH_URL, params=params, timeout=self.timeout)
        except Exception as exc:
            raise ImageProviderError(f"Pixabay network error: {exc}") from exc

        status = getattr(resp, "status_code", 500)
        if status == 400 or status == 401 or status == 403:
            raise ImageProviderError("Pixabay API request failed (check Pixabay API key).")
        if status == 429:
            raise ImageProviderError("Pixabay API rate limit reached (HTTP 429).")
        if status >= 400:
            raise ImageProviderError(f"Pixabay API returned HTTP {status}.")

        try:
            data = resp.json()
        except Exception as exc:
            raise ImageProviderError(f"Invalid JSON from Pixabay API: {exc}") from exc

        hits = data.get("hits", [])
        if not isinstance(hits, list):
            return []

        candidates: list[ImageCandidate] = []
        for hit in hits[:per_page]:
            if not isinstance(hit, dict):
                continue
            thumb_url = str(hit.get("previewURL") or hit.get("webformatURL") or "").strip()
            download_url = str(hit.get("webformatURL") or hit.get("largeImageURL") or "").strip()
            if not download_url:
                continue
            photo_id = str(hit.get("id") or "").strip()
            candidates.append(
                ImageCandidate(
                    provider="pixabay",
                    photo_id=photo_id,
                    thumbnail_url=thumb_url or download_url,
                    download_url=download_url,
                    page_url=str(hit.get("pageURL") or "").strip(),
                    photographer=str(hit.get("user") or "Unknown").strip(),
                    photographer_url=f"https://pixabay.com/users/{hit.get('user_id', '')}/" if hit.get("user_id") else "",
                    license_label="Pixabay Content License",
                    alt_text=str(hit.get("tags") or cleaned_query).strip(),
                    width=int(hit.get("webformatWidth") or 0),
                    height=int(hit.get("webformatHeight") or 0),
                )
            )
        return candidates

