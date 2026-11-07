"""Zero-key fallback client using official Openverse and Wikimedia Commons APIs for existing CC/PD photos."""

from __future__ import annotations

from typing import Any
import requests

from ..core.schemas import ImageCandidate
from .dictionary import HttpGetProtocol
from .pexels import ImageProviderError


USER_AGENT = "AnkiAIVocabImporter/1.0 (https://ankiweb.net; macOS Anki Add-on)"


class OpenverseClient:
    """Queries Openverse (api.openverse.org) and Wikimedia Commons for existing licensed photos without an API key."""

    OPENVERSE_URL = "https://api.openverse.org/v1/images/"
    WIKIMEDIA_URL = "https://commons.wikimedia.org/w/api.php"

    def __init__(self, timeout: float = 10.0, http_client: HttpGetProtocol | None = None) -> None:
        self.timeout = timeout
        self._http = http_client or requests.Session()

    def search_photos(self, query: str, per_page: int = 4) -> list[ImageCandidate]:
        """Search Openverse first, then Wikimedia Commons if needed."""
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return []

        candidates = self._search_openverse(cleaned_query, per_page=per_page)
        if candidates:
            return candidates
        return self._search_wikimedia(cleaned_query, per_page=per_page)

    def _search_openverse(self, query: str, per_page: int = 4) -> list[ImageCandidate]:
        params: dict[str, Any] = {
            "q": query[:100],
            "page_size": max(1, min(per_page, 10)),
        }
        headers = {"User-Agent": USER_AGENT}
        try:
            resp = self._http.get(self.OPENVERSE_URL, headers=headers, params=params, timeout=self.timeout)
        except Exception:
            return []

        if getattr(resp, "status_code", 500) != 200:
            return []

        try:
            data = resp.json()
        except Exception:
            return []

        results = data.get("results", [])
        if not isinstance(results, list):
            return []

        candidates: list[ImageCandidate] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            download_url = str(item.get("url") or "").strip()
            thumb_url = str(item.get("thumbnail") or download_url).strip()
            if not download_url:
                continue
            ext_lower = download_url.lower().split("?")[0]
            if ext_lower.endswith((".svg", ".gif", ".tif", ".tiff")):
                continue
            photo_id = str(item.get("id") or "")[:20]
            creator = str(item.get("creator") or "Unknown").strip()
            lic = str(item.get("license") or "cc").upper()
            lic_ver = str(item.get("license_version") or "").strip()
            license_label = f"CC {lic} {lic_ver}".strip() if lic not in ("PDM", "CC0") else lic

            candidates.append(
                ImageCandidate(
                    provider="openverse",
                    photo_id=photo_id,
                    thumbnail_url=thumb_url or download_url,
                    download_url=download_url,
                    page_url=str(item.get("foreign_landing_url") or item.get("detail_url") or "").strip(),
                    photographer=creator,
                    photographer_url=str(item.get("creator_url") or "").strip(),
                    license_label=license_label,
                    alt_text=str(item.get("title") or query).strip(),
                    width=int(item.get("width") or 0),
                    height=int(item.get("height") or 0),
                )
            )
            if len(candidates) >= per_page:
                break
        return candidates

    def _search_wikimedia(self, query: str, per_page: int = 4) -> list[ImageCandidate]:
        params: dict[str, Any] = {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {query[:80]}",
            "gsrnamespace": 6,
            "gsrlimit": max(1, min(per_page * 2, 10)),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|dimensions|mime",
            "iiurlwidth": 640,
            "format": "json",
        }
        headers = {"User-Agent": USER_AGENT}
        try:
            resp = self._http.get(self.WIKIMEDIA_URL, headers=headers, params=params, timeout=self.timeout)
        except Exception as exc:
            raise ImageProviderError(f"Wikimedia search failed: {exc}") from exc

        if getattr(resp, "status_code", 500) != 200:
            return []

        try:
            pages = resp.json().get("query", {}).get("pages", {})
        except Exception:
            return []

        if not isinstance(pages, dict):
            return []

        candidates: list[ImageCandidate] = []
        for page_id, page in pages.items():
            if not isinstance(page, dict):
                continue
            info_list = page.get("imageinfo")
            if not isinstance(info_list, list) or not info_list:
                continue
            ii = info_list[0]
            if not isinstance(ii, dict):
                continue
            mime = str(ii.get("mime") or "").lower()
            if mime not in ("image/jpeg", "image/png", "image/webp"):
                continue
            thumb_url = str(ii.get("thumburl") or ii.get("url") or "").strip()
            download_url = str(ii.get("thumburl") or ii.get("url") or "").strip()
            if not download_url:
                continue
            extmeta = ii.get("extmetadata") or {}
            artist_raw = ""
            lic_short = "Wikimedia Commons"
            if isinstance(extmeta, dict):
                artist_obj = extmeta.get("Artist")
                if isinstance(artist_obj, dict):
                    artist_raw = str(artist_obj.get("value") or "")
                lic_obj = extmeta.get("LicenseShortName")
                if isinstance(lic_obj, dict) and lic_obj.get("value"):
                    lic_short = str(lic_obj.get("value"))
            # Strip HTML tags from Wikimedia artist field
            import re
            photographer = re.sub(r"<[^>]+>", "", artist_raw).strip() or "Wikimedia Contributor"

            candidates.append(
                ImageCandidate(
                    provider="wikimedia",
                    photo_id=str(page_id),
                    thumbnail_url=thumb_url,
                    download_url=download_url,
                    page_url=str(ii.get("descriptionurl") or "").strip(),
                    photographer=photographer[:40],
                    license_label=lic_short,
                    alt_text=str(page.get("title") or query).removeprefix("File:"),
                    width=int(ii.get("thumbwidth") or ii.get("width") or 0),
                    height=int(ii.get("thumbheight") or ii.get("height") or 0),
                )
            )
            if len(candidates) >= per_page:
                break
        return candidates

