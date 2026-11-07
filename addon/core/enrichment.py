"""Enrichment pipeline coordinating LLM, Dictionary IPA, and Pexels/Pixabay image retrieval."""

from __future__ import annotations

from typing import Callable

from ..providers.dictionary import FreeDictionaryClient
from ..providers.llm import LLMProviderError, OpenAICompatibleLLMClient
from ..providers.openverse import OpenverseClient
from ..providers.pexels import ImageProviderError, PexelsClient
from ..providers.pixabay import PixabayClient
from .dedupe import compute_sense_key, normalize_pos
from .schemas import EnrichedItem, ImageCandidate, ParsedItem, ReviewState


ProgressCallback = Callable[[int, int, str], None]
CancelCheck = Callable[[], bool]


class EnrichmentService:
    """Coordinates LLM enrichment, verified IPA lookup, and Pexels/Pixabay/Openverse image retrieval."""

    def __init__(
        self,
        llm_client: OpenAICompatibleLLMClient,
        dictionary_client: FreeDictionaryClient | None = None,
        pexels_client: PexelsClient | None = None,
        pixabay_client: PixabayClient | None = None,
        openverse_client: OpenverseClient | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.dictionary_client = dictionary_client or FreeDictionaryClient()
        self.pexels_client = pexels_client
        self.pixabay_client = pixabay_client
        self.openverse_client = openverse_client

    def enrich_single(self, item: ParsedItem, locale: str = "en_US", max_images: int = 4) -> EnrichedItem:
        """Enrich one ParsedItem. Never raises on IPA or image failure."""
        try:
            payload = self.llm_client.enrich_item(item, locale=locale)
        except LLMProviderError as exc:
            item.review_state = ReviewState.ERROR.value
            return EnrichedItem(
                parsed=item,
                expression=item.expression,
                part_of_speech=item.part_of_speech,
                sense=item.sense_hint or item.proposed_sense,
                definition_en="",
                meaning_vi="",
                sense_key=compute_sense_key(item.expression, item.part_of_speech, item.sense_hint),
                error_message=str(exc),
            )

        expr = payload["expression"] or item.expression
        pos = normalize_pos(payload["part_of_speech"] or item.part_of_speech)
        sense = payload["sense"]
        definition_en = payload["definition_en"]
        meaning_vi = payload["meaning_vi"]
        examples = list(payload["examples"])
        image_query = payload.get("image_query", "").strip()

        # Respect non-negotiable: never silently change a user-specified sense
        if payload.get("sense_conflict") or item.review_state == ReviewState.CONFLICT_FLAGGED.value:
            item.review_state = ReviewState.CONFLICT_FLAGGED.value
            explanation = payload.get("conflict_explanation") or item.conflict_explanation
            if explanation:
                item.conflict_explanation = explanation
                warn_text = f"Sense conflict flagged: {explanation}"
                if warn_text not in item.warnings:
                    item.warnings.append(warn_text)
        elif payload.get("needs_review") or item.review_state == ReviewState.NEEDS_REVIEW.value:
            item.review_state = ReviewState.NEEDS_REVIEW.value
        else:
            item.review_state = ReviewState.ENRICHED.value

        # IPA Lookup: Free Dictionary API first (for single words), fallback to LLM (marked unverified)
        ipa = ""
        ipa_source = ""
        ipa_verified = False

        if self.dictionary_client is not None:
            dict_ipa, dict_source = self.dictionary_client.lookup_ipa(expr, locale=locale)
            if dict_ipa:
                ipa = dict_ipa
                ipa_source = dict_source
                ipa_verified = True

        if not ipa and payload.get("ipa_suggestion"):
            ipa = payload["ipa_suggestion"].strip()
            ipa_source = "LLM (unverified)"
            ipa_verified = False

        # Image Search: Pexels primary -> Pixabay fallback -> Openverse/Wikimedia zero-key fallback
        search_q = image_query or expr
        candidates, img_warning = self.search_images(search_q, max_images=max_images, fallback_query=expr)
        if img_warning and img_warning not in item.warnings:
            item.warnings.append(img_warning)

        selected_idx = 0 if candidates else -1
        sense_key = compute_sense_key(expr, pos, sense)

        return EnrichedItem(
            parsed=item,
            expression=expr,
            part_of_speech=pos,
            sense=sense,
            definition_en=definition_en,
            meaning_vi=meaning_vi,
            examples=examples,
            ipa=ipa,
            ipa_source=ipa_source,
            ipa_verified=ipa_verified,
            image_query=search_q,
            image_candidates=candidates,
            selected_image_index=selected_idx,
            sense_key=sense_key,
            suggested_deck=payload.get("suggested_deck", ""),
        )

    def search_images(self, query: str, max_images: int = 4, fallback_query: str = "") -> tuple[list[ImageCandidate], str]:
        """Search Pexels first, Pixabay second, and Openverse/Wikimedia third if no API keys or no hits."""
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return [], ""

        warnings: list[str] = []

        if self.pexels_client and self.pexels_client.is_configured:
            try:
                candidates = self.pexels_client.search_photos(cleaned_query, per_page=max_images)
                if candidates:
                    return candidates, ""
            except ImageProviderError as exc:
                warnings.append(str(exc))

        if self.pixabay_client and self.pixabay_client.is_configured:
            try:
                candidates = self.pixabay_client.search_photos(cleaned_query, per_page=max_images)
                if candidates:
                    return candidates, "; ".join(warnings)
            except ImageProviderError as exc:
                warnings.append(str(exc))

        if self.openverse_client is not None:
            try:
                candidates = self.openverse_client.search_photos(cleaned_query, per_page=max_images)
                if not candidates and fallback_query and fallback_query.lower() != cleaned_query.lower():
                    candidates = self.openverse_client.search_photos(fallback_query, per_page=max_images)
                if candidates:
                    return candidates, "; ".join(warnings)
            except ImageProviderError as exc:
                warnings.append(str(exc))

        return [], "; ".join(warnings)

    def enrich_batch(
        self,
        items: list[ParsedItem],
        locale: str = "en_US",
        max_images: int = 4,
        progress_cb: ProgressCallback | None = None,
        is_cancelled: CancelCheck | None = None,
    ) -> list[EnrichedItem]:
        """Enrich a list of ParsedItems sequentially with progress and cancellation checks."""
        results: list[EnrichedItem] = []
        selected_items = [it for it in items if it.selected]
        total = len(selected_items)

        for idx, item in enumerate(selected_items):
            if is_cancelled and is_cancelled():
                break
            if progress_cb:
                progress_cb(idx, total, f"Enriching '{item.expression}' ({idx + 1}/{total})...")
            enriched = self.enrich_single(item, locale=locale, max_images=max_images)
            results.append(enriched)

        if progress_cb:
            progress_cb(len(results), total, f"Completed {len(results)}/{total} items.")
        return results

