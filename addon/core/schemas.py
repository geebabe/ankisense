"""Core data structures and schema validation for Anki AI Vocabulary Importer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import html
import re
from typing import Any


NOTE_TYPE_NAME = "AI Vocabulary (v1)"

NOTE_FIELDS: tuple[str, ...] = (
    "Expression",
    "PartOfSpeech",
    "IPA",
    "DefinitionEN",
    "MeaningVI",
    "Example1",
    "Example2",
    "Example3",
    "Example4",
    "Example5",
    "Image",
    "Audio",
    "Sense",
    "Source",
    "SenseKey",
)


class Classification(str, Enum):
    WORD = "word"
    PHRASAL_VERB = "phrasal_verb"
    IDIOM = "idiom"
    COLLOCATION = "collocation"
    EXPRESSION = "expression"


class ReviewState(str, Enum):
    READY = "ready"
    NEEDS_REVIEW = "needs_review"
    CONFLICT_FLAGGED = "conflict_flagged"
    MERGED = "merged"
    ENRICHED = "enriched"
    ERROR = "error"


class DuplicatePolicy(str, Enum):
    SKIP = "skip"
    OVERWRITE = "overwrite"


@dataclass
class ParsedItem:
    """Represents a single vocabulary entry after deterministic + optional LLM parsing."""

    raw_input: str
    expression: str
    part_of_speech: str = ""
    sense_hint: str = ""
    classification: str = Classification.WORD.value
    confidence: float = 1.0
    warnings: list[str] = field(default_factory=list)
    review_state: str = ReviewState.READY.value
    conflict_explanation: str = ""
    proposed_sense: str = ""
    merge_candidate_key: str = ""
    selected: bool = True


@dataclass
class ImageCandidate:
    """Represents an existing licensed photo candidate from Pexels or Pixabay."""

    provider: str  # "pexels" | "pixabay"
    photo_id: str
    thumbnail_url: str
    download_url: str
    page_url: str
    photographer: str
    photographer_url: str = ""
    license_label: str = ""
    alt_text: str = ""
    width: int = 0
    height: int = 0
    thumbnail_bytes: bytes | None = None
    downloaded_bytes: bytes | None = None
    downloaded_ext: str = ""

    def format_attribution(self) -> str:
        """Return human-readable provenance string for the Source field."""
        provider_title = self.provider.capitalize()
        parts = [f"Image: {provider_title}"]
        if self.photographer:
            parts.append(f"by {self.photographer}")
        if self.page_url:
            parts.append(f"({self.page_url})")
        elif self.photo_id:
            parts.append(f"(id:{self.photo_id})")
        if self.license_label:
            parts.append(f"[{self.license_label}]")
        return " ".join(parts)


@dataclass
class EnrichedItem:
    """Full enriched card data ready for preview, editing, and Anki import."""

    parsed: ParsedItem
    expression: str
    part_of_speech: str
    sense: str
    definition_en: str
    meaning_vi: str
    examples: list[str] = field(default_factory=lambda: ["", "", "", "", ""])
    ipa: str = ""
    ipa_source: str = ""  # e.g., "dictionaryapi.dev (verified)" or "LLM (unverified)"
    ipa_verified: bool = False
    image_query: str = ""
    image_candidates: list[ImageCandidate] = field(default_factory=list)
    selected_image_index: int = -1  # -1 means "No image"
    local_image_filename: str = ""
    audio_field: str = ""  # Reserved for [sound:...] when user opts out of native TTS
    sense_key: str = ""
    duplicate_policy: str = ""  # Optional per-item override; defaults to batch policy
    is_duplicate_in_deck: bool = False
    existing_note_id: int | None = None
    error_message: str = ""
    suggested_deck: str = ""

    @property
    def selected_image(self) -> ImageCandidate | None:
        if 0 <= self.selected_image_index < len(self.image_candidates):
            return self.image_candidates[self.selected_image_index]
        return None

    def build_source_field(self) -> str:
        """Combine IPA provenance and image attribution into the Source field."""
        segments: list[str] = []
        if self.ipa and self.ipa_source:
            segments.append(f"IPA: {self.ipa_source}")
        img = self.selected_image
        if img is not None:
            segments.append(img.format_attribution())
        return " | ".join(segments)

    def build_image_html(self) -> str:
        """Return safe HTML <img> tag referencing only the local Anki media filename."""
        if not self.local_image_filename:
            return ""
        safe_fname = html.escape(self.local_image_filename, quote=True)
        img = self.selected_image
        alt = html.escape(img.alt_text or self.expression, quote=True) if img else html.escape(self.expression, quote=True)
        title = html.escape(img.format_attribution(), quote=True) if img else ""
        provider = html.escape(img.provider, quote=True) if img else ""
        photo_id = html.escape(img.photo_id, quote=True) if img else ""
        attrs = [f'src="{safe_fname}"', f'alt="{alt}"']
        if title:
            attrs.append(f'title="{title}"')
        if provider:
            attrs.append(f'data-provider="{provider}"')
        if photo_id:
            attrs.append(f'data-photo-id="{photo_id}"')
        return f"<img {' '.join(attrs)}>"

    def to_note_fields(self) -> dict[str, str]:
        """Serialize to the exact 15 fields of AI Vocabulary (v1), HTML-sanitized."""
        padded_examples = (list(self.examples) + ["", "", "", "", ""])[:5]
        return {
            "Expression": sanitize_plain_text(self.expression),
            "PartOfSpeech": sanitize_plain_text(self.part_of_speech),
            "IPA": sanitize_plain_text(self.ipa),
            "DefinitionEN": sanitize_plain_text(self.definition_en),
            "MeaningVI": sanitize_plain_text(self.meaning_vi),
            "Example1": sanitize_plain_text(padded_examples[0]),
            "Example2": sanitize_plain_text(padded_examples[1]),
            "Example3": sanitize_plain_text(padded_examples[2]),
            "Example4": sanitize_plain_text(padded_examples[3]),
            "Example5": sanitize_plain_text(padded_examples[4]),
            "Image": self.build_image_html(),
            "Audio": validate_audio_tag(self.audio_field),
            "Sense": sanitize_plain_text(self.sense),
            "Source": sanitize_plain_text(self.build_source_field()),
            "SenseKey": sanitize_plain_text(self.sense_key),
        }


_SCRIPT_TAG_RE = re.compile(r"<\s*/?\s*(script|iframe|object|embed|style|link|meta)\b[^>]*>", re.IGNORECASE)
_SOUND_TAG_RE = re.compile(r"^\[sound:[^\[\]\r\n]+\]$")


def sanitize_plain_text(value: str) -> str:
    """Escape HTML special characters so LLM or user text cannot inject arbitrary HTML/JS."""
    if not value:
        return ""
    cleaned = _SCRIPT_TAG_RE.sub("", value.strip())
    return html.escape(cleaned, quote=False)


def _mask_single_word(word: str) -> str:
    """Mask a single alphabetic token as first_letter + underscores + last_letter (e.g. crime -> c___e)."""
    if not word:
        return ""
    # Preserve leading/trailing punctuation if any
    prefix = ""
    suffix = ""
    core = word
    while core and not core[0].isalnum():
        prefix += core[0]
        core = core[1:]
    while core and not core[-1].isalnum():
        suffix = core[-1] + suffix
        core = core[:-1]

    if not core:
        return word
    n = len(core)
    if n == 1:
        masked = core
    elif n == 2:
        masked = f"{core[0]}_"
    else:
        masked = f"{core[0]}{'_' * (n - 2)}{core[-1]}"
    return f"{prefix}{masked}{suffix}"


def build_typing_hint(expression: str) -> str:
    """Build a masked typing hint like 'c___e' or 'b___k  i__o' without character counts."""
    cleaned = (expression or "").strip()
    if not cleaned:
        return ""
    words = cleaned.split()
    masked_words: list[str] = []

    for w in words:
        if "-" in w:
            parts = w.split("-")
            masked_words.append("-".join(_mask_single_word(p) for p in parts))
        else:
            masked_words.append(_mask_single_word(w))

    return "  ".join(masked_words)


def validate_audio_tag(value: str) -> str:
    """Allow only valid Anki [sound:...] references in the Audio field."""
    if not value:
        return ""
    stripped = value.strip()
    if _SOUND_TAG_RE.match(stripped):
        return stripped
    return ""


class SchemaValidationError(ValueError):
    """Raised when LLM output fails structured schema validation."""


def validate_enrichment_payload(payload: Any) -> dict[str, Any]:
    """Validate and normalize a single item's LLM enrichment JSON object.

    Expected keys:
      - expression (str, non-empty)
      - part_of_speech (str)
      - sense (str, non-empty)
      - definition_en (str, non-empty)
      - meaning_vi (str, non-empty)
      - examples (list of exactly 5 non-empty strings)
      - ipa_suggestion (optional str)
      - image_query (optional str)
      - sense_conflict (optional bool)
      - conflict_explanation (optional str)
      - needs_review (optional bool)
    """
    if not isinstance(payload, dict):
        raise SchemaValidationError(f"Expected JSON object from LLM, got {type(payload).__name__}")

    required_non_empty_str = ("expression", "sense", "definition_en", "meaning_vi")
    for key in required_non_empty_str:
        val = payload.get(key)
        if not isinstance(val, str) or not val.strip():
            raise SchemaValidationError(f"Missing or empty required string field '{key}'")

    pos = payload.get("part_of_speech", "")
    if pos is None:
        pos = ""
    if not isinstance(pos, str):
        raise SchemaValidationError("'part_of_speech' must be a string")

    examples = payload.get("examples")
    if not isinstance(examples, list) or len(examples) != 5:
        count = len(examples) if isinstance(examples, list) else "non-list"
        raise SchemaValidationError(f"'examples' must be a list of exactly 5 strings (got {count})")

    cleaned_examples: list[str] = []
    for idx, ex in enumerate(examples):
        if not isinstance(ex, str) or not ex.strip():
            raise SchemaValidationError(f"Example at index {idx} must be a non-empty string")
        cleaned_examples.append(ex.strip())

    # Ensure the 5 examples are distinct
    normalized_set = {ex.lower() for ex in cleaned_examples}
    if len(normalized_set) < 5:
        raise SchemaValidationError("All 5 examples must be distinct sentences")

    ipa_suggestion = payload.get("ipa_suggestion") or payload.get("ipa") or ""
    if not isinstance(ipa_suggestion, str):
        ipa_suggestion = ""

    image_query = payload.get("image_query") or ""
    if not isinstance(image_query, str):
        image_query = ""

    sense_conflict = bool(payload.get("sense_conflict", False))
    conflict_explanation = payload.get("conflict_explanation") or ""
    if not isinstance(conflict_explanation, str):
        conflict_explanation = str(conflict_explanation)

    needs_review = bool(payload.get("needs_review", False))
    suggested_deck = payload.get("suggested_deck") or ""
    if not isinstance(suggested_deck, str):
        suggested_deck = ""

    return {
        "expression": payload["expression"].strip(),
        "part_of_speech": pos.strip(),
        "sense": payload["sense"].strip(),
        "definition_en": payload["definition_en"].strip(),
        "meaning_vi": payload["meaning_vi"].strip(),
        "examples": cleaned_examples,
        "ipa_suggestion": ipa_suggestion.strip(),
        "image_query": image_query.strip(),
        "sense_conflict": sense_conflict,
        "conflict_explanation": conflict_explanation.strip(),
        "needs_review": needs_review,
        "suggested_deck": suggested_deck.strip(),
    }
