"""Idempotent sense-key generation and duplicate detection."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable

from .schemas import EnrichedItem, ParsedItem


_WHITESPACE_RE = re.compile(r"\s+")

# Canonical POS mapping so "(v)", "verb", "v." all normalize to "v"
POS_ALIASES: dict[str, str] = {
    "n": "n",
    "noun": "n",
    "n.": "n",
    "v": "v",
    "verb": "v",
    "v.": "v",
    "vt": "v",
    "vi": "v",
    "adj": "adj",
    "adjective": "adj",
    "adj.": "adj",
    "adv": "adv",
    "adverb": "adv",
    "adv.": "adv",
    "phr v": "phr_v",
    "phrasal verb": "phr_v",
    "phr. v.": "phr_v",
    "pv": "phr_v",
    "idiom": "idiom",
    "id": "idiom",
    "collocation": "collocation",
    "coll": "collocation",
    "prep": "prep",
    "preposition": "prep",
    "conj": "conj",
    "conjunction": "conj",
}


# Known common sense alias groups used for deterministic same-sense merge suggestions
# and contradiction checks before LLM enrichment.
KNOWN_SENSE_HINTS: dict[str, list[dict[str, object]]] = {
    "book": [
        {
            "pos": "n",
            "canonical_sense": "printed or digital work of pages",
            "synonyms": {"quyển sách", "cuốn sách", "sách", "printed work", "reading material"},
            "contradictions": {"đặt hàng": "Use 'order' for 'đặt hàng'. 'book (v)' means reserve/make a reservation ('đặt chỗ / đặt vé / đặt phòng')."},
        },
        {
            "pos": "v",
            "canonical_sense": "reserve accommodations, a place, or a ticket in advance",
            "synonyms": {"đặt chỗ", "đặt vé", "đặt phòng", "đặt trước", "reserve", "make a reservation"},
            "contradictions": {"đặt hàng": "'book' as a verb means to reserve a room, ticket, or table ('đặt chỗ / đặt phòng / đặt vé'), whereas 'đặt hàng' (purchasing goods) is typically 'order'."},
        },
    ],
    "run into": [
        {
            "pos": "phr_v",
            "canonical_sense": "meet someone unexpectedly by chance",
            "synonyms": {"tình cờ gặp", "bắt gặp", "gặp tình cờ", "meet by chance", "bump into"},
            "contradictions": {},
        }
    ],
    "sentence": [
        {
            "pos": "v",
            "canonical_sense": "declare the punishment decided for an offender in court",
            "synonyms": {"tuyên án", "kết án", "pronounce sentence", "punish by court"},
            "contradictions": {
                "bắt giữ": "'sentence (v)' means to pronounce punishment in court ('tuyên án'), whereas 'bắt giữ' (take into police custody) is 'arrest'."
            },
        },
        {
            "pos": "n",
            "canonical_sense": "punishment assigned to a defendant found guilty by a court",
            "synonyms": {"bản án", "mức án", "án phạt", "court punishment"},
            "contradictions": {},
        },
    ],
    "arrest": [
        {
            "pos": "v",
            "canonical_sense": "seize someone by legal authority and take into custody",
            "synonyms": {"bắt giữ", "bắt giam", "take into custody"},
            "contradictions": {
                "tuyên án": "'arrest' means to take someone into police custody ('bắt giữ'), whereas 'tuyên án' (declare punishment in court) is 'sentence'."
            },
        }
    ],
}


def normalize_text(text: str) -> str:
    """Normalize Unicode NFC, lowercase, and collapse internal whitespace."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFC", text).strip().lower()
    return _WHITESPACE_RE.sub(" ", norm)


def normalize_pos(pos: str) -> str:
    """Normalize part-of-speech tag to a canonical short code."""
    cleaned = normalize_text(pos).strip("().")
    return POS_ALIASES.get(cleaned, cleaned)


def compute_sense_key(expression: str, part_of_speech: str = "", sense: str = "") -> str:
    """Build a deterministic, human-inspectable + hash-suffixed SenseKey.

    Guarantees:
    - `book (n)` and `book (v)` produce distinct keys.
    - Re-importing the exact same expression + POS + sense produces the same key.
    """
    norm_expr = normalize_text(expression)
    norm_pos = normalize_pos(part_of_speech) or "any"
    norm_sense = normalize_text(sense) or "default"

    # Match against known canonical senses so `book (v)` and `book: đặt chỗ` converge
    known_list = KNOWN_SENSE_HINTS.get(norm_expr, [])
    for entry in known_list:
        entry_pos = str(entry["pos"])
        synonyms = entry["synonyms"]
        canonical = str(entry["canonical_sense"])
        if norm_sense in synonyms or norm_sense == normalize_text(canonical):
            norm_sense = normalize_text(canonical)
            if norm_pos == "any":
                norm_pos = entry_pos
            break
        if norm_pos == entry_pos and norm_sense == "default":
            norm_sense = normalize_text(canonical)
            break

    slug_expr = re.sub(r"[^a-z0-9]+", "-", norm_expr).strip("-")[:32] or "item"
    slug_pos = re.sub(r"[^a-z0-9_]+", "", norm_pos)[:10] or "any"
    digest = hashlib.sha256(f"{norm_expr}|{norm_pos}|{norm_sense}".encode("utf-8")).hexdigest()[:10]
    return f"{slug_expr}__{slug_pos}__{digest}"


def find_batch_merge_candidates(items: list[ParsedItem]) -> list[ParsedItem]:
    """Identify items in the same batch that may refer to the same expression + sense.

    - `book (v)` and `book: đặt chỗ` are flagged with the same `merge_candidate_key`
      and a helpful warning so the user can merge or keep separate.
    - `book (n)` and `book (v)` remain strictly distinct.
    """
    seen_by_key: dict[str, int] = {}
    for idx, item in enumerate(items):
        candidate_key = compute_sense_key(item.expression, item.part_of_speech, item.sense_hint or item.proposed_sense)
        item.merge_candidate_key = candidate_key
        if candidate_key in seen_by_key:
            first_idx = seen_by_key[candidate_key]
            first_item = items[first_idx]
            msg = (
                f"Same sense as row #{first_idx + 1} ('{first_item.raw_input}'); "
                "can be merged if confirmed."
            )
            if msg not in item.warnings:
                item.warnings.append(msg)
        else:
            seen_by_key[candidate_key] = idx
    return items


def merge_confirmed_duplicates(items: Iterable[ParsedItem]) -> list[ParsedItem]:
    """Merge selected items that share the exact same confirmed `merge_candidate_key`."""
    merged: list[ParsedItem] = []
    by_key: dict[str, ParsedItem] = {}

    for item in items:
        if not item.selected:
            continue
        key = compute_sense_key(item.expression, item.part_of_speech, item.sense_hint or item.proposed_sense)
        if key not in by_key:
            by_key[key] = item
            merged.append(item)
        else:
            primary = by_key[key]
            if not primary.part_of_speech and item.part_of_speech:
                primary.part_of_speech = item.part_of_speech
            if not primary.sense_hint and item.sense_hint:
                primary.sense_hint = item.sense_hint
            if item.raw_input not in primary.raw_input:
                primary.raw_input = f"{primary.raw_input} / {item.raw_input}"
    return merged


def mark_existing_deck_duplicates(
    enriched_items: list[EnrichedItem],
    existing_sense_keys: dict[str, int],
    existing_expr_pos_keys: dict[tuple[str, str], int] | None = None,
) -> None:
    """Annotate `EnrichedItem`s with duplicate status against existing notes in target deck."""
    existing_expr_pos_keys = existing_expr_pos_keys or {}
    for item in enriched_items:
        if not item.sense_key:
            item.sense_key = compute_sense_key(item.expression, item.part_of_speech, item.sense)
        if item.sense_key in existing_sense_keys:
            item.is_duplicate_in_deck = True
            item.existing_note_id = existing_sense_keys[item.sense_key]
            continue
        expr_pos = (normalize_text(item.expression), normalize_pos(item.part_of_speech))
        if expr_pos in existing_expr_pos_keys and not item.parsed.sense_hint:
            item.is_duplicate_in_deck = True
            item.existing_note_id = existing_expr_pos_keys[expr_pos]
        else:
            item.is_duplicate_in_deck = False
            item.existing_note_id = None

