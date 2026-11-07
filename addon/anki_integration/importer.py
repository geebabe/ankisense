"""Safe, idempotent Anki note importer with duplicate detection and media persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.dedupe import compute_sense_key, mark_existing_deck_duplicates, normalize_pos, normalize_text
from ..core.schemas import DuplicatePolicy, EnrichedItem
from .media import MediaPersistenceService, MediaValidationError
from .note_type import ensure_note_type


@dataclass
class ImportSummary:
    """Result report returned after importing a batch into an Anki collection."""

    deck_name: str
    created_count: int = 0
    updated_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    item_logs: list[str] = field(default_factory=list)


def scan_existing_deck_notes(col: Any, deck_name: str) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
    """Scan existing notes in `deck_name` to build lookup maps by SenseKey and (Expression, POS)."""
    sense_key_map: dict[str, int] = {}
    expr_pos_map: dict[tuple[str, str], int] = {}

    deck_id = col.decks.id_for_name(deck_name) if hasattr(col.decks, "id_for_name") else col.decks.by_name(deck_name)
    if not deck_id:
        return sense_key_map, expr_pos_map

    # Escape double quotes in deck name for Anki search query
    escaped_deck = deck_name.replace('"', '\\"')
    try:
        note_ids = col.find_notes(f'deck:"{escaped_deck}"')
    except Exception:
        note_ids = []

    for nid in note_ids:
        try:
            note = col.get_note(nid)
            note_keys = set(note.keys())
            if "SenseKey" in note_keys and note["SenseKey"].strip():
                sense_key_map[note["SenseKey"].strip()] = int(nid)
            if "Expression" in note_keys:
                expr = normalize_text(note["Expression"])
                pos = normalize_pos(note["PartOfSpeech"]) if "PartOfSpeech" in note_keys else ""
                if expr:
                    expr_pos_map[(expr, pos)] = int(nid)
        except Exception:
            continue

    return sense_key_map, expr_pos_map


def refresh_deck_duplicate_flags(col: Any, deck_name: str, items: list[EnrichedItem]) -> None:
    """Update `is_duplicate_in_deck` and `existing_note_id` on each EnrichedItem."""
    sense_map, expr_pos_map = scan_existing_deck_notes(col, deck_name)
    mark_existing_deck_duplicates(items, sense_map, expr_pos_map)


def import_enriched_items(
    col: Any,
    deck_name: str,
    items: list[EnrichedItem],
    tts_locale: str = "en_US",
    default_duplicate_policy: str = DuplicatePolicy.SKIP.value,
    media_service: MediaPersistenceService | None = None,
) -> ImportSummary:
    """Safely import confirmed `EnrichedItem`s into `deck_name` in `col`.

    Guarantees:
    - Creates/updates owned `AI Vocabulary (v1)` note type without touching other note types.
    - Downloads and persists chosen images via `col.media.write_data`; image failures never block card import.
    - Idempotent for the same deck + SenseKey (defaults to skipping duplicates).
    """
    summary = ImportSummary(deck_name=deck_name)
    media_svc = media_service or MediaPersistenceService()

    notetype = ensure_note_type(col, tts_locale=tts_locale)
    deck_id = col.decks.id(deck_name)
    if deck_id is None:
        raise RuntimeError(f"Could not resolve or create deck '{deck_name}'")

    sense_map, expr_pos_map = scan_existing_deck_notes(col, deck_name)

    for item in items:
        if not item.parsed.selected:
            continue

        # Validate minimum required fields before importing
        if not item.expression.strip() or not item.definition_en.strip() or not item.meaning_vi.strip():
            summary.failed_count += 1
            reason = item.error_message or "Missing required Expression, DefinitionEN, or MeaningVI"
            summary.item_logs.append(f"FAILED '{item.expression or item.parsed.raw_input}': {reason}")
            continue

        # Recompute sense key so edits in preview are reflected deterministically
        item.sense_key = compute_sense_key(item.expression, item.part_of_speech, item.sense)
        expr_pos = (normalize_text(item.expression), normalize_pos(item.part_of_speech))

        existing_nid = sense_map.get(item.sense_key)
        if existing_nid is None and not item.sense.strip():
            existing_nid = expr_pos_map.get(expr_pos)

        policy = item.duplicate_policy or default_duplicate_policy
        if existing_nid is not None and policy == DuplicatePolicy.SKIP.value:
            summary.skipped_count += 1
            summary.item_logs.append(f"SKIPPED duplicate '{item.expression}' ({item.sense_key})")
            continue

        # Handle optional image download & local media persistence
        if item.selected_image is not None and not item.local_image_filename:
            try:
                stored_fname = media_svc.save_candidate_to_anki_media(
                    col=col,
                    sense_key=item.sense_key,
                    candidate=item.selected_image,
                )
                item.local_image_filename = stored_fname
            except (MediaValidationError, Exception) as exc:
                # Non-negotiable: missing/failed image must NOT prevent a valid card from being imported
                item.local_image_filename = ""
                warn = f"Image skipped ({exc})"
                if warn not in item.parsed.warnings:
                    item.parsed.warnings.append(warn)
                summary.item_logs.append(f"WARNING '{item.expression}': {warn}")

        field_values = item.to_note_fields()

        try:
            if existing_nid is not None and policy == DuplicatePolicy.OVERWRITE.value:
                note = col.get_note(existing_nid)
                for k, v in field_values.items():
                    if k in note:
                        note[k] = v
                col.update_note(note)
                summary.updated_count += 1
                summary.item_logs.append(f"UPDATED '{item.expression}' ({item.sense_key})")
            else:
                note = col.new_note(notetype)
                for k, v in field_values.items():
                    if k in note:
                        note[k] = v
                col.add_note(note, deck_id)
                nid = int(note.id)
                sense_map[item.sense_key] = nid
                expr_pos_map[expr_pos] = nid
                summary.created_count += 1
                summary.item_logs.append(f"CREATED '{item.expression}' ({item.sense_key})")
        except Exception as exc:
            summary.failed_count += 1
            summary.item_logs.append(f"FAILED '{item.expression}': {exc}")

    return summary
