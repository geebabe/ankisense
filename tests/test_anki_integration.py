"""End-to-end integration tests with Anki Collection (real Anki 26.9.3 backend when available)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any
import unittest

from addon.anki_integration.importer import import_enriched_items
from addon.anki_integration.media import MediaPersistenceService, detect_image_extension, MediaValidationError
from addon.anki_integration.note_type import ensure_note_type
from addon.core.enrichment import EnrichmentService
from addon.core.parsing import parse_vocabulary_input
from addon.core.schemas import DuplicatePolicy, NOTE_FIELDS, NOTE_TYPE_NAME
from addon.providers.dictionary import FreeDictionaryClient
from addon.providers.llm import OpenAICompatibleLLMClient
from addon.providers.pexels import PexelsClient
from tests.test_providers_and_enrichment import DummyResponse, MockHttpClient

try:
    from anki.collection import Collection as RealAnkiCollection
except ImportError:
    RealAnkiCollection = None  # type: ignore[assignment]


# Valid 1x1 JPEG magic-bytes payload for testing Anki media import
VALID_JPEG_BYTES = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\xff\xd9"
)


M1_SMOKE_FIXTURES: dict[tuple[str, str], dict[str, Any]] = {
    ("book", "v"): {
        "expression": "book",
        "part_of_speech": "v",
        "sense": "reserve accommodations, a table, or a ticket in advance",
        "definition_en": "to arrange to have a seat, room, performer, etc. at a particular time in the future",
        "meaning_vi": "đặt chỗ, đặt vé, đặt phòng trước",
        "examples": [
            "She booked a flight to Hanoi for next Monday.",
            "We should book a table at the restaurant before Friday night.",
            "He booked two seats in the front row for the concert.",
            "I have already booked a hotel room near the conference center.",
            "You can book tickets online at a discounted rate.",
        ],
        "ipa_suggestion": "/bʊk/",
        "image_query": "hotel reception booking reservation",
    },
    ("book", "n"): {
        "expression": "book",
        "part_of_speech": "n",
        "sense": "printed or digital work of pages",
        "definition_en": "a written text that can be published in printed or electronic form",
        "meaning_vi": "quyển sách, cuốn sách",
        "examples": [
            "I borrowed a fascinating history book from the public library.",
            "She always reads a chapter of her book before going to sleep.",
            "He published a new book about artificial intelligence.",
            "The bookstore on the corner sells second-hand books.",
            "Please open your books to page forty-two.",
        ],
        "ipa_suggestion": "/bʊk/",
        "image_query": "open reading book pages library",
    },
    ("serendipity", ""): {
        "expression": "serendipity",
        "part_of_speech": "n",
        "sense": "finding valuable or agreeable things by chance",
        "definition_en": "the fact of finding interesting or valuable things by chance",
        "meaning_vi": "sự tình cờ may mắn",
        "examples": [
            "Meeting my co-founder at that coffee shop was pure serendipity.",
            "Many scientific discoveries happen through a mix of preparation and serendipity.",
            "By a stroke of serendipity, she found the exact vintage edition she needed.",
            "Travel often brings moments of unexpected serendipity.",
            "Their lifelong friendship began as a happy accident of serendipity.",
        ],
        "ipa_suggestion": "/ˌser.ənˈdɪp.ə.t̬i/",
        "image_query": "happy surprise discovery",
    },
    ("run into", "phr_v"): {
        "expression": "run into",
        "part_of_speech": "phr_v",
        "sense": "meet someone unexpectedly by chance",
        "definition_en": "to meet someone you know when you are not expecting to",
        "meaning_vi": "tình cờ gặp, bắt gặp",
        "examples": [
            "I ran into my high school teacher at the supermarket yesterday.",
            "You will never guess who I ran into on the train this morning.",
            "We ran into some old friends while walking through the Old Quarter.",
            "She was worried she might run into her former boss at the conference.",
            "It is always a pleasure to run into neighbors at the weekend market.",
        ],
        "ipa_suggestion": "/rʌn ˈɪn.tuː/",
        "image_query": "two friends meeting by chance street",
    },
    ("a blessing in disguise", "idiom"): {
        "expression": "a blessing in disguise",
        "part_of_speech": "idiom",
        "sense": "an apparent misfortune that eventually has good results",
        "definition_en": "something that seems bad or unlucky at first, but results in something good happening later",
        "meaning_vi": "trong cái rủi có cái may",
        "examples": [
            "Missing that flight turned out to be a blessing in disguise.",
            "Losing his dull office job was a blessing in disguise because it pushed him to start his own studio.",
            "The rainy weather was a blessing in disguise, giving us time to rest indoors.",
            "Being rejected from that program proved to be a blessing in disguise.",
            "At the time the delay felt frustrating, but it was truly a blessing in disguise.",
        ],
        "ipa_suggestion": "",
        "image_query": "sunlight breaking through storm clouds",
    },
}


@unittest.skipIf(RealAnkiCollection is None, "Real Anki package not on PYTHONPATH")
class RealAnkiVerticalSliceTests(unittest.TestCase):
    def test_m1_vertical_slice_and_p0_acceptance_on_real_anki(self) -> None:
        raw_input = (
            "book (v)\n"
            "book (n)\n"
            "serendipity\n"
            "run into: tình cờ gặp\n"
            "a blessing in disguise"
        )
        parsed_items = parse_vocabulary_input(raw_input)
        self.assertEqual(len(parsed_items), 5)

        def llm_post(url: str, **kw: Any) -> DummyResponse:
            prompt = kw["json"]["messages"][1]["content"]
            for (expr, pos), data in M1_SMOKE_FIXTURES.items():
                if f"Expression: {expr}\n" in prompt and (not pos or f"Part of Speech: {pos}\n" in prompt):
                    return DummyResponse(200, {"choices": [{"message": {"content": json.dumps(data)}}]})
            raise AssertionError(f"Unmatched prompt: {prompt}")

        def dict_get(url: str, **kw: Any) -> DummyResponse:
            if "serendipity" in url:
                return DummyResponse(
                    200,
                    [{"word": "serendipity", "phonetics": [{"text": "/ˌser.ənˈdɪp.ə.t̬i/", "audio": "serendipity-us.mp3"}]}],
                )
            if "book" in url:
                return DummyResponse(
                    200,
                    [{"word": "book", "phonetics": [{"text": "/bʊk/", "audio": "book-us.mp3"}]}],
                )
            return DummyResponse(404)

        def pexels_get(url: str, **kw: Any) -> DummyResponse:
            q = (kw.get("params") or {}).get("query", "vocab")
            slug = q.split()[0]
            return DummyResponse(
                200,
                {
                    "photos": [
                        {
                            "id": f"px_{slug}",
                            "width": 800,
                            "height": 600,
                            "url": f"https://www.pexels.com/photo/{slug}/",
                            "photographer": "Test Photographer",
                            "photographer_url": "https://www.pexels.com/@test",
                            "alt": q,
                            "src": {
                                "tiny": f"https://images.pexels.com/{slug}_tiny.jpg",
                                "medium": f"https://images.pexels.com/{slug}_med.jpg",
                            },
                        }
                    ]
                },
            )

        def media_get(url: str, **kw: Any) -> DummyResponse:
            return DummyResponse(200, content=VALID_JPEG_BYTES, headers={"Content-Type": "image/jpeg"})

        enrichment_svc = EnrichmentService(
            llm_client=OpenAICompatibleLLMClient(api_key="test-key", http_client=MockHttpClient(post_handler=llm_post)),
            dictionary_client=FreeDictionaryClient(http_client=MockHttpClient(get_handler=dict_get)),
            pexels_client=PexelsClient(api_key="pexels-key", http_client=MockHttpClient(get_handler=pexels_get)),
        )
        media_svc = MediaPersistenceService(http_client=MockHttpClient(get_handler=media_get))

        enriched_items = enrichment_svc.enrich_batch(parsed_items, locale="en_US")
        self.assertEqual(len(enriched_items), 5)

        # User leaves 'a blessing in disguise' without an image to verify optional image handling
        enriched_items[4].selected_image_index = -1

        with tempfile.TemporaryDirectory() as tmp:
            col_path = os.path.join(tmp, "collection.anki2")
            col = RealAnkiCollection(col_path)
            try:
                # 1. Verify owned note type creation with 15 exact fields
                nt = ensure_note_type(col, tts_locale="en_US")
                self.assertEqual(nt["name"], NOTE_TYPE_NAME)
                self.assertEqual(tuple(col.models.field_names(nt)), NOTE_FIELDS)

                # 2. Import the 5 enriched items into custom deck 'English::Vocabulary'
                summary1 = import_enriched_items(
                    col=col,
                    deck_name="English::Vocabulary",
                    items=enriched_items,
                    tts_locale="en_US",
                    media_service=media_svc,
                )
                self.assertEqual(summary1.created_count, 5)
                self.assertEqual(summary1.skipped_count, 0)
                self.assertEqual(summary1.failed_count, 0)

                # 3. Verify notes, local media files, and Native TTS rendering in Anki
                nids = col.find_notes('deck:"English::Vocabulary"')
                self.assertEqual(len(nids), 5)

                for nid in nids:
                    note = col.get_note(nid)
                    card = note.cards()[0]
                    rendered = card.render_output()
                    # Front has active-recall typing input [[type:Expression]], DefinitionEN, and MeaningVI
                    self.assertIn("[[type:Expression]]", rendered.question_text)
                    self.assertIn(note["DefinitionEN"], rendered.question_text)
                    self.assertIn(note["MeaningVI"], rendered.question_text)
                    # Native TTS plays on the Back (answer) side so it doesn't spoil the typing prompt on Front
                    self.assertEqual(len(rendered.question_av_tags), 0)
                    self.assertEqual(len(rendered.answer_av_tags), 1)
                    tts_tag = rendered.answer_av_tags[0]
                    self.assertEqual(tts_tag.lang, "en_US")
                    self.assertEqual(tts_tag.field_text, note["Expression"])
                    self.assertIn(note["Example1"], rendered.answer_text)

                    if note["Expression"] != "a blessing in disguise":
                        self.assertIn("<img src=\"ai_vocab_", note["Image"])
                        # Confirm local file actually exists in collection.media
                        media_dir = Path(col.media.dir())
                        files_in_img = col.media.files_in_str(nt["id"], note["Image"])
                        self.assertEqual(len(files_in_img), 1)
                        self.assertTrue((media_dir / files_in_img[0]).exists())
                    else:
                        self.assertEqual(note["Image"], "")

                # 4. Verify Audio field mutual exclusion with Native TTS on Back side
                sample_note = col.get_note(nids[0])
                sample_note["Audio"] = "[sound:custom_pronunciation.mp3]"
                col.update_note(sample_note)
                rendered_custom_audio = sample_note.cards()[0].render_output()
                # Only 1 audio tag (SoundOrVideoTag), NOT both SoundOrVideoTag and TTSTag!
                self.assertEqual(len(rendered_custom_audio.answer_av_tags), 1)
                self.assertNotEqual(type(rendered_custom_audio.answer_av_tags[0]).__name__, "TTSTag")

                # 5. Re-importing the exact same batch defaults to skipping all 5 duplicates
                summary2 = import_enriched_items(
                    col=col,
                    deck_name="English::Vocabulary",
                    items=enriched_items,
                    tts_locale="en_US",
                    default_duplicate_policy=DuplicatePolicy.SKIP.value,
                    media_service=media_svc,
                )
                self.assertEqual(summary2.created_count, 0)
                self.assertEqual(summary2.skipped_count, 5)
                self.assertEqual(len(col.find_notes('deck:"English::Vocabulary"')), 5)

                # 6. Overwrite policy updates existing notes without creating duplicates
                enriched_items[0].meaning_vi = "đặt chỗ trước (đã cập nhật)"
                summary3 = import_enriched_items(
                    col=col,
                    deck_name="English::Vocabulary",
                    items=[enriched_items[0]],
                    tts_locale="en_US",
                    default_duplicate_policy=DuplicatePolicy.OVERWRITE.value,
                    media_service=media_svc,
                )
                self.assertEqual(summary3.updated_count, 1)
                self.assertEqual(summary3.created_count, 0)
                self.assertEqual(len(col.find_notes('deck:"English::Vocabulary"')), 5)
            finally:
                col.close()

            # 7. Reopen Anki collection and verify persistence of notes and media
            col_reopened = RealAnkiCollection(col_path)
            try:
                reopened_nids = col_reopened.find_notes('deck:"English::Vocabulary"')
                self.assertEqual(len(reopened_nids), 5)
                updated_notes = [
                    col_reopened.get_note(nid)
                    for nid in reopened_nids
                    if "đã cập nhật" in col_reopened.get_note(nid)["MeaningVI"]
                ]
                self.assertEqual(len(updated_notes), 1)
            finally:
                col_reopened.close()

    def test_disguised_html_image_is_rejected_and_card_still_imports(self) -> None:
        with self.assertRaises(MediaValidationError):
            detect_image_extension(b"<html><script>alert(1)</script></html>", "image/jpeg")

        # Also verify that if image download returns bad bytes during Anki import, the card is still created
        parsed = parse_vocabulary_input("serendipity")
        data = M1_SMOKE_FIXTURES[("serendipity", "")]
        llm_http = MockHttpClient(
            post_handler=lambda u, **k: DummyResponse(200, {"choices": [{"message": {"content": json.dumps(data)}}]})
        )
        pexels_http = MockHttpClient(
            get_handler=lambda u, **k: DummyResponse(
                200,
                {
                    "photos": [
                        {
                            "id": "bad_img",
                            "url": "https://www.pexels.com/photo/bad/",
                            "photographer": "Bob",
                            "src": {"tiny": "https://images.pexels.com/bad.jpg", "medium": "https://images.pexels.com/bad.jpg"},
                        }
                    ]
                },
            )
        )
        bad_media_http = MockHttpClient(
            get_handler=lambda u, **k: DummyResponse(200, content=b"<html>not an image</html>", headers={"Content-Type": "text/html"})
        )
        svc = EnrichmentService(
            llm_client=OpenAICompatibleLLMClient(api_key="k", http_client=llm_http),
            dictionary_client=FreeDictionaryClient(http_client=MockHttpClient(get_handler=lambda u, **k: DummyResponse(404))),
            pexels_client=PexelsClient(api_key="pk", http_client=pexels_http),
        )
        enriched = svc.enrich_batch(parsed)
        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0].selected_image_index, 0)

        with tempfile.TemporaryDirectory() as tmp:
            col = RealAnkiCollection(os.path.join(tmp, "collection.anki2"))
            try:
                summary = import_enriched_items(
                    col=col,
                    deck_name="Default",
                    items=enriched,
                    media_service=MediaPersistenceService(http_client=bad_media_http),
                )
                self.assertEqual(summary.created_count, 1)
                self.assertEqual(summary.failed_count, 0)
                nids = col.find_notes('deck:"Default"')
                self.assertEqual(len(nids), 1)
                note = col.get_note(nids[0])
                self.assertEqual(note["Image"], "")
            finally:
                col.close()

    def test_partial_import_retry_is_idempotent(self) -> None:
        parsed = parse_vocabulary_input("book (v)\nbook (n)")
        svc = EnrichmentService(
            llm_client=OpenAICompatibleLLMClient(
                api_key="k",
                http_client=MockHttpClient(
                    post_handler=lambda u, **k: DummyResponse(
                        200,
                        {
                            "choices": [
                                {
                                    "message": {
                                        "content": json.dumps(
                                            M1_SMOKE_FIXTURES[("book", "v")]
                                            if "Part of Speech: v" in k["json"]["messages"][1]["content"]
                                            else M1_SMOKE_FIXTURES[("book", "n")]
                                        )
                                    }
                                }
                            ]
                        },
                    )
                ),
            ),
            dictionary_client=FreeDictionaryClient(http_client=MockHttpClient(get_handler=lambda u, **k: DummyResponse(404))),
        )
        items = svc.enrich_batch(parsed)
        # Simulate item #2 failing enrichment initially
        saved_def = items[1].definition_en
        items[1].definition_en = ""
        items[1].error_message = "Simulated timeout"

        with tempfile.TemporaryDirectory() as tmp:
            col = RealAnkiCollection(os.path.join(tmp, "collection.anki2"))
            try:
                first_run = import_enriched_items(col=col, deck_name="Default", items=items)
                self.assertEqual(first_run.created_count, 1)
                self.assertEqual(first_run.failed_count, 1)

                # Now item #2 succeeds on retry and user re-runs import on the batch
                items[1].definition_en = saved_def
                items[1].error_message = ""
                second_run = import_enriched_items(col=col, deck_name="Default", items=items)
                self.assertEqual(second_run.created_count, 1)
                self.assertEqual(second_run.skipped_count, 1)
                self.assertEqual(second_run.failed_count, 0)
                self.assertEqual(len(col.find_notes('deck:"Default"')), 2)
            finally:
                col.close()

    def test_qt_dialog_offscreen_smoke(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from aqt.qt import QApplication
        from addon.core.secrets import SecretStore
        from addon.ui.main_dialog import VocabImporterDialog, SAMPLE_M1_INPUT

        app = QApplication.instance() or QApplication([])
        self.assertIsNotNone(app)

        store = SecretStore(use_keychain=False, fallback_getter=lambda: {}, fallback_writer=lambda c: None)
        dlg = VocabImporterDialog(
            parent=None,
            mw=None,
            config={"default_tts_locale": "en_US"},
            secret_store=store,
            on_save_config=lambda c: None,
        )
        dlg.raw_input_edit.setPlainText(SAMPLE_M1_INPUT)
        dlg._on_parse_clicked()
        self.assertEqual(len(dlg.parsed_items), 5)
        self.assertEqual(dlg.parse_table.rowCount(), 5)
        dlg.close()


if __name__ == "__main__":
    unittest.main()
