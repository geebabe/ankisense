"""Unit tests for LLM schema validation, IPA provenance, Pexels/Pixabay fallback, and 50-item batches."""

from __future__ import annotations

import json
from typing import Any
import unittest

from addon.core.enrichment import EnrichmentService
from addon.core.parsing import parse_single_entry, parse_vocabulary_input
from addon.core.schemas import (
    ReviewState,
    SchemaValidationError,
    sanitize_plain_text,
    validate_enrichment_payload,
)
from addon.providers.dictionary import FreeDictionaryClient
from addon.providers.llm import OpenAICompatibleLLMClient
from addon.providers.pexels import PexelsClient
from addon.providers.pixabay import PixabayClient


class DummyResponse:
    def __init__(self, status_code: int, json_data: Any = None, content: bytes = b"", headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.content = content
        self.headers = headers or {}

    def json(self) -> Any:
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data


class MockHttpClient:
    def __init__(self, post_handler=None, get_handler=None) -> None:
        self._post_handler = post_handler
        self._get_handler = get_handler

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float) -> DummyResponse:
        return self._post_handler(url, headers=headers, json=json, timeout=timeout)

    def get(self, url: str, *, headers: dict[str, str] | None = None, params: dict[str, Any] | None = None, timeout: float = 10.0) -> DummyResponse:
        return self._get_handler(url, headers=headers, params=params, timeout=timeout)


def make_valid_llm_payload(expr: str = "serendipity", pos: str = "n", sense: str = "happy chance discovery") -> dict[str, Any]:
    return {
        "expression": expr,
        "part_of_speech": pos,
        "sense": sense,
        "definition_en": f"Definition of {expr} ({sense})",
        "meaning_vi": f"Nghĩa của {expr}",
        "examples": [
            f"Sentence one using {expr} in context.",
            f"Sentence two showing {expr} naturally.",
            f"Sentence three illustrating {expr} clearly.",
            f"Sentence four with {expr} in conversation.",
            f"Sentence five demonstrating {expr} effectively.",
        ],
        "ipa_suggestion": "/ˌser.ənˈdɪp.ə.t̬i/",
        "image_query": f"{expr} illustration photo",
        "sense_conflict": False,
        "conflict_explanation": "",
        "needs_review": False,
    }


class ProvidersAndEnrichmentTests(unittest.TestCase):
    def test_schema_validation_rejects_fewer_or_duplicate_examples(self) -> None:
        bad_count = make_valid_llm_payload()
        bad_count["examples"] = ["One", "Two", "Three"]
        with self.assertRaises(SchemaValidationError):
            validate_enrichment_payload(bad_count)

        dup_examples = make_valid_llm_payload()
        dup_examples["examples"] = ["Same", "same", "Two", "Three", "Four"]
        with self.assertRaises(SchemaValidationError):
            validate_enrichment_payload(dup_examples)

    def test_google_api_key_auto_configures_gemini_endpoint(self) -> None:
        for key in ("AIzaSyDummyGoogleKey12345", "AQ.Ab8RN6DummyGoogleKey12345"):
            client = OpenAICompatibleLLMClient(api_key=key, base_url="https://api.openai.com/v1", model="gpt-4o-mini")
            self.assertEqual(client.base_url, "https://generativelanguage.googleapis.com/v1beta")
            self.assertEqual(client.model, "gemini-3.5-flash-lite")

    def test_html_sanitization_strips_scripts_and_escapes_tags(self) -> None:
        raw = '<script>alert(1)</script><b>bold</b> & "quoted"'
        cleaned = sanitize_plain_text(raw)
        self.assertNotIn("<script>", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertIn("&lt;b&gt;bold&lt;/b&gt;", cleaned)

    def test_ipa_prefers_verified_dictionary_and_falls_back_to_unverified_llm(self) -> None:
        # 1. Single word where dictionary returns verified IPA
        dict_http = MockHttpClient(
            get_handler=lambda url, **kw: DummyResponse(
                200,
                [
                    {
                        "word": "serendipity",
                        "phonetics": [
                            {"text": "/ˌsɛɹ.ənˈdɪp.ɪ.ti/", "audio": "https://api.dictionaryapi.dev/media/pronunciations/en/serendipity-us.mp3"}
                        ],
                    }
                ],
            )
        )
        llm_http = MockHttpClient(
            post_handler=lambda url, **kw: DummyResponse(
                200,
                {"choices": [{"message": {"content": json.dumps(make_valid_llm_payload("serendipity"))}}]},
            )
        )
        svc = EnrichmentService(
            llm_client=OpenAICompatibleLLMClient(api_key="test-key", http_client=llm_http),
            dictionary_client=FreeDictionaryClient(http_client=dict_http),
        )
        enriched = svc.enrich_single(parse_single_entry("serendipity"))
        self.assertTrue(enriched.ipa_verified)
        self.assertEqual(enriched.ipa, "/ˌsɛɹ.ənˈdɪp.ɪ.ti/")
        self.assertEqual(enriched.ipa_source, "dictionaryapi.dev (verified)")

        # 2. Idiom where dictionary skips lookup and LLM suggestion is marked unverified
        llm_http_idiom = MockHttpClient(
            post_handler=lambda url, **kw: DummyResponse(
                200,
                {"choices": [{"message": {"content": json.dumps(make_valid_llm_payload("a blessing in disguise", "idiom"))}}]},
            )
        )
        svc_idiom = EnrichmentService(
            llm_client=OpenAICompatibleLLMClient(api_key="test-key", http_client=llm_http_idiom),
            dictionary_client=FreeDictionaryClient(http_client=dict_http),
        )
        enriched_idiom = svc_idiom.enrich_single(parse_single_entry("a blessing in disguise"))
        self.assertFalse(enriched_idiom.ipa_verified)
        self.assertEqual(enriched_idiom.ipa_source, "LLM (unverified)")

    def test_pexels_to_pixabay_fallback_and_missing_image_tolerance(self) -> None:
        # Pexels returns 429 rate limit -> Pixabay returns valid candidate
        pexels_http = MockHttpClient(get_handler=lambda url, **kw: DummyResponse(429, {}))
        pixabay_http = MockHttpClient(
            get_handler=lambda url, **kw: DummyResponse(
                200,
                {
                    "hits": [
                        {
                            "id": 98765,
                            "previewURL": "https://cdn.pixabay.com/preview.jpg",
                            "webformatURL": "https://cdn.pixabay.com/web.jpg",
                            "pageURL": "https://pixabay.com/photos/98765/",
                            "user": "alice_photo",
                            "user_id": 111,
                            "tags": "booking, hotel, reservation",
                        }
                    ]
                },
            )
        )
        llm_http = MockHttpClient(
            post_handler=lambda url, **kw: DummyResponse(
                200,
                {"choices": [{"message": {"content": json.dumps(make_valid_llm_payload("book", "v", "reserve"))}}]},
            )
        )
        svc = EnrichmentService(
            llm_client=OpenAICompatibleLLMClient(api_key="k", http_client=llm_http),
            dictionary_client=FreeDictionaryClient(http_client=MockHttpClient(get_handler=lambda url, **kw: DummyResponse(404))),
            pexels_client=PexelsClient(api_key="pexels-k", http_client=pexels_http),
            pixabay_client=PixabayClient(api_key="pixabay-k", http_client=pixabay_http),
        )
        enriched = svc.enrich_single(parse_single_entry("book (v)"))
        self.assertEqual(len(enriched.image_candidates), 1)
        self.assertEqual(enriched.image_candidates[0].provider, "pixabay")
        self.assertEqual(enriched.selected_image_index, 0)
        self.assertIn("Pixabay by alice_photo", enriched.build_source_field())

    def test_50_item_batch_progress_cancel_and_malformed_json_recovery(self) -> None:
        raw_50 = "\n".join(f"vocab_word_{i}" for i in range(50))
        parsed_50 = parse_vocabulary_input(raw_50)
        self.assertEqual(len(parsed_50), 50)

        call_counter = {"n": 0}

        def post_handler(url: str, **kw: Any) -> DummyResponse:
            call_counter["n"] += 1
            idx = call_counter["n"]
            # Simulate malformed JSON on item #7
            if idx == 7:
                return DummyResponse(200, {"choices": [{"message": {"content": "{not valid json"}}]})
            payload = make_valid_llm_payload(f"vocab_word_{idx - 1}", "n", f"sense {idx - 1}")
            return DummyResponse(200, {"choices": [{"message": {"content": json.dumps(payload)}}]})

        svc = EnrichmentService(
            llm_client=OpenAICompatibleLLMClient(api_key="k", http_client=MockHttpClient(post_handler=post_handler)),
            dictionary_client=FreeDictionaryClient(http_client=MockHttpClient(get_handler=lambda u, **k: DummyResponse(404))),
        )

        progress_events: list[tuple[int, int]] = []
        enriched_batch = svc.enrich_batch(
            parsed_50,
            progress_cb=lambda cur, tot, msg: progress_events.append((cur, tot)),
        )
        self.assertEqual(len(enriched_batch), 50)
        self.assertEqual(progress_events[-1], (50, 50))
        # Item #7 (index 6) recorded an error without aborting the remaining 43 items
        self.assertEqual(enriched_batch[6].parsed.review_state, ReviewState.ERROR.value)
        self.assertIn("Malformed JSON", enriched_batch[6].error_message)
        self.assertEqual(sum(1 for it in enriched_batch if not it.error_message), 49)

        # Verify cancellation stops batch cleanly
        cancel_after_5 = svc.enrich_batch(
            parsed_50,
            is_cancelled=lambda: len(progress_events) > 55,
            progress_cb=lambda cur, tot, msg: progress_events.append((cur, tot)),
        )
        self.assertLess(len(cancel_after_5), 50)

    def test_llm_retries_on_rate_limit_429(self) -> None:
        import unittest.mock
        attempts = {"count": 0}

        def post_with_429_then_success(url: str, **kw: Any) -> DummyResponse:
            attempts["count"] += 1
            if attempts["count"] == 1:
                return DummyResponse(429, {"error": "Rate limit"})
            payload = make_valid_llm_payload("verdict", "n", "decision of a jury")
            return DummyResponse(200, {"choices": [{"message": {"content": json.dumps(payload)}}]})

        client = OpenAICompatibleLLMClient(
            api_key="k",
            http_client=MockHttpClient(post_handler=post_with_429_then_success),
        )
        with unittest.mock.patch("time.sleep", return_value=None):
            result = client.enrich_item(parse_single_entry("verdict (n)"))
        self.assertEqual(result["expression"], "verdict")
        self.assertEqual(attempts["count"], 2)


if __name__ == "__main__":
    unittest.main()

