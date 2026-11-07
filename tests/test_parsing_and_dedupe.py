"""Unit tests for deterministic parsing, sense conflict detection, and deduplication."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from addon.core.dedupe import (
    compute_sense_key,
    merge_confirmed_duplicates,
)
from addon.core.parsing import (
    parse_vocabulary_input,
    read_vocab_file,
    split_raw_entries,
)
from addon.core.schemas import Classification, ReviewState, build_typing_hint


class ParsingAndDedupeTests(unittest.TestCase):
    def test_mixed_delimiters_quotes_and_blanks(self) -> None:
        raw = (
            '\ufeffbook (v), "run into: tình cờ gặp, bắt gặp",\r\n'
            "\r\n"
            "serendipity, a blessing in disguise\n"
            "'get the hang of something: hiểu cách làm, nắm bắt được'\n"
        )
        tokens = split_raw_entries(raw)
        self.assertEqual(
            tokens,
            [
                "book (v)",
                "run into: tình cờ gặp, bắt ngờ" if False else "run into: tình cờ gặp, bắt gặp",
                "serendipity",
                "a blessing in disguise",
                "get the hang of something: hiểu cách làm, nắm bắt được",
            ],
        )

    def test_read_utf8_bom_crlf_txt_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            txt_path = Path(tmp) / "vocab.txt"
            # Write UTF-8 with BOM and CRLF line endings
            content = "\ufeffbook (v), book (n)\r\nrun into: tình cờ gặp\r\na blessing in disguise\r\n"
            txt_path.write_bytes(content.encode("utf-8"))

            loaded = read_vocab_file(txt_path)
            self.assertNotIn("\ufeff", loaded)
            self.assertNotIn("\r", loaded)
            parsed = parse_vocabulary_input(loaded)
            self.assertEqual(len(parsed), 4)
            self.assertEqual(parsed[0].expression, "book")
            self.assertEqual(parsed[0].part_of_speech, "v")
            self.assertEqual(parsed[1].expression, "book")
            self.assertEqual(parsed[1].part_of_speech, "n")

    def test_book_noun_vs_verb_distinct_sense_keys(self) -> None:
        items = parse_vocabulary_input("book (n), book (v)")
        self.assertEqual(len(items), 2)
        key_n = compute_sense_key(items[0].expression, items[0].part_of_speech, items[0].proposed_sense)
        key_v = compute_sense_key(items[1].expression, items[1].part_of_speech, items[1].proposed_sense)
        self.assertNotEqual(key_n, key_v)
        self.assertIn("__n__", key_n)
        self.assertIn("__v__", key_v)

    def test_same_sense_merge_book_v_and_dat_cho(self) -> None:
        items = parse_vocabulary_input("book (v)\nbook: đặt chỗ\nbook (n)")
        self.assertEqual(len(items), 3)
        # First two share the same merge_candidate_key; third (book (n)) is distinct
        self.assertEqual(items[0].merge_candidate_key, items[1].merge_candidate_key)
        self.assertNotEqual(items[0].merge_candidate_key, items[2].merge_candidate_key)
        self.assertTrue(any("Same sense as row #1" in w for w in items[1].warnings))

        merged = merge_confirmed_duplicates(items)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].part_of_speech, "v")
        self.assertEqual(merged[0].sense_hint, "đặt chỗ")
        self.assertEqual(merged[1].part_of_speech, "n")

    def test_contradictory_sense_hint_flagged_never_silently_rewritten(self) -> None:
        items = parse_vocabulary_input("book: đặt hàng")
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item.review_state, ReviewState.CONFLICT_FLAGGED.value)
        # User's original sense hint is preserved for audit/review
        self.assertEqual(item.sense_hint, "đặt hàng")
        self.assertIn("reserve", item.conflict_explanation.lower())
        self.assertTrue(len(item.warnings) >= 1)

    def test_ambiguous_bare_word_flagged_for_review(self) -> None:
        items = parse_vocabulary_input("book")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].review_state, ReviewState.NEEDS_REVIEW.value)
        self.assertTrue(bool(items[0].proposed_sense))

    def test_classifications_word_phrasal_verb_idiom_collocation(self) -> None:
        items = parse_vocabulary_input(
            "serendipity\nrun into: tình cờ gặp\na blessing in disguise\nget the hang of something"
        )
        self.assertEqual(items[0].classification, Classification.WORD.value)
        self.assertEqual(items[1].classification, Classification.PHRASAL_VERB.value)
        self.assertEqual(items[2].classification, Classification.IDIOM.value)
        self.assertIn(
            items[3].classification,
            {Classification.COLLOCATION.value, Classification.EXPRESSION.value},
        )

    def test_build_typing_hint_masks_single_and_multiword_expressions(self) -> None:
        self.assertEqual(build_typing_hint("crime"), "c___e")
        self.assertEqual(build_typing_hint("break into"), "b___k  i__o")
        self.assertEqual(build_typing_hint("caught red-handed"), "c____t  r_d-h____d")


if __name__ == "__main__":
    unittest.main()

