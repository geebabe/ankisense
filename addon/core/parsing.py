"""Deterministic vocabulary input parser with quote-aware delimiter splitting and sense analysis."""

from __future__ import annotations

from pathlib import Path
import re

from .dedupe import KNOWN_SENSE_HINTS, find_batch_merge_candidates, normalize_pos, normalize_text
from .schemas import Classification, ParsedItem, ReviewState


# Trailing or inline parenthetical POS pattern, e.g. "book (v)" or "book (v): đặt chỗ"
_POS_PREFIX_RE = re.compile(
    r"^(?P<expr>.+?)\s*\(\s*(?P<pos>n|noun|v|verb|vt|vi|adj|adjective|adv|adverb|phr\.?\s*v\.?|phrasal\s+verb|idiom|collocation|prep|conj)\s*\)\s*(?::\s*(?P<hint>.+))?$",
    re.IGNORECASE,
)

# Colon-separated sense hint without explicit parenthetical POS, e.g. "run into: tình cờ gặp"
_COLON_HINT_RE = re.compile(r"^(?P<expr>[^:]+?)\s*:\s*(?P<hint>.+)$")

# Common English particles/prepositions used in phrasal verbs
_PHRASAL_PARTICLES = {
    "up", "down", "in", "out", "on", "off", "away", "back", "over",
    "into", "through", "after", "around", "about", "along", "across",
    "apart", "aside", "by", "forward", "together", "with", "for",
}

# Common verbs that frequently form phrasal verbs
_COMMON_PHRASAL_VERBS = {
    "run", "look", "get", "take", "make", "give", "put", "turn", "bring",
    "call", "come", "go", "keep", "set", "break", "carry", "catch", "check",
    "cut", "drop", "fall", "figure", "fill", "find", "hand", "hang", "hold",
    "let", "pass", "pick", "point", "pull", "Show", "stand", "throw", "work", "write",
}

# Known idioms or structural markers for idioms
_KNOWN_IDIOMS = {
    "a blessing in disguise",
    "break a leg",
    "bite the bullet",
    "under the weather",
    "spill the beans",
    "hit the nail on the head",
    "piece of cake",
    "once in a blue moon",
    "cost an arm and a leg",
    "burn the midnight oil",
    "caught red-handed",
    "catch someone red-handed",
    "above the law",
    "crime doesn't pay",
    "partner in crime",
    "on the run",
}

# Highly polysemous words that need review when entered bare without POS or sense hint
_AMBIGUOUS_BARE_WORDS: dict[str, str] = {
    "book": "printed work (noun) vs. reserve in advance (verb)",
    "bank": "financial institution (noun) vs. river slope (noun) vs. rely on (verb)",
    "run": "move fast on foot (verb) vs. manage/operate (verb)",
    "light": "illumination (noun) vs. not heavy (adjective) vs. ignite (verb)",
    "match": "contest/game (noun) vs. fire stick (noun) vs. correspond/pair (verb)",
    "date": "calendar day (noun) vs. romantic outing (noun) vs. sweet fruit (noun)",
    "spring": "season (noun) vs. coiled metal (noun) vs. jump suddenly (verb)",
    "bark": "dog sound (noun/verb) vs. tree outer layer (noun)",
    "bat": "flying mammal (noun) vs. sports club (noun)",
    "fair": "just/reasonable (adjective) vs. public exhibition/carnival (noun)",
    "charge": "formally accuse of a crime (verb/noun) vs. ask money as a price (verb) vs. electrical energy",
    "sentence": "court punishment (noun) vs. declare punishment (verb) vs. grammatical unit of words (noun)",
    "fine": "sum of money paid as penalty (noun/verb) vs. of high quality / satisfactory (adjective)",
    "convict": "declare guilty of a criminal offense (verb) vs. person serving a prison sentence (noun)",
}


def read_vocab_file(path: str | Path) -> str:
    """Read a UTF-8 or UTF-8-SIG (.txt) file and normalize line endings."""
    file_path = Path(path)
    raw_bytes = file_path.read_bytes()
    # Decode with utf-8-sig to transparently strip UTF-8 BOM if present
    text = raw_bytes.decode("utf-8-sig")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def split_raw_entries(raw_text: str) -> list[str]:
    """Split input by newlines and commas outside double quotes.

    Rules:
    - Strips UTF-8 BOM (`\\ufeff`) if present in pasted string.
    - Normalizes CRLF/CR to LF.
    - Commas inside double quotes (e.g., `"run into: gặp tình cờ, bất ngờ"`) do NOT split the entry.
    - Single quotes used as outer quotes (`'a, b'`) are also respected, while mid-word apostrophes
      (`don't`, `one's`) are treated as normal characters.
    - Surrounding outer quotes on a token are stripped after splitting.
    - Blank/whitespace-only entries are ignored.
    """
    if not raw_text:
        return []

    cleaned = raw_text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    tokens: list[str] = []
    buf: list[str] = []
    quote_char: str | None = None
    n = len(cleaned)

    for i, ch in enumerate(cleaned):
        if quote_char is not None:
            if ch == quote_char:
                quote_char = None
                buf.append(ch)
            elif ch == "\n":
                # Unclosed quote at end of line: close it implicitly and split on newline
                quote_char = None
                token = _finalize_token("".join(buf))
                if token:
                    tokens.append(token)
                buf = []
            else:
                buf.append(ch)
        else:
            if ch == '"':
                quote_char = '"'
                buf.append(ch)
            elif ch == "'" and (i == 0 or cleaned[i - 1] in (" ", "\t", "\n", ",")):
                # Only treat single quote as opening quote if at token boundary
                quote_char = "'"
                buf.append(ch)
            elif ch in (",", "\n"):
                token = _finalize_token("".join(buf))
                if token:
                    tokens.append(token)
                buf = []
            else:
                buf.append(ch)

    if buf:
        token = _finalize_token("".join(buf))
        if token:
            tokens.append(token)

    return tokens


def _finalize_token(token: str) -> str:
    stripped = token.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in ('"', "'"):
        stripped = stripped[1:-1].strip()
    return stripped


def classify_expression(expression: str, pos: str = "") -> str:
    """Classify an expression into word, phrasal_verb, idiom, collocation, or expression."""
    norm_pos = normalize_pos(pos)
    if norm_pos == "phr_v":
        return Classification.PHRASAL_VERB.value
    if norm_pos == "idiom":
        return Classification.IDIOM.value
    if norm_pos == "collocation":
        return Classification.COLLOCATION.value

    norm_expr = normalize_text(expression)
    if norm_expr in _KNOWN_IDIOMS:
        return Classification.IDIOM.value

    words = norm_expr.split()
    if len(words) <= 1:
        return Classification.WORD.value

    # Phrasal verb heuristic: 2-3 words starting with a common verb + particle
    if 2 <= len(words) <= 3 and words[0].lower() in _COMMON_PHRASAL_VERBS and words[1] in _PHRASAL_PARTICLES:
        return Classification.PHRASAL_VERB.value

    # Placeholder words like "something" / "somebody" / "one's" indicate idiom or collocation/expression
    if any(w in {"something", "somebody", "someone", "one's", "oneself", "sth", "sb"} for w in words):
        if len(words) >= 4:
            return Classification.EXPRESSION.value
        return Classification.COLLOCATION.value

    if len(words) >= 4:
        return Classification.IDIOM.value

    return Classification.COLLOCATION.value


def parse_single_entry(raw_entry: str) -> ParsedItem:
    """Parse one raw entry string into a structured `ParsedItem`."""
    raw = raw_entry.strip()
    expression = raw
    pos = ""
    sense_hint = ""

    m_pos = _POS_PREFIX_RE.match(raw)
    if m_pos:
        expression = m_pos.group("expr").strip()
        pos = normalize_pos(m_pos.group("pos"))
        sense_hint = (m_pos.group("hint") or "").strip()
    else:
        m_colon = _COLON_HINT_RE.match(raw)
        if m_colon:
            expression = m_colon.group("expr").strip()
            sense_hint = m_colon.group("hint").strip()
            # Check if expression itself has a trailing (pos)
            m_inner_pos = _POS_PREFIX_RE.match(expression)
            if m_inner_pos:
                expression = m_inner_pos.group("expr").strip()
                pos = normalize_pos(m_inner_pos.group("pos"))

    classification = classify_expression(expression, pos)
    if not pos and classification == Classification.PHRASAL_VERB.value:
        pos = "phr_v"
    elif not pos and classification == Classification.IDIOM.value:
        pos = "idiom"

    warnings: list[str] = []
    review_state = ReviewState.READY.value
    conflict_explanation = ""
    proposed_sense = ""
    confidence = 1.0

    norm_expr = normalize_text(expression)
    norm_hint = normalize_text(sense_hint)

    # 1. Check known sense hints & contradictions (e.g., `book: đặt hàng` or `book: đặt chỗ`)
    known_entries = KNOWN_SENSE_HINTS.get(norm_expr, [])
    matched_known = False
    for entry in known_entries:
        contradictions = entry.get("contradictions", {})
        if isinstance(contradictions, dict) and norm_hint in contradictions:
            review_state = ReviewState.CONFLICT_FLAGGED.value
            conflict_explanation = str(contradictions[norm_hint])
            warnings.append(f"Contradictory sense hint '{sense_hint}': {conflict_explanation}")
            confidence = 0.3
            matched_known = True
            break

        synonyms = entry.get("synonyms", set())
        if isinstance(synonyms, set) and norm_hint in synonyms:
            if not pos:
                pos = str(entry["pos"])
            proposed_sense = str(entry["canonical_sense"])
            matched_known = True
            break

        if pos and pos == str(entry["pos"]) and not sense_hint:
            proposed_sense = str(entry["canonical_sense"])
            matched_known = True
            break

    # 2. Check if bare polysemous word without POS or sense hint -> Needs Review
    if not matched_known and not pos and not sense_hint and norm_expr in _AMBIGUOUS_BARE_WORDS:
        review_state = ReviewState.NEEDS_REVIEW.value
        proposed_sense = _AMBIGUOUS_BARE_WORDS[norm_expr]
        warnings.append(
            f"Ambiguous expression '{expression}' without POS or sense hint. "
            f"Please specify intended sense ({proposed_sense})."
        )
        confidence = 0.5

    return ParsedItem(
        raw_input=raw,
        expression=expression,
        part_of_speech=pos,
        sense_hint=sense_hint,
        classification=classification,
        confidence=confidence,
        warnings=warnings,
        review_state=review_state,
        conflict_explanation=conflict_explanation,
        proposed_sense=proposed_sense,
    )


def parse_vocabulary_input(raw_text: str) -> list[ParsedItem]:
    """Parse multi-line / comma-delimited vocabulary input and annotate merge candidates."""
    raw_tokens = split_raw_entries(raw_text)
    parsed_items = [parse_single_entry(tok) for tok in raw_tokens]
    return find_batch_merge_candidates(parsed_items)

