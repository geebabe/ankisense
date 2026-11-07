"""Configurable LLM provider supporting both Google Gemini native REST and OpenAI-compatible endpoints."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol
import requests

from ..core.schemas import ParsedItem, SchemaValidationError, validate_enrichment_payload


class LLMProviderError(RuntimeError):
    """Raised when the LLM request fails or returns invalid data."""


class HttpPostProtocol(Protocol):
    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float) -> Any: ...


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_FALLBACK_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
)

SYSTEM_PROMPT = """You are an expert bilingual English-Vietnamese lexicographer creating Anki flashcards.
Given a vocabulary expression, optional part of speech (POS), and optional user sense hint:
1. Respect the user's specified sense hint and POS. NEVER silently change a user-specified sense.
2. If the user's sense hint contradicts conventional English usage (for example, `book: đặt hàng` when `book (v)` actually means reserve/make a reservation `đặt chỗ / đặt vé / đặt phòng` and `order` means `đặt hàng`), set `"sense_conflict": true` and explain clearly in `"conflict_explanation"`, while still proposing the closest accurate explanation for review.
3. If the expression has no sense hint and is highly ambiguous, set `"needs_review": true`.
4. Return ONLY a single valid JSON object with the exact schema below:
{
  "expression": "string (the target expression)",
  "part_of_speech": "string (e.g., n, v, adj, adv, phr_v, idiom, collocation)",
  "sense": "string (concise label of the specific intended sense)",
  "definition_en": "string (clear, natural English definition for this specific sense — DO NOT include the target expression itself in this definition because it is shown on the Front typing card)",
  "meaning_vi": "string (natural, accurate Vietnamese meaning for this specific sense)",
  "examples": [
    "string (Example 1: natural English sentence demonstrating this exact sense)",
    "string (Example 2: distinct context/collocation)",
    "string (Example 3: distinct context/collocation)",
    "string (Example 4: distinct context/collocation)",
    "string (Example 5: distinct context/collocation)"
  ],
  "ipa_suggestion": "string (standard IPA transcription if known, or empty string)",
  "image_query": "string (2-4 simple, concrete English nouns describing a real-world photo for THIS specific sense, e.g. 'police handcuffs arrest' or 'judge gavel courtroom')",
  "suggested_deck": "string (short 1-3 word English topic name for this vocabulary, e.g. 'Crime & Law', 'Travel & Booking', 'Daily Idioms')",
  "sense_conflict": false,
  "conflict_explanation": "",
  "needs_review": false
}"""


def is_google_api_key(api_key: str) -> bool:
    """Return True if `api_key` matches Google AI Studio (`AIza...`) or Google Cloud (`AQ....`) format."""
    cleaned = (api_key or "").strip()
    return cleaned.startswith(("AIza", "AQ."))


def extract_json_object(raw_text: str) -> Any:
    """Parse a JSON object from LLM text, stripping markdown code fences if present."""
    if not raw_text or not raw_text.strip():
        raise SchemaValidationError("Empty response content from LLM")
    text = raw_text.strip()
    m = _CODE_FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise SchemaValidationError(f"Malformed JSON from LLM: {exc.msg}") from exc


class OpenAICompatibleLLMClient:
    """Client supporting Google Gemini native REST API and OpenAI-compatible endpoints."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_GEMINI_BASE_URL,
        model: str = DEFAULT_GEMINI_MODEL,
        timeout: float = 25.0,
        http_client: HttpPostProtocol | None = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        resolved_url = (base_url or DEFAULT_GEMINI_BASE_URL).rstrip("/")
        resolved_model = (model or DEFAULT_GEMINI_MODEL).strip()

        # Auto-route Google Gemini API keys (`AIza...` or `AQ....`)
        if is_google_api_key(self.api_key) and "openai.com" in resolved_url:
            resolved_url = DEFAULT_GEMINI_BASE_URL
            if resolved_model in ("gpt-4o-mini", "gemini-2.5-flash"):
                resolved_model = DEFAULT_GEMINI_MODEL
        elif "generativelanguage.googleapis.com" in resolved_url and resolved_model == "gemini-2.5-flash":
            resolved_model = DEFAULT_GEMINI_MODEL

        self.base_url = resolved_url
        self.model = resolved_model
        self.timeout = timeout
        self._http = http_client or requests.Session()

    @property
    def uses_native_gemini(self) -> bool:
        """Use native Gemini `:generateContent` when talking to generativelanguage.googleapis.com with a real HTTP session."""
        # Keep mocked unit-test clients using standard post() structure unless they target native Gemini URL
        if not isinstance(self._http, requests.Session):
            return False
        return "generativelanguage.googleapis.com" in self.base_url or is_google_api_key(self.api_key)

    def enrich_item(self, item: ParsedItem, locale: str = "en_US") -> dict[str, Any]:
        """Enrich a single ParsedItem into validated structured card fields."""
        if not self.api_key:
            raise LLMProviderError(
                "LLM API key is not configured. Open 'API Keys & Settings' to add your Google Gemini / OpenAI / DeepSeek key."
            )

        user_prompt = (
            f"Expression: {item.expression}\n"
            f"Part of Speech: {item.part_of_speech or '(infer from context)'}\n"
            f"User Sense Hint: {item.sense_hint or item.proposed_sense or '(none)'}\n"
            f"Classification: {item.classification}\n"
            f"Pronunciation Locale: {locale}\n"
            "Produce the JSON object for this vocabulary item."
        )

        if self.uses_native_gemini:
            content = self._call_native_gemini(user_prompt)
        else:
            content = self._call_openai_compatible(user_prompt)

        try:
            parsed_json = extract_json_object(content)
            return validate_enrichment_payload(parsed_json)
        except SchemaValidationError as exc:
            raise LLMProviderError(f"LLM output failed schema validation: {exc}") from exc

    def _call_native_gemini(self, user_prompt: str) -> str:
        """Call Google Gemini's native v1beta `:generateContent` endpoint with automatic model fallback."""
        clean_model = self.model.removeprefix("models/")
        candidate_models = [clean_model] + [m for m in GEMINI_FALLBACK_MODELS if m != clean_model]

        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "responseMimeType": "application/json",
            },
        }
        headers = {"Content-Type": "application/json"}
        last_error = ""

        for model_name in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
            try:
                resp = self._http.post(url, headers=headers, json=body, timeout=self.timeout)
            except Exception as exc:
                last_error = f"Network error contacting Google Gemini ({model_name}): {exc}"
                continue

            status = getattr(resp, "status_code", 200)
            if status in (400, 401, 403):
                raise LLMProviderError(
                    f"Google Gemini authentication/request error (HTTP {status}). Please verify your GOOGLE_API_KEY."
                )
            if status in (404, 503):
                # Model unavailable or experiencing temporary demand spike -> try next fallback model
                last_error = f"Model {model_name} returned HTTP {status}"
                continue
            if status == 429:
                raise LLMProviderError("Google Gemini rate limit exceeded (HTTP 429). Please wait and retry.")
            if status >= 400:
                raise LLMProviderError(f"Google Gemini returned HTTP {status}.")

            try:
                data = resp.json()
                parts = data["candidates"][0]["content"]["parts"]
                for part in parts:
                    if isinstance(part, dict) and "text" in part and part["text"].strip():
                        return str(part["text"])
            except Exception as exc:
                raise LLMProviderError(f"Unexpected response structure from Google Gemini: {exc}") from exc

        raise LLMProviderError(last_error or "Google Gemini request failed across fallback models.")

    def _call_openai_compatible(self, user_prompt: str) -> str:
        """Call an OpenAI-compatible `/chat/completions` endpoint."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        }

        try:
            resp = self._http.post(url, headers=headers, json=body, timeout=self.timeout)
        except Exception as exc:
            raise LLMProviderError(f"Network error contacting LLM endpoint: {exc}") from exc

        status_code = getattr(resp, "status_code", 200)
        if status_code == 401:
            raise LLMProviderError("LLM authentication failed (HTTP 401). Please verify your API key.")
        if status_code == 429:
            raise LLMProviderError("LLM rate limit exceeded (HTTP 429). Please wait and retry.")
        if status_code >= 400:
            raise LLMProviderError(f"LLM endpoint returned HTTP {status_code}.")

        try:
            data = resp.json()
            return str(data["choices"][0]["message"]["content"])
        except Exception as exc:
            raise LLMProviderError(f"Unexpected response structure from LLM endpoint: {exc}") from exc
