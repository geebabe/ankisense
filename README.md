# AI Vocabulary Importer for Anki (macOS)

A local Anki Desktop add-on for macOS that turns pasted English vocabulary or UTF-8 `.txt` lists into sense-specific bilingual (English–Vietnamese) flashcards with five natural English examples, verified IPA when available, native Anki TTS, and existing licensed photos from Pexels (primary) and Pixabay (fallback).

---

## Features & Non-Negotiables

- **Local Anki Add-on:** Runs inside Anki Desktop (`Tools → AI Vocabulary Importer...`) using `aqt.qt` and supported `anki` collection APIs.
- **Deterministic Mixed-Delimiter Parser:** Splits by newlines and commas outside quotes, handles UTF-8 BOM and CRLF line endings, extracts explicit POS (`book (v)`, `book (n)`) and sense hints (`run into: tình cờ gặp`), flags contradictory hints (`book: đặt hàng`) without silently overwriting user intent, and identifies same-sense merge candidates (`book (v)` + `book: đặt chỗ`).
- **Validated Enrichment & Trusted IPA:**
  - Uses any OpenAI-compatible chat completions endpoint (OpenAI, DeepSeek, etc.) with strict JSON schema validation (requiring 5 distinct natural examples).
  - Queries **Free Dictionary API** (`api.dictionaryapi.dev`) first for verified single-word IPA transcriptions; falls back to LLM-suggested IPA explicitly marked `LLM (unverified)` in preview.
- **Existing Licensed Photos Only:**
  - Queries **Pexels API** primary and **Pixabay API** fallback (never generates AI images or scrapes Google Images).
  - Displays candidate thumbnails with photographer and license attribution; downloads and validates chosen photos (JPEG/PNG/WebP magic bytes & size checks) into `collection.media` via `col.media.write_data()`.
  - Missing or failed images never block card creation.
- **Owned Versioned Note Type (`AI Vocabulary (v1)`):**
  - Automatically creates/maintains `AI Vocabulary (v1)` with 15 ordered fields: `Expression`, `PartOfSpeech`, `IPA`, `DefinitionEN`, `MeaningVI`, `Example1`..`Example5`, `Image`, `Audio`, `Sense`, `Source`, `SenseKey`.
  - Uses native Anki TTS (`{{#Audio}}{{Audio}}{{/Audio}}{{^Audio}}{{tts en_US:Expression}}{{/Audio}}`) so stored audio and native TTS never play simultaneously.
- **Idempotent Import & Deduplication:**
  - Computes a deterministic `SenseKey` (`<slug>__<pos>__<hash>`) so `book (n)` and `book (v)` remain distinct while re-importing the same batch defaults to skipping duplicates safely.

---

## Installation (macOS)

### Option A: Development Symlink (Already Installed)
Run the packaging script with `--symlink`:
```bash
python3 scripts/package_and_install.py --symlink
```
This links `addon/` directly to `~/Library/Application Support/Anki2/addons21/anki_ai_vocab_importer`. Restart Anki Desktop to load the add-on.

### Option B: `.ankiaddon` Package File
1. Build the bundle:
   ```bash
   python3 scripts/package_and_install.py
   ```
2. In Anki Desktop, open **Tools → Add-ons → Install from file...** and select `dist/anki_ai_vocab_importer.ankiaddon`.

---

## Configuring API Keys (Google Gemini, OpenAI, or DeepSeek)

1. Open Anki Desktop and choose **Tools → AI Vocabulary Importer...**.
2. Click **API Keys & Settings...** in the top-right corner.
3. Choose a preset or paste your key:
   - **Google Gemini:** Click **Preset: Google Gemini** (sets Base URL to `https://generativelanguage.googleapis.com/v1beta/openai` and Model to `gemini-2.5-flash`) and paste your Google AI Studio key (`AIza...`). Even if you forget to click the preset button, pasting any `AIza...` key automatically configures the Google Gemini endpoint for you.
   - **OpenAI:** Click **Preset: OpenAI** (`https://api.openai.com/v1`, `gpt-4o-mini`) and paste your `sk-...` key.
   - **DeepSeek:** Click **Preset: DeepSeek** (`https://api.deepseek.com/v1`, `deepseek-chat`) and paste your key.
4. Optionally enter your **Pexels API Key** (primary image source) and **Pixabay API Key** (fallback image source).
5. Click **Save Settings**. Keys are stored in your macOS Keychain under service `anki-ai-vocab-importer` via `/usr/bin/security`.

---

## Running Automated Tests

Run the full unit, headless Anki 26.9.3 `Collection`, and offscreen Qt6 test suite:

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=/Applications/Anki.app/Contents/Resources/app_packages:. uv run --python 3.13 python -m unittest discover -s tests -v
```

---

## Documentation & Architecture

For in-depth architectural details, completed components, and future roadmap, see:
- [`docs/CONTEXT.md`](docs/CONTEXT.md) — Comprehensive architecture, implementation log, and future roadmap.
- [`docs/SPEC.md`](docs/SPEC.md) — Product specification and non-negotiable requirements.
- [`docs/TASKS.md`](docs/TASKS.md) — Milestone checklist.
- [`docs/AGENTS.md`](docs/AGENTS.md) — Agent development instructions and guidelines.

---

## Manual Anki Desktop & Sync Smoke-Test Checklist

Because live GUI audio playback and personal AnkiWeb account synchronization require interactive user verification in Anki Desktop, complete the following manual checks after restarting Anki:

- [ ] **Launch Add-on:** Open **Tools → AI Vocabulary Importer...** in Anki Desktop (`v26.9.3`).
- [ ] **Load 5-Item Smoke Sample:** Click **Load 5-Item Sample** (`book (v)`, `book (n)`, `serendipity`, `run into: tình cờ gặp`, `a blessing in disguise`) and click **1. Parse Vocabulary**.
- [ ] **Enrich & Review:** Click **2. Generate & Enrich Selected →**, verify all 5 cards populate in Tab 2, inspect candidate thumbnails, and toggle **Front** / **Back** live preview.
- [ ] **Import & Study TTS:** Click **Import to Anki**, open the target deck in Anki's Reviewer, and confirm native macOS TTS speaks the expression on the front side and shows bilingual definitions, image, and 5 examples on the back.
- [ ] **Reopen & AnkiWeb Sync:** Quit and reopen Anki Desktop, verify notes and local media (`ai_vocab_*`) persist in `Check Media`, and run **Sync** with AnkiWeb to verify synced playback on mobile/other clients.

