"""Management of the add-on owned 'AI Vocabulary (v1)' note type and card preview rendering."""

from __future__ import annotations

import base64
import html
from pathlib import Path
import re
from typing import Any

from ..core.schemas import NOTE_FIELDS, NOTE_TYPE_NAME, EnrichedItem, build_typing_hint


_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
_LOCALE_RE = re.compile(r"^[a-z]{2,3}_[A-Z]{2}$")


def sanitize_tts_locale(locale: str) -> str:
    """Validate TTS locale code (e.g. en_US, en_GB, en_AU); default to en_US."""
    cleaned = (locale or "").strip()
    if _LOCALE_RE.match(cleaned):
        return cleaned
    return "en_US"


def load_template_assets(tts_locale: str = "en_US") -> tuple[str, str, str]:
    """Load front.html, back.html, and style.css with the requested TTS locale."""
    safe_locale = sanitize_tts_locale(tts_locale)
    front_html = (_TEMPLATES_DIR / "front.html").read_text(encoding="utf-8")
    back_html = (_TEMPLATES_DIR / "back.html").read_text(encoding="utf-8")
    style_css = (_TEMPLATES_DIR / "style.css").read_text(encoding="utf-8")

    if safe_locale != "en_US":
        front_html = front_html.replace("{{tts en_US:Expression}}", f"{{{{tts {safe_locale}:Expression}}}}")
        back_html = back_html.replace("{{tts en_US:Expression}}", f"{{{{tts {safe_locale}:Expression}}}}")

    return front_html, back_html, style_css


def ensure_note_type(col: Any, tts_locale: str = "en_US") -> dict[str, Any]:
    """Create or safely migrate the owned 'AI Vocabulary (v1)' note type in `col`.

    Never deletes existing fields or overwrites unrelated note types.
    """
    mm = col.models
    front_html, back_html, style_css = load_template_assets(tts_locale=tts_locale)

    existing = mm.by_name(NOTE_TYPE_NAME)
    if existing is None:
        nt = mm.new(NOTE_TYPE_NAME)
        for field_name in NOTE_FIELDS:
            mm.add_field(nt, mm.new_field(field_name))

        tmpl = mm.new_template("Recognition")
        tmpl["qfmt"] = front_html
        tmpl["afmt"] = back_html
        mm.add_template(nt, tmpl)
        nt["css"] = style_css
        # Set sort field to Expression (index 0)
        nt["sortf"] = 0

        if hasattr(mm, "add_dict"):
            res = mm.add_dict(nt)
            return mm.get(res.id) or nt
        mm.add(nt)
        return mm.by_name(NOTE_TYPE_NAME) or nt

    # Safe non-destructive migration: add any missing fields from NOTE_FIELDS
    existing_field_names = set(mm.field_names(existing))
    changed = False
    for field_name in NOTE_FIELDS:
        if field_name not in existing_field_names:
            mm.add_field(existing, mm.new_field(field_name))
            changed = True

    if not existing.get("tmpls"):
        tmpl = mm.new_template("Recognition")
        tmpl["qfmt"] = front_html
        tmpl["afmt"] = back_html
        mm.add_template(existing, tmpl)
        changed = True
    else:
        tmpl0 = existing["tmpls"][0]
        if tmpl0.get("qfmt") != front_html or tmpl0.get("afmt") != back_html:
            tmpl0["qfmt"] = front_html
            tmpl0["afmt"] = back_html
            changed = True

    if existing.get("css") != style_css:
        existing["css"] = style_css
        changed = True

    if changed:
        if hasattr(mm, "update_dict"):
            mm.update_dict(existing)
        else:
            mm.save(existing)

    return mm.by_name(NOTE_TYPE_NAME) or existing


def render_card_preview_html(item: EnrichedItem, side: str = "back", tts_locale: str = "en_US") -> str:
    """Render a standalone HTML preview of the Front or Back card before Anki import."""
    _, _, style_css = load_template_assets(tts_locale=tts_locale)
    fields = item.to_note_fields()
    safe_locale = sanitize_tts_locale(tts_locale)

    # Preview image using thumbnail data URI (so QTextBrowser renders it without remote HTTP)
    image_html = fields["Image"]
    if item.selected_image is not None:
        cand = item.selected_image
        raw_bytes = cand.thumbnail_bytes or cand.downloaded_bytes
        alt = html.escape(cand.alt_text or item.expression, quote=True)
        if raw_bytes:
            b64 = base64.b64encode(raw_bytes).decode("ascii")
            mime = "image/png" if raw_bytes.startswith(b"\x89PNG") else "image/jpeg"
            image_html = f'<img src="data:{mime};base64,{b64}" alt="{alt}">'
        elif not image_html:
            src = html.escape(cand.thumbnail_url or cand.download_url, quote=True)
            image_html = f'<img src="{src}" alt="{alt}">'

    tts_badge = (
        f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;'
        f'background:#f1f5f9;color:#475569;font-size:12px;">🔊 Native TTS ({safe_locale})</span>'
        if not fields["Audio"]
        else f'<span style="font-size:12px;color:#475569;">🔊 {html.escape(fields["Audio"])}</span>'
    )

    ipa_block = f'<div class="ipa">{fields["IPA"]}</div>' if fields["IPA"] else ""
    pos_block = f'<span class="pos-badge">{fields["PartOfSpeech"]}</span>' if fields["PartOfSpeech"] else ""
    sense_block = (
        f'<div class="sense-banner"><span class="sense-label">Sense:</span> {fields["Sense"]}</div>'
        if fields["Sense"]
        else ""
    )

    if side.lower() == "front":
        masked_hint = html.escape(build_typing_hint(item.expression))
        body = f"""
        <div class="ai-vocab-card front-side">
          <div class="prompt-header">
            <div class="hint-row">
              <span class="typing-hint">{masked_hint}</span>
              {pos_block}
            </div>
          </div>
          <div class="meanings-box">
            <div class="definition-en">{fields["DefinitionEN"]}</div>
            <div class="meaning-vi">{fields["MeaningVI"]}</div>
          </div>
          {sense_block}
          <div class="typing-box">
            <div class="typing-label">Type the vocabulary:</div>
            <div style="display:inline-block;width:85%;max-width:420px;padding:8px 12px;border-radius:8px;border:1px solid #cbd5e1;background:#ffffff;color:#94a3b8;font-size:16px;">
              ⌨️ Type '{masked_hint}' here in Anki...
            </div>
          </div>
        </div>
        """
    else:
        img_block = f'<div class="vocab-image-container">{image_html}</div>' if image_html else ""
        ex_items = "".join(
            f"<li>{fields[f'Example{i}']}</li>" for i in range(1, 6) if fields.get(f"Example{i}")
        )
        src_block = f'<div class="source-footer">{fields["Source"]}</div>' if fields["Source"] else ""
        body = f"""
        <div class="ai-vocab-card back-side">
          <div class="expression-header">
            <div class="title-row">
              <h1 class="expression">{fields["Expression"]}</h1>
              {pos_block}
            </div>
            {ipa_block}
            <div class="tts-control">{tts_badge}</div>
          </div>
          <hr id="answer" class="divider">
          {sense_block}
          <div class="meanings-box">
            <div class="definition-en">{fields["DefinitionEN"]}</div>
            <div class="meaning-vi">{fields["MeaningVI"]}</div>
          </div>
          {img_block}
          <div class="examples-section">
            <div class="examples-title">Examples</div>
            <ol class="examples-list">{ex_items}</ol>
          </div>
          {src_block}
        </div>
        """

    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{style_css}</style></head><body class='card'>{body}</body></html>"

