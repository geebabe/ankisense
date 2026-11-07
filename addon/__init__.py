"""Anki Desktop Add-on Entry Point — AI Vocabulary Importer."""

from __future__ import annotations

from typing import Any

from .core.secrets import SecretStore


DEFAULT_CONFIG: dict[str, Any] = {
    "llm_base_url": "https://generativelanguage.googleapis.com/v1beta",
    "llm_model": "gemini-3.5-flash-lite",
    "default_deck": "Default",
    "default_tts_locale": "en_US",
    "llm_api_key": "",
    "pexels_api_key": "",
    "pixabay_api_key": "",
}

_active_dialog: Any = None


def _get_addon_config(mw: Any) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if mw and getattr(mw, "addonManager", None):
        try:
            loaded = mw.addonManager.getConfig(__name__)
            if isinstance(loaded, dict):
                cfg.update(loaded)
        except Exception:
            pass
    return cfg


def _save_addon_config(mw: Any, new_cfg: dict[str, Any]) -> None:
    if mw and getattr(mw, "addonManager", None):
        try:
            mw.addonManager.writeConfig(__name__, new_cfg)
        except Exception:
            pass


def open_importer_dialog() -> None:
    """Launch the AI Vocabulary Importer dialog from Anki's Tools menu or top toolbar."""
    global _active_dialog
    from aqt import mw
    from .ui.main_dialog import VocabImporterDialog

    if mw is None:
        return

    config = _get_addon_config(mw)
    secret_store = SecretStore(
        fallback_getter=lambda: _get_addon_config(mw),
        fallback_writer=lambda c: _save_addon_config(mw, c),
    )

    _active_dialog = VocabImporterDialog(
        parent=mw,
        mw=mw,
        config=config,
        secret_store=secret_store,
        on_save_config=lambda c: _save_addon_config(mw, c),
    )
    _active_dialog.show()
    _active_dialog.raise_()
    _active_dialog.activateWindow()


def open_settings_dialog() -> None:
    """Launch the API Keys & Settings dialog directly."""
    from aqt import mw
    from .ui.settings_dialog import SettingsDialog

    if mw is None:
        return

    config = _get_addon_config(mw)
    secret_store = SecretStore(
        fallback_getter=lambda: _get_addon_config(mw),
        fallback_writer=lambda c: _save_addon_config(mw, c),
    )
    dlg = SettingsDialog(
        parent=mw,
        config=config,
        secret_store=secret_store,
        on_save_config=lambda c: _save_addon_config(mw, c),
    )
    dlg.exec()


def setup_anki_menu() -> None:
    """Register Tools menu actions, top toolbar button, and Add-ons Config button in Anki."""
    try:
        from aqt import gui_hooks, mw
        from aqt.qt import QAction, QKeySequence
    except Exception:
        return

    if mw is None:
        return

    def _on_top_toolbar_init(links: list[str], toolbar: Any) -> None:
        try:
            link = toolbar.create_link(
                cmd="ai_vocab_importer",
                label="AI Vocab",
                func=open_importer_dialog,
                tip="Open AI Vocabulary Importer (Cmd+Shift+I)",
                id="ai_vocab_top_link",
            )
            links.append(link)
        except Exception:
            pass

    gui_hooks.top_toolbar_did_init_links.append(_on_top_toolbar_init)

    def _register_action() -> None:
        if getattr(mw, "addonManager", None):
            try:
                mw.addonManager.setConfigAction(__name__, open_settings_dialog)
            except Exception:
                pass

        if not getattr(mw, "form", None) or not getattr(mw.form, "menuTools", None):
            return
        for existing_act in mw.form.menuTools.actions():
            if existing_act.text() == "AI Vocabulary Importer...":
                return

        action = QAction("AI Vocabulary Importer...", mw)
        action.setShortcut(QKeySequence("Ctrl+Shift+I"))
        action.triggered.connect(open_importer_dialog)
        mw.form.menuTools.addAction(action)

        settings_action = QAction("AI Vocabulary Importer — API Keys & Settings...", mw)
        settings_action.triggered.connect(open_settings_dialog)
        mw.form.menuTools.addAction(settings_action)

    if getattr(mw, "form", None) and getattr(mw.form, "menuTools", None):
        _register_action()
    else:
        gui_hooks.main_window_did_init.append(_register_action)


setup_anki_menu()
