"""Settings dialog for LLM and image provider API credentials (stored in macOS Keychain)."""

from __future__ import annotations

from typing import Any, Callable

from aqt.qt import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.secrets import SecretStore, redact_secret
from ..providers.llm import DEFAULT_GEMINI_BASE_URL, DEFAULT_GEMINI_MODEL, is_google_api_key


class SettingsDialog(QDialog):
    """Dialog allowing the user to configure LLM endpoint/model and Keychain API keys."""

    def __init__(
        self,
        parent: QWidget | None,
        config: dict[str, Any],
        secret_store: SecretStore,
        on_save_config: Callable[[dict[str, Any]], None],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI Vocabulary Importer — API Keys & Settings")
        self.setMinimumWidth(520)
        self._config = dict(config)
        self._secrets = secret_store
        self._on_save_config = on_save_config

        self._build_ui()
        self._load_values()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        info = QLabel(
            "API keys are stored securely in your macOS Keychain ('anki-ai-vocab-importer') "
            "and are never written to logs."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()

        # Preset row
        preset_row = QHBoxLayout()
        btn_gemini = QPushButton("Preset: Google Gemini")
        btn_gemini.setToolTip("Sets Base URL to Google Gemini v1beta and model to gemini-3.5-flash-lite")
        btn_gemini.clicked.connect(
            lambda: self._apply_preset(
                DEFAULT_GEMINI_BASE_URL,
                DEFAULT_GEMINI_MODEL,
            )
        )
        btn_openai = QPushButton("Preset: OpenAI")
        btn_openai.clicked.connect(lambda: self._apply_preset("https://api.openai.com/v1", "gpt-4o-mini"))
        btn_deepseek = QPushButton("Preset: DeepSeek")
        btn_deepseek.clicked.connect(lambda: self._apply_preset("https://api.deepseek.com/v1", "deepseek-chat"))
        preset_row.addWidget(btn_gemini)
        preset_row.addWidget(btn_openai)
        preset_row.addWidget(btn_deepseek)
        form.addRow("Quick Presets:", preset_row)

        self.base_url_edit = QLineEdit()
        self.base_url_edit.setPlaceholderText("https://api.openai.com/v1")
        form.addRow("LLM Base URL:", self.base_url_edit)

        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("gpt-4o-mini")
        form.addRow("LLM Model:", self.model_edit)

        self.llm_key_edit = QLineEdit()
        self.llm_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.llm_key_edit.setPlaceholderText("AIza... (Google Gemini) or sk-... (OpenAI / DeepSeek)")
        form.addRow("LLM API Key:", self.llm_key_edit)

        self.pexels_key_edit = QLineEdit()
        self.pexels_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.pexels_key_edit.setPlaceholderText("Optional — Pexels API key (primary image source)")
        form.addRow("Pexels API Key:", self.pexels_key_edit)

        self.pixabay_key_edit = QLineEdit()
        self.pixabay_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.pixabay_key_edit.setPlaceholderText("Optional — Pixabay API key (fallback image source)")
        form.addRow("Pixabay API Key:", self.pixabay_key_edit)

        self.locale_combo = QComboBox()
        self.locale_combo.addItems(["en_US", "en_GB", "en_AU"])
        form.addRow("Default TTS Locale:", self.locale_combo)

        layout.addLayout(form)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("Save Settings")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

    def _apply_preset(self, base_url: str, model: str) -> None:
        self.base_url_edit.setText(base_url)
        self.model_edit.setText(model)

    def _load_values(self) -> None:
        self.base_url_edit.setText(str(self._config.get("llm_base_url") or DEFAULT_GEMINI_BASE_URL))
        self.model_edit.setText(str(self._config.get("llm_model") or DEFAULT_GEMINI_MODEL))
        locale = str(self._config.get("default_tts_locale") or "en_US")
        idx = self.locale_combo.findText(locale)
        if idx >= 0:
            self.locale_combo.setCurrentIndex(idx)

        llm_key = self._secrets.get_secret("llm_api_key")
        pexels_key = self._secrets.get_secret("pexels_api_key")
        pixabay_key = self._secrets.get_secret("pixabay_api_key")

        self.llm_key_edit.setText(llm_key)
        self.pexels_key_edit.setText(pexels_key)
        self.pixabay_key_edit.setText(pixabay_key)

        self.status_label.setText(
            f"Loaded status — LLM: {redact_secret(llm_key)} | "
            f"Pexels: {redact_secret(pexels_key)} | "
            f"Pixabay: {redact_secret(pixabay_key)}"
        )

    def _save(self) -> None:
        base_url = self.base_url_edit.text().strip() or DEFAULT_GEMINI_BASE_URL
        model = self.model_edit.text().strip() or DEFAULT_GEMINI_MODEL
        llm_key = self.llm_key_edit.text().strip()

        if is_google_api_key(llm_key) and "openai.com" in base_url:
            base_url = DEFAULT_GEMINI_BASE_URL
            if model in ("gpt-4o-mini", "gemini-2.5-flash"):
                model = DEFAULT_GEMINI_MODEL
            self.base_url_edit.setText(base_url)
            self.model_edit.setText(model)

        if not base_url.startswith(("https://", "http://")):
            QMessageBox.warning(self, "Invalid URL", "LLM Base URL must start with https:// or http://")
            return

        self._config["llm_base_url"] = base_url
        self._config["llm_model"] = model
        self._config["default_tts_locale"] = self.locale_combo.currentText()
        self._on_save_config(self._config)

        self._secrets.set_secret("llm_api_key", llm_key)
        self._secrets.set_secret("pexels_api_key", self.pexels_key_edit.text().strip())
        self._secrets.set_secret("pixabay_api_key", self.pixabay_key_edit.text().strip())

        self.accept()

