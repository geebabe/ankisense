"""Main multi-step workflow dialog for Anki AI Vocabulary Importer."""

from __future__ import annotations

from typing import Any, Callable

from aqt.qt import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPixmap,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QThread,
    QVBoxLayout,
    QWidget,
    Qt,
    pyqtSignal,
)

from ..anki_integration.importer import (
    ImportSummary,
    import_enriched_items,
    refresh_deck_duplicate_flags,
)
from ..anki_integration.media import MediaPersistenceService
from ..anki_integration.note_type import ensure_note_type, render_card_preview_html
from ..core.dedupe import compute_sense_key, find_batch_merge_candidates, merge_confirmed_duplicates
from ..core.enrichment import EnrichmentService
from ..core.parsing import classify_expression, parse_vocabulary_input, read_vocab_file
from ..core.schemas import DuplicatePolicy, EnrichedItem, ImageCandidate, ParsedItem, ReviewState
from ..core.secrets import SecretStore
from ..providers.dictionary import FreeDictionaryClient
from ..providers.llm import OpenAICompatibleLLMClient
from ..providers.openverse import OpenverseClient
from ..providers.pexels import PexelsClient
from ..providers.pixabay import PixabayClient
from .settings_dialog import SettingsDialog


SAMPLE_M1_INPUT = (
    "book (v)\n"
    "book (n)\n"
    "serendipity\n"
    "run into: tình cờ gặp\n"
    "a blessing in disguise"
)


class EnrichmentWorker(QThread):
    """Background worker thread running network enrichment + thumbnail prefetch off the UI thread."""

    progress_changed = pyqtSignal(int, int, str)
    item_finished = pyqtSignal(int, object)
    batch_finished = pyqtSignal(list)

    def __init__(
        self,
        service: EnrichmentService,
        media_service: MediaPersistenceService,
        items: list[tuple[int, ParsedItem]],
        locale: str = "en_US",
    ) -> None:
        super().__init__()
        self._service = service
        self._media = media_service
        self._items = items
        self._locale = locale
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        import time

        results: list[tuple[int, EnrichedItem]] = []
        total = len(self._items)
        start_time = time.time()
        for step_idx, (orig_idx, parsed_item) in enumerate(self._items):
            if self._cancelled:
                break
            eta_str = ""
            if step_idx > 0:
                elapsed = time.time() - start_time
                avg_sec = elapsed / step_idx
                rem_sec = int(avg_sec * (total - step_idx))
                if rem_sec >= 60:
                    eta_str = f" • ~{rem_sec // 60}m {rem_sec % 60:02d}s remaining"
                else:
                    eta_str = f" • ~{rem_sec}s remaining"

            self.progress_changed.emit(
                step_idx,
                total,
                f"Generating '{parsed_item.expression}' ({step_idx + 1}/{total}){eta_str}...",
            )
            enriched = self._service.enrich_single(parsed_item, locale=self._locale)
            # Prefetch small thumbnails in background so UI preview is snappy
            for cand in enriched.image_candidates:
                if self._cancelled:
                    break
                self._media.fetch_thumbnail(cand)
            results.append((orig_idx, enriched))
            self.item_finished.emit(orig_idx, enriched)

        self.progress_changed.emit(len(results), total, f"Finished {len(results)}/{total} items.")
        self.batch_finished.emit(results)


class ImageSearchWorker(QThread):
    """Background worker for re-querying Pexels/Pixabay for a single item."""

    finished_search = pyqtSignal(list, str)

    def __init__(
        self,
        service: EnrichmentService,
        media_service: MediaPersistenceService,
        query: str,
    ) -> None:
        super().__init__()
        self._service = service
        self._media = media_service
        self._query = query

    def run(self) -> None:
        candidates, warning = self._service.search_images(self._query, max_images=4)
        for cand in candidates:
            self._media.fetch_thumbnail(cand)
        self.finished_search.emit(candidates, warning)


class ImagePrefetchWorker(QThread):
    """Background worker that downloads selected full-size images off the UI thread before collection import."""

    progress_changed = pyqtSignal(int, int, str)
    finished_prefetch = pyqtSignal(bool)

    def __init__(
        self,
        media_service: MediaPersistenceService,
        items: list[EnrichedItem],
    ) -> None:
        super().__init__()
        self._media = media_service
        self._items = items
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        candidates_to_fetch = [
            (item.expression, item.selected_image)
            for item in self._items
            if item.parsed.selected and item.selected_image is not None and not item.local_image_filename
        ]
        total = len(candidates_to_fetch)
        for idx, (expr, cand) in enumerate(candidates_to_fetch):
            if self._cancelled:
                self.finished_prefetch.emit(True)
                return
            if cand is None:
                continue
            self.progress_changed.emit(idx, total, f"Downloading image for '{expr}' ({idx + 1}/{total})...")
            try:
                self._media.prefetch_candidate_image(cand)
            except Exception:
                # Leave downloaded_bytes empty; importer will record a non-fatal warning and still create the note
                pass
        self.progress_changed.emit(total, total, "Image downloads complete.")
        self.finished_prefetch.emit(False)


class VocabImporterDialog(QDialog):
    """Main non-blocking Qt dialog for parsing, enriching, previewing, and importing vocabulary."""

    def __init__(
        self,
        parent: QWidget | None,
        mw: Any,
        config: dict[str, Any],
        secret_store: SecretStore,
        on_save_config: Callable[[dict[str, Any]], None],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI Vocabulary Importer")
        self.resize(1160, 780)
        self._mw = mw
        self._config = dict(config)
        self._secrets = secret_store
        self._on_save_config = on_save_config

        self.parsed_items: list[ParsedItem] = []
        self.enriched_items: list[EnrichedItem] = []
        self._current_enriched_idx: int = -1
        self._updating_detail_ui: bool = False
        self._preview_side: str = "back"
        self._worker: EnrichmentWorker | None = None
        self._img_worker: ImageSearchWorker | None = None
        self._prefetch_worker: ImagePrefetchWorker | None = None

        self._build_ui()
        self._populate_decks()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # Top control bar: Deck, Locale, Duplicate policy, Settings
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Destination Deck (type new or pick):"))
        self.deck_combo = QComboBox()
        self.deck_combo.setEditable(True)
        self.deck_combo.setMinimumWidth(260)
        if self.deck_combo.lineEdit():
            self.deck_combo.lineEdit().setPlaceholderText("e.g. Vocabulary::Crime & Law")
        self.deck_combo.currentTextChanged.connect(self._on_deck_changed)
        top_bar.addWidget(self.deck_combo)

        top_bar.addWidget(QLabel("TTS Locale:"))
        self.locale_combo = QComboBox()
        self.locale_combo.addItems(["en_US", "en_GB", "en_AU"])
        default_locale = str(self._config.get("default_tts_locale") or "en_US")
        loc_idx = self.locale_combo.findText(default_locale)
        if loc_idx >= 0:
            self.locale_combo.setCurrentIndex(loc_idx)
        self.locale_combo.currentTextChanged.connect(lambda _: self._refresh_card_preview())
        top_bar.addWidget(self.locale_combo)

        top_bar.addWidget(QLabel("Duplicates:"))
        self.dup_policy_combo = QComboBox()
        self.dup_policy_combo.addItem("Skip existing (Default)", DuplicatePolicy.SKIP.value)
        self.dup_policy_combo.addItem("Overwrite existing note", DuplicatePolicy.OVERWRITE.value)
        top_bar.addWidget(self.dup_policy_combo)

        top_bar.addStretch()
        has_key = bool(self._secrets.get_secret("llm_api_key"))
        btn_label = "API Keys & Settings (Key Active)" if has_key else "API Keys & Settings..."
        self.settings_btn = QPushButton(btn_label)
        self.settings_btn.clicked.connect(self._open_settings)
        top_bar.addWidget(self.settings_btn)

        root.addLayout(top_bar)

        # Step workflow breadcrumbs
        step_header_widget = QWidget()
        step_header_layout = QHBoxLayout(step_header_widget)
        step_header_layout.setContentsMargins(4, 2, 4, 6)

        self.step1_pill = QLabel("  1️⃣ Step 1: Input & Parse  ")
        self.step2_pill = QLabel("  2️⃣ Step 2: Review & Customise  ")
        self.step3_pill = QLabel("  3️⃣ Step 3: Import to Deck  ")

        arrow_style = "font-weight: bold; color: #94a3b8; font-size: 13px;"
        lbl_arrow1 = QLabel("➔")
        lbl_arrow1.setStyleSheet(arrow_style)
        lbl_arrow2 = QLabel("➔")
        lbl_arrow2.setStyleSheet(arrow_style)

        step_header_layout.addWidget(self.step1_pill)
        step_header_layout.addWidget(lbl_arrow1)
        step_header_layout.addWidget(self.step2_pill)
        step_header_layout.addWidget(lbl_arrow2)
        step_header_layout.addWidget(self.step3_pill)
        step_header_layout.addStretch()

        root.addWidget(step_header_widget)

        # Tabs: Step 1 (Input & Parse) and Step 2 (Review, Images & Card Preview)
        self.tabs = QTabWidget()
        self._build_step1_tab()
        self._build_step2_tab()
        self.tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self.tabs, stretch=1)

        # Bottom status and action bar
        bottom_bar = QHBoxLayout()
        self.status_label = QLabel("Paste vocabulary or load a .txt file, then click '1. Parse Vocabulary'.")
        bottom_bar.addWidget(self.status_label, stretch=1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(260)
        self.progress_bar.setVisible(False)
        bottom_bar.addWidget(self.progress_bar)

        self.cancel_btn = QPushButton("Cancel Task")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel_background_work)
        bottom_bar.addWidget(self.cancel_btn)

        self.retry_failed_btn = QPushButton("Retry Failed")
        self.retry_failed_btn.setVisible(False)
        self.retry_failed_btn.clicked.connect(self._retry_failed_items)
        bottom_bar.addWidget(self.retry_failed_btn)

        self.import_btn = QPushButton("🚀 Import Ready Cards to Anki")
        self.import_btn.setEnabled(False)
        self.import_btn.setStyleSheet(
            "QPushButton { background-color: #16a34a; color: #ffffff; font-weight: bold; font-size: 13px; "
            "padding: 7px 18px; border-radius: 6px; border: 1px solid #15803d; } "
            "QPushButton:hover { background-color: #15803d; } "
            "QPushButton:disabled { background-color: #cbd5e1; color: #94a3b8; border: 1px solid #cbd5e1; }"
        )
        self.import_btn.clicked.connect(self._import_to_anki)
        bottom_bar.addWidget(self.import_btn)

        root.addLayout(bottom_bar)
        self._update_step_pills()

    def _build_step1_tab(self) -> None:
        tab1 = QWidget()
        layout = QVBoxLayout(tab1)

        # Guide banner at top of Step 1
        self.step1_banner = QLabel("📌 Step 1: Paste vocabulary below or load a .txt file, then click '1. Parse Vocabulary'.")
        self.step1_banner.setStyleSheet(
            "background-color: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; "
            "border-radius: 6px; padding: 7px 12px; font-weight: 500; font-size: 12.5px;"
        )
        layout.addWidget(self.step1_banner)

        input_header = QHBoxLayout()
        input_header.addWidget(
            QLabel("Paste words, phrasal verbs, idioms, or hints (e.g. 'book (v), run into: tình cờ gặp'):")
        )
        input_header.addStretch()

        sample_btn = QPushButton("Load 5-Item Sample")
        sample_btn.clicked.connect(lambda: self.raw_input_edit.setPlainText(SAMPLE_M1_INPUT))
        input_header.addWidget(sample_btn)

        load_txt_btn = QPushButton("Load UTF-8 .txt File...")
        load_txt_btn.clicked.connect(self._pick_txt_file)
        input_header.addWidget(load_txt_btn)
        layout.addLayout(input_header)

        self.raw_input_edit = QPlainTextEdit()
        self.raw_input_edit.setPlaceholderText(
            "Example:\nbook (v)\nbook (n)\nserendipity, \"run into: gặp tình cờ, bất ngờ\"\na blessing in disguise"
        )
        self.raw_input_edit.setMaximumHeight(140)
        layout.addWidget(self.raw_input_edit)

        parse_bar = QHBoxLayout()
        self.parse_btn = QPushButton("1. Parse Vocabulary")
        self.parse_btn.setStyleSheet(
            "QPushButton { background-color: #0284c7; color: white; font-weight: bold; font-size: 13px; "
            "padding: 7px 16px; border-radius: 6px; border: 1px solid #0369a1; } "
            "QPushButton:hover { background-color: #0369a1; }"
        )
        self.parse_btn.clicked.connect(self._on_parse_clicked)
        parse_bar.addWidget(self.parse_btn)

        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.setToolTip("Select all parsed rows for enrichment")
        self.btn_select_all.clicked.connect(self._select_all_parsed)
        parse_bar.addWidget(self.btn_select_all)

        self.btn_deselect_all = QPushButton("Deselect All")
        self.btn_deselect_all.setToolTip("Deselect all parsed rows")
        self.btn_deselect_all.clicked.connect(self._deselect_all_parsed)
        parse_bar.addWidget(self.btn_deselect_all)

        self.merge_dups_btn = QPushButton("Merge Same-Sense Duplicates")
        self.merge_dups_btn.setEnabled(False)
        self.merge_dups_btn.clicked.connect(self._on_merge_duplicates_clicked)
        parse_bar.addWidget(self.merge_dups_btn)

        parse_bar.addStretch()
        self.generate_btn = QPushButton("✨ 2. Generate && Enrich Selected (0) ➔")
        self.generate_btn.setEnabled(False)
        self.generate_btn.setStyleSheet(
            "QPushButton { background-color: #2563eb; color: #ffffff; font-weight: bold; font-size: 13px; "
            "padding: 8px 20px; border-radius: 6px; border: 1px solid #1d4ed8; } "
            "QPushButton:hover { background-color: #1d4ed8; } "
            "QPushButton:disabled { background-color: #cbd5e1; color: #94a3b8; border: 1px solid #cbd5e1; }"
        )
        self.generate_btn.clicked.connect(self._on_generate_clicked)
        parse_bar.addWidget(self.generate_btn)
        layout.addLayout(parse_bar)

        self.parse_table = QTableWidget(0, 6)
        self.parse_table.setHorizontalHeaderLabels(
            ["Select", "Expression", "POS", "Sense Hint / Proposed Sense", "Type", "Status & Warnings"]
        )
        header = self.parse_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.parse_table.itemChanged.connect(self._on_parse_table_edited)
        layout.addWidget(self.parse_table, stretch=1)

        self.tabs.addTab(tab1, "1. Input && Parse")

    def _build_step2_tab(self) -> None:
        tab2 = QWidget()
        layout = QVBoxLayout(tab2)

        # Step 2 header & prominent Next action bar
        step2_header = QHBoxLayout()
        self.btn_back_to_step1 = QPushButton("← Back to Step 1 (Input)")
        self.btn_back_to_step1.setStyleSheet(
            "QPushButton { background-color: #f1f5f9; color: #334155; font-weight: 600; font-size: 12.5px; "
            "padding: 6px 12px; border-radius: 6px; border: 1px solid #cbd5e1; } "
            "QPushButton:hover { background-color: #e2e8f0; }"
        )
        self.btn_back_to_step1.clicked.connect(lambda: self.tabs.setCurrentIndex(0))
        step2_header.addWidget(self.btn_back_to_step1)

        self.step2_guide_banner = QLabel("👉 Step 2 of 2: Click cards on the left to review & pick photos. When ready, click:")
        self.step2_guide_banner.setStyleSheet(
            "background-color: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; "
            "border-radius: 6px; padding: 7px 12px; font-weight: 600; font-size: 13px;"
        )
        step2_header.addWidget(self.step2_guide_banner, stretch=1)

        self.step2_import_btn = QPushButton("🚀 Import Ready Cards to Anki ➔")
        self.step2_import_btn.setStyleSheet(
            "QPushButton { background-color: #16a34a; color: #ffffff; font-weight: bold; font-size: 13px; "
            "padding: 8px 20px; border-radius: 6px; border: 1px solid #15803d; } "
            "QPushButton:hover { background-color: #15803d; } "
            "QPushButton:disabled { background-color: #cbd5e1; color: #94a3b8; border: 1px solid #cbd5e1; }"
        )
        self.step2_import_btn.setEnabled(False)
        self.step2_import_btn.clicked.connect(self._import_to_anki)
        step2_header.addWidget(self.step2_import_btn)
        layout.addLayout(step2_header)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Item list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Enriched Cards:"))
        self.enriched_list = QListWidget()
        self.enriched_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.enriched_list.currentRowChanged.connect(self._on_enriched_row_selected)
        left_layout.addWidget(self.enriched_list)
        splitter.addWidget(left_widget)

        # Center: Field editor + Image candidate picker (inside ScrollArea)
        center_scroll = QScrollArea()
        center_scroll.setWidgetResizable(True)
        center_container = QWidget()
        center_layout = QVBoxLayout(center_container)

        self.item_warning_banner = QLabel("")
        self.item_warning_banner.setWordWrap(True)
        self.item_warning_banner.setStyleSheet(
            "background-color: #fef3c7; color: #92400e; padding: 8px; border-radius: 6px; font-weight: 500;"
        )
        self.item_warning_banner.setVisible(False)
        center_layout.addWidget(self.item_warning_banner)

        form = QFormLayout()
        self.edit_expression = QLineEdit()
        self.edit_expression.textEdited.connect(self._on_detail_field_edited)
        form.addRow("Expression:", self.edit_expression)

        pos_sense_row = QHBoxLayout()
        self.edit_pos = QLineEdit()
        self.edit_pos.setPlaceholderText("n / v / adj / phr_v / idiom")
        self.edit_pos.setMaximumWidth(110)
        self.edit_pos.textEdited.connect(self._on_detail_field_edited)
        pos_sense_row.addWidget(self.edit_pos)
        pos_sense_row.addWidget(QLabel("Sense:"))
        self.edit_sense = QLineEdit()
        self.edit_sense.textEdited.connect(self._on_detail_field_edited)
        pos_sense_row.addWidget(self.edit_sense, stretch=1)
        form.addRow("POS & Sense:", pos_sense_row)

        ipa_row = QHBoxLayout()
        self.edit_ipa = QLineEdit()
        self.edit_ipa.setPlaceholderText("/.../")
        self.edit_ipa.textEdited.connect(self._on_detail_field_edited)
        ipa_row.addWidget(self.edit_ipa, stretch=1)
        self.ipa_provenance_label = QLabel("")
        self.ipa_provenance_label.setStyleSheet("font-size: 11.5px; color: #64748b;")
        ipa_row.addWidget(self.ipa_provenance_label)
        form.addRow("IPA:", ipa_row)

        self.edit_def_en = QLineEdit()
        self.edit_def_en.textEdited.connect(self._on_detail_field_edited)
        form.addRow("Definition (EN):", self.edit_def_en)

        self.edit_meaning_vi = QLineEdit()
        self.edit_meaning_vi.textEdited.connect(self._on_detail_field_edited)
        form.addRow("Meaning (VI):", self.edit_meaning_vi)

        self.example_edits: list[QLineEdit] = []
        for i in range(5):
            ex_edit = QLineEdit()
            ex_edit.textEdited.connect(self._on_detail_field_edited)
            self.example_edits.append(ex_edit)
            form.addRow(f"Example {i + 1}:", ex_edit)

        extra_row = QHBoxLayout()
        self.edit_audio = QLineEdit()
        self.edit_audio.setPlaceholderText("Leave empty for Native TTS, or [sound:file.mp3]")
        self.edit_audio.textEdited.connect(self._on_detail_field_edited)
        extra_row.addWidget(self.edit_audio, stretch=1)
        extra_row.addWidget(QLabel("Item Duplicate Policy:"))
        self.item_dup_combo = QComboBox()
        self.item_dup_combo.addItem("Use Batch Default", "")
        self.item_dup_combo.addItem("Skip", DuplicatePolicy.SKIP.value)
        self.item_dup_combo.addItem("Overwrite", DuplicatePolicy.OVERWRITE.value)
        self.item_dup_combo.currentIndexChanged.connect(lambda _: self._on_detail_field_edited())
        extra_row.addWidget(self.item_dup_combo)
        form.addRow("Audio Override:", extra_row)

        center_layout.addLayout(form)

        # Image candidate picker box
        img_group = QGroupBox("Illustrative Photo (Pexels / Pixabay — Existing Licensed Images Only)")
        img_layout = QVBoxLayout(img_group)

        query_row = QHBoxLayout()
        query_row.addWidget(QLabel("Search Query:"))
        self.edit_image_query = QLineEdit()
        query_row.addWidget(self.edit_image_query, stretch=1)
        self.search_img_btn = QPushButton("Search Images")
        self.search_img_btn.clicked.connect(self._on_manual_image_search)
        query_row.addWidget(self.search_img_btn)
        img_layout.addLayout(query_row)

        self.candidates_row = QHBoxLayout()
        self.image_btn_group = QButtonGroup(self)
        self.image_btn_group.idClicked.connect(self._on_image_candidate_selected)
        img_layout.addLayout(self.candidates_row)

        center_layout.addWidget(img_group)
        center_layout.addStretch()
        center_scroll.setWidget(center_container)
        splitter.addWidget(center_scroll)

        # Right: Live HTML Card Preview
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        preview_bar = QHBoxLayout()
        preview_bar.addWidget(QLabel("Live Card Preview:"))
        preview_bar.addStretch()
        self.btn_preview_front = QPushButton("Front")
        self.btn_preview_front.setCheckable(True)
        self.btn_preview_front.clicked.connect(lambda: self._set_preview_side("front"))
        self.btn_preview_back = QPushButton("Back")
        self.btn_preview_back.setCheckable(True)
        self.btn_preview_back.setChecked(True)
        self.btn_preview_back.clicked.connect(lambda: self._set_preview_side("back"))
        preview_bar.addWidget(self.btn_preview_front)
        preview_bar.addWidget(self.btn_preview_back)
        right_layout.addLayout(preview_bar)

        self.preview_browser = QTextBrowser()
        self.preview_browser.setOpenExternalLinks(True)
        right_layout.addWidget(self.preview_browser, stretch=1)
        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 3)

        layout.addWidget(splitter)
        self.tabs.addTab(tab2, "2. Review, Images && Preview")

    def _populate_decks(self) -> None:
        self.deck_combo.blockSignals(True)
        self.deck_combo.clear()
        deck_names: list[str] = ["Default"]
        if self._mw and getattr(self._mw, "col", None):
            try:
                ensure_note_type(self._mw.col, tts_locale=str(self._config.get("default_tts_locale") or "en_US"))
                decks = self._mw.col.decks.all_names_and_ids()
                deck_names = [d.name for d in decks] or ["Default"]
            except Exception:
                pass
        self.deck_combo.addItems(deck_names)
        saved_deck = str(self._config.get("default_deck") or "Default")
        idx = self.deck_combo.findText(saved_deck)
        if idx >= 0:
            self.deck_combo.setCurrentIndex(idx)
        self.deck_combo.blockSignals(False)

    def _on_deck_changed(self, deck_name: str) -> None:
        if self._mw and getattr(self._mw, "col", None) and self.enriched_items:
            refresh_deck_duplicate_flags(self._mw.col, deck_name, self.enriched_items)
            self._refresh_enriched_list_labels()

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self, self._config, self._secrets, self._on_save_config)
        if dlg.exec():
            has_key = bool(self._secrets.get_secret("llm_api_key"))
            self.settings_btn.setText("API Keys & Settings (Key Active)" if has_key else "API Keys & Settings...")
            self.status_label.setText("Updated API keys and settings.")

    def _pick_txt_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select UTF-8 Vocabulary .txt File",
            "",
            "Text Files (*.txt);;All Files (*)",
        )
        if not file_path:
            return
        try:
            content = read_vocab_file(file_path)
            self.raw_input_edit.setPlainText(content)
            # Suggest a deck name based on the .txt filename if currently Default
            from pathlib import Path
            import re
            stem = Path(file_path).stem
            cleaned_stem = re.sub(r"\b(sample|vocab|vocabulary|list|words)\b", "", stem, flags=re.IGNORECASE)
            cleaned_stem = re.sub(r"[_\-\s]+", " ", cleaned_stem).strip().title()
            if cleaned_stem and self.deck_combo.currentText().strip() in ("", "Default"):
                self.deck_combo.setEditText(f"Vocabulary::{cleaned_stem}")
            self.status_label.setText(f"Loaded '{file_path}'. Click '1. Parse Vocabulary' to inspect.")
        except Exception as exc:
            QMessageBox.critical(self, "File Read Error", f"Could not read UTF-8 file:\n{exc}")

    def _on_parse_clicked(self) -> None:
        raw_text = self.raw_input_edit.toPlainText()
        self.parsed_items = parse_vocabulary_input(raw_text)
        self._populate_parse_table()

        count = len(self.parsed_items)
        conflicts = sum(1 for it in self.parsed_items if it.review_state == ReviewState.CONFLICT_FLAGGED.value)
        reviews = sum(1 for it in self.parsed_items if it.review_state == ReviewState.NEEDS_REVIEW.value)
        self.merge_dups_btn.setEnabled(count > 1)
        self._update_generate_btn_label()

        if count == 0:
            self.step1_banner.setText("⚠️ Step 1: No valid vocabulary found. Please paste words or load a .txt file above.")
            self.status_label.setText("No items parsed.")
        else:
            self.step1_banner.setText(
                f"✅ Step 1 Complete: Parsed {count} item(s) ({conflicts} conflicts, {reviews} need review). "
                "Check/edit rows in table below, then click the blue '2. Generate & Enrich Selected' button ➔"
            )
            self.status_label.setText(
                f"Parsed {count} item(s). Click '2. Generate & Enrich Selected' to generate flashcard content."
            )

    def _update_step_pills(self) -> None:
        current_tab = self.tabs.currentIndex()
        active_style = (
            "background-color: #2563eb; color: #ffffff; font-weight: bold; "
            "font-size: 12px; padding: 4px 12px; border-radius: 12px;"
        )
        completed_style = (
            "background-color: #ecfdf5; color: #047857; font-weight: 600; "
            "font-size: 12px; padding: 4px 12px; border-radius: 12px; border: 1px solid #a7f3d0;"
        )
        inactive_style = (
            "background-color: #f1f5f9; color: #64748b; font-weight: 500; "
            "font-size: 12px; padding: 4px 12px; border-radius: 12px;"
        )
        ready_cta_style = (
            "background-color: #16a34a; color: #ffffff; font-weight: bold; "
            "font-size: 12px; padding: 4px 12px; border-radius: 12px;"
        )

        has_enriched = bool(self.enriched_items)
        if current_tab == 0:
            self.step1_pill.setStyleSheet(active_style)
            self.step2_pill.setStyleSheet(inactive_style)
            self.step3_pill.setStyleSheet(inactive_style)
        else:
            self.step1_pill.setStyleSheet(completed_style)
            self.step2_pill.setStyleSheet(active_style)
            self.step3_pill.setStyleSheet(ready_cta_style if has_enriched else inactive_style)

    def _on_tab_changed(self, _idx: int) -> None:
        self._update_step_pills()

    def _update_generate_btn_label(self) -> None:
        selected_count = sum(1 for it in self.parsed_items if it.selected)
        total_count = len(self.parsed_items)
        if total_count == 0:
            self.generate_btn.setText("✨ 2. Generate && Enrich Selected (0) ➔")
            self.generate_btn.setEnabled(False)
        else:
            self.generate_btn.setText(f"✨ 2. Generate && Enrich Selected ({selected_count}/{total_count}) ➔")
            self.generate_btn.setEnabled(selected_count > 0)

    def _select_all_parsed(self) -> None:
        self.parse_table.blockSignals(True)
        for row in range(self.parse_table.rowCount()):
            item = self.parse_table.item(row, 0)
            if item:
                item.setCheckState(Qt.CheckState.Checked)
        for it in self.parsed_items:
            it.selected = True
        self.parse_table.blockSignals(False)
        self._update_generate_btn_label()

    def _deselect_all_parsed(self) -> None:
        self.parse_table.blockSignals(True)
        for row in range(self.parse_table.rowCount()):
            item = self.parse_table.item(row, 0)
            if item:
                item.setCheckState(Qt.CheckState.Unchecked)
        for it in self.parsed_items:
            it.selected = False
        self.parse_table.blockSignals(False)
        self._update_generate_btn_label()

    def _populate_parse_table(self) -> None:
        self.parse_table.blockSignals(True)
        self.parse_table.setRowCount(len(self.parsed_items))

        for row, item in enumerate(self.parsed_items):
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            chk_item.setCheckState(Qt.CheckState.Checked if item.selected else Qt.CheckState.Unchecked)
            self.parse_table.setItem(row, 0, chk_item)

            self.parse_table.setItem(row, 1, QTableWidgetItem(item.expression))
            self.parse_table.setItem(row, 2, QTableWidgetItem(item.part_of_speech))
            hint_val = item.sense_hint or item.proposed_sense
            self.parse_table.setItem(row, 3, QTableWidgetItem(hint_val))

            type_item = QTableWidgetItem(item.classification)
            type_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.parse_table.setItem(row, 4, type_item)

            status_parts = [item.review_state.upper()]
            if item.warnings:
                status_parts.append(" — " + "; ".join(item.warnings))
            status_item = QTableWidgetItem("".join(status_parts))
            status_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.parse_table.setItem(row, 5, status_item)

        self.parse_table.blockSignals(False)
        self._update_generate_btn_label()

    def _on_parse_table_edited(self, table_item: QTableWidgetItem) -> None:
        row = table_item.row()
        col = table_item.column()
        if not (0 <= row < len(self.parsed_items)):
            return
        parsed = self.parsed_items[row]
        if col == 0:
            parsed.selected = table_item.checkState() == Qt.CheckState.Checked
            self._update_generate_btn_label()
        elif col == 1:
            parsed.expression = table_item.text().strip()
            parsed.classification = classify_expression(parsed.expression, parsed.part_of_speech)
        elif col == 2:
            parsed.part_of_speech = table_item.text().strip()
            parsed.classification = classify_expression(parsed.expression, parsed.part_of_speech)
        elif col == 3:
            parsed.sense_hint = table_item.text().strip()
            if parsed.sense_hint and parsed.review_state == ReviewState.NEEDS_REVIEW.value:
                parsed.review_state = ReviewState.READY.value

        find_batch_merge_candidates(self.parsed_items)

    def _on_merge_duplicates_clicked(self) -> None:
        before = len(self.parsed_items)
        self.parsed_items = merge_confirmed_duplicates(self.parsed_items)
        after = len(self.parsed_items)
        self._populate_parse_table()
        self._update_generate_btn_label()
        self.status_label.setText(f"Merged {before - after} duplicate same-sense row(s) ({after} remaining).")

    def _build_enrichment_service(self) -> tuple[EnrichmentService, MediaPersistenceService]:
        llm_key = self._secrets.get_secret("llm_api_key")
        pexels_key = self._secrets.get_secret("pexels_api_key")
        pixabay_key = self._secrets.get_secret("pixabay_api_key")
        base_url = str(self._config.get("llm_base_url") or "https://api.openai.com/v1")
        model = str(self._config.get("llm_model") or "gpt-4o-mini")

        llm_client = OpenAICompatibleLLMClient(api_key=llm_key, base_url=base_url, model=model)
        dict_client = FreeDictionaryClient()
        pexels_client = PexelsClient(api_key=pexels_key) if pexels_key else None
        pixabay_client = PixabayClient(api_key=pixabay_key) if pixabay_key else None
        openverse_client = OpenverseClient()
        return (
            EnrichmentService(
                llm_client=llm_client,
                dictionary_client=dict_client,
                pexels_client=pexels_client,
                pixabay_client=pixabay_client,
                openverse_client=openverse_client,
            ),
            MediaPersistenceService(),
        )

    def _on_generate_clicked(self) -> None:
        selected_pairs = [(idx, it) for idx, it in enumerate(self.parsed_items) if it.selected]
        if not selected_pairs:
            QMessageBox.information(self, "No Items Selected", "Please select at least one vocabulary row to generate.")
            return

        if not self._secrets.get_secret("llm_api_key"):
            QMessageBox.warning(
                self,
                "API Key Required",
                "Please configure your LLM API key in 'API Keys & Settings...' before generating.",
            )
            self._open_settings()
            if not self._secrets.get_secret("llm_api_key"):
                return

        self.enriched_items = [
            EnrichedItem(
                parsed=it,
                expression=it.expression,
                part_of_speech=it.part_of_speech,
                sense=it.sense_hint or it.proposed_sense,
                definition_en="",
                meaning_vi="",
            )
            for _, it in selected_pairs
        ]
        # Map worker index to self.enriched_items index
        worker_pairs = [(i, pair[1]) for i, pair in enumerate(selected_pairs)]
        self._start_enrichment_worker(worker_pairs)

    def _retry_failed_items(self) -> None:
        failed_pairs = [
            (idx, item.parsed)
            for idx, item in enumerate(self.enriched_items)
            if item.error_message or not item.definition_en
        ]
        if not failed_pairs:
            return
        self._start_enrichment_worker(failed_pairs)

    def _start_enrichment_worker(self, pairs: list[tuple[int, ParsedItem]]) -> None:
        service, media_svc = self._build_enrichment_service()
        locale = self.locale_combo.currentText()

        self.parse_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.import_btn.setEnabled(False)
        self.step2_import_btn.setEnabled(False)
        self.retry_failed_btn.setVisible(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(pairs))
        self.progress_bar.setValue(0)
        self.cancel_btn.setVisible(True)

        self._worker = EnrichmentWorker(service, media_svc, pairs, locale=locale)
        self._worker.progress_changed.connect(self._on_worker_progress)
        self._worker.item_finished.connect(self._on_worker_item_finished)
        self._worker.batch_finished.connect(self._on_worker_batch_finished)
        self._worker.start()

    def _on_worker_progress(self, current: int, total: int, message: str) -> None:
        self.progress_bar.setMaximum(max(1, total))
        self.progress_bar.setValue(current)
        self.status_label.setText(message)

    def _on_worker_item_finished(self, idx: int, enriched: EnrichedItem) -> None:
        if 0 <= idx < len(self.enriched_items):
            self.enriched_items[idx] = enriched

    def _on_worker_batch_finished(self, _results: list[ Any]) -> None:
        self.parse_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.cancel_btn.setVisible(False)

        # Auto-suggest topic-related deck name if deck is still "Default"
        current_deck = self.deck_combo.currentText().strip()
        if current_deck in ("", "Default"):
            from collections import Counter
            topic_counts = Counter(
                it.suggested_deck.strip()
                for it in self.enriched_items
                if getattr(it, "suggested_deck", "").strip()
            )
            if topic_counts:
                best_topic = topic_counts.most_common(1)[0][0]
                suggested_full = best_topic if "::" in best_topic else f"Vocabulary::{best_topic}"
                self.deck_combo.setEditText(suggested_full)

        if self._mw and getattr(self._mw, "col", None):
            refresh_deck_duplicate_flags(
                self._mw.col,
                self.deck_combo.currentText().strip() or "Default",
                self.enriched_items,
            )

        self._refresh_enriched_list_labels()
        if self.enriched_items:
            self.tabs.setCurrentIndex(1)
            self.enriched_list.setCurrentRow(0)

        failed = sum(1 for it in self.enriched_items if it.error_message)
        succeeded = len(self.enriched_items) - failed
        self.retry_failed_btn.setVisible(failed > 0)
        has_ready = succeeded > 0
        self.import_btn.setEnabled(has_ready)
        self.step2_import_btn.setEnabled(has_ready)

        import_text = f"🚀 Import {succeeded} Ready Cards to Anki ➔" if has_ready else "Import to Anki"
        self.import_btn.setText(import_text)
        self.step2_import_btn.setText(import_text)

        self.step2_guide_banner.setText(
            f"👉 Step 2 of 2: {succeeded} card(s) ready for import ({failed} failed). "
            "Review definitions or pick photos on the left, then click '🚀 Import Ready Cards to Anki' ➔"
        )
        self.status_label.setText(
            f"Enrichment complete: {succeeded} ready, {failed} failed. Review cards and click 'Import to Anki'."
        )
        self._update_step_pills()

    def _cancel_background_work(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self.status_label.setText("Cancelling after current item finishes...")
        if self._prefetch_worker and self._prefetch_worker.isRunning():
            self._prefetch_worker.cancel()
            self.status_label.setText("Cancelling image download before import...")

    def _refresh_enriched_list_labels(self) -> None:
        current_row = self.enriched_list.currentRow()
        self.enriched_list.blockSignals(True)
        self.enriched_list.clear()

        for item in self.enriched_items:
            if item.error_message:
                badge = "❌"
            elif item.parsed.review_state == ReviewState.CONFLICT_FLAGGED.value:
                badge = "⚠️"
            elif item.is_duplicate_in_deck:
                badge = "🔁"
            else:
                badge = "✅"
            pos_str = f" ({item.part_of_speech})" if item.part_of_speech else ""
            sense_str = f" — {item.sense}" if item.sense else ""
            list_item = QListWidgetItem(f"{badge} {item.expression}{pos_str}{sense_str}")
            self.enriched_list.addItem(list_item)

        self.enriched_list.blockSignals(False)
        if 0 <= current_row < len(self.enriched_items):
            self.enriched_list.setCurrentRow(current_row)

    def _on_enriched_row_selected(self, row: int) -> None:
        if not (0 <= row < len(self.enriched_items)):
            return
        self._current_enriched_idx = row
        item = self.enriched_items[row]

        self._updating_detail_ui = True
        self.edit_expression.setText(item.expression)
        self.edit_pos.setText(item.part_of_speech)
        self.edit_sense.setText(item.sense)
        self.edit_ipa.setText(item.ipa)
        self.ipa_provenance_label.setText(item.ipa_source or "No IPA")
        self.edit_def_en.setText(item.definition_en)
        self.edit_meaning_vi.setText(item.meaning_vi)

        padded = (list(item.examples) + ["", "", "", "", ""])[:5]
        for i in range(5):
            self.example_edits[i].setText(padded[i])

        self.edit_audio.setText(item.audio_field)
        dup_idx = self.item_dup_combo.findData(item.duplicate_policy)
        self.item_dup_combo.setCurrentIndex(dup_idx if dup_idx >= 0 else 0)

        self.edit_image_query.setText(item.image_query)

        # Show warnings/conflict/error banner if present
        notices: list[str] = []
        if item.error_message:
            notices.append(f"Error: {item.error_message}")
        if item.parsed.conflict_explanation:
            notices.append(f"Sense Conflict: {item.parsed.conflict_explanation}")
        for w in item.parsed.warnings:
            if w not in notices:
                notices.append(w)
        if item.is_duplicate_in_deck:
            notices.append("Duplicate detected in destination deck (will follow selected Duplicates policy).")

        if notices:
            self.item_warning_banner.setText("\n".join(notices))
            self.item_warning_banner.setVisible(True)
        else:
            self.item_warning_banner.setVisible(False)

        self._rebuild_image_candidates_ui(item)
        self._updating_detail_ui = False
        self._refresh_card_preview()

    def _rebuild_image_candidates_ui(self, item: EnrichedItem) -> None:
        # Clear existing candidate widgets
        for btn in self.image_btn_group.buttons():
            self.image_btn_group.removeButton(btn)
        while self.candidates_row.count():
            child = self.candidates_row.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Option -1: No Image
        no_img_box = QWidget()
        no_img_layout = QVBoxLayout(no_img_box)
        rb_none = QRadioButton("No Image")
        rb_none.setChecked(item.selected_image_index == -1)
        self.image_btn_group.addButton(rb_none, -1)
        no_img_layout.addWidget(rb_none)
        self.candidates_row.addWidget(no_img_box)

        for idx, cand in enumerate(item.image_candidates):
            card_w = QWidget()
            card_l = QVBoxLayout(card_w)
            thumb_lbl = QLabel()
            thumb_lbl.setFixedSize(110, 76)
            thumb_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb_lbl.setStyleSheet("border: 1px solid #cbd5e1; border-radius: 4px; background: #f8fafc;")
            if cand.thumbnail_bytes:
                pix = QPixmap()
                if pix.loadFromData(cand.thumbnail_bytes):
                    thumb_lbl.setPixmap(
                        pix.scaled(110, 76, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    )
                else:
                    thumb_lbl.setText(cand.provider.capitalize())
            else:
                thumb_lbl.setText(cand.provider.capitalize())
            card_l.addWidget(thumb_lbl)

            rb = QRadioButton(f"{cand.provider.capitalize()}\n{cand.photographer[:16]}")
            rb.setToolTip(cand.format_attribution())
            rb.setChecked(item.selected_image_index == idx)
            self.image_btn_group.addButton(rb, idx)
            card_l.addWidget(rb)
            self.candidates_row.addWidget(card_w)

        self.candidates_row.addStretch()

    def _on_image_candidate_selected(self, candidate_idx: int) -> None:
        if self._updating_detail_ui or not (0 <= self._current_enriched_idx < len(self.enriched_items)):
            return
        item = self.enriched_items[self._current_enriched_idx]
        item.selected_image_index = candidate_idx
        item.local_image_filename = ""
        self._refresh_card_preview()

    def _on_manual_image_search(self) -> None:
        if not (0 <= self._current_enriched_idx < len(self.enriched_items)):
            return
        query = self.edit_image_query.text().strip()
        if not query:
            return
        item = self.enriched_items[self._current_enriched_idx]
        item.image_query = query

        service, media_svc = self._build_enrichment_service()
        self.search_img_btn.setEnabled(False)
        self.status_label.setText(f"Searching Pexels/Pixabay for '{query}'...")

        self._img_worker = ImageSearchWorker(service, media_svc, query)
        self._img_worker.finished_search.connect(self._on_manual_image_search_done)
        self._img_worker.start()

    def _on_manual_image_search_done(self, candidates: list[ImageCandidate], warning: str) -> None:
        self.search_img_btn.setEnabled(True)
        if not (0 <= self._current_enriched_idx < len(self.enriched_items)):
            return
        item = self.enriched_items[self._current_enriched_idx]
        item.image_candidates = candidates
        item.selected_image_index = 0 if candidates else -1
        item.local_image_filename = ""
        self._rebuild_image_candidates_ui(item)
        self._refresh_card_preview()
        msg = f"Found {len(candidates)} image candidate(s)."
        if warning:
            msg += f" ({warning})"
        self.status_label.setText(msg)

    def _on_detail_field_edited(self) -> None:
        if self._updating_detail_ui or not (0 <= self._current_enriched_idx < len(self.enriched_items)):
            return
        item = self.enriched_items[self._current_enriched_idx]
        item.expression = self.edit_expression.text().strip()
        item.part_of_speech = self.edit_pos.text().strip()
        item.sense = self.edit_sense.text().strip()

        new_ipa = self.edit_ipa.text().strip()
        if new_ipa != item.ipa:
            item.ipa = new_ipa
            item.ipa_source = "User edited" if new_ipa else ""
            self.ipa_provenance_label.setText(item.ipa_source or "No IPA")

        item.definition_en = self.edit_def_en.text().strip()
        item.meaning_vi = self.edit_meaning_vi.text().strip()
        item.examples = [ed.text().strip() for ed in self.example_edits]
        item.audio_field = self.edit_audio.text().strip()
        item.duplicate_policy = str(self.item_dup_combo.currentData() or "")
        item.sense_key = compute_sense_key(item.expression, item.part_of_speech, item.sense)
        if item.error_message and item.definition_en and item.meaning_vi:
            item.error_message = ""

        self._refresh_enriched_list_labels()
        self._refresh_card_preview()

    def _set_preview_side(self, side: str) -> None:
        self._preview_side = side
        self.btn_preview_front.setChecked(side == "front")
        self.btn_preview_back.setChecked(side == "back")
        self._refresh_card_preview()

    def _refresh_card_preview(self) -> None:
        if not (0 <= self._current_enriched_idx < len(self.enriched_items)):
            self.preview_browser.setHtml("")
            return
        item = self.enriched_items[self._current_enriched_idx]
        locale = self.locale_combo.currentText()
        html_doc = render_card_preview_html(item, side=self._preview_side, tts_locale=locale)
        self.preview_browser.setHtml(html_doc)

    def _import_to_anki(self) -> None:
        if not self._mw or not getattr(self._mw, "col", None):
            QMessageBox.critical(self, "Anki Collection Unavailable", "Anki collection is not currently open.")
            return

        current_deck = self.deck_combo.currentText().strip() or "Vocabulary"
        chosen_deck, ok = QInputDialog.getText(
            self,
            "Confirm Destination Anki Deck",
            "Enter the destination deck name (use '::' for subdecks, e.g. 'Vocabulary::Crime & Law'):",
            QLineEdit.EchoMode.Normal,
            current_deck,
        )
        if not ok:
            return
        deck_name = chosen_deck.strip() or current_deck
        self.deck_combo.setEditText(deck_name)
        locale = self.locale_combo.currentText()

        self._config["default_deck"] = deck_name
        self._config["default_tts_locale"] = locale
        self._on_save_config(self._config)

        # Prefetch any selected full-size images on a background thread first so UI never blocks on HTTP I/O
        needs_download = [
            it
            for it in self.enriched_items
            if it.parsed.selected
            and it.selected_image is not None
            and not it.local_image_filename
            and not it.selected_image.downloaded_bytes
        ]
        if needs_download:
            self.import_btn.setEnabled(False)
            self.step2_import_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(len(needs_download))
            self.progress_bar.setValue(0)
            self.cancel_btn.setVisible(True)

            self._prefetch_worker = ImagePrefetchWorker(MediaPersistenceService(), self.enriched_items)
            self._prefetch_worker.progress_changed.connect(self._on_worker_progress)
            self._prefetch_worker.finished_prefetch.connect(self._on_prefetch_finished_run_import)
            self._prefetch_worker.start()
            return

        self._execute_collection_import()

    def _on_prefetch_finished_run_import(self, cancelled: bool) -> None:
        self.progress_bar.setVisible(False)
        self.cancel_btn.setVisible(False)
        if cancelled:
            self.import_btn.setEnabled(True)
            self.step2_import_btn.setEnabled(True)
            self.status_label.setText("Import cancelled before writing to Anki collection.")
            return
        self._execute_collection_import()

    def _execute_collection_import(self) -> None:
        deck_name = self.deck_combo.currentText().strip() or "Default"
        locale = self.locale_combo.currentText()
        policy = str(self.dup_policy_combo.currentData() or DuplicatePolicy.SKIP.value)

        self.import_btn.setEnabled(False)
        self.step2_import_btn.setEnabled(False)
        self.status_label.setText(f"Importing notes and saving media into '{deck_name}'...")

        try:
            summary: ImportSummary = import_enriched_items(
                col=self._mw.col,
                deck_name=deck_name,
                items=self.enriched_items,
                tts_locale=locale,
                default_duplicate_policy=policy,
            )
            if hasattr(self._mw, "reset"):
                self._mw.reset()
        except Exception as exc:
            self.import_btn.setEnabled(True)
            self.step2_import_btn.setEnabled(True)
            QMessageBox.critical(self, "Import Error", f"Failed to import notes:\n{exc}")
            return

        self.import_btn.setEnabled(True)
        self.step2_import_btn.setEnabled(True)
        refresh_deck_duplicate_flags(self._mw.col, deck_name, self.enriched_items)
        self._refresh_enriched_list_labels()

        report_lines = [
            f"Deck: {summary.deck_name}",
            f"Created: {summary.created_count}",
            f"Updated: {summary.updated_count}",
            f"Skipped (Duplicates): {summary.skipped_count}",
            f"Failed: {summary.failed_count}",
        ]
        if summary.item_logs:
            report_lines.append("\nDetails:")
            report_lines.extend(summary.item_logs[:20])

        self.status_label.setText(
            f"Import finished — Created: {summary.created_count}, "
            f"Updated: {summary.updated_count}, Skipped: {summary.skipped_count}, Failed: {summary.failed_count}"
        )
        QMessageBox.information(self, "Import Complete", "\n".join(report_lines))

        # Close the dialog if cards were successfully imported or processed
        if summary.created_count > 0 or summary.updated_count > 0 or summary.skipped_count > 0:
            self.accept()

