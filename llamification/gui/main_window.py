"""LLamification main window — PyQt6 GUI."""

import asyncio
import logging
import threading
from typing import List, Optional

from PyQt6.QtCore import QMetaObject, QModelIndex, QRect, Qt, Q_ARG, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QColor, QIcon, QPainter, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QStyleOptionViewItem,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    ADD_CUSTOM_KEY,
    custom_provider_id,
    host_from_url,
    is_custom_provider,
    load_config,
    new_custom_id,
    provider_display_name,
    resolve_base_url,
    sanitize_custom_name,
    save_config,
)
from ..proxy.providers import get_provider, LLMProvider
from ..proxy.server import ProxyServer
from .tray import create_tray_icon

logger = logging.getLogger("llamification.gui")


class StarDelegate(QStyledItemDelegate):
    """Item delegate that paints a clickable ★/☆ to the left of each row.

    The underlying item text stays a clean model id (so it is safe to feed
    ``model_combo.currentText()`` into the proxy). The star is painted in a
    dedicated leading column and toggled on left-click via ``editorEvent``.
    """

    STAR_COLUMN = 22  # pixels reserved on the left for the star
    STAR_GLYPH_ON = "★"
    STAR_GLYPH_OFF = "☆"

    def __init__(self, parent, is_favourite, on_toggle):
        """``is_favourite`` and ``on_toggle`` are callables taking a model id."""
        super().__init__(parent)
        self._is_favourite = is_favourite
        self._on_toggle = on_toggle

    def _model_id(self, index: QModelIndex) -> str:
        return index.data(Qt.ItemDataRole.DisplayRole) or ""

    def paint(self, painter: QPainter, option, index: QModelIndex):
        # Draw the row text shifted right to make room for the star column.
        text_rect = QRect(option.rect)
        text_rect.setLeft(text_rect.left() + self.STAR_COLUMN)
        text_opt = QStyleOptionViewItem(option)
        text_opt.rect = text_rect
        super().paint(painter, text_opt, index)

        # Draw the star glyph in the leading column.
        starred = self._is_favourite(self._model_id(index))
        glyph = self.STAR_GLYPH_ON if starred else self.STAR_GLYPH_OFF
        color = QColor("#d4a017") if starred else QColor("#aaaaaa")
        painter.save()
        painter.setPen(color)
        font = painter.font()
        font.setBold(starred)
        painter.setFont(font)
        painter.drawText(
            QRect(option.rect.left(), option.rect.top(),
                  self.STAR_COLUMN, option.rect.height()),
            Qt.AlignmentFlag.AlignCenter, glyph,
        )
        painter.restore()

    def editorEvent(self, event, model, option, index: QModelIndex) -> bool:
        # Only react to left-button presses inside the star column.
        if (event.type() == event.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
                and event.position().toPoint().x() - option.rect.left()
                <= self.STAR_COLUMN):
            self._on_toggle(self._model_id(index))
            return True  # consume the click so the row is not also selected
        return super().editorEvent(event, model, option, index)


class MainWindow(QMainWindow):
    """Main application window."""

    # Thread-safe signals (can be emitted from any thread)
    log_signal = pyqtSignal(str)
    models_fetched_signal = pyqtSignal(list)
    reset_refresh_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._server: Optional[ProxyServer] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server_thread: Optional[threading.Thread] = None
        self._server_stopping = threading.Event()  # signals loop to stop
        self._models_cache: List[str] = []
        self._provider_instance: Optional[LLMProvider] = None
        # Last real provider the user settled on; used to revert the combo if
        # the "Add Provider…" sentinel is cancelled.
        self._last_provider_key: str = ""

        self.setWindowTitle("LLamification")
        self.setMinimumSize(600, 500)

        self._build_ui()
        self._load_settings()
        self._setup_signals()
        self._setup_logging()

        # Connect signals. Use ``activated`` (not ``currentIndexChanged``) for
        # the provider combo so re-picking the current item still fires — the
        # "Add Provider…" sentinel can be the already-selected item on first
        # run, and a plain index-change signal would never emit for it.
        self.provider_combo.activated.connect(self._on_provider_changed)
        self.refresh_btn.clicked.connect(self._on_refresh_models)
        self.start_btn.clicked.connect(self._on_toggle_server)
        self.allow_override_cb.stateChanged.connect(self._on_override_setting_changed)
        self.show_tray_cb.stateChanged.connect(self._on_tray_setting_changed)
        self.model_combo.currentIndexChanged.connect(self._on_model_index_changed)

        # System tray (only created if enabled in config)
        self._tray_icon = None
        if load_config().get("show_tray_icon", True):
            self._tray_icon = create_tray_icon(self, self)

        # Auto-fetch models for the initial provider on startup.
        self._on_refresh_models()

    def _setup_signals(self):
        """Connect thread-safe signals to their slots."""
        self.log_signal.connect(self._append_log)
        self.models_fetched_signal.connect(self._populate_models)
        self.reset_refresh_signal.connect(self._reset_refresh_btn)

    def _build_ui(self):
        """Construct the UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # --- Provider & Model group ---
        group = QGroupBox("Provider Configuration")
        glayout = QVBoxLayout(group)

        # Provider row
        prow = QHBoxLayout()
        prow.addWidget(QLabel("Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.setMinimumWidth(200)
        prow.addWidget(self.provider_combo)
        prow.addStretch()
        glayout.addLayout(prow)

        # Custom Base URL row (shown only when a custom provider is selected).
        # Includes an inline ✕ remove button for the active custom provider.
        self.custom_url_row = QHBoxLayout()
        self.custom_url_label = QLabel("Base URL:")
        self.custom_url_row.addWidget(self.custom_url_label)
        self.custom_url_edit = QLineEdit()
        self.custom_url_edit.setPlaceholderText("https://your-provider.com/v1")
        self.custom_url_edit.setMinimumWidth(350)
        self.custom_url_row.addWidget(self.custom_url_edit)
        self.edit_custom_btn = QPushButton("✎ Edit")
        self.edit_custom_btn.setToolTip("Edit this custom provider's name and base URL")
        self.custom_url_row.addWidget(self.edit_custom_btn)
        self.remove_custom_btn = QPushButton("🗑 Remove")
        self.remove_custom_btn.setToolTip("Delete this custom provider")
        self.custom_url_row.addWidget(self.remove_custom_btn)
        self.custom_url_row.addStretch()
        glayout.addLayout(self.custom_url_row)
        # Track every widget in this row so visibility can be toggled together.
        self.custom_url_row_widgets = [
            self.custom_url_label,
            self.custom_url_edit,
            self.edit_custom_btn,
            self.remove_custom_btn,
        ]
        for w in self.custom_url_row_widgets:
            w.setVisible(False)

        # Model row
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(350)
        self.model_combo.setEditable(True)
        self.model_combo.setPlaceholderText("Select or type a model...")
        # Clickable ★/☆ next to each model; stars are favourites and float to top
        self.model_combo.setItemDelegate(
            StarDelegate(self.model_combo, self._is_favourite, self._toggle_favourite)
        )

        # Configure a custom QCompleter linked to the combobox model
        completer = QCompleter(self.model_combo.model(), self.model_combo)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.model_combo.setCompleter(completer)

        # Inline ✕ clear button inside the combo's text field.
        line_edit = self.model_combo.lineEdit()
        line_edit.setClearButtonEnabled(True)
        # Re-focus the field after the inline ✕ is clicked.
        from PyQt6.QtWidgets import QToolButton
        clear_btn = line_edit.findChild(QToolButton)
        if clear_btn is not None:
            clear_btn.clicked.connect(self._on_model_cleared)
        line_edit.textEdited.connect(self._on_model_text_edited)

        mrow.addWidget(self.model_combo)
        self.refresh_btn = QPushButton("⟳ Refresh Models")
        mrow.addWidget(self.refresh_btn)
        mrow.addStretch()
        glayout.addLayout(mrow)

        # Allow client override checkbox row
        self.allow_override_cb = QCheckBox("Allow client to select model (pass-through)")
        self.allow_override_cb.setChecked(True)
        glayout.addWidget(self.allow_override_cb)

        layout.addWidget(group)

        # --- Server group ---
        srv_group = QGroupBox("Server")
        srv_layout = QHBoxLayout(srv_group)
        srv_layout.addWidget(QLabel("Port:"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(11434)
        srv_layout.addWidget(self.port_spin)

        self.status_label = QLabel("● Stopped")
        self.status_label.setStyleSheet("color: #888; font-weight: bold;")
        srv_layout.addWidget(self.status_label)

        srv_layout.addStretch()

        self.start_btn = QPushButton("▶ Start Proxy")
        self.start_btn.setMinimumWidth(140)
        srv_layout.addWidget(self.start_btn)
        layout.addWidget(srv_group)

        # Tray icon toggle (lives in the Server group area)
        self.show_tray_cb = QCheckBox("Show tray icon")
        self.show_tray_cb.setToolTip("Show a system tray icon (hide to disable)")
        layout.addWidget(self.show_tray_cb)

        # --- Log area ---
        log_label = QLabel("Log:")
        layout.addWidget(log_label)
        self.log_area = QPlainTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumBlockCount(1000)
        self.log_area.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_area.setStyleSheet(
            "font-family: 'Consolas', 'Courier New', monospace; font-size: 11px;"
        )
        layout.addWidget(self.log_area, stretch=1)

        # Wire the custom-URL row actions.
        self.remove_custom_btn.clicked.connect(self._on_remove_custom_clicked)
        self.edit_custom_btn.clicked.connect(self._on_edit_custom_clicked)
        # Apply inline URL edits to the running proxy only (not persisted).
        self.custom_url_edit.editingFinished.connect(self._on_custom_url_edited)

    # --- Provider combo (built-ins + dynamic custom entries) ---

    def _rebuild_provider_combo(self, preserve_key: Optional[str] = None):
        """Rebuild the provider dropdown from stored custom providers.

        There are no built-in providers — every entry is user-defined.
        ``preserve_key`` (if given) is re-selected after the rebuild; otherwise
        the first custom provider is selected, or the "Add Provider…"
        sentinel when none exist yet. Signals are blocked so this does not
        trigger ``_on_provider_changed`` (and so landing on the sentinel does
        not auto-launch the add dialog).
        """
        cfg = load_config()
        self.provider_combo.blockSignals(True)
        try:
            self.provider_combo.clear()

            # User-defined custom providers
            custom_providers = cfg.get("custom_providers", {})

            # Sentinel entry to launch the "add custom" dialog. When there are
            # already providers, a separator is inserted before it.
            first_custom_idx = self.provider_combo.count()
            for cp_id, info in custom_providers.items():
                key = f"custom:{cp_id}"
                self.provider_combo.addItem(info.get("name") or key, key)

            self.provider_combo.insertSeparator(self.provider_combo.count())
            self.provider_combo.addItem("➕ Add Provider…", ADD_CUSTOM_KEY)

            # Restore selection
            target = preserve_key or self._last_provider_key
            idx = self.provider_combo.findData(target)
            if idx < 0:
                # Fall back to the first custom provider if any; otherwise land
                # on the sentinel but keep _last_provider_key empty so the
                # sentinel is never treated as a real provider.
                if custom_providers:
                    idx = first_custom_idx
                    target = self.provider_combo.itemData(idx)
                else:
                    idx = self.provider_combo.count() - 1
                    target = ""
            self.provider_combo.setCurrentIndex(idx)
            self._last_provider_key = target
        finally:
            self.provider_combo.blockSignals(False)

        # Keep the URL-row visibility in sync with the (possibly new)
        # selection, since the signal-driven refresh above was blocked.
        self._refresh_custom_url_ui()

    def _refresh_custom_url_ui(self):
        """Show the Base URL row + remove button only for custom providers."""
        provider_key = self.provider_combo.currentData()
        is_custom = is_custom_provider(provider_key)
        for w in self.custom_url_row_widgets:
            w.setVisible(is_custom)
        return is_custom

    def _on_custom_url_edited(self):
        """Apply an inline Base URL edit to the running proxy *only*.

        The change is **not** persisted to config — it resets when the app
        restarts or the provider is re-selected via the dropdown. Use the
        Edit Provider dialog to permanently change the saved URL.
        """
        provider_key = self.provider_combo.currentData()
        if not is_custom_provider(provider_key):
            return
        new_url = self.custom_url_edit.text().strip()
        if self._provider_instance:
            self._provider_instance.base_url = new_url.rstrip("/")

    def _on_remove_custom_clicked(self):
        """Delete the currently selected custom provider (after confirmation)."""
        provider_key = self.provider_combo.currentData()
        if not is_custom_provider(provider_key):
            return
        cfg = load_config()
        cp_id = custom_provider_id(provider_key)
        cp = cfg.get("custom_providers", {})
        info = cp.get(cp_id, {})
        name = info.get("name", provider_key)

        confirm = QMessageBox.question(
            self,
            "Remove custom provider",
            f"Remove the custom provider '{name}'?\n"
            f"This also deletes its saved API key and favourite models.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        # Stop the server if it is running for this provider.
        if self._server is not None:
            self._stop_server()

        cp.pop(cp_id, None)
        cfg.get("api_keys", {}).pop(provider_key, None)
        cfg.get("favourite_models", {}).pop(provider_key, None)
        if cfg.get("provider") == provider_key:
            cfg["provider"] = ""
        if self._last_provider_key == provider_key:
            self._last_provider_key = ""
        save_config(cfg)

        self._provider_instance = None
        self._models_cache = []
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.blockSignals(False)

        self._rebuild_provider_combo()
        # Only refresh the now-active provider if one still exists. When the
        # last provider was removed the combo lands on the sentinel; calling
        # _on_provider_changed() there would spuriously launch the add dialog.
        if self.provider_combo.currentData() != ADD_CUSTOM_KEY:
            self._on_provider_changed()
        else:
            # No providers left — clear the UI fields to a clean empty state.
            self.model_combo.blockSignals(True)
            self.model_combo.clear()
            self.model_combo.blockSignals(False)
        self.log_signal.emit(f"🗑 Removed custom provider '{name}'.")

    def _prompt_custom_provider(
        self,
        title: str,
        existing_names,
        initial_name: str = "",
        initial_url: str = "",
        initial_key: str = "",
        exclude_name: str = "",
    ) -> Optional[tuple]:
        """Open a dialog to capture a custom provider's name + base URL.

        Shared by Add and Edit. The Name field auto-fills with the URL's host
        (e.g. ``https://api.kilo.ai/...`` -> ``api.kilo.ai``) until the user
        types a custom name themselves. ``initial_name`` / ``initial_url``
        pre-fill the fields (for editing). ``exclude_name`` is removed from
        ``existing_names`` before de-duplication so re-saving the same name on
        edit does not gain a numeric suffix. Returns ``(name, base_url)`` on
        accept, or ``None`` if cancelled.
        """
        # Exclude the entry currently being edited from the uniqueness check.
        existing = [n for n in existing_names if n and n != exclude_name]

        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setMinimumWidth(420)
        form = QFormLayout(dlg)
        name_edit = QLineEdit(initial_name)
        name_edit.setPlaceholderText("auto from host, or type your own")
        url_edit = QLineEdit(initial_url)
        url_edit.setPlaceholderText("https://your-provider.com/v1")
        key_edit = QLineEdit(initial_key)
        key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        key_edit.setPlaceholderText("Optional — leave blank for local/no-auth providers")
        form.addRow("Name:", name_edit)
        form.addRow("Base URL:", url_edit)
        form.addRow("API Key:", key_edit)

        # Track whether the user has typed a name manually. Until they do, the
        # name field mirrors the URL's host. When editing an entry that already
        # has a name, we treat the pre-filled name as user-given so the URL
        # host does not silently overwrite it.
        state = {"user_named": bool(initial_name.strip())}

        def _on_url_changed(text: str):
            if state["user_named"]:
                return
            host = host_from_url(text)
            name_edit.blockSignals(True)
            name_edit.setText(host)
            name_edit.blockSignals(False)

        def _on_name_edited(_text: str):
            state["user_named"] = bool(name_edit.text().strip())

        url_edit.textChanged.connect(_on_url_changed)
        name_edit.textEdited.connect(_on_name_edited)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        url_edit.setFocus()
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None

        # Fall back to the URL host if the name field was left blank.
        name = name_edit.text().strip() or host_from_url(url_edit.text())
        name = sanitize_custom_name(name, existing)
        base_url = url_edit.text().strip()
        if not base_url:
            QMessageBox.warning(self, "Base URL required",
                                "A base URL is required for a custom provider.")
            return None
        api_key = key_edit.text().strip()
        return name, base_url, api_key

    def _prompt_add_custom(self) -> Optional[tuple]:
        """Open a dialog to capture a new custom provider's name + base URL."""
        cfg = load_config()
        existing_names = [
            (info or {}).get("name", "")
            for info in cfg.get("custom_providers", {}).values()
        ]
        return self._prompt_custom_provider(
            "Add Provider", existing_names
        )

    def _on_edit_custom_clicked(self):
        """Open the edit dialog for the currently selected custom provider."""
        provider_key = self.provider_combo.currentData()
        if not is_custom_provider(provider_key):
            return
        cfg = load_config()
        cp_id = custom_provider_id(provider_key)
        cp = cfg.get("custom_providers", {})
        info = cp.get(cp_id, {})
        current_name = info.get("name", "")
        current_url = info.get("base_url", "")

        existing_names = [
            (i or {}).get("name", "")
            for k, i in cp.items()
        ]
        result = self._prompt_custom_provider(
            "Edit Provider",
            existing_names,
            initial_name=current_name,
            initial_url=current_url,
            initial_key=cfg.get("api_keys", {}).get(provider_key, ""),
            exclude_name=current_name,
        )
        if result is None:
            return

        name, base_url, api_key = result

        # Reload in case config changed since the dialog opened.
        cfg = load_config()
        cp = cfg.setdefault("custom_providers", {})
        if cp_id not in cp:
            # Was deleted while the dialog was open — abort gracefully.
            self.log_signal.emit("⚠ Custom provider no longer exists; edit cancelled.")
            return
        cp[cp_id]["name"] = name
        cp[cp_id]["base_url"] = base_url
        # Persist the API key together with the name/base URL.
        api_keys = cfg.setdefault("api_keys", {})
        api_keys[provider_key] = api_key
        save_config(cfg)

        # Reflect changes in the UI.
        self.custom_url_edit.setText(base_url)
        idx = self.provider_combo.findData(provider_key)
        if idx >= 0:
            self.provider_combo.setItemText(idx, name)
        if self._provider_instance:
            self._provider_instance.base_url = base_url.rstrip("/")
            self._provider_instance.api_key = api_key

        self.log_signal.emit(f"✎ Updated custom provider '{name}' ({base_url}).")

    def _on_add_custom(self):
        """Handle the 'Add Provider…' sentinel selection."""
        result = self._prompt_add_custom()
        if result is None:
            # Revert to the previous real selection.
            self._rebuild_provider_combo(preserve_key=self._last_provider_key)
            self._refresh_custom_url_ui()
            return

        name, base_url, api_key = result
        if not base_url:
            QMessageBox.warning(self, "Base URL required",
                                "A base URL is required to add a custom provider.")
            self._rebuild_provider_combo(preserve_key=self._last_provider_key)
            self._refresh_custom_url_ui()
            return

        # Save the outgoing provider's model under its own key **before**
        # switching to the new provider, so the stale model does not leak into
        # the new provider's config entry.
        self._save_current_model_for_provider(self._last_provider_key)

        cfg = load_config()
        cp = cfg.setdefault("custom_providers", {})
        cp_id = new_custom_id(cp)
        cp[cp_id] = {"name": name, "base_url": base_url}
        # Persist the API key together with the new provider.
        new_key = f"custom:{cp_id}"
        cfg.setdefault("api_keys", {})[new_key] = api_key
        save_config(cfg)

        new_key = f"custom:{cp_id}"
        self._rebuild_provider_combo(preserve_key=new_key)
        self.log_signal.emit(f"➕ Added custom provider '{name}' ({base_url}).")
        # Behave as if the user just switched to it.
        self._on_provider_changed()

    def _setup_logging(self):
        """Redirect logging to the GUI log area."""
        class GuiLogHandler(logging.Handler):
            def __init__(self, signal):
                super().__init__()
                self.signal = signal

            def emit(self, record):
                msg = self.format(record)
                self.signal.emit(msg)

        handler = GuiLogHandler(self.log_signal)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        handler.setFormatter(formatter)
        logging.getLogger("llamification").addHandler(handler)
        logging.getLogger("llamification").setLevel(logging.INFO)

    @pyqtSlot(str)
    def _append_log(self, msg: str):
        """Append a message to the log area (called via signal from any thread)."""
        self.log_area.appendPlainText(msg)

    def _load_settings(self):
        """Load saved config into UI fields."""
        cfg = load_config()

        # Populate the combo from config, then select the saved provider.
        self._rebuild_provider_combo()
        provider_key = cfg.get("provider", "")
        idx = self.provider_combo.findData(provider_key)
        if idx < 0:
            # Saved provider no longer exists (e.g. deleted custom) — fall back.
            provider_key = ""
            idx = self.provider_combo.findData(provider_key)
        if idx >= 0:
            self.provider_combo.setCurrentIndex(idx)
        current = self.provider_combo.currentData()
        # Never let the sentinel ("Add Provider…") leak into the active
        # provider slot; treat an empty-state combo as no provider.
        self._last_provider_key = "" if current == ADD_CUSTOM_KEY else current

        # Load custom URL for the active provider (API key lives in config
        # and is edited via the Add/Edit dialog, not a main-window field).
        if is_custom_provider(self._last_provider_key):
            self.custom_url_edit.setText(resolve_base_url(self._last_provider_key, cfg))

        self.port_spin.setValue(cfg.get("port", 11434))

        # Show the URL row if a custom provider is selected
        self._refresh_custom_url_ui()

        # Load model combo if saved for the current provider
        models = cfg.get("models", {})
        saved_model = models.get(self._last_provider_key, "") if isinstance(models, dict) else ""
        if saved_model:
            self.model_combo.setCurrentText(saved_model)

        self.allow_override_cb.setChecked(cfg.get("allow_client_override", True))

        self.show_tray_cb.setChecked(cfg.get("show_tray_icon", True))

    def _save_settings(self):
        """Save current UI state to config."""
        cfg = load_config()  # load existing to preserve other provider data
        provider_key = self.provider_combo.currentData()
        if provider_key == ADD_CUSTOM_KEY:
            # Should never be the active key, but guard anyway.
            provider_key = self._last_provider_key

        cfg["provider"] = provider_key
        cfg["port"] = self.port_spin.value()

        # Store the selected model per-provider so switching providers restores it
        models = cfg.setdefault("models", {})
        models[provider_key] = self.model_combo.currentText().strip()
        cfg["models"] = models

        cfg["allow_client_override"] = self.allow_override_cb.isChecked()
        cfg["show_tray_icon"] = self.show_tray_cb.isChecked()

        save_config(cfg)
    def _on_provider_changed(self):
        """Handle provider selection: sentinel → add dialog; otherwise load
        saved key/URL/model and refresh the model list."""
        provider_key = self.provider_combo.currentData()

        # The "Add Provider…" sentinel launches the dialog and never
        # becomes the active provider.
        if provider_key == ADD_CUSTOM_KEY:
            self._on_add_custom()
            return

        # Persist the outgoing provider's current model before switching.
        self._save_current_model_for_provider(self._last_provider_key)

        self._last_provider_key = provider_key
        is_custom = self._refresh_custom_url_ui()

        # Load saved Base URL for custom providers (API key is read from
        # config on demand via _current_api_key()).
        cfg = load_config()
        if is_custom:
            self.custom_url_edit.setText(resolve_base_url(provider_key, cfg))

        # Clear stale model list from the previous provider.
        self._models_cache = []
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.blockSignals(False)

        # Restore this provider's last model as placeholder text (if any).
        # Once models are fetched, _populate_models will select it properly.
        models_map = cfg.get("models", {})
        saved = models_map.get(provider_key, "") if isinstance(models_map, dict) else ""
        if saved:
            self.model_combo.setCurrentText(saved)

        # Auto-fetch models for the newly selected provider.
        self._on_refresh_models()

    def _save_current_model_for_provider(self, provider_key: str):
        """Save the currently selected model for the specified provider key."""
        current_text = self.model_combo.currentText().strip()
        if not current_text or not provider_key:
            return
        cfg = load_config()
        models = cfg.setdefault("models", {})
        models[provider_key] = current_text
        cfg["models"] = models
        save_config(cfg)

    def _on_model_cleared(self):
        """After the inline ✕ clears the field, focus it so typing can resume."""
        self.model_combo.lineEdit().setFocus()
        self.log_signal.emit("✕ Model selection cleared.")

    def _on_model_text_edited(self, text: str):
        """Treat typing in the box as re-selecting a model (matches current
        text once it matches a known model id)."""
        model = text.strip()
        if model and model in self._models_cache:
            self.update_active_model(model)

    # --- Favourite models (per-provider, persisted in config) ---

    def _current_provider_key(self) -> str:
        data = self.provider_combo.currentData()
        if data == ADD_CUSTOM_KEY or data is None:
            return self._last_provider_key or ""
        return data

    def _favourites_for(self, provider_key: str) -> list:
        """Return the list of favourited model ids for the given provider."""
        cfg = load_config()
        favs = cfg.get("favourite_models", {})
        if not isinstance(favs, dict):
            return []
        entries = favs.get(provider_key, [])
        return entries if isinstance(entries, list) else []

    def _is_favourite(self, model_id: str) -> bool:
        return model_id in self._favourites_for(self._current_provider_key())

    def _sorted_model_ids(self, model_ids: list) -> list:
        """Lift favourites to the top, preserving the original order within
        each group (favourites first, then the rest). Stable by construction."""
        favourites = set(self._favourites_for(self._current_provider_key()))
        starred = [m for m in model_ids if m in favourites]
        rest = [m for m in model_ids if m not in favourites]
        return starred + rest

    def _toggle_favourite(self, model_id: str):
        """Flip a model's favourite status, persist, and re-sort the combo."""
        if not model_id:
            return
        cfg = load_config()
        favs = cfg.setdefault("favourite_models", {})
        provider_key = self._current_provider_key()
        entries = favs.get(provider_key, [])
        if model_id in entries:
            entries = [m for m in entries if m != model_id]
        else:
            entries = entries + [model_id]
        favs[provider_key] = entries
        save_config(cfg)
        self._resort_models()

    def _resort_models(self):
        """Re-sort the combo's existing entries using the current favourites,
        preserving the active selection text and the open popup state."""
        if not self._models_cache:
            return
        was_popup = self.model_combo.view().isVisible()
        self._populate_models(list(self._models_cache))
        if was_popup:
            self.model_combo.showPopup()

    def _current_api_key(self) -> str:
        """Return the saved API key for the currently selected provider."""
        provider_key = self.provider_combo.currentData()
        if provider_key == ADD_CUSTOM_KEY or provider_key is None:
            provider_key = self._last_provider_key
        cfg = load_config()
        return cfg.get("api_keys", {}).get(provider_key, "")

    def _current_base_url(self) -> str:
        """Resolve the base URL for the currently selected provider.

        Reads the (possibly edited) URL field for the active custom provider.
        Returns "" when no provider is selected.
        """
        provider_key = self.provider_combo.currentData()
        if is_custom_provider(provider_key):
            return self.custom_url_edit.text().strip()
        return ""

    def _on_refresh_models(self):
        """Fetch models from the selected provider in a background thread."""
        provider_key = self.provider_combo.currentData()
        if provider_key == ADD_CUSTOM_KEY:
            provider_key = self._last_provider_key

        # No real provider selected (first run, or the last one was removed) —
        # nothing to fetch. Stay quiet so a clean first run logs nothing.
        if not is_custom_provider(provider_key):
            return

        self.refresh_btn.setEnabled(False)
        self.refresh_btn.setText("⟳ Fetching...")
        self.log_signal.emit("Fetching available models...")

        api_key = self._current_api_key()

        base_url = ""
        if is_custom_provider(provider_key):
            base_url = self.custom_url_edit.text().strip()
            if not base_url:
                self.log_signal.emit("⚠ Custom URL is required.")
                self.refresh_btn.setEnabled(True)
                self.refresh_btn.setText("⟳ Refresh Models")
                return

        # Run model fetch in a thread
        threading.Thread(
            target=self._fetch_models_thread,
            args=(provider_key, api_key, base_url),
            daemon=True,
        ).start()

    def _fetch_models_thread(self, provider_key: str, api_key: str, base_url: str):
        """Fetch models in a background thread (runs async)."""
        try:
            provider = get_provider(provider_key, api_key, base_url)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            models = loop.run_until_complete(provider.fetch_models())
            loop.close()

            model_ids = [m["id"] for m in models]
            self.models_fetched_signal.emit(model_ids)
            self._provider_instance = provider
            self.log_signal.emit(f"✓ Fetched {len(model_ids)} models.")
        except Exception as e:
            self.log_signal.emit(f"✗ Failed to fetch models: {e}")
        finally:
            self.reset_refresh_signal.emit()

    @pyqtSlot(list)
    def _populate_models(self, model_ids: list):
        """Populate the model combo box with fetched models."""
        current = self.model_combo.currentText().strip()

        # Sort favourites to the top (per current provider).
        model_ids = self._sorted_model_ids(model_ids)

        # Block signals temporarily to prevent redundant configure triggers
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(model_ids)
        if current:
            idx = self.model_combo.findText(current)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
            else:
                # The saved model doesn't exist in the fetched list (e.g. a
                # stale model from a different provider leaked in). Clear the
                # combo text so no model is shown as selected.
                self.model_combo.setCurrentIndex(-1)
        self.model_combo.blockSignals(False)

        self._models_cache = model_ids
        # If server is running, notify it of model change
        if self._server and model_ids:
            active = self.model_combo.currentText().strip()
            if self._provider_instance:
                self._server.configure(self._provider_instance, model_ids, active, self.allow_override_cb.isChecked())

    @pyqtSlot()
    def _reset_refresh_btn(self):
        """Re-enable the refresh button."""
        self.refresh_btn.setEnabled(True)
        self.refresh_btn.setText("⟳ Refresh Models")

    def _on_toggle_server(self):
        """Start or stop the proxy server."""
        if self._server is not None:
            self._stop_server()
        else:
            self._start_server()

    def _start_server(self):
        """Start the proxy server in a background thread."""
        api_key = self._current_api_key()

        provider_key = self.provider_combo.currentData()
        if provider_key == ADD_CUSTOM_KEY:
            provider_key = self._last_provider_key

        base_url = ""
        if is_custom_provider(provider_key):
            base_url = self.custom_url_edit.text().strip()
            if not base_url:
                self.log_signal.emit("⚠ Custom URL is required.")
                return

        port = self.port_spin.value()
        model = self.model_combo.currentText().strip()
        if not model:
            self.log_signal.emit("⚠ No model selected.")
            return

        # Get or create provider instance
        try:
            if not self._provider_instance:
                self._provider_instance = get_provider(provider_key, api_key, base_url)
            else:
                # Update API key in case it changed
                self._provider_instance.api_key = api_key
        except ValueError as e:
            self.log_signal.emit(f"✗ {e}")
            return

        # Create event loop and server
        self._loop = asyncio.new_event_loop()
        self._server = ProxyServer(host="127.0.0.1", port=port)

        models = self._models_cache if self._models_cache else [model]
        self._server.configure(self._provider_instance, models, model, self.allow_override_cb.isChecked())

        # Auto-fetch models in the background if they haven't been fetched yet
        if not self._models_cache:
            self.log_signal.emit("Auto-fetching model list in the background...")
            threading.Thread(
                target=self._fetch_models_thread,
                args=(provider_key, api_key, base_url),
                daemon=True,
            ).start()

        # Save settings on start
        self._save_settings()

        # Run server in a thread
        self._server_thread = threading.Thread(
            target=self._run_server_loop,
            args=(self._loop, self._server),
            daemon=True,
        )
        self._server_thread.start()

        cfg = load_config()
        display = provider_display_name(provider_key, cfg)
        self.start_btn.setText("■ Stop Proxy")
        self.status_label.setText("● Running")
        self.status_label.setStyleSheet("color: #2a2; font-weight: bold;")
        self.log_signal.emit(f"✓ Proxy started on port {port}")
        self.log_signal.emit(f"  Serving model '{model}' via {display}")
        # Update tray toggle action
        if hasattr(self, "_tray_toggle_action"):
            self._tray_toggle_action.setText("Stop Proxy")

    def _run_server_loop(self, loop: asyncio.AbstractEventLoop, server: ProxyServer):
        """Run the asyncio event loop with the proxy server."""
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(server.start())
            # Run until stopped — check the event periodically
            while not self._server_stopping.is_set():
                loop.run_until_complete(asyncio.sleep(0.5))
        except Exception as e:
            logger.error(f"Server error: {e}")
            self.log_signal.emit(f"✗ Server error: {e}")
        finally:
            # Cancel all pending tasks and close
            try:
                for task in asyncio.all_tasks(loop):
                    task.cancel()
                loop.run_until_complete(asyncio.sleep(0.1))
            except Exception:
                pass
            loop.close()

    def _stop_server(self):
        """Stop the proxy server."""
        if self._server and self._loop:
            # Queue the server shutdown, then signal the loop to exit
            async def async_stop():
                await self._server.stop()

            try:
                asyncio.run_coroutine_threadsafe(async_stop(), self._loop)
            except RuntimeError:
                pass  # loop already closed

            self._server_stopping.set()

            # Wait briefly for the thread to finish
            if self._server_thread and self._server_thread.is_alive():
                self._server_thread.join(timeout=3)

        self._server = None
        self._server_thread = None
        self._loop = None
        self._server_stopping.clear()

        self.start_btn.setText("▶ Start Proxy")
        self.status_label.setText("● Stopped")
        self.status_label.setStyleSheet("color: #888; font-weight: bold;")
        self.log_signal.emit("■ Proxy stopped.")
        # Update tray toggle action
        if hasattr(self, "_tray_toggle_action"):
            self._tray_toggle_action.setText("Start Proxy")

    def update_active_model(self, model: str):
        """Update the active model on-the-fly (called from tray or combo)."""
        if self._server and model:
            self._server.configure(
                self._provider_instance,
                self._models_cache or [model],
                model,
                self.allow_override_cb.isChecked(),
            )
        # Block signals temporarily to prevent loop
        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentText(model)
        self.model_combo.blockSignals(False)
        
        self.log_signal.emit(f"→ Switched active model to '{model}'")

    def _on_override_setting_changed(self):
        """Handle toggle of client model override checkbox."""
        self._save_settings()
        if self._server:
            active = self.model_combo.currentText().strip()
            allow_override = self.allow_override_cb.isChecked()
            self._server.configure(
                self._provider_instance,
                self._models_cache or [active],
                active,
                allow_override
            )
            self.log_signal.emit(f"→ Client model override set to: {allow_override}")

    def _on_tray_setting_changed(self):
        """Handle toggle of the tray icon checkbox."""
        show = self.show_tray_cb.isChecked()
        if show and self._tray_icon is None:
            self._tray_icon = create_tray_icon(self, self)
            self.log_signal.emit("✓ Tray icon enabled.")
        elif not show and self._tray_icon is not None:
            self._tray_icon.hide()
            self._tray_icon.deleteLater()
            self._tray_icon = None
            self.log_signal.emit("✓ Tray icon disabled.")
        self._save_settings()

    def _on_model_index_changed(self):
        """Handle model selection change from the dropdown."""
        model = self.model_combo.currentText().strip()
        if model:
            self.update_active_model(model)

    def closeEvent(self, event):
        """Handle window close."""
        self._save_settings()
        if hasattr(self, "_tray_icon") and self._tray_icon and self._tray_icon.isVisible():
            self.hide()
            event.ignore()
        else:
            self._cleanup()
            event.accept()

    def _cleanup(self):
        """Clean up server on exit."""
        if self._server is not None:
            self._stop_server()
