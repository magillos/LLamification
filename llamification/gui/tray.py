"""System tray icon for LLamification."""

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QIcon, QPixmap
from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

logger = logging.getLogger("llamification.tray")


def create_tray_icon(parent, main_window) -> QSystemTrayIcon:
    """
    Create and return a system tray icon with context menu.
    
    The parent is the MainWindow. The tray provides:
      - Show window
      - Stop/Start proxy
      - Quit
    """
    tray = QSystemTrayIcon(parent)

    # Load the LLamification SVG icon
    icon = _make_icon()
    tray.setIcon(icon)
    tray.setToolTip("LLamification — Ollama-compatible proxy")

    # Build context menu
    menu = QMenu()

    show_action = QAction("Show Window", parent)
    show_action.triggered.connect(lambda: _show_window(main_window))
    menu.addAction(show_action)

    menu.addSeparator()

    # Proxy toggle
    main_window._tray_toggle_action = QAction("Start Proxy", parent)
    main_window._tray_toggle_action.triggered.connect(lambda: _toggle_server(main_window))
    menu.addAction(main_window._tray_toggle_action)

    menu.addSeparator()

    quit_action = QAction("Quit", parent)
    quit_action.triggered.connect(lambda: _quit_app(parent, main_window))
    menu.addAction(quit_action)

    tray.setContextMenu(menu)

    # Single-click toggles window visibility, double-click always shows
    tray.activated.connect(
        lambda reason: _toggle_window(main_window)
        if reason == QSystemTrayIcon.ActivationReason.Trigger
        else _show_window(main_window)
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick
        else None
    )

    tray.show()
    return tray


def _make_icon() -> QIcon:
    """Load the LLamification SVG icon."""
    from pathlib import Path

    from PyQt6.QtGui import QPainter
    from PyQt6.QtSvg import QSvgRenderer

    svg_path = Path(__file__).parent / "LLamification.svg"
    renderer = QSvgRenderer(str(svg_path))

    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()

    return QIcon(pixmap)


def _toggle_window(main_window):
    """Toggle main window visibility: hide if visible, show if hidden."""
    if main_window.isVisible():
        main_window.hide()
    else:
        _show_window(main_window)


def _show_window(main_window):
    """Show and raise the main window."""
    main_window.show()
    main_window.raise_()
    main_window.activateWindow()


def _toggle_server(main_window):
    """Toggle proxy on/off from tray menu."""
    if main_window._server is not None:
        main_window._stop_server()
        main_window._tray_toggle_action.setText("Start Proxy")
    else:
        main_window._start_server()
        main_window._tray_toggle_action.setText("Stop Proxy")


def _quit_app(parent, main_window):
    """Quit the application cleanly."""
    main_window._cleanup()
    QApplication.quit()