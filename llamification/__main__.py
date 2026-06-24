"""LLamification entry point: python -m llamification"""

import argparse
import asyncio
import logging
import signal
import sys

from .config import (
    CONFIG_FILE,
    custom_provider_id,
    is_custom_provider,
    load_config,
    provider_display_name,
    resolve_base_url,
)
from .proxy.providers import get_provider
from .proxy.server import ProxyServer

logger = logging.getLogger("llamification")

# Pastel ANSI colour helpers
_PASTEL_GREEN = "\033[38;2;140;210;140m"
_PASTEL_RED = "\033[38;2;230;140;140m"
_PASTEL_CYAN = "\033[38;2;140;200;230m"
_PASTEL_YELLOW = "\033[38;2;230;210;140m"
_PASTEL_MAGENTA = "\033[38;2;210;160;220m"
_RESET = "\033[0m"


def _colourise(text: str, colour: str) -> str:
    return f"{colour}{text}{_RESET}"


def _parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="llamification",
        description="LLamification — Ollama-compatible LLM proxy",
    )
    parser.add_argument(
        "-g", "--gui",
        action="store_true",
        help="Launch the graphical interface (default is headless)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Launch LLamification — headless by default, GUI with -g flag."""
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.gui:
        _run_gui()
    else:
        _run_headless()


def _run_gui():
    """Launch the LLamification GUI application."""
    import signal as _signal

    from pathlib import Path

    from PyQt6.QtCore import QTimer
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication

    from .gui.main_window import MainWindow

    # Allow Ctrl+C in terminal to kill the app
    _signal.signal(_signal.SIGINT, _signal.SIG_DFL)

    app = QApplication(sys.argv)
    app.setApplicationName("LLamification")
    app.setOrganizationName("LLamification")

    # Set window icon from the bundled SVG icon
    icon_path = str(Path(__file__).resolve().parent / "gui" / "LLamification.svg")
    app.setWindowIcon(QIcon(icon_path))

    window = MainWindow()
    window.show()

    # Periodic timer so the Python signal handler gets a chance to run
    # during the Qt event loop (required for SIGINT processing)
    _sigint_timer = QTimer()
    _sigint_timer.start(250)

    sys.exit(app.exec())


def _run_headless():
    """Run LLamification in headless mode: load config, start proxy, serve forever."""

    # --- Load and validate config ---
    if not CONFIG_FILE.exists():
        print(f"{_colourise('✗', _PASTEL_RED)} No configuration file found at {_colourise(str(CONFIG_FILE), _PASTEL_CYAN)}")
        print(f"  Run {_colourise('python -m llamification -g', _PASTEL_YELLOW)} first to configure the application.")
        sys.exit(1)

    cfg = load_config()

    provider_key = cfg.get("provider", "")
    if not provider_key:
        print(f"{_colourise('✗', _PASTEL_RED)} No provider configured.")
        print(f"  Run {_colourise('python -m llamification -g', _PASTEL_YELLOW)} first to set up a provider.")
        sys.exit(1)

    api_keys = cfg.get("api_keys", {})
    api_key = api_keys.get(provider_key, "")
    if not api_key:
        p_name = f"'{provider_key}'"
        print(f"{_colourise('✗', _PASTEL_RED)} No API key configured for provider {_colourise(p_name, _PASTEL_CYAN)}.")
        print(f"  Run {_colourise('python -m llamification -g', _PASTEL_YELLOW)} first to set up an API key.")
        sys.exit(1)

    models_map = cfg.get("models", {})
    model = models_map.get(provider_key, "") if isinstance(models_map, dict) else ""
    # Backward compat: fall back to the old flat "model" key if present.
    if not model:
        model = cfg.get("model", "")
    if not model:
        print(f"{_colourise('✗', _PASTEL_RED)} No model selected.")
        print(f"  Run {_colourise('python -m llamification -g', _PASTEL_YELLOW)} first to select a model.")
        sys.exit(1)

    port = cfg.get("port", 11434)
    allow_override = cfg.get("allow_client_override", True)

    # Resolve the base URL for the selected provider. Built-ins have a hard-
    # coded default; custom providers ("custom:<id>") read their stored URL.
    base_url = resolve_base_url(provider_key, cfg)
    if is_custom_provider(provider_key) and not base_url:
        p_name = f"'{provider_display_name(provider_key, cfg)}'"
        print(f"{_colourise('✗', _PASTEL_RED)} Custom provider {_colourise(p_name, _PASTEL_CYAN)} has no base URL configured.")
        print(f"  Run {_colourise('python -m llamification -g', _PASTEL_YELLOW)} first to set it up.")
        sys.exit(1)

    # --- Create provider and server ---
    try:
        provider = get_provider(provider_key, api_key, base_url)
    except ValueError as e:
        print(f"{_colourise('✗', _PASTEL_RED)} Failed to create provider: {e}")
        sys.exit(1)

    server = ProxyServer(host="127.0.0.1", port=port)
    server.configure(provider, [model], model, allow_override)

    logger.info(f"LLamification headless mode — provider={provider_key}, model={model}, port={port}")
    print(f"{_colourise('✓', _PASTEL_GREEN)} LLamification proxy started on {_colourise(f'127.0.0.1:{port}', _PASTEL_CYAN)}")
    print(f"  {_colourise('Provider', _PASTEL_MAGENTA)} : {_colourise(provider_display_name(provider_key, cfg), _PASTEL_CYAN)}")
    print(f"  {_colourise('Model', _PASTEL_MAGENTA)}    : {_colourise(model, _PASTEL_CYAN)}")
    print(f"  {_colourise('Override', _PASTEL_MAGENTA)} : {_colourise('allowed', _PASTEL_GREEN) if allow_override else _colourise('disabled', _PASTEL_RED)}")
    print(f"  {_colourise('Press Ctrl+C to stop.', _PASTEL_YELLOW)}")

    # --- Run the asyncio event loop with clean shutdown ---
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info("Received shutdown signal, stopping…")
        stop_event.set()

    # Handle SIGINT and SIGTERM for clean shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        loop.run_until_complete(server.start())
        # Block until stop_event is set by signal handler
        loop.run_until_complete(stop_event.wait())
    except Exception as e:
        logger.error(f"Server error: {e}")
        print(f"{_colourise('✗', _PASTEL_RED)} Server error: {e}", file=sys.stderr)
    finally:
        logger.info("Shutting down…")
        loop.run_until_complete(server.stop())
        # Cancel remaining tasks
        for task in asyncio.all_tasks(loop):
            task.cancel()
        try:
            loop.run_until_complete(asyncio.sleep(0.1))
        except Exception:
            pass
        loop.close()
        print(f"{_colourise('✓', _PASTEL_GREEN)} LLamification proxy stopped.")


if __name__ == "__main__":
    main()
