"""Configuration persistence for LLamification."""

import json
import re
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

CONFIG_DIR = Path.home() / ".config" / "llamification"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Built-in providers are always available. Custom providers are stored in
# ``custom_providers`` and addressed by the combo key ``"custom:<id>"``.
CUSTOM_PROVIDER_PREFIX = "custom:"

DEFAULT_CONFIG = {
    "provider": "",            # combo key of the active provider, e.g. "custom:cp1"; "" until one is added
    "api_keys": {},            # per-provider: {"custom:abc": "..."}
    "custom_providers": {},    # id -> {"name": str, "base_url": str}
    "port": 4001,
    "model": "",                # DEPRECATED — kept for backward compat; migrated into "models"
    "models": {},               # per-provider last model: {"custom:abc": "org/model", ...}
    "minimize_to_tray": True,
    "show_tray_icon": True,
    "allow_client_override": False,
    "favourite_models": {},    # per-provider: {"custom:abc": ["model/a", ...]}
}

# Sentinel combo key for the "Add Custom Provider..." entry. It is never
# selectable as a real provider; selecting it launches the add dialog.
ADD_CUSTOM_KEY = "__add_custom__"


def is_custom_provider(provider_key: str) -> bool:
    """True if the combo key refers to a user-defined custom provider."""
    return isinstance(provider_key, str) and provider_key.startswith(CUSTOM_PROVIDER_PREFIX)


def custom_provider_id(provider_key: str) -> str:
    """Extract the id portion from a ``custom:<id>`` key (empty if not custom)."""
    if is_custom_provider(provider_key):
        return provider_key[len(CUSTOM_PROVIDER_PREFIX):]
    return ""


def new_custom_id(existing: Dict[str, dict]) -> str:
    """Generate a unique, short id for a new custom provider.

    Ids are stable opaque tokens (``cp1``, ``cp2``, …) so renaming the display
    name later never orphans the stored API key or favourites.
    """
    used = set(existing.keys())
    n = 1
    while True:
        candidate = f"cp{n}"
        if candidate not in used:
            return candidate
        n += 1


def provider_display_name(provider_key: str, config: dict) -> str:
    """Human-friendly name shown in the provider combo / log messages.

    All providers are custom, so this just returns the stored display name
    (or falls back to the combo key itself when none is set).
    """
    if is_custom_provider(provider_key):
        cp = config.get("custom_providers", {}).get(custom_provider_id(provider_key))
        if cp:
            return cp.get("name") or provider_key
    return provider_key


def resolve_base_url(provider_key: str, config: dict) -> str:
    """Resolve the base URL to hit for a provider key.

    All providers are user-defined; the URL lives in ``custom_providers``.
    Returns "" when unknown (e.g. a custom that was deleted).
    """
    if is_custom_provider(provider_key):
        cp = config.get("custom_providers", {}).get(custom_provider_id(provider_key))
        if cp:
            return (cp.get("base_url") or "").strip()
    return ""


def sanitize_custom_name(name: str, existing_names) -> str:
    """Trim and de-duplicate a custom provider display name.

    Collisions with other custom names get a numeric suffix ("My API (2)").
    ``existing_names`` should already be lower-cased for comparison.
    """
    name = (name or "").strip()
    if not name:
        name = "Custom Provider"
    reserved = {n.lower() for n in existing_names if n}
    if name.lower() in reserved:
        base = name
        i = 2
        while f"{base} ({i})".lower() in reserved:
            i += 1
        name = f"{base} ({i})"
    return name


def host_from_url(url: str) -> str:
    """Extract the hostname portion of a URL for use as a default display name.

    e.g. "https://api.kilo.ai/api/gateway" -> "api.kilo.ai",
         "https://router.huggingface.co/v1" -> "router.huggingface.co".
    Returns an empty string when no plausible host can be parsed (the value
    must be ``localhost`` or contain a dot, to filter out free-form junk).
    """
    url = (url or "").strip()
    if not url:
        return ""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = (parsed.hostname or "").strip()
    if not host:
        return ""
    # Reject values that aren't a real host (e.g. "not a url at all" parsed as
    # host="not"). Accept localhost and any dotted hostname.
    if host == "localhost" or "." in host:
        return host
    return ""


def load_config() -> dict:
    """Load configuration from JSON file."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                cfg = json.load(f)
            # Merge with defaults so new keys get populated
            merged = {**DEFAULT_CONFIG, **cfg}

            # Migrate old single api_key to per-provider dict
            if not isinstance(merged.get("api_keys"), dict):
                old_key = merged.get("api_key", "")
                merged["api_keys"] = {}
                if old_key:
                    merged["api_keys"][merged.get("provider", "")] = old_key
            # Remove the old flat key if present
            merged.pop("api_key", None)

            # --- Migrate the old flat "model" key into per-provider "models" ---
            if not isinstance(merged.get("models"), dict):
                merged["models"] = {}
            old_model = merged.pop("model", None)
            if old_model and isinstance(old_model, str):
                pk = merged.get("provider", "")
                if pk not in merged["models"]:
                    merged["models"][pk] = old_model
            # Drop the flat key from defaults too (it stays in DEFAULT_CONFIG for
            # backward compat only).
            merged.pop("model", None)

            # --- Migrate the legacy single "custom" provider ---
            # Old schema stored one custom URL/key under the literal "custom"
            # key in custom_urls / api_keys. Promote it to a named entry in the
            # new custom_providers map so it keeps working. The display name is
            # derived from the URL's host (e.g. "https://api.kilo.ai/..." ->
            # "api.kilo.ai") so it reads well in the provider dropdown.
            if not isinstance(merged.get("custom_providers"), dict):
                merged["custom_providers"] = {}
            legacy_urls = merged.pop("custom_urls", None)
            if isinstance(legacy_urls, dict) and legacy_urls:
                old_url = legacy_urls.get("custom", "")
                if old_url:
                    cp_id = new_custom_id(merged["custom_providers"])
                    name = host_from_url(old_url) or "Custom Provider"
                    merged["custom_providers"][cp_id] = {
                        "name": name,
                        "base_url": old_url,
                    }
                    # Re-key the saved API key if the active provider was the
                    # legacy "custom" entry.
                    api_keys = merged.get("api_keys", {})
                    if "custom" in api_keys and f"custom:{cp_id}" not in api_keys:
                        api_keys[f"custom:{cp_id}"] = api_keys.pop("custom")
                        merged["api_keys"] = api_keys
                    if merged.get("provider") == "custom":
                        merged["provider"] = f"custom:{cp_id}"
                    # Favourites too, if present.
                    favs = merged.get("favourite_models", {})
                    if isinstance(favs, dict) and "custom" in favs:
                        favs[f"custom:{cp_id}"] = favs.pop("custom")
                        merged["favourite_models"] = favs
            merged.pop("custom_url", None)  # stray legacy flat key

            # Ensure nested dicts exist
            if not isinstance(merged.get("custom_providers"), dict):
                merged["custom_providers"] = {}
            if not isinstance(merged.get("favourite_models"), dict):
                merged["favourite_models"] = {}

            # Backfill host-derived names for any custom entry that still has a
            # placeholder display name (covers legacy entries migrated before
            # this naming logic existed, and any manually edited config).
            for cp_id, info in merged["custom_providers"].items():
                if not isinstance(info, dict):
                    continue
                current = (info.get("name") or "").strip()
                if current in ("", "Custom Provider", "custom"):
                    host = host_from_url(info.get("base_url", ""))
                    if host:
                        info["name"] = host

            return merged
        except (json.JSONDecodeError, OSError):
            return dict(DEFAULT_CONFIG)
    return dict(DEFAULT_CONFIG)


def save_config(config: dict) -> None:
    """Save configuration to JSON file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
