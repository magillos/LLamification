"""Provider abstractions for different LLM gateways.

LLamification ships no built-in providers. Every provider is a user-defined
OpenAI-compatible endpoint, addressed by the ``"custom:<id>"`` config key and
configured with a name, base URL, and API key. ``get_provider`` builds a single
generic :class:`OpenAICompatibleProvider` for any of them.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import aiohttp


class LLMProvider(ABC):
    """Base class for an LLM provider."""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    @abstractmethod
    def api_base(self) -> str:
        """The base URL for chat completions."""
        ...

    @abstractmethod
    def model_list_url(self) -> str:
        """URL to fetch available models."""
        ...

    @abstractmethod
    def prepare_chat_payload(
        self, model: str, messages: List[Dict[str, str]], stream: bool, options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the provider-specific request body for chat completions."""
        ...

    @abstractmethod
    def model_param(self, model: str) -> str:
        """Transform the model identifier as needed by this provider."""
        ...

    async def fetch_models(self) -> List[Dict[str, str]]:
        """Fetch available models from the provider."""
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(self.model_list_url(), timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Failed to fetch models: {resp.status} {await resp.text()}")
                data = await resp.json()
                return self._parse_models(data)

    @abstractmethod
    def _parse_models(self, data: dict) -> List[Dict[str, str]]:
        """Extract model names from the provider's response."""
        ...


class OpenAICompatibleProvider(LLMProvider):
    """Generic provider for any OpenAI-compatible ``/v1`` endpoint.

    Speaks the standard OpenAI schema (``/models``, ``/chat/completions``,
    responses shaped as ``{"data": [{"id": "...", "name": "...", ...}]}``),
    which covers virtually every hosted and self-hosted gateway the user may
    add.
    """

    def api_base(self) -> str:
        return f"{self.base_url}/chat/completions"

    def model_list_url(self) -> str:
        return f"{self.base_url}/models"

    def model_param(self, model: str) -> str:
        return model  # model IDs are used verbatim

    def prepare_chat_payload(
        self, model: str, messages: List[Dict[str, str]], stream: bool, options: Dict[str, Any]
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "max_tokens": options.get("num_predict", 2048),
            "temperature": options.get("temperature", 0.7),
            "top_p": options.get("top_p", 0.95),
        }
        if "stop" in options:
            payload["stop"] = options["stop"]
        # Forward function-calling fields so agentic clients (Cline, Continue,
        # etc.) can use tools through the proxy. These are only present when
        # the client explicitly requests them, so plain chat is unaffected.
        if "tools" in options:
            payload["tools"] = options["tools"]
        if "tool_choice" in options:
            payload["tool_choice"] = options["tool_choice"]
        # Forward structured-output / JSON-mode requests.
        if "response_format" in options:
            payload["response_format"] = options["response_format"]
        # Forward stream options (e.g. include_usage for token counting).
        if "stream_options" in options:
            payload["stream_options"] = options["stream_options"]
        # Forward logprobs requests for evaluation / debugging tools.
        if "logprobs" in options:
            payload["logprobs"] = options["logprobs"]
        if "top_logprobs" in options:
            payload["top_logprobs"] = options["top_logprobs"]
        # Forward additional sampling parameters that agentic clients may set.
        for key in ("presence_penalty", "frequency_penalty", "seed", "n"):
            if key in options:
                payload[key] = options[key]
        return payload

    def _parse_models(self, data: dict) -> List[Dict[str, str]]:
        models = []
        # OpenAI-compatible gateways return {"data": [{"id": "...", "name": "...", ...}]}
        for m in data.get("data", []):
            models.append({"id": m["id"], "name": m.get("name", m["id"])})
        return models


def get_provider(provider_name: str, api_key: str, base_url: str = "") -> LLMProvider:
    """Factory: build the provider instance for a config key.

    All providers are user-defined OpenAI-compatible endpoints addressed by a
    ``"custom:<id>"`` key (the legacy bare ``"custom"`` key is still accepted).
    They *require* an explicit ``base_url`` — there are no hard-coded defaults.
    """
    if provider_name == "custom" or (isinstance(provider_name, str) and provider_name.startswith("custom:")):
        if not base_url:
            raise ValueError("Custom provider requires a base URL")
        return OpenAICompatibleProvider(api_key=api_key, base_url=base_url)

    raise ValueError(f"Unknown provider: {provider_name}")
