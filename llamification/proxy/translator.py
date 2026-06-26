"""
Ollama ↔ Provider request/response translator.

Ollama API schemas (what clients expect):
  /api/tags    -> {"models": [{"name": "...", "modified_at": "...", ...}]}
  /api/generate -> input: {"model":"...", "prompt":"...", "stream":true, "options":{...}}
                   output (non-stream): {"response":"...", "done":true, ...}
                   output (stream):      {"response":"token", "done":false}
  /api/chat    -> input: {"model":"...", "messages":[...], "stream":true, "options":{...}}
                   output (non-stream): {"message":{"role":"assistant","content":"..."}, "done":true}
                   output (stream):      {"message":{"content":"token"},"done":false}

Provider API (OpenAI-compatible /v1/chat/completions):
  input:  {"model":"...", "messages":[...], "stream":true, "max_tokens":..., "temperature":...}
  output (non-stream): {"choices":[{"message":{"role":"assistant","content":"..."}, ...}]}
  output (stream):      {"choices":[{"delta":{"role":"assistant"},"index":0}]}
                        {"choices":[{"delta":{"content":"token"},"index":0}]}
                        {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}
"""

import json
from typing import Any, AsyncGenerator, Dict, List


def parse_ollama_generate_request(body: dict) -> tuple:
    """Parse an Ollama /api/generate request into (model, prompt, messages, stream, options)."""
    model = body.get("model", "default")
    prompt = body.get("prompt", "")
    stream = body.get("stream", False)
    options = body.get("options", {})
    # Convert prompt to a single user message for the provider
    messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
    if "system" in body:
        messages.insert(0, {"role": "system", "content": body["system"]})
    return model, prompt, messages, stream, options


def parse_ollama_chat_request(body: dict) -> tuple:
    """Parse an Ollama /api/chat request into (model, messages, stream, options).

    Also extracts Ollama-style ``tools`` and ``tool_choice`` from the request
    body so they can be forwarded to the upstream OpenAI-compatible provider.
    """
    model = body.get("model", "default")
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    options = body.get("options", {})
    # Ollama puts tools at the top level of the request body, not inside options.
    if "tools" in body:
        options["tools"] = body["tools"]
    if "tool_choice" in body:
        options["tool_choice"] = body["tool_choice"]
    return model, messages, stream, options


def make_ollama_tags_response(models: List[str], provider: str) -> dict:
    """Build the /api/tags response from a list of model names."""
    return {
        "models": [
            {
                "name": m,
                "model": m,
                "modified_at": "2024-01-01T00:00:00Z",
                "size": 0,
                "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": provider,
                    "families": [provider],
                    "parameter_size": "unknown",
                    "quantization_level": "unknown",
                },
            }
            for m in models
        ]
    }


def make_ollama_generate_response(provider_response: dict) -> dict:
    """Convert a non-streaming provider chat response to Ollama /api/generate format."""
    content = ""
    try:
        content = provider_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        pass
    return {
        "model": provider_response.get("model", "unknown"),
        "created_at": "2024-01-01T00:00:00Z",
        "response": content,
        "done": True,
        "done_reason": "stop",
        "context": [],
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": 0,
        "eval_count": len(content),
        "eval_duration": 0,
    }


def make_ollama_chat_response(provider_response: dict) -> dict:
    """Convert a non-streaming provider chat response to Ollama /api/chat format.

    Also translates OpenAI-style ``tool_calls`` into Ollama's
    ``message.tool_calls`` format so Ollama-native clients can use tools.
    """
    content = ""
    tool_calls = None
    done_reason = "stop"
    try:
        msg = provider_response["choices"][0]["message"]
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            done_reason = "stop"
    except (KeyError, IndexError):
        pass
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "model": provider_response.get("model", "unknown"),
        "created_at": "2024-01-01T00:00:00Z",
        "message": message,
        "done": True,
        "done_reason": done_reason,
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": 0,
        "eval_count": len(content),
        "eval_duration": 0,
    }


def make_ollama_stream_chunk(token: str, done: bool = False) -> str:
    """Build a single streaming JSON line for Ollama /api/generate."""
    chunk = {
        "response": token,
        "done": done,
    }
    if done:
        chunk["done_reason"] = "stop"
        chunk["context"] = []
        chunk["total_duration"] = 0
        chunk["eval_count"] = 0
        chunk["eval_duration"] = 0
    return json.dumps(chunk) + "\n"


def make_ollama_chat_stream_chunk(content: str, done: bool = False) -> str:
    """Build a single streaming JSON line for Ollama /api/chat."""
    chunk = {
        "message": {"role": "assistant", "content": content},
        "done": done,
    }
    if done:
        chunk["done_reason"] = "stop"
    return json.dumps(chunk) + "\n"