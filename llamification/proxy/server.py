"""Ollama-compatible HTTP proxy server using aiohttp."""

import asyncio
import copy
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import web

from .providers import LLMProvider
from .translator import (
    make_ollama_chat_response,
    make_ollama_chat_stream_chunk,
    make_ollama_generate_response,
    make_ollama_stream_chunk,
    make_ollama_tags_response,
    parse_ollama_chat_request,
    parse_ollama_generate_request,
)

logger = logging.getLogger("llamification.proxy")

_MISSING = object()


def _get_path(source: Any, path: List[str]) -> Any:
    """Return the value at a dotted path, or ``_MISSING`` when it is absent."""
    current = source
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _set_path(target: Dict[str, Any], path: List[str], value: Any) -> None:
    """Set ``value`` at a dotted path, creating intermediate dicts as needed."""
    current = target
    for key in path[:-1]:
        if not isinstance(current.get(key), dict):
            current[key] = {}
        current = current[key]
    current[path[-1]] = value


def _merge_prefer_client(default: Any, client: Any) -> Any:
    """Merge an ``extra_body`` default with a client value, client leaves winning.

    Nested dicts are merged so a client that overrides one sub-key still
    inherits the other sub-keys supplied by the proxy configuration.
    """
    if isinstance(default, dict):
        merged = copy.deepcopy(default)
        if isinstance(client, dict):
            for key, value in client.items():
                merged[key] = _merge_prefer_client(default.get(key), value)
        return merged
    return copy.deepcopy(client)


def _build_extra_body_patch(
    extra_body: Any, force_paths: Any, client_body: Any
) -> Optional[Dict[str, Any]]:
    """Build the upstream payload overlay for a model's ``extra_body`` setting.

    Fields the client supplied win over the configured defaults unless their
    key (or dotted path) is listed in ``force_extra_body``.
    """
    if not isinstance(extra_body, dict) or not extra_body:
        return None
    forced = (
        {p for p in force_paths if isinstance(p, str)}
        if isinstance(force_paths, (list, tuple, set))
        else set()
    )
    client = client_body if isinstance(client_body, dict) else {}
    patch: Dict[str, Any] = {}
    for key, value in extra_body.items():
        path = key.split(".")
        client_value = _get_path(client, path)
        if key in forced or client_value is _MISSING:
            _set_path(patch, path, copy.deepcopy(value))
        else:
            _set_path(patch, path, _merge_prefer_client(value, client_value))
    return patch or None


def _apply_payload_patch(payload: Dict[str, Any], patch: Dict[str, Any]) -> None:
    """Merge ``patch`` into ``payload``, patch leaves winning."""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            _apply_payload_patch(payload[key], value)
        else:
            payload[key] = copy.deepcopy(value)


@web.middleware
async def cors_middleware(request: web.Request, handler) -> web.StreamResponse:
    """Add CORS headers to all responses and handle OPTIONS preflight."""
    if request.method == "OPTIONS":
        origin = request.headers.get("Origin", "*")
        req_headers = request.headers.get("Access-Control-Request-Headers", "content-type, authorization")
        return web.Response(
            status=200,
            headers={
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": req_headers,
                "Access-Control-Max-Age": "86400",
            },
        )
    try:
        response = await handler(request)
        origin = request.headers.get("Origin", "*")
        response.headers["Access-Control-Allow-Origin"] = origin
        return response
    except web.HTTPException as exc:
        origin = request.headers.get("Origin", "*")
        exc.headers["Access-Control-Allow-Origin"] = origin  # type: ignore
        raise


class ProxyServer:
    """
    An async HTTP server that mimics both Ollama and OpenAI APIs.

    Ollama routes:
      GET  /                  -> "Ollama is running"
      GET  /api/tags          -> list available models
      POST /api/generate      -> generate text
      POST /api/chat          -> chat completion (with tools support)
      GET  /api/version       -> version info
      POST /api/embeddings    -> embeddings (proxied to upstream)

    OpenAI-compatible routes:
      GET  /v1/models         -> list available models
      POST /v1/chat/completions -> chat completion (with tools pass-through)
      POST /v1/embeddings     -> embeddings (proxied to upstream)
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11434,
        verbose: bool = False,
        debug_raw: bool = False,
    ):
        self.host = host
        self.port = port
        self._provider: Optional[LLMProvider] = None
        self._models: List[str] = []
        self._active_model: str = ""
        self._allow_client_override: bool = True
        self._model_aliases: Dict[str, str] = {}
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._verbose: bool = verbose
        self._debug_raw: bool = debug_raw
        self._settings: Dict[str, Any] = {}

        self.app = web.Application(middlewares=[cors_middleware])

        self.app.router.add_get("/", self.handle_root)
        self.app.router.add_get("/api/tags", self.handle_tags)
        self.app.router.add_get("/api/version", self.handle_version)
        self.app.router.add_post("/api/generate", self.handle_generate)
        self.app.router.add_post("/api/chat", self.handle_chat)
        self.app.router.add_post("/api/embeddings", self.handle_v1_embeddings)

        self.app.router.add_get("/v1", self.handle_v1_root)
        self.app.router.add_get("/v1/models", self.handle_v1_models)
        self.app.router.add_post("/v1/chat/completions", self.handle_v1_chat)
        self.app.router.add_post("/v1/embeddings", self.handle_v1_embeddings)

    def configure(
        self,
        provider: LLMProvider,
        models: List[str],
        active_model: str,
        allow_client_override: bool = True,
        settings: Optional[Dict[str, Any]] = None,
    ):
        """Point the server at a provider/model.

        ``settings`` carries the optional reasoning / extra_body configuration
        (``provider_key``, ``strip_reasoning``, ``fallback_to_reasoning`` and
        ``model_settings``). Omitted settings mean everything stays off.
        """
        self._provider = provider
        self._models = models
        self._active_model = active_model
        self._allow_client_override = allow_client_override
        self._settings = settings or {}
        self._model_aliases = {}
        for model in models:
            if "/" in model:
                short_name = model.split("/", 1)[1]
                self._model_aliases[short_name] = model
        if active_model and "/" in active_model:
            short_name = active_model.split("/", 1)[1]
            self._model_aliases[short_name] = active_model

    def get_active_model(self) -> str:
        return self._active_model

    def _resolve_model_alias(self, model: str) -> str:
        if model in self._model_aliases:
            resolved = self._model_aliases[model]
            logger.info(f"Model alias resolved: '{model}' -> '{resolved}'")
            return resolved
        return model

    def _resolve_to_active_model(self, model: str) -> str:
        """Upgrade a client-sent model to the active model when it's a prefix.

        Some providers accept suffixed model names (e.g. ``glm-5.2-short-flex``)
        that enable features like queued/discounted inference.  Clients like
        aurscan send the base name (``glm-5.2-short``) without the suffix, which
        causes the provider to use the non-suffixed variant.

        If the active model starts with ``model + "-"`` (e.g. ``glm-5.2-short``
        is a prefix of ``glm-5.2-short-flex``), replace the requested model
        with the active model so the suffix is always preserved.
        """
        if (
            self._active_model
            and model != self._active_model
            and self._active_model.startswith(model + "-")
        ):
            logger.info(
                f"Model prefix-matched: '{model}' -> active model '{self._active_model}'"
            )
            return self._active_model
        return model

    def _resolve_model_settings(self, actual_model: str, provider_model: str) -> Dict[str, Any]:
        """Resolve reasoning / extra_body settings for one request.

        Precedence: per-model entry > provider ``"*"`` entry > global keys.
        The active provider key comes from ``settings["provider_key"]``.
        """
        resolved: Dict[str, Any] = {
            "strip_reasoning": bool(self._settings.get("strip_reasoning", False)),
            "fallback_to_reasoning": bool(self._settings.get("fallback_to_reasoning", False)),
            "extra_body": {},
            "force_extra_body": [],
        }
        overrides = self._settings.get("model_settings")
        if not isinstance(overrides, dict):
            return resolved
        provider_key = self._settings.get("provider_key", "")
        entries = overrides.get(provider_key) if provider_key else None
        if not isinstance(entries, dict):
            return resolved
        # Apply the provider default first so a model entry always wins.
        for name in ("*", provider_model, actual_model):
            entry = entries.get(name)
            if not isinstance(entry, dict):
                continue
            for key in ("strip_reasoning", "fallback_to_reasoning"):
                if key in entry:
                    resolved[key] = bool(entry[key])
            if isinstance(entry.get("extra_body"), dict):
                resolved["extra_body"] = _merge_prefer_client(
                    resolved["extra_body"], entry["extra_body"]
                )
            if isinstance(entry.get("force_extra_body"), (list, tuple)):
                resolved["force_extra_body"] = list(entry["force_extra_body"])
        return resolved

    def _error_response(self, error: Exception, status: int = 502) -> web.Response:
        """Format an error as an OpenAI-style error response."""
        msg = str(error)
        code = None
        if "Provider returned" in msg:
            try:
                code = int(msg.split("Provider returned ")[1].split(":")[0])
            except (ValueError, IndexError):
                pass
        err_body = {
            "error": {
                "message": msg,
                "type": "upstream_error" if code else "proxy_error",
                "code": code,
            }
        }
        return web.json_response(err_body, status=status)

    async def _write_sse_line(self, response: web.StreamResponse, line: str) -> None:
        """Write one SSE line, echoing it when raw debugging is enabled."""
        if self._debug_raw:
            logger.info(f"[RAW CLIENT RESPONSE] {line.strip()}")
        await response.write(line.encode("utf-8"))

    async def handle_root(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "Ollama is running"}, status=200)

    async def handle_version(self, request: web.Request) -> web.Response:
        return web.json_response({"version": "0.1.0"}, status=200)

    async def handle_tags(self, request: web.Request) -> web.Response:
        if not self._models:
            return web.json_response({"models": []}, status=200)
        if self._allow_client_override:
            models_to_return = self._models
        else:
            models_to_return = [self._active_model] if self._active_model else [self._models[0]]
        provider_name = "llamification"
        if self._provider:
            provider_name = type(self._provider).__name__.replace("Provider", "").lower()
        return web.json_response(make_ollama_tags_response(models_to_return, provider_name), status=200)

    async def handle_v1_root(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "running", "provider": "llamification"}, status=200)

    async def handle_v1_models(self, request: web.Request) -> web.Response:
        if self._allow_client_override:
            models_list = self._models or [self._active_model] if self._active_model else []
        else:
            models_list = [self._active_model] if self._active_model else []
        data = [
            {"id": m, "object": "model", "created": 1700000000, "owned_by": "llamification"}
            for m in models_list
        ]
        return web.json_response({"object": "list", "data": data}, status=200)

    async def handle_v1_chat(self, request: web.Request) -> web.StreamResponse:
        """OpenAI-compatible POST /v1/chat/completions."""
        body = await request.json()
        if self._allow_client_override:
            model = body.get("model", self._active_model or "default")
        else:
            model = self._active_model or "default"
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        options = {
            "num_predict": body.get("max_tokens", 2048),
            "temperature": body.get("temperature", 0.7),
            "top_p": body.get("top_p", 0.95),
        }
        if "stop" in body:
            options["stop"] = body["stop"]
        if "tools" in body:
            options["tools"] = body["tools"]
        if "tool_choice" in body:
            options["tool_choice"] = body["tool_choice"]
        if "response_format" in body:
            options["response_format"] = body["response_format"]
        if "stream_options" in body:
            options["stream_options"] = body["stream_options"]
        if "logprobs" in body:
            options["logprobs"] = body["logprobs"]
        if "top_logprobs" in body:
            options["top_logprobs"] = body["top_logprobs"]
        for key in ("presence_penalty", "frequency_penalty", "seed", "n"):
            if key in body:
                options[key] = body[key]

        if not self._provider:
            return web.json_response({"error": "No provider configured"}, status=503)

        if model == "default":
            actual_model = self._active_model or "default"
        else:
            actual_model = self._resolve_model_alias(model)
            actual_model = self._resolve_to_active_model(actual_model)

        provider_model = self._provider.model_param(actual_model)

        # Per-model reasoning / extra_body settings. All of these are off by
        # default, so existing configurations behave exactly as before.
        model_settings = self._resolve_model_settings(actual_model, provider_model)
        strip_reasoning = model_settings["strip_reasoning"]
        fallback_reasoning = model_settings["fallback_to_reasoning"]
        extra_patch = _build_extra_body_patch(
            model_settings["extra_body"], model_settings["force_extra_body"], body
        )

        if stream:
            origin = request.headers.get("Origin", "*")
            response = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Credentials": "true",
                },
            )
            await response.prepare(request)

            # Emit initial role chunk as OpenAI does.
            role_chunk = {"choices": [{"delta": {"role": "assistant"}, "index": 0}]}
            await self._write_sse_line(response, f"data: {json.dumps(role_chunk)}\n\n")

            stream_opts = body.get("stream_options", {})
            include_usage = (
                stream_opts.get("include_usage", False) if isinstance(stream_opts, dict) else False
            )

            try:
                # Reasoning deltas are held back only when the fallback is
                # enabled: we must know whether real content ever arrives
                # before deciding which field carries the reasoning text.
                reasoning_buffer: List[str] = []
                saw_content = False

                async for token, reasoning_delta, done, tool_delta, usage in self._stream_chat(
                    provider_model, messages, options, extra_patch
                ):
                    if reasoning_delta:
                        if fallback_reasoning and not saw_content:
                            reasoning_buffer.append(reasoning_delta)
                        elif not strip_reasoning:
                            chunk_data = {
                                "choices": [
                                    {"delta": {"reasoning_content": reasoning_delta}, "index": 0}
                                ]
                            }
                            await self._write_sse_line(
                                response, f"data: {json.dumps(chunk_data)}\n\n"
                            )
                    if done:
                        if reasoning_buffer:
                            if fallback_reasoning and not saw_content:
                                logger.warning(
                                    "Upstream returned only reasoning for '%s'; falling back "
                                    "to reasoning_content as content "
                                    "(fallback_to_reasoning enabled)",
                                    actual_model,
                                )
                                for piece in reasoning_buffer:
                                    chunk_data = {
                                        "choices": [{"delta": {"content": piece}, "index": 0}]
                                    }
                                    await self._write_sse_line(
                                        response, f"data: {json.dumps(chunk_data)}\n\n"
                                    )
                            elif not strip_reasoning:
                                for piece in reasoning_buffer:
                                    chunk_data = {
                                        "choices": [
                                            {"delta": {"reasoning_content": piece}, "index": 0}
                                        ]
                                    }
                                    await self._write_sse_line(
                                        response, f"data: {json.dumps(chunk_data)}\n\n"
                                    )
                            reasoning_buffer = []
                        chunk_data = {
                            "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]
                        }
                    elif tool_delta is not None:
                        chunk_data = {
                            "choices": [{"delta": {"tool_calls": [tool_delta]}, "index": 0}]
                        }
                    elif token:
                        if reasoning_buffer:
                            if not strip_reasoning:
                                for piece in reasoning_buffer:
                                    held = {
                                        "choices": [
                                            {"delta": {"reasoning_content": piece}, "index": 0}
                                        ]
                                    }
                                    await self._write_sse_line(
                                        response, f"data: {json.dumps(held)}\n\n"
                                    )
                            reasoning_buffer = []
                        saw_content = True
                        chunk_data = {
                            "choices": [{"delta": {"content": token}, "index": 0}]
                        }
                    else:
                        continue
                    await self._write_sse_line(response, f"data: {json.dumps(chunk_data)}\n\n")
                    if done:
                        if include_usage and usage:
                            usage_chunk = {"choices": [], "usage": usage}
                            await self._write_sse_line(
                                response, f"data: {json.dumps(usage_chunk)}\n\n"
                            )
                        break
                await self._write_sse_line(response, "data: [DONE]\n\n")
            except (ConnectionError, asyncio.CancelledError, aiohttp.ClientConnectionError) as e:
                logger.info(f"V1 stream connection closed: {e}")
            except Exception as e:
                logger.error(f"V1 stream error: {e}", exc_info=self._verbose)
                try:
                    await response.write(
                        f"data: {json.dumps({'error': {'message': str(e), 'type': 'proxy_error'}})}\n\n".encode("utf-8")
                    )
                except Exception:
                    pass

            try:
                await response.write_eof()
            except Exception:
                pass
            return response
        else:
            try:
                provider_resp = await self._non_stream_chat(
                    provider_model, messages, options, extra_patch
                )
                content = ""
                reasoning = ""
                tool_calls = None
                finish_reason = "stop"
                try:
                    msg = provider_resp["choices"][0]["message"]
                    content = msg.get("content", "") or ""
                    reasoning = msg.get("reasoning_content", "") or ""
                    tool_calls = msg.get("tool_calls")
                    if tool_calls:
                        finish_reason = "tool_calls"
                except (KeyError, IndexError):
                    pass
                assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content}
                if reasoning and not strip_reasoning:
                    if not content and fallback_reasoning:
                        logger.warning(
                            "Upstream returned only reasoning for '%s'; falling back to "
                            "reasoning_content as content (fallback_to_reasoning enabled)",
                            actual_model,
                        )
                        assistant_msg["content"] = reasoning
                    else:
                        # Pass the field through untouched for clients that read it.
                        assistant_msg["reasoning_content"] = reasoning
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                openai_resp = {
                    "id": "chatcmpl-llamification",
                    "object": "chat.completion",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": actual_model,
                    "choices": [
                        {
                            "index": 0,
                            "message": assistant_msg,
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": provider_resp.get(
                        "usage",
                        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    ),
                }
                if self._debug_raw:
                    logger.info(f"[RAW CLIENT RESPONSE] {json.dumps(openai_resp)}")
                return web.json_response(openai_resp, status=200)
            except asyncio.TimeoutError as e:
                logger.error("V1 chat timed out waiting for upstream response", exc_info=self._verbose)
                return self._error_response(RuntimeError("Upstream response timed out after 600 seconds of inactivity"), 502)
            except Exception as e:
                logger.error(f"V1 chat error: {e!r}", exc_info=self._verbose)
                return self._error_response(e, 502)
    async def handle_v1_embeddings(self, request: web.Request) -> web.Response:
        if not self._provider:
            return web.json_response({"error": "No provider configured"}, status=503)
        body = await request.json()
        url = f"{self._provider.base_url}/embeddings"
        headers = {"Content-Type": "application/json", "User-Agent": "LLamification/0.1"}
        if self._provider.api_key:
            headers["Authorization"] = f"Bearer {self._provider.api_key}"
        logger.info(f"Embeddings POST {url}")
        connector = aiohttp.TCPConnector(force_close=True)
        try:
            async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
                async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if resp.status != 200:
                        err_text = await resp.text()
                        return web.json_response({"error": f"Provider returned {resp.status}: {err_text}"}, status=502)
                    data = await resp.json()
                    return web.json_response(data, status=200)
        except Exception as e:
            logger.error(f"Embeddings error: {e}")
            return self._error_response(e, 502)

    async def handle_generate(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        model, prompt, messages, stream, options = parse_ollama_generate_request(body)
        if not self._allow_client_override:
            model = self._active_model or "default"
        if not self._provider:
            return web.json_response({"error": "No provider configured"}, status=503)
        if model == "default":
            actual_model = self._active_model or "default"
        else:
            actual_model = self._resolve_model_alias(model)
            actual_model = self._resolve_to_active_model(actual_model)
        provider_model = self._provider.model_param(actual_model)
        if stream:
            response = web.StreamResponse(status=200, headers={"Content-Type": "application/x-ndjson", "Cache-Control": "no-cache"})
            await response.prepare(request)
            try:
                async for token, _reasoning, done, _, _ in self._stream_chat(provider_model, messages, options):
                    chunk = make_ollama_stream_chunk(token, done)
                    await response.write(chunk.encode("utf-8"))
                    if done:
                        break
            except Exception as e:
                logger.error(f"Stream error: {e}")
                err_chunk = make_ollama_stream_chunk(f"Error: {e}", done=True)
                try:
                    await response.write(err_chunk.encode("utf-8"))
                except Exception:
                    pass
            try:
                await response.write_eof()
            except Exception:
                pass
            return response
        else:
            try:
                provider_resp = await self._non_stream_chat(provider_model, messages, options)
                ollama_resp = make_ollama_generate_response(provider_resp)
                return web.json_response(ollama_resp, status=200)
            except Exception as e:
                logger.error(f"Generate error: {e}")
                return self._error_response(e, 502)

    async def handle_chat(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        model, messages, stream, options = parse_ollama_chat_request(body)
        if not self._allow_client_override:
            model = self._active_model or "default"
        if not self._provider:
            return web.json_response({"error": "No provider configured"}, status=503)
        if model == "default":
            actual_model = self._active_model or "default"
        else:
            actual_model = self._resolve_model_alias(model)
            actual_model = self._resolve_to_active_model(actual_model)
        provider_model = self._provider.model_param(actual_model)
        if stream:
            response = web.StreamResponse(status=200, headers={"Content-Type": "application/x-ndjson", "Cache-Control": "no-cache"})
            await response.prepare(request)
            try:
                async for token, _reasoning, done, _, _ in self._stream_chat(provider_model, messages, options):
                    chunk = make_ollama_chat_stream_chunk(token, done)
                    await response.write(chunk.encode("utf-8"))
                    if done:
                        break
            except Exception as e:
                logger.error(f"Chat stream error: {e}")
                err_chunk = make_ollama_chat_stream_chunk(f"Error: {e}", done=True)
                try:
                    await response.write(err_chunk.encode("utf-8"))
                except Exception:
                    pass
            try:
                await response.write_eof()
            except Exception:
                pass
            return response
        else:
            try:
                provider_resp = await self._non_stream_chat(provider_model, messages, options)
                ollama_resp = make_ollama_chat_response(provider_resp)
                return web.json_response(ollama_resp, status=200)
            except Exception as e:
                logger.error(f"Chat error: {e}")
                return self._error_response(e, 502)

    async def _non_stream_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Dict[str, Any],
        payload_patch: Optional[Dict[str, Any]] = None,
    ) -> dict:
        if not self._provider:
            raise RuntimeError("No provider configured")
        payload = self._provider.prepare_chat_payload(model, messages, False, options)
        if payload_patch:
            _apply_payload_patch(payload, payload_patch)
        headers = {"Content-Type": "application/json", "User-Agent": "LLamification/0.1"}
        if self._provider.api_key:
            headers["Authorization"] = f"Bearer {self._provider.api_key}"
        url = self._provider.api_base()
        logger.info(f"POST {url}")
        if self._verbose:
            logger.debug(f"Request payload: {json.dumps(payload)[:2000]}")
        # A completion may take longer than two minutes. Use a generous
        # inactivity timeout rather than a short total request deadline; active
        # upstream streaming is handled separately by _stream_chat.
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=600)
        connector = aiohttp.TCPConnector(force_close=True)
        async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    err_text = await resp.text()
                    if self._verbose:
                        logger.debug(f"Upstream error {resp.status}: {err_text[:1000]}")
                    raise RuntimeError(f"Provider returned {resp.status}: {err_text}")
                raw = await resp.text()
                if self._debug_raw:
                    logger.info(f"[RAW UPSTREAM] {raw}")
                try:
                    result = json.loads(raw)
                except json.JSONDecodeError:
                    raise RuntimeError(f"Provider returned invalid JSON: {raw[:500]}")
                if self._verbose:
                    logger.debug(f"Response: {json.dumps(result)[:2000]}")
                return result

    async def _stream_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Dict[str, Any],
        payload_patch: Optional[Dict[str, Any]] = None,
    ):
        """Yield ``(content, reasoning, done, tool_delta, usage)`` from upstream."""
        if not self._provider:
            raise RuntimeError("No provider configured")
        payload = self._provider.prepare_chat_payload(model, messages, True, options)
        if payload_patch:
            _apply_payload_patch(payload, payload_patch)
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream", "User-Agent": "LLamification/0.1"}
        if self._provider.api_key:
            headers["Authorization"] = f"Bearer {self._provider.api_key}"
        url = self._provider.api_base()
        logger.info(f"Stream POST {url}")
        if self._verbose:
            logger.debug(f"Stream request payload: {json.dumps(payload)[:2000]}")
        # Do not use a total timeout for a streamed completion. Long generations
        # can legitimately exceed it even while tokens are arriving regularly.
        # Keep a generous socket-read timeout instead, so a genuinely stalled
        # upstream still fails rather than hanging forever.
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=300)
        connector = aiohttp.TCPConnector(force_close=True)
        async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    err_text = await resp.text()
                    raise RuntimeError(f"Provider returned {resp.status}: {err_text}")
                buffer = ""
                async for chunk in resp.content:
                    buffer += chunk.decode("utf-8", errors="replace")
                    # SSE events are separated by a blank line. Parsing complete
                    # events (rather than individual TCP chunks) also handles
                    # JSON split across multiple network reads.
                    while "\n\n" in buffer or "\r\n\r\n" in buffer:
                        separator = "\r\n\r\n" if "\r\n\r\n" in buffer else "\n\n"
                        event, buffer = buffer.split(separator, 1)
                        data_lines = [
                            line[5:].lstrip() for line in event.replace("\r\n", "\n").split("\n")
                            if line.startswith("data:")
                        ]
                        if not data_lines:
                            continue
                        data_str = "\n".join(data_lines)
                        if data_str.strip() == "[DONE]":
                            yield ("", "", True, None, None)
                            return
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse SSE event: {data_str[:200]}")
                            continue
                        if self._debug_raw:
                            logger.info(f"[RAW UPSTREAM] {data_str}")
                        usage = data.get("usage")
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {}) or {}
                            content = delta.get("content", "") or ""
                            reasoning = delta.get("reasoning_content", "") or ""
                            tool_calls = delta.get("tool_calls")
                            finish_reason = choices[0].get("finish_reason")
                            if reasoning:
                                yield ("", reasoning, False, None, None)
                            if content:
                                yield (content, "", False, None, None)
                            if tool_calls:
                                for tc in tool_calls:
                                    yield ("", "", False, tc, None)
                            if finish_reason in ("stop", "tool_calls"):
                                yield ("", "", True, None, usage)
                                return
                        elif usage:
                            yield ("", "", True, None, usage)
                            return
                # Process a final event if the upstream closes without a trailing
                # blank line (some llama.cpp builds do this).
                event = buffer.replace("\r\n", "\n")
                data_lines = [line[5:].lstrip() for line in event.split("\n") if line.startswith("data:")]
                if data_lines and "[DONE]" in "\n".join(data_lines):
                    yield ("", "", True, None, None)
                    return
                yield ("", "", True, None, None)

    async def start(self) -> None:
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        logger.info(f"Proxy server started on {self.host}:{self.port}")

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            self._site = None
            logger.info("Proxy server stopped")