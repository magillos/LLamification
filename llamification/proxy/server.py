"""Ollama-compatible HTTP proxy server using aiohttp."""

import asyncio
import json
import logging
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


@web.middleware
async def cors_middleware(request: web.Request, handler) -> web.StreamResponse:
    """Add CORS headers to all responses and handle OPTIONS preflight."""
    if request.method == "OPTIONS":
        # Respond to CORS preflight with the requested origin echoed back
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
      POST /api/chat          -> chat completion

    OpenAI-compatible routes:
      GET  /v1/models         -> list available models
      POST /v1/chat/completions -> chat completion
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.host = host
        self.port = port
        self._provider: Optional[LLMProvider] = None
        self._models: List[str] = []
        self._active_model: str = ""
        self._allow_client_override: bool = True
        self._model_aliases: Dict[str, str] = {}  # alias -> full_model_id
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

        self.app = web.Application(middlewares=[cors_middleware])

        # Ollama endpoints
        self.app.router.add_get("/", self.handle_root)
        self.app.router.add_get("/api/tags", self.handle_tags)
        self.app.router.add_post("/api/generate", self.handle_generate)
        self.app.router.add_post("/api/chat", self.handle_chat)

        # OpenAI-compatible endpoints
        self.app.router.add_get("/v1", self.handle_v1_root)  # some apps check this
        self.app.router.add_get("/v1/models", self.handle_v1_models)
        self.app.router.add_post("/v1/chat/completions", self.handle_v1_chat)

    def configure(self, provider: LLMProvider, models: List[str], active_model: str, allow_client_override: bool = True):
        """Set the current provider, available models, active model, and override setting."""
        self._provider = provider
        self._models = models
        self._active_model = active_model
        self._allow_client_override = allow_client_override
        
        # Auto-generate aliases: extract short name from full model ID
        # e.g., "z-ai/glm-5.1" -> alias "glm-5.1"
        self._model_aliases = {}
        for model in models:
            if "/" in model:
                short_name = model.split("/", 1)[1]  # Get part after "/"
                self._model_aliases[short_name] = model
        
        # Also add the active model's short name if it has one
        if active_model and "/" in active_model:
            short_name = active_model.split("/", 1)[1]
            self._model_aliases[short_name] = active_model

    def get_active_model(self) -> str:
        """Return the currently active model name."""
        return self._active_model
    
    def _resolve_model_alias(self, model: str) -> str:
        """Resolve a model alias to its full ID, or return the model as-is."""
        # If the model is in our aliases, return the full ID
        if model in self._model_aliases:
            resolved = self._model_aliases[model]
            logger.info(f"Model alias resolved: '{model}' -> '{resolved}'")
            return resolved
        # Otherwise return as-is (could be full ID already, or "default")
        return model

    async def handle_root(self, request: web.Request) -> web.Response:
        return web.json_response(
            {"status": "Ollama is running"},
            status=200,
        )

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
        return web.json_response(
            make_ollama_tags_response(models_to_return, provider_name),
            status=200,
        )

    async def handle_v1_root(self, request: web.Request) -> web.Response:
        """OpenAI-compatible GET /v1 — some apps check this as a health endpoint."""
        return web.json_response(
            {"status": "running", "provider": "llamification"},
            status=200,
        )

    async def handle_v1_models(self, request: web.Request) -> web.Response:
        """OpenAI-compatible GET /v1/models."""
        if self._allow_client_override:
            models_list = self._models or [self._active_model] if self._active_model else []
        else:
            models_list = [self._active_model] if self._active_model else []
        data = [
            {
                "id": m,
                "object": "model",
                "created": 1700000000,
                "owned_by": "llamification",
            }
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

        if not self._provider:
            return web.json_response({"error": "No provider configured"}, status=503)

        # Resolve model: check aliases, fall back to active model if "default"
        if model == "default":
            actual_model = self._active_model or "default"
        else:
            actual_model = self._resolve_model_alias(model)
        
        provider_model = self._provider.model_param(actual_model)

        if stream:
            # Build CORS headers for streaming response
            origin = request.headers.get("Origin", "*")
            cors = {
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Credentials": "true",
            }
            response = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    **cors,
                },
            )
            await response.prepare(request)

            try:
                async for token, done in self._stream_chat(provider_model, messages, options):
                    if done:
                        chunk_data = {
                            "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
                        }
                    else:
                        chunk_data = {
                            "choices": [{"delta": {"content": token}, "index": 0}],
                        }
                    line = f"data: {json.dumps(chunk_data)}\n\n"
                    await response.write(line.encode("utf-8"))
                    if done:
                        break
                await response.write(b"data: [DONE]\n\n")
            except (ConnectionError, asyncio.CancelledError) as e:
                logger.info(f"V1 stream connection closed by client: {e}")
            except Exception as e:
                logger.error(f"V1 stream error: {e}")
                try:
                    await response.write(json.dumps({"error": str(e)}).encode("utf-8"))
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
                # Return in OpenAI format
                content = ""
                try:
                    content = provider_resp["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    pass
                openai_resp = {
                    "id": "chatcmpl-llamification",
                    "object": "chat.completion",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": actual_model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
                return web.json_response(openai_resp, status=200)
            except Exception as e:
                logger.error(f"V1 chat error: {e}")
                return web.json_response({"error": str(e)}, status=502)

    async def handle_generate(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        model, prompt, messages, stream, options = parse_ollama_generate_request(body)
        
        if not self._allow_client_override:
            model = self._active_model or "default"

        if not self._provider:
            return web.json_response(
                {"error": "No provider configured"}, status=503
            )

        # Resolve model: check aliases, fall back to active model if "default"
        if model == "default":
            actual_model = self._active_model or "default"
        else:
            actual_model = self._resolve_model_alias(model)
        
        provider_model = self._provider.model_param(actual_model)

        if stream:
            response = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "application/x-ndjson",
                    "Cache-Control": "no-cache",
                },
            )
            await response.prepare(request)

            try:
                async for token, done in self._stream_chat(provider_model, messages, options):
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
                return web.json_response({"error": str(e)}, status=502)

    async def handle_chat(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        model, messages, stream, options = parse_ollama_chat_request(body)
        
        if not self._allow_client_override:
            model = self._active_model or "default"

        if not self._provider:
            return web.json_response(
                {"error": "No provider configured"}, status=503
            )

        # Resolve model: check aliases, fall back to active model if "default"
        if model == "default":
            actual_model = self._active_model or "default"
        else:
            actual_model = self._resolve_model_alias(model)
        
        provider_model = self._provider.model_param(actual_model)

        if stream:
            response = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "application/x-ndjson",
                    "Cache-Control": "no-cache",
                },
            )
            await response.prepare(request)

            try:
                async for token, done in self._stream_chat(provider_model, messages, options):
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
                return web.json_response({"error": str(e)}, status=502)

    async def _non_stream_chat(
        self, model: str, messages: List[Dict[str, str]], options: Dict[str, Any]
    ) -> dict:
        """Make a non-streaming chat completion request to the provider."""
        if not self._provider:
            raise RuntimeError("No provider configured")

        payload = self._provider.prepare_chat_payload(model, messages, False, options)
        headers = {
            "Authorization": f"Bearer {self._provider.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "LLamification/0.1",
        }

        url = self._provider.api_base()
        logger.info(f"POST {url}")
        logger.info(f"Model: {payload.get('model')}, Stream: {payload.get('stream')}")

        connector = aiohttp.TCPConnector(force_close=True)
        async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    err_text = await resp.text()
                    logger.error(f"Provider {resp.status} from {url}: {err_text[:500]}")
                    raise RuntimeError(f"Provider returned {resp.status}: {err_text}")
                return await resp.json()

    async def _stream_chat(
        self, model: str, messages: List[Dict[str, str]], options: Dict[str, Any]
    ) -> "asyncio.AsyncIterator[tuple[str, bool]]":
        """Make a streaming chat completion, yielding (token, done) tuples."""
        if not self._provider:
            raise RuntimeError("No provider configured")

        payload = self._provider.prepare_chat_payload(model, messages, True, options)
        headers = {
            "Authorization": f"Bearer {self._provider.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "LLamification/0.1",
        }

        url = self._provider.api_base()
        logger.info(f"Stream POST {url}")
        logger.info(f"Model: {payload.get('model')}")

        connector = aiohttp.TCPConnector(force_close=True)
        async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    err_text = await resp.text()
                    logger.error(f"Provider stream {resp.status} from {url}: {err_text[:500]}")
                    raise RuntimeError(f"Provider returned {resp.status}: {err_text}")

                # Parse Server-Sent Events
                buffer = ""
                async for chunk in resp.content:
                    buffer += chunk.decode("utf-8", errors="replace")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:]  # strip "data: " prefix
                            if data_str.strip() == "[DONE]":
                                yield ("", True)
                                return
                            try:
                                data = json.loads(data_str)
                                # Extract content delta
                                choices = data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    finish_reason = choices[0].get("finish_reason")
                                    if content:
                                        yield (content, False)
                                    if finish_reason == "stop":
                                        yield ("", True)
                                        return
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse SSE: {data_str[:200]}")

                # If we get here without a finish, send done
                yield ("", True)

    async def start(self) -> None:
        """Start the server."""
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        logger.info(f"Proxy server started on {self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the server."""
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            self._site = None
            logger.info("Proxy server stopped")