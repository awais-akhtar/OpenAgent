"""Model provider gateway.

The gateway defaults to deterministic local behavior so OpenAgent can run
offline. Network providers are called only when a user explicitly selects them
or an auto-routing policy allows local network providers.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
from typing import Any, Iterator, Protocol

from .core_runtime import CPU_MODEL_STACK, model_request_options, smooth_runtime_profile
from .egress import EgressClient
from .adaptive_engine.failure_reporting import failure_detail, failure_summary
from .adaptive_engine.execution_service import ExecutionCancelled
from .inference_accelerator import (
    AdaptiveCPUInferenceEngine,
    active_runtime_options,
    estimate_messages_tokens,
    infer_task,
)


def _closing_provider_stream(chunks: Iterator["StreamChunk"]) -> Iterator["StreamChunk"]:
    """Forward provider chunks and deterministically release the upstream iterator.

    A caller closing the gateway stream (for example after an HTTP disconnect)
    must not leave a provider response/socket owned only by a suspended iterator.
    Cleanup failures are surfaced on normal completion, but never replace the
    provider/consumer exception that caused unwinding.
    """
    source = iter(chunks)
    primary: BaseException | None = None
    try:
        for chunk in source:
            yield chunk
    except BaseException as exc:
        primary = exc
        raise
    finally:
        close = getattr(source, "close", None)
        if callable(close):
            try:
                close()
            except BaseException as cleanup:
                if primary is None:
                    raise
                try:
                    primary.add_note("provider stream cleanup failed: " + failure_summary(cleanup))
                except BaseException:
                    pass


def _validate_stream_chunk(chunk: object, *, provider_id: str,
                           terminal_seen: bool) -> bool:
    """Validate provider event shape and enforce terminal-is-final ordering."""
    if terminal_seen:
        raise ProviderUnavailableError(
            f"Provider '{provider_id}' emitted an event after its terminal stream event"
        )
    if (not isinstance(chunk, StreamChunk) or type(chunk.provider) is not str
            or not chunk.provider or type(chunk.model) is not str or not chunk.model
            or type(chunk.content) is not str or type(chunk.done) is not bool
            or type(chunk.metadata) is not dict):
        raise ProviderUnavailableError(
            f"Provider '{provider_id}' emitted an invalid stream event"
        )
    return chunk.done


@dataclass(frozen=True)
class ModelResult:
    provider: str
    model: str
    content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class StreamChunk:
    provider: str
    model: str
    content: str
    done: bool
    metadata: dict[str, Any]


@dataclass(frozen=True)
class EmbeddingResult:
    provider: str
    model: str
    embeddings: list[list[float]]
    dimensions: int
    metadata: dict[str, Any]


class ModelProvider(Protocol):
    id: str
    label: str
    local: bool
    network: bool
    default_model: str
    privacy_warning: str

    def available(self) -> bool:
        ...

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
    ) -> ModelResult:
        ...

    def complete_vision(
        self,
        messages: list[dict[str, str]],
        image: bytes,
        *,
        media_type: str,
        model: str | None = None,
        max_output_tokens: int | None = None,
    ) -> ModelResult:
        ...

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
    ) -> Iterator[StreamChunk]:
        ...

    def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> EmbeddingResult:
        ...


class ProviderConfigurationError(RuntimeError):
    """Raised when a selected provider is missing required configuration."""


class ProviderUnavailableError(RuntimeError):
    """Raised when a selected provider cannot complete a request."""


_external_approval: ContextVar[bool] = ContextVar("model_external_approval", default=False)


@contextmanager
def _approve_model_egress(approved: bool) -> Iterator[None]:
    token = _external_approval.set(approved)
    try:
        yield
    finally:
        _external_approval.reset(token)


class BaseProvider:
    id = "base"
    label = "Base"
    local = True
    network = False
    default_model = "base"
    privacy_warning = "Provider behavior is not documented."

    def available(self) -> bool:
        return True

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
    ) -> Iterator[StreamChunk]:
        result = self.complete(messages, model=model)
        for part in chunk_text(result.content):
            yield StreamChunk(
                provider=result.provider,
                model=result.model,
                content=part,
                done=False,
                metadata=result.metadata,
            )
        yield StreamChunk(
            provider=result.provider,
            model=result.model,
            content="",
            done=True,
            metadata=result.metadata,
        )

    def complete_vision(
        self,
        messages: list[dict[str, str]],
        image: bytes,
        *,
        media_type: str,
        model: str | None = None,
        max_output_tokens: int | None = None,
    ) -> ModelResult:
        raise ProviderUnavailableError(f"Provider '{self.id}' does not support vision completion.")

    def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> EmbeddingResult:
        raise ProviderUnavailableError(f"Provider '{self.id}' does not support embeddings.")


class LocalProvider(BaseProvider):
    id = "local"
    label = "OpenAgent Local"
    local = True
    network = False
    default_model = "openagent-local"
    default_embedding_model = "openagent-local-hash-embedding"
    privacy_warning = "Offline deterministic provider. No network calls are made."

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
    ) -> ModelResult:
        user_message = next(
            (message["content"] for message in reversed(messages) if message["role"] == "user"),
            "",
        )
        system_context = "\n".join(
            message["content"] for message in messages if message["role"] == "system"
        )
        content = local_assistant_response(user_message, system_context)
        return ModelResult(
            provider=self.id,
            model=model or self.default_model,
            content=content,
            metadata={
                "offline": True,
                "network": False,
                "routed": False,
                "deterministic_fallback": True,
                "local_model_server": False,
                "smooth_runtime": smooth_runtime_profile(task="chat").to_dict(),
            },
        )

    def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> EmbeddingResult:
        dimensions = 64
        embeddings = [local_hash_embedding(text, dimensions=dimensions) for text in texts]
        return EmbeddingResult(
            provider=self.id,
            model=model or self.default_embedding_model,
            embeddings=embeddings,
            dimensions=dimensions,
            metadata={"offline": True, "network": False, "algorithm": "sha256-hashing"},
        )


def _ollama_vision_messages(messages: list[dict[str, str]], image: bytes) -> list[dict[str, Any]]:
    """Attach one image to the latest user turn using Ollama's image field."""
    encoded = base64.b64encode(image).decode("ascii")
    result = [dict(message) for message in messages]
    image_index = next(
        (index for index in range(len(result) - 1, -1, -1) if result[index].get("role") == "user"),
        None,
    )
    if image_index is None:
        result.append({"role": "user", "content": "Describe the supplied image.", "images": [encoded]})
    else:
        result[image_index]["images"] = [encoded]
    return result


def _openai_vision_messages(
    messages: list[dict[str, str]], image: bytes, media_type: str,
) -> list[dict[str, Any]]:
    """Attach one image to the latest user turn using OpenAI image_url parts."""
    data_url = f"data:{media_type};base64,{base64.b64encode(image).decode('ascii')}"
    result: list[dict[str, Any]] = []
    image_attached = False
    for message in reversed(messages):
        if not image_attached and message.get("role") == "user":
            content: list[dict[str, Any]] = []
            if message.get("content"):
                content.append({"type": "text", "text": str(message["content"])})
            content.append({"type": "image_url", "image_url": {"url": data_url}})
            result.append({"role": "user", "content": content})
            image_attached = True
        else:
            result.append(dict(message))
    result.reverse()
    if not image_attached:
        result.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the supplied image."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        })
    return result


class OllamaProvider(BaseProvider):
    id = "ollama"
    label = "Ollama"
    local = True
    network = True
    privacy_warning = "Calls the configured Ollama server, normally on localhost."

    def __init__(self) -> None:
        self.base_url = os.environ.get("OPENAGENT_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        self.default_model = normalize_model_name(os.environ.get("OPENAGENT_OLLAMA_MODEL", CPU_MODEL_STACK["chat"]))
        self.default_embedding_model = normalize_model_name(os.environ.get("OPENAGENT_OLLAMA_EMBED_MODEL", self.default_model))
        self._models_cache: list[str] = []
        self._models_cached_at = 0.0
        self._models_cached_approval = False
        self._cache_lock = threading.RLock()

    def available(self) -> bool:
        return bool(self.installed_models())

    def installed_models(self) -> list[str]:
        now = time.monotonic()
        with self._cache_lock:
            if now - self._models_cached_at <= 3.0 and self._models_cached_approval == _external_approval.get():
                return list(self._models_cache)
        try:
            data = get_json(f"{self.base_url}/api/tags", timeout_seconds=0.5)
            models = [str(item.get("name", "")).strip() for item in data.get("models", []) if str(item.get("name", "")).strip()]
        except Exception:
            models = []
        with self._cache_lock:
            self._models_cache = list(models)
            self._models_cached_at = now
            self._models_cached_approval = _external_approval.get()
        return models

    def resolve_model(self, requested: str | None) -> str:
        installed = self.installed_models()
        if not installed:
            return normalize_model_name(requested or self.default_model)
        wanted = normalize_model_name(requested or self.default_model)
        for name in installed:
            if model_name_matches(wanted, name):
                return name
        for preferred in preferred_ollama_models():
            for name in installed:
                if model_name_matches(preferred, name):
                    return name
        return installed[0]

    def resolve_vision_model(self, requested: str | None) -> str:
        installed = self.installed_models()
        if requested:
            wanted = normalize_model_name(requested)
            for name in installed:
                if model_name_matches(wanted, name):
                    return name
        vision_markers = ("llava", "qwen2.5vl", "qwen2-vl", "qwen3-vl", "minicpm-v", "moondream", "vision", "gemma3")
        for name in installed:
            compact = name.lower().replace("_", "-")
            if any(marker in compact for marker in vision_markers):
                return name
        return self.resolve_model(requested)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
    ) -> ModelResult:
        selected_model = self.resolve_model(model)
        options = provider_request_options(messages)
        keep_alive = os.environ.get("OPENAGENT_OLLAMA_KEEP_ALIVE", "30m").strip() or "30m"
        response = post_json(
            f"{self.base_url}/api/chat",
            {"model": selected_model, "messages": messages, "stream": False, "options": options, "keep_alive": keep_alive},
            headers={},
        )
        content = response.get("message", {}).get("content", "")
        if not content:
            content = json.dumps(response, sort_keys=True)
        return ModelResult(
            provider=self.id,
            model=selected_model,
            content=content,
            metadata={"base_url": self.base_url, "network": True, "local_network": True, "smooth_options": options},
        )

    def complete_vision(
        self,
        messages: list[dict[str, str]],
        image: bytes,
        *,
        media_type: str,
        model: str | None = None,
        max_output_tokens: int | None = None,
    ) -> ModelResult:
        selected_model = self.resolve_vision_model(model)
        options = provider_request_options(messages)
        if max_output_tokens is not None:
            options = {**options, "num_predict": max(1, int(max_output_tokens))}
        vision_messages = _ollama_vision_messages(messages, image)
        keep_alive = os.environ.get("OPENAGENT_OLLAMA_KEEP_ALIVE", "30m").strip() or "30m"
        response = post_json(
            f"{self.base_url}/api/chat",
            {
                "model": selected_model,
                "messages": vision_messages,
                "stream": False,
                "options": options,
                "keep_alive": keep_alive,
            },
            headers={},
            timeout_seconds=vision_model_timeout_seconds(),
        )
        content = str(response.get("message", {}).get("content", ""))
        if not content:
            raise ProviderUnavailableError(
                f"Ollama model '{selected_model}' returned no vision response. Confirm that the model supports images."
            )
        return ModelResult(
            provider=self.id,
            model=selected_model,
            content=content,
            metadata={
                "base_url": self.base_url,
                "network": True,
                "local_network": True,
                "modality": "vision",
                "media_type": media_type,
                "image_transport": "ollama-message-images",
                "smooth_options": options,
            },
        )

    def stream_complete_vision(
        self,
        messages: list[dict[str, str]],
        image: bytes,
        *,
        media_type: str,
        model: str | None = None,
        max_output_tokens: int | None = None,
        cancelled=None,
    ) -> Iterator[StreamChunk]:
        selected_model = self.resolve_vision_model(model)
        options = provider_request_options(messages)
        if max_output_tokens is not None:
            options = {**options, "num_predict": max(1, int(max_output_tokens))}
        keep_alive = os.environ.get("OPENAGENT_OLLAMA_KEEP_ALIVE", "30m").strip() or "30m"
        response = stream_json_lines(
            f"{self.base_url}/api/chat",
            {
                "model": selected_model,
                "messages": _ollama_vision_messages(messages, image),
                "stream": True,
                "options": options,
                "keep_alive": keep_alive,
            },
            headers={},
            cancelled=cancelled,
        )
        primary: BaseException | None = None
        done_seen = False
        try:
            for item in response:
                content = item.get("message", {}).get("content", "")
                done = bool(item.get("done"))
                done_seen = done_seen or done
                yield StreamChunk(
                    provider=self.id,
                    model=selected_model,
                    content=content,
                    done=done,
                    metadata={
                        "base_url": self.base_url,
                        "network": True,
                        "local_network": True,
                        "modality": "vision",
                        "media_type": media_type,
                        "image_transport": "ollama-message-images",
                        "smooth_options": options,
                    },
                )
                if done:
                    break
            if not done_seen:
                raise ProviderUnavailableError("Ollama vision stream ended without a done record")
        except BaseException as exc:
            primary = exc
            raise
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as cleanup:
                    if primary is None:
                        raise
                    try:
                        primary.add_note("Ollama vision stream cleanup failed: " + failure_summary(cleanup))
                    except BaseException:
                        pass

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
    ) -> Iterator[StreamChunk]:
        selected_model = self.resolve_model(model)
        options = provider_request_options(messages)
        keep_alive = os.environ.get("OPENAGENT_OLLAMA_KEEP_ALIVE", "30m").strip() or "30m"
        response = stream_json_lines(
            f"{self.base_url}/api/chat",
            {"model": selected_model, "messages": messages, "stream": True, "options": options, "keep_alive": keep_alive},
            headers={},
        )
        primary: BaseException | None = None
        done_seen = False
        try:
            for item in response:
                content = item.get("message", {}).get("content", "")
                done = bool(item.get("done"))
                if done:
                    done_seen = True
                yield StreamChunk(
                    provider=self.id,
                    model=selected_model,
                    content=content,
                    done=done,
                    metadata={"base_url": self.base_url, "network": True, "local_network": True, "smooth_options": options},
                )
                if done:
                    # Ollama's done record is the protocol terminal boundary;
                    # do not dispatch trailing records as more assistant text.
                    break
            if not done_seen:
                raise ProviderUnavailableError("Ollama stream ended without a done record")
        except BaseException as exc:
            primary = exc
            raise
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as cleanup:
                    if primary is None:
                        raise
                    try:
                        primary.add_note("Ollama stream cleanup failed: " + failure_summary(cleanup))
                    except BaseException:
                        pass

    def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> EmbeddingResult:
        selected_model = self.resolve_model(model or self.default_embedding_model)
        embeddings: list[list[float]] = []
        for text in texts:
            response = post_json(
                f"{self.base_url}/api/embeddings",
                {"model": selected_model, "prompt": text},
                headers={},
            )
            embeddings.append([float(value) for value in response.get("embedding", [])])
        dimensions = len(embeddings[0]) if embeddings else 0
        return EmbeddingResult(
            provider=self.id,
            model=selected_model,
            embeddings=embeddings,
            dimensions=dimensions,
            metadata={"base_url": self.base_url, "network": True, "local_network": True},
        )


class LlamaCppProvider(BaseProvider):
    id = "llamacpp"
    label = "llama.cpp"
    local = True
    network = True
    privacy_warning = "Calls the configured llama.cpp server, normally on localhost."

    def __init__(self) -> None:
        self.base_url = os.environ.get("OPENAGENT_LLAMACPP_URL", "http://127.0.0.1:8080").rstrip("/")
        self.default_model = os.environ.get("OPENAGENT_LLAMACPP_MODEL", CPU_MODEL_STACK["chat"])
        self._availability_cache = False
        self._availability_cached_at = 0.0
        self._availability_cached_approval = False
        self._cache_lock = threading.RLock()

    def available(self) -> bool:
        now = time.monotonic()
        with self._cache_lock:
            if now - self._availability_cached_at <= 3.0 and self._availability_cached_approval == _external_approval.get():
                return self._availability_cache
        available = local_http_available(f"{self.base_url}/v1/models")
        with self._cache_lock:
            self._availability_cache = available
            self._availability_cached_at = now
            self._availability_cached_approval = _external_approval.get()
        return available

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
    ) -> ModelResult:
        selected_model = model or self.default_model
        options = provider_request_options(messages)
        response = post_json(
            f"{self.base_url}/v1/chat/completions",
            {
                "model": selected_model,
                "messages": messages,
                "stream": False,
                "max_tokens": options["num_predict"],
                "temperature": options["temperature"],
                "top_p": options["top_p"],
                "cache_prompt": True,
            },
            headers={},
        )
        content = chat_completion_content(response)
        return ModelResult(
            provider=self.id,
            model=selected_model,
            content=content,
            metadata={"base_url": self.base_url, "network": True, "local_network": True, "smooth_options": options},
        )

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
    ) -> Iterator[StreamChunk]:
        selected_model = model or self.default_model
        options = provider_request_options(messages)
        response = stream_sse_json(
            f"{self.base_url}/v1/chat/completions",
            {
                "model": selected_model,
                "messages": messages,
                "stream": True,
                "max_tokens": options["num_predict"],
                "temperature": options["temperature"],
                "top_p": options["top_p"],
                "cache_prompt": True,
            },
            headers={},
        )
        primary: BaseException | None = None
        done_seen = False
        try:
            for item in response:
                content = delta_content(item)
                done = item.get("done") is True
                if done:
                    done_seen = True
                yield StreamChunk(
                    provider=self.id,
                    model=selected_model,
                    content=content,
                    done=done,
                    metadata={"base_url": self.base_url, "network": True, "local_network": True, "smooth_options": options},
                )
                if done:
                    break
            if not done_seen:
                raise ProviderUnavailableError("llama.cpp stream ended without a done marker")
        except BaseException as exc:
            primary = exc
            raise
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as cleanup:
                    if primary is None:
                        raise
                    try:
                        primary.add_note("llama.cpp stream cleanup failed: " + failure_summary(cleanup))
                    except BaseException:
                        pass


class OpenVINODirectProvider(BaseProvider):
    """Serve converted OpenVINO GenAI models through the resident native worker."""

    id = "openvino-direct"
    label = "OpenVINO Direct"
    local = True
    network = False
    privacy_warning = "Runs a converted OpenVINO model directly on local CPU, GPU, or NPU through inherited process pipes."

    def __init__(self, client: Any) -> None:
        self.client = client
        self.model_path = os.environ.get("OPENAGENT_OPENVINO_MODEL_DIR", "").strip()
        self.device = os.environ.get("OPENAGENT_OPENVINO_DEVICE", "GPU").strip() or "GPU"
        self.default_model = os.environ.get("OPENAGENT_OPENVINO_MODEL_ID", "openagent-direct-chat").strip() or "openagent-direct-chat"
        self._load_lock = threading.RLock()
        self._resident_model_ids: set[str] = set()

    def available(self) -> bool:
        if self.client is None or not self.client.available():
            return False
        status = self.client.status()
        if status.get("running"):
            try:
                live = self.client.status(start=True)
                if any(item.get("kind") == "llm" for item in live.get("models", [])):
                    return True
            except Exception:
                return False
        return bool(self.model_path and Path(self.model_path).expanduser().is_dir())

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
    ) -> ModelResult:
        model_id = model or self.default_model
        self._ensure_loaded(model_id)
        runtime_messages = self._runtime_messages(messages)
        options = provider_request_options(messages)
        generation = {
            "max_new_tokens": int(options.get("num_predict", 256)),
            "temperature": float(options.get("temperature", 0.7)),
            "top_p": float(options.get("top_p", 0.9)),
            "top_k": int(options.get("top_k", 50)),
            "do_sample": float(options.get("temperature", 0.0)) > 0.0,
        }
        result = self.client.generate(model_id, messages=runtime_messages, generation=generation)
        content = str(result.get("text", ""))
        if not content:
            raise ProviderUnavailableError("The direct OpenVINO model returned an empty response.")
        factual = self._runtime_fact_response(messages)
        if factual:
            content = factual
        return ModelResult(
            provider=self.id,
            model=model_id,
            content=content,
            metadata={
                "network": False,
                "direct_hardware": True,
                "device": result.get("device", self.device),
                "latency_ms": result.get("latency_ms"),
                "resident": True,
            },
        )

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
    ) -> Iterator[StreamChunk]:
        model_id = model or self.default_model
        self._ensure_loaded(model_id)
        runtime_messages = self._runtime_messages(messages)
        options = provider_request_options(messages)
        generation = {
            "max_new_tokens": int(options.get("num_predict", 256)),
            "temperature": float(options.get("temperature", 0.7)),
            "top_p": float(options.get("top_p", 0.9)),
            "top_k": int(options.get("top_k", 50)),
            "do_sample": float(options.get("temperature", 0.0)) > 0.0,
        }
        emitted = False
        final: dict[str, Any] = {}
        events = self.client.stream_generate(model_id, messages=runtime_messages, generation=generation)
        primary: BaseException | None = None
        try:
            for event in events:
                if event.get("type") == "token":
                    content = str(event.get("content", ""))
                    if content:
                        emitted = True
                        yield StreamChunk(
                            self.id, model_id, content, False,
                            {"network": False, "direct_hardware": True, "device": self.device, "streamed": True},
                        )
                elif event.get("type") == "done":
                    final = event
                    break
        except BaseException as exc:
            primary = exc
            raise
        finally:
            close = getattr(events, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as cleanup:
                    if primary is None:
                        raise
                    try:
                        primary.add_note("OpenVINO stream cleanup failed: " + failure_summary(cleanup))
                    except BaseException:
                        pass
        if not emitted and final.get("text"):
            yield StreamChunk(
                self.id, model_id, str(final["text"]), False,
                {"network": False, "direct_hardware": True, "device": final.get("device", self.device), "streamed": False},
            )
        yield StreamChunk(
            self.id, model_id, "", True,
            {
                "network": False,
                "direct_hardware": True,
                "device": final.get("device", self.device),
                "latency_ms": final.get("latency_ms"),
                "first_token_ms": final.get("first_token_ms"),
                "resident": True,
                "streamed": bool(final.get("streamed", emitted)),
            },
        )

    def configure(self, *, model_path: str = "", device: str = "", model_id: str = "") -> None:
        if model_path:
            self.model_path = model_path
        if device:
            self.device = device
        if model_id:
            self.default_model = model_id

    def _runtime_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        runtime_fact = {
            "role": "system",
            "content": (
                "Runtime facts you must not contradict: You are OpenAgent running fully locally on the user's machine. "
                f"This response is generated by OpenVINO Direct on the local {self.device.upper()} device. "
                "No cloud model or external provider is involved. Never claim cloud infrastructure. "
                "When asked about runtime hardware, state these facts briefly and accurately."
            ),
        }
        return [runtime_fact, *messages]

    def _runtime_fact_response(self, messages: list[dict[str, str]]) -> str:
        user = next((item.get("content", "") for item in reversed(messages) if item.get("role") == "user"), "").lower()
        asks_runtime = any(
            phrase in user
            for phrase in ("what hardware", "which hardware", "what device", "which device", "running locally", "running on cloud", "are you local")
        )
        if not asks_runtime:
            return ""
        return (
            f"I am running fully locally through OpenVINO Direct on this machine's {self.device.upper()} device; "
            "no cloud model or external provider handled this response."
        )

    def _ensure_loaded(self, model_id: str) -> None:
        with self._load_lock:
            try:
                status = self.client.status(start=True)
                if not status.get("running"):
                    self._resident_model_ids.clear()
                if any(item.get("model_id") == model_id and item.get("kind") == "llm" for item in status.get("models", [])):
                    self._resident_model_ids.add(model_id)
                    return
                if status.get("running") and model_id in self._resident_model_ids:
                    return
            except Exception as exc:
                raise ProviderUnavailableError(f"The direct accelerator worker is unavailable: {exc}") from exc
            path = Path(self.model_path).expanduser() if self.model_path else None
            if path is None or not path.is_dir():
                raise ProviderConfigurationError(
                    "Set OPENAGENT_OPENVINO_MODEL_DIR to a locally converted OpenVINO GenAI model directory."
                )
            try:
                self.client.load_llm(
                    str(path.resolve()),
                    device=self.device,
                    model_id=model_id,
                    scheduler={
                        "enabled": True,
                        "enable_prefix_caching": True,
                        "dynamic_split_fuse": True,
                        "max_num_seqs": 4,
                    },
                )
                self._resident_model_ids.add(model_id)
            except Exception as exc:
                raise ProviderUnavailableError(f"OpenVINO could not load the direct model: {exc}") from exc


_OLMOE_KERNEL_BY_FORMAT = {
    "float32": "native-cpu-numba",
    "int8-row": "native-cpu-numba",
    "int2-row": "int2-row-numba",
    "int3-grouped-g64": "int3-grouped-g64-numba",
    "int4-row": "packed-int4-numba",
    "int4-grouped-g64": "grouped-int4-numba",
    "mxfp4-e2m1-g32-ue8m0": "mxfp4-e2m1-g32-ue8m0-numba",
    "q4_0": "q4_0-numba",
    "q4_1": "q4_1-numba",
    "q5_0": "q5_0-numba",
    "q5_1": "q5_1-numba",
    "q8_0": "q8_0-numba",
    "q8_1": "q8_1-numba",
    "fp8-e4m3-b128-f32": "fp8-e4m3-b128-f32-numba",
}


def _olmoe_default_kernel_for_format(persisted_format: str) -> str:
    """Select the format's declared native consumer; never fall back implicitly."""
    try:
        return _OLMOE_KERNEL_BY_FORMAT[persisted_format]
    except KeyError:
        raise ProviderConfigurationError(
            f"OLMoE format {persisted_format!r} has no default native consumer") from None


class OlmoeDirectProvider(BaseProvider):
    """Run the independently implemented OLMoE decoder and streamed experts."""

    id = "olmoe-direct"
    label = "OLMoE Direct"
    local = True
    network = False
    default_model = "allenai/OLMoE-1B-7B-0125-Instruct"
    privacy_warning = "Runs the configured OLMoE checkpoint and expert container entirely on this machine."

    def __init__(self, direct_hardware: Any = None) -> None:
        self.direct_hardware = direct_hardware
        self.model_path = os.environ.get("OPENAGENT_OLMOE_MODEL_DIR", "").strip()
        self.manifest_path = os.environ.get("OPENAGENT_OLMOE_MANIFEST", "").strip()
        self.cache_gb = max(0.25, float(os.environ.get("OPENAGENT_OLMOE_EXPERT_CACHE_GB", "4")))
        self.context_tokens = max(1, int(os.environ.get("OPENAGENT_OLMOE_CONTEXT", "4096")))
        self.max_output_tokens = max(1, int(os.environ.get("OPENAGENT_OLMOE_MAX_OUTPUT", "128")))
        self.io_mode = os.environ.get("OPENAGENT_OLMOE_IO_MODE", "mmap").strip().lower()
        if self.io_mode not in {"buffered", "mmap", "direct", "uring"}:
            self.io_mode = "mmap"
        self.prefetch_workers = max(1, int(os.environ.get("OPENAGENT_OLMOE_PREFETCH_WORKERS", "2")))
        self.prefetch_depth = max(
            self.prefetch_workers,
            int(os.environ.get("OPENAGENT_OLMOE_PREFETCH_DEPTH", "8")),
        )
        self.storage_roots = _olmoe_storage_roots(os.environ.get("OPENAGENT_OLMOE_STORAGE_ROOTS", ""))
        self.preferred_kernel = os.environ.get("OPENAGENT_OLMOE_KERNEL", "").strip() or None
        self.accelerator_cache_gb = max(
            0.25, float(os.environ.get("OPENAGENT_OLMOE_ACCELERATOR_CACHE_GB", "2"))
        )
        self._runtime: Any = None
        self._lock = threading.RLock()
        self._artifact_signature = None
        self._artifact_ready = False

    def available(self) -> bool:
        root = Path(self.model_path) if self.model_path else None
        manifest = Path(self.manifest_path) if self.manifest_path else None
        configured = bool(
            root and root.is_dir() and (root / "config.json").is_file()
            and (root / "tokenizer.json").is_file() and list(root.glob("*.safetensors"))
            and manifest and manifest.is_file()
        )
        if not configured:
            return False
        try:
            files = [root / "config.json", manifest, *sorted(root.glob("*.safetensors"))]
            index = root / "model.safetensors.index.json"
            if index.is_file():
                files.append(index)
            signature = tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in files)
            with self._lock:
                if signature != self._artifact_signature:
                    from .adaptive_engine.expert_packing import _scan_directory
                    from .adaptive_engine.expert_runtime import SparseExpertManifest
                    from .adaptive_engine.olmoe import OlmoeConfig, validate_olmoe_tensors, olmoe_metadata_fingerprint

                    self._artifact_ready = False
                    self._artifact_signature = signature
                    validate_olmoe_tensors(OlmoeConfig.read(root), _scan_directory(root))
                    parsed = SparseExpertManifest.read(manifest)
                    if parsed.architecture != "olmoe" or parsed.fingerprint != olmoe_metadata_fingerprint(root):
                        return False
                    self._artifact_ready = True
                return self._artifact_ready
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def complete(self, messages: list[dict[str, str]], *, model: str | None = None) -> ModelResult:
        if not self.available():
            raise ProviderUnavailableError(
                "Configure OPENAGENT_OLMOE_MODEL_DIR and OPENAGENT_OLMOE_MANIFEST with completed local artifacts."
            )
        prompt = _olmoe_prompt(messages)
        started = time.perf_counter()
        with self._lock:
            runtime = self._load()
            content = runtime.generate(prompt, max_tokens=self.max_output_tokens)
            status = runtime.status()
        if not content.strip():
            raise ProviderUnavailableError("OLMoE completed without producing text.")
        return ModelResult(
            self.id,
            model or self.default_model,
            content,
            {
                "network": False,
                "offline": True,
                "engine": "openagent-olmoe-streaming",
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "runtime": status,
            },
        )

    def _load(self):
        if self._runtime is None:
            from .adaptive_engine import ExpertKernelRegistry, OlmoeModel, OpenVINOExpertBackend

            if self.preferred_kernel is None:
                from .adaptive_engine.expert_runtime import SparseExpertManifest

                manifest = SparseExpertManifest.read(self.manifest_path)
                formats = {layer.dtype for layer in manifest.layers}
                if len(formats) != 1:
                    raise ProviderConfigurationError(
                        "OLMoE direct provider requires one homogeneous persisted expert format")
                self.preferred_kernel = _olmoe_default_kernel_for_format(next(iter(formats)))

            kernels = None
            if self.preferred_kernel == "vulkan-wgpu":
                from .adaptive_engine.vulkan_expert import VulkanRoutedSwiGLUBackend

                backend = VulkanRoutedSwiGLUBackend()
                if not backend.info().available:
                    raise ProviderConfigurationError(backend.info().unavailable_reason)
                kernels = ExpertKernelRegistry([backend])
            if self.preferred_kernel == "torch-expert":
                from .adaptive_engine.torch_expert import TorchExpertBackend

                backend = TorchExpertBackend()
                if not backend.info().available:
                    raise ProviderConfigurationError(backend.info().unavailable_reason)
                kernels = ExpertKernelRegistry([backend])
            if self.preferred_kernel == "native-cpu-numba":
                from .adaptive_engine import NativeExpertBackend

                backend = NativeExpertBackend()
                if not backend.info().available:
                    raise ProviderConfigurationError(backend.info().unavailable_reason)
                kernels = ExpertKernelRegistry([backend])
            if self.preferred_kernel == "vectorized-cpu-numba":
                from .adaptive_engine.vectorized_expert import VectorizedExpertBackend

                backend = VectorizedExpertBackend()
                if not backend.info().available:
                    raise ProviderConfigurationError(backend.info().unavailable_reason)
                kernels = ExpertKernelRegistry([backend])
            if self.preferred_kernel == "packed-int4-numba":
                from .adaptive_engine.native_packed_expert import NativePackedExpertBackend

                backend = NativePackedExpertBackend()
                if not backend.info().available:
                    raise ProviderConfigurationError(backend.info().unavailable_reason)
                kernels = ExpertKernelRegistry([backend])
            if self.preferred_kernel == "q8_0-numba":
                from .adaptive_engine.native_q8_0_expert import NativeQ8_0ExpertBackend

                backend = NativeQ8_0ExpertBackend()
                if not backend.info().available:
                    raise ProviderConfigurationError(backend.info().unavailable_reason)
                kernels = ExpertKernelRegistry([backend])
            if self.preferred_kernel == "q4_0-numba":
                from .adaptive_engine.native_q4_0_expert import NativeQ4_0ExpertBackend

                backend = NativeQ4_0ExpertBackend()
                if not backend.info().available:
                    raise ProviderConfigurationError(backend.info().unavailable_reason)
                kernels = ExpertKernelRegistry([backend])
            if self.preferred_kernel == "q4_1-numba":
                from .adaptive_engine.native_q4_1_expert import NativeQ4_1ExpertBackend

                backend = NativeQ4_1ExpertBackend()
                if not backend.info().available:
                    raise ProviderConfigurationError(backend.info().unavailable_reason)
                kernels = ExpertKernelRegistry([backend])
            if self.preferred_kernel == "q5_0-numba":
                from .adaptive_engine.native_q5_0_expert import NativeQ5_0ExpertBackend

                backend = NativeQ5_0ExpertBackend()
                if not backend.info().available:
                    raise ProviderConfigurationError(backend.info().unavailable_reason)
                kernels = ExpertKernelRegistry([backend])
            if self.preferred_kernel == "q5_1-numba":
                from .adaptive_engine.native_q5_1_expert import NativeQ5_1ExpertBackend

                backend = NativeQ5_1ExpertBackend()
                if not backend.info().available:
                    raise ProviderConfigurationError(backend.info().unavailable_reason)
                kernels = ExpertKernelRegistry([backend])
            if self.preferred_kernel == "q8_1-numba":
                from .adaptive_engine.native_q8_1_expert import NativeQ8_1ExpertBackend

                backend = NativeQ8_1ExpertBackend()
                if not backend.info().available:
                    raise ProviderConfigurationError(backend.info().unavailable_reason)
                kernels = ExpertKernelRegistry([backend])
            if self.preferred_kernel in {"int2-row-numba", "int3-grouped-g64-numba"}:
                from .adaptive_engine.native_lowbit_expert import NativeLowBitExpertBackend

                backend = NativeLowBitExpertBackend(self.preferred_kernel.removesuffix("-numba"))
                if not backend.info().available:
                    raise ProviderConfigurationError(backend.info().unavailable_reason)
                kernels = ExpertKernelRegistry([backend])
            if self.preferred_kernel == "grouped-int4-numba":
                from .adaptive_engine.native_grouped_int4_expert import NativeGroupedInt4ExpertBackend

                backend = NativeGroupedInt4ExpertBackend()
                if not backend.info().available:
                    raise ProviderConfigurationError(backend.info().unavailable_reason)
                kernels = ExpertKernelRegistry([backend])
            if self.preferred_kernel == "mxfp4-e2m1-g32-ue8m0-numba":
                from .adaptive_engine.native_mxfp4_expert import NativeMXFP4ExpertBackend

                backend = NativeMXFP4ExpertBackend()
                if not backend.info().available:
                    raise ProviderConfigurationError(backend.info().unavailable_reason)
                kernels = ExpertKernelRegistry([backend])
            if self.preferred_kernel == "fp8-e4m3-b128-f32-numba":
                from .adaptive_engine.native_fp8_block_expert import NativeFP8BlockExpertBackend

                backend = NativeFP8BlockExpertBackend()
                if not backend.info().available:
                    raise ProviderConfigurationError(backend.info().unavailable_reason)
                kernels = ExpertKernelRegistry([backend])
            if self.preferred_kernel and self.preferred_kernel.startswith("openvino-"):
                if self.direct_hardware is None:
                    raise ProviderConfigurationError(
                        "The requested OpenVINO expert kernel needs the local direct-hardware runtime."
                    )
                device = self.preferred_kernel.removeprefix("openvino-").upper()
                kernels = ExpertKernelRegistry([
                    OpenVINOExpertBackend(
                        self.direct_hardware,
                        device=device,
                        capacity_bytes=int(self.accelerator_cache_gb * 1024**3),
                    )
                ])

            self._runtime = OlmoeModel(
                self.model_path,
                self.manifest_path,
                expert_cache_bytes=int(self.cache_gb * 1024**3),
                context_tokens=self.context_tokens,
                io_mode=self.io_mode,
                prefetch_workers=self.prefetch_workers,
                prefetch_depth=self.prefetch_depth,
                storage_roots=self.storage_roots,
                kernels=kernels,
                preferred_kernel=self.preferred_kernel,
            )
        return self._runtime


class OpenAICompatibleProvider(BaseProvider):
    id = "openai-compatible"
    label = "OpenAI Compatible"
    local = False
    network = True
    privacy_warning = "External provider. Prompts may leave user infrastructure."

    def __init__(self) -> None:
        self.base_url = os.environ.get("OPENAGENT_OPENAI_BASE_URL", "").rstrip("/")
        self.api_key = os.environ.get("OPENAGENT_OPENAI_API_KEY", "")
        self.default_model = os.environ.get("OPENAGENT_OPENAI_MODEL", "gpt-4o-mini")
        self.default_embedding_model = os.environ.get(
            "OPENAGENT_OPENAI_EMBED_MODEL",
            "text-embedding-3-small",
        )

    def available(self) -> bool:
        return bool(self.base_url and self.api_key)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
    ) -> ModelResult:
        self.require_configured()
        selected_model = model or self.default_model
        response = post_json(
            f"{self.base_url}/chat/completions",
            {"model": selected_model, "messages": messages, "stream": False},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        return ModelResult(
            provider=self.id,
            model=selected_model,
            content=chat_completion_content(response),
            metadata={"base_url": self.base_url, "network": True, "external": True},
        )

    def complete_vision(
        self,
        messages: list[dict[str, str]],
        image: bytes,
        *,
        media_type: str,
        model: str | None = None,
        max_output_tokens: int | None = None,
    ) -> ModelResult:
        self.require_configured()
        selected_model = model or self.default_model
        vision_messages = _openai_vision_messages(messages, image, media_type)
        payload: dict[str, Any] = {"model": selected_model, "messages": vision_messages, "stream": False}
        if max_output_tokens is not None:
            payload["max_tokens"] = max(1, int(max_output_tokens))
        response = post_json(
            f"{self.base_url}/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        content = chat_completion_content(response)
        if not content:
            raise ProviderUnavailableError(
                f"OpenAI-compatible model '{selected_model}' returned no vision response. Confirm that the endpoint and model support image_url content."
            )
        return ModelResult(
            provider=self.id,
            model=selected_model,
            content=content,
            metadata={
                "base_url": self.base_url,
                "network": True,
                "external": True,
                "modality": "vision",
                "media_type": media_type,
                "image_transport": "openai-image-url",
            },
        )

    def stream_complete_vision(
        self,
        messages: list[dict[str, str]],
        image: bytes,
        *,
        media_type: str,
        model: str | None = None,
        max_output_tokens: int | None = None,
        cancelled=None,
    ) -> Iterator[StreamChunk]:
        self.require_configured()
        selected_model = model or self.default_model
        payload: dict[str, Any] = {
            "model": selected_model,
            "messages": _openai_vision_messages(messages, image, media_type),
            "stream": True,
        }
        if max_output_tokens is not None:
            payload["max_tokens"] = max(1, int(max_output_tokens))
        response = stream_sse_json(
            f"{self.base_url}/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            cancelled=cancelled,
        )
        primary: BaseException | None = None
        done_seen = False
        try:
            for item in response:
                done = item.get("done") is True
                done_seen = done_seen or done
                yield StreamChunk(
                    provider=self.id,
                    model=selected_model,
                    content=delta_content(item),
                    done=done,
                    metadata={
                        "base_url": self.base_url,
                        "network": True,
                        "external": True,
                        "modality": "vision",
                        "media_type": media_type,
                        "image_transport": "openai-image-url",
                    },
                )
                if done:
                    break
            if not done_seen:
                raise ProviderUnavailableError("OpenAI-compatible vision stream ended without a terminal event")
        except BaseException as exc:
            primary = exc
            raise
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as cleanup:
                    if primary is None:
                        raise
                    try:
                        primary.add_note("OpenAI-compatible vision stream cleanup failed: " + failure_summary(cleanup))
                    except BaseException:
                        pass

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
    ) -> Iterator[StreamChunk]:
        self.require_configured()
        selected_model = model or self.default_model
        response = stream_sse_json(
            f"{self.base_url}/chat/completions",
            {"model": selected_model, "messages": messages, "stream": True},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        primary: BaseException | None = None
        try:
            for item in response:
                yield StreamChunk(
                    provider=self.id,
                    model=selected_model,
                    content=delta_content(item),
                    done=item.get("done") is True,
                    metadata={"base_url": self.base_url, "network": True, "external": True},
                )
        except BaseException as exc:
            primary = exc
            raise
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as cleanup:
                    if primary is None:
                        raise
                    try:
                        primary.add_note("OpenAI-compatible stream cleanup failed: " + failure_summary(cleanup))
                    except BaseException:
                        pass

    def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> EmbeddingResult:
        self.require_configured()
        selected_model = model or self.default_embedding_model
        response = post_json(
            f"{self.base_url}/embeddings",
            {"model": selected_model, "input": texts},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        data = response.get("data", [])
        embeddings = [[float(value) for value in item.get("embedding", [])] for item in data]
        dimensions = len(embeddings[0]) if embeddings else 0
        return EmbeddingResult(
            provider=self.id,
            model=selected_model,
            embeddings=embeddings,
            dimensions=dimensions,
            metadata={"base_url": self.base_url, "network": True, "external": True},
        )

    def require_configured(self) -> None:
        if not self.available():
            raise ProviderConfigurationError(
                "OPENAGENT_OPENAI_BASE_URL and OPENAGENT_OPENAI_API_KEY are required."
            )


class ConfiguredOpenAICompatibleProvider(OpenAICompatibleProvider):
    """A named provider using the OpenAI-compatible chat and embedding API."""

    def __init__(
        self,
        *,
        provider_id: str,
        label: str,
        environment_prefix: str,
        default_base_url: str,
        default_model: str,
    ) -> None:
        self.id = provider_id
        self.label = label
        self.environment_prefix = environment_prefix
        self.base_url = os.environ.get(f"{environment_prefix}_BASE_URL", default_base_url).rstrip("/")
        self.api_key = os.environ.get(f"{environment_prefix}_API_KEY", "")
        self.default_model = os.environ.get(f"{environment_prefix}_MODEL", default_model)
        self.default_embedding_model = os.environ.get(f"{environment_prefix}_EMBED_MODEL", "")

    def embed(self, texts: list[str], *, model: str | None = None) -> EmbeddingResult:
        if not (model or self.default_embedding_model):
            raise ProviderConfigurationError(
                f"{self.environment_prefix}_EMBED_MODEL is required because {self.label} embedding defaults are not assumed."
            )
        return super().embed(texts, model=model or self.default_embedding_model)

    def require_configured(self) -> None:
        if not self.available():
            raise ProviderConfigurationError(f"{self.environment_prefix}_API_KEY is required for provider '{self.id}'.")


COMPATIBLE_PROVIDER_SPECS = (
    ("deepseek", "DeepSeek", "OPENAGENT_DEEPSEEK", "https://api.deepseek.com/v1", "deepseek-chat"),
    ("mistral", "Mistral", "OPENAGENT_MISTRAL", "https://api.mistral.ai/v1", "mistral-small-latest"),
    ("groq", "Groq", "OPENAGENT_GROQ", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    ("together", "Together AI", "OPENAGENT_TOGETHER", "https://api.together.xyz/v1", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    ("openrouter", "OpenRouter", "OPENAGENT_OPENROUTER", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
    ("xai", "xAI", "OPENAGENT_XAI", "https://api.x.ai/v1", "grok-2-latest"),
    ("fireworks", "Fireworks AI", "OPENAGENT_FIREWORKS", "https://api.fireworks.ai/inference/v1", "accounts/fireworks/models/llama-v3p1-8b-instruct"),
    ("perplexity", "Perplexity", "OPENAGENT_PERPLEXITY", "https://api.perplexity.ai", "sonar"),
    ("qwen", "Qwen", "OPENAGENT_QWEN", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
)


class ModelGateway:
    def __init__(self, *, direct_hardware: Any = None) -> None:
        self.providers: dict[str, ModelProvider] = {
            "local": LocalProvider(),
            "openvino-direct": OpenVINODirectProvider(direct_hardware),
            "olmoe-direct": OlmoeDirectProvider(direct_hardware),
            "ollama": OllamaProvider(),
            "llamacpp": LlamaCppProvider(),
            "openai-compatible": OpenAICompatibleProvider(),
        }
        self.providers.update(
            {
                provider_id: ConfiguredOpenAICompatibleProvider(
                    provider_id=provider_id,
                    label=label,
                    environment_prefix=environment_prefix,
                    default_base_url=base_url,
                    default_model=default_model,
                )
                for provider_id, label, environment_prefix, base_url, default_model in COMPATIBLE_PROVIDER_SPECS
            }
        )
        if os.environ.get("OPENAGENT_HF_MOE_MODEL_DIR"):
            from .adaptive_engine.hf_moe_provider import HFMoeProvider

            self.providers["hf-moe-direct"] = HFMoeProvider()
        if os.environ.get("OPENAGENT_NATIVE_PUBLISHER_CONFIG"):
            from .adaptive_engine.publisher_provider import NativePublisherProvider

            self.providers["native-publisher"] = NativePublisherProvider()
        self.routing_policy = os.environ.get("OPENAGENT_MODEL_ROUTING", "local-first")
        self.fallback_policy = os.environ.get("OPENAGENT_MODEL_FALLBACK", "local")
        self.inference = AdaptiveCPUInferenceEngine()

    def specs(self) -> list[dict[str, Any]]:
        specs = []
        for provider in self.providers.values():
            spec = {
                "id": provider.id,
                "label": provider.label,
                "local": provider.local,
                "network": provider.network,
                "available": provider.available(),
                "default_model": provider.default_model,
                "smooth_runtime": smooth_runtime_profile(task="chat").to_dict(),
                "privacy_warning": provider.privacy_warning,
            }
            if isinstance(provider, OllamaProvider):
                installed = provider.installed_models()
                spec.update(
                    {
                        "installed_models": installed,
                        "selected_model": provider.resolve_model(None) if installed else provider.default_model,
                        "default_model_ready": any(model_name_matches(provider.default_model, name) for name in installed),
                    }
                )
            if provider.id == 'native-publisher':
                spec['vision_configured'] = provider.supports_vision()
            specs.append(spec)
        return [
            {
                "id": "auto",
                "label": "Auto Route",
                "local": True,
                "network": False,
                "available": True,
                "default_model": "policy-selected",
                "smooth_runtime": smooth_runtime_profile(task="chat").to_dict(),
                "privacy_warning": "Routes by policy. Default policy prefers local model servers, then the offline fallback.",
            },
            *specs,
        ]

    def policies(self) -> dict[str, Any]:
        external_providers = [provider_id for provider_id, provider in self.providers.items() if not provider.local]
        return {
            "routing_policy": self.routing_policy,
            "fallback_policy": self.fallback_policy,
            "routes": {
                "local-first": ["openvino-direct", "olmoe-direct", "ollama", "llamacpp", "local"],
                "local-server-first": ["openvino-direct", "olmoe-direct", "ollama", "llamacpp", "local"],
                "offline": ["local"],
                **({"native-publisher": ["native-publisher"]}
                   if "native-publisher" in self.providers else {}),
                "cloud-allowed": [*external_providers, "openvino-direct", "olmoe-direct", "ollama", "llamacpp", "local"],
            },
            "fallbacks": {
                "none": [],
                "local": ["local"],
            },
            "smooth_runtime": smooth_runtime_profile(task="chat").to_dict(),
            "runtime_settings": runtime_settings_snapshot(),
        }

    def apply_runtime_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        mapping = {
            "ollama_url": "OPENAGENT_OLLAMA_URL",
            "ollama_model": "OPENAGENT_OLLAMA_MODEL",
            "llamacpp_url": "OPENAGENT_LLAMACPP_URL",
            "llamacpp_model": "OPENAGENT_LLAMACPP_MODEL",
            "chat_model": "OPENAGENT_SMOOTH_CHAT_MODEL",
            "code_model": "OPENAGENT_SMOOTH_CODE_MODEL",
            "research_model": "OPENAGENT_SMOOTH_RESEARCH_MODEL",
            "voice_model": "OPENAGENT_SMOOTH_VOICE_ENGINE",
            "quantization_mode": "OPENAGENT_QUANTIZATION_MODE",
            "target_ram_gb": "OPENAGENT_TOTAL_RAM_GB",
            "model_timeout_seconds": "OPENAGENT_MODEL_TIMEOUT_SECONDS",
            "openvino_model_dir": "OPENAGENT_OPENVINO_MODEL_DIR",
            "openvino_device": "OPENAGENT_OPENVINO_DEVICE",
            "openvino_model_id": "OPENAGENT_OPENVINO_MODEL_ID",
            "olmoe_model_dir": "OPENAGENT_OLMOE_MODEL_DIR",
            "olmoe_manifest": "OPENAGENT_OLMOE_MANIFEST",
            "olmoe_expert_cache_gb": "OPENAGENT_OLMOE_EXPERT_CACHE_GB",
            "olmoe_context": "OPENAGENT_OLMOE_CONTEXT",
            "olmoe_io_mode": "OPENAGENT_OLMOE_IO_MODE",
            "olmoe_prefetch_workers": "OPENAGENT_OLMOE_PREFETCH_WORKERS",
            "olmoe_prefetch_depth": "OPENAGENT_OLMOE_PREFETCH_DEPTH",
            "olmoe_storage_roots": "OPENAGENT_OLMOE_STORAGE_ROOTS",
            "olmoe_kernel": "OPENAGENT_OLMOE_KERNEL",
            "olmoe_accelerator_cache_gb": "OPENAGENT_OLMOE_ACCELERATOR_CACHE_GB",
        }
        for key, env_key in mapping.items():
            value = str(settings.get(key, "")).strip()
            if value:
                os.environ[env_key] = value
        ollama = self.providers.get("ollama")
        if ollama and str(settings.get("ollama_url", "")).strip():
            setattr(ollama, "base_url", str(settings["ollama_url"]).strip().rstrip("/"))
        if ollama and str(settings.get("ollama_model", "")).strip():
            setattr(ollama, "default_model", normalize_model_name(str(settings["ollama_model"]).strip()))
        llamacpp = self.providers.get("llamacpp")
        if llamacpp and str(settings.get("llamacpp_url", "")).strip():
            setattr(llamacpp, "base_url", str(settings["llamacpp_url"]).strip().rstrip("/"))
        if llamacpp and str(settings.get("llamacpp_model", "")).strip():
            setattr(llamacpp, "default_model", str(settings["llamacpp_model"]).strip())
        direct = self.providers.get("openvino-direct")
        if isinstance(direct, OpenVINODirectProvider):
            direct.configure(
                model_path=str(settings.get("openvino_model_dir", "")).strip(),
                device=str(settings.get("openvino_device", "")).strip(),
                model_id=str(settings.get("openvino_model_id", "")).strip(),
            )
        olmoe = self.providers.get("olmoe-direct")
        if isinstance(olmoe, OlmoeDirectProvider):
            if str(settings.get("olmoe_model_dir", "")).strip():
                olmoe.model_path = str(settings["olmoe_model_dir"]).strip()
            if str(settings.get("olmoe_manifest", "")).strip():
                olmoe.manifest_path = str(settings["olmoe_manifest"]).strip()
            if str(settings.get("olmoe_expert_cache_gb", "")).strip():
                olmoe.cache_gb = max(0.25, float(settings["olmoe_expert_cache_gb"]))
            if str(settings.get("olmoe_context", "")).strip():
                olmoe.context_tokens = max(1, int(settings["olmoe_context"]))
            if str(settings.get("olmoe_io_mode", "")).strip():
                value = str(settings["olmoe_io_mode"]).strip().lower()
                if value not in {"buffered", "mmap", "direct", "uring"}:
                    raise ValueError("olmoe_io_mode must be buffered, mmap, direct or uring")
                olmoe.io_mode = value
            if str(settings.get("olmoe_prefetch_workers", "")).strip():
                olmoe.prefetch_workers = max(1, int(settings["olmoe_prefetch_workers"]))
            if str(settings.get("olmoe_prefetch_depth", "")).strip():
                olmoe.prefetch_depth = max(1, int(settings["olmoe_prefetch_depth"]))
            if olmoe.prefetch_depth < olmoe.prefetch_workers:
                raise ValueError("olmoe_prefetch_depth must be at least olmoe_prefetch_workers")
            if "olmoe_storage_roots" in settings:
                olmoe.storage_roots = _olmoe_storage_roots(str(settings.get("olmoe_storage_roots", "")))
            if "olmoe_kernel" in settings:
                olmoe.preferred_kernel = str(settings.get("olmoe_kernel", "")).strip() or None
            if str(settings.get("olmoe_accelerator_cache_gb", "")).strip():
                olmoe.accelerator_cache_gb = max(
                    0.25, float(settings["olmoe_accelerator_cache_gb"])
                )
            with olmoe._lock:
                if olmoe._runtime is not None:
                    olmoe._runtime.close()
                    olmoe._runtime = None
        return runtime_settings_snapshot()

    def discover_models(self) -> list[dict[str, Any]]:
        """Auto-discover available local models from Ollama and llama.cpp.

        Attempts to query each local server provider and returns the list of
        available models. Does not call external cloud providers.
        """
        discovered: list[dict[str, Any]] = []
        # Always include the offline local provider
        discovered.append({
            "provider": "local",
            "model": "openagent-local",
            "available": True,
            "type": "offline",
        })
        direct_provider = self.providers.get("openvino-direct")
        if isinstance(direct_provider, OpenVINODirectProvider) and direct_provider.available():
            discovered.append({
                "provider": direct_provider.id,
                "model": direct_provider.default_model,
                "available": True,
                "type": "direct-hardware",
                "device": direct_provider.device,
                "path": direct_provider.model_path,
            })
        olmoe_provider = self.providers.get("olmoe-direct")
        if isinstance(olmoe_provider, OlmoeDirectProvider) and olmoe_provider.available():
            discovered.append({
                "provider": olmoe_provider.id,
                "model": olmoe_provider.default_model,
                "available": True,
                "type": "direct-sparse-expert",
                "path": olmoe_provider.model_path,
            })
        # Try Ollama model list
        ollama_provider = next(
            (p for p in self.providers.values() if p.id == "ollama"), None
        )
        if ollama_provider is not None and local_http_available(f"{getattr(ollama_provider, 'base_url', 'http://127.0.0.1:11434')}/api/tags"):
            try:
                base_url = getattr(ollama_provider, "base_url", "http://127.0.0.1:11434")
                data = get_json(f"{base_url}/api/tags", timeout_seconds=4)
                for model in data.get("models", []):
                    discovered.append({
                        "provider": "ollama",
                        "model": model.get("name", "unknown"),
                        "available": True,
                        "type": "local-server",
                        "size": model.get("size"),
                    })
            except Exception:
                pass
        # Try llama.cpp model info
        llamacpp_provider = next(
            (p for p in self.providers.values() if p.id == "llamacpp"), None
        )
        if llamacpp_provider is not None and llamacpp_provider.available():
            try:
                base_url = getattr(llamacpp_provider, "base_url", "http://127.0.0.1:8080")
                data = get_json(f"{base_url}/v1/models", timeout_seconds=4)
                for model in data.get("data", []):
                    discovered.append({
                        "provider": "llamacpp",
                        "model": model.get("id", "llama.cpp"),
                        "available": True,
                        "type": "local-server",
                    })
            except Exception:
                pass
        return discovered

    def score_brio(self, state: Any, options: list[str] | tuple[str, ...], *,
                   question: str = "", model: str | None = None,
                   normalize: str = "mean", cancelled: Any = None,
                   cache_slot: int | None = None) -> dict[str, Any]:
        """Route closed-set scoring only to the explicitly configured native engine."""
        provider = self.providers.get("native-publisher")
        if provider is None:
            raise ProviderUnavailableError("native publisher scoring is not configured")
        scorer = getattr(provider, "score_brio", None)
        if not callable(scorer):
            raise ProviderUnavailableError("native publisher does not expose closed-set scoring")
        return scorer(state, options, question=question, model=model,
                      normalize=normalize, cancelled=cancelled, cache_slot=cache_slot)

    def score_brio_questions(self, state: Any, questions: list[dict[str, Any]], *,
                             model: str | None = None, normalize: str = "mean",
                             cancelled: Any = None, option_limit: int = 64,
                             cache_slot: int | None = None,
                             jev_model_alias: bool = False) -> dict[str, Any]:
        provider = self.providers.get("native-publisher")
        scorer = getattr(provider, "score_brio_questions", None)
        if not callable(scorer):
            raise ProviderUnavailableError("native publisher batch scoring is not configured")
        return scorer(state, questions, model=model, normalize=normalize, cancelled=cancelled,
                      option_limit=option_limit, cache_slot=cache_slot,
                      jev_model_alias=jev_model_alias)

    def score_brio_schema(self, state: Any, schema: dict[str, Any], *, task: str = "",
                          model: str | None = None, normalize: str = "mean",
                          cancelled: Any = None, cache_slot: int | None = None) -> dict[str, Any]:
        provider = self.providers.get("native-publisher")
        scorer = getattr(provider, "score_brio_schema", None)
        if not callable(scorer):
            raise ProviderUnavailableError("native publisher schema scoring is not configured")
        return scorer(state, schema, task=task, model=model,
                      normalize=normalize, cancelled=cancelled, cache_slot=cache_slot)

    @staticmethod
    def serialize_jev_state(value: Any) -> str:
        if type(value) is str:
            return value
        if type(value) in (dict, list):
            try:
                rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ProviderConfigurationError("Jev state/instructions must be JSON-serializable") from error
            if len(rendered.encode("utf-8")) > 1024**2:
                raise ProviderConfigurationError("serialized Jev text exceeds the 1 MiB limit")
            return rendered
        raise ProviderConfigurationError("Jev state/instructions must be text, object or array")

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        provider_id: str = "local",
        model: str | None = None,
        allow_external: bool = False,
        max_output_tokens: int | None = None,
        cancelled=None,
    ) -> ModelResult:
        if cancelled is not None and not callable(cancelled):
            raise ValueError("completion cancellation must be callable")

        def check_cancelled() -> None:
            if cancelled is not None and cancelled():
                raise ExecutionCancelled("completion request cancelled")

        check_cancelled()
        provider = self.select_provider(provider_id, allow_external=allow_external)
        selected_model = self._effective_provider_model(provider, provider_id=provider_id, model=model)
        plan = self.inference.plan(
            messages,
            provider_id=provider.id,
            model=selected_model,
            allow_external=allow_external,
            max_output_tokens=max_output_tokens,
        )
        started = time.perf_counter()
        cached = self.inference.get(plan.cache_key)
        if cached is not None:
            check_cancelled()
            latency_ms = (time.perf_counter() - started) * 1000
            metadata = {
                **cached.metadata,
                "cache_hit": True,
                "inference": plan.metadata(),
                "routing_policy": self.routing_policy,
                "fallback_policy": self.fallback_policy,
            }
            self.inference.record(
                plan,
                provider=cached.provider,
                model=cached.model,
                latency_ms=latency_ms,
                first_token_ms=latency_ms,
                cache_hit=True,
            )
            return ModelResult(cached.provider, cached.model, cached.content, metadata)
        fallback_used = False
        try:
            check_cancelled()
            provider_options = {"model": selected_model}
            if provider.id == "native-publisher" and cancelled is not None:
                provider_options["cancelled"] = cancelled
            with self.inference.activate(plan), _approve_model_egress(allow_external):
                result = provider.complete(plan.messages, **provider_options)
            check_cancelled()
            metadata = {
                **result.metadata,
                "routing_policy": self.routing_policy,
                "fallback_policy": self.fallback_policy,
            }
        except Exception as exc:
            # Cancellation is a request outcome, not a provider failure to
            # recover by replaying the prompt against a fallback backend.
            if isinstance(exc, (InterruptedError, ExecutionCancelled, _ModelStreamCancelled)):
                raise
            check_cancelled()
            fallback = self.select_fallback(provider.id)
            if fallback is None:
                raise
            fallback_used = True
            check_cancelled()
            with self.inference.activate(plan):
                result = fallback.complete(plan.messages, model=None)
            check_cancelled()
            metadata = {
                **result.metadata,
                "fallback_from": provider.id,
                "fallback_error": failure_detail(exc),
                "routing_policy": self.routing_policy,
                "fallback_policy": self.fallback_policy,
            }
        latency_ms = (time.perf_counter() - started) * 1000
        metadata = {
            **metadata,
            "cache_hit": False,
            "latency_ms": round(latency_ms, 3),
            "inference": plan.metadata(),
        }
        if not fallback_used:
            self.inference.put(
                plan.cache_key,
                provider=result.provider,
                model=result.model,
                content=result.content,
                metadata=metadata,
            )
        self.inference.record(
            plan,
            provider=result.provider,
            model=result.model,
            latency_ms=latency_ms,
            first_token_ms=latency_ms,
            cache_hit=False,
        )
        return ModelResult(result.provider, result.model, result.content, metadata)

    def complete_vision(
        self,
        messages: list[dict[str, str]],
        image: bytes,
        *,
        media_type: str = "image/png",
        provider_id: str = "auto",
        model: str | None = None,
        allow_external: bool = False,
        max_output_tokens: int | None = None,
    ) -> ModelResult:
        """Complete a request through a provider's native image protocol.

        Vision requests never fall back to a text provider because doing so
        would silently discard the image. External endpoints retain the same
        explicit approval boundary as text completion.
        """

        if not image:
            raise ValueError("Vision completion requires non-empty image bytes.")
        if not media_type.startswith("image/"):
            raise ValueError("Vision completion media_type must be an image MIME type.")
        provider = self._select_vision_provider(provider_id, allow_external=allow_external)
        selected_model = self._effective_provider_model(provider, provider_id=provider_id, model=model)
        if isinstance(provider, OllamaProvider):
            with _approve_model_egress(allow_external):
                selected_model = provider.resolve_vision_model(selected_model)
        started = time.perf_counter()
        try:
            with _approve_model_egress(allow_external):
                result = provider.complete_vision(
                    messages,
                    image,
                    media_type=media_type,
                    model=selected_model,
                    max_output_tokens=max_output_tokens,
                )
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Vision completion failed through provider '{provider.id}' using model "
                f"'{selected_model or provider.default_model}': {exc}"
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        return ModelResult(
            provider=result.provider,
            model=result.model,
            content=result.content,
            metadata={
                **result.metadata,
                "provider": result.provider,
                "model": result.model,
                "modality": "vision",
                "latency_ms": round(latency_ms, 3),
                "routing_policy": self.routing_policy,
                "external_approved": bool(allow_external and not provider.local),
            },
        )

    def stream_complete_vision(
        self,
        messages: list[dict[str, str]],
        image: bytes,
        *,
        media_type: str = "image/png",
        provider_id: str = "auto",
        model: str | None = None,
        allow_external: bool = False,
        max_output_tokens: int | None = None,
        cancelled=None,
    ) -> Iterator[StreamChunk]:
        """Route image streams through the admitted image-aware provider."""
        if cancelled is not None and not callable(cancelled):
            raise ValueError("stream cancellation must be callable")
        def check_cancelled() -> None:
            if cancelled is not None and cancelled():
                raise ExecutionCancelled("vision stream request cancelled")
        check_cancelled()
        if type(image) is not bytes or not image or len(image) > 256*1024**2:
            raise ValueError("Vision streaming requires bounded non-empty image bytes.")
        if type(media_type) is not str or not media_type.startswith("image/"):
            raise ValueError("Vision streaming requires an image MIME type.")
        if max_output_tokens is not None and (type(max_output_tokens) is not int or max_output_tokens <= 0):
            raise ValueError("Vision streaming requires a positive output token limit.")
        provider = self._select_vision_provider(provider_id, allow_external=allow_external)
        selected_model = self._effective_provider_model(provider, provider_id=provider_id, model=model)
        if isinstance(provider, OllamaProvider):
            with _approve_model_egress(allow_external):
                selected_model = provider.resolve_vision_model(selected_model)
        stream = getattr(provider, 'stream_complete_vision', None)
        started = time.perf_counter()
        if not callable(stream):
            result = self.complete_vision(messages, image, media_type=media_type,
                provider_id=provider.id, model=selected_model, allow_external=allow_external,
                max_output_tokens=max_output_tokens)
            metadata = {**result.metadata, 'streaming_mode': 'buffered-vision'}
            for part in chunk_text(result.content):
                check_cancelled()
                yield StreamChunk(result.provider, result.model, part, False, metadata)
            check_cancelled()
            yield StreamChunk(result.provider, result.model, '', True, metadata)
            return
        iterator = None
        first_token_ms = None
        primary = None
        terminal_seen = False
        try:
            check_cancelled()
            with _approve_model_egress(allow_external):
                stream_options = dict(media_type=media_type, model=selected_model,
                                      max_output_tokens=max_output_tokens)
                if provider.id in {'native-publisher', 'ollama', 'openai-compatible'}:
                    stream_options['cancelled'] = cancelled
                iterator = _closing_provider_stream(iter(stream(messages, image, **stream_options)))
                for chunk in iterator:
                    check_cancelled()
                    terminal_seen = _validate_stream_chunk(
                        chunk, provider_id=provider.id, terminal_seen=terminal_seen)
                    if chunk.content and first_token_ms is None:
                        first_token_ms = (time.perf_counter()-started)*1000
                    metadata = {**chunk.metadata, 'modality': 'vision',
                        'routing_policy': self.routing_policy,
                        'external_approved': bool(allow_external and not provider.local),
                        'streaming_mode': 'native-vision'}
                    if terminal_seen:
                        metadata.update(latency_ms=round((time.perf_counter()-started)*1000, 3),
                                        first_token_ms=None if first_token_ms is None else round(first_token_ms, 3))
                    yield StreamChunk(chunk.provider, chunk.model, chunk.content, chunk.done, metadata)
                    if terminal_seen:
                        return
                raise ProviderUnavailableError('Vision stream ended without a completion event.')
        except ProviderUnavailableError as exc:
            primary = exc
            raise
        except ExecutionCancelled as exc:
            primary = exc
            raise
        except Exception as exc:
            primary = ProviderUnavailableError(
                f"Vision streaming failed through provider '{provider.id}': {exc}")
            raise primary from exc
        except BaseException as exc:
            primary = exc
            raise
        finally:
            close = getattr(iterator, 'close', None)
            if callable(close):
                try:
                    close()
                except BaseException as cleanup:
                    if primary is None:
                        raise
                    try:
                        primary.add_note('vision provider iterator cleanup failed: '+
                                         failure_summary(cleanup))
                    except BaseException:
                        pass

    def _select_vision_provider(self, provider_id: str, *, allow_external: bool) -> ModelProvider:
        if provider_id == "auto":
            ollama = self.providers["ollama"]
            if ollama.available():
                return ollama
            native = self.providers.get('native-publisher')
            if (native is not None and native.available() and
                    callable(getattr(native, 'supports_vision', None)) and native.supports_vision()):
                return native
            external = self.providers["openai-compatible"]
            if allow_external and external.available():
                return external
            raise ProviderUnavailableError(
                "No vision provider is available. Configure native publisher vision, start Ollama with a vision-capable model, or configure an "
                "OpenAI-compatible vision endpoint and pass allow_external=true."
            )
        provider = self.select_provider(provider_id, allow_external=allow_external)
        if provider.id == 'native-publisher':
            if not provider.supports_vision():
                raise ProviderUnavailableError('Native publisher vision requires an image placeholder and vision configuration.')
            return provider
        if provider.id not in {"ollama", "openai-compatible"}:
            raise ProviderUnavailableError(
                f"Provider '{provider.id}' does not support native vision completion. "
                "Select 'ollama' or a configured 'openai-compatible' vision provider."
            )
        return provider

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        *,
        provider_id: str = "local",
        model: str | None = None,
        allow_external: bool = False,
        max_output_tokens: int | None = None,
        cancelled=None,
    ) -> Iterator[StreamChunk]:
        if cancelled is not None and not callable(cancelled):
            raise ValueError("stream cancellation must be callable")
        def check_cancelled() -> None:
            if cancelled is not None and cancelled():
                raise ExecutionCancelled("stream request cancelled")
        check_cancelled()
        provider = self.select_provider(provider_id, allow_external=allow_external)
        selected_model = self._effective_provider_model(provider, provider_id=provider_id, model=model)
        plan = self.inference.plan(
            messages,
            provider_id=provider.id,
            model=selected_model,
            allow_external=allow_external,
            max_output_tokens=max_output_tokens,
        )
        started = time.perf_counter()
        cached = self.inference.get(plan.cache_key)
        if cached is not None:
            first_token_ms: float | None = None
            metadata = {
                **cached.metadata,
                "cache_hit": True,
                "inference": plan.metadata(),
                "routing_policy": self.routing_policy,
                "fallback_policy": self.fallback_policy,
            }
            for part in chunk_text(cached.content):
                check_cancelled()
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000
                yield StreamChunk(cached.provider, cached.model, part, False, metadata)
            check_cancelled()
            yield StreamChunk(cached.provider, cached.model, "", True, metadata)
            latency_ms = (time.perf_counter() - started) * 1000
            self.inference.record(
                plan,
                provider=cached.provider,
                model=cached.model,
                latency_ms=latency_ms,
                first_token_ms=first_token_ms or latency_ms,
                cache_hit=True,
            )
            return
        parts: list[str] = []
        first_token_ms: float | None = None
        final_provider = provider.id
        final_model = selected_model or provider.default_model
        final_metadata: dict[str, Any] = {}
        fallback_used = False
        stream_started = False
        terminal_seen = False
        try:
            check_cancelled()
            provider_options = {"model": selected_model}
            if provider.id == 'native-publisher':
                provider_options['cancelled'] = cancelled
            stream = _closing_provider_stream(
                provider.stream_complete(plan.messages, **provider_options))
            try:
                with self.inference.activate(plan), _approve_model_egress(allow_external):
                    for chunk in stream:
                        check_cancelled()
                        terminal_seen = _validate_stream_chunk(
                            chunk, provider_id=provider.id, terminal_seen=terminal_seen)
                        if chunk.content:
                            parts.append(chunk.content)
                            if first_token_ms is None:
                                first_token_ms = (time.perf_counter() - started) * 1000
                        final_provider = chunk.provider
                        final_model = chunk.model
                        final_metadata = {
                            **chunk.metadata,
                            "cache_hit": False,
                            "inference": plan.metadata(),
                            "routing_policy": self.routing_policy,
                            "fallback_policy": self.fallback_policy,
                        }
                        # Once any event is visible to the caller, replaying the
                        # whole prompt through a fallback would splice two
                        # providers' output into one response. Fail the stream
                        # instead of publishing a mixed/duplicated completion.
                        stream_started = True
                        yield StreamChunk(chunk.provider, chunk.model, chunk.content, chunk.done, final_metadata)
                    if not terminal_seen:
                        raise ProviderUnavailableError(
                            f"Provider '{provider.id}' stream ended without a terminal event"
                        )
            finally:
                stream.close()
        except ExecutionCancelled:
            raise
        except Exception as exc:
            # Cancellation wins over provider recovery. Starting fallback
            # after the client has gone away would create fresh work with no
            # consumer and could publish a second provider's output later.
            check_cancelled()
            fallback = self.select_fallback(provider.id)
            if fallback is None or stream_started:
                raise
            fallback_used = True
            parts = []
            terminal_seen = False
            stream = _closing_provider_stream(
                fallback.stream_complete(plan.messages, model=None))
            try:
                with self.inference.activate(plan):
                    for chunk in stream:
                        check_cancelled()
                        terminal_seen = _validate_stream_chunk(
                            chunk, provider_id=fallback.id, terminal_seen=terminal_seen)
                        if chunk.content:
                            parts.append(chunk.content)
                            if first_token_ms is None:
                                first_token_ms = (time.perf_counter() - started) * 1000
                        final_provider = chunk.provider
                        final_model = chunk.model
                        final_metadata = {
                            **chunk.metadata,
                            "fallback_from": provider.id,
                            "fallback_error": failure_detail(exc),
                            "cache_hit": False,
                            "inference": plan.metadata(),
                            "routing_policy": self.routing_policy,
                            "fallback_policy": self.fallback_policy,
                        }
                        yield StreamChunk(chunk.provider, chunk.model, chunk.content, chunk.done, final_metadata)
                    if not terminal_seen:
                        raise ProviderUnavailableError(
                            f"Fallback provider '{fallback.id}' stream ended without a terminal event"
                        )
            finally:
                stream.close()
        latency_ms = (time.perf_counter() - started) * 1000
        final_metadata = {**final_metadata, "latency_ms": round(latency_ms, 3)}
        if parts and not fallback_used:
            self.inference.put(
                plan.cache_key,
                provider=final_provider,
                model=final_model,
                content="".join(parts),
                metadata=final_metadata,
            )
        self.inference.record(
            plan,
            provider=final_provider,
            model=final_model,
            latency_ms=latency_ms,
            first_token_ms=first_token_ms,
            cache_hit=False,
        )

    def embed(
        self,
        texts: list[str],
        *,
        provider_id: str = "local",
        model: str | None = None,
        allow_external: bool = False,
    ) -> EmbeddingResult:
        if not texts:
            raise ValueError("At least one text input is required.")
        provider = self.select_provider(provider_id, allow_external=allow_external)
        cache_key = self.inference.embedding_key(
            texts,
            provider_id=provider.id,
            model=model,
            allow_external=allow_external,
        )
        cached = self.inference.get_embedding(cache_key)
        if cached is not None:
            return EmbeddingResult(
                cached.provider,
                cached.model,
                [list(vector) for vector in cached.embeddings],
                cached.dimensions,
                {
                    **cached.metadata,
                    "cache_hit": True,
                    "inference": {
                        "engine": "openagent-adaptive-cpu-v1",
                        "operation": "embedding",
                        "batch_size": len(texts),
                        "input_characters": sum(len(text) for text in texts),
                    },
                },
            )
        started = time.perf_counter()
        fallback_used = False
        try:
            with _approve_model_egress(allow_external):
                result = provider.embed(texts, model=model)
            metadata = dict(result.metadata)
        except Exception as exc:
            fallback = self.select_fallback(provider.id)
            if fallback is None:
                raise
            fallback_used = True
            result = fallback.embed(texts, model=None)
            metadata = {**result.metadata, "fallback_from": provider.id,
                        "fallback_error": failure_detail(exc)}
        latency_ms = (time.perf_counter() - started) * 1000
        metadata = {
            **metadata,
            "cache_hit": False,
            "latency_ms": round(latency_ms, 3),
            "inference": {
                "engine": "openagent-adaptive-cpu-v1",
                "operation": "embedding",
                "batch_size": len(texts),
                "input_characters": sum(len(text) for text in texts),
                "value_preserving_cache": True,
            },
            "routing_policy": self.routing_policy,
            "fallback_policy": self.fallback_policy,
        }
        if not fallback_used:
            self.inference.put_embedding(
                cache_key,
                provider=result.provider,
                model=result.model,
                embeddings=result.embeddings,
                dimensions=result.dimensions,
                metadata=metadata,
            )
        return EmbeddingResult(result.provider, result.model, result.embeddings, result.dimensions, metadata)

    def inference_status(self) -> dict[str, Any]:
        return self.inference.status()

    def benchmark_inference_engine(self, *, iterations: int = 100) -> dict[str, Any]:
        return self.inference.benchmark(iterations=iterations)

    def clear_inference_cache(self) -> dict[str, Any]:
        return self.inference.clear()

    def warmup(self, *, provider_id: str = "auto", model: str | None = None) -> dict[str, Any]:
        provider = self.select_provider(provider_id, allow_external=False)
        selected_model = self._effective_provider_model(provider, provider_id=provider_id, model=model)
        if not provider.local:
            raise ProviderConfigurationError("Inference warmup is restricted to local providers.")
        messages = [
            {"role": "system", "content": "This is a local runtime warmup."},
            {"role": "user", "content": "Reply only READY."},
        ]
        plan = self.inference.plan(
            messages,
            provider_id=provider.id,
            model=selected_model,
            allow_external=False,
        )
        started = time.perf_counter()
        with self.inference.activate(plan):
            result = provider.complete(plan.messages, model=selected_model)
        latency_ms = (time.perf_counter() - started) * 1000
        metadata = {**result.metadata, "warmup": True, "inference": plan.metadata()}
        self.inference.put(
            plan.cache_key,
            provider=result.provider,
            model=result.model,
            content=result.content,
            metadata=metadata,
        )
        self.inference.record(
            plan,
            provider=result.provider,
            model=result.model,
            latency_ms=latency_ms,
            first_token_ms=latency_ms,
            cache_hit=False,
        )
        return {
            "status": "ready",
            "provider": result.provider,
            "model": result.model,
            "latency_ms": round(latency_ms, 3),
            "target_first_token_ms": plan.target_first_token_ms,
            "within_target": latency_ms <= plan.target_first_token_ms,
            "privacy_boundary": "Warmup contacted only the selected local model provider.",
        }

    @staticmethod
    def _effective_provider_model(
        provider: ModelProvider,
        *,
        provider_id: str,
        model: str | None,
    ) -> str | None:
        """Preserve the real model identity when Auto selects a direct graph."""

        if provider_id == "auto" and provider.id in {"openvino-direct", "olmoe-direct", "hf-moe-direct"}:
            requested = str(model or "").strip()
            return requested if requested == provider.default_model else None
        return model

    def select_provider(self, provider_id: str, *, allow_external: bool = False) -> ModelProvider:
        if provider_id == "auto":
            for candidate_id in self.route_order(allow_external=allow_external):
                provider = self.providers[candidate_id]
                with _approve_model_egress(allow_external):
                    available = provider.available()
                if available:
                    return provider
            return self.providers["local"]

        provider = self.providers.get(provider_id)
        if provider is None:
            known = ", ".join(["auto", *sorted(self.providers)])
            raise ValueError(f"Unknown model provider '{provider_id}'. Known providers: {known}")
        if not provider.local and not allow_external:
            raise ProviderConfigurationError(
                f"Provider '{provider_id}' crosses the local data boundary. "
                "Pass allow_external=true to use it."
            )
        return provider

    def route_order(self, *, allow_external: bool) -> list[str]:
        policies = self.policies()["routes"]
        configured = policies.get(self.routing_policy, policies["local-first"])
        if "hf-moe-direct" in self.providers and self.routing_policy != "local":
            configured = ["hf-moe-direct", *configured]
        if allow_external:
            return list(configured)
        return [provider_id for provider_id in configured if self.providers[provider_id].local]

    def select_fallback(self, failed_provider_id: str) -> ModelProvider | None:
        if self.fallback_policy == "none":
            return None
        fallback = self.providers.get("local")
        if fallback and fallback.id != failed_provider_id:
            return fallback
        return None


def _olmoe_prompt(messages: list[dict[str, str]]) -> str:
    """Render OLMoE-Instruct's user/assistant turn structure."""
    parts = ["|||IP_ADDRESS|||"]
    rendered: list[tuple[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported OLMoE message role: {role}")
        rendered.append((str(role), content))
    if not rendered or rendered[-1][0] != "user":
        raise ValueError("OLMoE completion requires a final user message")
    previous = ""
    for role, content in rendered:
        if role == "system":
            if previous:
                raise ValueError("OLMoE system messages must precede conversation turns")
            parts.append(f"<|system|>\n{content}\n")
        elif role == "user":
            if previous not in {"", "system", "assistant"}:
                raise ValueError("OLMoE user and assistant messages must alternate")
            parts.append(f"<|user|>\n{content}\n")
        else:
            if previous != "user":
                raise ValueError("OLMoE user and assistant messages must alternate")
            parts.append(f"<|assistant|>\n{content}|||IP_ADDRESS|||\n")
        previous = role
    parts.append("<|assistant|>\n")
    return "".join(parts)


def _olmoe_storage_roots(value: str) -> dict[str, Path]:
    if not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("OPENAGENT_OLMOE_STORAGE_ROOTS must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("OPENAGENT_OLMOE_STORAGE_ROOTS must be a JSON object")
    roots: dict[str, Path] = {}
    for name, raw_path in payload.items():
        if not isinstance(name, str) or not name or "/" in name or "\\" in name:
            raise ValueError("OLMoE storage root names must be simple identifiers")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("OLMoE storage root paths must be non-empty strings")
        roots[name] = Path(raw_path).expanduser().resolve(strict=True)
        if not roots[name].is_dir():
            raise ValueError(f"OLMoE storage root is not a directory: {roots[name]}")
    return roots


def chunk_text(text: str, *, size: int = 80) -> Iterator[str]:
    if not text:
        return
    for start in range(0, len(text), size):
        yield text[start : start + size]


def provider_request_options(messages: list[dict[str, str]]) -> dict[str, Any]:
    active = active_runtime_options()
    if active is not None:
        return active
    task = infer_task(messages)
    return model_request_options(
        task=task,
        prompt_tokens=estimate_messages_tokens(messages),
        low_latency=True,
    )


def chat_completion_content(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if choices:
        return str(choices[0].get("message", {}).get("content", ""))
    return json.dumps(response, sort_keys=True)


def delta_content(item: dict[str, Any]) -> str:
    choices = item.get("choices", [])
    if not choices:
        return ""
    return str(choices[0].get("delta", {}).get("content", ""))


def local_http_available(url: str, *, timeout: float = 0.35) -> bool:
    try:
        response = EgressClient().fetch(
            url, approved=_external_approval.get(), allow_loopback=True,
            purpose="model provider availability", timeout_seconds=timeout,
            max_response_bytes=1024 * 1024, max_redirects=0,
            headers={"Accept": "application/json"},
        )
        return 200 <= response.status < 300
    except Exception:
        return False


def model_name_matches(requested: str, available: str) -> bool:
    left = normalize_model_name(requested).lower()
    right = normalize_model_name(available).lower()
    if not left or not right:
        return False
    return left == right or f"{left}:latest" == right or left == right.removesuffix(":latest")


def preferred_ollama_models() -> list[str]:
    return [
        CPU_MODEL_STACK["chat"],
        CPU_MODEL_STACK["speed"],
        CPU_MODEL_STACK["code"],
        CPU_MODEL_STACK["research"],
        CPU_MODEL_STACK["extraction"],
        CPU_MODEL_STACK["low_code"],
        CPU_MODEL_STACK["high_research"],
    ]


def normalize_model_name(model_name: str | None) -> str:
    value = str(model_name or "").strip()
    aliases = {
        "phi4-mini:3.8b-instruct-q4_k_m": "phi4-mini",
        "qwen2.5-coder:3b-instruct-q4_k_m": "qwen2.5-coder:3b",
        "gemma3:2b-it-q4_k_m": "gemma3:2b",
        "deepseek-coder:1.3b-instruct-q4_k_m": "deepseek-coder:1.3b",
        "llama3.2:3b-instruct-q4_k_m": "llama3.2:3b",
        "qwen2.5:7b-instruct-q4_k_m": "qwen2.5:7b",
    }
    lower = value.lower()
    if lower.endswith(":latest"):
        lower_without_latest = lower.removesuffix(":latest")
        return aliases.get(lower_without_latest, value.removesuffix(":latest"))
    return aliases.get(lower, value)


def local_assistant_response(user_message: str, system_context: str = "") -> str:
    prompt = user_message.strip()
    if not prompt:
        return (
            "Tell me what you want to build, debug, research, summarize, or plan.\n\n"
            "No local model server answered yet. Use Developer > Download Chat Model to pull the default Ollama model."
        )

    lower = prompt.lower()
    fallback_note = (
        "\n\nOffline fallback: no Ollama or llama.cpp model answered yet. Use Developer > Download Chat Model, then keep provider on Auto."
    )

    greetings = {"hi", "hello", "hey", "salam", "assalamualaikum", "asalamualaikum"}
    if lower in greetings or lower.startswith(("hi ", "hello ", "hey ")):
        return (
            "Hi. I am ready in the local workspace.\n\n"
            "You can ask me to draft, code, research, summarize, plan a workflow, inspect memory, or prepare an action with approval."
            + fallback_note
        )

    if any(word in lower for word in ["joke", "make me laugh", "funny"]):
        return (
            "Here is one:\n\n"
            "Why did the local AI refuse to gossip? Because it was trained to keep everything on-device.\n\n"
            "I can do better once a local model is pulled and running."
            + fallback_note
        )

    if any(word in lower for word in ["research", "investigate", "source", "find out", "compare", "analyze"]):
        return (
            f"Research plan for: {prompt}\n\n"
            "1. Define the exact question, assumptions, and success criteria.\n"
            "2. Search local knowledge first, then request explicit approval before any external source lookup.\n"
            "3. Capture source notes with dates, authors, and reliability signals.\n"
            "4. Synthesize findings into claims, evidence, uncertainties, and next actions.\n"
            "5. Save the final report and audit trail inside the project workspace.\n\n"
            "Use the Deep Research tab when you want this run as a structured research job."
            + fallback_note
        )

    if any(word in lower for word in ["code", "bug", "error", "traceback", "function", "python", "javascript", "typescript", "test"]):
        return (
            f"Code task understood: {prompt}\n\n"
            "Suggested local workflow:\n"
            "1. Reproduce the issue or define the expected behavior.\n"
            "2. Inspect the smallest relevant files and tests.\n"
            "3. Make a scoped change, preserving user edits.\n"
            "4. Run focused tests first, then the broader suite when the change touches shared behavior.\n"
            "5. Record the important outcome in audit/memory if it changes project behavior."
            + fallback_note
        )

    if any(word in lower for word in ["summarize", "summary", "brief", "extract"]):
        return (
            f"Summary target: {prompt}\n\n"
            "I can turn this into a concise brief with: key points, decisions, open questions, risks, and next actions. "
            "Paste the source text or point me to the local document/workspace item to summarize."
            + fallback_note
        )

    if any(word in lower for word in ["plan", "roadmap", "steps", "todo", "milestone"]):
        return (
            f"Plan for: {prompt}\n\n"
            "1. Clarify the final outcome and the non-negotiable constraints.\n"
            "2. Break it into runnable stages with visible checkpoints.\n"
            "3. Put risky or uncertain work early.\n"
            "4. Add verification for each stage.\n"
            "5. Keep privacy boundaries explicit for every action that can leave the local system."
            + fallback_note
        )

    return (
        f"I do not have a local model server online yet, but I can still route this request: {prompt}\n\n"
        "For real model responses, open Developer and click Download Chat Model. After Ollama has the model, keep Chat provider on Auto."
        + fallback_note
    )


def local_hash_embedding(text: str, *, dimensions: int) -> list[float]:
    values = [0.0 for _ in range(dimensions)]
    tokens = text.lower().split()
    for token in tokens or [text]:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for index, byte in enumerate(digest):
            bucket = index % dimensions
            values[bucket] += (byte - 127.5) / 127.5
    magnitude = sum(value * value for value in values) ** 0.5 or 1.0
    return [round(value / magnitude, 6) for value in values]


def get_json(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    response = EgressClient().fetch(
        url, approved=_external_approval.get(), allow_loopback=True,
        purpose="model provider discovery", timeout_seconds=timeout_seconds,
        max_response_bytes=4 * 1024 * 1024, max_redirects=0,
        headers={"Accept": "application/json"},
    )
    _require_model_http_success(response.status)
    return json.loads(response.body.decode("utf-8"))


def _require_model_http_success(status: int) -> None:
    if not 200 <= status < 300:
        raise ProviderUnavailableError(f"Model provider returned HTTP {status}.")


class _ModelStreamCancelled(Exception):
    pass


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    response = EgressClient().fetch(
        url, approved=_external_approval.get(), allow_loopback=True,
        purpose="model provider completion", method="POST",
        body=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        timeout_seconds=timeout_seconds or model_timeout_seconds(),
        max_response_bytes=32 * 1024 * 1024, max_redirects=0,
    )
    _require_model_http_success(response.status)
    return json.loads(response.body.decode("utf-8"))


def _stream_model_lines(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    cancelled=None,
) -> Iterator[str]:
    if cancelled is not None and not callable(cancelled):
        raise ValueError("model stream cancellation must be callable")
    events: Queue[tuple[str | None, BaseException | None]] = Queue(maxsize=64)
    stream_cancelled = threading.Event()
    request_cancelled = threading.Event()
    approved = _external_approval.get()

    def publish(line: str | None, error: BaseException | None = None) -> None:
        while not stream_cancelled.is_set():
            try:
                events.put((line, error), timeout=0.1)
                return
            except Full:
                continue
        raise _ModelStreamCancelled()

    def fetch_lines() -> None:
        try:
            response = EgressClient().fetch(
                url, approved=approved, allow_loopback=True,
                purpose="model provider streaming", method="POST",
                body=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", **headers},
                timeout_seconds=model_timeout_seconds(),
                max_response_bytes=32 * 1024 * 1024, max_redirects=0,
                on_line=lambda line: publish(line),
                # The caller's predicate is polled only by the consumer thread
                # below. Egress may read this event from both its socket reader
                # and cancellation watcher, so never hand it an arbitrary
                # caller callback.
                cancelled=request_cancelled.is_set,
            )
            _require_model_http_success(response.status)
        except _ModelStreamCancelled:
            return
        except InterruptedError as exc:
            if request_cancelled.is_set():
                try:
                    publish(None, ExecutionCancelled("model provider stream cancelled"))
                except _ModelStreamCancelled:
                    pass
            elif not stream_cancelled.is_set():
                try:
                    publish(None, exc)
                except _ModelStreamCancelled:
                    pass
        except Exception as exc:
            if not stream_cancelled.is_set():
                try:
                    publish(None, exc)
                except _ModelStreamCancelled:
                    pass
        finally:
            if not stream_cancelled.is_set():
                try:
                    publish(None)
                except _ModelStreamCancelled:
                    pass

    threading.Thread(target=fetch_lines, daemon=True, name="model-provider-stream").start()
    try:
        while True:
            try:
                line, error = events.get(timeout=0.1)
            except Empty:
                if cancelled is not None:
                    try:
                        if cancelled():
                            request_cancelled.set()
                            stream_cancelled.set()
                            raise ExecutionCancelled("model provider stream cancelled")
                    except ExecutionCancelled:
                        raise
                    except BaseException as exc:
                        request_cancelled.set()
                        stream_cancelled.set()
                        raise ExecutionCancelled("model stream cancellation callback failed") from exc
                continue
            if error is not None:
                raise error
            if line is None:
                break
            yield line
    finally:
        request_cancelled.set()
        stream_cancelled.set()


def _closing_model_lines(lines: Iterator[str]) -> Iterator[str]:
    """Ensure parser/consumer exits cancel the underlying HTTP stream."""
    source = iter(lines)
    primary: BaseException | None = None
    try:
        yield from source
    except BaseException as exc:
        primary = exc
        raise
    finally:
        close = getattr(source, "close", None)
        if callable(close):
            try:
                close()
            except BaseException as cleanup:
                if primary is None:
                    raise
                try:
                    primary.add_note("model line stream cleanup failed: " + failure_summary(cleanup))
                except BaseException:
                    pass


def stream_json_lines(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    cancelled=None,
) -> Iterator[dict[str, Any]]:
    for line in _closing_model_lines(_stream_model_lines(
        url, payload, headers=headers, cancelled=cancelled,
    )):
        raw = line.strip()
        if raw:
            yield json.loads(raw)


def stream_sse_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    cancelled=None,
) -> Iterator[dict[str, Any]]:
    done_seen = False
    bare_terminal_seen = False
    for line in _closing_model_lines(_stream_model_lines(
        url, payload, headers=headers, cancelled=cancelled,
    )):
        raw = line.strip()
        if not raw:
            continue
        if raw.startswith(":") or raw.startswith(("event:", "id:", "retry:")):
            # SSE comments and control fields do not carry model payload.
            continue
        if raw.startswith("data:"):
            data = raw[5:].lstrip()
            if data == "[DONE]":
                done_seen = True
                yield {"done": True}
                break
            yield json.loads(data)
        else:
            # Some local OpenAI-compatible servers emit bare JSONL records.
            # Keep that supported when they provide their own done record.
            item = json.loads(raw)
            bare_terminal_seen = bare_terminal_seen or (
                isinstance(item, dict) and item.get("done") is True
            )
            yield item
    if not done_seen and not bare_terminal_seen:
        raise ProviderUnavailableError(
            "OpenAI-compatible stream ended without a [DONE] marker or bare-JSON done record"
        )


def model_timeout_seconds() -> float:
    try:
        return max(5.0, min(120.0, float(os.environ.get("OPENAGENT_MODEL_TIMEOUT_SECONDS", "35"))))
    except ValueError:
        return 35.0


def vision_model_timeout_seconds() -> float:
    try:
        configured = float(os.environ.get("OPENAGENT_VISION_TIMEOUT_SECONDS", "180"))
    except ValueError:
        configured = 180.0
    return max(model_timeout_seconds(), min(900.0, max(30.0, configured)))


def runtime_settings_snapshot() -> dict[str, Any]:
    return {
        "ollama_url": os.environ.get("OPENAGENT_OLLAMA_URL", "http://127.0.0.1:11434"),
        "ollama_model": normalize_model_name(os.environ.get("OPENAGENT_OLLAMA_MODEL", CPU_MODEL_STACK["chat"])),
        "llamacpp_url": os.environ.get("OPENAGENT_LLAMACPP_URL", "http://127.0.0.1:8080"),
        "llamacpp_model": normalize_model_name(os.environ.get("OPENAGENT_LLAMACPP_MODEL", CPU_MODEL_STACK["chat"])),
        "chat_model": normalize_model_name(os.environ.get("OPENAGENT_SMOOTH_CHAT_MODEL", CPU_MODEL_STACK["chat"])),
        "code_model": normalize_model_name(os.environ.get("OPENAGENT_SMOOTH_CODE_MODEL", CPU_MODEL_STACK["code"])),
        "research_model": normalize_model_name(os.environ.get("OPENAGENT_SMOOTH_RESEARCH_MODEL", CPU_MODEL_STACK["research"])),
        "voice_model": os.environ.get("OPENAGENT_SMOOTH_VOICE_ENGINE", CPU_MODEL_STACK["voice"]),
        "speed_model": os.environ.get("OPENAGENT_SMOOTH_SPEED_MODEL", CPU_MODEL_STACK["speed"]),
        "low_resource_code_model": os.environ.get("OPENAGENT_SMOOTH_LOW_CODE_MODEL", CPU_MODEL_STACK["low_code"]),
        "extraction_model": os.environ.get("OPENAGENT_SMOOTH_EXTRACTION_MODEL", CPU_MODEL_STACK["extraction"]),
        "high_context_research_model": os.environ.get("OPENAGENT_SMOOTH_HIGH_RESEARCH_MODEL", CPU_MODEL_STACK["high_research"]),
        "quantization_mode": os.environ.get("OPENAGENT_QUANTIZATION_MODE", "auto"),
        "target_ram_gb": os.environ.get("OPENAGENT_TOTAL_RAM_GB", ""),
        "model_timeout_seconds": model_timeout_seconds(),
        "openvino_model_dir": os.environ.get("OPENAGENT_OPENVINO_MODEL_DIR", ""),
        "openvino_device": os.environ.get("OPENAGENT_OPENVINO_DEVICE", "GPU"),
        "openvino_model_id": os.environ.get("OPENAGENT_OPENVINO_MODEL_ID", "openagent-direct-chat"),
        "olmoe_model_dir": os.environ.get("OPENAGENT_OLMOE_MODEL_DIR", ""),
        "olmoe_manifest": os.environ.get("OPENAGENT_OLMOE_MANIFEST", ""),
        "olmoe_expert_cache_gb": os.environ.get("OPENAGENT_OLMOE_EXPERT_CACHE_GB", "4"),
        "olmoe_context": os.environ.get("OPENAGENT_OLMOE_CONTEXT", "4096"),
        "olmoe_io_mode": os.environ.get("OPENAGENT_OLMOE_IO_MODE", "mmap"),
        "olmoe_prefetch_workers": os.environ.get("OPENAGENT_OLMOE_PREFETCH_WORKERS", "2"),
        "olmoe_prefetch_depth": os.environ.get("OPENAGENT_OLMOE_PREFETCH_DEPTH", "8"),
        "olmoe_storage_roots": os.environ.get("OPENAGENT_OLMOE_STORAGE_ROOTS", ""),
        "olmoe_kernel": os.environ.get("OPENAGENT_OLMOE_KERNEL", ""),
        "olmoe_accelerator_cache_gb": os.environ.get("OPENAGENT_OLMOE_ACCELERATOR_CACHE_GB", "2"),
        "privacy_boundary": "Runtime settings update only the local OpenAgent process environment.",
    }
