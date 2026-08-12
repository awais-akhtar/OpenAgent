"""Model provider gateway.

The gateway defaults to deterministic local behavior so OpenAgent can run
offline. Network providers are called only when a user explicitly selects them
or an auto-routing policy allows local network providers.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterator, Protocol
from urllib.request import Request, urlopen

from .core_runtime import CPU_MODEL_STACK, model_request_options, smooth_runtime_profile
from .inference_accelerator import (
    AdaptiveCPUInferenceEngine,
    active_runtime_options,
    estimate_messages_tokens,
    infer_task,
)


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
        self._cache_lock = threading.RLock()

    def available(self) -> bool:
        return bool(self.installed_models())

    def installed_models(self) -> list[str]:
        now = time.monotonic()
        with self._cache_lock:
            if now - self._models_cached_at <= 3.0:
                return list(self._models_cache)
        try:
            request = Request(f"{self.base_url}/api/tags", headers={"Accept": "application/json"})
            with urlopen(request, timeout=0.5) as response:
                data = json.loads(response.read().decode("utf-8"))
            models = [str(item.get("name", "")).strip() for item in data.get("models", []) if str(item.get("name", "")).strip()]
        except Exception:
            models = []
        with self._cache_lock:
            self._models_cache = list(models)
            self._models_cached_at = now
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
        encoded = base64.b64encode(image).decode("ascii")
        vision_messages = [dict(message) for message in messages]
        image_index = next(
            (index for index in range(len(vision_messages) - 1, -1, -1) if vision_messages[index].get("role") == "user"),
            None,
        )
        if image_index is None:
            vision_messages.append({"role": "user", "content": "Describe the supplied image.", "images": [encoded]})
        else:
            vision_messages[image_index]["images"] = [encoded]
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
        for item in response:
            content = item.get("message", {}).get("content", "")
            done = bool(item.get("done"))
            yield StreamChunk(
                provider=self.id,
                model=selected_model,
                content=content,
                done=done,
                metadata={"base_url": self.base_url, "network": True, "local_network": True, "smooth_options": options},
            )

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
        self._cache_lock = threading.RLock()

    def available(self) -> bool:
        now = time.monotonic()
        with self._cache_lock:
            if now - self._availability_cached_at <= 3.0:
                return self._availability_cache
        available = local_http_available(f"{self.base_url}/v1/models")
        with self._cache_lock:
            self._availability_cache = available
            self._availability_cached_at = now
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
        for item in stream_sse_json(
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
        ):
            content = delta_content(item)
            done = item.get("done") is True
            yield StreamChunk(
                provider=self.id,
                model=selected_model,
                content=content,
                done=done,
                metadata={"base_url": self.base_url, "network": True, "local_network": True, "smooth_options": options},
            )


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
        for event in self.client.stream_generate(model_id, messages=runtime_messages, generation=generation):
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
        data_url = f"data:{media_type};base64,{base64.b64encode(image).decode('ascii')}"
        vision_messages: list[dict[str, Any]] = []
        image_attached = False
        for message in reversed(messages):
            if not image_attached and message.get("role") == "user":
                content: list[dict[str, Any]] = []
                if message.get("content"):
                    content.append({"type": "text", "text": str(message["content"])})
                content.append({"type": "image_url", "image_url": {"url": data_url}})
                vision_messages.append({"role": "user", "content": content})
                image_attached = True
            else:
                vision_messages.append(dict(message))
        vision_messages.reverse()
        if not image_attached:
            vision_messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the supplied image."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            )
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

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
    ) -> Iterator[StreamChunk]:
        self.require_configured()
        selected_model = model or self.default_model
        for item in stream_sse_json(
            f"{self.base_url}/chat/completions",
            {"model": selected_model, "messages": messages, "stream": True},
            headers={"Authorization": f"Bearer {self.api_key}"},
        ):
            yield StreamChunk(
                provider=self.id,
                model=selected_model,
                content=delta_content(item),
                done=item.get("done") is True,
                metadata={"base_url": self.base_url, "network": True, "external": True},
            )

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


class ModelGateway:
    def __init__(self, *, direct_hardware: Any = None) -> None:
        self.providers: dict[str, ModelProvider] = {
            "local": LocalProvider(),
            "openvino-direct": OpenVINODirectProvider(direct_hardware),
            "ollama": OllamaProvider(),
            "llamacpp": LlamaCppProvider(),
            "openai-compatible": OpenAICompatibleProvider(),
        }
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
        return {
            "routing_policy": self.routing_policy,
            "fallback_policy": self.fallback_policy,
            "routes": {
                "local-first": ["openvino-direct", "ollama", "llamacpp", "local"],
                "local-server-first": ["openvino-direct", "ollama", "llamacpp", "local"],
                "offline": ["local"],
                "cloud-allowed": ["openai-compatible", "openvino-direct", "ollama", "llamacpp", "local"],
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
        # Try Ollama model list
        ollama_provider = next(
            (p for p in self.providers.values() if p.id == "ollama"), None
        )
        if ollama_provider is not None and local_http_available(f"{getattr(ollama_provider, 'base_url', 'http://127.0.0.1:11434')}/api/tags"):
            try:
                base_url = getattr(ollama_provider, "base_url", "http://127.0.0.1:11434")
                req = Request(f"{base_url}/api/tags", headers={"Accept": "application/json"})
                with urlopen(req, timeout=4) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
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
                req = Request(f"{base_url}/v1/models", headers={"Accept": "application/json"})
                with urlopen(req, timeout=4) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
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

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        provider_id: str = "local",
        model: str | None = None,
        allow_external: bool = False,
        max_output_tokens: int | None = None,
    ) -> ModelResult:
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
            with self.inference.activate(plan):
                result = provider.complete(plan.messages, model=selected_model)
            metadata = {
                **result.metadata,
                "routing_policy": self.routing_policy,
                "fallback_policy": self.fallback_policy,
            }
        except Exception as exc:
            fallback = self.select_fallback(provider.id)
            if fallback is None:
                raise
            fallback_used = True
            with self.inference.activate(plan):
                result = fallback.complete(plan.messages, model=None)
            metadata = {
                **result.metadata,
                "fallback_from": provider.id,
                "fallback_error": str(exc),
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
            selected_model = provider.resolve_vision_model(selected_model)
        started = time.perf_counter()
        try:
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

    def _select_vision_provider(self, provider_id: str, *, allow_external: bool) -> ModelProvider:
        if provider_id == "auto":
            ollama = self.providers["ollama"]
            if ollama.available():
                return ollama
            external = self.providers["openai-compatible"]
            if allow_external and external.available():
                return external
            raise ProviderUnavailableError(
                "No vision provider is available. Start Ollama with a vision-capable model, or configure an "
                "OpenAI-compatible vision endpoint and pass allow_external=true."
            )
        provider = self.select_provider(provider_id, allow_external=allow_external)
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
    ) -> Iterator[StreamChunk]:
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
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000
                yield StreamChunk(cached.provider, cached.model, part, False, metadata)
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
        try:
            with self.inference.activate(plan):
                for chunk in provider.stream_complete(plan.messages, model=selected_model):
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
                    yield StreamChunk(chunk.provider, chunk.model, chunk.content, chunk.done, final_metadata)
        except Exception as exc:
            fallback = self.select_fallback(provider.id)
            if fallback is None:
                raise
            fallback_used = True
            parts = []
            with self.inference.activate(plan):
                for chunk in fallback.stream_complete(plan.messages, model=None):
                    if chunk.content:
                        parts.append(chunk.content)
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - started) * 1000
                    final_provider = chunk.provider
                    final_model = chunk.model
                    final_metadata = {
                        **chunk.metadata,
                        "fallback_from": provider.id,
                        "fallback_error": str(exc),
                        "cache_hit": False,
                        "inference": plan.metadata(),
                        "routing_policy": self.routing_policy,
                        "fallback_policy": self.fallback_policy,
                    }
                    yield StreamChunk(chunk.provider, chunk.model, chunk.content, chunk.done, final_metadata)
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
            result = provider.embed(texts, model=model)
            metadata = dict(result.metadata)
        except Exception as exc:
            fallback = self.select_fallback(provider.id)
            if fallback is None:
                raise
            fallback_used = True
            result = fallback.embed(texts, model=None)
            metadata = {**result.metadata, "fallback_from": provider.id, "fallback_error": str(exc)}
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

        if provider_id == "auto" and provider.id == "openvino-direct":
            requested = str(model or "").strip()
            return requested if requested == provider.default_model else None
        return model

    def select_provider(self, provider_id: str, *, allow_external: bool = False) -> ModelProvider:
        if provider_id == "auto":
            for candidate_id in self.route_order(allow_external=allow_external):
                provider = self.providers[candidate_id]
                if provider.available():
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
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=timeout):
            return True
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


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds or model_timeout_seconds()) as response:
        return json.loads(response.read().decode("utf-8"))


def stream_json_lines(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
) -> Iterator[dict[str, Any]]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urlopen(request, timeout=model_timeout_seconds()) as response:
        for line in response:
            raw = line.decode("utf-8").strip()
            if raw:
                yield json.loads(raw)


def stream_sse_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
) -> Iterator[dict[str, Any]]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urlopen(request, timeout=model_timeout_seconds()) as response:
        for line in response:
            raw = line.decode("utf-8").strip()
            if not raw:
                continue
            if raw == "data: [DONE]":
                yield {"done": True}
                break
            if raw.startswith("data: "):
                yield json.loads(raw.removeprefix("data: "))
            else:
                yield json.loads(raw)


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
        "privacy_boundary": "Runtime settings update only the local OpenAgent process environment.",
    }
