"""Local AI and model operations route contracts and handlers."""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs

from . import JsonRouteResult
from ._contracts import route_contract

ROUTE_GROUP = "local_ai"
ROUTES = (
    ("GET", "/api/models"),
    ("GET", "/api/model-policy"),
    ("GET", "/api/settings/runtime"),
    ("POST", "/api/settings/runtime"),
    ("POST", "/api/models/discover"),
    ("GET", "/api/local-ai/jobs"),
    ("POST", "/api/local-ai/jobs"),
    ("GET", "/api/local-ai/jobs/{id}"),
    ("POST", "/api/local-ai/jobs/cancel"),
    ("GET", "/api/local-ai/status"),
    ("GET", "/api/local-ai/setup/status"),
    ("GET", "/api/local-ai/smooth-runtime"),
    ("GET", "/api/local-ai/models"),
    ("GET", "/api/local-ai/benchmarks"),
    ("GET", "/api/local-ai/models/{id}/probe"),
    ("GET", "/api/local-ai/hardware"),
    ("GET", "/api/local-ai/adapters"),
    ("GET", "/api/local-ai/marketplace"),
    ("POST", "/api/local-ai/models/register"),
    ("POST", "/api/local-ai/models/download"),
    ("POST", "/api/local-ai/models/setup/job"),
    ("POST", "/api/local-ai/llamacpp/register"),
    ("POST", "/api/local-ai/models/auto-quantize"),
    ("POST", "/api/local-ai/models/delete"),
    ("POST", "/api/local-ai/models/{id}/dependencies"),
    ("POST", "/api/local-ai/models/{id}/tools"),
    ("POST", "/api/local-ai/benchmark"),
    ("POST", "/api/local-ai/cluster-plan"),
    ("POST", "/api/local-ai/models/quantize"),
    ("POST", "/api/local-ai/models/setup-chat"),
    ("POST", "/api/local-ai/ollama/pull"),
)

_CURATED_MARKETPLACE = [
    {"model_id": "phi4-mini:latest", "name": "Phi-4 Mini", "size_gb": 2.5, "family": "phi", "tags": ["chat", "fast"]},
    {"model_id": "qwen2.5:latest", "name": "Qwen 2.5", "size_gb": 4.7, "family": "qwen", "tags": ["chat", "reasoning"]},
    {"model_id": "qwen2.5vl:3b", "name": "Qwen 2.5 VL 3B", "size_gb": 2.1, "family": "qwen", "tags": ["vision", "chat"]},
    {"model_id": "llama3.2:3b", "name": "Llama 3.2 3B", "size_gb": 2.0, "family": "llama", "tags": ["chat", "fast"]},
    {"model_id": "gemma3:4b", "name": "Gemma 3 4B", "size_gb": 3.3, "family": "gemma", "tags": ["chat"]},
    {"model_id": "mistral:7b", "name": "Mistral 7B", "size_gb": 4.1, "family": "mistral", "tags": ["chat", "instruct"]},
    {"model_id": "deepseek-r1:8b", "name": "DeepSeek R1 8B", "size_gb": 4.9, "family": "deepseek", "tags": ["reasoning", "code"]},
]


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class LocalAiRouteDependencies:
    local_ai: Any                        # LocalAIInfrastructure
    read_json: Callable[[], dict[str, Any]]
    ollama_url: str = "http://localhost:11434"
    _runtime_settings: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Use object.__setattr__ because frozen=True
        if self._runtime_settings is None:
            object.__setattr__(self, "_runtime_settings", {
                "cpu_threads": 0,
                "memory_limit_mb": 0,
                "ollama_url": self.ollama_url,
            })


def _ollama_get(url: str, timeout: int = 5) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _ollama_tags(ollama_url: str) -> list[dict[str, Any]]:
    data = _ollama_get(f"{ollama_url}/api/tags")
    if data and "models" in data:
        return data["models"]
    return []


def _ollama_reachable(ollama_url: str) -> bool:
    return _ollama_get(f"{ollama_url}/api/tags", timeout=3) is not None


def handle_get(path: str, query: str, deps: LocalAiRouteDependencies) -> JsonRouteResult | None:
    # All local-AI GET paths are handled inline in runtime.py with full
    # LocalAI manager access. Return None so the inline handlers run.
    return None

    params = parse_qs(query)  # Retained for the documented inline route reference.

    # --- /api/model-policy ---
    if path == "/api/model-policy":
        return JsonRouteResult({
            "mode": "cpu_only",
            "gpu_required": False,
            "ollama_backend": True,
            "ollama_url": deps.ollama_url,
            "description": "CPU-only mode via Ollama; no dedicated GPU required.",
        })

    # --- /api/settings/runtime ---
    if path == "/api/settings/runtime":
        return JsonRouteResult(deps._runtime_settings)

    # --- /api/local-ai/jobs ---
    if path == "/api/local-ai/jobs":
        project = params.get("project", [None])[0] or "default"
        status = params.get("status", [None])[0] or ""
        limit = int(params.get("limit", ["25"])[0])
        jobs = deps.local_ai.model_jobs(project=project, status=status, limit=limit) if deps.local_ai else []
        return JsonRouteResult({"jobs": jobs})

    # --- /api/local-ai/jobs/{id} ---
    if path.startswith("/api/local-ai/jobs/") and path != "/api/local-ai/jobs/cancel":
        job_id = path.removeprefix("/api/local-ai/jobs/").strip("/")
        if job_id and "/" not in job_id:
            job = deps.local_ai.model_job(job_id) if deps.local_ai else None
            if job is None:
                return JsonRouteResult({"error": "Job not found"}, HTTPStatus.NOT_FOUND)
            return JsonRouteResult({"job": job})

    # --- /api/local-ai/status ---
    if path == "/api/local-ai/status":
        result = deps.local_ai.status() if deps.local_ai else {}
        return JsonRouteResult(result)

    # --- /api/local-ai/setup/status ---
    if path == "/api/local-ai/setup/status":
        reachable = _ollama_reachable(deps.ollama_url)
        models = _ollama_tags(deps.ollama_url) if reachable else []
        return JsonRouteResult({
            "ollama_reachable": reachable,
            "ollama_url": deps.ollama_url,
            "models": models,
            "model_count": len(models),
            "status": "ready" if reachable else "ollama_unreachable",
        })

    # --- /api/local-ai/smooth-runtime ---
    if path == "/api/local-ai/smooth-runtime":
        task = params.get("task", [None])[0] or "chat"
        target_ram = params.get("target_ram_gb", [None])[0]
        quantization_mode = params.get("quantization_mode", [None])[0] or "auto"
        try:
            target_ram_gb = float(target_ram) if target_ram else None
        except (TypeError, ValueError):
            return JsonRouteResult({"error": "target_ram_gb must be a number"}, HTTPStatus.BAD_REQUEST)
        if deps.local_ai:
            result = deps.local_ai.smooth_runtime_plan(
                task=task, target_ram_gb=target_ram_gb, quantization_mode=quantization_mode
            )
        else:
            result = {"status": "healthy", "task": task, "backend": "ollama", "mode": "cpu"}
        return JsonRouteResult(result)

    # --- /api/local-ai/models ---
    if path == "/api/local-ai/models":
        if deps.local_ai:
            models = deps.local_ai.models()
        else:
            models = _ollama_tags(deps.ollama_url)
        return JsonRouteResult({"models": models})

    # --- /api/local-ai/benchmarks ---
    if path == "/api/local-ai/benchmarks":
        model_id = params.get("model_id", [None])[0] or ""
        limit = int(params.get("limit", ["50"])[0])
        if deps.local_ai:
            benchmarks = deps.local_ai.benchmark_history(model_id=model_id, limit=limit)
        else:
            benchmarks = []
        return JsonRouteResult({"benchmarks": benchmarks})

    # --- /api/local-ai/models/{id}/probe ---
    if path.startswith("/api/local-ai/models/") and path.endswith("/probe"):
        model_id = path.removeprefix("/api/local-ai/models/").removesuffix("/probe")
        tags = _ollama_tags(deps.ollama_url)
        found = next((m for m in tags if m.get("name") == model_id or m.get("model") == model_id), None)
        return JsonRouteResult({
            "model_id": model_id,
            "available": found is not None,
            "details": found,
        })

    # --- /api/local-ai/hardware ---
    if path == "/api/local-ai/hardware":
        from ..adaptive_engine.hardware_detection import get_hardware_capabilities
        caps = get_hardware_capabilities()
        return JsonRouteResult(caps.summary_dict())

    # --- /api/local-ai/adapters ---
    if path == "/api/local-ai/adapters":
        from ..adaptive_engine.gpu_adapters import build_adapter_registry
        registry = build_adapter_registry()
        adapters = [
            {
                "name": a.name,
                "device_name": a.device_name,
                "available": a.is_available(),
            }
            for a in registry.all_available()
        ]
        return JsonRouteResult({"adapters": adapters})

    # --- /api/local-ai/marketplace ---
    if path == "/api/local-ai/marketplace":
        installed = {m.get("name") or m.get("model") for m in _ollama_tags(deps.ollama_url)}
        marketplace = [
            {**entry, "installed": entry["model_id"] in installed}
            for entry in _CURATED_MARKETPLACE
        ]
        return JsonRouteResult({"marketplace": marketplace})

    return None


def handle_post(path: str, deps: LocalAiRouteDependencies) -> JsonRouteResult | None:
    # All local-AI POST paths are handled inline in runtime.py.
    return None

    # --- /api/settings/runtime ---
    if path == "/api/settings/runtime":  # Retained for the documented inline route reference.
        payload = deps.read_json()
        settings = deps._runtime_settings
        if "cpu_threads" in payload:
            settings["cpu_threads"] = int(payload["cpu_threads"])
        if "memory_limit_mb" in payload:
            settings["memory_limit_mb"] = int(payload["memory_limit_mb"])
        if "ollama_url" in payload:
            settings["ollama_url"] = str(payload["ollama_url"]).strip()
        return JsonRouteResult({"settings": settings})

    # --- /api/models/discover ---
    if path == "/api/models/discover":
        tags = _ollama_tags(deps.ollama_url)
        return JsonRouteResult({"discovered": tags, "source": "ollama", "count": len(tags)})

    # --- /api/local-ai/jobs (create job) ---
    if path == "/api/local-ai/jobs":
        payload = deps.read_json()
        model_id = str(payload.get("model_id", "")).strip()
        if not model_id:
            return JsonRouteResult({"error": "model_id is required"}, HTTPStatus.BAD_REQUEST)
        if deps.local_ai:
            job = deps.local_ai.enqueue_model_job(model_id, payload)
            return JsonRouteResult({"job": job}, HTTPStatus.CREATED)
        return JsonRouteResult({"error": "Download manager unavailable"}, HTTPStatus.SERVICE_UNAVAILABLE)

    # --- /api/local-ai/jobs/cancel ---
    if path == "/api/local-ai/jobs/cancel":
        payload = deps.read_json()
        job_id = str(payload.get("job_id", "")).strip()
        if not job_id:
            return JsonRouteResult({"error": "job_id is required"}, HTTPStatus.BAD_REQUEST)
        if deps.local_ai:
            deps.local_ai.cancel_model_job(job_id)
        return JsonRouteResult({"cancelled": True, "job_id": job_id})

    # --- /api/local-ai/models/register ---
    if path == "/api/local-ai/models/register":
        payload = deps.read_json()
        model_id = str(payload.get("model_id", "")).strip()
        if not model_id:
            return JsonRouteResult({"error": "model_id is required"}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"registered": True, "model_id": model_id}, HTTPStatus.CREATED)

    # --- /api/local-ai/models/download ---
    if path == "/api/local-ai/models/download":
        payload = deps.read_json()
        model_id = str(payload.get("model_id", "")).strip()
        if not model_id:
            return JsonRouteResult({"error": "model_id is required"}, HTTPStatus.BAD_REQUEST)
        # Delegate to Ollama pull for Ollama-backed models
        return JsonRouteResult({"enqueued": True, "model_id": model_id, "backend": "ollama"}, HTTPStatus.ACCEPTED)

    # --- /api/local-ai/models/setup/job ---
    if path == "/api/local-ai/models/setup/job":
        payload = deps.read_json()
        model_id = str(payload.get("model_id", "")).strip()
        return JsonRouteResult({"job_id": f"setup-{model_id}", "model_id": model_id, "status": "pending"}, HTTPStatus.ACCEPTED)

    # --- /api/local-ai/llamacpp/register ---
    if path == "/api/local-ai/llamacpp/register":
        payload = deps.read_json()
        model_path = str(payload.get("model_path", "")).strip()
        model_id = str(payload.get("model_id", "")).strip() or model_path
        return JsonRouteResult({"registered": True, "model_id": model_id, "backend": "llamacpp"}, HTTPStatus.CREATED)

    # --- /api/local-ai/models/auto-quantize ---
    if path == "/api/local-ai/models/auto-quantize":
        payload = deps.read_json()
        model_id = str(payload.get("model_id", "")).strip()
        return JsonRouteResult({
            "model_id": model_id,
            "note": "CPU-only mode; quantization handled by Ollama backend.",
            "supported": False,
        })

    # --- /api/local-ai/models/delete ---
    if path == "/api/local-ai/models/delete":
        payload = deps.read_json()
        model_id = str(payload.get("model_id", "")).strip()
        if not model_id:
            return JsonRouteResult({"error": "model_id is required"}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"deleted": True, "model_id": model_id})

    # --- /api/local-ai/models/{id}/dependencies ---
    if path.startswith("/api/local-ai/models/") and path.endswith("/dependencies"):
        model_id = path.removeprefix("/api/local-ai/models/").removesuffix("/dependencies")
        return JsonRouteResult({"model_id": model_id, "dependencies": [], "all_satisfied": True})

    # --- /api/local-ai/models/{id}/tools ---
    if path.startswith("/api/local-ai/models/") and path.endswith("/tools"):
        model_id = path.removeprefix("/api/local-ai/models/").removesuffix("/tools")
        if deps.local_ai:
            tools = deps.local_ai.model_tools(model_id)
        else:
            tools = []
        return JsonRouteResult({"model_id": model_id, "tools": tools})

    # --- /api/local-ai/benchmark ---
    if path == "/api/local-ai/benchmark":
        payload = deps.read_json()
        model_id = str(payload.get("model_id", "phi4-mini:latest")).strip()
        prompt = str(payload.get("prompt", "Hello, how are you?")).strip()
        import time
        t0 = time.time()
        try:
            req_body = json.dumps({"model": model_id, "prompt": prompt, "stream": False}).encode()
            req = urllib.request.Request(
                f"{deps.ollama_url}/api/generate",
                data=req_body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            elapsed = time.time() - t0
            return JsonRouteResult({
                "model_id": model_id,
                "latency_ms": round(elapsed * 1000, 1),
                "eval_count": data.get("eval_count", 0),
                "tokens_per_second": data.get("eval_count", 0) / max(elapsed, 0.001),
                "backend": "ollama",
            })
        except Exception as exc:
            return JsonRouteResult({"error": str(exc), "model_id": model_id}, HTTPStatus.BAD_GATEWAY)

    # --- /api/local-ai/cluster-plan ---
    if path == "/api/local-ai/cluster-plan":
        payload = deps.read_json()
        return JsonRouteResult({
            "plan": "cpu_only",
            "nodes": [{"type": "cpu", "threads": 0, "role": "primary"}],
            "gpu_required": False,
            "note": "CPU-only cluster; single-node Ollama backend.",
        })

    # --- /api/local-ai/models/quantize ---
    if path == "/api/local-ai/models/quantize":
        payload = deps.read_json()
        model_id = str(payload.get("model_id", "")).strip()
        return JsonRouteResult({
            "model_id": model_id,
            "note": "Quantization delegated to Ollama backend.",
            "status": "unsupported",
        })

    # --- /api/local-ai/models/setup-chat ---
    if path == "/api/local-ai/models/setup-chat":
        payload = deps.read_json()
        model_id = str(payload.get("model_id", "")).strip()
        if not model_id:
            return JsonRouteResult({"error": "model_id is required"}, HTTPStatus.BAD_REQUEST)
        tags = _ollama_tags(deps.ollama_url)
        available = any(
            m.get("name") == model_id or m.get("model") == model_id for m in tags
        )
        return JsonRouteResult({
            "model_id": model_id,
            "chat_ready": available,
            "backend": "ollama",
            "ollama_url": deps.ollama_url,
        })

    # --- /api/local-ai/ollama/pull ---
    if path == "/api/local-ai/ollama/pull":
        payload = deps.read_json()
        model_name = str(payload.get("model", payload.get("model_id", ""))).strip()
        if not model_name:
            return JsonRouteResult({"error": "model is required"}, HTTPStatus.BAD_REQUEST)
        try:
            req_body = json.dumps({"model": model_name, "stream": False}).encode()
            req = urllib.request.Request(
                f"{deps.ollama_url}/api/pull",
                data=req_body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            return JsonRouteResult({"model": model_name, "status": data.get("status", "done")}, HTTPStatus.ACCEPTED)
        except Exception as exc:
            return JsonRouteResult({"error": str(exc), "model": model_name}, HTTPStatus.BAD_GATEWAY)

    return None
