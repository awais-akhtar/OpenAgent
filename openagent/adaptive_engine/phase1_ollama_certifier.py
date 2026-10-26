"""Deprecated Ollama probe entry point; Ollama probes cannot certify Phase 1.

Phase 1 evidence must be captured through the exact admitted OpenAgent route,
including its pinned checkpoint, loader, backend, cache, ABI, recovery and
placement observations. A generic request to an Ollama server proves none of
those properties, even when the model name resembles a family.
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPTS_PATH = REPOSITORY_ROOT / "phase1_ollama_receipts.json"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Informational name hints only; never used to admit or certify a route.
OLLAMA_FAMILY_MAP: dict[str, str] = {
    "qwen3.5": "qwen35",
    "qwen3-coder": "qwen4",
    "phi4-mini": "inkling",
    "gpt-oss": "olmoe",
    "qwen2.5vl": "qwen35",
}

PROBE_PROMPT = "The capital of France is"
PROBE_MAX_TOKENS = 8

def _ollama_request(url: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    """POST JSON to Ollama and return parsed response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_ollama_connectivity(ollama_url: str = DEFAULT_OLLAMA_URL) -> list[str]:
    """Return list of available Ollama model names, or raise on failure."""
    req = urllib.request.Request(f"{ollama_url}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach Ollama at {ollama_url}: {exc}") from exc
    models = data.get("models") or []
    return [m["name"] for m in models if isinstance(m, dict) and "name" in m]


def map_ollama_models_to_families(model_names: list[str]) -> dict[str, str]:
    """Return {ollama_model_name: phase1_family} for recognizable models."""
    result: dict[str, str] = {}
    for name in model_names:
        name_lower = name.lower()
        for pattern, family in OLLAMA_FAMILY_MAP.items():
            if pattern in name_lower:
                result[name] = family
                break
    return result


def _probe_model(model_name: str, ollama_url: str) -> dict[str, Any]:
    """Run a single generate probe; return timing + response info."""
    payload = {
        "model": model_name,
        "prompt": PROBE_PROMPT,
        "stream": False,
        "options": {"num_predict": PROBE_MAX_TOKENS},
    }
    t0 = time.monotonic()
    response = _ollama_request(f"{ollama_url}/api/generate", payload, timeout=180)
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    resp_text: str = response.get("response", "")
    eval_count: int = response.get("eval_count", 0) or 0
    eval_duration_ns: int = response.get("eval_duration", 0) or 0
    tokens_per_second = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0.0

    return {
        "latency_ms": round(elapsed_ms, 1),
        "tokens": eval_count,
        "tokens_per_second": round(tokens_per_second, 2),
        "response_preview": resp_text[:80].strip(),
    }


def certify_bindings_with_ollama(
    ollama_url: str = DEFAULT_OLLAMA_URL,
    receipts_path: Path | None = None,
) -> dict[str, int]:
    """Reject the legacy API because a generic Ollama probe is not certification.

    The parameters remain for source compatibility. Nothing is read, probed,
    or written: use ``phase1_capture`` with the exact admitted checkpoint route.
    """
    del ollama_url, receipts_path
    raise RuntimeError(
        "Ollama smoke probes cannot certify Phase 1 routes; capture evidence "
        "through the exact admitted OpenAgent route with phase1_capture"
    )


__all__ = [
    "check_ollama_connectivity",
    "map_ollama_models_to_families",
    "certify_bindings_with_ollama",
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_RECEIPTS_PATH",
    "OLLAMA_FAMILY_MAP",
]
