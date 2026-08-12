"""Adaptive CPU inference planning, prompt compaction, caching, and telemetry."""

from __future__ import annotations

from collections import OrderedDict, deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import os
import re
import threading
import time
from typing import Any, Iterator

from .core_runtime import model_request_options, smooth_runtime_profile


_ACTIVE_OPTIONS: ContextVar[dict[str, Any] | None] = ContextVar(
    "openagent_inference_options", default=None
)


@dataclass(frozen=True)
class InferencePlan:
    messages: list[dict[str, str]]
    task: str
    cache_key: str
    options: dict[str, Any]
    original_tokens: int
    optimized_tokens: int
    compacted_messages: int
    target_first_token_ms: int

    def metadata(self) -> dict[str, Any]:
        return {
            "engine": "openagent-adaptive-cpu-v1",
            "task": self.task,
            "original_prompt_tokens_estimate": self.original_tokens,
            "optimized_prompt_tokens_estimate": self.optimized_tokens,
            "compacted_messages": self.compacted_messages,
            "target_first_token_ms": self.target_first_token_ms,
            "options": self.options,
            "techniques": [
                "adaptive-context-window",
                "intent-output-budget",
                "physical-core-thread-budget",
                "warm-session-keepalive",
                "exact-response-cache",
                "value-preserving-embedding-cache",
                "prefix-preserving-compaction",
                "stream-first-delivery",
            ],
        }


@dataclass(frozen=True)
class CachedCompletion:
    provider: str
    model: str
    content: str
    metadata: dict[str, Any]
    created_monotonic: float


@dataclass(frozen=True)
class CachedEmbedding:
    provider: str
    model: str
    embeddings: tuple[tuple[float, ...], ...]
    dimensions: int
    metadata: dict[str, Any]
    created_monotonic: float


class AdaptiveCPUInferenceEngine:
    """Optimize interactive CPU requests without changing answer semantics."""

    def __init__(self) -> None:
        self.max_entries = _env_int("OPENAGENT_INFERENCE_CACHE_ENTRIES", 128, 0, 2048)
        self.ttl_seconds = _env_int("OPENAGENT_INFERENCE_CACHE_TTL_SECONDS", 900, 1, 86400)
        self.target_first_token_ms = _env_int("OPENAGENT_FIRST_TOKEN_TARGET_MS", 1000, 100, 10000)
        self._cache: OrderedDict[str, CachedCompletion] = OrderedDict()
        self._embedding_cache: OrderedDict[str, CachedEmbedding] = OrderedDict()
        self._events: deque[dict[str, Any]] = deque(maxlen=500)
        self._lock = threading.RLock()
        self._plans = 0
        self._hits = 0
        self._misses = 0
        self._embedding_hits = 0
        self._embedding_misses = 0

    def plan(
        self,
        messages: list[dict[str, str]],
        *,
        provider_id: str,
        model: str | None,
        allow_external: bool,
        max_output_tokens: int | None = None,
    ) -> InferencePlan:
        cleaned = [
            {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
            for item in messages
            if str(item.get("content", "")).strip()
        ]
        task = infer_task(cleaned)
        profile = smooth_runtime_profile(task=task)
        if max_output_tokens is None:
            output_tokens = adaptive_output_tokens(cleaned, task=task, maximum=profile.output_tokens)
        else:
            output_tokens = max(32, min(int(max_output_tokens), profile.context_tokens // 2))
        context_limit = max(512, profile.context_tokens - output_tokens - 96)
        optimized, compacted = compact_messages(cleaned, token_budget=context_limit)
        original_tokens = estimate_messages_tokens(cleaned)
        optimized_tokens = estimate_messages_tokens(optimized)
        options = model_request_options(
            task=task,
            prompt_tokens=optimized_tokens,
            output_tokens=output_tokens,
            low_latency=True,
        )
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "provider": provider_id,
                    "model": model or "",
                    "external": allow_external,
                    "messages": cleaned,
                    "runtime": {
                        "task": task,
                        "num_ctx": options["num_ctx"],
                        "num_predict": options["num_predict"],
                    },
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        with self._lock:
            self._plans += 1
        return InferencePlan(
            messages=optimized,
            task=task,
            cache_key=cache_key,
            options=options,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            compacted_messages=compacted,
            target_first_token_ms=self.target_first_token_ms,
        )

    @contextmanager
    def activate(self, plan: InferencePlan) -> Iterator[None]:
        token = _ACTIVE_OPTIONS.set(dict(plan.options))
        try:
            yield
        finally:
            _ACTIVE_OPTIONS.reset(token)

    def get(self, cache_key: str) -> CachedCompletion | None:
        if self.max_entries <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is None:
                self._misses += 1
                return None
            if now - cached.created_monotonic > self.ttl_seconds:
                self._cache.pop(cache_key, None)
                self._misses += 1
                return None
            self._cache.move_to_end(cache_key)
            self._hits += 1
            return cached

    def put(
        self,
        cache_key: str,
        *,
        provider: str,
        model: str,
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        if self.max_entries <= 0 or not content:
            return
        with self._lock:
            self._cache[cache_key] = CachedCompletion(
                provider=provider,
                model=model,
                content=content,
                metadata=dict(metadata),
                created_monotonic=time.monotonic(),
            )
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)

    def embedding_key(
        self,
        texts: list[str],
        *,
        provider_id: str,
        model: str | None,
        allow_external: bool,
    ) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "operation": "embedding",
                    "provider": provider_id,
                    "model": model or "",
                    "external": allow_external,
                    "texts": texts,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    def get_embedding(self, cache_key: str) -> CachedEmbedding | None:
        if self.max_entries <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            cached = self._embedding_cache.get(cache_key)
            if cached is None:
                self._embedding_misses += 1
                return None
            if now - cached.created_monotonic > self.ttl_seconds:
                self._embedding_cache.pop(cache_key, None)
                self._embedding_misses += 1
                return None
            self._embedding_cache.move_to_end(cache_key)
            self._embedding_hits += 1
            return cached

    def put_embedding(
        self,
        cache_key: str,
        *,
        provider: str,
        model: str,
        embeddings: list[list[float]],
        dimensions: int,
        metadata: dict[str, Any],
    ) -> None:
        if self.max_entries <= 0 or not embeddings:
            return
        frozen = tuple(tuple(float(value) for value in vector) for vector in embeddings)
        with self._lock:
            self._embedding_cache[cache_key] = CachedEmbedding(
                provider=provider,
                model=model,
                embeddings=frozen,
                dimensions=int(dimensions),
                metadata=dict(metadata),
                created_monotonic=time.monotonic(),
            )
            self._embedding_cache.move_to_end(cache_key)
            while len(self._embedding_cache) > self.max_entries:
                self._embedding_cache.popitem(last=False)

    def record(
        self,
        plan: InferencePlan,
        *,
        provider: str,
        model: str,
        latency_ms: float,
        first_token_ms: float | None,
        cache_hit: bool,
    ) -> None:
        event = {
            "task": plan.task,
            "provider": provider,
            "model": model,
            "latency_ms": round(latency_ms, 3),
            "first_token_ms": round(first_token_ms, 3) if first_token_ms is not None else None,
            "cache_hit": cache_hit,
            "prompt_tokens": plan.optimized_tokens,
            "saved_prompt_tokens": max(0, plan.original_tokens - plan.optimized_tokens),
            "at": time.time(),
        }
        with self._lock:
            self._events.append(event)

    def status(self) -> dict[str, Any]:
        with self._lock:
            events = list(self._events)
            latencies = sorted(float(event["latency_ms"]) for event in events)
            first_tokens = sorted(
                float(event["first_token_ms"])
                for event in events
                if event.get("first_token_ms") is not None
            )
            return {
                "status": "ready",
                "engine": "openagent-adaptive-cpu-v1",
                "plans": self._plans,
                "cache": {
                    "entries": len(self._cache),
                    "max_entries": self.max_entries,
                    "ttl_seconds": self.ttl_seconds,
                    "hits": self._hits,
                    "misses": self._misses,
                    "hit_rate": round(self._hits / max(1, self._hits + self._misses), 6),
                },
                "embedding_cache": {
                    "entries": len(self._embedding_cache),
                    "hits": self._embedding_hits,
                    "misses": self._embedding_misses,
                    "hit_rate": round(
                        self._embedding_hits / max(1, self._embedding_hits + self._embedding_misses),
                        6,
                    ),
                },
                "latency": {
                    "samples": len(events),
                    "p50_ms": percentile(latencies, 0.50),
                    "p95_ms": percentile(latencies, 0.95),
                    "first_token_p50_ms": percentile(first_tokens, 0.50),
                    "first_token_p95_ms": percentile(first_tokens, 0.95),
                    "target_first_token_ms": self.target_first_token_ms,
                },
                "last_events": events[-10:],
                "privacy_boundary": "Prompts, responses, and cached vectors remain in process memory and are never sent to telemetry services.",
            }

    def clear(self) -> dict[str, Any]:
        with self._lock:
            completion_removed = len(self._cache)
            embedding_removed = len(self._embedding_cache)
            self._cache.clear()
            self._embedding_cache.clear()
        return {
            "cleared": completion_removed + embedding_removed,
            "completion_entries": completion_removed,
            "embedding_entries": embedding_removed,
            "remaining": 0,
        }

    def benchmark(self, *, iterations: int = 100) -> dict[str, Any]:
        iterations = max(1, min(int(iterations), 10000))
        messages = [
            {"role": "system", "content": "Answer concisely and stay local."},
            {"role": "user", "content": "Explain how OpenAgent keeps retrieval private."},
        ]
        planning = []
        plan: InferencePlan | None = None
        for _ in range(iterations):
            started = time.perf_counter()
            plan = self.plan(messages, provider_id="auto", model=None, allow_external=False)
            planning.append((time.perf_counter() - started) * 1000)
        assert plan is not None
        self.put(
            plan.cache_key,
            provider="benchmark",
            model="benchmark",
            content="cached benchmark response",
            metadata={},
        )
        cache_started = time.perf_counter()
        for _ in range(iterations):
            self.get(plan.cache_key)
        cache_total = (time.perf_counter() - cache_started) * 1000
        return {
            "status": "ok",
            "iterations": iterations,
            "planning_ms": {
                "min": round(min(planning), 6),
                "mean": round(sum(planning) / len(planning), 6),
                "max": round(max(planning), 6),
            },
            "cache_lookup_mean_ms": round(cache_total / iterations, 6),
            "plan": plan.metadata(),
            "scope": "Engine overhead only; model token generation is hardware/model dependent.",
        }


def active_runtime_options() -> dict[str, Any] | None:
    options = _ACTIVE_OPTIONS.get()
    return dict(options) if options is not None else None


def infer_task(messages: list[dict[str, str]]) -> str:
    user = next(
        (item["content"].lower() for item in reversed(messages) if item["role"] == "user"),
        "",
    )
    if any(term in user for term in ("code", "function", "class ", "debug", "python", "javascript", "sql", "error")):
        return "code"
    if any(term in user for term in ("research", "compare", "evidence", "sources", "analyze", "investigate")):
        return "research"
    if re.search(r"\bagents?\b|\bworkflow\b|\btool calls?\b|\bexecute\b", user):
        return "agent"
    return "chat"


def compact_messages(
    messages: list[dict[str, str]], *, token_budget: int
) -> tuple[list[dict[str, str]], int]:
    if estimate_messages_tokens(messages) <= token_budget:
        return messages, 0
    systems = [item for item in messages if item["role"] == "system"]
    dialogue = [item for item in messages if item["role"] != "system"]
    system_budget = max(192, token_budget // 3)
    if estimate_messages_tokens(systems) > system_budget:
        joined = "\n\n".join(item["content"] for item in systems)
        systems = [{"role": "system", "content": truncate_middle_tokens(joined, system_budget)}]
    reserved = estimate_messages_tokens(systems)
    available = max(128, token_budget - reserved)
    kept_reversed: list[dict[str, str]] = []
    used = 0
    omitted: list[dict[str, str]] = []
    for item in reversed(dialogue):
        cost = estimate_tokens(item["content"]) + 4
        if not kept_reversed and cost > available:
            kept_reversed.append({
                "role": item["role"],
                "content": truncate_middle_tokens(item["content"], max(32, available - 4)),
            })
            used = available
            continue
        if kept_reversed and used + cost > available:
            omitted.append(item)
            continue
        if not kept_reversed or used + cost <= available:
            kept_reversed.append(item)
            used += cost
        else:
            omitted.append(item)
    kept = list(reversed(kept_reversed))
    if omitted:
        digest_budget = max(48, min(256, available // 4))
        digest_parts = []
        for item in reversed(omitted):
            excerpt = " ".join(item["content"].split())[:240]
            digest_parts.append(f"{item['role']}: {excerpt}")
        digest = truncate_tokens(" | ".join(digest_parts), digest_budget)
        systems = [
            *systems,
            {
                "role": "system",
                "content": f"Earlier conversation digest (lossy; prefer recent messages): {digest}",
            },
        ]
    return [*systems, *kept], len(omitted)


def truncate_middle_tokens(text: str, token_budget: int) -> str:
    max_chars = max(32, token_budget * 4)
    if len(text) <= max_chars:
        return text
    marker = "\n...[middle context compacted]...\n"
    remaining = max(16, max_chars - len(marker))
    head = remaining // 2
    tail = remaining - head
    return text[:head].rstrip() + marker + text[-tail:].lstrip()


def adaptive_output_tokens(
    messages: list[dict[str, str]], *, task: str, maximum: int
) -> int:
    user = next(
        (item["content"] for item in reversed(messages) if item["role"] == "user"), ""
    )
    user_tokens = estimate_tokens(user)
    concise_request = any(
        phrase in user.lower()
        for phrase in ("one sentence", "short sentence", "briefly", "be concise", "short answer")
    )
    if concise_request and task == "chat":
        desired = 48
    elif user_tokens <= 4 and task == "chat":
        desired = 64
    elif user_tokens <= 24 and task == "chat":
        desired = 128
    elif task == "code":
        desired = 384
    elif task == "research":
        desired = 256
    else:
        desired = 192
    override = os.environ.get(f"OPENAGENT_{task.upper()}_MAX_OUTPUT_TOKENS", "").strip()
    if override:
        try:
            desired = int(override)
        except ValueError:
            pass
    return max(32, min(int(maximum), desired))


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    return sum(estimate_tokens(item.get("content", "")) + 4 for item in messages) + 3


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def truncate_tokens(text: str, token_budget: int) -> str:
    max_chars = max(16, token_budget * 4)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rsplit(' ', 1)[0].strip()}..."


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    index = min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))
    return round(values[index], 3)


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))
