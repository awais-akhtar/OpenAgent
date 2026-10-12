"""Local, workload-specific CPU BLAS tuning; never a global optimum claim.

The caller must own an idle model and ensure there are no other BLAS users
in the process for the entire call. A lock only serializes this module's calls.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import tempfile
import threading
from time import perf_counter

import numpy as np

from .cpu_threads import blas_thread_scope

_LOCK = threading.Lock()


def _integer(value, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError("budgets and thread counts must be integers in range")


def _configuration(prompt_ids, candidates, max_tokens, warmup, runs, vocab, context):
    for value, minimum in ((max_tokens, 1), (warmup, 0), (runs, 1), (vocab, 1), (context, 1)):
        _integer(value, minimum)
    if not isinstance(candidates, (tuple, list)) or not candidates:
        raise ValueError("candidates must be a non-empty sequence")
    for value in candidates:
        _integer(value)
    if len(set(candidates)) != len(candidates):
        raise ValueError("duplicate candidates")
    if not isinstance(prompt_ids, (list, tuple, np.ndarray)):
        raise ValueError("prompt_ids must be a non-empty integer vector")
    ids = np.asarray(prompt_ids)
    if ids.ndim != 1 or not ids.size or ids.dtype.kind not in "iu":
        raise ValueError("prompt_ids must be a non-empty integer vector")
    if np.any(ids < 0) or np.any(ids >= vocab):
        raise ValueError("prompt token outside vocabulary")
    if len(ids) + max_tokens > context:
        raise ValueError("generation exceeds session context")
    return dict(prompt_ids=[int(x) for x in ids], candidates=list(candidates),
                max_tokens=max_tokens, warmup=warmup, runs=runs,
                vocab_size=vocab, context_tokens=context)


def _environment(model):
    try:
        from threadpoolctl import __version__, threadpool_info
    except ImportError as exc:
        raise RuntimeError("BLAS controls require the engine extra (threadpoolctl)") from exc

    fingerprint = model.manifest.fingerprint
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("manifest fingerprint is required")
    blas = [{key: str(pool.get(key) or "") for key in
             ("internal_api", "prefix", "version", "architecture", "threading_layer")}
            for pool in threadpool_info() if pool.get("user_api") == "blas"]
    if not blas:
        raise RuntimeError("no CPU BLAS implementation detected")
    return dict(manifest_fingerprint=fingerprint, cpu_architecture=platform.machine(),
                cpu_processor=platform.processor(), numpy_version=np.__version__,
                threadpoolctl_version=__version__,
                blas=sorted(blas, key=lambda item: json.dumps(item, sort_keys=True)))


def _key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _validate_runtime(runtime):
    if type(runtime) is not dict or set(runtime) != {
            "preferred_kernel", "selected_backends", "io_mode", "capacity_bytes",
            "prefetch_workers", "prefetch_depth"}:
        raise ValueError("invalid runtime identity fields")
    preferred = runtime["preferred_kernel"]
    if preferred is not None and (type(preferred) is not str or not preferred):
        raise ValueError("invalid preferred kernel")
    if runtime["io_mode"] not in ("buffered", "mmap", "direct", "uring"):
        raise ValueError("invalid runtime io_mode")
    for name in ("capacity_bytes", "prefetch_workers", "prefetch_depth"):
        _integer(runtime[name])
    if runtime["prefetch_depth"] < runtime["prefetch_workers"]:
        raise ValueError("invalid runtime prefetch configuration")
    backends = runtime["selected_backends"]
    if type(backends) is not list or not backends:
        raise ValueError("missing selected expert backend")
    layers = set()
    for backend in backends:
        if type(backend) is not dict or set(backend) != {
                "layer_id", "dtype", "name", "provider", "compiled", "native_simd"}:
            raise ValueError("invalid selected backend fields")
        for name in ("layer_id", "dtype", "name", "provider"):
            if type(backend[name]) is not str or not backend[name]:
                raise ValueError("invalid selected backend identity")
        if backend["layer_id"] in layers:
            raise ValueError("duplicate runtime layer")
        layers.add(backend["layer_id"])
        if any(type(backend[name]) is not bool for name in ("compiled", "native_simd")):
            raise ValueError("invalid backend capabilities")


def _runtime_identity(model):
    """Use reported store configuration and the actual batch kernel resolver."""
    from .expert_store import DEFAULT_EXPERT_KERNELS

    try:
        experts = model.experts
        status = experts.status()
        store = status["store"]
        if store["closed"] is not False:
            raise ValueError("expert store must be open")
        runtime = {name: store[name] for name in
                   ("io_mode", "capacity_bytes", "prefetch_workers", "prefetch_depth")}
        runtime["preferred_kernel"] = status["preferred_kernel"]
        registry = experts.kernels if experts.kernels is not None else DEFAULT_EXPERT_KERNELS
        selected = []
        for layer in model.manifest.layers:
            info = registry.resolve(dtype=layer.dtype, batch=True,
                                    preferred=runtime["preferred_kernel"]).info()
            # The status description must corroborate the resolved implementation.
            reported = next(kernel for kernel in status["kernels"] if kernel["name"] == info.name)
            if reported["available"] is not True:
                raise ValueError("selected expert backend is unavailable")
            selected.append(dict(layer_id=layer.id, dtype=layer.dtype,
                                 **{name: reported[name] for name in
                                    ("name", "provider", "compiled", "native_simd")}))
        runtime["selected_backends"] = selected
        _validate_runtime(runtime)
        return runtime
    except (AttributeError, KeyError, TypeError, StopIteration) as exc:
        raise ValueError("expert runtime lacks required identity capability") from exc


def tune_olmoe_threads(model, prompt_ids, candidates=(1, 2, 4), max_tokens=2,
                       warmup=1, runs=2, *, no_active_external_users=False):
    """Measure greedy generation, returning a JSON-compatible local profile.

    Reject output-changing candidates, including changes during warmup. The
    first candidate's first run establishes the target IDs. Reset KV state
    before and after every run, even when inference or thread control fails.
    """
    if no_active_external_users is not True:
        raise ValueError("requires no_active_external_users=True for process-wide BLAS limits")
    config = _configuration(prompt_ids, candidates, max_tokens, warmup, runs,
                            model.config.vocab_size, model.context_tokens)
    if type(model.position) is not int or model.position != 0:
        raise ValueError("model session must be idle with empty context")
    with _LOCK:
        if type(model.position) is not int or model.position != 0:
            raise ValueError("model session must be idle with empty context")
        identity = dict(environment=_environment(model), configuration=config,
                        runtime=_runtime_identity(model))
        results = []
        baseline = None
        try:
            for threads in candidates:
                measurements = []
                accepted = True
                with blas_thread_scope(threads):
                    for index in range(warmup + runs):
                        model.reset()
                        stream = None
                        try:
                            started = perf_counter()
                            stream = model.generate_id_stream(config["prompt_ids"],
                                                              max_tokens=max_tokens, temperature=0.0)
                            try:
                                ids = [int(next(stream))]
                            except StopIteration as exc:
                                raise ValueError("prompt generated no tokens; TTFT cannot be measured") from exc
                            ttft = perf_counter() - started
                            ids.extend(int(token) for token in stream)
                            elapsed = perf_counter() - started
                            if not ids or len(ids) > max_tokens or any(
                                    token < 0 or token >= model.config.vocab_size for token in ids):
                                raise ValueError("invalid generated token IDs")
                            if elapsed <= 0 or ttft < 0 or elapsed < ttft:
                                raise RuntimeError("invalid perf_counter measurement")
                            if baseline is None:
                                baseline = ids
                            accepted = accepted and ids == baseline
                            if index >= warmup:
                                measurements.append(dict(token_ids=ids, ttft_seconds=ttft,
                                                         elapsed_seconds=elapsed,
                                                         tokens_per_second=len(ids) / elapsed))
                        finally:
                            try:
                                if stream is not None and hasattr(stream, "close"):
                                    stream.close()
                            finally:
                                model.reset()
                results.append(dict(threads=threads, accepted=accepted, measurements=measurements))
        finally:
            model.reset()
        if _runtime_identity(model) != identity["runtime"]:
            raise RuntimeError("expert runtime configuration changed during tuning")
        valid = [item for item in results if item["accepted"]]
        if not valid:
            raise RuntimeError("no candidate consistently matched baseline token IDs")
        selected = max(valid, key=lambda item: statistics.median(
            run["tokens_per_second"] for run in item["measurements"]))["threads"]
        return dict(schema_version=2, scope="measured-workload-only", identity=identity,
                    cache_key=_key(identity), baseline_token_ids=baseline,
                    results=results, recommended_threads=selected)


def _validate(profile):
    """Validate structure and derived evidence before accepting persisted data."""
    def fields(value, names):
        if type(value) is not dict or set(value) != set(names.split()):
            raise ValueError("invalid profile fields")

    fields(profile, "schema_version scope identity cache_key baseline_token_ids results recommended_threads")
    if type(profile["schema_version"]) is not int or profile["schema_version"] != 2 or profile["scope"] != "measured-workload-only":
        raise ValueError("unsupported profile")
    identity = profile["identity"]
    fields(identity, "environment configuration runtime")
    _validate_runtime(identity["runtime"])
    env = identity["environment"]
    fields(env, "manifest_fingerprint cpu_architecture cpu_processor numpy_version threadpoolctl_version blas")
    for key in set(env) - {"blas"}:
        if type(env[key]) is not str or (key != "cpu_processor" and not env[key]):
            raise ValueError("invalid environment identity")
    if type(env["blas"]) is not list or not env["blas"]:
        raise ValueError("invalid BLAS identity")
    for pool in env["blas"]:
        fields(pool, "internal_api prefix version architecture threading_layer")
        if any(type(value) is not str for value in pool.values()) or not pool["internal_api"]:
            raise ValueError("invalid BLAS identity")
    config = identity["configuration"]
    fields(config, "prompt_ids candidates max_tokens warmup runs vocab_size context_tokens")
    checked = _configuration(config["prompt_ids"], config["candidates"], config["max_tokens"],
                             config["warmup"], config["runs"], config["vocab_size"], config["context_tokens"])
    if checked != config or profile["cache_key"] != _key(identity):
        raise ValueError("profile identity mismatch")
    def tokens(ids):
        if type(ids) is not list or not 1 <= len(ids) <= config["max_tokens"] or any(
                type(x) is not int or not 0 <= x < config["vocab_size"] for x in ids):
            raise ValueError("invalid profile tokens")
    tokens(profile["baseline_token_ids"])
    results = profile["results"]
    if type(results) is not list or len(results) != len(config["candidates"]):
        raise ValueError("invalid candidate results")
    for candidate, result in zip(config["candidates"], results):
        fields(result, "threads accepted measurements")
        if type(result["threads"]) is not int or result["threads"] != candidate or type(result["accepted"]) is not bool:
            raise ValueError("invalid candidate result")
        if type(result["measurements"]) is not list or len(result["measurements"]) != config["runs"]:
            raise ValueError("invalid run count")
        for run in result["measurements"]:
            fields(run, "token_ids ttft_seconds elapsed_seconds tokens_per_second")
            tokens(run["token_ids"])
            for key in ("ttft_seconds", "elapsed_seconds", "tokens_per_second"):
                if type(run[key]) not in (int, float) or not math.isfinite(run[key]) or run[key] < 0:
                    raise ValueError("invalid timing")
            if run["elapsed_seconds"] <= 0 or run["ttft_seconds"] > run["elapsed_seconds"] or not math.isclose(
                    run["tokens_per_second"], len(run["token_ids"]) / run["elapsed_seconds"], rel_tol=1e-12):
                raise ValueError("inconsistent timing")
            if result["accepted"] and run["token_ids"] != profile["baseline_token_ids"]:
                raise ValueError("accepted candidate changed output")
    valid = [r for r in results if r["accepted"]]
    if not valid or type(profile["recommended_threads"]) is not int or profile["recommended_threads"] != max(
            valid, key=lambda r: statistics.median(m["tokens_per_second"] for m in r["measurements"]))["threads"]:
        raise ValueError("invalid recommendation")


def save_profile(profile, path):
    """Atomically replace a strictly validated JSON profile."""
    _validate(profile)
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            json.dump(profile, handle, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def load_profile(path, *, expected_identity):
    """Load only a profile matching the caller's exact expected identity."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    with Path(path).open(encoding="utf-8") as handle:
        profile = json.load(handle, object_pairs_hook=unique)
    _validate(profile)
    if profile["identity"] != expected_identity:
        raise ValueError("cached profile identity mismatch")
    return profile
