"""Production universal adapters for every verified-admitted model type.

Each admitted checkpoint ``model_type`` in the canonical source matrix gets:

* an exact route resolver that reads the checkpoint inventory (and, for
  OLMoE, its packed expert manifest) to name the six-field route, and
* an opener that constructs the same family owner the production gateway and
  CLIs use, after the same P02 checkpoint-inventory gate.

Openers return transactional owners (``snapshot``/``restore``) so the
universal session can roll back cancelled or failed requests.

Nothing here infers admission: the set of registered types is derived from
the matrix's ``verified-admitted`` rows, and ``UniversalEngine.open`` still
checks the exact route against the matrix before calling any opener.
"""
from __future__ import annotations

import copy
import hashlib
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from .failure_reporting import retain_cleanup_failure
from .universal_engine import ModelRoute


_ORDINARY_DTYPES = ("F32", "F16", "BF16")
_FP8_ROUTE = ("F8_E4M3", "fp8-e4m3-b128-f32")
_HF_METADATA_NAMES = frozenset({
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "preprocessor_config.json", "processor_config.json",
    "chat_template.json", "added_tokens.json", "vocab.json"})
_PUBLISHER_REQUEST_FIELDS = frozenset({
    "temperature", "top_k", "top_p", "min_p", "repetition_penalty", "frequency_penalty",
    "presence_penalty", "seed", "stop_token_id", "stop_prefix", "continue_session"})
_V41_REQUEST_FIELDS = frozenset({"temperature", "seed", "stop_prefix", "dspark"})
_STREAM_QUEUE_EVENTS = 256


def admitted_universal_model_types() -> tuple[str, ...]:
    """Exact checkpoint identities with at least one verified-admitted row."""
    from .source_scope import source_scope_matrix
    return tuple(sorted({case["model_type"] for case in source_scope_matrix()["execution_cases"]
                         if case["combination_status"] == "verified-admitted"}))


# ---------------------------------------------------------------------------
# Route resolution
# ---------------------------------------------------------------------------

def _inventory_dtypes(directory: Path, *, max_metadata_bytes: int) -> set[str]:
    from .publisher_checkpoint.storage import snapshot
    inventory = snapshot(directory, max_metadata_bytes=max_metadata_bytes,
                         max_tensors=1_000_000, max_shards=4096)
    return {record.dtype for record in inventory.tensors.values()}


def _ordinary_source_format(dtypes: set[str]) -> str:
    ordinary = sorted(dtype for dtype in dtypes if dtype in _ORDINARY_DTYPES)
    if len(ordinary) != 1:
        raise ValueError("checkpoint mixes ordinary source dtypes " + repr(ordinary) +
                         "; pass an explicit route")
    return ordinary[0]


def expert_manifests(directory: Path, *, max_bytes: int = 64 * 1024 * 1024) -> dict[Path, str]:
    """Return every valid sparse-expert manifest in a checkpoint with its format."""
    from .expert_runtime import MANIFEST_FORMAT, SparseExpertManifest
    found: dict[Path, str] = {}
    for path in sorted(Path(directory).iterdir()):
        if (path.suffix != ".json" or path.name in _HF_METADATA_NAMES or
                path.name.endswith(".index.json") or not path.is_file()):
            continue
        if path.stat().st_size > max_bytes:
            continue
        try:
            payload = json.loads(path.read_bytes())
        except (OSError, UnicodeError, ValueError):
            continue
        if type(payload) is dict and payload.get("format") == MANIFEST_FORMAT:
            SparseExpertManifest.read(path, max_bytes=max_bytes)
            found[path.resolve()] = _manifest_persisted_format(path)
    return found


def _require_manifest_owner(path: Path, directory: Path) -> None:
    """A manifest belongs to a checkpoint only if it lives there and binds its metadata."""
    from .expert_runtime import SparseExpertManifest
    from .olmoe import olmoe_metadata_fingerprint
    if path.parent != directory:
        raise ValueError("expert manifest must be a direct file of the admitted checkpoint")
    packed = SparseExpertManifest.read(path)
    if packed.architecture != "olmoe" or packed.fingerprint != olmoe_metadata_fingerprint(directory):
        raise ValueError(f"expert manifest {path.name} does not belong to this checkpoint "
                         "(architecture or metadata fingerprint differs)")


def select_expert_manifest(directory: Path, *, manifest: str | Path | None = None,
                           persisted_format: str | None = None) -> tuple[Path, str]:
    """Choose exactly one packed-expert manifest, or fail closed.

    The chosen manifest must be a direct file of the checkpoint (symlinks are
    resolved first) and must bind that checkpoint's architecture and metadata
    fingerprint.  An explicit ``manifest`` wins and must exist; when ``persisted_format`` is
    also given (from an explicit route) the manifest must persist exactly that
    format.  Otherwise the selection must be unique: the only manifest in the
    checkpoint, or the only one with the requested persisted format.
    """
    directory = Path(directory).resolve(strict=True)
    if manifest is not None:
        path = Path(manifest).resolve(strict=True)
        _require_manifest_owner(path, directory)
        persisted = _manifest_persisted_format(path)
        if persisted_format is not None and persisted != persisted_format:
            raise ValueError(f"expert manifest {path.name} persists {persisted!r}, "
                             f"route admits {persisted_format!r}")
        return path, persisted
    available = expert_manifests(directory)
    candidates = {path: fmt for path, fmt in available.items()
                  if persisted_format is None or fmt == persisted_format}
    if len(candidates) == 1:
        path, persisted = next(iter(candidates.items()))
        _require_manifest_owner(path, directory)
        return path, persisted
    listing = ", ".join(f"{path.name}={fmt}" for path, fmt in sorted(available.items())) or "none"
    wanted = "" if persisted_format is None else f" with persisted format {persisted_format!r}"
    raise ValueError(f"exactly one sparse-expert manifest{wanted} is required, found "
                     f"{len(candidates)}; pass manifest=<path> to choose one of: {listing}")


def single_expert_manifest(directory: Path, *, max_bytes: int = 64 * 1024 * 1024) -> Path:
    """Return the checkpoint's only sparse-expert manifest, or fail closed."""
    return select_expert_manifest(directory)[0]


def _manifest_persisted_format(path: Path) -> str:
    from .expert_runtime import SparseExpertManifest
    manifest = SparseExpertManifest.read(path)
    formats = {layer.dtype for layer in manifest.layers}
    if len(formats) != 1:
        raise ValueError("sparse-expert manifest mixes persisted formats " + repr(sorted(formats)))
    return formats.pop()


def resolve_checkpoint_route(config: dict[str, Any], directory: Path, *,
                             options: dict[str, Any] | None = None,
                             route: ModelRoute | None = None,
                             max_metadata_bytes: int = 256 * 1024 ** 2) -> ModelRoute:
    """Resolve the default native route of one checkpoint from its own files.

    Device, public-range and other non-default routes are never guessed;
    callers request them with an explicit ``route``.  For OLMoE the persisted
    format is read from the selected packed manifest: the explicit
    ``manifest`` opening option, a manifest matching an explicitly requested
    route, or the checkpoint's only manifest.
    """
    from .source_scope import source_family_for_identity
    model_type = config.get("model_type")
    if type(model_type) is not str or not model_type:
        raise ValueError("model config lacks exact model_type")
    family = source_family_for_identity(model_type)
    directory = Path(directory)
    if model_type == "deepseek_v41":
        # V4.1 executes three publisher banks together; the opener gates all
        # three rows. The route names the BF16 resident bank.
        return ModelRoute(family, model_type, "native", "publisher-bf16-fp8-fp4", "BF16", "bf16")
    dtypes = _inventory_dtypes(directory, max_metadata_bytes=max_metadata_bytes)
    if model_type == "olmoe":
        persisted_format = None if route is None else route.persisted_format
        _, persisted = select_expert_manifest(
            directory, manifest=(options or {}).get("manifest"),
            persisted_format=persisted_format)
        return ModelRoute(family, model_type, "native", "fp32",
                          _ordinary_source_format(dtypes), persisted)
    if not any(dtype in _ORDINARY_DTYPES for dtype in dtypes) and _FP8_ROUTE[0] in dtypes:
        return ModelRoute(family, model_type, "native", "fp32", *_FP8_ROUTE)
    return ModelRoute(family, model_type, "native", "fp32", _ordinary_source_format(dtypes), "float32")


# ---------------------------------------------------------------------------
# Owners
# ---------------------------------------------------------------------------

DEFAULT_STREAM_JOIN_SECONDS = 30.0


class _StreamGuard:
    """Own one owner's stream worker and its quarantine state.

    A stream worker that ignores cancellation after its consumer leaves is
    never raced: the owner is quarantined, rollback is refused while the
    worker runs, and ``close`` waits (retryably) for the worker to exit.
    """

    def __init__(self, join_seconds: float = DEFAULT_STREAM_JOIN_SECONDS) -> None:
        self.join_seconds = join_seconds
        self.thread: threading.Thread | None = None
        self.quarantine_reason: str | None = None

    def check(self) -> None:
        if self.quarantine_reason is not None:
            raise RuntimeError("owner quarantined: " + self.quarantine_reason)
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("owner has a running stream worker")

    def release_for_close(self, timeout: float) -> None:
        thread = self.thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                raise RuntimeError("stream worker is still running; close can be retried")
            self.thread = None


def _run_stream(generate: Callable[..., Any], token_ids: Any, *, guard: _StreamGuard,
                cancelled: Callable[[], bool] | None, **request: Any) -> Iterator[dict[str, Any]]:
    """Adapt a synchronous ``emit`` request into a bounded event iterator.

    The worker observes consumer abandonment through its cancellation hook.
    The iterator waits at most ``guard.join_seconds`` for it after abandonment;
    a worker that is still running then quarantines the owner instead of
    letting a rollback race it.
    """
    guard.check()
    events: queue.Queue = queue.Queue(maxsize=_STREAM_QUEUE_EVENTS)
    abandoned = threading.Event()
    cancellation_requested = threading.Event()
    outcome: dict[str, Any] = {}

    def is_cancelled() -> bool:
        # The owner worker must not call the caller's predicate: it may be
        # thread-affine, and UniversalSession also evaluates it on the
        # consumer thread.  Poll it only in the iterator and signal the worker
        # through this event.
        return abandoned.is_set() or cancellation_requested.is_set()

    def emit(event: dict[str, Any]) -> None:
        while not abandoned.is_set():
            try:
                events.put(("event", event), timeout=0.05)
                return
            except queue.Full:
                continue

    def worker() -> None:
        try:
            outcome["result"] = generate(token_ids, cancelled=is_cancelled, emit=emit,
                                         _stream_worker=True, **request)
        except BaseException as error:
            outcome["error"] = error
        finally:
            while not abandoned.is_set():
                try:
                    events.put(("done", None), timeout=0.05)
                    break
                except queue.Full:
                    continue

    thread = threading.Thread(target=worker, name="openagent-universal-stream", daemon=True)
    guard.thread = thread
    thread.start()
    try:
        while True:
            try:
                kind, event = events.get(timeout=0.1)
            except queue.Empty:
                if cancelled is not None and cancelled():
                    cancellation_requested.set()
                    from .universal_engine import GenerationCancelled
                    raise GenerationCancelled("generation cancelled while waiting for stream output")
                continue
            if kind == "done":
                break
            yield event
        if "error" in outcome:
            raise outcome["error"]
        yield {"event": "result", "result": outcome["result"]}
    finally:
        abandoned.set()
        deadline = time.monotonic() + guard.join_seconds
        while thread.is_alive() and time.monotonic() < deadline:
            try:
                while True:
                    events.get_nowait()
            except queue.Empty:
                pass
            thread.join(timeout=0.05)
        if thread.is_alive():
            guard.quarantine_reason = "stream worker ignored cancellation"
            raise RuntimeError("stream worker ignored cancellation; owner quarantined")
        guard.thread = None


class PublisherServiceOwner:
    """Transactional owner over a service-backed family ``PublisherSession``.

    Used for ``native_family_cli`` and ``native_conditional_cli`` sessions,
    whose rollback receipts are ``NativeExecutionService`` snapshots.
    """

    def __init__(self, operator: Any, args: Any, session: Any) -> None:
        self._operator, self._args, self._session = operator, args, session
        self.family = args.family
        self._guard = _StreamGuard()

    @property
    def checkpoint_identity(self) -> str | None:
        return getattr(self._session, "checkpoint_identity", None)

    def _request(self, token_ids: Any, *, max_tokens: int | None, fields: dict[str, Any]) -> Any:
        unknown = set(fields) - _PUBLISHER_REQUEST_FIELDS
        if unknown:
            raise ValueError("unsupported universal request fields: " + ", ".join(sorted(unknown)))
        tokens = list(token_ids)
        if not tokens or any(type(token) is not int for token in tokens):
            raise ValueError("nonempty integer prompt token IDs required")
        request = copy.copy(self._args)
        request.token_ids = json.dumps(tokens)
        if max_tokens is not None:
            if type(max_tokens) is not int or max_tokens <= 0:
                raise ValueError("positive max_tokens required")
            request.max_tokens = max_tokens
        for name, value in fields.items():
            setattr(request, name, value)
        return request

    def generate(self, token_ids: Any, *, cancelled: Callable[[], bool] | None = None,
                 max_tokens: int | None = None, emit: Callable[[dict], None] | None = None,
                 _stream_worker: bool = False, **fields: Any) -> dict[str, Any]:
        if not _stream_worker:
            self._guard.check()
        request = self._request(token_ids, max_tokens=max_tokens, fields=fields)
        return self._session.run(request, emit=emit, cancelled=cancelled)

    def stream(self, token_ids: Any, *, cancelled: Callable[[], bool] | None = None,
               **request: Any) -> Iterator[dict[str, Any]]:
        return _run_stream(self.generate, token_ids, guard=self._guard, cancelled=cancelled, **request)

    def _service(self) -> tuple[Any, Any]:
        self._guard.check()
        service, session_id = self._session.service, self._session.session_id
        if service is None or session_id is None:
            raise RuntimeError("publisher session has no open execution service")
        return service, session_id

    def snapshot(self) -> Any:
        service, session_id = self._service()
        return service.snapshot(session_id)

    def restore(self, receipt: Any) -> None:
        service, session_id = self._service()
        service.restore(session_id, receipt)

    def release_snapshot(self, receipt: Any) -> None:
        service, session_id = self._service()
        service.release_snapshot(session_id, receipt)

    def expert_telemetry_sources(self) -> tuple[Any, ...]:
        sources = getattr(self._session, "expert_telemetry_sources", None)
        return tuple(sources()) if callable(sources) else ()

    def close(self) -> None:
        self._guard.release_for_close(self._guard.join_seconds)
        self._session.close()


class _ResetStateReceipt:
    """Receipt for an owner that resets to one canonical state per request."""
    __slots__ = ("identity", "requests")

    def __init__(self, identity: str, requests: int) -> None:
        self.identity, self.requests = identity, requests


class PublisherStatelessOwner:
    """Transactional owner over the DeepSeek V4.1 ``PublisherSession``.

    That session resets target and draft state before and after every request
    (and closes itself if a reset fails), so its only persistent state is the
    canonical reset state; restoring re-applies that reset.
    """

    def __init__(self, operator: Any, args: Any, session: Any) -> None:
        self._operator, self._args, self._session = operator, args, session
        self.family = "deepseek_v41"
        self._guard = _StreamGuard()

    @property
    def checkpoint_identity(self) -> str | None:
        value = getattr(self._session, "_signature_value", None)
        return None if value is None else repr(value)

    def generate(self, token_ids: Any, *, cancelled: Callable[[], bool] | None = None,
                 max_tokens: int | None = None, emit: Callable[[dict], None] | None = None,
                 _stream_worker: bool = False, **fields: Any) -> dict[str, Any]:
        if not _stream_worker:
            self._guard.check()
        unknown = set(fields) - _V41_REQUEST_FIELDS
        if unknown:
            raise ValueError("unsupported universal request fields: " + ", ".join(sorted(unknown)))
        tokens = list(token_ids)
        if not tokens or any(type(token) is not int for token in tokens):
            raise ValueError("nonempty integer prompt token IDs required")
        request = copy.copy(self._args)
        request.token_ids = json.dumps(tokens)
        if max_tokens is not None:
            if type(max_tokens) is not int or max_tokens <= 0:
                raise ValueError("positive max_tokens required")
            request.max_tokens = max_tokens
        for name, value in fields.items():
            setattr(request, name, value)
        return self._session.run(request, emit=emit, cancelled=cancelled)

    def stream(self, token_ids: Any, *, cancelled: Callable[[], bool] | None = None,
               **request: Any) -> Iterator[dict[str, Any]]:
        return _run_stream(self.generate, token_ids, guard=self._guard, cancelled=cancelled, **request)

    def snapshot(self) -> _ResetStateReceipt:
        self._guard.check()
        if getattr(self._session, "_closed", False):
            raise RuntimeError("publisher session is closed")
        return _ResetStateReceipt(repr(getattr(self._session, "_signature_value", None)),
                                  getattr(self._session, "_requests", 0))

    def restore(self, receipt: Any) -> None:
        self._guard.check()
        if type(receipt) is not _ResetStateReceipt:
            raise ValueError("V4.1 restore requires its own reset-state receipt")
        if receipt.identity != repr(getattr(self._session, "_signature_value", None)):
            raise ValueError("V4.1 reset-state receipt belongs to another session")
        self._session._reset_request()

    def expert_telemetry_sources(self) -> tuple[Any, ...]:
        return ()

    def close(self) -> None:
        self._guard.release_for_close(self._guard.join_seconds)
        self._session.close()


class OlmoeNativeOwner:
    """Persistent, transactional owner for the admitted native OLMoE route.

    Builds the same leased segment engine, edge and decoder session as
    ``native_olmoe_cli`` and decodes with the same greedy argmax policy.
    """

    def __init__(self, directory: Path, *, manifest: Path, library: Path, context_tokens: int,
                 resident_mib: int, cache_mib: int, transfer_mib: int, staging_mib: int,
                 conversion_mib: int | None = None) -> None:
        from contextlib import ExitStack
        from .native_leased_olmoe_segments import NativeLeasedOlmoeSegmentEngine
        from .native_olmoe_segments import NativeOlmoeEdge
        from .olmoe import OlmoeConfig
        for name, value in (("context_tokens", context_tokens), ("resident_mib", resident_mib),
                            ("cache_mib", cache_mib), ("transfer_mib", transfer_mib),
                            ("staging_mib", staging_mib)):
            if type(value) is not int or value <= 0:
                raise ValueError("positive explicit OLMoE admission required: " + name)
        if conversion_mib is not None:
            if type(conversion_mib) is not int or conversion_mib <= 0:
                raise ValueError("positive explicit OLMoE admission required: conversion_mib")
        self.config = OlmoeConfig.read(directory)
        if context_tokens > self.config.max_position_embeddings:
            raise ValueError("context capacity exceeds the checkpoint limit")
        mib = 1024 ** 2
        self._stack = ExitStack()
        try:
            self._engine = self._stack.enter_context(NativeLeasedOlmoeSegmentEngine(
                directory, manifest, 0, self.config.num_hidden_layers, Path(library).resolve(strict=True),
                max_resident_bytes=resident_mib * mib, expert_cache_bytes=cache_mib * mib,
                context_tokens=context_tokens, max_bytes=transfer_mib * mib,
                max_staging_bytes=staging_mib * mib))
            self._edge = self._stack.enter_context(NativeOlmoeEdge(
                directory, Path(library).resolve(strict=True), max_resident_bytes=resident_mib * mib,
                max_bytes=transfer_mib * mib,
                max_conversion_bytes=None if conversion_mib is None else conversion_mib * mib))
            self._decoder = self._stack.enter_context(self._engine.create_decoder_session(self._edge))
        except BaseException:
            self._stack.close()
            raise
        self.context_tokens = context_tokens
        self.family = "olmoe"
        self.manifest_path = Path(manifest)
        self._guard = _StreamGuard()

    @property
    def checkpoint_identity(self) -> str:
        return self._edge.execution_identity

    def generate(self, token_ids: Any, *, cancelled: Callable[[], bool] | None = None,
                 max_tokens: int = 32, emit: Callable[[dict], None] | None = None,
                 continue_session: bool = False, capture: list | None = None,
                 _stream_worker: bool = False) -> dict[str, Any]:
        """Greedy decode; ``capture`` (a list) receives each step's FP32 logits row."""
        import numpy as np
        if not _stream_worker:
            self._guard.check()
        tokens = list(token_ids)
        vocab = self.config.vocab_size
        if not tokens or any(type(token) is not int or not 0 <= token < vocab for token in tokens):
            raise ValueError("token IDs must be integers in the checkpoint vocabulary")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("positive max_tokens required")
        if not continue_session:
            self._decoder.reset()
        if self._decoder.position + len(tokens) + max_tokens - 1 > self.context_tokens:
            raise ValueError("prompt and generation exceed the selected context capacity")
        generated: list[int] = []
        pending = tokens
        finish_reason = "length"
        for _ in range(max_tokens):
            logits = self._decoder.forward(pending, cancelled=cancelled)
            if capture is not None:
                capture.append(np.array(logits[-1], dtype=np.float32, copy=True))
            token = int(np.argmax(logits[-1]))
            generated.append(token)
            if emit is not None:
                emit({"event": "tokens", "token_ids": [token], "committed": False})
            pending = [token]
            if token == self.config.eos_token_id:
                finish_reason = "eos"
                break
        return {"generated_ids": generated, "finish_reason": finish_reason,
                "context_tokens": self._decoder.position, "family": "olmoe",
                "implementation": self._engine.numeric_class,
                "numerical_parity_verified": False, "platform_certified": False}

    def stream(self, token_ids: Any, *, cancelled: Callable[[], bool] | None = None,
               **request: Any) -> Iterator[dict[str, Any]]:
        return _run_stream(self.generate, token_ids, guard=self._guard, cancelled=cancelled, **request)

    @property
    def position(self) -> int:
        return int(self._decoder.position)

    @property
    def numeric_class(self) -> str:
        return str(self._engine.numeric_class)

    def cache_observation(self) -> dict[str, Any]:
        """Measure the physical KV cache: position, per-layer digests and bytes."""
        import hashlib
        import numpy as np
        self._guard.check()
        state = self._decoder.snapshot_state()["range_state"]
        keys, values = state["keys"], state["values"]
        position = int(self._decoder.position)
        digests, total = [], 0
        for key, value in zip(keys, values):
            used_key = np.ascontiguousarray(np.asarray(key)[..., :position, :]) if np.asarray(key).ndim >= 2 else np.asarray(key)
            used_value = np.ascontiguousarray(np.asarray(value)[..., :position, :]) if np.asarray(value).ndim >= 2 else np.asarray(value)
            digest = hashlib.sha256(used_key.tobytes())
            digest.update(used_value.tobytes())
            digests.append(digest.hexdigest())
            total += used_key.nbytes + used_value.nbytes
        return {"position": position, "layers": len(digests), "layer_digests": digests,
                "bytes": int(total)}

    def snapshot(self) -> dict[str, Any]:
        self._guard.check()
        return self._decoder.snapshot_state()

    def restore(self, receipt: Any) -> None:
        self._guard.check()
        self._decoder.restore_state(receipt)

    def expert_telemetry_sources(self) -> tuple[Any, ...]:
        loader = getattr(self._engine, "_loader", None)
        experts = getattr(loader, "experts", None)
        return (experts,) if experts is not None else ()

    def close(self) -> None:
        self._guard.release_for_close(self._guard.join_seconds)
        self._stack.close()


class OlmoeDirectOwner:
    """Transactional owner for OLMoE native formats without a leased ABI path.

    The direct production provider selects an explicit persisted-format kernel
    and disables registry fallback. It supplies the complete NumPy decoder
    while retaining the same isolated KV snapshot/restore contract.
    """

    def __init__(self, directory: Path, *, manifest: Path, persisted_format: str,
                 context_tokens: int, cache_mib: int, io_mode: str = "mmap",
                 prefetch_workers: int = 2, prefetch_depth: int = 8,
                 storage_roots: dict[str, str] | None = None) -> None:
        from ..model_gateway import OlmoeDirectProvider, _olmoe_default_kernel_for_format
        from .olmoe import OlmoeConfig

        if type(context_tokens) is not int or context_tokens <= 0:
            raise ValueError("positive OLMoE context admission required")
        if type(cache_mib) is not int or cache_mib <= 0:
            raise ValueError("positive OLMoE expert-cache admission required")
        if type(prefetch_workers) is not int or prefetch_workers <= 0:
            raise ValueError("positive OLMoE prefetch worker count required")
        if type(prefetch_depth) is not int or prefetch_depth < prefetch_workers:
            raise ValueError("OLMoE prefetch depth must cover all workers")
        if type(io_mode) is not str or io_mode not in {"buffered", "mmap", "direct", "uring"}:
            raise ValueError("unsupported OLMoE expert I/O mode")
        if storage_roots is not None and (type(storage_roots) is not dict or
                any(type(key) is not str or not isinstance(value, (str, Path))
                    for key, value in storage_roots.items())):
            raise ValueError("OLMoE storage roots must map strings to paths")

        provider = OlmoeDirectProvider()
        provider.model_path = str(Path(directory).resolve(strict=True))
        provider.manifest_path = str(Path(manifest).resolve(strict=True))
        provider.context_tokens = context_tokens
        provider.cache_gb = cache_mib / 1024
        provider.io_mode = io_mode
        provider.prefetch_workers = prefetch_workers
        provider.prefetch_depth = prefetch_depth
        provider.storage_roots = storage_roots
        # This route is not permitted to fall back to the NumPy reference
        # kernel: OlmoeDirectProvider passes this explicit preferred kernel to
        # OlmoeModel, whose registry resolution is fail-closed in that mode.
        provider.preferred_kernel = _olmoe_default_kernel_for_format(persisted_format)
        self._provider = provider
        self._model = provider._load()
        self._config = OlmoeConfig.read(directory)
        self._guard = _StreamGuard()
        self.family = "olmoe"
        self.persisted_format = persisted_format

    @property
    def checkpoint_identity(self) -> str:
        return str(self._model._state_identity)

    @property
    def position(self) -> int:
        return int(self._model.position)

    def generate(self, token_ids: Any, *, cancelled: Callable[[], bool] | None = None,
                 max_tokens: int = 32, emit: Callable[[dict], None] | None = None,
                 continue_session: bool = False, capture: list | None = None,
                 _stream_worker: bool = False) -> dict[str, Any]:
        import numpy as np
        from .universal_engine import GenerationCancelled

        if not _stream_worker:
            self._guard.check()
        tokens = list(token_ids)
        if not tokens or any(type(token) is not int or
                             not 0 <= token < self._config.vocab_size for token in tokens):
            raise ValueError("token IDs must be integers in the OLMoE vocabulary")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("positive max_tokens required")
        if not continue_session:
            self._model.reset()
        if self._model.position + len(tokens) + max_tokens - 1 > self._model.context_tokens:
            raise ValueError("prompt and generation exceed the selected context capacity")

        def check_cancelled() -> None:
            if cancelled is not None and cancelled():
                raise GenerationCancelled("OLMoE generation cancelled")

        check_cancelled()
        logits = self._model.forward(tokens)
        generated: list[int] = []
        finish_reason = "length"
        for index in range(max_tokens):
            check_cancelled()
            row = np.asarray(logits, dtype=np.float32)
            if capture is not None:
                capture.append(np.array(row, copy=True))
            token = int(np.argmax(row))
            generated.append(token)
            if emit is not None:
                emit({"event": "tokens", "token_ids": [token], "committed": False})
            if token == self._config.eos_token_id:
                finish_reason = "eos"
                break
            if index + 1 < max_tokens:
                check_cancelled()
                logits = self._model.forward([token])
        check_cancelled()
        return {"generated_ids": generated, "finish_reason": finish_reason,
                "context_tokens": int(self._model.position), "family": self.family,
                "implementation": self._provider.preferred_kernel,
                "numerical_parity_verified": False, "platform_certified": False}

    def stream(self, token_ids: Any, *, cancelled: Callable[[], bool] | None = None,
               **request: Any) -> Iterator[dict[str, Any]]:
        return _run_stream(self.generate, token_ids, guard=self._guard,
                           cancelled=cancelled, **request)

    def snapshot(self) -> dict[str, Any]:
        self._guard.check()
        return self._model.snapshot_state()

    def restore(self, receipt: Any) -> None:
        self._guard.check()
        self._model.restore_state(receipt)

    def release_snapshot(self, receipt: Any) -> None:
        # OlmoeModel snapshots are owned copies, not native handles.
        return None

    def cache_observation(self) -> dict[str, Any]:
        self._guard.check()
        state = self._model.snapshot_state()
        digests, total = [], 0
        for key, value in zip(state["keys"], state["values"]):
            digest = hashlib.sha256(key.tobytes())
            digest.update(value.tobytes())
            digests.append(digest.hexdigest())
            total += key.nbytes + value.nbytes
        return {"position": int(state["position"]), "layers": len(digests),
                "layer_digests": digests, "bytes": int(total)}

    def expert_telemetry_sources(self) -> tuple[Any, ...]:
        experts = getattr(self._model, "experts", None)
        return (experts,) if experts is not None else ()

    def close(self) -> None:
        self._guard.release_for_close(self._guard.join_seconds)
        self._model.close()


# ---------------------------------------------------------------------------
# Openers
# ---------------------------------------------------------------------------

def _options(admission: dict[str, Any], allowed: frozenset[str], required: frozenset[str]) -> dict[str, Any]:
    options = admission.get("options")
    if type(options) is not dict:
        raise TypeError("universal admission options must be a dictionary")
    unknown = set(options) - allowed
    if unknown:
        raise ValueError("undeclared adapter options: " + ", ".join(sorted(unknown)))
    missing = required - set(options)
    if missing:
        raise ValueError("required adapter options missing: " + ", ".join(sorted(missing)))
    return options


def open_publisher_owner(directory: Path, *, admission: dict[str, Any]) -> Any:
    """Open a publisher-family checkpoint exactly as the gateway does."""
    from .publisher_provider import (NativePublisherProvider, parse_publisher_operator,
                                     publisher_operator_backend)
    manifest = admission["manifest"]
    route = manifest.route
    options = _options(admission, frozenset({"argv"}), frozenset({"argv"}))
    argv = options["argv"]
    if type(argv) is not list or any(type(arg) is not str for arg in argv):
        raise TypeError("argv must be a list of operator argument strings")
    if any(arg == "--checkpoint" or arg.startswith("--checkpoint=") or arg == "--family" or
           arg.startswith("--family=") for arg in argv):
        raise ValueError("the universal engine supplies the admitted checkpoint and family")
    model_type = manifest.model_type
    checkpoint_args = ([str(directory)] if model_type == "deepseek_v41"
                       else ["--checkpoint", str(directory)])
    operator, args = parse_publisher_operator(model_type, [*checkpoint_args, *argv])
    backend = publisher_operator_backend(model_type, args)
    if backend != route.backend:
        raise ValueError(f"operator arguments select backend {backend!r}, route admits {route.backend!r}")
    NativePublisherProvider._require_source_route({"family": model_type}, args, Path(directory))
    session = operator.PublisherSession(args, cancelled=None)
    try:
        if model_type == "deepseek_v41":
            return PublisherStatelessOwner(operator, args, session)
        return PublisherServiceOwner(operator, args, session)
    except BaseException as primary:
        try:
            session.close()
        except BaseException as cleanup:
            retain_cleanup_failure(
                primary, session.close, cleanup,
                "publisher owner construction cleanup remains retryable: ")
        raise


_OLMOE_OPTIONS = frozenset({"library", "manifest", "context_tokens", "resident_mib", "cache_mib",
                            "transfer_mib", "staging_mib", "conversion_mib", "io_mode",
                            "prefetch_workers", "prefetch_depth", "storage_roots"})


def open_olmoe_owner(directory: Path, *, admission: dict[str, Any]) -> OlmoeNativeOwner:
    """Open the admitted native OLMoE route with exactly one packed manifest.

    The manifest is the explicit ``manifest`` option or the unique manifest
    that persists the route's format; it must persist exactly that format.
    """
    manifest = admission["manifest"]
    route = manifest.route
    if route.backend != "native":
        raise ValueError("the universal OLMoE adapter admits only the native backend")
    options = _options(admission, _OLMOE_OPTIONS, frozenset())
    source_format = _ordinary_source_format(
        _inventory_dtypes(Path(directory), max_metadata_bytes=256 * 1024 ** 2))
    if source_format != route.source_format:
        raise ValueError(f"checkpoint source format {source_format!r} differs from admitted route "
                         f"{route.source_format!r}")
    manifest_path, _ = select_expert_manifest(Path(directory), manifest=options.get("manifest"),
                                              persisted_format=route.persisted_format)
    context_tokens = options.get("context_tokens", 512)
    cache_mib = options.get("cache_mib", 256)
    if options.get("library") is not None:
        from .native_leased_olmoe_segments import _APPLY_ATTRIBUTES
        if route.persisted_format not in _APPLY_ATTRIBUTES:
            raise ValueError("the leased Segment/Edge consumer does not implement "
                             f"{route.persisted_format!r}; omit library to select its direct native consumer "
                             "(OlmoeDirectProvider)")
        return OlmoeNativeOwner(
            Path(directory), manifest=manifest_path, library=Path(options["library"]),
            context_tokens=context_tokens, resident_mib=options.get("resident_mib", 2048),
            cache_mib=cache_mib, transfer_mib=options.get("transfer_mib", 2048),
            staging_mib=options.get("staging_mib", 256), conversion_mib=options.get("conversion_mib"))
    direct_only = {"io_mode", "prefetch_workers", "prefetch_depth", "storage_roots"}
    leased_only = {"resident_mib", "transfer_mib", "staging_mib", "conversion_mib"}
    unexpected = set(options) & leased_only
    if unexpected:
        raise ValueError("leased-runtime budgets require library=: " + ", ".join(sorted(unexpected)))
    direct_options = set(options) - {"manifest", "context_tokens", "cache_mib"} - direct_only
    if direct_options:
        raise ValueError("unsupported OLMoE direct-owner options: " + ", ".join(sorted(direct_options)))
    return OlmoeDirectOwner(
        Path(directory), manifest=manifest_path, persisted_format=route.persisted_format,
        context_tokens=context_tokens, cache_mib=cache_mib,
        io_mode=options.get("io_mode", "mmap"),
        prefetch_workers=options.get("prefetch_workers", 2),
        prefetch_depth=options.get("prefetch_depth", 8),
        storage_roots=options.get("storage_roots"))


def register_production_universal_adapters(engine: Any) -> tuple[str, ...]:
    """Register exact openers and route resolvers for every admitted model type.

    ``engine`` is a ``UniversalEngine``.  Returns the registered types, which
    equal :func:`admitted_universal_model_types` exactly.
    """
    registered = []
    for model_type in admitted_universal_model_types():
        opener = open_olmoe_owner if model_type == "olmoe" else open_publisher_owner
        engine.register(model_type, opener, route_resolver=resolve_checkpoint_route)
        registered.append(model_type)
    return tuple(registered)


__all__ = ["DEFAULT_STREAM_JOIN_SECONDS", "OlmoeDirectOwner", "OlmoeNativeOwner", "PublisherServiceOwner",
           "PublisherStatelessOwner", "admitted_universal_model_types", "expert_manifests",
           "open_olmoe_owner", "open_publisher_owner", "register_production_universal_adapters",
           "resolve_checkpoint_route", "select_expert_manifest", "single_expert_manifest"]
