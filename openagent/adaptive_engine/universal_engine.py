"""One user-facing engine boundary for all exact family adapters.

The universal engine owns admission, warm-session reuse, request transactions,
recovery receipts and retryable cleanup.  Family adapters own model execution.

Owner protocol
--------------
An opener returns an owner exposing:

``generate(*args, cancelled=..., **kwargs)`` or ``run(...)``
    Execute one request and return its result.
``stream(*args, cancelled=..., **kwargs)`` or ``stream_generate(...)`` (optional)
    Return an iterator of events for one request.
``snapshot() -> receipt``
    Capture the owner's exact current request state.
``restore(receipt)``
    Restore a receipt previously returned by ``snapshot`` on this owner.
``release_snapshot(receipt)`` (optional)
    Release resources retained by a receipt.
``close()``
    Release every owned resource.  Raising leaves the owner retryable.

``snapshot``/``restore`` are mandatory: every request is transactional, so an
owner that cannot roll back is not admitted.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path, PurePath
from typing import Any, Callable, Iterator, Protocol
from uuid import uuid4

from .failure_reporting import add_failure_note, failure_detail, retain_cleanup_failure


UNIVERSAL_RECEIPT_VERSION = "openagent.universal-receipt.v1"
DEFAULT_RECEIPT_MAX_AGE_SECONDS = 3600.0


def _universal_lifecycle(method):
    """Serialize lifecycle calls and reject same-thread callback reentry."""
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self._lock:
            if self._active_request_thread is not None:
                raise RuntimeError("universal session lifecycle reentry is unsupported")
            self._active_request_thread = threading.get_ident()
            try:
                return method(self, *args, **kwargs)
            finally:
                self._active_request_thread = None
    return guarded


def _universal_request(method):
    """Own one request boundary, including the full lifetime of a stream."""
    if method.__name__ == "stream":
        @wraps(method)
        def stream_guard(self, *args, **kwargs):
            def iterate():
                with self._lock:
                    if self._active_request_thread == threading.get_ident():
                        raise RuntimeError("universal session request reentry is unsupported")
                self._request_gate.acquire()
                try:
                    with self._lock:
                        self._active_request_thread = threading.get_ident()
                    try:
                        # Do not retain an RLock across yield: consumers are
                        # allowed to close an iterator from a cancellation or
                        # teardown thread. The active marker still excludes
                        # lifecycle changes for the full stream lifetime.
                        yield from method(self, *args, **kwargs)
                    finally:
                        with self._lock:
                            self._active_request_thread = None
                finally:
                    self._request_gate.release()
            return iterate()
        return stream_guard

    @wraps(method)
    def request_guard(self, *args, **kwargs):
        with self._lock:
            if self._active_request_thread == threading.get_ident():
                raise RuntimeError("universal session request reentry is unsupported")
        self._request_gate.acquire()
        try:
            with self._lock:
                self._active_request_thread = threading.get_ident()
            try:
                return method(self, *args, **kwargs)
            finally:
                with self._lock:
                    self._active_request_thread = None
        finally:
            self._request_gate.release()
    return request_guard


class UniversalAdapter(Protocol):
    def open(self, model_directory: Path, *, admission: dict[str, Any]) -> Any: ...


class GenerationCancelled(RuntimeError):
    """A request was cancelled and its owner state was rolled back."""


class PublicationFailed(RuntimeError):
    """The output sink rejected a result; owner state was rolled back."""


@dataclass(frozen=True)
class ModelRoute:
    family: str
    model_type: str
    backend: str
    arithmetic: str
    source_format: str
    persisted_format: str

    def __post_init__(self) -> None:
        for name in ("family", "model_type", "backend", "arithmetic",
                     "source_format", "persisted_format"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError("exact non-empty route field required: " + name)

    def as_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in (
            "family", "model_type", "backend", "arithmetic",
            "source_format", "persisted_format")}

    def as_tuple(self) -> tuple[str, ...]:
        return tuple(self.as_dict().values())


@dataclass(frozen=True)
class ModelManifest:
    directory: Path
    model_type: str
    architecture: str
    route: ModelRoute
    config: dict[str, Any]
    config_sha256: str = ""

    @property
    def model_identity(self) -> str:
        """Checkpoint identity: resolved directory plus exact config digest."""
        return hashlib.sha256(
            (str(self.directory) + "\0" + self.config_sha256).encode("utf-8")).hexdigest()


def freeze_options(value: Any) -> Any:
    """Return a canonical, hashable identity for nested opening options.

    Types are tagged so ``1``, ``1.0`` and ``True`` stay distinct, dictionary
    and set order never matters, and sequence order always does.  Values with
    no stable identity are rejected rather than keyed by ``repr``.
    """
    if value is None:
        return ("none",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", value)
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError("non-finite float options have no stable session identity")
        return ("float", value.hex())
    if type(value) is str:
        return ("str", value)
    if type(value) is bytes:
        return ("bytes", value)
    if isinstance(value, PurePath):
        return ("path", str(value))
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("session option dictionaries require string keys")
            items.append((key, freeze_options(item)))
        return ("dict", tuple(sorted(items, key=lambda pair: pair[0])))
    if isinstance(value, (list, tuple)):
        return ("seq", tuple(freeze_options(item) for item in value))
    if isinstance(value, (set, frozenset)):
        frozen = [freeze_options(item) for item in value]
        return ("set", tuple(sorted(frozen, key=repr)))
    raise TypeError("session option has no canonical identity: " + type(value).__name__)


@dataclass(frozen=True, eq=False)
class UniversalReceipt:
    """Recovery receipt bound to one route, checkpoint, session and epoch.

    Equality is identity: a receipt is only valid as the exact object the
    session issued, so copies, forgeries and deserialized lookalikes fail.
    """
    version: str
    receipt_id: str
    route: tuple[str, ...]
    model_identity: str
    session_id: str
    epoch: int
    state_digest: str
    created_monotonic: float
    max_age_seconds: float
    _owner_receipt: Any = field(repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return {"version": self.version, "receipt_id": self.receipt_id,
                "route": list(self.route), "model_identity": self.model_identity,
                "session_id": self.session_id, "epoch": self.epoch,
                "state_digest": self.state_digest,
                "max_age_seconds": self.max_age_seconds}


def _accepts_options(resolver: Callable[..., Any]) -> bool:
    import inspect as _inspect
    try:
        parameters = _inspect.signature(resolver).parameters
    except (TypeError, ValueError):
        return False
    return "options" in parameters and parameters["options"].kind in (
        _inspect.Parameter.KEYWORD_ONLY, _inspect.Parameter.POSITIONAL_OR_KEYWORD)


def _accepts_route(resolver: Callable[..., Any]) -> bool:
    import inspect as _inspect
    try:
        parameters = _inspect.signature(resolver).parameters
    except (TypeError, ValueError):
        return False
    return "route" in parameters and parameters["route"].kind in (
        _inspect.Parameter.KEYWORD_ONLY, _inspect.Parameter.POSITIONAL_OR_KEYWORD)


def _as_cancellation(error: BaseException, cancelled: Callable[[], bool] | None) -> GenerationCancelled | None:
    """Normalize a family-specific cancellation into ``GenerationCancelled``."""
    if isinstance(error, GenerationCancelled) or not isinstance(error, Exception):
        return None
    try:
        requested = cancelled is not None and bool(cancelled())
    except Exception:
        return None
    return GenerationCancelled("generation cancelled; owner state rolled back") if requested else None


def _state_digest(owner_receipt: Any) -> str:
    """Digest the owner receipt's identity fields without arbitrary __repr__."""
    digest = hashlib.sha256(type(owner_receipt).__qualname__.encode("utf-8"))
    for name in ("service_id", "session_id", "snapshot_id", "identity"):
        value = getattr(owner_receipt, name, None)
        if type(value) is str:
            digest.update(b"\0" + name.encode() + b"=" + value.encode("utf-8"))
    tokens = getattr(owner_receipt, "token_ids", None)
    if type(tokens) is tuple and all(type(token) is int for token in tokens):
        digest.update(b"\0tokens=" + json.dumps(tokens).encode())
    digest.update(b"\0object=" + str(id(owner_receipt)).encode())
    return digest.hexdigest()


class UniversalEngine:
    """Shared launcher and exact adapter admission boundary."""

    def __init__(self, *, route_admission: Callable[[ModelRoute], None] | None = None) -> None:
        self._adapters: dict[str, Callable[..., Any]] = {}
        self._route_resolvers: dict[str, Callable[[dict[str, Any], Path], ModelRoute]] = {}
        self._sessions: dict[tuple[Any, ...], UniversalSession] = {}
        # Sessions replaced while their close is pending retry stay owned here
        # until a pool close succeeds, so no failed cleanup becomes unreachable.
        self._retired: list[UniversalSession] = []
        self._lock = threading.RLock()
        self._route_admission = route_admission or require_universal_route_admission

    def register(self, model_type: str, opener: Callable[..., Any], *,
                 route_resolver: Callable[[dict[str, Any], Path], ModelRoute] | None = None) -> None:
        if type(model_type) is not str or not model_type.strip():
            raise ValueError("exact model_type is required")
        if not callable(opener):
            raise TypeError("adapter opener must be callable")
        if route_resolver is not None and not callable(route_resolver):
            raise TypeError("route resolver must be callable")
        with self._lock:
            if model_type in self._adapters:
                raise ValueError("model_type adapter already registered: " + model_type)
            self._adapters[model_type] = opener
            if route_resolver is not None:
                self._route_resolvers[model_type] = route_resolver

    def inspect(self, model_directory: str | Path, *, route: ModelRoute | None = None,
                options: dict[str, Any] | None = None) -> ModelManifest:
        """Resolve the exact route for one checkpoint.

        A resolver that declares an ``options`` keyword receives a copy of the
        opening options, so a caller's explicit selection (for example one of
        several packed expert manifests) determines the route instead of being
        guessed.
        """
        directory = Path(model_directory).resolve(strict=True)
        config_path = directory / "config.json"
        if not config_path.is_file():
            raise ValueError("universal admission requires config.json")
        try:
            raw = config_path.read_bytes()
            config = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid model config for universal admission") from error
        if type(config) is not dict:
            raise ValueError("model config must be an object")
        model_type = config.get("model_type")
        architectures = config.get("architectures")
        architecture = architectures[0] if type(architectures) is list and architectures else None
        if type(model_type) is not str or not model_type:
            raise ValueError("model config lacks exact model_type")
        if type(architecture) is not str or not architecture:
            architecture = model_type
        with self._lock:
            registered = model_type in self._adapters
            resolver = self._route_resolvers.get(model_type)
        if not registered:
            from .source_scope import source_family_for_identity
            # Report a reviewed out-of-scope or unknown identity with its
            # catalog reason before the generic missing-adapter error.
            source_family_for_identity(model_type)
            raise ValueError("no exact universal adapter registered for " + model_type)
        if route is None:
            if resolver is None:
                raise ValueError("exact route resolver is required for " + model_type)
            if _accepts_options(resolver):
                route = resolver(copy.deepcopy(config), directory,
                                 options=copy.deepcopy(options or {}))
            else:
                route = resolver(copy.deepcopy(config), directory)
        if not isinstance(route, ModelRoute) or route.model_type != model_type:
            raise ValueError("requested route model_type differs from checkpoint identity")
        return ModelManifest(directory, model_type, architecture, route, copy.deepcopy(config),
                             hashlib.sha256(raw).hexdigest())

    def open(self, model_directory: str | Path, *, route: ModelRoute | None = None,
             **options: Any) -> "UniversalSession":
        manifest = self.inspect(model_directory, route=route, options=options)
        # An explicit route selects a tuple, but it does not bind that tuple
        # to this checkpoint's tensor inventory. Require the registered
        # resolver on every open, not only when the caller asks us to choose
        # a default route. Otherwise a custom opener with an explicit route
        # could reach its loader without ever resolving source provenance.
        with self._lock:
            resolver = self._route_resolvers.get(manifest.model_type)
        if resolver is None:
            raise ValueError("exact route resolver is required to bind checkpoint inventory for " +
                             manifest.model_type)
        # Admission failures propagate unchanged; a missing helper must never
        # be reported as an ordinary route rejection.
        self._route_admission(manifest.route)
        if route is not None:
            # A canonical route row proves that a tuple is allowed in general;
            # it does not prove that this checkpoint has the tuple's source
            # dtype. Resolve the checkpoint inventory even for explicit
            # backend/format selections, and compare the immutable source
            # identity before any family owner is constructed.
            resolver_options = {}
            if _accepts_options(resolver):
                resolver_options["options"] = copy.deepcopy(options)
            if _accepts_route(resolver):
                resolver_options["route"] = manifest.route
            observed_route = resolver(copy.deepcopy(manifest.config),
                                      manifest.directory, **resolver_options)
            if (not isinstance(observed_route, ModelRoute) or
                    observed_route.model_type != manifest.model_type or
                    observed_route.family != manifest.route.family or
                    observed_route.source_format != manifest.route.source_format):
                observed_source = (observed_route.source_format
                                   if isinstance(observed_route, ModelRoute) else "unknown")
                raise ValueError("explicit route source format " +
                                 repr(manifest.route.source_format) +
                                 " differs from checkpoint inventory " + repr(observed_source))
        option_key = freeze_options(options)
        key = (str(manifest.directory), manifest.config_sha256, manifest.route.as_tuple(), option_key)
        with self._lock:
            existing = self._sessions.get(key)
            if existing is not None and existing.is_usable():
                return existing
            opener = self._adapters[manifest.model_type]
        owner = opener(manifest.directory, admission={
            "manifest": manifest, "route": manifest.route.as_dict(),
            "options": copy.deepcopy(options)})
        try:
            session = UniversalSession(manifest, owner)
        except BaseException as primary:
            close = getattr(owner, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as cleanup:
                    retain_cleanup_failure(
                        primary, close, cleanup,
                        "rejected universal owner cleanup remains retryable: ")
            raise
        with self._lock:
            previous = self._sessions.get(key)
            if previous is not None and previous.is_usable():
                losing = session
                session = previous
            else:
                losing = None
                if previous is not None and not previous.status()["closed"]:
                    self._retired.append(previous)
                self._sessions[key] = session
        if losing is not None:
            try:
                losing.close()
            except BaseException as primary:
                # Concurrent opens can construct the same warm-session key
                # before either owner is published. If this duplicate owner
                # cannot close, keep it reachable for engine/model teardown;
                # otherwise its physical resources are orphaned after this
                # stack unwinds.
                with self._lock:
                    if all(retired is not losing for retired in self._retired):
                        self._retired.append(losing)
                add_failure_note(
                    primary,
                    "duplicate universal owner retained for retry by close_model()/close()")
                raise
        return session

    def registered_model_types(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._adapters))

    def resolver_model_types(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._route_resolvers))

    def _close_pooled(self, keys: list[tuple[Any, ...]], *, root: str | None = None) -> None:
        """Close pooled sessions, removing each only after it closed."""
        primary = None
        with self._lock:
            retired = [session for session in self._retired
                       if root is None or str(session.manifest.directory) == root]
        for session in retired:
            try:
                session.close()
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    add_failure_note(primary, "additional universal session close failure: " + failure_detail(error))
                continue
            with self._lock:
                self._retired = [item for item in self._retired if item is not session]
        for key in keys:
            with self._lock:
                session = self._sessions.get(key)
            if session is None:
                continue
            try:
                session.close()
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    add_failure_note(primary, "additional universal session close failure: " + failure_detail(error))
                continue
            with self._lock:
                if self._sessions.get(key) is session:
                    del self._sessions[key]
        if primary is not None:
            raise primary

    def close_model(self, model_directory: str | Path, *, route: ModelRoute | None = None) -> None:
        root = str(Path(model_directory).resolve())
        with self._lock:
            keys = [key for key in self._sessions if key[0] == root and
                    (route is None or key[2] == route.as_tuple())]
        self._close_pooled(keys, root=root)

    def close_all(self) -> None:
        with self._lock:
            keys = list(self._sessions)
        self._close_pooled(keys)

    def pooled_sessions(self) -> int:
        """Sessions still owned by the pool, including retired close retries."""
        with self._lock:
            return len(self._sessions) + len(self._retired)


def require_universal_route_admission(route: ModelRoute) -> None:
    """Fail closed unless the exact six-field route is verified-admitted."""
    from .source_scope import require_source_matrix_admission, source_family_for_identity
    family = source_family_for_identity(route.model_type)
    if family != route.family:
        raise ValueError("route family differs from the canonical checkpoint identity family")
    try:
        require_source_matrix_admission(
            family=route.family, model_type=route.model_type, backend=route.backend,
            arithmetic=route.arithmetic, source_format=route.source_format,
            persisted_format=route.persisted_format)
    except KeyError as error:
        raise ValueError("route is absent from the canonical source matrix: " +
                         "/".join(route.as_tuple())) from error


class UniversalSession:
    """Serial, transactional lifecycle owner shared by every family adapter."""

    def __init__(self, manifest: ModelManifest, owner: Any, *,
                 receipt_max_age_seconds: float = DEFAULT_RECEIPT_MAX_AGE_SECONDS) -> None:
        if owner is None:
            raise ValueError("adapter opener returned no owned session")
        if not callable(getattr(owner, "snapshot", None)) or not callable(getattr(owner, "restore", None)):
            raise TypeError("universal owners must provide snapshot() and restore() for request rollback")
        if (type(receipt_max_age_seconds) not in (int, float) or
                not math.isfinite(receipt_max_age_seconds) or receipt_max_age_seconds <= 0):
            raise ValueError("positive finite receipt freshness bound required")
        self.manifest, self._owner = manifest, owner
        self.session_id = uuid4().hex
        self._lock = threading.RLock()
        # A plain lock deliberately protects request ownership across stream
        # yields: unlike RLock, it may safely be released by a cancellation
        # thread that closes the paused stream iterator.
        self._request_gate = threading.Lock()
        self._closed = False
        self._closing = False
        self._close_error: BaseException | None = None
        self._active_request_thread: int | None = None
        self._epoch = 0
        self._receipts: dict[str, UniversalReceipt] = {}
        # Transaction receipts are intentionally not exposed through
        # ``_receipts``. Keep failed releases independently reachable so a
        # later close can retry them instead of losing a family-owned cache
        # snapshot after the request frame unwinds.
        self._pending_receipt_releases: dict[int, Any] = {}
        # If rollback itself failed, the owner's request worker may still be
        # using its physical state. Such receipts are released only through
        # the owner's close contract, never raced with an active worker.
        self._deferred_receipt_releases: set[int] = set()
        self._receipt_max_age = float(receipt_max_age_seconds)

    # -- admission -----------------------------------------------------
    def is_usable(self) -> bool:
        with self._lock:
            return not self._closed and not self._closing

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("universal session is closed")
        if self._closing:
            raise RuntimeError("universal session close is pending retry")

    @property
    def owner(self) -> Any:
        return self._owner

    # -- request transactions -----------------------------------------
    def _owner_snapshot(self) -> Any:
        receipt = self._owner.snapshot()
        if receipt is None:
            raise RuntimeError("adapter owner returned no rollback receipt")
        return receipt

    def _release_owner_receipt(self, receipt: Any, primary: BaseException | None) -> None:
        try:
            release = getattr(self._owner, "release_snapshot", None)
            if not callable(release):
                return
            release(receipt)
        except BaseException as error:
            self._pending_receipt_releases[id(receipt)] = receipt
            self._closing = True
            if primary is not None:
                retain_cleanup_failure(
                    primary,
                    lambda receipt=receipt: self._retry_owner_receipt_release(receipt),
                    error,
                    "request rollback receipt release remains retryable: ")
            else:
                raise
        else:
            self._pending_receipt_releases.pop(id(receipt), None)

    def _retry_owner_receipt_release(self, receipt: Any) -> None:
        """Retry one failed physical receipt release without reopening the session."""
        with self._lock:
            key = id(receipt)
            if self._pending_receipt_releases.get(key) is not receipt:
                return
            release = getattr(self._owner, "release_snapshot", None)
            if not callable(release):
                raise RuntimeError("owner no longer supports retrying its retained snapshot release")
            release(receipt)
            self._pending_receipt_releases.pop(key, None)

    def _defer_owner_receipt_release(self, receipt: Any, primary: BaseException | None) -> None:
        """Retain a receipt for owner teardown when rollback left work in flight."""
        key = id(receipt)
        self._pending_receipt_releases[key] = receipt
        self._deferred_receipt_releases.add(key)
        self._closing = True
        if primary is not None:
            add_failure_note(primary, "rollback receipt retained until owner teardown")
            try:
                primary.cleanup = self.close
            except BaseException:
                pass

    def _rollback(self, receipt: Any, primary: BaseException) -> bool:
        try:
            self._owner.restore(receipt)
        except BaseException as error:
            # A failed rollback leaves owner state unknown; quarantine it.
            self._closing = True
            add_failure_note(primary, "request rollback failed; session quarantined: " + failure_detail(error))
            return False
        return True

    @_universal_request
    def generate(self, *args: Any, cancelled: Callable[[], bool] | None = None,
                 publish: Callable[[Any], None] | None = None, **kwargs: Any) -> Any:
        """Run one request atomically.

        The owner state is committed only when the request completes, is not
        cancelled, and ``publish`` (when given) accepts the result.  Any
        cancellation or failure restores the pre-dispatch snapshot first.
        """
        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be callable")
        if publish is not None and not callable(publish):
            raise TypeError("publish must be callable")
        with self._lock:
            self._check_open()
            if cancelled is not None and cancelled():
                raise GenerationCancelled("generation cancelled before adapter dispatch")
            generate = getattr(self._owner, "generate", None) or getattr(self._owner, "run", None)
            if not callable(generate):
                raise TypeError("registered adapter session lacks generate/run")
            receipt = self._owner_snapshot()
            primary: BaseException | None = None
            rollback_succeeded = True
            try:
                result = generate(*args, cancelled=cancelled, **kwargs)
                if cancelled is not None and cancelled():
                    raise GenerationCancelled("generation cancelled before universal publication")
                if publish is not None:
                    try:
                        publish(result)
                    except BaseException as error:
                        raise PublicationFailed("result publication failed") from error
                self._epoch += 1
                return result
            except BaseException as error:
                primary = error
                rollback_succeeded = self._rollback(receipt, error)
                cancelled_error = _as_cancellation(error, cancelled)
                if cancelled_error is not None:
                    primary = cancelled_error
                    raise cancelled_error from error
                raise
            finally:
                if rollback_succeeded:
                    self._release_owner_receipt(receipt, primary)
                else:
                    self._defer_owner_receipt_release(receipt, primary)

    @_universal_request
    def stream(self, *args: Any, cancelled: Callable[[], bool] | None = None,
               **kwargs: Any) -> Iterator[Any]:
        """Stream one request; roll back unless the consumer reaches the end.

        Cancellation is checked before dispatch and before every published
        event.  Cancellation, owner failure, or the consumer abandoning the
        stream restores the exact pre-request snapshot.
        """
        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be callable")
        with self._lock:
            self._check_open()
            if cancelled is not None and cancelled():
                raise GenerationCancelled("generation cancelled before adapter dispatch")
            stream = getattr(self._owner, "stream", None) or getattr(self._owner, "stream_generate", None)
            if not callable(stream):
                raise TypeError("registered adapter session lacks stream/stream_generate")
            receipt = self._owner_snapshot()
        primary: BaseException | None = None
        iterator = None
        completed = False
        rollback_succeeded = True
        cleanup_error: BaseException | None = None
        try:
            iterator = iter(stream(*args, cancelled=cancelled, **kwargs))
            for event in iterator:
                if cancelled is not None and cancelled():
                    raise GenerationCancelled("generation cancelled during universal stream")
                # Adapter stream wrappers publish this envelope only after
                # the worker has returned successfully. Treat it as the
                # commit boundary before yielding it: consumers commonly
                # stop as soon as they receive the final result and close
                # the iterator without asking for one more item. Waiting
                # for StopIteration in that case would incorrectly roll
                # back a request whose committed result was already sent.
                terminal_result = (
                    type(event) is dict and event.get("event") == "result"
                    and "result" in event
                )
                if terminal_result and not completed:
                    completed = True
                    self._epoch += 1
                yield event
            if cancelled is not None and cancelled():
                raise GenerationCancelled("generation cancelled before stream completion")
            if not completed:
                completed = True
                self._epoch += 1
        except BaseException as error:
            # GeneratorExit means the consumer abandoned the stream, not that
            # the backend operation itself failed. Keep it out of `primary`
            # so a backend close failure can be reported to the cancelling
            # caller after rollback completes.
            primary = None if isinstance(error, GeneratorExit) else error
            cancelled_error = _as_cancellation(error, cancelled)
            if cancelled_error is not None:
                primary = cancelled_error
                raise cancelled_error from error
            raise
        finally:
            if iterator is not None:
                try:
                    close = getattr(iterator, "close", None)
                    if callable(close):
                        close()
                except BaseException as error:
                    if primary is None:
                        primary = error
                        cleanup_error = error
                    else:
                        add_failure_note(primary, "stream iterator close failed: " + failure_detail(error))
            with self._lock:
                if not completed:
                    rollback_cause = primary or GenerationCancelled("stream abandoned before completion")
                    rollback_succeeded = self._rollback(receipt, rollback_cause)
                if rollback_succeeded:
                    self._release_owner_receipt(receipt, primary)
                else:
                    self._defer_owner_receipt_release(receipt, primary)
            if cleanup_error is not None:
                raise cleanup_error

    # -- recovery receipts ----------------------------------------------
    @_universal_lifecycle
    def snapshot(self) -> UniversalReceipt:
        with self._lock:
            self._check_open()
            owner_receipt = self._owner_snapshot()
            try:
                receipt = UniversalReceipt(
                    version=UNIVERSAL_RECEIPT_VERSION, receipt_id=uuid4().hex,
                    route=self.manifest.route.as_tuple(),
                    model_identity=self.manifest.model_identity,
                    session_id=self.session_id, epoch=self._epoch,
                    state_digest=_state_digest(owner_receipt),
                    created_monotonic=time.monotonic(),
                    max_age_seconds=self._receipt_max_age,
                    _owner_receipt=owner_receipt)
                self._receipts[receipt.receipt_id] = receipt
            except BaseException as primary:
                self._release_owner_receipt(owner_receipt, primary)
                raise
            return receipt

    def _validate_receipt(self, receipt: Any, *, allow_expired: bool = False) -> UniversalReceipt:
        if type(receipt) is not UniversalReceipt:
            raise ValueError("restore requires a universal receipt issued by this session")
        if receipt.version != UNIVERSAL_RECEIPT_VERSION:
            raise ValueError("restore receipt version is not supported")
        if receipt.route != self.manifest.route.as_tuple():
            raise ValueError("restore receipt belongs to a different route")
        if receipt.model_identity != self.manifest.model_identity:
            raise ValueError("restore receipt belongs to a different checkpoint")
        if receipt.session_id != self.session_id:
            raise ValueError("restore receipt belongs to a different session")
        if self._receipts.get(receipt.receipt_id) is not receipt:
            raise ValueError("restore receipt is unknown, released, or forged")
        if receipt.epoch > self._epoch:
            raise ValueError("restore receipt epoch is ahead of this session")
        if not allow_expired and time.monotonic() - receipt.created_monotonic > receipt.max_age_seconds:
            raise ValueError("restore receipt is stale")
        if _state_digest(receipt._owner_receipt) != receipt.state_digest:
            raise ValueError("restore receipt state digest no longer matches")
        return receipt

    @_universal_lifecycle
    def restore(self, receipt: Any) -> None:
        with self._lock:
            self._check_open()
            valid = self._validate_receipt(receipt)
            try:
                self._owner.restore(valid._owner_receipt)
            except BaseException as primary:
                # A family restore may have installed only part of its
                # physical cache before failing. Until that owner is torn
                # down, no later request or restore can safely assume which
                # physical boundary is live. Keep the public receipt retained
                # for close-time release and make cleanup reachable from the
                # original restore error.
                self._closing = True
                add_failure_note(primary,
                    "universal receipt restore failed; physical owner quarantined")
                try:
                    primary.cleanup = self.close
                except BaseException:
                    pass
                raise
            # Restoring is itself a committed state change; the epoch only
            # moves forward so receipts can never claim a future state.
            self._epoch += 1

    @_universal_lifecycle
    def release_snapshot(self, receipt: Any) -> None:
        with self._lock:
            # Freshness constrains restoration, not disposal of an exact owned
            # receipt. Expiration must not pin physical snapshots indefinitely.
            valid = self._validate_receipt(receipt, allow_expired=True) if not self._closed else None
            if valid is None:
                return
            self._release_owner_receipt(valid._owner_receipt, None)
            del self._receipts[valid.receipt_id]

    # -- status / lifecycle -------------------------------------------
    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"closed": self._closed, "closing": self._closing,
                    "close_error": None if self._close_error is None else type(self._close_error).__name__,
                    "session_id": self.session_id, "epoch": self._epoch,
                    "live_receipts": len(self._receipts),
                    "pending_cleanup_receipts": len(self._pending_receipt_releases),
                    "model_type": self.manifest.model_type,
                    "architecture": self.manifest.architecture,
                    "model_identity": self.manifest.model_identity,
                    "route": self.manifest.route.as_dict()}

    @_universal_lifecycle
    def close(self) -> None:
        """Close the owner; on failure the session stays retryable."""
        with self._lock:
            if self._closed:
                return
            self._closing = True
            primary: BaseException | None = None
            for receipt_id, receipt in list(self._receipts.items()):
                try:
                    self._release_owner_receipt(receipt._owner_receipt, None)
                    del self._receipts[receipt_id]
                except BaseException as error:
                    if primary is None:
                        primary = error
                    else:
                        add_failure_note(primary, "additional receipt release failure: " + failure_detail(error))
            registered = {id(receipt._owner_receipt) for receipt in self._receipts.values()}
            for receipt_id, owner_receipt in list(self._pending_receipt_releases.items()):
                if (receipt_id in registered or
                        receipt_id in self._deferred_receipt_releases):
                    continue
                try:
                    self._release_owner_receipt(owner_receipt, None)
                except BaseException as error:
                    if primary is None:
                        primary = error
                    else:
                        add_failure_note(primary, "additional transaction receipt release failure: " +
                                         failure_detail(error))
            if primary is None:
                try:
                    close = getattr(self._owner, "close", None)
                    if callable(close):
                        close()
                    elif self._deferred_receipt_releases:
                        raise RuntimeError(
                            "owner lacks close() required to retire deferred rollback receipts")
                except BaseException as error:
                    primary = error
            if primary is not None:
                self._close_error = primary
                raise primary
            self._close_error = None
            self._receipts.clear()
            self._pending_receipt_releases.clear()
            self._deferred_receipt_releases.clear()
            self._closed = True
            self._closing = False

    def __enter__(self) -> "UniversalSession":
        with self._lock:
            self._check_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            if exc is None:
                raise
            add_failure_note(exc, "universal session close failed: " + failure_detail(cleanup))
            try:
                exc.cleanup = self.close
            except BaseException:
                pass


__all__ = ["DEFAULT_RECEIPT_MAX_AGE_SECONDS", "GenerationCancelled", "ModelManifest",
           "ModelRoute", "PublicationFailed", "UNIVERSAL_RECEIPT_VERSION", "UniversalAdapter",
           "UniversalEngine", "UniversalReceipt", "UniversalSession", "freeze_options",
           "require_universal_route_admission"]
