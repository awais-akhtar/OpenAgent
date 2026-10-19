"""One user-facing engine boundary for all exact family adapters."""
from __future__ import annotations

import copy
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol


class UniversalAdapter(Protocol):
    def open(self, model_directory: Path, *, admission: dict[str, Any]) -> Any: ...


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


@dataclass(frozen=True)
class ModelManifest:
    directory: Path
    model_type: str
    architecture: str
    route: ModelRoute
    config: dict[str, Any]


class UniversalEngine:
    """Shared launcher and exact adapter admission boundary."""

    def __init__(self) -> None:
        self._adapters: dict[str, Callable[..., Any]] = {}
        self._route_resolvers: dict[str, Callable[[dict[str, Any], Path], ModelRoute]] = {}
        self._sessions: dict[tuple[str, tuple[str, ...]], UniversalSession] = {}
        self._lock = threading.RLock()

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

    def inspect(self, model_directory: str | Path, *, route: ModelRoute | None = None) -> ModelManifest:
        directory = Path(model_directory).resolve(strict=True)
        config_path = directory / "config.json"
        if not config_path.is_file():
            raise ValueError("universal admission requires config.json")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid model config for universal admission") from error
        if type(config) is not dict:
            raise ValueError("model config must be an object")
        model_type = config.get("model_type")
        architecture = config.get("architectures", [None])[0]
        if type(model_type) is not str or not model_type:
            raise ValueError("model config lacks exact model_type")
        if type(architecture) is not str or not architecture:
            architecture = model_type
        if route is None:
            with self._lock:
                resolver = self._route_resolvers.get(model_type)
            if resolver is None:
                raise ValueError("exact route resolver is required for " + model_type)
            route = resolver(copy.deepcopy(config), directory)
        if not isinstance(route, ModelRoute) or route.model_type != model_type:
            raise ValueError("requested route model_type differs from checkpoint identity")
        with self._lock:
            if model_type not in self._adapters:
                raise ValueError("no exact universal adapter registered for " + model_type)
        return ModelManifest(directory, model_type, architecture, route, copy.deepcopy(config))

    def open(self, model_directory: str | Path, *, route: ModelRoute | None = None, **options: Any) -> "UniversalSession":
        manifest = self.inspect(model_directory, route=route)
        key = (str(manifest.directory), tuple(manifest.route.as_dict().values()))
        with self._lock:
            existing = self._sessions.get(key)
            if existing is not None and not existing.status()["closed"]:
                return existing
            opener = self._adapters[manifest.model_type]
        try:
            owner = opener(manifest.directory, admission={"manifest": manifest,
                "route": manifest.route.as_dict(), **options})
        except BaseException:
            # The adapter owns any resources it staged before returning only
            # after a successful open; failed construction must not leak a
            # partially returned owner through the universal boundary.
            raise
        session = UniversalSession(manifest, owner)
        with self._lock:
            previous = self._sessions.get(key)
            if previous is not None and not previous.status()["closed"]:
                session.close()
                return previous
            self._sessions[key] = session
        return session

    def registered_model_types(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._adapters))

    def close_model(self, model_directory: str | Path, *, route: ModelRoute | None = None) -> None:
        root = str(Path(model_directory).resolve())
        with self._lock:
            keys = [key for key in self._sessions if key[0] == root and
                    (route is None or key[1] == tuple(route.as_dict().values()))]
            sessions = [self._sessions.pop(key) for key in keys]
        for session in sessions:
            session.close()

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        primary = None
        for session in sessions:
            try:
                session.close()
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    primary.add_note("additional universal session close failure: " + repr(error))
        if primary is not None:
            raise primary


class UniversalSession:
    """Serial lifecycle owner shared by every family-specific adapter."""

    def __init__(self, manifest: ModelManifest, owner: Any) -> None:
        if owner is None:
            raise ValueError("adapter opener returned no owned session")
        self.manifest, self._owner = manifest, owner
        self._lock, self._closed = threading.RLock(), False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("universal session is closed")

    def generate(self, *args: Any, cancelled: Callable[[], bool] | None = None, **kwargs: Any) -> Any:
        with self._lock:
            self._check_open()
            if cancelled is not None and cancelled():
                raise RuntimeError("generation cancelled before adapter dispatch")
            generate = getattr(self._owner, "generate", None) or getattr(self._owner, "run", None)
            if not callable(generate):
                raise TypeError("registered adapter session lacks generate/run")
            result = generate(*args, cancelled=cancelled, **kwargs)
            if cancelled is not None and cancelled():
                raise RuntimeError("generation cancelled before universal publication")
            return result

    def stream(self, *args: Any, cancelled: Callable[[], bool] | None = None, **kwargs: Any) -> Iterator[Any]:
        self._lock.acquire()
        iterator = None
        try:
            self._check_open()
            stream = getattr(self._owner, "stream", None) or getattr(self._owner, "stream_generate", None)
            if not callable(stream):
                raise TypeError("registered adapter session lacks stream/stream_generate")
            iterator = iter(stream(*args, cancelled=cancelled, **kwargs))
            try:
                for event in iterator:
                    if cancelled is not None and cancelled():
                        raise RuntimeError("generation cancelled during universal stream")
                    yield event
            finally:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
        finally:
            self._lock.release()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"closed": self._closed, "model_type": self.manifest.model_type,
                    "architecture": self.manifest.architecture,
                    "route": self.manifest.route.as_dict()}

    def snapshot(self) -> Any:
        with self._lock:
            self._check_open()
            snapshot = getattr(self._owner, "snapshot", None)
            if not callable(snapshot):
                raise TypeError("adapter session lacks snapshot")
            return snapshot()

    def restore(self, receipt: Any) -> None:
        with self._lock:
            self._check_open()
            restore = getattr(self._owner, "restore", None)
            if not callable(restore):
                raise TypeError("adapter session lacks restore")
            restore(receipt)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            close = getattr(self._owner, "close", None)
            if callable(close):
                close()
            self._closed = True

    def __enter__(self) -> "UniversalSession":
        self._check_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
