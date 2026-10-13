"""Backend contract and reference kernels for routed SwiGLU experts.

The module deliberately separates computation from expert storage. A payload
source only needs to expose an ``acquire(key)`` context manager, which keeps the
kernel compatible with disk-backed, mapped, or resident stores without taking
ownership of their buffers.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import math
from numbers import Real
from typing import ContextManager, Iterable, Iterator, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


SUPPORTED_PAYLOAD_DTYPES = frozenset({"float32", "float16", "int8-row"})
REGISTERED_PAYLOAD_DTYPES = SUPPORTED_PAYLOAD_DTYPES | {"e8-lattice-caller-v1", "iq3-format6-98-v1", "int2-row", "int3-grouped-g64", "int4-row", "int4-grouped", "int4-grouped-g64", "mxfp4-e2m1-g32-ue8m0", "q8_0", "q8_1", "q4_0", "q4_1", "q5_0", "q5_1", "fp8-e4m3-b128-f32"}


@dataclass(frozen=True)
class KernelEvidence:
    """A provider-supplied fact supporting an implementation capability."""

    kind: str
    detail: str
    verified: bool = False

    def __post_init__(self) -> None:
        if not self.kind.strip() or not self.detail.strip():
            raise ValueError("kernel evidence requires a kind and detail")


@dataclass(frozen=True)
class KernelCapabilities:
    """Execution features exposed by one backend."""

    payload_dtypes: frozenset[str]
    batch_execution: bool
    deterministic: bool
    compiled: bool = False
    native_simd: bool = False

    def __post_init__(self) -> None:
        dtypes = frozenset(self.payload_dtypes)
        if not dtypes or not dtypes <= REGISTERED_PAYLOAD_DTYPES:
            raise ValueError("backend declares an unsupported payload dtype")
        object.__setattr__(self, "payload_dtypes", dtypes)
        if self.native_simd and not self.compiled:
            raise ValueError("native SIMD requires a compiled backend")


@dataclass(frozen=True)
class KernelBackendInfo:
    """Discoverable backend state used for selection and operator reporting."""

    name: str
    provider: str
    available: bool
    priority: int
    capabilities: KernelCapabilities
    evidence: tuple[KernelEvidence, ...] = ()
    unavailable_reason: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.provider.strip():
            raise ValueError("backend name and provider are required")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("backend priority must be an integer")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if not self.available and not self.unavailable_reason.strip():
            raise ValueError("an unavailable backend must explain why")
        if self.capabilities.native_simd and not any(item.verified for item in self.evidence):
            raise ValueError("native SIMD claims require verified provider evidence")


@runtime_checkable
class ExpertPayloadSource(Protocol):
    """Lease payload bytes for an expert key."""

    def acquire(self, key: str) -> ContextManager[memoryview]: ...


@runtime_checkable
class RoutedSwiGLUKernelBackend(Protocol):
    """Compute explicitly routed SwiGLU layers without changing router output."""

    def info(self) -> KernelBackendInfo: ...

    def execute(
        self,
        source: ExpertPayloadSource,
        inputs,
        routes: Sequence[tuple[str, float]],
        *,
        hidden_size: int,
        intermediate_size: int,
        dtype: str,
    ) -> np.ndarray: ...

    def execute_batch(
        self,
        source: ExpertPayloadSource,
        inputs,
        route_batch: Sequence[Sequence[tuple[str, float]]],
        *,
        hidden_size: int,
        intermediate_size: int,
        dtype: str,
    ) -> np.ndarray: ...


class MappingPayloadSource:
    """Small resident payload source useful for tests and embedding runtimes."""

    def __init__(self, payloads: Mapping[str, bytes | bytearray | memoryview]) -> None:
        self._payloads = dict(payloads)

    @contextmanager
    def acquire(self, key: str) -> Iterator[memoryview]:
        try:
            payload = self._payloads[key]
        except KeyError:
            raise KeyError(key) from None
        view = memoryview(payload).toreadonly()
        try:
            yield view
        finally:
            view.release()


class NumPyRoutedSwiGLUBackend:
    """Portable correctness backend using NumPy FP32 accumulation."""

    NAME = "numpy-reference"

    def info(self) -> KernelBackendInfo:
        return KernelBackendInfo(
            name=self.NAME,
            provider="openagent",
            available=True,
            priority=-10_000,
            capabilities=KernelCapabilities(
                payload_dtypes=SUPPORTED_PAYLOAD_DTYPES,
                batch_execution=True,
                deterministic=True,
            ),
            evidence=(KernelEvidence("implementation", "NumPy reference operations", verified=True),),
        )

    def execute(
        self,
        source: ExpertPayloadSource,
        inputs,
        routes: Sequence[tuple[str, float]],
        *,
        hidden_size: int,
        intermediate_size: int,
        dtype: str,
    ) -> np.ndarray:
        hidden_size, intermediate_size, expected_size = _validate_configuration(
            source, hidden_size, intermediate_size, dtype
        )
        vector = _finite_array(inputs, (hidden_size,), "inputs")
        normalized = _validate_routes(routes, "routes")
        output = np.zeros(hidden_size, dtype=np.float32)
        for key, weight in normalized:
            with _lease(source, key) as payload:
                matrices = _decode_payload(payload, hidden_size, intermediate_size, dtype, expected_size)
                output += np.float32(weight) * _single_expert(vector, *matrices)
        return _finite_output(output)

    def execute_batch(
        self,
        source: ExpertPayloadSource,
        inputs,
        route_batch: Sequence[Sequence[tuple[str, float]]],
        *,
        hidden_size: int,
        intermediate_size: int,
        dtype: str,
    ) -> np.ndarray:
        hidden_size, intermediate_size, expected_size = _validate_configuration(
            source, hidden_size, intermediate_size, dtype
        )
        batch = _finite_matrix(inputs, hidden_size)
        try:
            route_rows = list(route_batch)
        except TypeError:
            raise TypeError("route_batch must be an iterable of route lists") from None
        if len(route_rows) != batch.shape[0]:
            raise ValueError("one route list is required for each input row")

        grouped: OrderedDict[str, list[tuple[int, float]]] = OrderedDict()
        for row, routes in enumerate(route_rows):
            for key, weight in _validate_routes(routes, f"route_batch[{row}]"):
                grouped.setdefault(key, []).append((row, weight))

        output = np.zeros((batch.shape[0], hidden_size), dtype=np.float32)
        for key, assignments in grouped.items():
            with _lease(source, key) as payload:
                matrices = _decode_payload(payload, hidden_size, intermediate_size, dtype, expected_size)
                rows = np.fromiter((row for row, _ in assignments), dtype=np.intp)
                weights = np.fromiter((weight for _, weight in assignments), dtype=np.float32)[:, None]
                output[rows] += weights * _batch_expert(batch[rows], *matrices)
        return _finite_output(output)


class ExpertKernelRegistry:
    """Register optional kernels and select one with an explicit safe fallback."""

    def __init__(self, backends: Iterable[RoutedSwiGLUKernelBackend] = ()) -> None:
        self._backends: dict[str, RoutedSwiGLUKernelBackend] = {}
        self.register(NumPyRoutedSwiGLUBackend())
        from .packed_expert import PackedRoutedSwiGLUBackend

        self.register(PackedRoutedSwiGLUBackend())
        from .q8_0_expert import Q8_0ExpertBackend

        self.register(Q8_0ExpertBackend())
        from .q8_1_expert import Q8_1ExpertBackend

        self.register(Q8_1ExpertBackend())
        from .q4_0_expert import Q4_0ExpertBackend

        self.register(Q4_0ExpertBackend())
        from .q4_1_expert import Q4_1ExpertBackend

        self.register(Q4_1ExpertBackend())
        from .q5_0_expert import Q5_0ExpertBackend

        self.register(Q5_0ExpertBackend())
        from .q5_1_expert import Q5_1ExpertBackend

        self.register(Q5_1ExpertBackend())
        from .grouped_int4_expert import GroupedInt4ExpertBackend
        from .mxfp4_expert import MXFP4ExpertBackend

        self.register(GroupedInt4ExpertBackend())
        self.register(MXFP4ExpertBackend())
        from .lowbit_expert import LowBitExpertBackend, INT2, INT3

        self.register(LowBitExpertBackend(INT2))
        self.register(LowBitExpertBackend(INT3))
        for backend in backends:
            self.register(backend)

    def register(self, backend: RoutedSwiGLUKernelBackend, *, replace: bool = False) -> None:
        if not isinstance(backend, RoutedSwiGLUKernelBackend):
            raise TypeError("backend does not implement the routed SwiGLU protocol")
        info = backend.info()
        if not isinstance(info, KernelBackendInfo):
            raise TypeError("backend info must be KernelBackendInfo")
        if info.name == NumPyRoutedSwiGLUBackend.NAME and not isinstance(backend, NumPyRoutedSwiGLUBackend):
            raise ValueError("the reference backend name is reserved")
        if info.name in self._backends and not replace:
            raise ValueError(f"backend already registered: {info.name}")
        self._backends[info.name] = backend

    def unregister(self, name: str) -> None:
        if name == NumPyRoutedSwiGLUBackend.NAME:
            raise ValueError("the deterministic reference fallback cannot be removed")
        try:
            del self._backends[name]
        except KeyError:
            raise KeyError(name) from None

    def close(self) -> None:
        """Release backend resources owned by this registry."""
        errors = []
        for backend in self._backends.values():
            close = getattr(backend, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise RuntimeError("expert backend cleanup failed") from errors[0]

    def describe(self) -> tuple[KernelBackendInfo, ...]:
        return tuple(
            sorted((backend.info() for backend in self._backends.values()), key=lambda item: item.name)
        )

    def resolve(
        self,
        *,
        dtype: str,
        batch: bool = False,
        preferred: str | None = None,
        allow_fallback: bool = True,
    ) -> RoutedSwiGLUKernelBackend:
        if dtype not in REGISTERED_PAYLOAD_DTYPES:
            raise ValueError(f"unsupported expert payload dtype: {dtype}")
        if dtype in {"e8-lattice-caller-v1", "iq3-format6-98-v1"}:
            if preferred is None:
                raise ValueError("lattice execution requires an explicit contract-bound backend")
            # Another lattice backend may use a different admitted grid.
            allow_fallback = False
        if preferred is not None and preferred not in self._backends:
            if not allow_fallback:
                raise KeyError(preferred)
        elif preferred is not None:
            backend = self._backends[preferred]
            if _supports(backend.info(), dtype, batch):
                return backend
            if not allow_fallback:
                raise RuntimeError(f"requested backend is unavailable or incompatible: {preferred}")

        candidates = [
            backend for name, backend in self._backends.items()
            if name != NumPyRoutedSwiGLUBackend.NAME and _supports(backend.info(), dtype, batch)
        ]
        if candidates:
            return min(candidates, key=lambda item: (-item.info().priority, item.info().name))
        if not allow_fallback:
            raise RuntimeError(f"no available explicit backend supports {dtype}")
        fallback = self._backends[NumPyRoutedSwiGLUBackend.NAME]
        if not _supports(fallback.info(), dtype, batch):
            raise RuntimeError(f"no available backend supports {dtype}")
        return fallback


def _supports(info: KernelBackendInfo, dtype: str, batch: bool) -> bool:
    return (
        info.available
        and dtype in info.capabilities.payload_dtypes
        and (not batch or info.capabilities.batch_execution)
    )


def _validate_configuration(source, hidden_size, intermediate_size, dtype) -> tuple[int, int, int]:
    if not isinstance(source, ExpertPayloadSource):
        raise TypeError("source must implement acquire(key)")
    for value, name in ((hidden_size, "hidden_size"), (intermediate_size, "intermediate_size")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if dtype not in SUPPORTED_PAYLOAD_DTYPES:
        raise ValueError(f"unsupported expert payload dtype: {dtype}")
    width = hidden_size * intermediate_size
    size = 3 * width + 4 * (2 * intermediate_size + hidden_size) if dtype == "int8-row" else 3 * width * (4 if dtype == "float32" else 2)
    return hidden_size, intermediate_size, size


def _finite_array(value, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        raise TypeError(f"{label} must be numeric") from None
    if array.dtype.kind not in "biuf" or array.shape != shape:
        raise ValueError(f"{label} must have shape {shape} and contain real numbers")
    result = np.asarray(array, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must contain only finite values")
    return result


def _finite_matrix(value, hidden_size: int) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        raise TypeError("inputs must be numeric") from None
    if array.dtype.kind not in "biuf" or array.ndim != 2 or array.shape[1] != hidden_size:
        raise ValueError("inputs must be a real [batch, hidden_size] matrix")
    result = np.asarray(array, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("inputs must contain only finite values")
    return result


def _validate_routes(routes, label: str) -> list[tuple[str, float]]:
    try:
        values = list(routes)
    except TypeError:
        raise TypeError(f"{label} must be iterable") from None
    if not values:
        raise ValueError(f"{label} cannot be empty")
    normalized: list[tuple[str, float]] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(f"{label} entries must be (expert_key, weight) pairs")
        key, weight = item
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError(f"{label} must contain distinct non-empty expert keys")
        if isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(float(weight)) or weight < 0:
            raise ValueError(f"{label} weights must be finite and nonnegative")
        seen.add(key)
        normalized.append((key, float(weight)))
    return normalized


@contextmanager
def _lease(source: ExpertPayloadSource, key: str) -> Iterator[memoryview]:
    lease = source.acquire(key)
    if not hasattr(lease, "__enter__") or not hasattr(lease, "__exit__"):
        raise TypeError("source.acquire(key) must return a context manager")
    with lease as payload:
        try:
            view = memoryview(payload)
        except TypeError:
            raise TypeError("expert payload must support the buffer protocol") from None
        if not view.c_contiguous:
            view.release()
            raise ValueError("expert payload must be C-contiguous")
        try:
            yield view
        finally:
            view.release()


def _decode_payload(payload: memoryview, hidden: int, intermediate: int, dtype: str, expected: int):
    if payload.nbytes != expected:
        raise ValueError(f"expert payload size mismatch: expected {expected}, received {payload.nbytes}")
    width = hidden * intermediate
    if dtype == "int8-row":
        values = np.frombuffer(payload, dtype=np.int8, count=3 * width)
        scales = np.frombuffer(payload, dtype="<f4", count=2 * intermediate + hidden, offset=3 * width)
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("row-wise INT8 scales must be finite and positive")
        gate = values[:width].reshape(intermediate, hidden)
        up = values[width:2 * width].reshape(intermediate, hidden)
        down = values[2 * width:].reshape(hidden, intermediate)
        return gate, up, down, scales[:intermediate], scales[intermediate:2 * intermediate], scales[2 * intermediate:]
    values = np.frombuffer(payload, dtype="<f4" if dtype == "float32" else "<f2", count=3 * width)
    if not np.isfinite(values).all():
        raise ValueError("floating-point expert payload contains non-finite values")
    one = np.float32(1.0)
    return (
        values[:width].reshape(intermediate, hidden),
        values[width:2 * width].reshape(intermediate, hidden),
        values[2 * width:].reshape(hidden, intermediate),
        one, one, one,
    )


def _single_expert(x, gate_matrix, up_matrix, down_matrix, gate_scale, up_scale, down_scale):
    gate = (gate_matrix @ x) * gate_scale
    up = (up_matrix @ x) * up_scale
    activated = gate * np.exp(-np.logaddexp(0, -gate)) * up
    return (down_matrix @ activated) * down_scale


def _batch_expert(x, gate_matrix, up_matrix, down_matrix, gate_scale, up_scale, down_scale):
    gate = (x @ gate_matrix.T) * gate_scale
    up = (x @ up_matrix.T) * up_scale
    activated = gate * np.exp(-np.logaddexp(0, -gate)) * up
    return (activated @ down_matrix.T) * down_scale


def _finite_output(output: np.ndarray) -> np.ndarray:
    result = np.asarray(output, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ArithmeticError("expert computation produced non-finite output")
    return result


__all__ = [
    "ExpertKernelRegistry",
    "ExpertPayloadSource",
    "KernelBackendInfo",
    "KernelCapabilities",
    "KernelEvidence",
    "MappingPayloadSource",
    "NumPyRoutedSwiGLUBackend",
    "RoutedSwiGLUKernelBackend",
    "SUPPORTED_PAYLOAD_DTYPES",
]
