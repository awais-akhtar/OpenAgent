"""OpenVINO execution backend for disk-described routed experts."""

from __future__ import annotations

import hashlib
import math
import threading
from uuid import uuid4
from numbers import Real
from collections import OrderedDict
from itertools import islice
from typing import Any, Sequence

import numpy as np

from .expert_kernels import (
    ExpertPayloadSource,
    KernelBackendInfo,
    KernelCapabilities,
    KernelEvidence,
)


class OpenVINOExpertBackend:
    """Keep compiled fixed-batch expert graphs resident in the native worker.

    The payload source must expose ``describe_region(key)``. This limits the
    pipe protocol to small activation vectors while the worker reopens the
    already validated local byte range for each first-time expert load.

    Capacity budgets estimated FP32 graph constants, not total native memory.
    Compilation, infer requests, device allocations and temporary buffers are
    additional and are not bounded by this backend.
    """

    def __init__(
        self,
        client: Any,
        *,
        device: str = "GPU",
        capacity_bytes: int = 2 * 1024**3,
    ) -> None:
        self.client = client
        self.device = str(device).strip().upper()
        if not self.device:
            raise ValueError("OpenVINO expert device is required")
        if type(capacity_bytes) is not int or capacity_bytes <= 0:
            raise ValueError("OpenVINO expert capacity must be positive")
        self.name = f"openvino-{self.device.lower()}"
        self.capacity_bytes = int(capacity_bytes)
        self._resident: OrderedDict[object, tuple[str, int]] = OrderedDict()
        self._resident_bytes = 0
        self._lock = threading.RLock()
        self._closed = False
        self._namespace = uuid4().hex[:16]
        self._worker = getattr(client, "_process", None)

    def info(self) -> KernelBackendInfo:
        available = bool(not self._closed and self.client is not None and getattr(self.client, "available", lambda: False)())
        return KernelBackendInfo(
            name=self.name,
            provider="openvino",
            available=available,
            priority=500,
            capabilities=KernelCapabilities(
                payload_dtypes=frozenset({"float32", "float16", "int8-row"}),
                batch_execution=True,
                deterministic=True,
                compiled=True,
            ),
            evidence=(
                KernelEvidence(
                    "implementation",
                    "OpenVINO graph compilation in the isolated local hardware worker",
                    verified=False,
                ),
            ),
            unavailable_reason="isolated OpenVINO runtime is not installed" if not available else "",
        )

    def execute(
        self, source: ExpertPayloadSource, inputs,
        routes: Sequence[tuple[str, float]], *, hidden_size: int,
        intermediate_size: int, dtype: str,
    ) -> np.ndarray:
        with self._lock:
            if self._closed:
                raise RuntimeError("OpenVINO expert backend is closed")
            self._sync_worker()
            return self._execute_locked(
                source, inputs, routes, hidden_size=hidden_size,
                intermediate_size=intermediate_size, dtype=dtype,
            )

    def _sync_worker(self) -> None:
        worker = getattr(self.client, "_process", None)
        if worker is not self._worker or (worker is not None and worker.poll() is not None):
            self._resident.clear()
            self._resident_bytes = 0
        self._worker = worker

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._sync_worker()
            error = None
            for key, (expert_id, size) in list(self._resident.items()):
                try:
                    self.client.unload(expert_id)
                except Exception as exc:
                    error = exc
                else:
                    del self._resident[key]
                    self._resident_bytes -= size
            if error is not None:
                raise error

    def _execute_locked(
        self,
        source: ExpertPayloadSource,
        inputs,
        routes: Sequence[tuple[str, float]],
        *,
        hidden_size: int,
        intermediate_size: int,
        dtype: str,
    ) -> np.ndarray:
        if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0
               for n in (hidden_size, intermediate_size)):
            raise ValueError("expert dimensions must be positive integers")
        if dtype not in {"float32", "float16", "int8-row"}:
            raise ValueError("unsupported expert payload dtype")
        vector = np.asarray(inputs, dtype=np.float32)
        if vector.shape != (hidden_size,) or not np.isfinite(vector).all():
            raise ValueError("inputs must be one finite hidden-size vector")
        route_values = self._validate_routes(routes)
        if not route_values:
            raise ValueError("routes cannot be empty")
        output = np.zeros(hidden_size, dtype=np.float32)
        seen: set[str] = set()
        for key, raw_weight in route_values:
            if not isinstance(key, str) or not key or key in seen:
                raise ValueError("routes must contain distinct non-empty expert keys")
            if isinstance(raw_weight, bool) or not isinstance(raw_weight, Real):
                raise ValueError("route weights must be finite and nonnegative")
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("route weights must be finite and nonnegative")
            seen.add(key)
            if weight == 0:
                continue
            expert_id = self._ensure_resident(
                source, key, hidden_size=hidden_size,
                intermediate_size=intermediate_size, dtype=dtype,
            )
            result = self.client.run_expert(expert_id, vector.tolist())
            value = np.asarray(result.get("output"), dtype=np.float32)
            if value.shape != (hidden_size,) or not np.isfinite(value).all():
                raise ArithmeticError("OpenVINO expert returned an invalid output")
            output += np.float32(weight) * value
        if not np.isfinite(output).all():
            raise ArithmeticError("weighted expert output overflowed")
        return output

    @staticmethod
    def _validate_routes(routes) -> list[tuple[str, float]]:
        values = list(islice(iter(routes), 129))
        if not values or len(values) > 128:
            raise ValueError("each token requires between one and 128 routes")
        checked = []
        seen = set()
        for route in values:
            if not isinstance(route, (tuple, list)) or len(route) != 2:
                raise ValueError("each route must contain an expert key and weight")
            key, weight = route
            if not isinstance(key, str) or not key or key in seen:
                raise ValueError("routes must contain distinct non-empty expert keys")
            if isinstance(weight, bool) or not isinstance(weight, Real):
                raise ValueError("route weights must be finite and nonnegative")
            weight = float(weight)
            if not math.isfinite(weight) or not 0 <= weight <= np.finfo(np.float32).max:
                raise ValueError("route weights must fit finite nonnegative FP32")
            seen.add(key)
            checked.append((key, weight))
        return checked

    def execute_batch(
        self, source: ExpertPayloadSource, inputs, route_batch, *,
        hidden_size: int, intermediate_size: int, dtype: str,
    ) -> np.ndarray:
        with self._lock:
            if self._closed:
                raise RuntimeError("OpenVINO expert backend is closed")
            if any(type(n) is not int or n <= 0 for n in (hidden_size, intermediate_size)):
                raise ValueError("expert dimensions must be positive integers")
            if dtype not in {"float32", "float16", "int8-row"}:
                raise ValueError("unsupported expert payload dtype")
            vectors = np.asarray(inputs, dtype=np.float32)
            if (vectors.ndim != 2 or vectors.shape[1] != hidden_size
                    or not 1 <= vectors.shape[0] <= 64 or vectors.size > 65536
                    or not np.isfinite(vectors).all()):
                raise ValueError("expert batch must contain one to 64 finite bounded vectors")
            route_rows = list(islice(iter(route_batch), 65))
            if len(route_rows) != vectors.shape[0]:
                raise ValueError("route rows must match the input batch")
            checked = [self._validate_routes(row) for row in route_rows]
            self._sync_worker()
            output = np.zeros_like(vectors)
            # Group within each route slot to preserve every row's summation order.
            for slot in range(max(map(len, checked))):
                groups = OrderedDict()
                for row, values in enumerate(checked):
                    if slot < len(values) and values[slot][1] != 0:
                        key, weight = values[slot]
                        groups.setdefault(key, []).append((row, weight))
                for key, members in groups.items():
                    rows = [row for row, _ in members]
                    expert_id = self._ensure_resident(
                        source, key, hidden_size=hidden_size,
                        intermediate_size=intermediate_size, dtype=dtype,
                        batch_size=len(rows),
                    )
                    if len(rows) == 1:
                        result = self.client.run_expert(expert_id, vectors[rows[0]].tolist())
                        values = np.asarray(result.get("output"), dtype=np.float32)[None, :]
                    else:
                        result = self.client.run_expert_batch(expert_id, vectors[rows].tolist())
                        values = np.asarray(result.get("output"), dtype=np.float32)
                    if values.shape != (len(rows), hidden_size) or not np.isfinite(values).all():
                        raise ArithmeticError("OpenVINO expert returned an invalid batch")
                    for index, (row, weight) in enumerate(members):
                        output[row] += np.float32(weight) * values[index]
            if not np.isfinite(output).all():
                raise ArithmeticError("weighted expert batch overflowed")
            return output

    def _ensure_resident(
        self,
        source: ExpertPayloadSource,
        key: str,
        *,
        hidden_size: int,
        intermediate_size: int,
        dtype: str,
        batch_size: int = 1,
    ) -> str:
        describe = getattr(source, "describe_region", None)
        if not callable(describe):
            raise TypeError("OpenVINO experts require a disk source with describe_region(key)")
        descriptor = describe(key)
        signature = hashlib.sha256(
            (
                f"{descriptor.path}:{descriptor.offset}:{descriptor.size}:{descriptor.sha256}:"
                f"{hidden_size}:{intermediate_size}:{dtype}:{self.device}:{batch_size}"
            ).encode("utf-8")
        ).hexdigest()
        expert_id = f"expert-{self._namespace}-{hashlib.sha256(key.encode()).hexdigest()[:16]}-{signature[:32]}"
        constant_bytes = 3 * hidden_size * intermediate_size * 4
        resident_key = key if batch_size == 1 else (key, batch_size)
        resident = self._resident.get(resident_key)
        if resident is not None and resident[0] == expert_id:
            self._resident.move_to_end(resident_key)
            return expert_id
        if constant_bytes > self.capacity_bytes:
            raise MemoryError("one expert exceeds the configured OpenVINO residency budget")
        if resident is not None:
            old_id, old_size = self._resident[resident_key]
            self.client.unload(old_id)
            del self._resident[resident_key]
            self._resident_bytes -= old_size
        while self._resident and self._resident_bytes + constant_bytes > self.capacity_bytes:
            evicted_key = next(iter(self._resident))
            evicted_id, evicted_size = self._resident[evicted_key]
            self.client.unload(evicted_id)
            del self._resident[evicted_key]
            self._resident_bytes -= evicted_size
        self.client.load_expert(
            expert_id=expert_id,
            path=descriptor.path,
            offset=descriptor.offset,
            size=descriptor.size,
            sha256=descriptor.sha256,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            device=self.device,
            **({"batch_size": batch_size} if batch_size != 1 else {}),
        )
        self._sync_worker()
        self._resident[resident_key] = (expert_id, constant_bytes)
        self._resident_bytes += constant_bytes
        return expert_id


__all__ = ["OpenVINOExpertBackend"]
