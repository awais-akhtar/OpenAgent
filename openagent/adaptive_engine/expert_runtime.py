"""Validated sparse-expert manifests and adaptive routed execution."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import os
import tempfile
from pathlib import Path
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

from .expert_store import DiskExpertStore, ExpertRegion, routed_swiglu, routed_swiglu_batch
from .expert_kernels import ExpertKernelRegistry


MANIFEST_FORMAT = "openagent.sparse-experts.v1"


def _unique_fields(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate sparse metadata field: '+key)
        result[key] = value
    return result


@dataclass(frozen=True)
class ExpertLayer:
    """The exact on-disk layout and compute contract for one MoE layer."""

    id: str
    hidden_size: int
    intermediate_size: int
    experts_used: int
    dtype: str
    activation: str
    regions: Mapping[str, ExpertRegion]
    contract: Mapping[str, Any] | None = None

    def validate(self) -> None:
        for name in ("hidden_size", "intermediate_size", "experts_used"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if not self.id or "/" in self.id:
            raise ValueError("layer id must be non-empty and cannot contain '/'")
        if self.hidden_size <= 0 or self.intermediate_size <= 0:
            raise ValueError("layer dimensions must be positive")
        if self.experts_used <= 0 or self.experts_used > len(self.regions):
            raise ValueError("experts_used must fit the declared expert set")
        if self.dtype not in {"float16", "float32", "int8-row", "int2-row", "int3-grouped-g64", "int4-row", "int4-grouped-g64", "mxfp4-e2m1-g32-ue8m0", "q8_0", "q8_1", "q4_0", "q4_1", "q5_0", "q5_1", "fp8-e4m3-b128-f32", "e8-lattice-caller-v1", "iq3-format6-98-v1"}:
            raise ValueError("unsupported expert payload dtype")
        if self.activation != "swiglu":
            raise ValueError("only the swiglu payload contract is executable")
        width = self.hidden_size * self.intermediate_size
        expected = (
            3 * width + 4 * (2 * self.intermediate_size + self.hidden_size)
            if self.dtype == "int8-row"
            else 3 * width * (2 if self.dtype == "float16" else 4)
        )
        if self.dtype == "e8-lattice-caller-v1":
            from .e8_lattice_expert import payload_size, validate_contract_metadata

            validate_contract_metadata(self.contract)
            expected = payload_size(self.hidden_size, self.intermediate_size)
        elif self.dtype == "iq3-format6-98-v1":
            from .iq3_lattice import payload_size, validate_contract_metadata
            validate_contract_metadata(self.contract)
            expected = (2*payload_size(self.intermediate_size, self.hidden_size)+
                        payload_size(self.hidden_size, self.intermediate_size))
        elif self.contract is not None:
            raise ValueError("this expert dtype does not accept a lattice contract")
        if self.dtype in {"int2-row", "int3-grouped-g64"}:
            from .lowbit_expert import payload_size

            expected = payload_size(self.hidden_size, self.intermediate_size, dtype=self.dtype)
        elif self.dtype == "int4-row":
            from .packed_expert import payload_size

            expected = payload_size(self.hidden_size, self.intermediate_size)
        elif self.dtype == "q8_0":
            from .q8_0_expert import payload_size

            expected = payload_size(self.hidden_size, self.intermediate_size)
        elif self.dtype == "q8_1":
            from .q8_1_expert import payload_size

            expected = payload_size(self.hidden_size, self.intermediate_size)
        elif self.dtype == "q4_0":
            from .q4_0_expert import payload_size

            expected = payload_size(self.hidden_size, self.intermediate_size)
        elif self.dtype == "q4_1":
            from .q4_1_expert import payload_size

            expected = payload_size(self.hidden_size, self.intermediate_size)
        elif self.dtype == "q5_0":
            from .q5_0_expert import payload_size

            expected = payload_size(self.hidden_size, self.intermediate_size)
        elif self.dtype == "q5_1":
            from .q5_1_expert import payload_size

            expected = payload_size(self.hidden_size, self.intermediate_size)
        elif self.dtype == "int4-grouped-g64":
            from .grouped_int4_expert import payload_size

            expected = payload_size(self.hidden_size, self.intermediate_size, group_size=64)
        elif self.dtype == "mxfp4-e2m1-g32-ue8m0":
            from .mxfp4_expert import payload_size

            expected = payload_size(self.hidden_size, self.intermediate_size)
        elif self.dtype == "fp8-e4m3-b128-f32":
            from .fp8_block_expert import payload_size

            expected = payload_size(self.hidden_size, self.intermediate_size)
        for expert_id, region in self.regions.items():
            if not expert_id or "/" in expert_id:
                raise ValueError("expert ids must be non-empty and cannot contain '/'")
            if region.size != expected:
                raise ValueError(f"expert {self.id}/{expert_id} has {region.size} bytes; expected {expected}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "experts_used": self.experts_used,
            "dtype": self.dtype,
            "activation": self.activation,
            **({"contract": dict(self.contract)} if self.contract is not None else {}),
            "experts": {
                key: {
                    "path": value.path,
                    "offset": value.offset,
                    "size": value.size,
                    "sha256": value.sha256,
                    **({"storage": value.storage} if value.storage else {}),
                }
                for key, value in sorted(self.regions.items())
            },
        }


@dataclass(frozen=True)
class SparseExpertManifest:
    """Portable metadata for expert payloads; model math remains adapter-owned."""

    model_id: str
    architecture: str
    fingerprint: str
    layers: tuple[ExpertLayer, ...]
    format: str = MANIFEST_FORMAT

    def validate(self) -> None:
        if self.format != MANIFEST_FORMAT:
            raise ValueError(f"unsupported sparse expert manifest format: {self.format}")
        if not self.model_id or not self.architecture or not self.fingerprint:
            raise ValueError("model_id, architecture and fingerprint are required")
        if not self.layers:
            raise ValueError("at least one expert layer is required")
        ids = [layer.id for layer in self.layers]
        if len(set(ids)) != len(ids):
            raise ValueError("layer ids must be unique")
        for layer in self.layers:
            layer.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "model_id": self.model_id,
            "architecture": self.architecture,
            "fingerprint": self.fingerprint,
            "layers": [layer.to_dict() for layer in self.layers],
        }

    def write(self, path: str | Path) -> Path:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                    dir=destination.parent, prefix='.experts-', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(self.to_dict(), stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def read(cls, path: str | Path, *, max_bytes=64*1024*1024,
             max_layers=4096, max_experts=1048576) -> "SparseExpertManifest":
        for value in (max_bytes, max_layers, max_experts):
            if type(value) is not int or value <= 0:
                raise ValueError('positive manifest byte and inventory limits required')
        source = Path(path)
        try:
            from .publisher_media_files import read_regular
            encoded=read_regular(source,max_bytes=max_bytes)
            if len(encoded) > max_bytes:
                raise MemoryError('sparse expert manifest exceeds byte admission')
            payload = json.loads(encoded, object_pairs_hook=_unique_fields,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            raise ValueError(f"cannot read sparse expert manifest: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("sparse expert manifest must be a JSON object")
        layers: list[ExpertLayer] = []
        records = _records(payload.get('layers'))
        if len(records) > max_layers:
            raise MemoryError('sparse expert layer inventory exceeds admission')
        expert_count = 0
        for raw_layer in records:
            regions: dict[str, ExpertRegion] = {}
            experts = raw_layer.get("experts")
            if not isinstance(experts, dict):
                raise ValueError("layer experts must be an object")
            expert_count += len(experts)
            if expert_count > max_experts:
                raise MemoryError('sparse expert inventory exceeds admission')
            for expert_id, raw_region in experts.items():
                if not isinstance(raw_region, dict):
                    raise ValueError("expert region must be an object")
                regions[str(expert_id)] = ExpertRegion(
                    path=str(raw_region.get("path", "")),
                    offset=_integer(raw_region.get("offset"), "offset", minimum=0),
                    size=_integer(raw_region.get("size"), "size", minimum=1),
                    sha256=str(raw_region.get("sha256", "")),
                    storage=str(raw_region.get("storage", "")),
                )
            layers.append(ExpertLayer(
                id=str(raw_layer.get("id", "")),
                hidden_size=_integer(raw_layer.get("hidden_size"), "hidden_size", minimum=1),
                intermediate_size=_integer(raw_layer.get("intermediate_size"), "intermediate_size", minimum=1),
                experts_used=_integer(raw_layer.get("experts_used"), "experts_used", minimum=1),
                dtype=str(raw_layer.get("dtype", "")),
                activation=str(raw_layer.get("activation", "")),
                regions=regions,
                contract=raw_layer.get("contract"),
            ))
        manifest = cls(
            model_id=str(payload.get("model_id", "")),
            architecture=str(payload.get("architecture", "")),
            fingerprint=str(payload.get("fingerprint", "")),
            layers=tuple(layers),
            format=str(payload.get("format", "")),
        )
        manifest.validate()
        return manifest


class RoutePredictor:
    """Small local transition model used only for advisory expert reads."""

    def __init__(self) -> None:
        self._previous: dict[str, tuple[str, ...]] = {}
        self._transitions: dict[str, dict[tuple[str, ...], Counter[str]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        self._usage: dict[str, Counter[str]] = defaultdict(Counter)
        self._lock = threading.Lock()

    def observe(self, layer_id: str, expert_ids: Iterable[str]) -> None:
        current = tuple(sorted(dict.fromkeys(expert_ids)))
        if not current:
            return
        with self._lock:
            previous = self._previous.get(layer_id)
            if previous is not None:
                self._transitions[layer_id][previous].update(current)
            self._usage[layer_id].update(current)
            self._previous[layer_id] = current

    def predict(self, layer_id: str, after: Iterable[str], *, limit: int = 2) -> list[str]:
        state = tuple(sorted(dict.fromkeys(after)))
        if limit <= 0:
            return []
        with self._lock:
            learned = self._transitions[layer_id].get(state)
            source = learned if learned else self._usage[layer_id]
            return [key for key, _ in source.most_common(limit)]

    def hot(self, layer_id: str, *, limit: int) -> list[str]:
        with self._lock:
            return [key for key, _ in self._usage[layer_id].most_common(max(0, limit))]

    def ranked(self) -> list[tuple[str, str, int]]:
        with self._lock:
            return [
                (layer_id, expert_id, count)
                for layer_id, counts in self._usage.items()
                for expert_id, count in counts.items()
            ]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "usage": {layer: dict(counts) for layer, counts in self._usage.items()},
                "transitions": {
                    layer: [
                        {"after": list(state), "next": dict(counts)}
                        for state, counts in states.items()
                    ]
                    for layer, states in self._transitions.items()
                },
            }

    def restore(self, payload: Mapping[str, Any]) -> None:
        usage = payload.get("usage")
        transitions = payload.get("transitions")
        if not isinstance(usage, dict) or not isinstance(transitions, dict):
            raise ValueError("route profile is missing usage or transitions")
        restored_usage: dict[str, Counter[str]] = defaultdict(Counter)
        restored_transitions: dict[str, dict[tuple[str, ...], Counter[str]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        for layer, counts in usage.items():
            if not isinstance(counts, dict):
                raise ValueError("route profile usage must be an object")
            for expert, count in counts.items():
                value = _integer(count, "route count", minimum=0)
                if value:
                    restored_usage[str(layer)][str(expert)] = value
        for layer, records in transitions.items():
            if not isinstance(records, list):
                raise ValueError("route profile transitions must be arrays")
            for record in records:
                if not isinstance(record, dict) or not isinstance(record.get("after"), list) or not isinstance(record.get("next"), dict):
                    raise ValueError("route transition record is invalid")
                state = tuple(sorted(str(value) for value in record["after"]))
                for expert, count in record["next"].items():
                    value = _integer(count, "transition count", minimum=0)
                    if value:
                        restored_transitions[str(layer)][state][str(expert)] = value
        with self._lock:
            self._usage = restored_usage
            self._transitions = restored_transitions
            self._previous.clear()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                layer: {
                    "observations": sum(counts.values()),
                    "experts_seen": len(counts),
                    "states": len(self._transitions[layer]),
                }
                for layer, counts in self._usage.items()
            }


class SparseExpertRuntime:
    """Execute validated routed FFN layers with adaptive local residency."""

    def __init__(
        self,
        root: str | Path,
        manifest: SparseExpertManifest,
        *,
        capacity_bytes: int,
        io_mode: str = "buffered",
        storage_roots: dict[str, str | Path] | None = None,
        prefetch_workers: int = 1,
        prefetch_depth: int = 2,
        kernels: ExpertKernelRegistry | None = None,
        preferred_kernel: str | None = None,
        allow_kernel_fallback: bool = True,
    ) -> None:
        manifest.validate()
        # The generic kernel registry is intentionally broader than the
        # per-family admitted format matrix. For families with native sparse
        # execution routes, bind every layer to its family/native consumer
        # policy before creating DiskExpertStore (which opens artifact paths).
        source_family = {'mixtral':'mixtral', 'olmoe':'olmoe'}.get(manifest.architecture)
        if source_family is not None:
            from .source_scope import require_format_execution_consumer
            for layer in manifest.layers:
                require_format_execution_consumer(
                    family=source_family, backend='native',
                    persisted_format=layer.dtype)
        for layer in manifest.layers:
            self._check_contract(layer, kernels, preferred_kernel, allow_kernel_fallback)
            # Resolve the exact format/backend pair before DiskExpertStore
            # opens any expert container.  Previously only lattice contract
            # metadata was preflighted here; ordinary registered formats such
            # as block-FP8 or MXFP4 could pass manifest admission, open and map
            # every expert region, and only fail when the first routed token
            # reached the provider.  Keep the normal fallback policy explicit,
            # but require that the selected execution policy can actually
            # consume every admitted layer format up front.
            from .expert_store import DEFAULT_EXPERT_KERNELS

            registry = kernels or DEFAULT_EXPERT_KERNELS
            registry.resolve(
                dtype=layer.dtype,
                preferred=preferred_kernel,
                allow_fallback=allow_kernel_fallback,
            )
        self.manifest = manifest
        self.layers = {layer.id: layer for layer in manifest.layers}
        regions = {
            self.key(layer.id, expert_id): region
            for layer in manifest.layers
            for expert_id, region in layer.regions.items()
        }
        self.store = DiskExpertStore(
            root,
            regions,
            capacity_bytes=capacity_bytes,
            io_mode=io_mode,
            storage_roots=storage_roots,
            prefetch_workers=prefetch_workers,
            prefetch_depth=prefetch_depth,
        )
        self.predictor = RoutePredictor()
        self.kernels = kernels
        self.preferred_kernel = preferred_kernel
        self.allow_kernel_fallback = allow_kernel_fallback
        self._stats: dict[str, Any] = {
            "tokens": 0,
            "layer_calls": 0,
            "expert_calls": 0,
            "compute_seconds": 0.0,
            "layers": defaultdict(lambda: {"calls": 0, "expert_calls": 0, "seconds": 0.0}),
        }
        self._lock = threading.Lock()
        self._adaptive_pins: set[str] = set()
        self._closed = False
        self._closing = False

    @staticmethod
    def _check_contract(layer, kernels, preferred, fallback, *, batch=False):
        if layer.dtype not in {"e8-lattice-caller-v1", "iq3-format6-98-v1"}:
            return
        if kernels is None or preferred is None or fallback:
            raise ValueError("persisted lattice experts require an explicit backend with fallback disabled")
        backend = kernels.resolve(dtype=layer.dtype, batch=batch, preferred=preferred, allow_fallback=False)
        contract = getattr(backend, "contract", None)
        if contract is None or contract.metadata() != dict(layer.contract):
            raise ValueError("expert manifest and backend lattice contracts differ")

    @staticmethod
    def key(layer_id: str, expert_id: str) -> str:
        return f"{layer_id}/{expert_id}"

    def _check_open(self) -> None:
        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError('sparse expert runtime is closed')

    def execute(self, layer_id: str, hidden, routes: Sequence[tuple[str, float]]):
        self._check_open()
        layer = self.layers.get(layer_id)
        if layer is None:
            raise KeyError(layer_id)
        self._check_contract(layer, self.kernels, self.preferred_kernel, self.allow_kernel_fallback)
        routes = list(routes)
        if len(routes) > layer.experts_used:
            raise ValueError("router selected more experts than the manifest permits")
        qualified = [(self.key(layer_id, expert_id), weight) for expert_id, weight in routes]
        for expert_id, _ in routes:
            if expert_id not in layer.regions:
                raise KeyError(expert_id)
        started = time.perf_counter()
        output = routed_swiglu(
            self.store,
            hidden,
            qualified,
            hidden_size=layer.hidden_size,
            intermediate_size=layer.intermediate_size,
            dtype=layer.dtype,
            kernels=self.kernels,
            preferred_backend=self.preferred_kernel,
            allow_fallback=self.allow_kernel_fallback,
        )
        elapsed = time.perf_counter() - started
        selected = [expert_id for expert_id, _ in routes]
        self.predictor.observe(layer_id, selected)
        predictions = self.predictor.predict(layer_id, selected, limit=2)
        self.store.prefetch_many(self.key(layer_id, expert_id) for expert_id in predictions)
        with self._lock:
            record = self._stats["layers"][layer_id]
            record["calls"] += 1
            record["expert_calls"] += len(routes)
            record["seconds"] += elapsed
            self._stats["tokens"] += 1
            self._stats["layer_calls"] += 1
            self._stats["expert_calls"] += len(routes)
            self._stats["compute_seconds"] += elapsed
        return output

    def execute_batch(self, layer_id: str, hidden_batch, route_batch: Sequence[Sequence[tuple[str, float]]]):
        self._check_open()
        if len(hidden_batch) != len(route_batch):
            raise ValueError("hidden and route batches must have equal length")
        layer = self.layers.get(layer_id)
        if layer is None:
            raise KeyError(layer_id)
        self._check_contract(layer, self.kernels, self.preferred_kernel, self.allow_kernel_fallback, batch=True)
        normalized = [list(routes) for routes in route_batch]
        for routes in normalized:
            if len(routes) > layer.experts_used:
                raise ValueError("router selected more experts than the manifest permits")
            for expert_id, _ in routes:
                if expert_id not in layer.regions:
                    raise KeyError(expert_id)
        qualified = [
            [(self.key(layer_id, expert_id), weight) for expert_id, weight in routes]
            for routes in normalized
        ]
        started = time.perf_counter()
        output = routed_swiglu_batch(
            self.store,
            hidden_batch,
            qualified,
            hidden_size=layer.hidden_size,
            intermediate_size=layer.intermediate_size,
            dtype=layer.dtype,
            kernels=self.kernels,
            preferred_backend=self.preferred_kernel,
            allow_fallback=self.allow_kernel_fallback,
        )
        elapsed = time.perf_counter() - started
        expert_calls = sum(len(routes) for routes in normalized)
        for routes in normalized:
            selected = [expert_id for expert_id, _ in routes]
            self.predictor.observe(layer_id, selected)
            predictions = self.predictor.predict(layer_id, selected, limit=2)
            self.store.prefetch_many(self.key(layer_id, expert_id) for expert_id in predictions)
        with self._lock:
            record = self._stats["layers"][layer_id]
            record["calls"] += len(normalized)
            record["expert_calls"] += expert_calls
            record["seconds"] += elapsed
            self._stats["tokens"] += len(normalized)
            self._stats["layer_calls"] += len(normalized)
            self._stats["expert_calls"] += expert_calls
            self._stats["compute_seconds"] += elapsed
        return output

    def pin_hot(self, layer_id: str, *, limit: int, load: bool = False) -> list[str]:
        self._check_open()
        if layer_id not in self.layers:
            raise KeyError(layer_id)
        selected = self.predictor.hot(layer_id, limit=limit)
        qualified = [self.key(layer_id, expert_id) for expert_id in selected]
        self.store.pin(qualified, load=load)
        return selected

    def rebalance_hot(self, *, budget_bytes: int, load: bool = False) -> list[str]:
        """Promote the most frequently useful experts per resident byte."""
        self._check_open()
        if budget_bytes < 0 or budget_bytes > self.store.capacity_bytes:
            raise ValueError("hot-set budget must fit the expert cache")
        candidates: list[tuple[float, str, int]] = []
        for layer_id, expert_id, count in self.predictor.ranked():
            key = self.key(layer_id, expert_id)
            size = self.store.regions[key].size
            candidates.append((count / size, key, size))
        selected: list[str] = []
        used = 0
        for _, key, size in sorted(candidates, key=lambda item: (-item[0], item[1])):
            if used + size <= budget_bytes:
                selected.append(key)
                used += size
        self.store.unpin(self._adaptive_pins)
        self.store.pin(selected, load=load)
        self._adaptive_pins = set(selected)
        return selected

    def save_route_profile(self, path: str | Path) -> Path:
        self._check_open()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        payload = {
            "format": "openagent.route-profile.v1",
            "model_id": self.manifest.model_id,
            "fingerprint": self.manifest.fingerprint,
            "routes": self.predictor.snapshot(),
        }
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(destination)
        return destination

    def load_route_profile(self, path: str | Path, *, max_bytes=16*1024*1024) -> None:
        self._check_open()
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('positive route profile byte limit required')
        try:
            with Path(path).open('rb') as stream:
                encoded = stream.read(max_bytes + 1)
            if len(encoded) > max_bytes:
                raise MemoryError('route profile exceeds byte admission')
            payload = json.loads(encoded, object_pairs_hook=_unique_fields,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            raise ValueError(f"cannot read route profile: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("format") != "openagent.route-profile.v1":
            raise ValueError("unsupported route profile format")
        if payload.get("model_id") != self.manifest.model_id or payload.get("fingerprint") != self.manifest.fingerprint:
            raise ValueError("route profile belongs to a different model")
        routes = payload.get("routes")
        if not isinstance(routes, dict):
            raise ValueError("route profile has no route data")
        restored = RoutePredictor()
        restored.restore(routes)
        for layer_id, expert_id, _ in restored.ranked():
            layer = self.layers.get(layer_id)
            if layer is None or expert_id not in layer.regions:
                raise ValueError("route profile references an unknown layer or expert")
        for layer_id, transitions in restored._transitions.items():
            layer = self.layers.get(layer_id)
            if layer is None:
                raise ValueError('route profile transition references an unknown layer')
            for state, counts in transitions.items():
                if len(state) > layer.experts_used or any(
                        expert not in layer.regions for expert in (*state, *counts)):
                    raise ValueError('route profile transition references an invalid route')
        self.predictor = restored

    def status(self) -> dict[str, Any]:
        with self._lock:
            layer_stats = {key: dict(value) for key, value in self._stats["layers"].items()}
            compute = {key: value for key, value in self._stats.items() if key != "layers"}
            closed = self._closed
            closing = self._closing
        if closed or closing:
            return {
                "model_id": self.manifest.model_id,
                "closed": closed,
                "closing": closing,
                "architecture": self.manifest.architecture,
                "compute": {**compute, "layers": layer_stats},
                "preferred_kernel": self.preferred_kernel,
            }
        store = self.store.status()
        from .expert_store import DEFAULT_EXPERT_KERNELS

        registry = self.kernels or DEFAULT_EXPERT_KERNELS
        return {
            "model_id": self.manifest.model_id,
            "closed": False,
            "architecture": self.manifest.architecture,
            "compute": {
                **compute,
                "expert_load_reuse": (
                    max(0.0, 1.0 - store["misses"] / compute["expert_calls"])
                    if compute["expert_calls"] else 0.0
                ),
                "layers": layer_stats,
            },
            "routes": self.predictor.status(),
            "store": store,
            "kernels": [
                {
                    "name": info.name,
                    "provider": info.provider,
                    "available": info.available,
                    "priority": info.priority,
                    "compiled": info.capabilities.compiled,
                    "native_simd": info.capabilities.native_simd,
                    "evidence": [
                        {"kind": item.kind, "detail": item.detail, "verified": item.verified}
                        for item in info.evidence
                    ],
                }
                for info in registry.describe()
            ],
            "preferred_kernel": self.preferred_kernel,
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._closing:
                raise RuntimeError('sparse expert runtime cleanup is already active')
            self._closing = True
            errors = []
        try:
            try:
                self.store.close()
            except BaseException as error:
                errors.append(error)
            if self.kernels is not None:
                try:
                    self.kernels.close()
                except BaseException as error:
                    errors.append(error)
            if errors:
                raise RuntimeError('sparse expert runtime cleanup remains incomplete') from errors[0]
        finally:
            with self._lock:
                if not errors:
                    self._closed = True
                self._closing = False

    def __enter__(self) -> "SparseExpertRuntime":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value or not all(isinstance(item, dict) for item in value):
        raise ValueError("manifest layers must be a non-empty array of objects")
    return value


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    result = value
    if result < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return result
