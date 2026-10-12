"""Exact native K3 hybrid snapshot codec and resident-weight fixture planner.

Expanded MLA is explicitly selected. This adapter does not admit publisher
checkpoints, infer compressed layouts, or stream resident native MoE weights.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import math
import struct
import threading

import numpy as np

from ..native_kimi_k3_hybrid import NativeKimiK3Hybrid, KimiK3HybridSnapshot
from .facade import ExecutionPlan, ScalableSession
from .resources import CacheLayout, Geometry, positive
from .scheduler import DispatchStep


_MAGIC = b"OAK3S001"
_STEP_MAGIC = b"OAK3P001"
_LAYOUT = "expanded-head-kv-v1"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


class K3HybridSnapshotCodec:
    """Bounded canonical little-endian FP32 snapshots; never pickle.

    The template is borrowed and must remain open/immutable for the binding's
    lifetime. Its config, weight identity and KDA component identities are
    recorded in the contract. Only the exact current hybrid class is accepted.
    """

    def __init__(self, template, *, mla_layout, max_payload_bytes):
        if type(template) is not NativeKimiK3Hybrid:
            raise TypeError("exact NativeKimiK3Hybrid backend required")
        if mla_layout != _LAYOUT:
            raise ValueError("compressed MLA requires its own versioned native snapshot codec")
        self.template = template
        self.config = template.config
        self.max_payload_bytes = positive(max_payload_bytes, "max_payload_bytes")
        self.config_fingerprint = hashlib.sha256(_json(asdict(self.config))).hexdigest()
        self.model_identity = template._identity
        c, k = self.config.mla, self.config.kda
        self._component_ids = tuple(None if mixer is None else mixer._identity
                                    for mixer in template._mixers)
        self._full_specs, self._step_specs = [], []
        for index, kind in enumerate(self.config.layer_types):
            if kind == "mla":
                specs = [(f"{index}.keys", (c.capacity, c.heads, c.nope + c.rope)),
                         (f"{index}.values", (c.capacity, c.heads, c.value))]
                rows = [(name, shape[1:]) for name, shape in specs]
                self._full_specs.extend(specs)
                self._step_specs.extend(rows)
                # Detect another agent changing the native cache layout, rather
                # than silently serializing rank-compressed arrays as head KV.
                if (type(template._keys[index]) is not np.ndarray or
                        template._keys[index].shape != specs[0][1] or
                        type(template._values[index]) is not np.ndarray or
                        template._values[index].shape != specs[1][1]):
                    raise ValueError("native MLA snapshot is not expanded-head-kv-v1")
            else:
                specs = [(f"{index}.state", (k.heads, k.key, k.value)),
                         (f"{index}.q_history", (k.kernel - 1, k.heads, k.key)),
                         (f"{index}.k_history", (k.kernel - 1, k.heads, k.key)),
                         (f"{index}.v_history", (k.kernel - 1, k.heads, k.value))]
                self._full_specs.extend(specs)
                self._step_specs.extend(specs)
        self._full_header = self._header("full-capacity-snapshot", self._full_specs)
        self._step_header = self._header("one-token-mla-row-and-kda-state", self._step_specs)
        self.snapshot_bytes = self._size(self._full_header, self._full_specs)
        self.step_bytes = self._size(self._step_header, self._step_specs)
        if max(self.snapshot_bytes, self.step_bytes) > self.max_payload_bytes:
            raise MemoryError("K3 snapshot/step representation exceeds payload admission")
        self.backend_identity = hashlib.sha256(self._step_header).hexdigest()

    def _header(self, mode, specs):
        offset = 0
        arrays = []
        for name, shape in specs:
            size = math.prod(shape) * 4
            arrays.append(dict(name=name, dtype="<f4", shape=shape, layout="C",
                               offset=offset, size=size))
            offset += size
        body = _json(dict(version=1, mode=mode, mla_layout=_LAYOUT,
            config=asdict(self.config), config_fingerprint=self.config_fingerprint,
            model_identity=self.model_identity, component_identities=self._component_ids,
            arrays=arrays, array_bytes=offset))
        if len(body) > 64 * 1024:
            raise MemoryError("K3 snapshot metadata exceeds 64 KiB")
        return body

    @staticmethod
    def _size(header, specs):
        return 8 + 4 + len(header) + 8 + sum(math.prod(shape) * 4 for _, shape in specs) + 32

    def _binding(self, model):
        if (type(model) is not NativeKimiK3Hybrid or model.config != self.config or
                model._identity != self.model_identity):
            raise ValueError("K3 model/config/weight identity mismatch")
        if tuple(None if mixer is None else mixer._identity for mixer in model._mixers) != self._component_ids:
            raise ValueError("K3 recurrent component identity mismatch")

    def _arrays(self, snapshot):
        if (type(snapshot) is not KimiK3HybridSnapshot or snapshot.identity != self.model_identity
                or type(snapshot.length) is not int or not 0 <= snapshot.length <= self.config.mla.capacity):
            raise ValueError("foreign or invalid K3 snapshot")
        c = self.config.mla
        if any(type(items) is not tuple or len(items) != c.layers for items in
               (snapshot.keys, snapshot.values, snapshot.recurrent)):
            raise ValueError("K3 snapshot layer inventory mismatch")
        arrays = []
        for index, kind in enumerate(self.config.layer_types):
            if kind == "mla":
                if snapshot.recurrent[index] is not None:
                    raise ValueError("MLA layer has unexpected recurrent state")
                arrays.extend((snapshot.keys[index], snapshot.values[index]))
            else:
                if snapshot.keys[index] is not None or snapshot.values[index] is not None:
                    raise ValueError("KDA layer has unexpected MLA buffers")
                item = snapshot.recurrent[index]
                if (type(item) is not tuple or len(item) != 2 or item[0] != self._component_ids[index]
                        or type(item[1]) is not tuple or len(item[1]) != 3):
                    raise ValueError("KDA snapshot identity/inventory mismatch")
                position, state, history = item[1]
                if type(position) is not int or position != snapshot.length:
                    raise ValueError("KDA snapshot position differs from hybrid position")
                if type(history) is not tuple or len(history) != 3:
                    raise ValueError("KDA convolution history inventory mismatch")
                arrays.extend((state, *history))
        for array, (name, shape) in zip(arrays, self._full_specs):
            self._array(array, shape, name)
            if snapshot.length == 0 and array.any():
                raise ValueError("nonzero state at empty K3 position")
            if (name.endswith(".keys") or name.endswith(".values")) and array[snapshot.length:].any():
                raise ValueError("nonzero uncommitted MLA capacity")
        return arrays

    @staticmethod
    def _array(array, shape, name):
        if (type(array) is not np.ndarray or array.dtype != np.float32 or
                array.shape != shape or not array.flags.c_contiguous):
            raise ValueError("invalid typed C-layout K3 array: " + name)
        # No vocabulary/context-sized temporary boolean tensor.
        with np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"],
                       op_flags=["readonly"], buffersize=4096) as iterator:
            if any(not np.isfinite(chunk).all() for chunk in iterator):
                raise ValueError("nonfinite K3 state: " + name)

    def _pack(self, length, arrays, *, step):
        header = self._step_header if step else self._full_header
        specs = self._step_specs if step else self._full_specs
        size = self.step_bytes if step else self.snapshot_bytes
        output = bytearray(size)
        output[:8] = _STEP_MAGIC if step else _MAGIC
        struct.pack_into("<I", output, 8, len(header))
        output[12:12 + len(header)] = header
        struct.pack_into("<Q", output, 12 + len(header), length)
        offset = 20 + len(header)
        for array, (name, shape) in zip(arrays, specs):
            self._array(array, shape, name)
            # Explicit canonical conversion; object/native-pointer dtypes never enter.
            encoded = array.astype("<f4", copy=False).tobytes(order="C")
            output[offset:offset + len(encoded)] = encoded
            offset += len(encoded)
        if offset != size - 32:
            raise ValueError("K3 array inventory does not fill admitted frame")
        with memoryview(output)[:offset] as view:
            output[offset:] = hashlib.sha256(view).digest()
        return output

    def _unpack(self, payload, *, step, expected_length=None):
        header = self._step_header if step else self._full_header
        specs = self._step_specs if step else self._full_specs
        size = self.step_bytes if step else self.snapshot_bytes
        with memoryview(payload) as original, original.cast("B") as view:
            if view.nbytes != size or view[:8] != (_STEP_MAGIC if step else _MAGIC):
                raise ValueError("K3 frame magic/size/layout mismatch")
            if struct.unpack_from("<I", view, 8)[0] != len(header) or view[12:12 + len(header)] != header:
                raise ValueError("K3 typed metadata/config fingerprint mismatch")
            with view[:-32] as body:
                if hashlib.sha256(body).digest() != view[-32:]:
                    raise ValueError("K3 snapshot checksum mismatch")
            length = struct.unpack_from("<Q", view, 12 + len(header))[0]
            if not 0 <= length <= self.config.mla.capacity or (expected_length is not None and length != expected_length):
                raise ValueError("K3 frame position mismatch")
            arrays = []
            offset = 20 + len(header)
            for name, shape in specs:
                count = math.prod(shape)
                array = np.frombuffer(view, dtype="<f4", count=count, offset=offset).reshape(shape)
                array = np.array(array, dtype=np.float32, copy=True, order="C")
                self._array(array, shape, name)
                arrays.append(array)
                offset += count * 4
            return length, arrays

    def _snapshot(self, length, arrays):
        keys, values, recurrent = [], [], []
        iterator = iter(arrays)
        for index, kind in enumerate(self.config.layer_types):
            if kind == "mla":
                keys.append(next(iterator))
                values.append(next(iterator))
                recurrent.append(None)
            else:
                state = next(iterator)
                history = tuple(next(iterator) for _ in range(3))
                keys.append(None)
                values.append(None)
                recurrent.append((self._component_ids[index], (length, state, history)))
        return KimiK3HybridSnapshot(self.model_identity, length, tuple(keys), tuple(values), tuple(recurrent))

    def encode_snapshot(self, snapshot):
        return self._pack(snapshot.length, self._arrays(snapshot), step=False)

    def decode_snapshot(self, payload):
        length, arrays = self._unpack(payload, step=False)
        snapshot = self._snapshot(length, arrays)
        self._arrays(snapshot)
        return snapshot

    def restore(self, model, payload):
        """Parse/validate everything first, then native atomic whole-state restore."""
        self._binding(model)
        candidate = self.decode_snapshot(payload)
        model.restore(candidate)

    def encode_step(self, snapshot):
        arrays = self._arrays(snapshot)
        if snapshot.length < 1:
            raise ValueError("cannot encode an empty token step")
        rows = [np.ascontiguousarray(array[snapshot.length - 1])
                if name.endswith(".keys") or name.endswith(".values") else array
                for array, (name, _) in zip(arrays, self._full_specs)]
        return self._pack(snapshot.length, rows, step=True)

    def geometry(self, *, max_batch_tokens):
        c = self.config.mla
        positive(max_batch_tokens, "max_batch_tokens")
        if max_batch_tokens > min(128, c.capacity):
            raise ValueError("native K3 batch bound exceeded")
        sizes = [0] * c.layers
        for name, shape in self._step_specs:
            sizes[int(name.split(".")[0])] += math.prod(shape) * 4
        sizes[0] += self.step_bytes - sum(sizes)
        layouts = tuple(CacheLayout("custom" if kind == "mla" else "recurrent-state", size,
            f"k3-step-v1:{self.backend_identity}:layer-{index}:{kind}")
            for index, (kind, size) in enumerate(zip(self.config.layer_types, sizes)))
        return Geometry(self.backend_identity, c.hidden, self.config.moe.intermediate,
            c.layers, c.heads, c.heads, max(1, c.nope + c.rope), 4, c.capacity,
            max_batch_tokens, frozenset({"k3-native-resident-fp32"}), layouts)

    def fork_empty(self):
        """Isolate mutable caches; borrow immutable admitted weights/native code.

        Native reset/restore replace caches and clone recurrent mixers. The
        exact-class binding deliberately depends on that current native API.
        """
        with self.template._lock:
            self._binding(self.template)
            candidate = copy.copy(self.template)
            candidate._lock = threading.RLock()
            candidate._mutation_active = False
            candidate.reset()
            return candidate


class K3HybridPlanner:
    """Actual native token-to-logit dispatch; native kernel limits still apply."""

    def __init__(self, codec, geometry):
        if not isinstance(codec, K3HybridSnapshotCodec) or codec.geometry(
                max_batch_tokens=geometry.max_batch_tokens) != geometry:
            raise ValueError("exact K3 codec/geometry binding required")
        self.codec, self.geometry = codec, geometry

    def scratch_required(self, tokens):
        model, c = self.codec.template, self.codec.config.mla
        # Keep native's own weight/cache workspace allowance; additionally admit
        # full snapshot reconstruction, codec frames, and output/finite checks.
        native = (5 * model._weights_bytes + 5 * model._cache_bytes +
                  4 * tokens * (c.vocab + 16 * c.hidden) + 1024 * 1024)
        return native + 8 * self.codec.snapshot_bytes + 3 * self.codec.step_bytes + 8 * tokens * c.vocab + 32768

    def __call__(self, token_ids):
        holder = {}

        def cleanup():
            # Never close borrowed experts/libraries; discard only request caches.
            holder.clear()

        def restore(context, _):
            context.defer(cleanup)
            native = self.codec.fork_empty()
            holder["model"] = native
            snapshot = native.snapshot()
            arrays = self.codec._arrays(snapshot)
            restored = 0
            for metadata in context.cache.page_metadata():
                context.check_cancel()
                if len(metadata.token_ids) != 1 or metadata.position != restored:
                    raise ValueError("K3 cache requires consecutive one-token records")
                if not 0 <= metadata.token_ids[0] < self.codec.config.mla.vocab:
                    raise ValueError("K3 recovered token is outside vocabulary")
                with context.cache.read_page(metadata.ordinal) as payload:
                    length, rows = self.codec._unpack(payload, step=True, expected_length=restored + 1)
                for index, ((name, _), row) in enumerate(zip(self.codec._full_specs, rows)):
                    if name.endswith(".keys") or name.endswith(".values"):
                        arrays[index][restored] = row
                    else:
                        arrays[index] = row
                restored = length
            if restored != context.position:
                raise ValueError("K3 reconstructed position differs from committed cache")
            native.restore(self.codec._snapshot(restored, arrays))
            holder["logits"] = np.empty((len(context.token_ids), self.codec.config.mla.vocab), np.float32)

        def execute(context, _):
            native = holder["model"]
            for index, token in enumerate(context.token_ids):
                context.check_cancel()
                output = native.forward(np.array([token], np.int64))
                context.check_cancel()
                holder["logits"][index] = output[0]
                payload = self.codec.encode_step(native.snapshot())
                context.cache.append_page((token,), payload)
                del payload

        return ExecutionPlan([DispatchStep(None, restore), DispatchStep(None, execute)],
            lambda context: holder["logits"], self.scratch_required(len(token_ids)))


def bind_k3_session(executor, codec, session_id, owner, *, create=True, prefix_id=None):
    """Concrete coordinator facade using native K3 math and exact page recovery."""
    planner = K3HybridPlanner(codec, executor.geometry)
    return ScalableSession(executor, session_id, owner, planner,
        vocab_size=codec.config.mla.vocab, create=create, prefix_id=prefix_id)
