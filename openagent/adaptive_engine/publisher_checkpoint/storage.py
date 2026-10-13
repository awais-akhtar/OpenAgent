"""Bounded metadata snapshots and local SafeTensors range reads."""
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

from ..native_checkpoint import _json, _open_local, _pairs, _stamp
from ..session_recovery import _path

WIDTHS = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
          "I32": 4, "U32": 4, "I64": 8, "U64": 8,
          "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
          "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1}


def positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return value


def freeze(value):
    if type(value) is dict:
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if type(value) is list:
        return tuple(freeze(v) for v in value)
    return value


def thaw(value):
    """Detach immutable metadata into ordinary JSON-compatible containers."""
    if isinstance(value, (dict, MappingProxyType)):
        return {key:thaw(item) for key,item in value.items()}
    if isinstance(value, (list,tuple)):
        return [thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    shard: str
    dtype: str
    shape: tuple
    offsets: tuple
    base: int
    stamp: tuple

    @property
    def nbytes(self):
        return self.offsets[1] - self.offsets[0]


@dataclass(frozen=True)
class Snapshot:
    root: Path
    config: object
    config_sha256: str
    tensors: object
    files: object
    metadata_bytes: int

    def assert_unchanged(self):
        if (self.root / "model.safetensors.index.json").exists() != (
                "model.safetensors.index.json" in self.files):
            raise ValueError("checkpoint index presence changed since planning")
        for filename, stamp in self.files.items():
            with ExitStack() as stack:
                _, current = _open_local(self.root, self.root / filename, stack)
                if current != stamp:
                    raise ValueError("checkpoint changed since planning: " + filename)
        if {p.name for p in self.root.glob("*.safetensors")} != {
                t.shard for t in self.tensors.values()}:
            raise ValueError("checkpoint shard inventory changed since planning")

    def read_range(self, name, offset, length):
        if type(name) is not str or name not in self.tensors:
            raise ValueError("unknown planned checkpoint tensor")
        if (type(offset) is not int or type(length) is not int or
                offset < 0 or length < 0):
            raise ValueError("nonnegative integer tensor byte range required")
        tensor = self.tensors[name]
        if not offset <= offset + length <= tensor.nbytes:
            raise ValueError("tensor read outside declared extent")
        with ExitStack() as stack:
            source, stamp = _open_local(self.root, self.root / tensor.shard, stack)
            if stamp != tensor.stamp:
                raise ValueError("checkpoint changed since planning: " + tensor.shard)
            source.seek(tensor.base + tensor.offsets[0] + offset)
            raw = source.read(length)
            if len(raw) != length:
                raise OSError("checkpoint payload truncated")
            if _stamp(source) != stamp:
                raise ValueError("checkpoint changed during payload read")
            return raw


def snapshot(directory, *, max_metadata_bytes, max_tensors, max_shards):
    for name, value in (("max_metadata_bytes", max_metadata_bytes),
                        ("max_tensors", max_tensors), ("max_shards", max_shards)):
        positive(value, name)
    root = _path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("checkpoint must be a local directory")
    files, inventory, used = {}, {}, 0

    def debit(size):
        nonlocal used
        used += size
        if used > max_metadata_bytes:
            raise MemoryError("publisher metadata budget exceeded")

    with ExitStack() as stack:
        source, stamp = _open_local(root, root / "config.json", stack)
        files["config.json"] = stamp
        debit(stamp[2])
        config = _json(source, stamp)
        if type(config) is not dict:
            raise ValueError("publisher config must be an object")
        source.seek(0)
        digest = hashlib.sha256(source.read(stamp[2])).hexdigest()
        if _stamp(source) != stamp:
            raise ValueError("config changed during read")
    shards = sorted(root.glob("*.safetensors"))
    if not shards or len(shards) > max_shards:
        raise ValueError("missing shards or shard count budget exceeded")
    for shard in shards:
        with ExitStack() as stack:
            source, stamp = _open_local(root, shard, stack)
            files[shard.name] = stamp
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ValueError("missing SafeTensors header length")
            length = int.from_bytes(prefix, "little")
            if not 1 < length <= min(16 * 1024**2, stamp[2] - 8):
                raise ValueError("invalid SafeTensors header length")
            debit(length + 8)
            header = json.loads(source.read(length), object_pairs_hook=_pairs,
                                parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            if type(header) is not dict:
                raise ValueError("SafeTensors header must be an object")
            base, ranges = length + 8, []
            for name, entry in header.items():
                if name == "__metadata__":
                    if (type(entry) is not dict or
                            any(type(k) is not str or not k or type(v) is not str
                                for k, v in entry.items())):
                        raise ValueError("invalid SafeTensors metadata")
                    continue
                if type(name) is not str or not name or name in inventory or type(entry) is not dict:
                    raise ValueError("duplicate or invalid publisher tensor: " + str(name))
                dtype, shape, offsets = (entry.get(k) for k in ("dtype", "shape", "data_offsets"))
                if (type(dtype) is not str or dtype not in WIDTHS
                        or type(shape) is not list or len(shape) > 32
                        or any(type(n) is not int or not 0 <= n < 2**63 for n in shape)
                        or type(offsets) is not list or len(offsets) != 2
                        or any(type(n) is not int for n in offsets)):
                    raise ValueError("invalid tensor metadata: " + name)
                start, end = offsets
                if not 0 <= start <= end <= stamp[2] - base or end-start != math.prod(shape)*WIDTHS[dtype]:
                    raise ValueError("invalid tensor extent: " + name)
                ranges.append((start, end))
                inventory[name] = TensorMetadata(name, shard.name, dtype, tuple(shape),
                                                 tuple(offsets), base, stamp)
                if len(inventory) > max_tensors:
                    raise MemoryError("publisher tensor count budget exceeded")
            position = 0
            for start, end in sorted(ranges):
                if start != position:
                    raise ValueError("overlapping or noncontiguous SafeTensors payload")
                position = end
            if position != stamp[2] - base:
                raise ValueError("unaccounted SafeTensors payload bytes")
            if _stamp(source) != stamp:
                raise ValueError("shard changed during metadata read")
    if not inventory:
        raise ValueError("checkpoint has no tensors")
    path = root / "model.safetensors.index.json"
    if path.exists():
        with ExitStack() as stack:
            source, stamp = _open_local(root, path, stack)
            files[path.name] = stamp
            debit(stamp[2])
            document = _json(source, stamp)
            mapping = document.get("weight_map") if type(document) is dict else None
            if type(mapping) is not dict or set(mapping) != set(inventory):
                raise ValueError("publisher index differs from shard inventory")
            if any(mapping[n] != t.shard for n, t in inventory.items()):
                raise ValueError("publisher index contains mismatched or nonlocal paths")
            if _stamp(source) != stamp:
                raise ValueError("index changed during read")
    result = Snapshot(root, freeze(config), digest, MappingProxyType(inventory),
                      MappingProxyType(files), used)
    result.assert_unchanged()
    return result
