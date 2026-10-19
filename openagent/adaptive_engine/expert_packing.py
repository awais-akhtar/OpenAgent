"""Pack explicitly mapped SafeTensors experts into aligned runtime payloads."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import math
import os
import json
from uuid import uuid4
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Mapping

from .expert_runtime import ExpertLayer, SparseExpertManifest
from .expert_store import ExpertRegion
from .failure_reporting import add_failure_note, failure_summary


_DTYPES = {"F16": ("float16", 2), "F32": ("float32", 4)}
_SOURCE_ITEM_BYTES = {"F16": 2, "BF16": 2, "F32": 4}
_HEADER_ITEM_BYTES={**_SOURCE_ITEM_BYTES,'F64':8,'I64':8,'U64':8,'I32':4,'U32':4,
    'I16':2,'U16':2,'I8':1,'U8':1,'BOOL':1,'F8_E4M3':1,'F8_E5M2':1,'F8_E8M0':1}


@dataclass(frozen=True)
class TensorRangeBinding:
    """Explicit contiguous matrix view inside a declared source tensor bank."""
    name: str
    source_shape: tuple[int, ...]
    offset_elements: int
    shape: tuple[int, int]


@dataclass(frozen=True)
class ExpertTensorBinding:
    id: str
    gate: str | TensorRangeBinding
    up: str | TensorRangeBinding
    down: str | TensorRangeBinding


@dataclass(frozen=True)
class ExpertLayerBinding:
    id: str
    hidden_size: int
    intermediate_size: int
    experts_used: int
    experts: tuple[ExpertTensorBinding, ...]


def pack_safetensors_experts(
    source: str | Path,
    destination: str | Path,
    *,
    model_id: str,
    architecture: str,
    fingerprint: str,
    layers: Iterable[ExpertLayerBinding],
    alignment: int = 4096,
) -> SparseExpertManifest:
    """Create one immutable expert container without changing tensor values.

    Tensor names are supplied by a model-family adapter. The packer does not
    infer architecture semantics. Gate and up tensors must be [I,H], down must
    be [H,I], and every mapped tensor must use the same F16 or F32 dtype.
    """
    if type(alignment) is not int or not 0<alignment<=1024**2 or alignment & (alignment-1):
        raise ValueError('alignment must be a power of two no larger than one MiB')
    source_path = Path(source).resolve(strict=True)
    destination_path = Path(destination).resolve()
    if source_path == destination_path:
        raise ValueError("source and destination must differ")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    source_stamp=_source_stamp(source_path.stat())
    header,data_base=_read_header(source_path)
    if _source_stamp(source_path.stat())!=source_stamp:
        raise ValueError('checkpoint source changed during complete header admission')
    layer_specs = tuple(layers)
    if not layer_specs:
        raise ValueError("at least one layer binding is required")
    _admit_binding_inventory(layer_specs,{name:tensor for name,tensor in header.items() if name!='__metadata__'},
        allowed_dtypes=_DTYPES,uniform_layer_dtype=True)
    temporary = destination_path.with_name(destination_path.name+'.tmp-'+uuid4().hex)
    manifest_layers: list[ExpertLayer] = []
    try:
        with source_path.open("rb", buffering=0) as reader, temporary.open("xb", buffering=0) as writer:
            if _source_stamp(os.fstat(reader.fileno()))!=source_stamp:
                raise ValueError('checkpoint source changed before expert packing')
            for layer in layer_specs:
                if not layer.experts:
                    raise ValueError(f"layer {layer.id} has no expert bindings")
                regions: dict[str, ExpertRegion] = {}
                layer_dtype = ""
                for expert in layer.experts:
                    _align(writer, alignment)
                    offset = writer.tell()
                    digest = hashlib.sha256()
                    dtypes: list[str] = []
                    for role, name, shape in (
                        ("gate", expert.gate, [layer.intermediate_size, layer.hidden_size]),
                        ("up", expert.up, [layer.intermediate_size, layer.hidden_size]),
                        ("down", expert.down, [layer.hidden_size, layer.intermediate_size]),
                    ):
                        tensor = _tensor(header, name)
                        if tensor["shape"] != shape:
                            raise ValueError(
                                f"{layer.id}/{expert.id} {role} tensor {name} has shape "
                                f"{tensor['shape']}; expected {shape}"
                            )
                        dtype = str(tensor["dtype"])
                        if dtype not in _DTYPES:
                            raise ValueError(f"tensor {name} uses unsupported dtype {dtype}")
                        dtypes.append(dtype)
                        start, end = tensor["data_offsets"]
                        _copy_range(reader,writer,data_base+start,end-start,digest,dtype=dtype)
                    if len(set(dtypes)) != 1:
                        raise ValueError(f"{layer.id}/{expert.id} mixes tensor dtypes")
                    runtime_dtype = _DTYPES[dtypes[0]][0]
                    if layer_dtype and runtime_dtype != layer_dtype:
                        raise ValueError(f"layer {layer.id} mixes expert dtypes")
                    layer_dtype = runtime_dtype
                    regions[expert.id] = ExpertRegion(
                        path=destination_path.name,
                        offset=offset,
                        size=writer.tell() - offset,
                        sha256=digest.hexdigest(),
                    )
                manifest_layers.append(ExpertLayer(
                    id=layer.id,
                    hidden_size=layer.hidden_size,
                    intermediate_size=layer.intermediate_size,
                    experts_used=layer.experts_used,
                    dtype=layer_dtype,
                    activation="swiglu",
                    regions=regions,
                ))
            writer.flush()
            os.fsync(writer.fileno())
        manifest = SparseExpertManifest(
            model_id=model_id,
            architecture=architecture,
            fingerprint=fingerprint,
            layers=tuple(manifest_layers),
        )
        manifest.validate()
        if _source_stamp(source_path.stat())!=source_stamp:
            raise ValueError('checkpoint source changed before expert publication')
        temporary.replace(destination_path)
        from .checkpoint_recovery import _sync_directory
        _sync_directory(destination_path.parent)
        return manifest
    except BaseException as primary:
        _cleanup_temporaries((temporary,),primary)
        raise


def pack_safetensors_directory_int8(
    source_directory: str | Path,
    destination: str | Path,
    *,
    model_id: str,
    architecture: str,
    fingerprint: str,
    layers: Iterable[ExpertLayerBinding],
    alignment: int = 4096,
    progress: Callable[[int, int, str, str], None] | None = None,
    storage_destinations: Mapping[str, str | Path] | None = None,
    dtype: str = "int8-row",
    lattice_contract=None,
    iq3_grid=None,
    max_conversion_bytes: int = 1024*1024*1024,
) -> SparseExpertManifest:
    """Quantize mapped F16/BF16/F32 experts to row INT8/INT4 or block Q8/Q4.

    Source shards are indexed by reading only their SafeTensors headers. One
    projection is materialized at a time, keeping conversion memory bounded by
    the largest mapped matrix rather than the whole checkpoint.
    """
    import numpy as np

    if type(max_conversion_bytes) is not int or max_conversion_bytes <= 0:
        raise ValueError('positive expert conversion working-memory limit required')
    if dtype not in {"int8-row", "int2-row", "int3-grouped-g64", "int4-row", "int4-grouped-g64", "mxfp4-e2m1-g32-ue8m0", "q8_0", "q8_1", "q4_0", "q4_1", "q5_0", "q5_1", "fp8-e4m3-b128-f32", "e8-lattice-caller-v1", "iq3-format6-98-v1"}:
        raise ValueError("unsupported expert packing dtype")
    if dtype == "e8-lattice-caller-v1":
        from .e8_lattice_expert import LatticeContract

        if not isinstance(lattice_contract, LatticeContract):
            raise TypeError("E8 packing requires an explicit LatticeContract")
    elif lattice_contract is not None:
        raise ValueError("lattice_contract is only accepted for E8 packing")
    if dtype == 'iq3-format6-98-v1':
        from .iq3_lattice import IQ3Grid
        if type(iq3_grid) is not IQ3Grid:
            raise ValueError('IQ3 packing requires an explicit admitted grid')
    elif iq3_grid is not None:
        raise ValueError('iq3_grid is only accepted for IQ3 packing')
    root = Path(source_directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("SafeTensors source must be a directory")
    if type(alignment) is not int or not 0<alignment<=1024**2 or alignment & (alignment-1):
        raise ValueError('alignment must be a power of two no larger than one MiB')
    index = _scan_directory(root)
    layer_specs = tuple(layers)
    if not layer_specs:
        raise ValueError("at least one layer binding is required")
    _admit_binding_inventory(layer_specs,{name:tensor for name,(_,tensor,_) in index.items()},
        allowed_dtypes=_SOURCE_ITEM_BYTES,max_conversion_bytes=max_conversion_bytes,
        conversion_extra=0 if lattice_contract is None else 128*len(lattice_contract.codebook),target_dtype=dtype)
    total_experts = sum(len(layer.experts) for layer in layer_specs)
    completed_experts = 0
    destination_path = Path(destination).resolve()
    destinations = {"": destination_path}
    for name, value in (storage_destinations or {}).items():
        if not name or "/" in name or "\\" in name:
            raise ValueError("storage destination names must be simple non-empty identifiers")
        destinations[name] = Path(value).resolve()
    if len(set(destinations.values())) != len(destinations):
        raise ValueError("each storage destination must use a different file")
    if len(destinations)>1 and any(path.exists() for path in destinations.values()):
        raise ValueError('multi-container publication requires unused immutable destination names')
    source_paths = {path.resolve() for path, _, _ in index.values()}
    if source_paths.intersection(destinations.values()):
        raise ValueError('expert containers cannot overwrite checkpoint shards')
    for path in destinations.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    temporaries = {
        name: path.with_name(path.name+'.tmp-'+uuid4().hex)
        for name, path in destinations.items()
    }
    target_names = tuple(destinations)
    manifest_layers: list[ExpertLayer] = []
    try:
        with _open_expert_staging_writers(temporaries) as writers:
            for layer in layer_specs:
                if not layer.experts:
                    raise ValueError(f"layer {layer.id} has no expert bindings")
                regions: dict[str, ExpertRegion] = {}
                for expert in layer.experts:
                    storage = target_names[completed_experts % len(target_names)]
                    writer = writers[storage]
                    _align(writer, alignment)
                    offset = writer.tell()
                    digest = hashlib.sha256()
                    projections = (
                        ("gate", expert.gate, [layer.intermediate_size, layer.hidden_size]),
                        ("up", expert.up, [layer.intermediate_size, layer.hidden_size]),
                        ("down", expert.down, [layer.hidden_size, layer.intermediate_size]),
                    )
                    scales: list[np.ndarray] = []
                    for role, name, shape in projections:
                        # Include source bytes, promoted matrix, quantizer
                        # temporaries, encoded output and retained row scales.
                        # E8 nearest-point work also depends on the codebook.
                        required = 64*shape[0]*shape[1]+4096
                        if lattice_contract is not None:
                            required += 128*len(lattice_contract.codebook)
                        if required > max_conversion_bytes:
                            raise MemoryError('expert matrix conversion exceeds working-memory admission')
                        matrix = _read_float_matrix(index, name, expected_shape=shape,
                                                    max_bytes=max_conversion_bytes)
                        if list(matrix.shape) != shape:
                            raise ValueError(
                                f"{layer.id}/{expert.id} {role} tensor {name} has shape "
                                f"{list(matrix.shape)}; expected {shape}"
                            )
                        if not np.isfinite(matrix).all():
                            raise ValueError(f"tensor {name} must contain only finite weights")
                        if dtype == "iq3-format6-98-v1":
                            from .iq3_lattice import encode_iq3
                            raw = encode_iq3(matrix, grid=iq3_grid,
                                             max_bytes=max_conversion_bytes).tobytes()
                        elif dtype == "e8-lattice-caller-v1":
                            from .e8_lattice_expert import pack_matrix

                            raw = pack_matrix(matrix, lattice_contract)
                        elif dtype == "q8_0":
                            from .q8_0_expert import pack_matrix

                            raw = pack_matrix(matrix)
                        elif dtype == "q8_1":
                            from .q8_1_expert import pack_matrix

                            raw = pack_matrix(matrix)
                        elif dtype == "q4_0":
                            from .q4_0_expert import pack_matrix

                            raw = pack_matrix(matrix)
                        elif dtype == "q4_1":
                            from .q4_1_expert import pack_matrix

                            raw = pack_matrix(matrix)
                        elif dtype == "q5_0":
                            from .q5_0_expert import pack_matrix

                            raw = pack_matrix(matrix)
                        elif dtype == "q5_1":
                            from .q5_1_expert import pack_matrix

                            raw = pack_matrix(matrix)
                        elif dtype == "fp8-e4m3-b128-f32":
                            from .fp8_block_expert import pack_matrix

                            raw = pack_matrix(matrix)
                        elif dtype in {"int2-row", "int3-grouped-g64"}:
                            from .lowbit_expert import pack_matrix

                            raw, scale = pack_matrix(matrix, dtype=dtype)
                        elif dtype == "int4-grouped-g64":
                            from .grouped_int4_expert import pack_matrix

                            raw, scale = pack_matrix(matrix, group_size=64)
                        elif dtype == "mxfp4-e2m1-g32-ue8m0":
                            from .mxfp4_expert import pack_matrix

                            raw = pack_matrix(matrix)
                        elif dtype == "int4-row":
                            from .packed_expert import pack_matrix

                            raw, scale = pack_matrix(matrix)
                        else:
                            row_max = np.max(np.abs(matrix), axis=1)
                            scale = np.maximum(row_max, np.float32(1e-12)) / np.float32(127.0)
                            q = np.clip(np.rint(matrix / scale[:, None]), -128, 127).astype(np.int8)
                            raw = q.tobytes(order="C")
                        if dtype not in {"q8_0", "q8_1", "q4_0", "q4_1", "q5_0", "q5_1", "mxfp4-e2m1-g32-ue8m0", "fp8-e4m3-b128-f32", "e8-lattice-caller-v1", "iq3-format6-98-v1"}:
                            scales.append(scale.astype("<f4", copy=False))
                        _write_all(writer,raw)
                        digest.update(raw)
                        del matrix, raw
                        if dtype == "int8-row":
                            del q, row_max
                    for scale in scales:
                        raw = scale.tobytes(order="C")
                        _write_all(writer,raw)
                        digest.update(raw)
                    regions[expert.id] = ExpertRegion(
                        path=destinations[storage].name,
                        offset=offset,
                        size=writer.tell() - offset,
                        sha256=digest.hexdigest(),
                        storage=storage,
                    )
                    completed_experts += 1
                    if progress is not None:
                        progress(completed_experts, total_experts, layer.id, expert.id)
                manifest_layers.append(ExpertLayer(
                    id=layer.id,
                    hidden_size=layer.hidden_size,
                    intermediate_size=layer.intermediate_size,
                    experts_used=layer.experts_used,
                    dtype=dtype,
                    activation="swiglu",
                    regions=regions,
                    contract=(iq3_grid.metadata() if iq3_grid is not None else
                              lattice_contract.metadata() if lattice_contract is not None else None),
                ))
            for writer in writers.values():
                writer.flush()
                os.fsync(writer.fileno())
        manifest = SparseExpertManifest(
            model_id=model_id,
            architecture=architecture,
            fingerprint=fingerprint,
            layers=tuple(manifest_layers),
        )
        manifest.validate()
        source_stamps = {source_path: source_tensor['_source_stamp']
                         for source_path, source_tensor, _ in index.values()}
        for source_path, stamp in source_stamps.items():
            if _source_stamp(source_path.stat()) != stamp:
                raise ValueError('checkpoint shard changed before expert container publication')
        _publish_containers(temporaries,destinations)
        return manifest
    except BaseException as primary:
        _cleanup_temporaries(temporaries.values(),primary)
        raise


def _source_stamp(info):
    # Windows exposes st_ctime as creation time (and Python 3.14 may report
    # different values for path.stat() and fstat() on the same file). Use the
    # explicit birth time there so opening an unchanged checkpoint is stable.
    creation_time = getattr(info, "st_birthtime_ns", info.st_ctime_ns)
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, creation_time)


@contextmanager
def _open_expert_staging_writers(temporaries):
    """Close every opened staging writer without hiding the conversion error."""
    writers = {}
    primary = None
    try:
        for name, path in temporaries.items():
            writers[name] = path.open("xb", buffering=0)
        yield writers
    except BaseException as exc:
        primary = exc
        raise
    finally:
        cleanup_error = None
        for writer in writers.values():
            try:
                writer.close()
            except BaseException as cleanup:
                if primary is not None:
                    add_failure_note(primary, "expert staging writer close failed: " + failure_summary(cleanup))
                elif cleanup_error is None:
                    cleanup_error = cleanup
                else:
                    add_failure_note(cleanup_error, "another expert staging writer close failed: " + failure_summary(cleanup))
        if primary is None and cleanup_error is not None:
            raise cleanup_error


def _cleanup_temporaries(paths,primary):
    pending=[]
    for path in paths:
        try:path.unlink(missing_ok=True)
        except BaseException as cleanup:
            pending.append(str(path));add_failure_note(primary, 'expert staging cleanup failed: '+failure_summary(cleanup))
    if pending:primary.uncommitted_staging=tuple(pending)


def _publish_containers(temporaries,destinations):
    """Publish a complete immutable set without replacing one member early.

    The manifest is returned only after all members exist. Exclusive links
    prevent a concurrent publisher from replacing an admitted destination.
    Single-file replacement retains its existing atomic update contract.
    """
    if len(destinations)==1:
        name=next(iter(destinations));temporaries[name].replace(destinations[name])
        from .checkpoint_recovery import _sync_directory
        _sync_directory(destinations[name].parent)
        return
    published=[]
    try:
        for name,temporary in temporaries.items():
            destination=destinations[name]
            os.link(temporary,destination)
            published.append((temporary,destination,_source_stamp(destination.stat())))
        from .checkpoint_recovery import _sync_directory
        for parent in {path.parent for path in destinations.values()}:_sync_directory(parent)
        for temporary in temporaries.values():temporary.unlink()
    except BaseException as primary:
        pending=[]
        for temporary,destination,stamp in reversed(published):
            try:
                # Removing a staged hard link changes ctime/link count without
                # changing the admitted published inode or payload identity.
                if (_source_stamp(destination.stat())[:4]!=stamp[:4] or
                        temporary.exists() and not os.path.samefile(temporary,destination)):
                    raise RuntimeError('published container changed before failed-set cleanup')
                destination.unlink()
            except BaseException as cleanup:
                pending.append(str(destination));add_failure_note(primary, 'uncommitted container cleanup failed: '+failure_summary(cleanup))
        if pending:primary.uncommitted_containers=tuple(pending)
        raise


def _admit_binding_inventory(layers,header,*,allowed_dtypes,uniform_layer_dtype=False,
        max_conversion_bytes=None,conversion_extra=0,target_dtype=None):
    """Admit every mapped original matrix before selected payload reads."""
    names=set()
    for layer in layers:
        if (type(layer) is not ExpertLayerBinding or type(layer.id) is not str or not layer.id or
                '/' in layer.id or layer.id in names or type(layer.experts) is not tuple or not layer.experts or
                any(type(value) is not int or value<=0 for value in
                    (layer.hidden_size,layer.intermediate_size,layer.experts_used)) or
                layer.experts_used>len(layer.experts)):
            raise ValueError('complete unique expert layer binding and geometry required')
        names.add(layer.id);experts=set();dtypes=set()
        if (target_dtype=='iq3-format6-98-v1' and (layer.hidden_size%256 or layer.intermediate_size%256) or
                target_dtype=='e8-lattice-caller-v1' and (layer.hidden_size%8 or layer.intermediate_size%8)):
            raise ValueError('complete mapped lattice packing block alignment required')
        for expert in layer.experts:
            if type(expert) is not ExpertTensorBinding or type(expert.id) is not str or not expert.id or expert.id in experts:
                raise ValueError('complete unique original expert binding required')
            experts.add(expert.id)
            for name,shape in ((expert.gate,[layer.intermediate_size,layer.hidden_size]),
                    (expert.up,[layer.intermediate_size,layer.hidden_size]),(expert.down,[layer.hidden_size,layer.intermediate_size])):
                if type(name) not in (str,TensorRangeBinding):raise ValueError('explicit original tensor name or bank view required')
                tensor=_tensor(header,name);dtype=tensor.get('dtype')
                if tensor.get('shape')!=shape or dtype not in allowed_dtypes:
                    raise ValueError('mapped original expert shape/dtype mismatch: '+str(name))
                dtypes.add(dtype)
                if max_conversion_bytes is not None and 64*shape[0]*shape[1]+4096+conversion_extra>max_conversion_bytes:
                    raise MemoryError('complete mapped expert conversion exceeds working admission')
        if uniform_layer_dtype and len(dtypes)!=1:raise ValueError('one original floating dtype per copied expert layer required')


def _unique_header(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate SafeTensors metadata field: '+key)
        result[key] = value
    return result


def _read_header(path: Path) -> tuple[dict[str, Any], int]:
    before=path.stat();size=before.st_size
    with path.open("rb") as source:
        if _source_stamp(os.fstat(source.fileno()))!=_source_stamp(before):
            raise ValueError('checkpoint source changed before header admission')
        raw_length = source.read(8)
        if len(raw_length) != 8:
            raise ValueError("SafeTensors header length is missing")
        length = int.from_bytes(raw_length, "little")
        if length <= 1 or length > min(size - 8, 256 * 1024 * 1024):
            raise ValueError("SafeTensors header length is invalid")
        try:
            header = json.loads(source.read(length).decode("utf-8"),object_pairs_hook=_unique_header,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("SafeTensors header is invalid JSON") from exc
        if (_source_stamp(os.fstat(source.fileno()))!=_source_stamp(before) or
                _source_stamp(path.stat())!=_source_stamp(before)):
            raise ValueError('checkpoint source changed during header admission')
    if not isinstance(header, dict):
        raise ValueError("SafeTensors header must be an object")
    data_base = 8 + length
    ranges=[]
    for name, tensor in header.items():
        if name == "__metadata__":
            if type(tensor) is not dict or any(type(key) is not str or type(value) is not str for key,value in tensor.items()):
                raise ValueError('SafeTensors user metadata must contain string fields')
            continue
        if not isinstance(tensor, dict):
            raise ValueError(f"tensor {name} metadata must be an object")
        offsets = tensor.get("data_offsets")
        shape = tensor.get("shape")
        if (
            not isinstance(offsets, list) or len(offsets) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in offsets)
            or offsets[0] < 0 or offsets[1] < offsets[0]
            or data_base + offsets[1] > size
            or not isinstance(shape, list)
            or len(shape)>8 or not all(type(value) is int and 0<=value<2**63 for value in shape)
        ):
            raise ValueError(f"tensor {name} metadata is invalid")
        dtype = str(tensor.get("dtype", ""))
        if dtype not in _HEADER_ITEM_BYTES:raise ValueError('unadmitted original SafeTensors dtype: '+dtype)
        elements=1
        for dimension in shape:elements*=dimension
        if offsets[1]-offsets[0]!=elements*_HEADER_ITEM_BYTES[dtype]:
            raise ValueError(f"tensor {name} byte range does not match its shape and dtype")
        if offsets[1]>offsets[0]:ranges.append(tuple(offsets))
    end=0
    for first,last in sorted(ranges):
        if first!=end:raise ValueError('SafeTensors original payload has overlapping or missing tensor ranges')
        end=last
    if data_base+end!=size:raise ValueError('SafeTensors payload extent differs from its complete header inventory')
    return header, data_base


def _scan_directory(root: Path) -> dict[str, tuple[Path, dict[str, Any], int]]:
    index: dict[str, tuple[Path, dict[str, Any], int]] = {}
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise ValueError("SafeTensors directory contains no shards")
    for shard in shards:
        stamp = _source_stamp(shard.stat())
        header, data_base = _read_header(shard)
        if _source_stamp(shard.stat()) != stamp:
            raise ValueError('checkpoint shard changed during header admission')
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if name in index:
                raise ValueError(f"tensor appears in more than one shard: {name}")
            index[name] = (shard, dict(tensor, _source_stamp=stamp), data_base)
    return index


def _read_float_matrix(index: dict[str, tuple[Path, dict[str, Any], int]], name,
                       *, expected_shape=None, max_bytes=1024*1024*1024):
    import numpy as np

    source_name = name.name if type(name) is TensorRangeBinding else name
    record = index.get(source_name)
    if record is None:
        raise ValueError(f"mapped tensor not found: {name}")
    path, tensor, data_base = record
    stamp = tensor.get('_source_stamp')
    if type(name) is TensorRangeBinding:
        tensor = _tensor({source_name: tensor}, name)
    dtype = str(tensor.get("dtype", ""))
    if dtype not in _SOURCE_ITEM_BYTES:
        raise ValueError(f"tensor {name} uses unsupported source dtype {dtype}")
    shape = tensor['shape']
    if (not isinstance(shape, list) or not shape or
            any(type(value) is not int or value <= 0 for value in shape) or
            (expected_shape is not None and shape != list(expected_shape))):
        raise ValueError(f'tensor {name} has an invalid mapped shape')
    start, end = tensor["data_offsets"]
    elements = math.prod(shape)
    if end-start != elements*_SOURCE_ITEM_BYTES[dtype]:
        raise ValueError('source matrix byte extent differs from its shape')
    if type(max_bytes) is not int or max_bytes <= 0 or elements*12 > max_bytes:
        raise MemoryError('source matrix decoding exceeds admission')
    with path.open("rb", buffering=0) as source:
        if stamp is not None and _source_stamp(os.fstat(source.fileno())) != stamp:
            raise ValueError('checkpoint shard changed before matrix read')
        source.seek(data_base + start)
        raw = source.read(end - start)
        if stamp is not None and (_source_stamp(os.fstat(source.fileno())) != stamp or
                _source_stamp(path.stat()) != stamp):
            raise ValueError('checkpoint shard changed during matrix read')
    if len(raw) != end - start:
        raise OSError(f"tensor {name} ended before its declared byte range")
    if dtype == "BF16":
        words = np.frombuffer(raw, dtype="<u2")
        values = (words.astype(np.uint32) << np.uint32(16)).view("<f4")
    else:
        values = np.frombuffer(raw, dtype="<f2" if dtype == "F16" else "<f4").astype(np.float32)
    return values.reshape(tuple(tensor["shape"]))


def _tensor(header: dict[str, Any], name) -> dict[str, Any]:
    source_name = name.name if type(name) is TensorRangeBinding else name
    value = header.get(source_name)
    if not isinstance(value, dict):
        raise ValueError(f"mapped tensor not found: {name}")
    if type(name) is TensorRangeBinding:
        import math
        if (type(name.name) is not str or not name.name or
                type(name.source_shape) is not tuple or not name.source_shape or
                any(type(extent) is not int or extent <= 0 for extent in name.source_shape) or
                tuple(value['shape']) != name.source_shape or
                type(name.shape) is not tuple or len(name.shape) != 2 or
                any(type(extent) is not int or extent <= 0 for extent in name.shape) or
                type(name.offset_elements) is not int or name.offset_elements < 0):
            raise ValueError('invalid explicit source-bank matrix view')
        width = _SOURCE_ITEM_BYTES.get(value['dtype'])
        if width is None:
            raise ValueError('unsupported source-bank matrix encoding')
        count = math.prod(name.shape)
        first, last = value['data_offsets']
        if (last-first != math.prod(name.source_shape)*width or
                name.offset_elements+count > math.prod(name.source_shape)):
            raise ValueError('source-bank matrix view exceeds its tensor extent')
        first += name.offset_elements*width
        return dict(dtype=value['dtype'], shape=list(name.shape),
                    data_offsets=[first, first+count*width])
    return value


def _align(destination: BinaryIO, alignment: int) -> None:
    padding = (-destination.tell()) % alignment
    if padding:
        _write_all(destination,bytes(padding))


def _write_all(destination,value):
    view=memoryview(value).cast('B')
    while view:
        count=destination.write(view)
        if type(count) is not int or not 0<count<=len(view):raise OSError('expert container writer made invalid progress')
        view=view[count:]


def _copy_range(source: BinaryIO,destination: BinaryIO,offset: int,size: int,digest: Any,*,dtype) -> None:
    import numpy as np
    source.seek(offset)
    remaining = size
    while remaining:
        block = source.read(min(remaining, 1024 * 1024))
        if not block:
            raise OSError("SafeTensors payload ended before the declared tensor range")
        if not np.isfinite(np.frombuffer(block,dtype='<f2' if dtype=='F16' else '<f4')).all():
            raise ValueError('original copied expert payload contains nonfinite weights')
        _write_all(destination,block)
        digest.update(block)
        remaining -= len(block)
