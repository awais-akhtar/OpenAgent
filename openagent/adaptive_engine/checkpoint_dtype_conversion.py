"""Bounded, atomic materialization of FP16 or FP32 from a local BF16 checkpoint.

The converter preserves every tensor name, shape and shard assignment. It
updates tensor extents and the index's ``total_size`` when expanding BF16 to
F32. It changes only BF16 payloads, corresponding SafeTensors dtype tags, and
the checkpoint's declared ``torch_dtype``. The source tree is read-only; the
output must be a new sibling directory. This supports route validation and
deployment preparation; it is not a model quantizer.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

from .expert_packing import _HEADER_ITEM_BYTES, _read_header, _source_stamp


class CheckpointConversionError(RuntimeError):
    """The checkpoint could not be converted without violating its contract."""


def _atomic_json(path: Path, document: dict) -> None:
    temporary = path.with_name(path.name + ".tmp-" + uuid4().hex)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException as error:
        temporary.unlink(missing_ok=True)
        raise error


def _convert_bf16_payload(raw: bytes, target_dtype: str):
    import numpy as np

    words = np.frombuffer(raw, dtype="<u2")
    values = (words.astype(np.uint32) << np.uint32(16)).view("<f4")
    if not np.isfinite(values).all():
        raise CheckpointConversionError("BF16 source tensor contains a non-finite value")
    if target_dtype == "F16":
        with np.errstate(over="ignore", invalid="ignore"):
            converted = values.astype("<f2")
        if not np.isfinite(converted).all():
            raise CheckpointConversionError("FP16 conversion overflows a finite BF16 source value")
    elif target_dtype == "F32":
        converted = values
    else:
        raise ValueError("destination dtype must be F16 or F32")
    return converted.tobytes(order="C")


def _convert_shard(source: Path, destination: Path, *, chunk_bytes: int,
                   target_dtype: str) -> tuple[int, int, int]:
    before = _source_stamp(source.stat())
    header, source_data_base = _read_header(source)
    tensors = [(name, record) for name, record in header.items() if name != "__metadata__"]
    bf16_count = sum(record["dtype"] == "BF16" for _, record in tensors)
    if not bf16_count:
        raise CheckpointConversionError(f"SafeTensors shard contains no BF16 tensors: {source.name}")
    bf16_names = {name for name, record in tensors if record["dtype"] == "BF16"}
    for _, record in tensors:
        if record["dtype"] not in _HEADER_ITEM_BYTES:
            raise CheckpointConversionError("source shard contains an unsupported SafeTensors dtype")
        if record["dtype"] == "BF16":
            record["dtype"] = target_dtype
    ordered = sorted((record["data_offsets"][0], record["data_offsets"][1], name, record)
                     for name, record in tensors)
    expected_data_bytes = source.stat().st_size - source_data_base
    if sum(end - start for start, end, _, _ in ordered) != expected_data_bytes:
        raise CheckpointConversionError("tensor ranges do not cover the complete SafeTensors payload")
    source_cursor = output_cursor = 0
    output_data_bytes = 0
    for first, last, name, record in ordered:
        if first != source_cursor or last < first:
            raise CheckpointConversionError("source tensor ranges contain a gap, overlap, or reversal")
        extent = last - first
        if name in bf16_names:
            if extent % 2:
                raise CheckpointConversionError("BF16 tensor has a partial element extent")
            extent = extent // 2 * _HEADER_ITEM_BYTES[target_dtype]
        record["data_offsets"] = [output_cursor, output_cursor + extent]
        source_cursor = last
        output_cursor += extent
    output_data_bytes = output_cursor
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    tensors_written = 0
    with source.open("rb", buffering=0) as reader, destination.open("xb", buffering=0) as writer:
        if _source_stamp(os.fstat(reader.fileno())) != before:
            raise CheckpointConversionError("source shard changed before conversion")
        writer.write(len(encoded).to_bytes(8, "little"))
        writer.write(encoded)
        cursor = 0
        for first, last, name, record in ordered:
            if first != cursor:
                raise CheckpointConversionError("source tensor ranges contain a gap or overlap")
            reader.seek(source_data_base + first)
            remaining = last - first
            if name in bf16_names:
                while remaining:
                    amount = min(remaining, chunk_bytes)
                    amount -= amount % 2
                    if amount <= 0:
                        raise CheckpointConversionError("conversion chunk must contain a complete BF16 value")
                    raw = reader.read(amount)
                    if len(raw) != amount:
                        raise CheckpointConversionError("source ended inside a BF16 tensor")
                    writer.write(_convert_bf16_payload(raw, target_dtype))
                    remaining -= amount
                tensors_written += 1
            else:
                while remaining:
                    amount = min(remaining, chunk_bytes)
                    raw = reader.read(amount)
                    if len(raw) != amount:
                        raise CheckpointConversionError("source ended inside a tensor")
                    writer.write(raw)
                    remaining -= amount
            cursor = last
        if _source_stamp(os.fstat(reader.fileno())) != before or _source_stamp(source.stat()) != before:
            raise CheckpointConversionError("source shard changed during conversion")
        writer.flush()
        os.fsync(writer.fileno())
    if destination.stat().st_size != 8 + len(encoded) + output_data_bytes:
        raise CheckpointConversionError("converted shard size differs from its complete tensor inventory")
    return tensors_written, destination.stat().st_size, output_data_bytes


def convert_bf16_checkpoint(source_directory: str | Path,
                            destination_directory: str | Path, *, target_dtype: str,
                            chunk_bytes: int = 32 * 1024 * 1024) -> dict:
    """Convert all BF16 tensors to F16 or F32 with bounded atomic publication."""
    if target_dtype not in {"F16", "F32"}:
        raise ValueError("target_dtype must be F16 or F32")
    source = Path(source_directory).resolve(strict=True)
    destination = Path(destination_directory).resolve()
    if (not source.is_dir() or source == destination or destination.exists() or
            source.parent != destination.parent):
        raise ValueError("source checkpoint and unused sibling destination directory are required")
    if type(chunk_bytes) is not int or chunk_bytes < 2 or chunk_bytes % 2:
        raise ValueError("chunk_bytes must be a positive even integer")
    shards = sorted(source.glob("*.safetensors"))
    if not shards:
        raise ValueError("checkpoint contains no SafeTensors shards")
    config_path = source / "config.json"
    index_path = source / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise ValueError("checkpoint config and SafeTensors index are required")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("torch_dtype") != "bfloat16":
        raise ValueError("checkpoint config must declare torch_dtype=bfloat16")
    staging = destination.with_name("." + destination.name + ".stage-" + uuid4().hex)
    if staging.parent != destination.parent or not staging.name.startswith("." + destination.name + ".stage-"):
        raise ValueError("invalid checkpoint staging path")
    staging.mkdir(parents=False, exist_ok=False)
    total_tensors = total_bytes = total_payload_bytes = 0
    try:
        for shard in shards:
            count, size, data_bytes = _convert_shard(
                shard, staging / shard.name, chunk_bytes=chunk_bytes, target_dtype=target_dtype)
            total_tensors += count
            total_bytes += size
            total_payload_bytes += data_bytes
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if type(index) is not dict:
            raise CheckpointConversionError("SafeTensors index must be a JSON object")
        metadata = index.get("metadata")
        if metadata is not None:
            if type(metadata) is not dict:
                raise CheckpointConversionError("SafeTensors index metadata must be a JSON object")
            if "total_size" in metadata:
                metadata["total_size"] = total_payload_bytes
        _atomic_json(staging / "model.safetensors.index.json", index)
        converted_config = dict(config)
        converted_config["torch_dtype"] = "float16" if target_dtype == "F16" else "float32"
        _atomic_json(staging / "config.json", converted_config)
        for name in ("generation_config.json", "special_tokens_map.json",
                     "tokenizer.json", "tokenizer_config.json"):
            item = source / name
            if item.is_file():
                shutil.copy2(item, staging / name)
        if not (staging / "tokenizer.json").is_file():
            raise CheckpointConversionError("checkpoint tokenizer.json is required")
        if destination.exists():
            raise FileExistsError("destination appeared while conversion was running")
        staging.replace(destination)
    except BaseException as error:
        try:
            if staging.parent == destination.parent and staging.name.startswith("." + destination.name + ".stage-"):
                shutil.rmtree(staging)
        except BaseException as cleanup:
            if hasattr(error, "add_note"):
                error.add_note("staging cleanup failed: " + str(cleanup))
        raise
    return {"source": str(source), "destination": str(destination),
            "converted_bf16_tensors": total_tensors, "safetensors_shards": len(shards),
            "safetensors_bytes": total_bytes, "source_dtype": "BF16", "destination_dtype": target_dtype}


def convert_bf16_checkpoint_to_fp16(source_directory: str | Path,
                                    destination_directory: str | Path, *,
                                    chunk_bytes: int = 32 * 1024 * 1024) -> dict:
    return convert_bf16_checkpoint(source_directory, destination_directory,
                                   target_dtype="F16", chunk_bytes=chunk_bytes)


def convert_bf16_checkpoint_to_fp32(source_directory: str | Path,
                                    destination_directory: str | Path, *,
                                    chunk_bytes: int = 32 * 1024 * 1024) -> dict:
    return convert_bf16_checkpoint(source_directory, destination_directory,
                                   target_dtype="F32", chunk_bytes=chunk_bytes)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--target-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--chunk-mib", type=int, default=32)
    args = parser.parse_args(argv)
    result = convert_bf16_checkpoint(args.source, args.destination,
        target_dtype="F16" if args.target_dtype == "fp16" else "F32",
        chunk_bytes=args.chunk_mib * 1024 * 1024)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
