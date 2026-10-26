"""Bounded, transactional SafeTensors BF16-to-FP16 conversion contracts."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from openagent.adaptive_engine.checkpoint_dtype_conversion import (
    CheckpointConversionError, convert_bf16_checkpoint_to_fp16, convert_bf16_checkpoint_to_fp32,
)
from openagent.adaptive_engine.expert_packing import _read_header


def bf16_bytes(values):
    words = np.asarray(values, dtype="<f4").view("<u4")
    rounded = words + np.uint32(0x7FFF) + ((words >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype("<u2").tobytes()


def write_safetensors(path: Path, tensors):
    header, payload, offset = {}, bytearray(), 0
    for name, dtype, shape, raw in tensors:
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + len(raw)]}
        payload.extend(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


class CheckpointDtypeConversionTests(unittest.TestCase):
    def _source(self, root, bf_values, *, include_half=True):
        root.mkdir()
        (root / "config.json").write_text(json.dumps({"model_type": "olmoe",
            "torch_dtype": "bfloat16"}), encoding="utf-8")
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": 8}, "weight_map": {"bf": "model-00001-of-00001.safetensors"}
        }), encoding="utf-8")
        (root / "tokenizer.json").write_text("{}", encoding="utf-8")
        tensors = [("bf", "BF16", (len(bf_values),), bf16_bytes(bf_values))]
        if include_half:
            half = np.array([1.25, -2.5], dtype="<f2")
            tensors.append(("already_half", "F16", half.shape, half.tobytes()))
        integer = np.array([7, 11], dtype="<i8")
        tensors.append(("ids", "I64", integer.shape, integer.tobytes()))
        shard = root / "model-00001-of-00001.safetensors"
        write_safetensors(shard, tensors)
        return shard, tensors

    def test_converts_only_bf16_and_preserves_inventory_with_tiny_chunks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, expected = root / "source", root / "converted"
            self._source(source, [1.0, -3.25, .125, 7.5])
            result = convert_bf16_checkpoint_to_fp16(source, expected, chunk_bytes=4)
            self.assertEqual(result["converted_bf16_tensors"], 1)
            self.assertEqual(result["safetensors_shards"], 1)
            self.assertEqual(json.loads((expected / "config.json").read_text())["torch_dtype"], "float16")
            header, data_base = _read_header(expected / "model-00001-of-00001.safetensors")
            self.assertEqual(header["bf"]["dtype"], "F16")
            self.assertEqual(header["already_half"]["dtype"], "F16")
            self.assertEqual(header["ids"]["dtype"], "I64")
            with (expected / "model-00001-of-00001.safetensors").open("rb") as stream:
                stream.seek(data_base + header["bf"]["data_offsets"][0])
                actual_bf = np.frombuffer(stream.read(8), dtype="<f2")
                stream.seek(data_base + header["already_half"]["data_offsets"][0])
                actual_half = stream.read(4)
                stream.seek(data_base + header["ids"]["data_offsets"][0])
                actual_ids = stream.read(16)
            np.testing.assert_array_equal(actual_bf, np.array([1.0, -3.25, .125, 7.5], dtype="<f2"))
            np.testing.assert_array_equal(np.frombuffer(actual_half, dtype="<f2"), [1.25, -2.5])
            np.testing.assert_array_equal(np.frombuffer(actual_ids, dtype="<i8"), [7, 11])

    def test_fp16_overflow_cleans_staging_without_publishing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / "source", root / "converted"
            self._source(source, [70000.0], include_half=False)
            with self.assertRaises(CheckpointConversionError):
                convert_bf16_checkpoint_to_fp16(source, destination, chunk_bytes=2)
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".converted.stage-*")), [])

    def test_fp32_materialization_reflows_offsets_and_updates_index_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / "source", root / "converted"
            self._source(source, [1.0, -3.25, .125, 7.5])
            result = convert_bf16_checkpoint_to_fp32(source, destination, chunk_bytes=4)
            self.assertEqual(result["destination_dtype"], "F32")
            self.assertEqual(json.loads((destination / "config.json").read_text())["torch_dtype"],
                             "float32")
            shard_path = destination / "model-00001-of-00001.safetensors"
            header, data_base = _read_header(shard_path)
            self.assertEqual(header["bf"]["dtype"], "F32")
            self.assertEqual(header["bf"]["data_offsets"], [0, 16])
            self.assertEqual(header["already_half"]["data_offsets"], [16, 20])
            self.assertEqual(header["ids"]["data_offsets"], [20, 36])
            with shard_path.open("rb") as stream:
                stream.seek(data_base)
                bf_actual = np.frombuffer(stream.read(16), dtype="<f4")
                half_actual = stream.read(4)
                ids_actual = stream.read(16)
            np.testing.assert_array_equal(bf_actual, [1.0, -3.25, .125, 7.5])
            np.testing.assert_array_equal(np.frombuffer(half_actual, dtype="<f2"), [1.25, -2.5])
            np.testing.assert_array_equal(np.frombuffer(ids_actual, dtype="<i8"), [7, 11])
            index = json.loads((destination / "model.safetensors.index.json").read_text())
            self.assertEqual(index["metadata"]["total_size"], 36)

    def test_rejects_existing_destination_and_invalid_chunk_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / "source", root / "existing"
            self._source(source, [1.0], include_half=False)
            destination.mkdir()
            (destination / "sentinel").write_text("preserve", encoding="utf-8")
            with self.assertRaises(ValueError):
                convert_bf16_checkpoint_to_fp16(source, destination)
            self.assertEqual((destination / "sentinel").read_text(encoding="utf-8"), "preserve")
            with self.assertRaises(ValueError):
                convert_bf16_checkpoint_to_fp16(source, root / "bad-chunk", chunk_bytes=3)

    def test_rejects_nested_destination_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            shard, _ = self._source(source, [1.0], include_half=False)
            with self.assertRaisesRegex(ValueError, "sibling destination"):
                convert_bf16_checkpoint_to_fp16(source, source / "nested-output", chunk_bytes=2)
            self.assertEqual(sorted(source.iterdir(), key=lambda path: path.name), sorted(
                [source / "config.json", source / "model.safetensors.index.json",
                 source / "tokenizer.json", shard], key=lambda path: path.name))


if __name__ == "__main__":
    unittest.main()
