"""Independent integration contract probes; no quantization oracle is copied."""

import inspect
import json
from dataclasses import replace
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.expert_kernels import ExpertKernelRegistry
from openagent.adaptive_engine.expert_packing import pack_safetensors_directory_int8
from openagent.adaptive_engine.expert_store import DiskExpertStore
from openagent.adaptive_engine.olmoe import OlmoeModel, pack_olmoe_experts
from openagent.adaptive_engine.olmoe_cli import build_parser
from openagent.adaptive_engine.q5_0_expert import Q5_0ExpertBackend
from tests.test_q8_0_olmoe import checkpoint


def poison_tensor(root, name, value):
    path = root / "model.safetensors"
    with path.open("r+b") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(length))
        stream.seek(8 + length + header[name]["data_offsets"][0])
        stream.write(struct.pack("<f", value))


class MutableQ5Backend(Q5_0ExpertBackend):
    NAME = "q5-integration-mutable"
    available = True

    def info(self):
        return replace(super().info(), name=self.NAME, available=self.available,
                       unavailable_reason="" if self.available else "disabled after load")


class Q5IntegrationAdversarialTests(unittest.TestCase):
    def _assert_postload_backend_loss_is_strict(self, batch):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint(root)
            manifest = pack_olmoe_experts(root, root / "experts.q5", dtype="q5_0")
            for mutation in ("unavailable", "unregistered"):
                with self.subTest(mutation=mutation):
                    selected = MutableQ5Backend()
                    registry = ExpertKernelRegistry([selected])
                    fallback = registry.resolve(dtype="q5_0", preferred="q5_0-reference",
                                                allow_fallback=False)
                    with OlmoeModel(root, manifest, kernels=registry,
                                    preferred_kernel=selected.NAME,
                                    expert_cache_bytes=20000) as model:
                        runtime = model.experts
                        layer = manifest.layers[0]
                        inputs = np.zeros((2, layer.hidden_size) if batch else
                                          (layer.hidden_size,), np.float32)
                        routes = [[("0", 1.0)], [("1", 1.0)]] if batch else [("0", 1.0)]
                        execute = runtime.execute_batch if batch else runtime.execute
                        self.assertTrue(np.isfinite(execute(layer.id, inputs, routes)).all())
                        self.assertFalse(runtime.allow_kernel_fallback)
                        if mutation == "unavailable":
                            selected.available = False
                            error, message = RuntimeError, "unavailable or incompatible"
                        else:
                            registry.unregister(selected.NAME)
                            error, message = KeyError, selected.NAME
                        with patch.object(fallback, "execute") as scalar_fallback, \
                                patch.object(fallback, "execute_batch") as batch_fallback:
                            with self.assertRaisesRegex(error, message):
                                execute(layer.id, inputs, routes)
                            scalar_fallback.assert_not_called()
                            batch_fallback.assert_not_called()

    def test_scalar_runtime_selected_backend_lost_after_model_load_raises(self):
        self._assert_postload_backend_loss_is_strict(batch=False)

    def test_batch_runtime_selected_backend_lost_after_model_load_raises(self):
        self._assert_postload_backend_loss_is_strict(batch=True)

    def test_existing_packing_defaults_preserved(self):
        self.assertEqual(build_parser().parse_args(["pack", "model"]).dtype, "int8-row")
        for function in (pack_olmoe_experts, pack_safetensors_directory_int8):
            self.assertEqual(inspect.signature(function).parameters["dtype"].default, "int8-row")

    def test_registry_rejects_unknown_dtype_and_strict_incompatible_backend(self):
        registry = ExpertKernelRegistry()
        try:
            with self.assertRaises(ValueError):
                registry.resolve(dtype="q5_typo")
            for preferred in ("numpy-reference", "q4_0-reference", "missing-kernel"):
                with self.subTest(preferred=preferred), self.assertRaises((RuntimeError, KeyError)):
                    registry.resolve(dtype="q5_0", preferred=preferred, allow_fallback=False)
            registry.unregister("q5_0-reference")
            with self.assertRaises(RuntimeError):
                registry.resolve(dtype="q5_0")
        finally:
            registry.close()

    def test_explicit_incompatible_model_backend_must_not_silently_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint(root)
            manifest = pack_olmoe_experts(root, root / "experts.q5", dtype="q5_0")
            for preferred in ("numpy-reference", "q4_0-reference", "missing-kernel"):
                with self.subTest(preferred=preferred), self.assertRaises((ValueError, RuntimeError, KeyError)):
                    with OlmoeModel(root, manifest, expert_cache_bytes=20000,
                                    preferred_kernel=preferred) as model:
                        model.forward([1])

    def test_explicit_q5_backend_must_reject_int8_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint(root)
            manifest = pack_olmoe_experts(root, root / "experts.int8")
            with self.assertRaises((ValueError, RuntimeError)):
                with OlmoeModel(root, manifest, expert_cache_bytes=100000,
                                preferred_kernel="q5_0-reference") as model:
                    model.forward([1])

    def test_nonfinite_expert_sources_rejected_without_replacing_destination(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint(root)
                poison_tensor(root, "model.layers.0.mlp.experts.0.gate_proj.weight", value)
                destination = root / "experts.q5"
                destination.write_bytes(b"existing-output")
                with self.assertRaisesRegex(ValueError, "finite"):
                    pack_olmoe_experts(root, destination, dtype="q5_0")
                self.assertEqual(destination.read_bytes(), b"existing-output")
                self.assertFalse((root / "experts.q5.tmp").exists())

    def test_gateway_explicit_incompatible_backend_must_not_fallback(self):
        from openagent.model_gateway import OlmoeDirectProvider

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint(root)
            manifest = pack_olmoe_experts(root, root / "experts.q5", dtype="q5_0")
            path = manifest.write(root / "manifest.json")
            provider = OlmoeDirectProvider()
            provider.model_path = str(root)
            provider.manifest_path = str(path)
            provider.preferred_kernel = "numpy-reference"
            provider.cache_gb = 0.001
            provider.context_tokens = 16
            try:
                with self.assertRaises((ValueError, RuntimeError, KeyError)):
                    provider._load().forward([1])
            finally:
                if provider._runtime is not None:
                    provider._runtime.close()

    def test_nonfinite_dense_source_must_be_rejected_at_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint(root)
            poison_tensor(root, "model.embed_tokens.weight", float("nan"))
            manifest = pack_olmoe_experts(root, root / "experts.q5", dtype="q5_0")
            with self.assertRaisesRegex(ValueError, "finite"):
                with OlmoeModel(root, manifest, expert_cache_bytes=20000):
                    pass

    def test_corrupt_q5_payload_rejected_on_first_acquire(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint(root)
            manifest = pack_olmoe_experts(root, root / "experts.q5", dtype="q5_0")
            region = manifest.layers[0].regions["0"]
            with (root / region.path).open("r+b") as stream:
                stream.seek(region.offset + 6)
                original = stream.read(1)
                stream.seek(region.offset + 6)
                stream.write(bytes([original[0] ^ 1]))
            for mode in ("buffered", "mmap"):
                with self.subTest(mode=mode), DiskExpertStore(
                        root, {"expert": region}, capacity_bytes=region.size,
                        io_mode=mode) as store:
                    with self.assertRaisesRegex(ValueError, "checksum"):
                        with store.acquire("expert"):
                            pass

    def test_checksum_must_be_revalidated_after_eviction_and_reread(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint(root)
            manifest = pack_olmoe_experts(root, root / "experts.q5", dtype="q5_0")
            region = manifest.layers[0].regions["0"]
            with DiskExpertStore(root, {"expert": region}, capacity_bytes=region.size,
                                 io_mode="buffered") as store:
                with store.acquire("expert"):
                    pass
                self.assertTrue(store.evict("expert"))
                with (root / region.path).open("r+b") as stream:
                    stream.seek(region.offset + 6)
                    original = stream.read(1)
                    stream.seek(region.offset + 6)
                    stream.write(bytes([original[0] ^ 1]))
                with self.assertRaisesRegex(ValueError, "checksum|file changed"):
                    with store.acquire("expert"):
                        pass


if __name__ == "__main__":
    unittest.main()
