"""Exact manifest bytes, independent decoded execution and snapshot identity."""

from dataclasses import replace
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.expert_kernels import (
    ExpertKernelRegistry, MappingPayloadSource, NumPyRoutedSwiGLUBackend, _lease,
)
from openagent.adaptive_engine.olmoe import OlmoeModel, pack_olmoe_experts
from openagent.adaptive_engine.olmoe_cli import main, _kernel_resources
from openagent.adaptive_engine import q8_1_expert, grouped_int4_expert, mxfp4_expert, lowbit_expert
from openagent.adaptive_engine.native_q8_1_expert import NativeQ8_1ExpertBackend
from openagent.adaptive_engine.native_grouped_int4_expert import NativeGroupedInt4ExpertBackend
from openagent.adaptive_engine.native_mxfp4_expert import NativeMXFP4ExpertBackend
from openagent.adaptive_engine.native_lowbit_expert import NativeLowBitExpertBackend
from tests.test_q8_0_olmoe import checkpoint
from tests.test_q8_1_expert import oracle as q8_oracle
from tests.test_lowbit_expert import oracle as lowbit_oracle
from tests.test_grouped_int4_expert import oracle_decode as grouped_oracle


def mxfp_oracle(payload, hidden, intermediate):
    position, result = 0, []
    levels = (0., .5, 1., 1.5, 2., 3., 4., 6.)
    for rows, columns in ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate)):
        matrix = np.empty((rows, columns), np.float32)
        for row in range(rows):
            for start in range(0, columns, 32):
                scale = 2. ** (payload[position] - 127)
                for lane in range(min(32, columns - start)):
                    code = (payload[position + 1 + lane // 2] >> ((lane % 2) * 4)) & 15
                    matrix[row, start + lane] = np.float32(scale * levels[code & 7] * (-1 if code & 8 else 1))
                position += 17
        result.append(matrix)
    return result


FORMATS = (
    ("q8_1", q8_1_expert, q8_1_expert.Q8_1ExpertBackend(), NativeQ8_1ExpertBackend(), q8_oracle, "q81"),
    ("int4-grouped-g64", grouped_int4_expert, grouped_int4_expert.GroupedInt4ExpertBackend(),
     NativeGroupedInt4ExpertBackend(), lambda data, h, i: grouped_oracle(data, h, i, 64), "gi4"),
    ("mxfp4-e2m1-g32-ue8m0", mxfp4_expert, mxfp4_expert.MXFP4ExpertBackend(),
     NativeMXFP4ExpertBackend(), mxfp_oracle, "mxfp4"),
    ("int2-row", lowbit_expert, lowbit_expert.LowBitExpertBackend("int2-row"),
     NativeLowBitExpertBackend("int2-row"), lambda data, h, i: lowbit_oracle(data, h, i, "int2-row"), "i2"),
    ("int3-grouped-g64", lowbit_expert, lowbit_expert.LowBitExpertBackend("int3-grouped-g64"),
     NativeLowBitExpertBackend("int3-grouped-g64"), lambda data, h, i: lowbit_oracle(data, h, i, "int3-grouped-g64"), "gi3"),
)


class DecodedOracle:
    def __init__(self, backend, decode):
        self.backend, self.decode = backend, decode
        self.NAME = backend.NAME + "-oracle"

    def info(self):
        return replace(self.backend.info(), name=self.NAME)

    def execute(self, source, inputs, routes, **kwargs):
        return self.execute_batch(source, np.asarray(inputs)[None], [routes], **kwargs)[0]

    def execute_batch(self, source, inputs, route_batch, *, hidden_size, intermediate_size, dtype):
        blobs = {}
        for routes in route_batch:
            for key, _ in routes:
                if key not in blobs:
                    with _lease(source, key) as payload:
                        matrices = self.decode(bytes(payload), hidden_size, intermediate_size)
                    blobs[key] = b"".join(matrix.astype("<f4").tobytes() for matrix in matrices)
        return NumPyRoutedSwiGLUBackend().execute_batch(MappingPayloadSource(blobs), inputs, route_batch,
            hidden_size=hidden_size, intermediate_size=intermediate_size, dtype="float32")


class ExtendedPackedModelTests(unittest.TestCase):
    def test_real_pack_full_and_cached_logits_against_independent_decoder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors = checkpoint(root)
            previous_format_state = None
            for dtype, module, reference, native, decode, extension in FORMATS:
                with self.subTest(dtype=dtype):
                    manifest = pack_olmoe_experts(root, root / ("experts." + extension), dtype=dtype)
                    for layer in manifest.layers:
                        for key, region in layer.regions.items():
                            with (root / region.path).open("rb") as stream:
                                stream.seek(region.offset)
                                payload = stream.read(region.size)
                            matrices = [tensors[f"{layer.id}.mlp.experts.{key}.{role}_proj.weight"]
                                        for role in ("gate", "up", "down")]
                            kwargs = {"dtype": dtype} if module is lowbit_expert else {}
                            self.assertEqual(payload, module.pack_expert(*matrices, **kwargs))
                            self.assertEqual(len(region.sha256), 64)
                    oracle = DecodedOracle(reference, decode)
                    choices = [reference] + ([native] if native.info().available else [])
                    for backend in choices:
                        registry = ExpertKernelRegistry()
                        registry.register(backend, replace=True)
                        with OlmoeModel(root, manifest, kernels=registry, preferred_kernel=backend.NAME,
                                        context_tokens=16, expert_cache_bytes=1024 * 1024) as actual, OlmoeModel(
                                root, manifest, kernels=ExpertKernelRegistry([oracle]), preferred_kernel=oracle.NAME,
                                context_tokens=16, expert_cache_bytes=1024 * 1024) as expected:
                            for tokens in ([1, 4], [7], [3, 9]):
                                np.testing.assert_allclose(actual.forward(tokens), expected.forward(tokens),
                                                           rtol=2e-4, atol=2e-5)
                            snapshot = actual.snapshot_state()
                            self.assertEqual(len(snapshot["execution_identity"]), 64)
                            if previous_format_state is not None:
                                position_before = actual.position
                                keys_before = [value.copy() for value in actual._k]
                                with self.assertRaisesRegex(ValueError, "precision/backend"):
                                    actual.restore_state(previous_format_state)
                                self.assertEqual(actual.position, position_before)
                                for value, previous in zip(actual._k, keys_before):
                                    np.testing.assert_array_equal(value, previous)
                            changed = dict(snapshot)
                            changed["execution_identity"] = "invalid"
                            position = actual.position
                            with self.assertRaisesRegex(ValueError, "precision/backend"):
                                actual.restore_state(changed)
                            self.assertEqual(actual.position, position)
                            continuation = actual.forward([2])
                            actual.restore_state(snapshot)
                            np.testing.assert_array_equal(actual.forward([2]), continuation)
                    previous_format_state = snapshot

    def test_cli_gateway_native_selection_and_distinct_defaults(self):
        from openagent.model_gateway import OlmoeDirectProvider
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint(root)
            protected = root / "openagent-experts.q8"
            protected.write_bytes(b"previous-format")
            for dtype, _, _, native, _, extension in FORMATS:
                with self.subTest(dtype=dtype):
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(main(["pack", str(root), "--dtype", dtype]), 0)
                    self.assertEqual(protected.read_bytes(), b"previous-format")
                    if not native.info().available:
                        continue
                    registry, resource = _kernel_resources(root, native.NAME, 1)
                    self.assertIsNone(resource)
                    self.assertTrue(registry.resolve(dtype=dtype, preferred=native.NAME, allow_fallback=False).info().capabilities.compiled)
                    manifest = root / ("openagent-experts." + extension + ".manifest.json")
                    with patch.dict(os.environ, {"OPENAGENT_OLMOE_MODEL_DIR": str(root),
                            "OPENAGENT_OLMOE_MANIFEST": str(manifest), "OPENAGENT_OLMOE_KERNEL": native.NAME,
                            "OPENAGENT_OLMOE_CONTEXT": "16", "OPENAGENT_OLMOE_EXPERT_CACHE_GB": "0.001",
                            "OPENAGENT_OLMOE_IO_MODE": "mmap", "OPENAGENT_OLMOE_STORAGE_ROOTS": ""}):
                        provider = OlmoeDirectProvider()
                        try:
                            loaded = provider._load()
                            with OlmoeModel(root, manifest, kernels=registry, preferred_kernel=native.NAME,
                                            context_tokens=16, expert_cache_bytes=1024 * 1024) as direct:
                                np.testing.assert_array_equal(loaded.forward([1, 4]), direct.forward([1, 4]))
                        finally:
                            if provider._runtime is not None:
                                provider._runtime.close()

    def test_unscoped_group_size_not_persisted(self):
        from openagent.adaptive_engine.expert_runtime import ExpertLayer
        from openagent.adaptive_engine.expert_store import ExpertRegion
        with self.assertRaises(ValueError):
            ExpertLayer("layer", 5, 5, 1, "int4-grouped", "swiglu", {"a": ExpertRegion("bytes", 0, 100)}).validate()
        registry = ExpertKernelRegistry()
        registry.register(grouped_int4_expert.GroupedInt4ExpertBackend(group_size=32), replace=True)
        with self.assertRaises(RuntimeError):
            registry.resolve(dtype="int4-grouped-g64", preferred="grouped-int4-reference", allow_fallback=False)
