"""Native K3 adapter harness, written but not run in this pass.

Native cases require an existing library path; this module never compiles one.
"""
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ..native_kimi_k3_decoder import KimiK3DenseMLAConfig
from ..native_kimi_k3_kda import KimiK3KDAConfig
from ..native_kimi_k3_moe import KimiK3MoEConfig
from ..native_kimi_k3_hybrid import (KimiK3HybridConfig, KimiK3HybridSnapshot,
                                     NativeKimiK3Hybrid, hybrid_tensor_shapes)
from . import (Budget, DurablePagedCache, K3HybridSnapshotCodec, K3HybridPlanner,
               PackedStore, ScalableExecutor, bind_k3_session)


def fixture_config():
    return KimiK3HybridConfig(KimiK3DenseMLAConfig(capacity=8),
        KimiK3KDAConfig(capacity=8), KimiK3MoEConfig(), ("kda", "mla", "mla"))


class K3SerializationContracts(unittest.TestCase):
    def setUp(self):
        # Serialization alone does not require loading native code. This object
        # supplies the exact type/config/identity/buffer contract, not model math.
        template = NativeKimiK3Hybrid.__new__(NativeKimiK3Hybrid)
        template.config = fixture_config()
        template._identity = "a" * 64
        template._mixers = [type("Component", (), {"_identity": "b" * 64})(), None, None]
        c, k = template.config.mla, template.config.kda
        template._keys = [None, *[np.zeros((c.capacity, c.heads, c.nope + c.rope), np.float32) for _ in range(2)]]
        template._values = [None, *[np.zeros((c.capacity, c.heads, c.value), np.float32) for _ in range(2)]]
        recurrent = ("b" * 64, (1, np.ones((k.heads, k.key, k.value), np.float32),
            tuple(np.ones((k.kernel - 1, k.heads, width), np.float32) for width in (k.key, k.key, k.value))))
        keys = tuple(None if a is None else a.copy() for a in template._keys)
        values = tuple(None if a is None else a.copy() for a in template._values)
        for a in (*keys, *values):
            if a is not None:
                a[0] = 0.25
        self.snapshot = KimiK3HybridSnapshot(template._identity, 1, keys, values, (recurrent, None, None))
        self.codec = K3HybridSnapshotCodec(template, mla_layout="expanded-head-kv-v1",
                                          max_payload_bytes=1024 * 1024)

    def test_typed_roundtrip(self):
        frame = self.codec.encode_snapshot(self.snapshot)
        decoded = self.codec.decode_snapshot(frame)
        self.assertEqual(decoded.length, 1)
        for actual, expected in zip(self.codec._arrays(decoded), self.codec._arrays(self.snapshot)):
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.dtype, np.dtype(np.float32))
            self.assertTrue(actual.flags.c_contiguous)

    def test_corruption_and_foreign_config_rejected(self):
        frame = self.codec.encode_snapshot(self.snapshot)
        frame[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.codec.decode_snapshot(frame)
        changed = replace(self.snapshot, identity="c" * 64)
        with self.assertRaises(ValueError):
            self.codec.encode_snapshot(changed)

    def test_compressed_layout_not_implicitly_admitted(self):
        with self.assertRaisesRegex(ValueError, "compressed MLA"):
            K3HybridSnapshotCodec(self.codec.template, mla_layout="compressed-latent-v1",
                                  max_payload_bytes=1024 * 1024)

    def test_payload_budget_before_serialization(self):
        with self.assertRaises(MemoryError):
            K3HybridSnapshotCodec(self.codec.template, mla_layout="expanded-head-kv-v1",
                                  max_payload_bytes=64)


@unittest.skipUnless(os.environ.get("OPENAGENT_K3_TEST_LIBRARY"), "requires explicit existing K3 library")
class K3NativeBindingContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        config = fixture_config()
        rng = np.random.default_rng(73)
        weights = {name: rng.normal(0, 0.01, shape).astype(np.float32)
                   for name, shape in hybrid_tensor_shapes(config).items()}
        for name, array in weights.items():
            if "norm.weight" in name:
                array.fill(1)
        self.model = NativeKimiK3Hybrid(config, weights,
            library=os.environ["OPENAGENT_K3_TEST_LIBRARY"], max_bytes=128 * 1024 * 1024)
        self.codec = K3HybridSnapshotCodec(self.model, mla_layout="expanded-head-kv-v1",
                                          max_payload_bytes=1024 * 1024)
        geometry = self.codec.geometry(max_batch_tokens=2)
        self.cache = DurablePagedCache(root / "cache", geometry,
            capacity_bytes=8 * self.codec.step_bytes, page_tokens=1,
            read_budget=Budget(2 * self.codec.step_bytes))
        self.store = PackedStore(root, {}, mapped_budget=Budget(4096))
        self.executor = ScalableExecutor(geometry, self.store, self.cache,
            scratch_budget=Budget(K3HybridPlanner(self.codec, geometry).scratch_required(2)))
        self.session = bind_k3_session(self.executor, self.codec, "native", "owner")

    def tearDown(self):
        self.session.close()
        self.executor.close()
        self.cache.close()
        self.store.close()
        self.temp.cleanup()

    def test_native_logits_recovery_and_generation_rollback(self):
        baseline = self.model.snapshot()
        actual = self.session.forward((1, 2))
        expected = self.model.forward(np.array([1, 2], np.int64))
        np.testing.assert_array_equal(actual, expected)
        self.model.restore(baseline)
        checkpoint = self.session.snapshot()
        try:
            next_logits = self.session.forward((3,))
            self.session.restore(checkpoint)
            np.testing.assert_array_equal(self.session.forward((3,)), next_logits)
            self.assertEqual(self.session.position, 3)
            self.assertEqual(self.model.position, 0)
        finally:
            self.session.release_snapshot(checkpoint)

    def test_bad_snapshot_does_not_mutate_native_target(self):
        frame = self.codec.encode_snapshot(self.model.snapshot())
        frame[-1] ^= 1
        with self.assertRaises(ValueError):
            self.codec.restore(self.model, frame)
        self.assertEqual(self.model.position, 0)
