"""Threading and ownership regressions using the real native executor."""
from concurrent.futures import ThreadPoolExecutor
import gc
import tempfile
import threading
import unittest
import weakref

import numpy as np

from openagent.adaptive_engine.native_segment_executor import NativeSegmentExecutor
from tests.test_native_segment_executor import Reference, build_executor, fixture


class NativeSegmentAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Windows keeps a loaded DLL locked until process exit.
        cls.build_directory = tempfile.mkdtemp(prefix="oa-native-adversarial-")
        cls.library = build_executor(cls.build_directory)

    def make(self):
        cfg, weights = fixture()
        executor = NativeSegmentExecutor(self.library)
        engine = executor.create_range(weights, **cfg)
        return cfg, weights, executor, engine

    def assert_cache_equal(self, actual, expected):
        self.assertEqual(actual.position, expected.position)
        np.testing.assert_array_equal(actual._keys, expected._keys)
        np.testing.assert_array_equal(actual._values, expected._values)

    def test_shared_session_eight_calls_are_serialized(self):
        cfg, _, _, engine = self.make()
        shared, reference = engine.create_session(), engine.create_session()
        x = np.random.default_rng(73).normal(size=(1, cfg["hidden"])).astype(np.float32)
        expected = np.concatenate([reference.forward(x) for _ in range(8)])
        barrier = threading.Barrier(8)

        def forward(_):
            barrier.wait(timeout=30)
            return shared.forward(x)

        with ThreadPoolExecutor(max_workers=8) as pool:
            actual = np.concatenate(list(pool.map(forward, range(8))))
        # Lock acquisition order is unspecified; compare the output multiset.
        expected_order = np.lexsort(expected.T[::-1])
        actual_order = np.lexsort(actual.T[::-1])
        np.testing.assert_array_equal(actual[actual_order], expected[expected_order])
        self.assertEqual(shared.position, 8)
        self.assert_cache_equal(shared, reference)

    def test_eight_disjoint_sessions_match_native_and_numeric_references(self):
        cfg, weights, _, engine = self.make()
        sessions = [engine.create_session() for _ in range(8)]
        rng = np.random.default_rng(74)
        inputs = [rng.normal(size=(3, cfg["hidden"])).astype(np.float32)
                  for _ in sessions]
        references = [engine.create_session() for _ in sessions]
        expected = [session.forward(x) for session, x in zip(references, inputs)]
        numeric = [Reference(cfg, weights).forward(x) for x in inputs]
        barrier = threading.Barrier(8)

        def forward(index):
            barrier.wait(timeout=30)
            return sessions[index].forward(inputs[index])

        with ThreadPoolExecutor(max_workers=8) as pool:
            actual = list(pool.map(forward, range(8)))
        for index, output in enumerate(actual):
            np.testing.assert_array_equal(output, expected[index])
            np.testing.assert_allclose(output, numeric[index], atol=3e-6, rtol=4e-5)
            self.assert_cache_equal(sessions[index], references[index])
            self.assertEqual(sessions[index].position, 3)

    def test_registration_owns_weights_after_originals_are_mutated(self):
        cfg, weights, _, engine = self.make()
        owned = [array.copy() for array in engine._weights]
        x = np.ones((2, cfg["hidden"]), np.float32)
        reference = engine.create_session()
        expected = reference.forward(x)
        for layer in weights:
            for array in layer.values():
                array.fill(np.nan)
        for array, before in zip(engine._weights, owned):
            np.testing.assert_array_equal(array, before)
            self.assertFalse(array.flags.writeable)
            self.assertTrue(array.flags.owndata)
            for layer in weights:
                for original in layer.values():
                    self.assertFalse(np.shares_memory(array, original))
        session = engine.create_session()
        np.testing.assert_array_equal(session.forward(x), expected)
        self.assert_cache_equal(session, reference)

    def test_close_is_idempotent_and_rejects_mutations(self):
        cfg, _, _, engine = self.make()
        session = engine.create_session()
        x = np.ones((1, cfg["hidden"]), np.float32)
        session.forward(x)
        session.close()
        session.close()
        self.assertIsNone(session._keys)
        self.assertIsNone(session._values)
        self.assertEqual(session.position, 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.forward(x)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.reset()

    def test_retained_session_pins_engine_executor_library_and_weights(self):
        cfg, weights, executor, engine = self.make()
        session = engine.create_session()
        x = np.ones((2, cfg["hidden"]), np.float32)
        expected = Reference(cfg, weights).forward(x)
        engine_ref = weakref.ref(engine)
        executor_ref = weakref.ref(executor)
        library_ref = weakref.ref(executor.lib)
        weight_refs = [weakref.ref(array) for array in engine._weights]
        del weights, engine, executor
        gc.collect()
        self.assertIs(session.engine, engine_ref())
        self.assertIs(session.engine.executor, executor_ref())
        self.assertIs(session.engine.executor.lib, library_ref())
        self.assertTrue(all(ref() is not None for ref in weight_refs))
        np.testing.assert_allclose(session.forward(x), expected, atol=3e-6, rtol=4e-5)
        self.assertEqual(session.position, 2)
        session.close()
        del session
        gc.collect()
        self.assertIsNone(engine_ref())
        self.assertIsNone(executor_ref())
        self.assertTrue(all(ref() is None for ref in weight_refs))


if __name__ == "__main__":
    unittest.main()
