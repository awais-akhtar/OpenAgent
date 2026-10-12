from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace

import numpy as np

from openagent.adaptive_engine.openvino_expert import OpenVINOExpertBackend


class FakeProcess:
    def poll(self):
        return None


class FakeClient:
    def __init__(self):
        self._process = FakeProcess()
        self.loaded = {}
        self.loads = []
        self.unloads = []
        self.fail_unload = False
        self.fail_load = False
        self.fail_run = False
        self.output = [2, 4]
        self.entered = None
        self.release = None

    def available(self):
        return True

    def load_expert(self, **kwargs):
        if self.fail_load:
            raise RuntimeError("load failed")
        self.loads.append(kwargs)
        self.loaded[kwargs["expert_id"]] = kwargs

    def unload(self, expert_id):
        if self.fail_unload:
            raise RuntimeError("unload failed")
        self.unloads.append(expert_id)
        self.loaded.pop(expert_id, None)

    def run_expert(self, expert_id, inputs):
        if self.entered is not None:
            self.entered.set()
            if not self.release.wait(5):
                raise TimeoutError("test release timed out")
        if self.fail_run:
            raise RuntimeError("run failed")
        if expert_id not in self.loaded:
            raise KeyError("not resident")
        return {"output": self.output}


class FakeSource:
    def __init__(self, size=28):
        self.size = size
        self.revision = "a"

    def describe_region(self, key):
        return SimpleNamespace(path="experts.bin", offset=0, size=self.size,
                               sha256=self.revision)


class OpenVINOExpertTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.source = FakeSource()
        self.backend = OpenVINOExpertBackend(self.client, capacity_bytes=48)

    def execute(self, key="a", dtype="int8-row", routes=None):
        return self.backend.execute(self.source, [1, 1], routes or [(key, 1)],
                                    hidden_size=2, intermediate_size=2, dtype=dtype)

    def test_fp32_constant_accounting_and_file_descriptor(self):
        for dtype, size in (("int8-row", 28), ("float16", 24), ("float32", 48)):
            with self.subTest(dtype=dtype):
                self.source.size = size
                self.execute(dtype=dtype)
                self.assertEqual(self.backend._resident_bytes, 48)
                self.assertEqual(self.client.loads[-1]["size"], size)
        self.assertEqual(len(self.client.unloads), 2)

    def test_oversized_constants_rejected_before_loading(self):
        self.backend.capacity_bytes = 47
        with self.assertRaises(MemoryError):
            self.execute()
        self.assertEqual(self.client.loads, [])

    def test_cache_hit_and_eviction(self):
        self.execute()
        self.execute()
        self.assertEqual(len(self.client.loads), 1)
        self.execute("b")
        self.assertEqual(list(self.backend._resident), ["b"])
        self.assertEqual(len(self.client.loaded), 1)
        self.assertEqual(self.backend._resident_bytes, 48)

    def test_worker_replacement_invalidates_residency(self):
        self.execute()
        self.client._process = FakeProcess()
        self.client.loaded.clear()
        self.execute()
        self.assertEqual(len(self.client.loads), 2)
        self.assertEqual(self.backend._resident_bytes, 48)

    def test_lru_hits_refresh_eviction_order(self):
        self.backend.capacity_bytes = 96
        for key in ("a", "b", "a", "c"):
            self.execute(key)
        self.assertEqual(list(self.backend._resident), ["a", "c"])
        self.assertEqual(self.backend._resident_bytes, 96)
        self.assertEqual(self.client.unloads, [self.client.loads[1]["expert_id"]])

    def test_shared_client_backends_have_independent_ids(self):
        other = OpenVINOExpertBackend(self.client, capacity_bytes=48)
        self.execute()
        other.execute(self.source, [1, 1], [("a", 1)], hidden_size=2,
                      intermediate_size=2, dtype="int8-row")
        self.assertEqual(len(self.client.loaded), 2)
        self.backend.close()
        self.assertEqual(len(self.client.loaded), 1)
        other.close()

    def test_close_unloads_is_idempotent_and_disables_execution(self):
        self.execute()
        self.backend.close()
        self.backend.close()
        self.assertEqual(len(self.client.unloads), 1)
        self.assertEqual(self.backend._resident_bytes, 0)
        self.assertFalse(self.backend.info().available)
        with self.assertRaises(RuntimeError):
            self.execute()

    def test_failed_unload_retains_accounting_and_close_can_retry(self):
        self.execute()
        self.client.fail_unload = True
        with self.assertRaises(RuntimeError):
            self.execute("b")
        self.assertEqual(list(self.backend._resident), ["a"])
        self.assertEqual(self.backend._resident_bytes, 48)
        with self.assertRaises(RuntimeError):
            self.backend.close()
        self.client.fail_unload = False
        self.backend.close()
        self.assertEqual(self.backend._resident_bytes, 0)

    def test_failed_load_not_recorded_and_failed_run_not_replayed(self):
        self.client.fail_load = True
        with self.assertRaises(RuntimeError):
            self.execute()
        self.assertEqual(self.backend._resident_bytes, 0)
        self.client.fail_load = False
        self.client.fail_run = True
        with self.assertRaises(RuntimeError):
            self.execute()
        self.assertEqual(len(self.client.loads), 1)

    def test_weighted_output_and_invalid_native_output(self):
        np.testing.assert_array_equal(self.execute(routes=[("a", .25), ("b", .75)]), [2, 4])
        for output in ([1], [float("nan"), 1]):
            self.client.output = output
            with self.assertRaises(ArithmeticError):
                self.execute()

    def test_close_waits_for_inflight_inference(self):
        self.client.entered = threading.Event()
        self.client.release = threading.Event()
        errors = []

        def run():
            try:
                self.execute()
            except Exception as exc:
                errors.append(exc)

        inference = threading.Thread(target=run)
        closer = threading.Thread(target=self.backend.close)
        inference.start()
        try:
            self.assertTrue(self.client.entered.wait(2))
            closer.start()
            closer.join(.1)
            self.assertTrue(closer.is_alive())
            self.assertEqual(self.client.unloads, [])
        finally:
            self.client.release.set()
            inference.join(5)
            if closer.ident is not None:
                closer.join(5)
        self.assertFalse(inference.is_alive())
        self.assertFalse(closer.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(self.client.unloads), 1)


if __name__ == "__main__":
    unittest.main()
