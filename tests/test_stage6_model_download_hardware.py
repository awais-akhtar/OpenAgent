from __future__ import annotations

import hashlib
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from openagent.adaptive_engine.gpu_adapters import CpuFallbackAdapter, IntelArcAdapter, build_adapter_registry
from openagent.adaptive_engine.hardware_detection import (
    CpuInfo,
    CudaDeviceInfo,
    HardwareCapabilities,
    IntelGpuInfo,
)
from openagent.adaptive_engine.model_download_manager import (
    DownloadStatus,
    ModelDownloadManager,
    ModelRegistryEntry,
    ModelSource,
    SourceKind,
    default_registry_entries,
    resolve_huggingface_url,
)


class _RangeServer:
    def __init__(self, payload: bytes, *, ignore_range: bool = False):
        self.payload = payload
        self.ignore_range = ignore_range
        self.ranges: list[str | None] = []

        payload_ref = self.payload
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                range_header = self.headers.get("Range")
                owner.ranges.append(range_header)
                if range_header and not owner.ignore_range:
                    start = int(range_header.removeprefix("bytes=").removesuffix("-"))
                    body = payload_ref[start:]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(payload_ref) - 1}/{len(payload_ref)}")
                else:
                    body = payload_ref
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: object) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/weights.bin"

    def __enter__(self) -> "_RangeServer":
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class Stage6ModelDownloadTests(unittest.TestCase):
    def _entry(self, model_id: str, source: ModelSource, *, expected_bytes: int | None = None) -> ModelRegistryEntry:
        return ModelRegistryEntry(
            model_id=model_id,
            family="test-family",
            sources=(ModelSource(source.kind, source.uri, source.expected_sha256, expected_bytes, source.priority),),
        )

    def test_default_registry_covers_nine_admitted_families_and_hf_url_is_deterministic(self) -> None:
        entries = default_registry_entries()
        self.assertEqual(len(entries), 9)
        self.assertEqual(len({entry.family for entry in entries}), 9)
        self.assertEqual(
            resolve_huggingface_url("owner/repo", "nested/model.safetensors", revision="v1"),
            "https://huggingface.co/owner/repo/resolve/v1/nested/model.safetensors",
        )

    def test_local_download_state_and_integrity_survive_manager_restart(self) -> None:
        payload = b"synthetic local model artifact"
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "source.bin"
            source_path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            manager = ModelDownloadManager(root / "state")
            source = ModelSource(SourceKind.LOCAL_PATH, str(source_path), expected_sha256=digest)
            manager.register(self._entry("local-fixture", source, expected_bytes=len(payload)))
            job_id = manager.enqueue("local-fixture", root / "downloads")

            completed = manager.run_job(job_id)
            self.assertEqual(completed.status, DownloadStatus.COMPLETE)
            self.assertEqual(completed.dest_path.read_bytes(), payload)

            restarted = ModelDownloadManager(root / "state")
            restored = restarted.get_job(job_id)
            self.assertIsNotNone(restored)
            self.assertEqual(restored.status, DownloadStatus.COMPLETE)
            self.assertEqual(restored.source.expected_sha256, digest)
            self.assertEqual(restored.source.expected_bytes, len(payload))

    def test_restart_resumes_from_partial_file_and_validates_range(self) -> None:
        payload = b"0123456789" * 4096
        with _RangeServer(payload) as server, TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = ModelDownloadManager(root / "state")
            digest = hashlib.sha256(payload).hexdigest()
            source = ModelSource(SourceKind.DIRECT_URL, server.url, expected_sha256=digest)
            manager.register(self._entry("resume-fixture", source, expected_bytes=len(payload)))
            job_id = manager.enqueue("resume-fixture", root / "downloads")
            job = manager.get_job(job_id)
            assert job is not None
            partial = payload[:4096]
            job.dest_path.parent.mkdir(parents=True, exist_ok=True)
            job.dest_path.write_bytes(partial)

            restarted = ModelDownloadManager(root / "state")
            result = restarted.run_job(job_id)
            self.assertEqual(result.status, DownloadStatus.COMPLETE)
            self.assertEqual(result.dest_path.read_bytes(), payload)
            self.assertEqual(server.ranges, [f"bytes={len(partial)}-"])
            self.assertEqual(result.bytes_downloaded, len(payload))

    def test_range_ignored_server_restarts_write_instead_of_appending_corrupt_bytes(self) -> None:
        payload = b"canonical model content" * 200
        with _RangeServer(payload, ignore_range=True) as server, TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = ModelDownloadManager(root / "state")
            digest = hashlib.sha256(payload).hexdigest()
            source = ModelSource(SourceKind.DIRECT_URL, server.url, expected_sha256=digest)
            manager.register(self._entry("ignore-range-fixture", source, expected_bytes=len(payload)))
            job_id = manager.enqueue("ignore-range-fixture", root / "downloads")
            job = manager.get_job(job_id)
            assert job is not None
            job.dest_path.parent.mkdir(parents=True, exist_ok=True)
            job.dest_path.write_bytes(payload[:23])

            result = manager.run_job(job_id)
            self.assertEqual(result.status, DownloadStatus.COMPLETE)
            self.assertEqual(result.dest_path.read_bytes(), payload)
            self.assertEqual(result.bytes_downloaded, len(payload))

    def test_expected_size_mismatch_fails_and_pending_cancel_persists(self) -> None:
        payload = b"small local file"
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "source.bin"
            source_path.write_bytes(payload)
            manager = ModelDownloadManager(root / "state")
            source = ModelSource(SourceKind.LOCAL_PATH, str(source_path))
            manager.register(self._entry("size-mismatch", source, expected_bytes=len(payload) + 1))
            mismatch_id = manager.enqueue("size-mismatch", root / "downloads")
            mismatch = manager.run_job(mismatch_id)
            self.assertEqual(mismatch.status, DownloadStatus.FAILED)
            self.assertIn("size", (mismatch.error or "").lower())

            cancel_id = manager.enqueue("size-mismatch", root / "downloads")
            manager.cancel(cancel_id)
            restarted = ModelDownloadManager(root / "state")
            cancelled = restarted.get_job(cancel_id)
            self.assertIsNotNone(cancelled)
            self.assertEqual(cancelled.status, DownloadStatus.CANCELLED)


class Stage6HardwareAdapterTests(unittest.TestCase):
    def test_intel_adapter_executes_projection_through_compiled_openvino_device_model(self) -> None:
        class FakeNode:
            def __init__(self, value=None):
                self.value = value

        class FakeOps:
            @staticmethod
            def parameter(*_args: object, **_kwargs: object) -> FakeNode:
                return FakeNode()

            @staticmethod
            def constant(value: object, **_kwargs: object) -> FakeNode:
                return FakeNode(np.asarray(value))

            @staticmethod
            def matmul(weight: FakeNode, _input: FakeNode, *_args: object) -> FakeNode:
                return FakeNode(weight.value)

            @staticmethod
            def convert(node: FakeNode, _dtype: object) -> FakeNode:
                return node

        calls: list[str] = []

        class FakeCompiled:
            def input(self, _index: int) -> str:
                return "input"

            def output(self, _index: int) -> str:
                return "output"

            def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
                calls.append("inference")
                return {"output": np.asarray(compiled_model.weight) @ inputs["input"]}

        compiled_model = FakeNode()

        class FakeCore:
            available_devices = ["GPU.0"]

            def get_property(self, *_args: object) -> str:
                return "Intel Arc fixture"

            def compile_model(self, model: object, device: str) -> FakeCompiled:
                self.assert_device(device)
                compiled_model.weight = np.asarray(model[0].value)
                calls.append("compiled")
                return FakeCompiled()

            @staticmethod
            def assert_device(device: str) -> None:
                if device != "GPU.0":
                    raise AssertionError(f"unexpected device {device}")

        class FakeModel:
            def __init__(self, results: list[FakeNode], _parameters: list[FakeNode], _name: str):
                self.results = results

            def __getitem__(self, index: int) -> FakeNode:
                return self.results[index]

        fake_openvino = type("FakeOpenVINO", (), {
            "Core": FakeCore,
            "Model": FakeModel,
            "PartialShape": staticmethod(lambda shape: shape),
            "Type": type("Type", (), {"f16": "f16", "f32": "f32", "bf16": "bf16"}),
            "opset13": FakeOps,
        })()
        fake_runtime = type("FakeOpenVINORuntime", (), {"Core": FakeCore})()
        with mock.patch.dict(sys.modules, {
            "openvino": fake_openvino,
            "openvino.runtime": fake_runtime,
        }):
            adapter = IntelArcAdapter(ov_device_id="GPU.0")
            weight = np.asarray([[2.0, -1.0], [0.5, 3.0]], dtype=np.float32)
            values = np.asarray([[4.0], [2.0]], dtype=np.float32)
            result = adapter.projection(weight, values)
            np.testing.assert_allclose(result, weight @ values)
            self.assertEqual(calls, ["compiled", "inference"])
            adapter.unload()

    def test_windows_cpu_brand_registry_fallback_is_used_when_wmi_is_unavailable(self) -> None:
        from openagent.adaptive_engine import hardware_detection

        key = mock.MagicMock()
        key.__enter__.return_value = key
        winreg = mock.MagicMock()
        winreg.HKEY_LOCAL_MACHINE = object()
        winreg.OpenKey.return_value = key
        winreg.QueryValueEx.return_value = ("Intel(R) Core(TM) Ultra 7 255H", 1)
        with mock.patch.dict(sys.modules, {"winreg": winreg}), \
             mock.patch.object(hardware_detection.os, "name", "nt"), \
             mock.patch.object(hardware_detection.platform, "processor", return_value="generic CPU"), \
             mock.patch.object(hardware_detection.platform, "machine", return_value="AMD64"), \
             mock.patch.object(hardware_detection.shutil, "which", return_value=None), \
             mock.patch.object(hardware_detection, "_cpu_flags", return_value={
                 "avx512": False, "amx": False, "neon": False,
             }):
            cpu, errors = hardware_detection._detect_cpu()

        self.assertEqual(cpu.name, "Intel(R) Core(TM) Ultra 7 255H")
        self.assertEqual(errors, [])

    def test_placement_uses_available_accelerators_then_native_fallback(self) -> None:
        cpu = CpuInfo("test cpu", physical_cores=4, logical_cores=8)
        native = HardwareCapabilities(cpu=cpu)
        self.assertEqual(native.best_placement, "native")
        registry = build_adapter_registry(native)
        self.assertIsInstance(registry.best_available(), CpuFallbackAdapter)

        cuda = HardwareCapabilities(cpu=cpu, cuda_devices=[CudaDeviceInfo(0, "fixture CUDA", 1, "8.0", "test")])
        self.assertEqual(cuda.best_placement, "device")
        arc_without_openvino = HardwareCapabilities(
            cpu=cpu,
            intel_gpus=[IntelGpuInfo("fixture Arc", 1, "test", True, "GPU.0")],
            openvino_available=False,
        )
        self.assertEqual(arc_without_openvino.best_placement, "native")

    def test_cpu_adapter_projection_matches_numpy_and_small_benchmark_is_finite(self) -> None:
        adapter = CpuFallbackAdapter()
        weight = np.asarray([[2.0, -1.0], [0.5, 3.0]], dtype=np.float32)
        values = np.asarray([[4.0], [2.0]], dtype=np.float32)
        np.testing.assert_allclose(adapter.projection(weight, values), weight @ values)
        benchmark = adapter.benchmark(matrix_size=8)
        self.assertEqual(benchmark.adapter_name, "cpu")
        self.assertEqual(benchmark.matrix_size, 8)
        self.assertGreaterEqual(benchmark.achieved_tflops, 0.0)
        self.assertGreaterEqual(benchmark.latency_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
