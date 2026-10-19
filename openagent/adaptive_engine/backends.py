"""Backend capability and compilation adapters for the adaptive engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .hardware import DeviceProfile, HardwareProfiler, HardwareProfile


@dataclass(frozen=True)
class BackendCapability:
    id: str
    device_id: str
    device_name: str
    model_formats: tuple[str, ...]
    phases: tuple[str, ...]
    measured: bool
    available: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model_formats"] = list(self.model_formats)
        payload["phases"] = list(self.phases)
        return payload


class BackendRegistry:
    def capabilities(self, hardware: HardwareProfile) -> list[BackendCapability]:
        result: list[BackendCapability] = []
        for device in hardware.devices:
            for backend in device.backends:
                if backend == "openvino":
                    formats = ("onnx", "openvino")
                    phases = ("prefill", "decode")
                    notes = "Vendor compiler and runtime; model compilation must succeed before use."
                elif backend.startswith("llama-"):
                    formats = ("gguf",)
                    phases = ("prefill", "decode")
                    notes = "llama.cpp runtime; model load and generation must be validated separately."
                elif backend == "onnxruntime":
                    formats = ("onnx",)
                    phases = ("prefill", "decode")
                    notes = "ONNX Runtime provider available in the main OpenAgent process."
                else:
                    formats = ()
                    phases = ()
                    notes = "Discovered backend with no registered model frontend."
                result.append(
                    BackendCapability(
                        id=backend,
                        device_id=device.id,
                        device_name=device.name,
                        model_formats=formats,
                        # Discovery is not proof that a backend has a model
                        # frontend for embedding or media execution. Advertise
                        # only the text-generation phases represented by these
                        # registered format frontends; unknown backends remain
                        # discoverable but are not admitted for any phase.
                        phases=phases,
                        measured=device.estimate_source.startswith("measured"),
                        available=device.available,
                        notes=notes,
                    )
                )
        return result


class OpenVINOBackend:
    """Use the isolated OpenVINO environment without importing it in the app."""

    def __init__(self, profiler: HardwareProfiler, cache_root: str | Path) -> None:
        self.profiler = profiler
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def compile(self, model_path: str | Path, device: DeviceProfile) -> dict[str, Any]:
        target = openvino_device_id(device)
        if not target:
            raise RuntimeError(f"{device.name} is not exposed by OpenVINO")
        return self.profiler.compile_openvino(
            str(Path(model_path).expanduser().resolve()),
            target,
            cache_dir=str(self.cache_root / _safe_name(target)),
        )

    def benchmark(
        self,
        device: DeviceProfile,
        *,
        iterations: int = 20,
        dimension: int = 256,
        performance_hint: str = "LATENCY",
        num_streams: str = "",
    ) -> dict[str, Any]:
        target = openvino_device_id(device)
        if not target:
            raise RuntimeError(f"{device.name} is not exposed by OpenVINO")
        return self.profiler.benchmark_openvino(
            target,
            iterations=iterations,
            dimension=dimension,
            performance_hint=performance_hint,
            num_streams=num_streams,
        )


def openvino_device_id(device: DeviceProfile) -> str:
    if "openvino" not in device.backends:
        return ""
    details = device.capabilities.get("openvino", {})
    if isinstance(details, dict):
        target = details.get("id")
        if isinstance(target, str) and target.strip():
            return target.strip()
    # CPU is a singleton in the OpenVINO device namespace. GPU/NPU kind names
    # are not identities: synthesizing "GPU" from a profile such as gpu:1 can
    # silently execute on GPU.0 (or another provider-selected device).
    if device.kind == "cpu" and "openvino" in device.backends:
        return "CPU"
    return ""


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
