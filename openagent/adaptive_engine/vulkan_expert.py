"""Optional, streaming FP32 SwiGLU compute on a verified Vulkan adapter.

Payloads are decoded on the host; all matrix products and activation run in
WGSL. Only the device and pipelines persist, never expert weights or buffers.
"""

from __future__ import annotations

from collections import OrderedDict
import importlib
import os
import threading

import numpy as np

from .expert_kernels import (
    KernelBackendInfo, KernelCapabilities, KernelEvidence,
    SUPPORTED_PAYLOAD_DTYPES, _decode_payload, _finite_array, _finite_matrix,
    _finite_output, _lease, _validate_configuration, _validate_routes,
)


_PROBE_LOCK = threading.Lock()
_GATE_UP = """
struct Params { hidden: u32, intermediate: u32, tokens: u32, padding: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> matrices: array<f32>;
@group(0) @binding(2) var<storage, read_write> activation: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= p.tokens * p.intermediate) { return; }
    let token = index / p.intermediate;
    let row = index % p.intermediate;
    let width = p.hidden * p.intermediate;
    var gate = 0.0;
    var up = 0.0;
    for (var k = 0u; k < p.hidden; k++) {
        let value = x[token * p.hidden + k];
        gate += matrices[row * p.hidden + k] * value;
        up += matrices[width + row * p.hidden + k] * value;
    }
    gate *= matrices[3u * width + row];
    up *= matrices[3u * width + p.intermediate + row];
    let sigmoid = select(exp(gate) / (1.0 + exp(gate)),
                         1.0 / (1.0 + exp(-gate)), gate >= 0.0);
    activation[index] = gate * sigmoid * up;
}
"""
_DOWN = """
struct Params { hidden: u32, intermediate: u32, tokens: u32, padding: u32 }
@group(0) @binding(0) var<storage, read> activation: array<f32>;
@group(0) @binding(1) var<storage, read> matrices: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= p.tokens * p.hidden) { return; }
    let token = index / p.hidden;
    let row = index % p.hidden;
    let width = p.hidden * p.intermediate;
    var value = 0.0;
    for (var k = 0u; k < p.intermediate; k++) {
        value += matrices[2u * width + row * p.intermediate + k]
                 * activation[token * p.intermediate + k];
    }
    output[index] = weights[token] *
        (value * matrices[3u * width + 2u * p.intermediate + row]);
}
"""


class VulkanRoutedSwiGLUBackend:
    """Two-pass Vulkan expert backend, with no CPU computation fallback."""

    NAME = "vulkan-wgpu"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._device = None
        self.adapter_info = {}
        self._reason = ""
        try:
            self._wgpu = importlib.import_module("wgpu")
            # wgpu-native reads this at adapter request time, even after import.
            with _PROBE_LOCK:
                previous = os.environ.get("WGPU_BACKEND_TYPE")
                os.environ["WGPU_BACKEND_TYPE"] = "Vulkan"
                try:
                    adapter = self._wgpu.gpu.request_adapter_sync(
                        power_preference="high-performance", force_fallback_adapter=False
                    )
                finally:
                    if previous is None:
                        os.environ.pop("WGPU_BACKEND_TYPE", None)
                    else:
                        os.environ["WGPU_BACKEND_TYPE"] = previous
            self.adapter_info = dict(adapter.info)
            if self.adapter_info.get("backend_type") != "Vulkan":
                raise RuntimeError("requested Vulkan but adapter is not Vulkan")
            if self.adapter_info.get("adapter_type") not in {"IntegratedGPU", "DiscreteGPU"}:
                raise RuntimeError("Vulkan adapter is not a hardware GPU")
            self._device = adapter.request_device_sync()
            self._pipelines = tuple(
                self._device.create_compute_pipeline(
                    layout="auto", compute={"module": self._device.create_shader_module(code=code),
                                            "entry_point": "main"}
                ) for code in (_GATE_UP, _DOWN)
            )
        except Exception as exc:
            self._reason = f"Vulkan wgpu unavailable: {type(exc).__name__}: {exc}"
            if self._device is not None:
                self._device.destroy()
                self._device = None

    def info(self) -> KernelBackendInfo:
        return KernelBackendInfo(
            name=self.NAME, provider="wgpu-py", available=self._device is not None,
            priority=100, capabilities=KernelCapabilities(
                payload_dtypes=SUPPORTED_PAYLOAD_DTYPES, batch_execution=True,
                deterministic=False, compiled=True,
            ), evidence=(KernelEvidence("adapter", str(self.adapter_info),
                                        verified=self._device is not None),),
            unavailable_reason=self._reason,
        )

    def close(self) -> None:
        with self._lock:
            if self._device is not None:
                self._device.destroy()
                self._device = None
            self._reason = "Vulkan backend is closed"

    def execute(self, source, inputs, routes, *, hidden_size, intermediate_size, dtype):
        _validate_configuration(source, hidden_size, intermediate_size, dtype)
        vector = _finite_array(inputs, (hidden_size,), "inputs")
        return self.execute_batch(source, vector[None, :], [routes], hidden_size=hidden_size,
                                  intermediate_size=intermediate_size, dtype=dtype)[0]

    def execute_batch(self, source, inputs, route_batch, *, hidden_size, intermediate_size, dtype):
        hidden, intermediate, expected = _validate_configuration(
            source, hidden_size, intermediate_size, dtype
        )
        batch = _finite_matrix(inputs, hidden)
        try:
            route_rows = list(route_batch)
        except TypeError:
            raise TypeError("route_batch must be an iterable of route lists") from None
        if len(route_rows) != len(batch):
            raise ValueError("one route list is required for each input row")
        grouped = OrderedDict()
        for row, routes in enumerate(route_rows):
            for key, weight in _validate_routes(routes, f"route_batch[{row}]"):
                grouped.setdefault(key, []).append((row, weight))
        with self._lock:
            if self._device is None:
                raise RuntimeError(self._reason)
            output = np.zeros(batch.shape, dtype=np.float32)
            for key, assignments in grouped.items():
                with _lease(source, key) as payload:
                    gate, up, down, gs, us, ds = _decode_payload(
                        payload, hidden, intermediate, dtype, expected
                    )
                    packed = np.concatenate([
                        np.asarray(m, dtype=np.float32).ravel() for m in (gate, up, down)
                    ] + [np.broadcast_to(s, (n,)).astype(np.float32)
                         for s, n in ((gs, intermediate), (us, intermediate), (ds, hidden))])
                    rows = np.asarray([row for row, _ in assignments], dtype=np.intp)
                    weights = np.asarray([w for _, w in assignments], dtype=np.float32)
                    result = self._compute(batch[rows], packed, weights, hidden, intermediate)
                    output[rows] += result
            return _finite_output(output)

    def _compute(self, inputs, packed, weights, hidden, intermediate):
        device = self._device
        usage = self._wgpu.BufferUsage
        tokens = len(inputs)
        sizes = [inputs.nbytes, packed.nbytes, tokens * intermediate * 4,
                 tokens * hidden * 4, weights.nbytes]
        limit = min(device.limits["max-buffer-size"],
                    device.limits["max-storage-buffer-binding-size"])
        dispatches = [(tokens * intermediate + 63) // 64, (tokens * hidden + 63) // 64]
        if max(sizes) > limit or max(dispatches) > device.limits["max-compute-workgroups-per-dimension"]:
            raise ValueError("expert group exceeds Vulkan device buffer or dispatch limits")
        buffers = []
        try:
            def upload(data, flags=usage.STORAGE):
                buffer = device.create_buffer_with_data(data=data, usage=flags)
                buffers.append(buffer)
                return buffer

            x = upload(np.ascontiguousarray(inputs))
            matrices = upload(packed)
            activation = device.create_buffer(size=sizes[2], usage=usage.STORAGE)
            buffers.append(activation)
            result = device.create_buffer(size=sizes[3], usage=usage.STORAGE | usage.COPY_SRC)
            buffers.append(result)
            router = upload(weights)
            params = upload(np.asarray([hidden, intermediate, tokens, 0], dtype=np.uint32), usage.UNIFORM)
            encoder = device.create_command_encoder()
            for pipeline, resources, count in zip(
                self._pipelines, ((x, matrices, activation, params),
                                  (activation, matrices, result, router, params)), dispatches
            ):
                group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
                    {"binding": i, "resource": {"buffer": buffer}}
                    for i, buffer in enumerate(resources)
                ])
                compute = encoder.begin_compute_pass()
                compute.set_pipeline(pipeline)
                compute.set_bind_group(0, group)
                compute.dispatch_workgroups(count)
                compute.end()
            device.queue.submit([encoder.finish()])
            return np.frombuffer(device.queue.read_buffer(result), dtype=np.float32).copy().reshape(tokens, hidden)
        finally:
            for buffer in buffers:
                buffer.destroy()


__all__ = ["VulkanRoutedSwiGLUBackend"]
