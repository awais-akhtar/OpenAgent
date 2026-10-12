"""Optional PyTorch device kernels for explicitly routed SwiGLU experts."""

from __future__ import annotations

from collections import OrderedDict
import importlib

import numpy as np

from .expert_kernels import (
    KernelBackendInfo, KernelCapabilities, KernelEvidence,
    SUPPORTED_PAYLOAD_DTYPES, _decode_payload, _finite_array, _finite_matrix,
    _finite_output, _lease, _validate_configuration, _validate_routes,
)


class TorchExpertBackend:
    """Stream experts to CUDA (including HIP) or Metal with FP32 arithmetic.

    No weights are cached or retained between calls. Only one expert's weights
    are live at a time; activations and PyTorch's allocator pool are additional.
    ``allow_cpu_for_tests`` enables an explicitly labelled differential-test
    device, never an automatic CPU fallback or a claim of GPU availability.
    """

    def __init__(self, *, device: str | None = None,
                 allow_cpu_for_tests: bool = False) -> None:
        self._torch = None
        self.device = None
        self._reason = ""
        self._closed = False
        if device is not None and not isinstance(device, str):
            raise TypeError("device must be a string or None")
        if device == "cpu" and not allow_cpu_for_tests:
            raise ValueError("CPU is permitted only with allow_cpu_for_tests=True")
        if device is not None and device not in {"cpu", "mps", "cuda"}:
            if not device.startswith("cuda:") or not device[5:].isdigit():
                raise ValueError("device must be cuda, cuda:N, mps, or explicit test cpu")
        try:
            torch = importlib.import_module("torch")
            self._torch = torch
            cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
            mps = getattr(torch.backends, "mps", None)
            metal = mps is not None and mps.is_available()
            selected = device or ("cuda" if cuda else "mps" if metal else None)
            if selected is None:
                raise RuntimeError("no available CUDA/HIP or MPS/Metal device")
            if selected.startswith("cuda"):
                if not cuda:
                    raise RuntimeError(f"CUDA/HIP device {selected} is unavailable")
                index = int(selected.split(":")[1]) if ":" in selected else torch.cuda.current_device()
                if index >= torch.cuda.device_count():
                    raise RuntimeError(f"CUDA/HIP device {selected} is unavailable")
                selected = f"cuda:{index}"
            elif selected == "mps" and not metal:
                raise RuntimeError("MPS/Metal device is unavailable")
            # Probe actual allocation, execution and host transfer, not build flags.
            probe = torch.ones((1, 1), device=selected, dtype=torch.float32)
            (probe @ probe).cpu().numpy()
            self.device = selected
        except Exception as exc:
            self._reason = f"PyTorch device unavailable: {exc}"

    def info(self) -> KernelBackendInfo:
        available = self.device is not None and not self._closed
        test_cpu = self.device == "cpu"
        return KernelBackendInfo(
            name="torch-cpu-test" if test_cpu else "torch-expert",
            provider="pytorch", available=available,
            priority=-20_000 if test_cpu else 600,
            capabilities=KernelCapabilities(
                payload_dtypes=SUPPORTED_PAYLOAD_DTYPES,
                batch_execution=True, deterministic=False,
            ),
            evidence=(KernelEvidence(
                "device", f"Executed FP32 device probe on {self.device}"
                + (" (CPU differential tests only)" if test_cpu else ""),
                verified=True,
            ),) if available else (),
            unavailable_reason=("backend is closed" if self._closed else self._reason)
            if not available else "",
        )

    def close(self) -> None:
        self._closed = True

    def execute(self, source, inputs, routes, *, hidden_size: int,
                intermediate_size: int, dtype: str) -> np.ndarray:
        _validate_configuration(source, hidden_size, intermediate_size, dtype)
        vector = _finite_array(inputs, (hidden_size,), "inputs")
        normalized = _validate_routes(routes, "routes")
        return self.execute_batch(
            source, vector[None, :], [normalized], hidden_size=hidden_size,
            intermediate_size=intermediate_size, dtype=dtype,
        )[0]

    def execute_batch(self, source, inputs, route_batch, *, hidden_size: int,
                      intermediate_size: int, dtype: str) -> np.ndarray:
        hidden_size, intermediate_size, expected = _validate_configuration(
            source, hidden_size, intermediate_size, dtype,
        )
        batch = _finite_matrix(inputs, hidden_size)
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
        if not self.info().available:
            raise RuntimeError(self.info().unavailable_reason)
        torch = self._torch
        with torch.inference_mode():
            x = torch.tensor(batch, dtype=torch.float32, device=self.device)
            output = torch.zeros_like(x)
            for key, assignments in grouped.items():
                with _lease(source, key) as payload:
                    decoded = _decode_payload(
                        payload, hidden_size, intermediate_size, dtype, expected,
                    )
                    # Own all tensor storage before releasing the payload lease.
                    tensors = [torch.tensor(np.array(a, dtype=np.float32, copy=True),
                                            device=self.device) for a in decoded]
                    del decoded
                gate_matrix, up_matrix, down_matrix, gate_scale, up_scale, down_scale = tensors
                rows = torch.tensor([row for row, _ in assignments],
                                    dtype=torch.long, device=self.device)
                weights = torch.tensor([weight for _, weight in assignments],
                                       dtype=torch.float32, device=self.device)[:, None]
                selected = x[rows]
                gate = (selected @ gate_matrix.T) * gate_scale
                up = (selected @ up_matrix.T) * up_scale
                activated = torch.nn.functional.silu(gate) * up
                result = (activated @ down_matrix.T) * down_scale
                # Route validation guarantees distinct row indices per expert.
                output[rows] = output[rows] + weights * result
                del tensors, gate_matrix, up_matrix, down_matrix
                del gate_scale, up_scale, down_scale, selected, gate, up, activated, result
            return _finite_output(output.cpu().numpy())


__all__ = ["TorchExpertBackend"]
