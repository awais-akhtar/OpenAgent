# CUDA/HIP Backend Validation

## Scope

`tests/test_cuda_hip_certification.py` is an opt-in differential suite for
`TorchExpertBackend`, invoked directly rather than through a registry that may
select a reference fallback. HIP uses PyTorch's `cuda:N` device API too.
Only the explicitly selected CUDA/HIP device is covered by each run.

This backend decodes payload bytes on the host, transfers owned tensors to the
device, performs FP32 matmul and SiLU operations, and returns host NumPy FP32
arrays. Float16 and row-wise INT8 are storage formats, not FP16 or INT8 device
arithmetic. Host decoding and the final device-to-host transfer are intentional.
This suite does **not** certify optimized native kernel parity, fused kernels,
throughput, memory bounds, full-model accuracy, or bitwise determinism.

## Strict Run

Use a Python environment with NumPy and a hardware-compatible PyTorch runtime.
From the repository root, in PowerShell:

```powershell
$env:OPENAGENT_CUDA_HIP_CERTIFY = '1'
$env:OPENAGENT_CUDA_HIP_DEVICE = 'cuda:0'
python -m tests.test_cuda_hip_certification
```

Without the opt-in value exactly `1`, the five tests skip. With it, missing
PyTorch, an unusable accelerator, invalid selection, or failed device probe is
a setup error and a nonzero exit, never a successful hardware skip. CPU and
Metal are disallowed. No dependencies, drivers, or runtimes are installed by
the suite. Repeat with other device indices to validate other devices.

The module entry point rejects every skip and empty suite with exit 1, including
when opt-in is omitted. Ordinary unittest discovery is not this strict runner.
Runner contract tests cover success, skips (including class setup), empty suites,
errors and numerical failures; those self-tests are not hardware evidence.

The suite prints `CUDA_HIP_RUNTIME` and `CUDA_HIP_DEVICE` JSON with Python,
platform, PyTorch version, CUDA/HIP build-runtime versions, availability,
device count, selected device, device name, memory, and capability. These are
runtime/build facts, not an independently queried driver version. Setup
failures print `CUDA_HIP_BLOCKED` with the reason. `CUDA_HIP_PASS` lines describe
completed checks; certification requires the final unittest run to succeed
with no failures or skips, not merely the presence of a device or PASS line.

## Tested Semantics

- Real dispatch inspection checks FP32 matmul and SiLU input/output placement
  on the selected device. Every nonempty successful execution must dispatch
  both operations. TF32 matmul is disabled for the suite and restored afterward.
- Float32, float16, and INT8-row serialized payloads use deterministic fixtures;
  INT8 includes -128 and 127 and distinct positive gate/up/down row scales.
- Scalar, batch, and NumPy reference results agree within `rtol=5e-5`,
  `atol=5e-6`. Shapes include 1x1, 3x4, 17x33, and 64x65, exercising small,
  odd, and alignment-adjacent dimensions. These are bounded fixtures, not an
  exhaustive dimension or floating-point range proof.
- Noncontiguous inputs, shared experts across rows, reversed route order,
  unnormalized weights, zero-weight routes, and unused experts are covered.
  Outputs are finite host FP32 arrays with the expected shape.
- Exact payload boundaries reject truncated and oversized data, NaN/infinite
  floating weights, and zero/negative/NaN/infinite INT8 scales. Missing experts
  raise errors and payload leases are released on failures.
- Invalid input shapes/nonfinite values, route counts, empty routes, duplicate
  or empty keys, negative/nonfinite/bool weights, unsupported dtypes, and invalid
  dimensions are rejected before acquiring payloads. Empty batches are accepted.
- Each grouped call leases only routed experts, once per expert, with at most
  one active lease. Changed payloads are reloaded, overflow output is rejected,
  and idempotently closed backends raise instead of falling back.
- An injected allocation failure propagates. This negative-path injection is
  not hardware numerical evidence; the differential cases use actual device
  operations without mocked execution.

## Local Evidence: 2026-10-02

Status: **hardware certification blocked; no CUDA or HIP numerical pass**.

- Default interpreter `C:\Python314\python.exe` has no PyTorch installed.
- Existing `D:\openagent\.accelerator-venv\Scripts\python.exe`: Python 3.12.10,
  Windows 11 build 26200, PyTorch `2.14.1+cpu`, CUDA runtime null, HIP runtime
  null, accelerator availability false, device count zero.
- Strict invocation above using that environment exited 1 at setup with
  `strict hardware certification blocked: no usable CUDA/HIP device`;
  zero hardware test methods executed.
- Windows `Win32_VideoController` inventory was denied. Neither `nvidia-smi`
  nor `rocm-smi` was found on PATH. Physical GPU absence is **not** established;
  only this environment's inability to execute CUDA/HIP is established.
- Disabled suite: five skips, exit 0. Existing `test_torch_expert.py` regression
  suite: 12 tests, 11 passes and one unavailable-accelerator skip, exit 0.
  CPU differential results do not count as CUDA/HIP certification.

A CUDA/HIP-capable runner with a matching PyTorch build is needed to obtain
actual device and numerical evidence. Retain its complete output and exit
status; CUDA results cannot certify HIP, or vice versa.

### Strict Runner Recheck On 2026-10-02

Source HEAD: `b1b2495880631078414d3a7428580aea61e67353`, with existing
uncommitted changes retained. The existing accelerator environment reported
Python 3.12.10, NumPy 2.4.6, torch `2.14.1+cpu`, CUDA/HIP null, availability
false, and device count zero. `nvcc`, `hipcc`, `nvidia-smi`, and `rocm-smi`
were absent from PATH. `Get-CimInstance Win32_VideoController` returned
Access denied, so physical GPU absence is not established.

With `OPENAGENT_CUDA_HIP_CERTIFY=1`, the new module entry point exited 1
at class setup: `no usable CUDA/HIP device`; zero hardware tests executed.
CUDA and HIP remain **unverified independently**. No dependencies or drivers
were installed. Combined runner contracts and torch regressions ran 19 tests:
18 passed, one unavailable-accelerator skip. Those results are not certification.
