# Metal Validation

Status: **not hardware-certified**. This implementation was prepared on Windows;
Windows cannot execute Apple Metal/MPS. Ordinary skipped tests, CPU differential
tests, mocked discovery, and CUDA execution are not Metal certification.

## Run On A Real Mac

Use a macOS host with an actual MPS-supported GPU and Python 3.11 or later.
From the repository root, in a fresh shell and virtual environment:

```sh
python3 -m venv .venv-metal
. .venv-metal/bin/activate
python -m pip install -e '.[engine-torch]'
python -m pip freeze
sw_vers
system_profiler SPHardwareDataType SPDisplaysDataType
OPENAGENT_CERTIFY_METAL=1 PYTORCH_ENABLE_MPS_FALLBACK=0 python -m tests.test_metal_certification
```

Preserve the command, exit status, complete stdout/stderr, hardware/OS details,
dependency versions, and the tested source revision (including any uncommitted
changes). Reuse the recorded dependency versions for repeat runs. No model
downloads or external datasets are needed.

## Acceptance

The selected suite must execute both tests with exit status zero, no skips, and
no errors or failures. Opt-in on Windows, missing torch/MPS, allocation failures,
or enabled/unset CPU fallback must fail, never skip. Without opt-in the suite
skips intentionally under ordinary discovery: that result means **not verified**,
even if unittest says OK. The module entry point rejects every skip and empty
suite with exit 1. Use that entry point for certification, as CI does.

The suite requires macOS, MPS build/runtime availability, real MPS allocation,
matrix multiplication, synchronization and host transfer. It explicitly selects
`TorchExpertBackend(device="mps")`, never automatic discovery or test CPU.
Fallback is disabled before importing torch. Real scalar and batch matmul and
SiLU dispatch inputs and outputs must reside on MPS with FP32 dtype; both
operations must be observed. Their implementation is not substituted.

Seeded nontrivial scalar and grouped-batch SwiGLU execution is compared with the
NumPy backend for float32, float16 and int8-row payloads, using FP32 arithmetic.
Every output and reference must be finite, have matching shape and FP32 output
dtype, and satisfy `abs(actual-reference) <= 2e-5 + 2e-4 * abs(reference)`.
These are bounded fixture tolerances, not a universal model accuracy guarantee.
Tests cover weighted/reordered routes, a zero-weight row, unused experts,
payload lease release, malformed/nonfinite payload rejection, overflow rejection,
weight refresh without caching, and idempotent close with execution rejected
after close. Each test closes its backend, synchronizes and clears the MPS
allocator cache; this does not assert a particular OS memory footprint.

A successful run certifies only this backend and these fixtures on the recorded
Mac/software combination. It does not certify full-model generation, performance,
all Apple GPUs, or other engine backends. No successful real-Mac run has been
recorded here.

## Windows Check Recorded

On this Windows workspace, using `.venv\Scripts\python.exe`, ordinary discovery
reported two intentional skips. With `OPENAGENT_CERTIFY_METAL=1` and
`PYTORCH_ENABLE_MPS_FALLBACK=0`, discovery exited 1 with
`cannot verify Metal on Windows` before any hardware tests ran. Existing torch
discovery tests passed (six tests); numerical tests skipped because torch was
not installed. The system Python also lacked NumPy, so checks used the repository
virtual environment. None of these results is Metal certification.

### Strict Runner Recheck On 2026-10-02

Source HEAD: `b1b2495880631078414d3a7428580aea61e67353`, with existing
uncommitted changes retained. The existing accelerator environment has Python
3.12.10, NumPy 2.4.6 and torch `2.14.1+cpu`; MPS build and availability are
both false. With `OPENAGENT_CERTIFY_METAL=1` and
`PYTORCH_ENABLE_MPS_FALLBACK=0`, `python -m tests.test_metal_certification`
exited 1 at class setup: `cannot verify Metal on Windows`; zero hardware tests
executed. Metal remains **unverified**. The strengthened MPS math-placement
contract still requires execution on a real Mac; its source is not evidence of
a successful MPS run. No dependencies or operating systems were installed.
