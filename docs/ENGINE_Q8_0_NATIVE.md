# Optional Q8_0 CPU Kernel

`NativeQ8_0ExpertBackend` lives in
`openagent.adaptive_engine.native_q8_0_expert`. It uses the locally installed
Numba runtime, with nopython compilation, `nogil=True`, `fastmath=False`, and
no disk JIT cache. It does not download dependencies, build a C library or
register itself. Missing Numba or `NUMBA_DISABLE_JIT=1` makes the backend
unavailable; execution then raises `RuntimeError`, not a Python-loop fallback.
Specializations compile lazily on first use, so compilation may fail even when
the runtime is available. Such failures are not silently replaced by another
implementation.

Unexpected dispatcher/compilation exceptions are sanitized at the lease boundary:
original traceback frames and cause/context chains are cleared, borrowed byte
views are dropped, and the original exception is raised outside the lease with
its type and message preserved and no chained exception. This deliberately
discards compiler tracebacks to prevent retained exceptions from pinning mmap
storage; no decoded payload copy is made. A regression retains both the reported
exception and the original chained exceptions and closes a real mmap store,
including a nested callback retaining payload and argument views in its frames.

## Explicit Selection

```python
from openagent.adaptive_engine.expert_kernels import ExpertKernelRegistry
from openagent.adaptive_engine.native_q8_0_expert import NativeQ8_0ExpertBackend

registry = ExpertKernelRegistry()
registry.register(NativeQ8_0ExpertBackend(max_scratch_bytes=1024 * 1024))
backend = registry.resolve(
    dtype="q8_0", preferred="q8_0-numba", allow_fallback=False,
)
output = backend.execute(
    source, inputs, routes, hidden_size=hidden, intermediate_size=intermediate,
)
```

Registry priority is 100, affecting only explicitly registered Q8_0 backends.
After registration, ordinary Q8_0 resolution may select it even without a
preferred name. The default registry is unchanged and still resolves Q8_0 to
`q8_0-numpy`; no original INT8 or INT4 backend is replaced. The backend reports
compiled capability when the optional JIT runtime is enabled, not a measured
speedup or a native SIMD claim.

The OLMoE CLI now explicitly registers this backend for
`generate` / `verify --kernel q8_0-numba` (the option follows the subcommand's
positional arguments). Packing uses `pack --dtype q8_0`, with distinct default
container `openagent-experts.q8` and JSON `openagent-experts.q8.manifest.json`.
`--kernel q8_0-numpy` remains supported without optional registration. Full
`OlmoeModel` state binds expert content, dtype and selected backend as well as
dense-weight identity; backend/precision/content changes cannot be restored into
the existing KV state. Segments remain INT8-only and reject Q8_0. See the
[integration commands and state contract](ENGINE_Q8_0.md).

## Numerical And Memory Contract

The byte contract is the existing [Q8_0 expert format](ENGINE_Q8_0.md).
The compiled kernel borrows a uint8 view of the leased payload and decodes each
binary16 scale from its integer sign, exponent and fraction fields. No Numba
float16 arrays or float16 operations are required. Weight codes are decoded as
signed INT8, including -128; the scale is multiplied before the activation
operand and reduction. FP32 products and serial reductions use no fastmath.
SiLU uses a sign-dependent sigmoid branch to avoid exponential overflow.
Different reduction orders from NumPy/BLAS need not produce identical bits.

Exact explicit numerical scratch is `4 * intermediate_size` bytes: one FP32
activation vector reused across every token and expert. `scratch_bytes(h, i)`
reports this size. `max_scratch_bytes` rejects oversized execution before
acquiring a lease. There are no decoded matrices or scale/code arrays; only a
borrowed uint8 payload view. Inputs/output, input conversion/contiguity copies,
Python routes, Numba compilation/runtime resources, payload storage and the OS
page cache are outside this scratch budget. One expert lease is active at a
time, and no borrowed array is kept after a call.

Payload length and dimensions are checked before native access. Every scale is
checked in the compiled kernel before arithmetic, including zero-weight routes.
Nonfinite scale encodings return a status converted to `ValueError` only after
the borrowed array has been dropped. Nonfinite projections, activation, down
projection or weighted accumulation return a status converted to
`ArithmeticError`. No output is returned on failure; the caller's input is not
mutated. Partial internal output may have been computed and is discarded.
As elsewhere, callers must keep leased payload contents immutable during use.

## Verification And Limits

```powershell
.venv/Scripts/python.exe -W error -m unittest tests.test_native_q8_0_expert -v
```

Tests execute actual Numba nopython code. They exhaust all 65,536 binary16 bit
patterns against Python's struct decoder, check hand-built Q8_0 payloads against
the existing independent scalar-byte oracle and FP32 backend, and exercise
finite extreme scales, signed codes, corruption, overflow, scratch preflight,
and real buffered/mmap DiskExpertStore lease cleanup. Generated LLVM is checked
for absence of Numba array-allocation calls in the numerical function. Disabled
JIT is exercised in a separate process. These are implementation-specific tests,
not a universal guarantee about compiler-generated workspace on other versions.

Verified locally with Numba 0.68.0. This work does not supply a C/ctypes backend,
controlled benchmark, upstream execution parity, model-quality evaluation,
accelerator kernel or cross-platform compiler certification. Model packer/CLI
integration is maintained separately and was not modified by these native files.

The reported public-checkpoint integration smoke run at
`allenai/OLMoE-1B-7B-0125-Instruct` revision
`b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e` passed the four-token oracle with
`q8_0-numba` in 25.291 seconds including JIT and loading. Conversion took
116.823 seconds and produced 1,024 experts totaling 6,845,104,128 bytes. This is
one reported local smoke result, not an upstream-parity, quality or benchmark
claim, and it was not rerun by this native fixture test suite.
