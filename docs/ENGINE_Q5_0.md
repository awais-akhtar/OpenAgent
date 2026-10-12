# Independent Q5_0

The public layout reference is [GGML block_q5_0 field metadata](https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h)
(checked 2026-10-02). Only the format declarations were consulted, not upstream
quantizer or execution implementations. The implementation and independent
bit-plane tests below establish the explicit contract, not upstream byte or
execution parity.

## Byte Contract

Each block describes 32 weights in 22 little-endian bytes:

| Offset | Field | Meaning |
| --- | --- | --- |
| 0 | 2-byte IEEE binary16 d | Signed finite delta |
| 2 | 4-byte qh | Bit k is the fifth code bit for lane k |
| 6 | 16-byte qs | Low nibbles for lanes 0..15; high for lanes 16..31 |

For lane k, q is its low nibble plus 16 times its qh bit. Weight is
FP32(q-16)*FP32(d). All codes 0..31 are legal, representing centered values
-16..15. Finite negative, zero, negative-zero and subnormal deltas are legal;
NaN and infinity are not. The contract uses explicit bytes, not native-endian
struct reinterpretation. Tests verify every fifth-bit lane independently.

Expert layout is row-major gate [i,h], up [i,h], then down [h,i], without
headers, alignment or padding. Trusted external dimensions must be positive
integers divisible by 32. Size is 3*h*i/32*22. Equal-size dimension swaps cannot
be detected from bytes alone. Fifth-bit/nibble corruption may remain legal;
integrity protection belongs to the region SHA256 contract.

## APIs

- `q5_0_expert`: `DTYPE="q5_0"`, `BLOCK_WIDTH=32`, `BLOCK_BYTES=22`.
- `payload_size(hidden_size, intermediate_size)`.
- `pack_matrix(matrix, block_count=64)` and `pack_expert(gate, up, down, block_count=64)` return bytes; block_count is keyword-only.
- `decode_expert(payload, hidden_size, intermediate_size)` explicitly allocates owned FP32 matrices.
- `Q5_0ExpertBackend(max_scratch_bytes=...)`, scalar name `q5_0-reference`.
- `native_q5_0_expert.NativeQ5_0ExpertBackend(max_scratch_bytes=...)`, native name `q5_0-numba`.
- Both modules expose `scratch_bytes(hidden_size, intermediate_size)`, which is 4*i.
- `q5_0_conversion.convert_experts(experts, output, hidden_size=..., intermediate_size=..., block_count=64)` returns `Q5_0ConversionResult`.

Converter fields: path, hidden_size, intermediate_size, immutable regions
mapping, dtype and activation="swiglu". Regions are standard ExpertRegion
objects with relative filename, consecutive offsets, exact sizes and SHA256.
One packed expert is held while writing. Conversion uses same-directory
temporary creation, file fsync and exclusive hard-link publication. It refuses
existing outputs, removes temporary files on validation/publication failure,
and does not claim crash-durable directory publication. Filesystems lacking
hard-link support fail with OSError rather than publishing a partial output.

## Quantization Policy

This is independently specified: positive max(abs(block))/15, rounded to
binary16; nonzero underflow deltas floored to 2^-24. Codes are ties-to-even,
computed against the stored delta and clipped to [-16,15], then biased by 16.
Zero blocks store d=0 and q=16. Weights must be real, finite and representable
as FP32; binary16 delta overflow is an error. Conversion staging is bounded by
block_count*32 values plus fixed block temporaries, excluding caller matrices
and returned bytes. `pack_expert` holds its three projection byte results
while joining. This is not an upstream quantizer reproduction.

## Kernels

The scalar and compiled numerical routines are distinct implementations of
the byte contract. They share lease/input/route orchestration through the
reference backend class. Scalar execution reads one fixed-size block tuple;
compiled execution extracts bytes and bits directly. Neither allocates decoded
weight arrays or scale arrays. Explicit FP32 activation scratch is one i-value
vector, checked before acquiring leases. Input normalization, caller output,
routes, fixed Python block objects, borrowed payload and compiler workspace
are separate. One expert lease is held at a time.

Every scale is validated before arithmetic, including zero-weight routes.
Nonfinite projection, SwiGLU or weighted-output arithmetic raises
ArithmeticError. Exceptions from the numerical routine detach chained
borrowed frames before leaving the lease, allowing mmap closure while the
exception remains referenced. Returned partial outputs are never exposed.

The native backend requires active Numba and never falls back to Python.
Missing Numba and NUMBA_DISABLE_JIT=1 mark it unavailable; execution raises
RuntimeError. Numba uses nogil, fastmath=False and cache=False. Lazy compiler
failures propagate, not scalar execution. No SIMD or speed claim is made.

## Shared Integration Requirements

Shared registry, manifest, packer, model and gateway changes are maintained
separately. The recorded integration connects the following mappings; its
packing, model and gateway paths passed the format integration tests:

- Add q5_0 to REGISTERED_PAYLOAD_DTYPES and manifest dtype validation.
- Default-register Q5_0ExpertBackend; keep NativeQ5_0ExpertBackend opt-in.
- Dispatch payload sizing to q5_0_expert.payload_size.
- SafeTensors packing must call q5_0_expert.pack_matrix, concatenate the three projections, and not append row scales.
- Accept q5_0 in OLMoE/CLI dtype allowlists and q5_0-reference/q5_0-numba in runtime selection.
- Add native CLI/provider registration and lazy exports. Suggested converter export: convert_q5_0_experts.
- Keep artifacts distinct; suggested filenames are openagent-experts.q50 and openagent-experts.q50.manifest.json.

The standalone converter is available. The integration layer constructs ExpertLayer and
SparseExpertManifest with caller-supplied model/layer/routing identity from
its returned regions. Shared integration tests require no mocks for packing,
model math or gateway loading. The shared dtype gate now passes, so packing,
model logit differential and compiled gateway loading all execute. On a
checkout lacking shared integration they skip explicitly until the dtype is
registered; once accepted, missing integration produces a failure.

Gateway selection after shared runtime integration:

```text
OPENAGENT_OLMOE_MODEL_DIR=<checkpoint directory>
OPENAGENT_OLMOE_MANIFEST=<Q5_0 manifest path>
OPENAGENT_OLMOE_KERNEL=q5_0-numba
OPENAGENT_OLMOE_IO_MODE=mmap
```

## Verification

```powershell
.venv/Scripts/python.exe -W error -m unittest tests.test_q5_0_expert tests.test_native_q5_0_expert tests.test_q5_0_olmoe -v
```

Format tests cover all 32 codes, every high-bit lane, extreme finite/signed
deltas, exact quantizer bytes, ties-even, zero/subnormal/maximum deltas,
chunk-independent conversion and seeded quantization error bounds. They reject
invalid shapes/dimensions, nonfinite/unrepresentable weights, oversized or
truncated payloads and strided buffers. Conversion tests cover offsets/hashes,
exclusive publication, late validation failure and hard-link failure cleanup.

Execution tests use an independent NumPy unpackbits bit-plane decoder and the
existing FP32 SwiGLU executor, non-square multi-block projections, multiple
experts/routes/tokens, zero routes, every half pattern, malformed/nonfinite
scales, scratch/route/input limits, arithmetic overflow, actual compiled
signatures and LLVM allocation inspection, real buffered/mmap stores, retained
chained exceptions, and subprocesses for missing/disabled Numba.

Tolerance is rtol=5e-5/atol=2e-7 versus decoded FP32 matrix reductions and
rtol=5e-6/atol=1e-9 between scalar/native experts, allowing reduction ordering
and FP32 sigmoid rounding. Model tests use rtol=8e-5/atol=8e-6 against
an independently decoded test-only backend. Identical native gateway/direct
configurations are compared exactly; no scalar/native bitwise claim follows.

Observed verification on 2026-10-02: all 17 Q5_0 tests pass with `-W error`,
including actual compiled execution, custom registry, SafeTensors/model and
gateway checks, with no skips. The combined Q4_0/Q4_1/Q5_0/Q8_0 format/native/
OLMoE suites pass 69 tests with `-W error`. This is synthetic contract and
regression evidence, not public-checkpoint quality or performance evidence.

## Remaining Gaps

Q5_0 provides one format within the broader format requirements. At the time
of the recorded assessment, Q5_1, Q8_1, Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K, IQ variants,
Q1_0/Q2_0, ternary TQ1_0/TQ2_0, MXFP4/NVFP4/other FP4 and FP8 E4M3/E5M2
formats and their scale/container contracts are not implemented here.
Subsequent format implementations require separate implementation and verification evidence.
GPU packed kernels, GGUF import, public-checkpoint quality/performance
certification and upstream quantizer comparison remain absent.

Existing Q4_0/Q4_1/Q8_0 contracts were inspected and regression-tested without
reimplementation. The recorded evidence covers these inspections and test runs.
