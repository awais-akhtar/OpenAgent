# Bounded V4.1 Local FP4 Codec

This is an original standalone arithmetic codec, not a decoder or a checkpoint
reader. There is no decoder registration or checkpoint acceptance, no V4 alias,
and no publisher implementation import or reference-source translation.
`NativeDeepSeekV41Quant` is available through the public lazy
`openagent.adaptive_engine` export for runtime composition. Importing the class
does not compile or load a native library; construction requires an explicit
caller-provided compiled library file. There is no automatic compilation.

## Evidence

- [V4.1 report v1, section 2.4.4](https://arxiv.org/html/2609.19969v1#S2.SS4.SSS4)
  specifies E2M1 main KV with one E4M3 scale per 16 channels, after RoPE,
  without a second global scale. The product maximum is 2688. This codec does
  not perform RMSNorm or RoPE; those belong to later runtime composition.
- [OCP MX specification v1.0, sections 5.3.1 and 5.3.3](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
  supplies E4M3 and E2M1 definitions, subnormal support, and FP4 ties-to-even
  rounding and saturation. The E4M3 table reproduces the OCP FP8 definition.
- [Public PyTorch dtype documentation](https://docs.pytorch.org/docs/stable/tensor_attributes.html)
  identifies `float8_e4m3fn`. Tests use installed, independently maintained
  PyTorch conversion as an external E4M3 format oracle, not as a V4.1 oracle.
  No source from that implementation is copied, translated, or imported by
  the runtime module. No independent executable FP4 oracle was run: E2M1
  tests are analytical format fixtures, not external numerical certification.

The report does not independently establish published packing, scale selection,
ties, or fused-kernel arithmetic here. No checkpoint compatibility or publisher
kernel parity is claimed. This is not MXFP4 (which uses E8M0 and group 32), nor
the full NVFP4 recipe with a global scale.

## Local Contract

Inputs are contiguous row-major FP32, rows 1..32, channels 16..512 divisible by
16. Groups never cross rows. Every local group is **9 bytes**: positive E4M3
scale first, then eight bytes of E2M1 payload. The even channel occupies the
low nibble; the odd channel occupies the high nibble. This is explicitly a
local serialization, not an assertion about any published tensor.

E2M1 uses sign bit 3, exponent bits 2..1 (bias 1), fraction bit 0. Positive
codes 0..7 represent 0, 0.5, 1, 1.5, 2, 3, 4, 6. Code 8 is negative zero.
E4M3 uses sign bit 7, exponent bits 6..3 (bias 7), fraction bits 2..0.
Scale codes accepted are 0..126 only: nonnegative finite values, including
subnormals (minimum 1/512), minimum normal 1/64 and maximum 448. Negative
scales, negative zero scales and NaN codes 127/255 are rejected.

`oa_v41_quant_encode` accepts caller-supplied scale bytes. Values are divided
by the exact decoded scale in FP64, rounded to nearest E2M1 with ties resolved
to the even low significand bit, and saturated at magnitude 6. Underflow rounds
to signed zero; signed zero is preserved. NaN/Inf inputs are rejected even when
saturation would otherwise apply. Zero scale permits only signed-zero inputs.

`oa_v41_quant_scales` is a **local optional policy**, not a report-derived
algorithm: take group max absolute value / 6 in FP64, round to nearest finite
positive E4M3, ties-even; saturate at 448. For a nonzero group whose scale would
round to zero, use minimum subnormal scale 1/512. All-zero groups use zero.
Scale rounding can produce clipping; no no-clipping guarantee is implied.
The encoder supports arbitrary finite FP32 inputs, including those exceeding
2688, with explicit saturation. Decode multiplies exact scale and E2M1 values;
these bounded binary products are exactly representable in FP32.

## ABI And Composition

The C file exports three codec functions returning int status and a private
uint32_t ABI-version handshake. Python loads only an
explicit caller-provided compiled library path; it never compiles or falls back.
The path is strictly resolved and must name an existing regular local file;
Windows UNC/network paths are rejected. A bare library search name is not passed
to CDLL. This does not sandbox native code or authenticate its provenance: the
caller must trust the selected library and its platform-resolved dependencies.

`oa_v41_quant_abi_version(void)` returns uint32_t **1**, the private local ABI
version, unrelated to checkpoint/model versions. The wrapper checks it before
binding codec functions. Missing or mismatched handshake rejects construction
and releases the acquired handle. A `weakref.finalize` callback owns that handle
without retaining the wrapper, calling FreeLibrary on Windows or dlclose on
POSIX exactly once. `close()` is idempotent; wrapper calls after close fail.
Do not retain or invoke raw `library` function objects beyond wrapper lifetime,
and do not race close/collection with native calls. No thread-safe shutdown is
claimed. A matching handshake alone cannot prove that a library is correct.

Python bool values are rejected for rows, channels and budget before integer
conversion. Value iterables consume at most element_count+1 items before count
rejection. Scale/packed buffer byte extents are checked before copying; generic
byte iterables consume at most expected_count+1 items. Arbitrary `__bytes__`
conversions are not invoked. Validated payload sizes remain bounded by geometry.
Custom iterator/index conversion code itself remains caller-controlled.
The private ABI signatures in the source are the authoritative declarations:

```text
scales(x, float_count, rows, channels, scales, scale_count, budget_bytes)
encode(x, float_count, scales, scale_count, rows, channels,
       packed_output, packed_byte_count, budget_bytes)
decode(packed, packed_byte_count, rows, channels,
       float_output, float_count, budget_bytes)
```

Symbols have prefix `oa_v41_quant_`. Dimensions are uint32_t, lengths and
budget are size_t; pointers name float or uint8_t arrays. Counts must match
geometry exactly, not merely exceed it. Status 1 is invalid geometry, count,
null pointer, float alignment or address-extent arithmetic; 2 is insufficient
staging budget; 3 is nonfinite input or invalid scale/value combination.
Validation order is geometry/extents, budget, then numerical content.

Logical staging budgets are group_count bytes for scales, 9*group_count for
encode, and 4*element_count for decode. No heap allocation occurs. Fixed stack
arrays reserve up to 1024, 9216 and 65536 bytes respectively even on small
calls; budget limits logical staging, not actual stack reservation. Scalar
locals/compiler overhead are additional. Python's input/output allocations are
not counted by this native scratch budget. Wrapper auto-scale calls are serial.

All output is staged before publication, including a late bad value/scale.
Failure leaves output bytes unchanged. Inputs may alias output because all reads
finish first. Callers must supply truthful live allocation extents, float
alignment, and exclusive access; the ABI cannot inspect allocation ownership or
make invalid pointers safe. No hidden state or global mutable cache exists.
Dimensions cap multiplication before pointer checks. The codec can be composed
with a runtime-owned cache without taking ownership of cache lifecycle.

## Verification And Limits

Run `.accelerator-venv/Scripts/python.exe tests/test_native_deepseek_v41_quant.py -v`.
The tests compile the actual C source independently using MSVC `/W4 /WX /O2`
or C99 `-Wall -Wextra -Werror -O2`; a missing compiler fails the test.
The external Torch oracle is optional in the base suite: `find_spec` absence
produces an explicit unittest skip, never a parity pass. An installed but broken
Torch import or failed numerical comparison still fails. Strict acceptance must
use the accelerator environment with Torch installed and count any skip as
failure.
The Python module is loaded directly,
without shared package exports or a model loader.

Observed on 2026-10-03: Windows MSVC strict compiled accelerator run, eleven tests
passed with zero skips. The absence-guard regression test verifies the exact reason;
external format oracle was PyTorch 2.14.1+cpu. No sanitizer or alternate-platform
build was run.

Fixtures cover all 16 FP4 codes, signed zeros, subnormals, every positive/negative
FP4 midpoint and adjacent probes, finite saturation, all 127 accepted scales,
E4M3 midpoint conversions against PyTorch, maximum geometry, row-chunk equality,
explicit-scale re-encoding, wrapper validation, input/output aliasing, pointer
alignment/address overflow, exact counts/budgets, canaries, and late numerical
rollback. These establish only the documented local codec. There is no claim
of full decoder logits, GPU equivalence, checkpoint acceptance, cache integration,
official scale policy, sanitizer coverage, or independent executable E2M1 parity.
Additional wrapper fixtures verify resolved file loading, actual native version
handshake, mocked missing/mismatched handshake rollback, real OS handle release
on GC and explicit close, double-close prevention, closed-call rejection, bool
rejection, infinite iterable bounds, and oversized buffer rejection before copy.
The POSIX dlclose branch has not been executed in this Windows environment.

Wrapper operations and explicit close share a lock. Close waits for another
thread's active call; a close invoked re-entrantly by an input iterator is
rejected before unloading the library. Tests exercise both cases. This protects
wrapper calls, not raw native function pointers retained by callers.
Scale and packed-byte iterators exercise the same protection; conversion
exceptions unwind the active-call counter before subsequent use or close.
