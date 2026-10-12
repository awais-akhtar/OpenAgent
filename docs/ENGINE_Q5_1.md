# Independent Q5_1

The implementation supports the affine Q5_1 format. Public declarations
were checked on 2026-10-02 in [block_q5_1 field metadata](https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h#L223-L235)
and the [ggml_type enum](https://github.com/ggml-org/ggml/blob/master/include/ggml.h#L367-L412).
Only declarations were used, not reference quantizer/kernel implementation.
The links are mutable upstream refs; the explicit inventory below is the
finite inspected snapshot, not a claim that future upstream additions are
covered. The metadata-only fetch did not provide a pinned commit identifier.

## Byte Contract

Each 32-value block occupies 24 bytes, little-endian:

| Offset | Bytes | Field |
| --- | --- | --- |
| 0 | 2 | IEEE binary16 delta d |
| 2 | 2 | IEEE binary16 minimum m |
| 4 | 4 | Fifth-bit plane qh: bit k belongs to lane k |
| 8 | 16 | Split nibbles qs: low for lanes 0..15, high for lanes 16..31 |

Code q is the lane's nibble plus 16 times its fifth bit, in [0,31]. Decode is
FP32(q*float(d)+float(m)), using binary64 affine arithmetic then one FP32
rounding. Both metadata fields may be signed finite normals, subnormals or
zero. No centered-code bias is applied. NaN/infinity in either field is
rejected before arithmetic, including on zero-weight routes. Valid negative
deltas and offsets remain legal even though the converter generates d>=0.
Q5_0 bytes cannot be reinterpreted as this format.

Experts concatenate row-major gate [i,h], up [i,h], down [h,i], with no header,
padding, inter-matrix alignment or appended row scales. h and i must be
positive integers divisible by 32. Size is 3*h*i/32*24. External metadata binds
shape and identity; legal bit-plane corruption requires SHA256 protection to
detect, and equal-sized dimension swaps cannot be detected from bytes alone.

## Public APIs

- `q5_1_expert.DTYPE="q5_1"`, `BLOCK_WIDTH=32`, `BLOCK_BYTES=24`.
- `payload_size(hidden_size, intermediate_size)` returns exact expert bytes.
- `pack_matrix(matrix, block_count=64)` and `pack_expert(gate, up, down, block_count=64)` return bytes; block_count is keyword-only.
- `decode_expert(payload, hidden_size, intermediate_size)` returns three full owned FP32 matrices, not kernel scratch.
- `Q5_1ExpertBackend(max_scratch_bytes=...)` selects scalar `q5_1-reference`.
- `native_q5_1_expert.NativeQ5_1ExpertBackend(max_scratch_bytes=...)` selects compiled `q5_1-numba`.
- Both format/kernel modules export `scratch_bytes(hidden_size, intermediate_size)`, exactly 4*i explicit activation bytes.
- `q5_1_conversion.convert_experts(experts, output, hidden_size=..., intermediate_size=..., block_count=64)` returns `Q5_1ConversionResult`.

Both backends expose info(), execute() and execute_batch() with the shared
lease/source and routing contract. The converter result exposes path,
hidden_size, intermediate_size, immutable regions, dtype and activation.
Each ExpertRegion has a relative filename, consecutive offset, exact size
and SHA256. The integration layer supplies model/layer/routing identity when adapting these
regions into ExpertLayer/SparseExpertManifest.

## Quantization Policy

Independently specified, not upstream quantizer parity: m=half(min(block)),
d=half((max(block)-min(block))/31). For nonconstant blocks, floor a nonzero
delta rounding to zero to 2^-24, assign ties-to-even codes against stored
metadata, and clip to [0,31]. Constant blocks have d=0 and all-zero codes,
with the constant rounded to half in m. Subtraction/division for code
selection uses binary64 to avoid overflowing finite FP32 endpoint differences.
Nonfinite input, FP32-unrepresentable input or half metadata overflow is an
error. A representable minimum does not permit an overflowing delta.

Conversion staging is bounded by block_count*32 plus fixed block temporaries,
excluding source normalization and returned packed bytes. Non-array source
normalization can allocate a full matrix. pack_expert holds its three packed
projection byte results while joining. The file converter holds one packed
expert while writing, then releases it. Publication uses a same-directory
temporary, file fsync and exclusive hard link; existing files are preserved.
Validation/publication failures remove temporaries. Filesystems without
hard-link support fail. No crash-durable directory-publication claim is made.

## Execution And Reuse

Scalar and compiled Q5_1 projections are dedicated affine byte routines.
They never allocate decoded weight matrices or scale arrays. Scalar execution
unpacks a fixed-size block tuple. Compiled projection reads d/m once per
block, extracts codes from leased bytes, and uses ordered FP32 products and
accumulation. SwiGLU is stable for both gate signs. Nonfinite projections,
activation or weighted output raise ArithmeticError; partial results are
never returned.

Shared functionality from the independent Q5_0 modules is limited to positive-integer validation,
chained borrowed-frame cleanup and binary16 field reading. None of these
interprets Q5_0 layout when invoked from Q5_1. All affine field offsets,
24-byte strides, quantization, decode and numerical execution remain Q5_1.
No reference implementation is imported or copied.

Explicit array scratch is one i-value FP32 activation vector, checked before
lease acquisition. Inputs/output, routes, borrowed payload, fixed Python block
objects and compiler workspace are separate. Input conversion can allocate a
contiguous FP32 batch. One expert lease is held at a time; retained scalar or
compiler exceptions detach chained borrowed frames before releasing mmap.

Native execution requires active Numba. Missing Numba, NUMBA_DISABLE_JIT=1 or
lazy compilation failure never invokes scalar fallback. An unavailable native
backend reports unavailable and raises RuntimeError. JIT uses nogil,
fastmath=False and cache=False. There is no SIMD or performance claim.

## Shared Integration

The shared integration layer manages registries, manifests, the SafeTensors
packer, OLMoE, CLI, gateway and exports. Recorded integration work connects q5_1
using the names above and the distinct defaults openagent-experts.q51 and
openagent-experts.q51.manifest.json. Required
dispatch is payload sizing to q5_1_expert.payload_size, and SafeTensors packing
to pack_matrix without appended row scales. Default registration uses the
scalar class; compiled registration is explicit. Converter export may be
convert_q5_1_experts. The owned custom-registry test passes; end-to-end model
and gateway tests are tracked separately from the format-specific test evidence.

Gateway selection after shared integration:

```text
OPENAGENT_OLMOE_MODEL_DIR=<checkpoint directory>
OPENAGENT_OLMOE_MANIFEST=<Q5_1 manifest path>
OPENAGENT_OLMOE_KERNEL=q5_1-numba
OPENAGENT_OLMOE_IO_MODE=mmap
```

Use q5_1-reference for explicit scalar execution.

## Verification

```powershell
.venv/Scripts/python.exe -W error -m unittest tests.test_q5_1_expert tests.test_native_q5_1_expert -v
```

All 16 owned tests pass with strict warnings and no skips in the integrated
workspace. Tests cover all 32 codes, every fifth-bit lane, signed/extreme/
subnormal metadata, zero delta with nonzero minimum, affine rounding, exhaustive
half reading for both field offsets, exact quantizer bytes, ties-even/constants,
metadata overflow, strided source and payload behavior, malformed lengths,
nonfinite input/metadata, zero-route validation, scratch/route limits,
arithmetic overflow, multiple experts/routes/tokens, nonsquare multiblock
projections, independent bit-plane FP32 differentials, real buffered/mmap
leases, retained chained failures, converter offsets/hashes/exclusive
publication/failure cleanup, missing/disabled/lazy-failed JIT, nopython
signatures and LLVM allocation inspection.

Independent decoding uses NumPy unpackbits and binary64 affine reconstruction,
then the established FP32 SwiGLU executor. Expert tolerances are rtol=5e-5,
atol=2e-7 versus matrix reductions, and rtol=5e-6, atol=1e-9 scalar/native,
allowing FP32 reduction and sigmoid rounding differences. An ordered-dot
fixture and basis projections are exact. These tests do not claim upstream
quantizer/byte/execution parity or public-checkpoint quality/performance.

## Actual Colibri Target Scope

The actual comparison target is the reviewed Colibri registry, not the GGML
quantized enum. Metadata-only source is
[Colibri docs/FORMATS.md at ce370e87d7b623d7759b52ec2007d75fc5b0e87e](https://github.com/JustVugg/colibri/blob/ce370e87d7b623d7759b52ec2007d75fc5b0e87e/docs/FORMATS.md).
The matching local checkout's commit was verified. No Colibri implementation
files were read for this scope correction, imported, copied or translated.

The finite ordinal scope is these **nine** format names, 0 through 8:

| Ordinal | Target Name | Metadata Geometry | OpenAgent Track Status |
| --- | --- | --- | --- |
| 0 | f32 | O*I float32 values, no scales | Independent float32 path exists; no Colibri container/parity certification |
| 1 | int8-row | O*I signed bytes, O float32 scales | Independent path exists; no universal parity claim |
| 2 | int4-row | O*ceil(I/2) nibble bytes, O float32 scales | Independent path exists with its own reserved-code/quantizer policy; no universal parity claim |
| 3 | int2-row | O*ceil(I/4) bytes, O float32 scales | Missing |
| 4 | int4-grouped | O*ceil(I/2) nibble bytes, O*ceil(I/gs) float32 scales; gs is per-tensor | Independent path now added; shared integration/Colibri parity remain separate gates |
| 5 | int3-g64 | O*ceil(I/64)*24 bytes; 16-byte low plane plus 8-byte high plane per group; O*ceil(I/64) float32 scales | Missing |
| 6 | e8-iq3-lattice | O*ceil(I/256)*98-byte super-blocks with embedded half scales, lattice/sign metadata; rotated weights and activations | Missing |
| 7 | mxfp4 | O*ceil(I/2) E2M1 nibble bytes; UE8M0 byte scales per 32-input group in container | Missing |
| 8 | fp8-e4m3-b128 | O*I raw E4M3 bytes; ceil(O/128)*ceil(I/128) block scales | Missing |

At the scope correction, six targets were missing. The grouped INT4 track
has since added independent code and kernel tests, leaving int2-row,
int3-g64, e8-iq3-lattice, mxfp4 and fp8-e4m3-b128 without completed evidence
within this format-specific scope. FP8 implementation is tracked separately.
ENGINE_PARITY.md provides the authoritative current inventory.
Q5_1 is useful extra support, not completion of a Colibri ordinal target.
Next priorities are int2-row and int3-g64, then MXFP4 UE8M0; E8 lattice remains an
explicit target requiring public codebook/bit mapping and FWHT rotation math.
Byte geometry alone does not establish signed-code mapping, padding semantics,
lattice codebook or quantizer parity. Missing declarations must be resolved
from public metadata/math, not by copying kernels.

MXFP4's reviewed path is Vulkan-specific; implementing an independent CPU
kernel must not imply reviewed CPU/Metal parity. Its container UE8M0 group
scales are distinct from a generic GGML block wrapper. FP8's reviewed format
implements float32 block scales and recognizes but refuses UE8M0 block scales;
support for those encodings must be declared separately. Names and scaling
geometry, not byte count alone, must identify ambiguous INT8/FP8 payloads.

The same document lists int4-rans256-g0 without a public ordinal as a merged
offline-tools-only, stamp-required entropy container. It is not one of the
nine ordinal targets and remains a separate explicit container/codec gap.
Its data-dependent size must not be assigned a fabricated fixed-size formula.

## Supplementary GGML Scope

For a separate optional extension inventory, use active ggml_type enum entries
paired with block declarations, including intermediate dot-product formats
Q8_1 and Q8_K. This yields **27 packed formats**. The implementation set recorded
in this assessment comprises Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0: five formats,
leaving **22** outside these format implementations. This is **not** the Colibri parity
target inventory above:

| Family | Remaining Active Metadata Names (Enum IDs) |
| --- | --- |
| Basic | Q8_1 (9), Q1_0 (41), Q2_0 (42) |
| K blocks | Q2_K (10), Q3_K (11), Q4_K (12), Q5_K (13), Q6_K (14), Q8_K (15) |
| IQ blocks | IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ1_S (19), IQ4_NL (20), IQ3_S (21), IQ2_S (22), IQ4_XS (23), IQ1_M (29) |
| Ternary | TQ1_0 (34), TQ2_0 (35) |
| FP4 | MXFP4 (39), NVFP4 (40) |

The implemented IDs are 2, 3, 6, 7 and 8 respectively. Dense/scalar entries
F32, F16, I8, I16, I32, I64, F64 and BF16 are outside the packed-block scope.
Removed/commented names are not targets: Q4_2/Q4_3 and interleaved
Q4_0_4_4/Q4_0_4_8/Q4_0_8_8, IQ4_NL_4_4/IQ4_NL_4_8/IQ4_NL_8_8.
GGML_TYPE_COUNT=43 is an enum sentinel, not 43 active formats.

Standalone FP8 E4M3/E5M2, arbitrary other FP4 variants, GPTQ/AWQ and other
container contracts do not appear as packed enum types in this GGML snapshot.
NVFP4's E4M3 scale field is not a standalone FP8 weight format. Such requests
cannot be inferred from this GGML checklist. In particular, fp8-e4m3-b128 is
already an actual Colibri target above, even though absent from this enum.
Other additions need explicit separate public contracts. IQ declarations may require
additional public codebook/bit-mapping metadata before clean-room execution;
this inventory is not evidence those metadata alone specify their decode.

GPU kernels, GGUF import, public-checkpoint quality/performance certification
and upstream numerical parity remain separate unimplemented work. Other
tracks can change implementation status, but must report their own evidence.
