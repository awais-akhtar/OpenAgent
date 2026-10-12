# Independent Grouped INT4

Public geometry is from [reviewed Colibri docs/FORMATS.md, ordinal 4](https://github.com/JustVugg/colibri/blob/ce370e87d7b623d7759b52ec2007d75fc5b0e87e/docs/FORMATS.md).
Only that document's format metadata was used. No reference implementation
was inspected, imported or copied. This is a direct-byte OpenAgent grouped
codec and quantizer; matching geometry is not universal Colibri byte,
quantizer, container or execution parity.

## Contract

Each [rows,columns] projection has rows*ceil(columns/2) bytes, even columns
in low nibbles and odd columns in high nibbles, and rows*ceil(columns/gs)
little-endian FP32 scales. gs is a positive per-tensor/group-size parameter,
not necessarily 64 or a divisor of row width. Final groups may be partial.
Odd rows have a zero high padding nibble, with no padding between groups.
Groups can therefore split a packed byte between its two nibbles.

The independent OpenAgent code convention is signed two's-complement
nibbles [-8,7], including -8. Weight = FP32(code*FP32(scale)), with scale
applied before input multiplication and reduction. All finite signed FP32
scales, including signed zero and subnormals, are accepted. Nonfinite
scales and nonzero odd-row padding are rejected before any arithmetic,
including zero-weight routes. Finite scales whose actual codes overflow
FP32 weight decoding raise ArithmeticError; finite maximum scales with
zero codes remain legal. This code convention and strict padding policy
are explicit, not inferred as proven Colibri parity from byte counts.

Expert bytes concatenate gate [i,h], up [i,h], down [h,i] code arrays,
then gate/up/down scale arrays, each scale array in row/group order. No
headers or inter-array alignment are present; scales may be unaligned.
All dimensions are positive integers, including 1 and odd dimensions.
Size is

```text
2*i*ceil(h/2) + h*ceil(i/2)
+ 4*(2*i*ceil(h/gs) + h*ceil(i/gs))
```

Group size is external trusted metadata. Wrong group sizes sometimes have
identical byte counts and cannot be detected from payload alone. Format
identity, group size, model fingerprint and snapshot execution identity
must be persisted and validated by shared runtime integration. A headerless
payload is not a Colibri container importer and does not carry ordinals.

## APIs

- `grouped_int4_expert.DTYPE="int4-grouped"`, `DEFAULT_GROUP_SIZE=64`.
- `payload_size(h,i, group_size=64)` and `scratch_bytes(h,i, group_size=64)` use keyword-only group_size.
- `pack_matrix(matrix, group_size=64)` returns `(packed_bytes, scales)`; scales is an owned little-endian FP32 [rows,ceil(columns/gs)] array.
- `pack_expert(gate,up,down, group_size=64)` returns the complete expert bytes.
- `decode_expert(payload,h,i, group_size=64)` explicitly returns full owned FP32 matrices, not kernel scratch.
- `GroupedInt4ExpertBackend(group_size=64, max_scratch_bytes=...)`, name `grouped-int4-reference`.
- `native_grouped_int4_expert.NativeGroupedInt4ExpertBackend(group_size=64, max_scratch_bytes=...)`, name `grouped-int4-numba`.
- `grouped_int4_conversion.convert_experts(experts,output, hidden_size=..., intermediate_size=..., group_size=64)` returns `GroupedInt4ConversionResult`.

The conversion result exposes path, hidden_size, intermediate_size,
group_size, immutable ExpertRegion mapping, dtype and activation="swiglu".
Regions have consecutive offsets, exact sizes and SHA256. Existing outputs
are preserved; same-directory temporary creation, file fsync and exclusive
hard-link publication prevent partial publication. Failures remove the
temporary; missing hard-link support raises OSError. Directory crash
durability is not promised. Model, layer and routing identity is supplied by the integration layer.

## Quantization And Scratch

Independent positive per-group max-abs/7 FP32 scales, ties-to-even [-7,7]
codes, zero groups scale 1, nonzero underflow scales floored to the smallest
FP32 subnormal. Generated weight overflow is rejected, not emitted. Real
finite FP32-representable weights are required; extremely large finite
inputs can still fail the quantized reconstruction check. The decoder's
-8 code is valid although this symmetric quantizer does not emit it.

pack_matrix stages one group of source values/codes at a time, apart from
source normalization, output nibble bytes and output scale array. Non-array
normalization may allocate a full source matrix. pack_expert holds projection
bytes/scales while joining; file conversion holds one packed expert at a time.

Both executors hold one expert lease and one i-value FP32 activation array
(4*i bytes), independent of gs and expert count. They allocate no decoded
weight or scale arrays. Input normalization, output, routing, borrowed payload,
fixed Python objects and compiler workspace are separate. Scalar uses bounded
struct reads; native reads FP32 scale bits directly, including unaligned fields.
SwiGLU is stable, weights are scaled before reduction, and all intermediate
or weighted-output overflow is an error. Chained borrowed exception frames
are detached before mmap release.

Compiled selection requires active Numba, no implicit scalar fallback.
Missing/disabled JIT marks the native backend unavailable and execution raises
RuntimeError. nogil=True, fastmath=False, cache=False. No SIMD, throughput or
quality claim is made. Scalar execution remains explicitly available.

## Runtime Integration

No shared integration files were edited. The execution service must add int4-grouped dtype
capability/manifest validation, default scalar and opt-in native registration,
SafeTensors pack dispatch and group metadata. `pack_matrix` returns bytes plus
scales like row INT4, not an inline-metadata Q-format. The packer must write
all three code arrays before the three group-scale arrays. Suggested artifact
extension is .gi4, separate from row INT4 and Q4 formats.

For initial default integration, gs=64 is explicit and fixed. Supporting other
group sizes requires a schema field and roundtrip validation, configured
backend dispatch per layer, and binding group_size in cache/snapshot identity.
Never silently execute a variable-gs payload with the default gs or infer gs
from size. Shared registry names alone do not distinguish gs configurations.
The converter result already includes gs for adaptation. The custom-registry
test skips until execution-service dtype acceptance; thereafter missing dispatch fails.

## Verification

```powershell
.venv/Scripts/python.exe -W error -m unittest tests.test_grouped_int4_expert tests.test_native_grouped_int4_expert -v
```

Evidence includes independently hand-built bytes and vectorized signed-code/
group-scale oracle; all nibbles, signed/zero/subnormal and extreme scales;
odd rows, partial groups, non-square 65/129 dimensions, groups splitting bytes;
quantizer bytes/ties/subnormal error bounds; malformed sizes/padding/scales;
route/input/scratch validation before leases; overflow and zero-route validation;
random FP32 bit patterns and boundary-field reader comparisons; real buffered/
mmap stores, retained chained exceptions; hashes/atomic publication cleanup;
missing/disabled JIT subprocesses; actual nopython signatures and LLVM
allocation inspection. All mathematical tests execute directly even before
shared registration. Tolerances are rtol=5e-5/atol=2e-7 versus FP32 matrix
reductions and rtol=5e-6/atol=1e-9 scalar/native, not bitwise parity promises.

Observed run: 17 owned tests under `-W error`, 16 passed and one registry
integration test explicitly skipped pending execution-service dtype integration. The combined
grouped/Q5_1/Q5_0/Q4_1/Q4_0/Q8_0 regression run passes 102 tests with that one
skip. Ruff passes for the owned grouped and Q5_1 modules/tests.

This adds an independent implementation for the actual int4-grouped target
geometry. INT2 row, INT3 g64, E8 lattice and MXFP4 remain outside this component;
FP8 E4M3 block-128 implementation is tracked separately. Model, CLI and gateway
integration evidence is maintained separately. ENGINE_PARITY.md provides the
authoritative target inventory.
