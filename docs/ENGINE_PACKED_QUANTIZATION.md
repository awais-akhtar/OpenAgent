# Packed Quantization

The engine uses explicit dtype names for every packed layout. The row INT4
layout described first is an owned OpenAgent format, not GPTQ, AWQ, GGUF, or
an upstream checkpoint ABI. Dtype: `int4-row`. Backend: `PackedInt4Backend`, reported as
`packed-int4-reference`. `PackedRoutedSwiGLUBackend` is an alias for the same
class, used by the kernel registry. NumPy FP32 arithmetic, not native
packed dot products.

## Layout

Dimensions are external metadata: gate/up `[intermediate, hidden]`, down
`[hidden, intermediate]`. No header. Concatenate gate bytes, up bytes, down bytes,
then gate scales, up scales, down scales. Scales are little-endian FP32.
Each row occupies `ceil(columns/2)` bytes: even column in low nibble, odd in high.
Signed two's-complement codes represent -7 through 7; code 8 (-8) is reserved.
Odd row widths require a zero high padding nibble on every row.
Exact payload length is
`2*intermediate*ceil(hidden/2) + hidden*ceil(intermediate/2) + 4*(2*intermediate+hidden)`.
Length, reserved codes, padding, and all scales are checked before execution.
Scales must be positive, finite and satisfy `7*scale <= float32_max`.
There is no checksum here; use store-region SHA256 for corruption detection.
Equal-length dimension permutations cannot be detected without trusted metadata.

## APIs

- `pack_matrix(matrix, block_rows=32) -> (bytes, scale_array)` processes one
  projection with row-block scratch. Scale array is little-endian FP32.
- `pack_expert(gate, up, down, block_rows=32) -> bytes` composes that layout.
- `payload_size(hidden, intermediate) -> int` computes exact byte size.
- `decode_expert(payload, hidden, intermediate, block_rows=32)` returns three
  owned FP32 matrices. This convenience API allocates full decoded weights.
- `PackedInt4Backend(block_rows=32)` implements `info`, `execute`, `execute_batch`
  using the existing `acquire(key)` context-manager source interface.

Packing rejects non-real, non-finite and FP32-unrepresentable inputs. Quantize
FP32 weights with row scale max(abs(row))/7, round ties to even, clamp [-7,7].
Zero rows use scale 1. Subnormal scales are floored to the smallest positive
FP32 value; extreme scales are capped to permit finite dequantization.
For ordinary rows the absolute element error is at most scale/2 plus FP32
rounding (tests allow `2*eps*max(abs(row))`). Subnormal rows need an absolute
rounding allowance of one smallest positive FP32 value. End-to-end relative
SwiGLU error has no universal bound near cancellation: tests check a seeded
moderate-weight workload below 20% relative L2 error and compare decoded
execution against the floating backend independently.
Tests also propagate an absolute bound: projection errors are
`e_g=abs(delta_gate)@abs(x)` and `e_u=abs(delta_up)@abs(x)`;
`e_a=2*e_g*(abs(up)+e_u)+abs(silu(gate))*e_u` uses a conservative global
SiLU Lipschitz constant of 2. Output error is bounded by
`abs(decoded_down)@e_a + abs(delta_down)@abs(original_activation)`, plus
floating-point arithmetic allowance. This is input-dependent, not a model
quality guarantee.

Execution decodes at most `block_rows * max(hidden, intermediate)` weights at
a time. Scratch excludes input/output and leased payload: O(block_rows*max(h,i)
+ intermediate), independent of expert count and batch size. Batch execution
loops token/expert pairs, trading throughput for bounded scratch. No decoded
cache, normalization, router changes, fused-kernel or acceleration claims.
Non-finite intermediate/output arithmetic raises ArithmeticError.

## Engine Integration

The kernel registry includes the INT4 reference backend without expanding the
original NumPy backend's supported dtype set. Manifest validation uses the
exact payload size. The existing SafeTensors directory packer accepts
`dtype="int4-row"`; INT8 remains the default. It reads one projection at a time,
writes packed bytes and appends the three scale arrays. DiskExpertStore retains
its normal lease and checksum checks.

```text
SafeTensors -> row INT4 packer -> aligned expert container
                                      |
                         manifest + leased expert bytes
                                      |
                reference blocks / compiled nibble reductions
                                      |
                             routed SwiGLU output
```

```sh
openagent-olmoe pack MODEL --dtype int4-row --output experts.i4 --manifest experts.i4.json
openagent-olmoe generate MODEL MODEL/experts.i4.json "Hello" --kernel packed-int4-reference
openagent-olmoe generate MODEL MODEL/experts.i4.json "Hello" --kernel packed-int4-numba
```

`NativePackedExpertBackend` (`packed-int4-numba`) is an optional Numba CPU
kernel. It extracts signed nibbles inside its dot-product loops, uses FP32
reductions without fastmath, and keeps weight bytes borrowed for the lease
lifetime. It does not expand expert matrices or cache decoded weights. Scratch
contains the intermediate activation vector and routing assignments. Odd and
even shape tests compare actual JIT results with the reference backend; this
does not establish a universal speedup or unchanged model quality after INT4
quantization. The gateway accepts `OPENAGENT_OLMOE_KERNEL=packed-int4-numba`.
Segment snapshot precision remains INT8 and rejects INT4 manifests. Full-model
state v2 binds loaded dense values, expert content hashes, dtype and selected backend names;
cross-precision restore is rejected before KV mutation. Legacy full-model state
v1 is rejected because it lacks that execution identity. Segment state and
portable snapshots also use v2, binding loaded dense values and selected expert content hashes and sizes;
legacy segment v1 snapshots are rejected before changing KV.

## Public Checkpoint Smoke Test

The local `allenai/OLMoE-1B-7B-0125-Instruct` checkpoint at revision
`b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e` was packed into 1,024 experts.
The aligned INT4 container occupied 3,238,002,688 bytes, compared with
6,459,228,160 bytes for its INT8 container. Conversion took 56.669 seconds.
The compiled INT4 CPU backend produced `[19, 5043, 374, 18207]` for the
four-token arithmetic smoke prompt, matching the separate Transformers BF16
oracle. The final verification run took 20.163 seconds including loading and
initial JIT. Row scales are applied before reduction to avoid overflow in
unscaled nibble sums for tiny weights and large finite activations.
This is one short output check, not universal preservation of model quality or
a controlled speed comparison. INT4 remains an explicit choice.

## Additional Fixed-Geometry Formats

These formats use the same manifest, leased store and routed execution path.
The original INT8 default has not changed. Each optional compiled backend is
selected explicitly; unavailable or disabled Numba raises an error rather
than silently executing a Python kernel.

| Manifest dtype | Reference backend | Compiled CPU backend | Scale geometry |
| --- | --- | --- | --- |
| `int2-row` | `int2-row-reference` | `int2-row-numba` | One little-endian FP32 scale per row |
| `int3-grouped-g64` | `int3-grouped-g64-reference` | `int3-grouped-g64-numba` | One FP32 scale per 64 columns, including a partial final group |
| `int4-grouped-g64` | `grouped-int4-reference` | `grouped-int4-numba` | One FP32 scale per 64 columns, including a partial final group |
| `mxfp4-e2m1-g32-ue8m0` | `mxfp4-e2m1-g32-ue8m0-reference` | `mxfp4-e2m1-g32-ue8m0-numba` | One UE8M0 byte per padded 32-element row group |
| `q8_1` | `q8_1-reference` | `q8_1-numba` | Two binary16 fields and 32 signed INT8 codes per block |

Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0 implementations also remain available. Their
formats are distinct from row/group INT4. FP8 E4M3 with FP32 128-by-128 block
scales uses the explicit dtype `fp8-e4m3-b128-f32`.

### INT2 and INT3

`lowbit_expert.py` implements two owned layouts. Gate/up shapes are `[i,h]`;
down is `[h,i]`. Each row is an LSB-first stream of two's-complement codes,
occupying `ceil(columns*bits/8)` bytes. INT3 codes may cross byte boundaries.
Unused high bits in the final byte must be zero. Gate, up and down code bytes
come first; their little-endian FP32 scale arrays follow in the same order.
All bit patterns represent signed values, including -2 for INT2 and -4 for
INT3. Packing uses symmetric ranges [-1,1] and [-3,3], respectively, with
max-absolute scaling and ties-to-even rounding. Zero groups have scale 1.
Nonzero underflow scales floor to the smallest FP32 subnormal. Non-finite
weights and quantized values outside FP32 range are rejected.

INT2 uses one scale per row. INT3 fixes its group size at 64 in the dtype.
Scales are applied before multiplication and accumulation, not after an
unscaled integer reduction. Execution validates every scale and padding bit,
even for a zero-weight route. One activation vector of `4*i` bytes is reused
across token/expert pairs. No expanded matrix or decoded-weight cache is
allocated by either execution kernel. The packer returns caller-sized bytes
and scale arrays; its memory cost is separate from execution scratch.

### Grouped INT4

The persisted contract is `int4-grouped-g64`, not the unscoped
`int4-grouped` name. The standalone API still accepts other group sizes with
explicit caller metadata. Those payloads cannot be persisted under the
fixed-64 manifest dtype or selected through a fixed-64 backend capability.
This prevents equal-size payloads from being reinterpreted with another group
size. Signed nibble codes use adjacent low/high lanes; scales follow all
three projection code arrays. Odd rows require zero high padding nibbles.
There is no claim of GPTQ, AWQ or reference-container byte compatibility.

### MXFP4

The owned MXFP4 layout stores 17 bytes per 32-element row group: one UE8M0
exponent byte followed by 16 adjacent-lane E2M1 nibble bytes. Row tails are
padded with zero codes. E2M1 magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4 and 6;
the sign bit changes their sign. UE8M0 scale is `2^(byte-127)`; byte 255 is
rejected. Non-finite reconstructed FP32 values and nonzero padding are also
rejected. Packing chooses a power-of-two ceiling scale and nearest E2M1
values with ties-to-even code selection. Scale layout and nibble ordering
are explicit OpenAgent metadata, not a promise of compatibility with a
different MXFP4 container, checkpoint or GPU kernel.

### Q8_1

The [public block declaration](https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h)
defines 36 bytes: little-endian binary16 `d`, binary16 `s`, and 32 signed INT8
codes. This implementation decodes each weight as FP32 `d*code`. The `s`
field is finite-validated metadata; it is not used in a floating activation
dot product and is not required to equal a recomputed sum when importing
caller bytes. The independent packer stores half-rounded `d*sum(codes)` and
rejects overflow in either half field. This is not upstream quantizer byte
parity. Row widths must be multiples of 32. Expert projection blocks are
concatenated without headers or padding. There is no GGUF importer.

## Packing, Loading and State

```text
Local SafeTensors
       |
Explicit dtype + fixed scale geometry
       |
Projection conversion -> aligned expert containers + SHA256 regions
       |
Manifest size validation -> leased expert bytes
       |
Scalar / explicit Numba kernel -> routed SwiGLU -> model logits
       |
Snapshot identity: dense values + expert hashes/sizes + dtype + backend
```

```sh
openagent-olmoe pack MODEL --dtype int3-grouped-g64
openagent-olmoe generate MODEL MODEL/openagent-experts.gi3.manifest.json "Hello" --kernel int3-grouped-g64-numba
openagent-olmoe pack MODEL --dtype mxfp4-e2m1-g32-ue8m0
openagent-olmoe generate MODEL MODEL/openagent-experts.mxfp4.manifest.json "Hello" --kernel mxfp4-e2m1-g32-ue8m0-numba
```

Defaults use distinct `.i2`, `.gi3`, `.gi4`, `.mxfp4` and `.q81` container
names. The gateway accepts the compiled backend names through
`OPENAGENT_OLMOE_KERNEL`. Full-model snapshots bind their format and backend
through the execution-identity hash and reject altered identities before
changing KV state. No new Segment precision support is implied here; range
execution and portable Segment snapshots have their own accepted contracts.

## Validation Scope

`test_lowbit_expert.py` checks hand-built streams against an independent
integer/struct oracle, including cross-byte INT3 codes, odd row padding,
partial groups, signed scales, overflow, actual nopython execution and retained
mmap exception cleanup. `test_q8_1_expert.py` covers signed-byte extrema,
both metadata fields, malformed lengths, scratch limits, disabled JIT and
real mmap leases. Neither compiled kernel allocates Numba-managed numeric
arrays inside its execution loop.

`test_extended_packed_olmoe.py` packs a nontrivial two-layer SafeTensors fixture
for all five formats. It checks exact projection bytes and region hashes,
compares full and incremental logits with independently decoded FP32 expert
execution, tests snapshot continuation and identity rejection, and loads the
same model through CLI selection and the gateway. These are deterministic
software correctness checks, not public-checkpoint quality benchmarks or
controlled speed measurements.

The [caller-configured E8 API](ENGINE_E8_CALLER.md) provides independently
validated lattice tables, an explicit orthogonal activation transform and
scalar/compiled CPU execution. It is a standalone owned format, not the
reference E8/IQ3 codec, and is not accepted by persisted model manifests.
Exact IQ3 codebook/bit-layout/rotation compatibility remains unimplemented.
Native GPU kernels for these new formats are also not implemented. Format
support is not a claim that all reference packed categories, execution devices
or model families are covered.
