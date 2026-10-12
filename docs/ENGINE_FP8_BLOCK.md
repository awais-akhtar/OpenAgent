# FP8 Block128 Experts

Independent numeric source: [ONNX E4M3FN equations](https://onnx.ai/onnx/technical/float8.html).
No reference/colibri implementation was used. This is an owned format, not an
assertion of checkpoint byte compatibility.

## Integration Contract

- Dtype: `fp8-e4m3-b128-f32`; backend: `fp8-e4m3-b128-f32-numba`.
- Scalar reference class: `fp8_block_expert.FP8BlockExpertBackend`, backend
  `fp8-e4m3-b128-f32-reference`. Always available without Numba; compiled=False.
- `fp8_block_expert`: `matrix_payload_size(O,I)`, `payload_size(hidden_size,intermediate_size)`,
  `pack_matrix(matrix, *, scale_encoding="f32")`, `pack_expert(gate,up,down, *, scale_encoding="f32")`.
- `native_fp8_block_expert`: `NativeFP8BlockExpertBackend`, `scratch_bytes(h,i)`.
- Registration, manifests, packer orchestration, CLI and gateways belong to the integration layer.
  Neither backend registers itself. The integration layer may register the reference for
  default manifest execution and expose native only through explicit selection.

## Byte Layout

Each projection [O,I] consists of exactly O*I uint8 E4M3FN weight bytes in C
row-major order, immediately followed by little-endian IEEE binary32 scale
bytes in C row-major grid order [ceil(O/128),ceil(I/128)]. There is no header,
padding, alignment gap, or padded edge block. Its size is
O*I + 4*ceil(O/128)*ceil(I/128). Weight [o,i] uses scale [o//128,i//128].
An expert concatenates gate [intermediate,hidden], up [intermediate,hidden],
then down [hidden,intermediate], each with its own trailing scale grid.
The down offset is twice the gate projection size; its own size may differ.
External dimensions and integrity metadata are mandatory. `f32` is the only
accepted scale encoding; UE8M0 and unknown encodings are rejected, never aliased.

## Numeric Policy

E4M3FN has sign bit 7, exponent bits 6:3 with bias 7, fraction bits 2:0.
Exponent zero decodes as (-1)^sign * fraction * 2^-9; otherwise it is
(-1)^sign * (8+fraction) * 2^(exponent-10). Magnitudes 127 are NaN;
all other patterns are finite, including signed zeros and maximum +/-448.
Execution rejects either NaN code or nonfinite scale, even on zero-weight
routes. Finite signed and zero FP32 scales are valid. Scaled weights, products,
reductions, activation and route accumulation use FP32 with fastmath disabled;
nonfinite intermediate/output arithmetic raises ArithmeticError.

Packing requires finite FP32-representable real matrices. Each 128x128 edge
clipped block uses FP32(max(abs(block))/448), floored to the smallest FP32
subnormal when needed; zero blocks use zero. Codes use nearest finite level,
ties to even low significand bit, saturation to +/-448, preserving signed zero.
This is an owned quantization policy, not upstream byte parity.

## Memory And Failures

Numba is optional. Missing or disabled JIT reports unavailable and execution
raises; there is no Python execution fallback or JIT disk cache. The genuine
nopython CPU kernel reads codes and scales directly from a borrowed uint8
view. It allocates no decoded matrices or scale grids. Explicit numerical
scratch is exactly 4*intermediate bytes, checked before acquisition. Inputs,
output, routing metadata, compiler/runtime workspace and borrowed payload are
excluded. One expert lease is held at a time. Store-provided checksums remain
the store's responsibility. Compiler exceptions have borrowed traceback frames
cleared before rethrow outside the lease, preserving exception type.
The separately selected scalar reference reads codes directly and uses
`struct.unpack_from("<f", ...)` for scalar scales. It has the same explicit
4*intermediate activation scratch limit and validates the entire payload before
arithmetic, including zero-weight routes. It neither imports nor invokes Numba,
and is not a fallback or availability alias for the native backend. Both clear
borrowed exception frames and reject nonfinite arithmetic.
Reference/native results are compared with numerical tolerance, not byte parity;
platform transcendental evaluation can differ in the final FP32 rounding.
Packing allocates output-sized bytes plus bounded 128x128 conversion scratch.
No SIMD, performance, quality, or upstream execution-parity claim is made.
