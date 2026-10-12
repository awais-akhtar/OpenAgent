# Native Rowwise MXFP4

Original C implementation in `native_mxfp4.c`, explicitly loaded through
`native_mxfp4.py`. No vendor code, Numba, Torch execution fallback, shared
checkpoint loader or automatic compilation. The class is exported by the engine
package and its compiled tests are included in CI. This is a weight decoder for
the K3 format boundary, not full K3 checkpoint admission or a complete expert
runtime. `matvec` evaluates the packed linear weights directly without creating
a dense FP32 weight matrix.

## Verified Contract

[OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
Table 1 and sections 5.3.3/5.4.1 specify E2M1, block size 32, E8M0.
E2M1 magnitude codes 0..7 mean 0, 0.5, 1, 1.5, 2, 3, 4, 6; bit 3 is
sign, including negative zero. E8M0 bytes 0..254 mean 2^(byte-127);
255 is NaN, not zero. OCP does not establish this application's byte order.
[vLLM's upstream format utility](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/mxfp4_utils.py)
independently verifies adjacent even/odd packing as even | (odd << 4).
[K3 publisher config](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)
identifies MXFP4 pack quantization. These are format references only, not
implementation dependencies; no source was copied.

Accepted inputs are separate contiguous U8 arrays `(rows, cols/2)` packed
low nibble first and `(rows, cols/32)` scales, dense row-major. Columns must
be a positive multiple of 32; no partial blocks, swizzles, transposes, NVFP4,
affine INT4, or format guessing. This differs from `packed_expert.py`'s
OpenAgent INT4 payload and is deliberately not registered as that backend.

## Execution And Safety

`NativeMXFP4(library=path, max_bytes=...).decode(packed, scales,
rows=R, cols=C, out=optional_fp32)` calls C directly. Compile as a shared
library using MSVC `/TC /W4 /WX /O2 /fp:strict /LD`, or C99
`cc -Wall -Wextra -Werror -O2 -shared -fPIC native_mxfp4.c -lm`.
Do not enable fast-math or flush subnormals to zero.

ABI v1 lengths and budget are byte counts. Status 1 means invalid geometry,
length, pointer, or host float contract; 2 output/input overlap; 3 NaN scale
or FP32 overflow; 4 budget exceeded. Required budget is
`R*C/2 + R*C/32 + 4*R*C`, covering borrowed inputs and output. C has constant
scratch and no allocations. Python checks the budget before output allocation.
Full validation precedes writes, preserving caller output on errors without
a dense scratch matrix. Valid signed zeros and subnormal results are preserved;
values outside finite FP32 are rejected rather than saturated.

Caller must supply truthful accessible lengths and keep inputs stable during
the call (including while ctypes releases the GIL). C cannot validate pointer
ownership or prevent concurrent mutation; atomicity assumes this contract.
Input/input alias is harmless; output/input overlap is forbidden, even partial.
Output uses memcpy so the byte-oriented ABI does not require aligned pointers.

`matvec(packed, scales, x, rows=R, cols=C, out=...)` accepts an FP32 input vector
of length C and returns R FP32 outputs. It accumulates in double and stages R
results before publishing. Budget is `R*C/2 + R*C/32 + 4*C + 8*R` bytes,
covering packed weights, scales, input, output and row staging. Output must not
overlap any input. Status 5 reports staging allocation failure. This kernel
does not implement routing, activation quantization or shared-expert fusion.

## Verification

`python -m unittest discover -s tests -p test_native_mxfp4.py -v` compiles and
executes the actual C library, failing rather than skipping without a compiler.
Tests cover every nibble, finite scale range, signed zero, row/block indexing,
late failures, exact budgets, overflow geometry, aliases, lengths and canaries.
An optional independent Torch equation oracle checks randomized matrices;
it is not a publisher checkpoint oracle or evidence of full model parity.
