# Native Qwen3.5 Vision Vertical Subset

This is a tested **bounded synthetic encoder subset**, not pretrained
Qwen3.5 multimodal checkpoint support. Public exports, checkpoint loaders,
conformance registration, README and CI integration are tracked separately.
No runtime Torch or Transformers imports, dependency installation, JIT build,
or NumPy model arithmetic. The C implementation is original; the locally
installed official Transformers 5.16.1 was inspected and used only as a test
oracle, via `.accelerator-venv` and ignored `.tmp-native-decoder-oracle`.

## Exact Boundary

`NativeQwen35Vision` takes the wrapper config and official state-dict tensor
names, plus explicitly caller-built vision and resident text shared libraries.
Supported wrapper: untied `qwen3_5_moe`, with the existing bounded native
`qwen3_5_moe_text` decoder and text hidden width 24. Vision configuration:

| Field | Required Value |
| --- | --- |
| model_type | qwen3_5_moe_vision |
| depth | integer 0-4 |
| hidden_size / num_heads | 8 / 2 |
| intermediate_size / hidden_act | 16 / gelu_pytorch_tanh |
| in_channels | 3 |
| patch_size / temporal_patch_size | 1 / 1 |
| spatial_merge_size | 2 |
| out_hidden_size | 24 |
| num_position_embeddings | 4 |

Batch one, one initial image placeholder, one image grid `[1,2,2]`, processed
finite FP32 patches of shape `[4,3]`. Input preprocessing is external: this is
not a raw-image loader, resize/normalization pipeline or image processor.
The positional grid exactly matches the learned 2x2 table, so interpolation
reduces to direct row selection. Depths 1-4 execute real vision attention/MLP
blocks; see ENGINE_NATIVE_QWEN35_MULTIMODAL.md for the extended contract.
The merger still executes pre-shuffle LayerNorm, biased projection, exact
erf GELU and output projection in C.

The image produces one merged token. Its temporal, height and width positions
all coincide with its text-sequence index. Thus interleaved mRoPE is exactly
scalar RoPE, and the official cached position delta is zero. A bounded C
position builder supplies those indices; the existing resident text C kernel
performs rotary math. This is **not general three-axis spatial mRoPE**.

C copies the embedding table and replaces only the image placeholder row with
the encoded image. A transactional candidate resident text decoder then
executes prefill and owns the normal hybrid attention/recurrent cache. Cached
calls accept text only. Image ID use without pixels is rejected, preventing
the temporary embedding substitution from leaking into ordinary text input.
Reset clears both active and base caches and restores the original-weight
resident decoder, including the modality embedding row. The next image
prefill starts from original text weights. The untied output head is never
changed. Both original text and vision buffers are pinned using immutable
bytes-backed FP32 arrays; caller tensor mutations and writable-flag changes
cannot alter those buffers. The exposed vision owner collection is a tuple.

Image candidates shallow-copy resident decoder control/bindings and share
the original text DLL, but receive fresh cache state, a lock and an immutable
merged embedding table. No DLL is acquired for candidate creation or reset.
Failure leaves the previously active decoder installed. Each successfully
acquired wrapper-owned library has a handle-only weakref finalizer: release
occurs once the library and its dependent decoder/function references become
unreachable, not at reset or while an externally retained candidate remains
live. This is GC-lifetime cleanup, not a deterministic close API. The existing
text constructor's internal failure behavior remains the responsibility of the
text decoder implementation.

Explicitly rejected: depth outside 0-4, different patch/position/merge
geometry, multiple images, batch greater than one, video tensors or tokens,
quantization, tied embeddings, non-FP32/nonfinite pixels, mismatched modality
types, context overflow, and images introduced after prefill. Padding masks
and caller-defined position IDs are not API parameters. Weight geometry,
names and finiteness are checked; wrapper memory admission reserves source
and candidate ownership and limits each decoder to one third of its budget.
This private family ABI makes no streamed Segment ABI claim.

## Verification

On Windows PowerShell, using the already-installed oracle:

```powershell
$env:PYTHONPATH='D:\openagent\.tmp-native-decoder-oracle;D:\openagent'
& .accelerator-venv/Scripts/python.exe -m unittest tests.test_native_qwen35_vision -v
```

The tests compile both original C sources with MSVC `/W4 /WX /O2 /fp:precise`
(or C99 `-Wall -Wextra -Werror` on Unix). They compare the vision feature,
image/text logits, all exposed hidden states and full-attention probabilities
against `Qwen3_5MoeForConditionalGeneration`, then compare three cached text
steps. T/H/W positions and zero delta are checked against official
`get_rope_index`. Failure cases preserve cache position. Runtime guards forbid
independent NumPy decoder fallback and common NumPy math operations.
Additional regressions cover image/reset/new-image and image/reset/pure-text
oracle comparisons, exact original embedding restoration, immutable weight
pinning despite caller mutation, candidate execution failure, DLL sharing and
last-reference finalizer release.

The oracle test requires exactly Transformers 5.16.1 and actual compiler
availability; a compiler absence skips rather than claiming parity. These
results certify this synthetic geometry only, not a real pretrained model.
