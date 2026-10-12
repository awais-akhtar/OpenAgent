# Native Family Execution

A **resident CPU FP32 native text-decoder path** has been implemented.
Completion of other model families, production certification, quantized formats,
and Segment ABI support is tracked separately.

## Exact Executable Path

`openagent.adaptive_engine.native_qwen35.NativeQwen35TextDecoder` accepts only
`qwen3_5_moe_text`. It loads an explicitly supplied caller-built shared library
from `native_qwen35.c`. It never builds, downloads or falls back at inference time.

Every transformer arithmetic operation in its forward path executes in C:

- Embedding lookup, offset RMSNorm and final vocabulary projection.
- Full GQA attention: per-head offset Q/K RMSNorm, partial split-half RoPE,
  causal/padding masks, softmax, value accumulation and sigmoid output gates.
- Linear layers: masked projections, depthwise causal convolution/history,
  L2-normalized Q/K, grouped value heads, scalar decay/write gates, gated-delta
  state transitions, output RMSNorm/SiLU gating and output projection.
- Router logits, normalized softmax top-k selection, packed gate/up routed banks,
  SiLU expert evaluation, down projections and sigmoid-gated shared experts.
- Residual additions throughout the complete layer schedule.

Python controls buffer validation, allocation, layer dispatch, cache commits and
greedy token selection. NumPy owns buffers; it does not perform transformer math.
The existing independent decoder supplies the exact tensor-contract validator
and greedy control flow only. Its NumPy forward/attention/recurrence is not called.
No reference implementation was translated, copied or imported into production.

```python
from openagent.adaptive_engine.native_qwen35 import NativeQwen35TextDecoder

decoder = NativeQwen35TextDecoder(config_dict, tensor_arrays,
    library="/absolute/path/to/qwen35.dll", max_bytes=256 * 1024 * 1024)
result = decoder.forward([1, 4, 7])
tokens = decoder.generate_ids([1, 4, 7], max_tokens=4)
```

Status reports:

```text
model_type = qwen3_5_moe_text
implementation = c-qwen35-fp32-double-dot
native_range_execution = True
native_streamed_range_execution = False
upstream_abi_compatible = False
```

The native-range claim applies only to these private, counted-buffer, resident
family kernels. It does not enable the shared conformance gate, imply upstream
ABI equivalence, certify streaming experts or authorize other model identities.
The earlier `IndependentMixtralDecoder` and `IndependentQwen35TextDecoder` remain
NumPy implementations with `native_range_execution=False`. The earlier compiled
gated-delta step remains a primitive, not a full decoder.

## Bounds And Failure Behavior

The facade validates all names/shapes and finite FP32-compatible weights before
loading the library. C validates geometry, exact tensor counts, finite buffers,
positions/masks and allocation results. Weights remain pinned/read-only. Each
forward works on private cache copies and commits only after every layer/head
succeeds. Failed input validation, budget checks, descriptor checks and late native
failure leave committed position, mask and caches unchanged. Calls and complete
generation runs on one decoder are serialized with a reentrant lock.

Default budget: 256 MiB, accounting for resident weights, original/new caches,
diagnostics, working activations, logits, input buffers and the maximum layer
scratch allocation before allocating call buffers. This is an explicit allocation
budget, **not measured process RSS** or a production-scale peak-memory guarantee.
Caller-retained previous outputs are not included. C also caps each scanned tensor
and scratch allocation at 67,108,864 float elements.

The compiled subset is intentionally bounded: hidden/head dimensions <= 4096,
head counts <= 128, batch <= 128, convolution kernel <= 64, experts <= 4096,
context <= 65,536, vocabulary <= 1,048,576 and unquantized resident expert banks.
Default partial RoPE and 2D text positions only. Multimodal wrappers, scaled RoPE,
vision/video encoders, quantization, sharded checkpoint loading, disk-backed
experts, eviction/reload and snapshot interchange are not implemented here.

## Independent Numerical Evidence

Eight native tests compile real C under MSVC with `/W4 /WX /O2 /fp:precise` and
compare against separately installed Transformers 5.16.1 `Qwen3_5MoeForCausalLM`.
The additional admission regressions check rejection before default mask and
position allocation, exact budget boundaries, and inclusion of the previous
cached mask. The forward budget includes incoming and committed mask storage,
the old cached mask, bounded binary-mask validation and default position seeds.
Masks accept NumPy arrays or nested numeric lists/tuples. Nested masks convert
directly into admitted UINT8 storage instead of a retained INT64 intermediate;
non-binary, non-real and malformed mask values fail before cache commit.
Fixtures use random tiny weights, never reference/Colibri saved outputs or copied
source. Optional dependency/compiler absence skips explicitly, not as success.

Coverage includes batched padded prefill, chunked and one-token continuation,
explicit positions, all hidden states, logits, full-attention probabilities,
router logits/expert choices/weights, convolution history, recurrent matrices,
KV tensors, greedy IDs, reset, tied embeddings, asymmetric key/value/full-head
widths, convolution size one, a mixed linear/full/linear schedule and an all-full
schedule. Nonzero normalization offsets are tested. Decoder tolerance is
`atol=4e-6, rtol=5e-5`; state tolerances are `3e-6`.

Rejection tests cover exact unsupported identities, missing/shape-corrupt/nonfinite
weights, malformed tokens/masks, corrupted native tensor counts, memory-budget
failure and failure after an earlier C layer has already executed on private
state. No-fallback tests disable the independent NumPy forward and NumPy exponent/
einsum helpers during native execution.

Pure-linear cached scheduling was not oracle-certified: Transformers 5.16.1
`Cache.get_seq_length()` raises when its cache contains no attention layer. The
native equations support that schedule but this is not a numerical acceptance
claim for the unverified variant. Stable low-index router ties are defined in C;
exact Torch tie ordering parity is not established.

```powershell
$env:PYTHONPATH='D:\openagent\.tmp-native-decoder-oracle'
.\.accelerator-venv\Scripts\python.exe -m unittest discover -s tests -p test_native_qwen35.py -v
```

Existing dependency environments and pins are unchanged. The newer oracle lives
only in the existing ignored `.tmp-native-decoder-oracle` directory. Windows C
runtime behavior is tested; Linux/macOS compilation/runtime remain unverified.

## Exact Remaining Inventory

The pinned 5.16.1 mapping inventory was re-probed; mapping presence is not decoder
execution. All identities below remain unsupported by this new C decoder.

| Identity | Exact outstanding native work |
| --- | --- |
| `qwen3_5_moe` | A depth-zero, single-merged-image C fixture exists; see ENGINE_NATIVE_QWEN35_VISION.md. Pretrained vision blocks, general spatial positions, video and checkpoint loading remain unsupported. |
| `glm_moe_dsa` | Independent compressed-latent NumPy and resident C decoders support default/static YaRN RoPE; see ENGINE_NATIVE_GLM_DSA.md. Other rotary contracts, packed streaming and broad public-checkpoint validation remain outstanding. |
| `glm5_next_text` | Bounded compiled KDA/KPool/mHC and dense/MoE execution exists; see ENGINE_NATIVE_GLM5_NEXT.md. Masks, packed streaming and broad checkpoint validation remain outstanding. |
| `glm5_next` | Full visual wrapper remains unsupported; text arithmetic does not establish wrapper ownership. |
| `glm5_moe`, registry `glm` | `glm5_moe` is absent; `glm` maps to older dense GlmForCausalLM. No alias to GLM5 DSA is permitted. |
| `inkling_text` | Resident C text subset exists with convolution/KV state and dense/MoE execution; see ENGINE_NATIVE_INKLING.md. This separate exact identity does not enable the multimodal or legacy identities. |
| `inkling_mm_model`, `inkling` | Multimodal and legacy identity contracts remain unsupported. |
| `kimi_linear` | Bounded resident C token-to-logit subset has pinned fused-schema vendor-oracle tests; see ENGINE_NATIVE_KIMI.md. Complete published-checkpoint compatibility and broader configuration support remain unverified. |
| `kimi_k3` | A bounded C AttnRes aggregation primitive exists; see ENGINE_NATIVE_KIMI_K3.md. Full K3 decoding and an exact runnable oracle remain outstanding; it is not a Kimi Linear alias. |
| `qwen4_exp_text` | Resident NumPy/C linear, QSA and PLE execution exists; see ENGINE_NATIVE_QWEN4.md. Packed streaming, non-default rotary scaling and broad checkpoint validation remain outstanding. |
| `qwen4_exp` | The visual wrapper is not implemented. Text decoder support does not establish multimodal support. |
| `deepseek_v4` | Bounded resident C attention, indexer, low-rank output, hash/corrected routing and hyperconnections exist; see ENGINE_NATIVE_DEEPSEEK_V4.md. Its current state strategy replays token history. Optimized KV, quantization and production-scale validation remain outstanding. |
| `deepseek_v41`, `deepseek_v41_text` | A bounded C Full/Reindex/Reuse publication primitive exists; see ENGINE_NATIVE_DEEPSEEK_V41.md. Complete decoder arithmetic, compression, engrams, quantization and an exact oracle remain outstanding. |
| `mixtral` | Bounded resident C decoder supports fused and split expert banks, GQA, default RoPE and sliding attention; see ENGINE_NATIVE_MIXTRAL.md. Masks, rotary scaling, packed streaming and broader geometry remain unsupported. |

The package-data rules include root and nested C/header sources in wheels and
source archives. Shared Segment/packed-format execution remains a separate
contract from these resident text decoders. Local checkpoint loading is
documented in [Native Text Checkpoints](ENGINE_NATIVE_CHECKPOINTS.md).

## Primary Specifications

- [Official Qwen3.5 MoE architecture documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5_moe)
- [Official checkpoint tensor/config geometry](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json)
- [Published gated-delta state equation](https://arxiv.org/abs/2412.06464)

These sources and independently observed tensor contracts inform the equations.
The separately licensed installed Transformers model is a black-box numerical
oracle only, not bundled production decoder code. Native math is independently
written; no remaining-family completion is inferred from names or metadata.
