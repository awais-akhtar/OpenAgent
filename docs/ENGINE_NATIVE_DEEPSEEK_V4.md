# Native DeepSeek V4: Bounded Resident FP32 Replay

Implemented original NumPy and C decoders for a deliberately bounded subset.
All native model arithmetic runs in the new C file; there is no Transformers
runtime adapter, identity alias, or Python arithmetic fallback. This is not
production-scale V4 support. The decoder is exported by
`openagent.adaptive_engine` and available through the shared
[native text checkpoint loader](ENGINE_NATIVE_CHECKPOINTS.md), including I64
hash-routing tables.

## Constructor And Identities

```python
from openagent.adaptive_engine.independent_deepseek_v4 import IndependentDeepseekV4Decoder
from openagent.adaptive_engine.native_deepseek_v4 import NativeDeepseekV4Decoder

python_decoder = IndependentDeepseekV4Decoder(
    config, tensors, max_context=128, max_bytes=512*1024*1024)
native_decoder = NativeDeepseekV4Decoder(
    config, tensors, library=compiled_library_path,
    max_context=128, max_bytes=512*1024*1024)
result = native_decoder.forward([[2, 3, 4]])  # logits shape [1, 3, vocab]
```

Config is a mapping or an object with `to_dict()`. Floating tensors must already
be FP32 with exact manifest shapes. No F16/BF16 casting or quantization occurs.
CPU Torch tensors are accepted at construction without importing Torch in either
runtime module. Weights are copied into immutable bytes-backed arrays.

`tie_word_embeddings` accepts only a boolean. When true, the embedding is
required and `lm_head.weight` may be absent, matching official tied checkpoint
serialization. A supplied head must still be finite FP32 with the exact shape
and elementwise identical to the embedding (no tolerance); mismatches reject.
Both weight names then reference the same immutable allocation, and the C
embedding/head descriptors point to that allocation. Resident memory counts it
once. When false (the default), the head is required and copied independently;
equal values do not implicitly enable tying.

Hash routing requires `model.layers.{i}.mlp.gate.tid2eid`, exact **int64**,
shape `[vocab_size, num_experts_per_tok]`. IDs must be in range and unique within
each row. Ordinary routing instead requires the FP32 vector
`model.layers.{i}.mlp.gate.e_score_correction_bias`. Expert shapes are
`[experts, 2*moe_intermediate_size, hidden_size]` and
`[experts, hidden_size, moe_intermediate_size]`. `layer_spec()` defines all
accepted layer names/shapes. Extra, missing, malformed and nonfinite tensors reject.

Native identity: `compiled-c-deepseek-v4-token-replay`.
Python identity: `independent-numpy-deepseek-v4-token-replay`.
Private C ABI version: 2 (activation selector; version-1 libraries reject).
Both report `native_range_execution=False` and
`native_streamed_range_execution=False`. API: `forward`, `reset`, `position`,
`status`; no range execution, generation adapter, or cache import/export API.

## Exact Subset

- Batch one, nonempty integer tokens, unmasked contiguous absolute positions from
  zero. Caller-supplied masks/positions reject, even all-ones masks.
- Sliding, CSA and HCA attention schedules; hash and ordinary MoE schedules.
- One KV head and one shared expert; SiLU or ReLU, sqrt-softplus affinity and normalized
  routing; no attention/MLP biases, attention dropout or router jitter.
- Default unscaled RoPE, consistent partial-rotary metadata, positive even rotary
  width fitting attention and index heads. Heads divide evenly into output groups.
- No MTP (`num_nextn_predict_layers=0`) or quantization metadata. Tied and untied
  embedding/head weights are supported under the explicit constructor contract.
- Router/indexer exact top-k cutoff ties reject during inference rather than
  inventing portable Torch tie ordering. Ties within the selected set are accepted.
- Numerical epsilons, activation limit, routing scale and RoPE theta must remain
  finite and positive after FP32 conversion. Boolean and nonnumeric scalars reject;
  both implementations use the converted values, including at the C ABI boundary.

Hard positive-integer dimension limits:

| Dimension | Ceiling |
| --- | ---: |
| Resident context (`max_context`) | 256 |
| Hidden/query-low-rank/output-low-rank widths | 128 |
| Attention heads/output groups/index heads | 8 |
| Attention/index/rotary head widths | 64 |
| Hyperconnection streams/layers | 4 / 8 |
| Routed experts/experts per token | 32 / 8 |
| Expert intermediate width/vocabulary | 256 / 4096 |
| Window/either compression rate/index top-k | 256 |
| Sinkhorn iterations | 64 |

Actual context also cannot exceed config `max_position_embeddings` (accepted
ceiling 1048576). Flash/Pro real checkpoints exceed these bounds and/or use
unsupported quantization: they are explicitly outside this checkpoint.

## Numerical Contract

Target: installed Transformers **5.16.1 eager FP32**, not mutable main behavior.
The [official architecture documentation](https://huggingface.co/docs/transformers/main/model_doc/deepseek_v4)
and installed Apache-2.0 modeling source were inspected with user authorization
to understand equations. Original scalar loops and a separate NumPy design were
written; no source blocks were copied or mechanically translated. Repository
reference implementations were not read. The linked official report and the
alternative official Pro PDF URL returned 404; report verification is not claimed.

Attention equations, with query position `t`, compression rate `m`, entry `w`:

1. Learned RMS-normalize the low-rank query projection; expand to heads and apply
   unweighted per-head RMS. Learned RMS-normalize the single KV projection, used
   identically as K and V. RMS uses mean squares plus `rms_norm_eps`.
2. Rotate trailing interleaved pairs. Sliding-only uses main theta; CSA/HCA use
   compression theta for both local and compressed branches.
3. HCA pools each disjoint `m`-token window by per-channel softmax of gate plus
   offset bias. CSA pools previous-window A channels with current-window B
   channels; entry zero has no previous contribution. RMS-normalize pooled KV
   and rotate at `w*m`. Incomplete windows produce no entry.
4. Local visibility is `max(0,t+1-window)..t`. Compressed visibility requires
   `w < floor((t+1)/m)`. Branches are concatenated, not deduplicated.
5. CSA index pooling uses the same overlap at index width. Its queries expand
   from the normalized low-rank residual without further query RMS. Scores sum
   weighted ReLU query/key dot products, scaled by inverse square roots of index
   head width and head count. Only ready entries compete for `index_topk`;
   HCA attends all ready entries.
6. Main attention scales dot products by inverse square root of head width.
   Softmax includes the learned per-head sink in its denominator but no sink
   value. Inverse-rotate the output at `-t`; apply grouped low-rank then shared
   output projection.

Installed 5.16.1 CSA exposes a pool plus sparse mask; current main documentation
describes a different gather/mask layout. The original engine evaluates the
verified selected-entry result without depending on either tensor layout.

Hyperconnection equations:

- RMS-normalize flattened streams and project pre/post/combine logits with
  independently learned segment scales and biases. Pre is sigmoid plus `hc_eps`;
  post is twice sigmoid.
- Combine starts as row-softmax plus epsilon; normalize columns first, then do
  `hc_sinkhorn_iters-1` row/column passes. Every denominator adds `hc_eps`.
- Pre collapses the branch input. Output stream `k` is
  `post[k]*branch + sum_j combine[j,k]*old_stream[j]`; combine is not symmetric.
- The final head computes its own sigmoid-plus-epsilon collapse from flattened
  RMS streams, followed by model RMS and language-model projection.

Routing equations:

- Affinity is `sqrt(softplus(logit))`. Correction bias affects selection only.
  Hash selection uses the I64 table but still uses learned affinities for weights.
- Normalize selected affinities with denominator sum plus `1e-20`, multiply by
  `routed_scaling_factor`, sum expert outputs, and add the shared expert output.
- Gate clamps above `swiglu_limit`; up clamps on both sides. Evaluate
  `activation(gate)*up` before down projection, for both routed and shared experts.
  `hidden_act="silu"` is the default; `"relu"` uses the official ReLU registry
  semantics with the same clamps, routing and tensor layout. Other activations
  remain rejected. All native activation arithmetic executes in C.

## State And Memory

Persistent state is a transactional token log, **not an optimized compressed KV
cache**. Each call evaluates the pending full prefix in one stateless C call.
Only successful finite logits cause the wrapper to replace the log. Invalid
inputs, budget/capacity failures, C errors, nonfinite outputs and cutoff ties
leave prior state unchanged. Calls/reset/status are lock-serialized; outputs
contain only new rows. Reset discards history. Replay gets slower as the prefix
grows and is unsuitable for large-context serving.

Construction checks a two-weight-copy peak plus maximum-context workspace before
copying. Calls check resident weights plus conservative workspace before replay.
Workspace includes output copies, logs, NumPy MoE intermediates and fixed scratch.
C allocates less than this ceiling, reserving output/log/stack space externally.
The private C entry point checks geometry, descriptor lengths, finite weights,
hash contents, token range, output length and budget before allocation.
`max_bytes` conservatively bounds owned numerical buffers, not total process RSS,
the interpreter/library, or caller-owned checkpoint storage.

## Verified Checkpoint

Windows CPU strict run: **22 tests passed, no skips**, including NumPy/C/oracle
prefill, token/chunk replay, CSA/HCA/window boundaries, hash/corrected routing,
grouped projection, multiple stream counts/rates/top-k values, clamped experts,
asymmetric mHC component checks, compressor pool/index visibility, numerical
scorer/compressor interventions, manifest/dtype/budget rejects, raw C rejects,
reset and failure rollback. Patching the Python evaluator proves no C fallback.
Additional regressions cover FP32 metadata underflow/overflow, exact V4.1 identity
rejection, malformed top-k counts, nonfinite unselected candidates, and real
router-dot-product overflow in both engines without history commit. C top-k
validates every candidate, not only selected scores; index dot products are
checked before ReLU can hide NaNs. SwiGLU clamps preserve NaN propagation.
Private ABI error 3 becomes `FloatingPointError`; errors 1, 2 and 4 remain
descriptor/geometry failure, memory rejection and ambiguous cutoff respectively.
Tied tests verify actual official parameter identity, `save_pretrained` head
omission, restored tying on official reload, full and cached-chunk oracle parity,
both supplied-equal and omitted-head decoder inputs, shared immutable storage/C
pointers, exact resident budget boundaries and malformed/unequal-head rejects.
Official serialization also converts layer names back to its release layout;
tests use the reloaded model's normalized state-dict names for decoder input.
This verifies the decoder contract, not shared-loader release-name conversion
or tied-checkpoint integration (those files remain owner-managed). The original
C tying arithmetic is unchanged: tying only shares the existing embedding and
final projection descriptors, with no Python math fallback. The subsequent ReLU
extension adds a checked activation selector in ABI v2; stale v1 libraries reject
before geometry or inference. ReLU tests cover hash/corrected routing schedules,
negative and upper-clamped gates, component MoE parity, full/cached-chunk oracle
parity, unchanged source identity and no Python fallback. A SiLU comparison
confirms the new activation is not an identity alias.

On 2026-10-03, softmax routing was investigated but not retained: although the
installed config documents it, 5.16.1 router construction raises `KeyError`
because its `ACT2FN` registry lacks `softmax`. No oracle patch or invented
semantics were used to make it pass. Alternate routing, additional activations
and multi-shared-expert semantics remain unsupported pending exact evidence.
Tolerance is `atol=8e-6, rtol=8e-5`: numerical FP32 parity, not bit-identical
BLAS reduction ordering or whole-family support.

```powershell
$env:PYTHONPATH = "$PWD/.tmp-native-decoder-oracle"
$env:OA_DEEPSEEK_V4_STRICT = "1"
.accelerator-venv/Scripts/python.exe -m unittest discover -s tests -p test_native_deepseek_v4.py -v
```

Tests compile only the new C file: MSVC `/W4 /WX /O2 /fp:precise`, or C99
`-Wall -Wextra -Werror -O2 -ffp-contract=off`. Strict mode makes absent exact
Transformers 5.16.1/Torch/compiler requirements errors. Base `.venv` skips the
two optional oracle classes cleanly; dependency-policy/tie tests still run.
No Linux, sanitizer, accelerator, streamed-range, real quantized-checkpoint or
large-model certification is implied. No commit or push was made.

## V4.1 Exact-Oracle Investigation

Checked on 2026-10-02: installed Transformers 5.16.1 has no `deepseek_v41` or
`deepseek_v41_text` configuration registration or modeling source. The
[official Flash checkpoint config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json)
does declare nested `deepseek_v41_text`; this is an identity declaration, not
an executable oracle. The [upstream implementation PR](https://github.com/huggingface/transformers/pull/48721)
is closed and unmerged at inspection. Its description discusses shared CSA2
state, Engram memory, single-pass mHC and output-visible activation quantization,
so substituting V4 equations would not establish architectural parity.

No trustworthy installed exact V4.1 oracle was available under the requested
verification setup. Both constructors explicitly reject those identities;
no alias or speculative implementation was added. Reference/Colibri code was
not read or copied. Shared conformance, exports and checkpoint loaders were
left to their owner. Outstanding V4 contracts remain optimized compressed KV
state, streamed/range execution, broader RoPE/precision/geometry support and
cross-platform certification; V4.1 additionally requires its own exact oracle.
