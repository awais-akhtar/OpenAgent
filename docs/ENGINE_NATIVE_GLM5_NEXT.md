# Native GLM5-next Text Subset

## Identity And Scope

This is an independently written, compiled C implementation of a bounded
GLM5-next text architecture, not an alias to GLM-DSA, Qwen, or another family.
It implements KDA, NoPE MLA/DSA with learned KPool compression, MHC residual
streams, dense and routed/shared expert MLPs, final stream collapse, and an
optional genuine LM head. It is not a full-size GLM-5.3-Flash serving engine.
Production checkpoint defaults exceed the deliberately small native bounds
and are rejected. No vision, MTP, tokenizer, generation loop, accelerator
kernels, quantization, or streamed expert execution is implemented here.
Local sharded loading is available through the shared
[native text checkpoint loader](ENGINE_NATIVE_CHECKPOINTS.md).

Public constructor:

```python
from openagent.adaptive_engine.native_glm5_next import NativeGlm5NextTextDecoder

decoder = NativeGlm5NextTextDecoder(
    config, tensors, library="/absolute/path/to/compiled/library",
    max_tokens=256, max_bytes=512*1024*1024,
)
result = decoder.forward([[1, 2, 3]], use_cache=True)
```

The config input is an explicit `glm5_next_text` dictionary or an object whose
`to_dict()` returns that dictionary. `glm5_next` multimodal configs are NOT
unwrapped. Layer, MLP, and indexer schedules must already be explicit. Unknown
config extensions fail closed. The contract/buffer owner
`IndependentGlm5NextTextDecoder(config, tensors, *, max_tokens=256,
max_bytes=536870912)` is not a Python arithmetic fallback; its `forward` raises.

`tensors` must be a text-only state_dict mapping of finite NumPy FP32 arrays
with exact shapes. Canonical names include `embed_tokens.weight`, `norm.weight`,
and `layers.N.*`; the layout is enumerated in `tensor_layout`. Bare names and
the explicit prefixes `model.`, `language_model.`, or `model.language_model.`
are normalized by this module, so the checkpoint loader need not rename them.
Duplicate normalized names, missing tensors, wrong dtype/shape, and extra
tensors (including vision weights) fail closed. The caller must supply the
text tensor subset and the explicit text config, not an entire multimodal map.

`lm_head.weight` is optional and NEVER manufactured from embedding weights.
Without it, `status()['output_scope']` is `text-hidden-only`, and the result
contains `last_hidden_state` and `position`, with no `logits` key. With a
supplied head it is `text-causal-lm-head`, and `logits` is present. A supplied
tied head must equal the embeddings. Head presence does not imply support for
vision, generation, or a full-size checkpoint.

## Supported Bounds

- Resident FP32; one unpadded batch row; nonempty integer token arrays.
- SiLU, NoPE (`qk_rope_head_dim=0`), equal MLA attention/KV head counts,
  no attention biases or dropout. Grouped expert routing is supported: expert
  count must divide evenly into groups, each group must contain at least two
  experts, and selected groups must contain enough experts for token top-k.
- Always-selected incomplete KPool tails. Complete pools are actually ranked
  and pruned when their count exceeds the selection budget.
- A shared DSA layer must immediately follow a full-indexer DSA layer;
  shared chains or intervening KDA layers are rejected.
- Both finite-negative safe forget-gate bounds and the unbounded softplus
  variant are supported. Routed weights may be normalized or unnormalized.
- At most 64 layers, hidden width 256, vocabulary 65536, eight MHC streams,
  64 Sinkhorn iterations, 32 convolution taps or pool positions. Scalar
  dimensions otherwise cap at 256, except index_topk (4096) and configured
  maximum positions (1048576). KDA projection width caps at 1024; expanded
  MLA width caps at 2048. Context capacity is separately bounded to 4096 and
  the configured maximum; the constructor defaults to 256.
- Positive numerical constants are bounded to [1e-12, 1e12]; a non-null safe
  bound is in [-1e12, -1e-12]. Finite input is not a promise of finite execution:
  overflow/non-finite state is rejected without committing the transaction.
- Unsupported masks, position overrides, outer configs, unknown config fields,
  alternate/quantized contracts, shapes, geometry, and selection-boundary ties
  fail closed. Ties wholly inside a selected set do not affect its mask/sum.

## Independent Equations

All matrices use output-row/input-column storage. For each token the C engine
owns a scratch arena and evaluates the entire layer stack before advancing to
the next token. This differs from the oracle's layer-major, chunk-parallel
evaluation. There is no copied chunk algorithm, generated translation, or
runtime call to Transformers, PyTorch, or another native family.

Let X contain R residual streams of width H. MHC computes z=RMS(flatten(X))
without a learned norm weight, then a learned linear map produces pre, post,
and residual logits. Learned scales and bases give
pre=sigmoid(z_pre*s_pre+b_pre)+hc_eps and
post=2*sigmoid(z_post*s_post+b_post). The residual matrix starts with row
softmax plus hc_eps, then column-normalizes once. The remaining iterations
alternate row and column normalization, each with hc_eps added to the divisor.
The sublayer input is sum_r(pre_r*X_r), followed by its learned RMS norm.
The new stream matrix is post[:,None]*sublayer + residual_matrix.T*X.
This operation is repeated for the feed-forward site. Embeddings initialize
every stream identically; final collapse is the unweighted stream mean followed
by learned RMS normalization.

For KDA, separate Q/K/V projections pass through causal depthwise convolution
and SiLU. Q and K are divided by sqrt(sum of squares + 1e-6); Q also has the
1/sqrt(head_dim) scale. Per-head memory S has [key_dim,value_dim] axes. For a
token, the row-wise decay and delta update are:

```text
rate = exp(A_log)
u = F_b(F_a(x)) + dt_bias
log_decay = lower_bound * sigmoid(rate*u)             # safe variant
log_decay = -rate * softplus(u)                        # unbounded variant
S_decayed = diag(exp(log_decay)) * S
delta = sigmoid(B*x) * (v - k.T*S_decayed)
S_new = S_decayed + k*delta.T
core = q.T*S_new
output = O * (weighted_RMS(core) * sigmoid(G_b(G_a(x))))
```

Depthwise history stores the raw projected Q/K/V channels, oldest to newest,
including the current token; convolution coefficients follow that same order.
The C engine uses the same recurrent equation for prefill and continuation.

For NoPE MLA, q=Q_b*RMS(Q_a*x), latent=RMS(KV_a*x), and KV_b expands latent
into each head's key/value pair. The full indexer independently forms
queries from the query residual, LayerNorm-normalized keys from x, per-head
weights from x, and per-channel compression logits from x. For each complete
pool and key channel, compression is a softmax-weighted average of that pool's
keys, using token gates plus learned within-pool positional biases. A pool's
score sums head_weight/sqrt(index_heads) times
ReLU(query_head dot pool_key/sqrt(index_dim)). The largest
index_topk/index_kpool complete pools are expanded into token positions;
the current visible incomplete pool is appended. A shared layer uses the
immediately preceding full layer's selection, not its own key scoring.
Attention uses softmax(q dot k/sqrt(qk_nope_head_dim)) over selected positions.

Dense/shared/routed MLPs clamp the gate only ABOVE swiglu_limit, clamp the up
projection on BOTH sides, then apply SiLU(gate)*up before the down projection.
Routing ranks sigmoid(router_logits)+correction_bias, but combines experts
using the original sigmoid scores, optionally normalized, and multiplied by
routed_scaling_factor. The shared expert MLP is added without a separate gate.
Each group's score is the sum of its two largest corrected expert scores.
The top `topk_group` groups restrict the subsequent expert top-k selection.
Group-boundary ties are rejected, just like expert-boundary ties; ties within
the two largest values of a group do not change its summed score.

## Cache Transactions And Memory

Python owns frozen weight buffers, cache buffers, validation and commit; all
model arithmetic runs in C. NumPy is used for allocation, copies, layout and
validation only, never as a mathematical execution fallback. Compilation is
caller-owned; the module does not auto-compile or silently select a backend.

Each cacheful `forward` copies committed state into a candidate transaction.
Every native token transition mutates only that candidate. Position and cache
are published only after the entire chunk succeeds, including its hidden/head
outputs. Any validation, native status, allocation or later-token exception
discards the candidate. `use_cache=False` always starts from empty state and
leaves committed state untouched. Cache snapshots are detached copies.
`reset` and forward are serialized with a reentrant lock.

KDA caches store convolution history and recurrent memory. DSA caches store
expanded KV and raw indexer keys/gates; complete pools are recomputed at each
token. This is a correctness-oriented resident design, not a compressed-cache
or throughput claim. The budget includes owned weights, committed/candidate
caches, an 8 MiB C workspace and transaction outputs. C validates geometry,
descriptor sizes, schedules, finite buffers, ABI version and final state.
The exported step function is a PRIVATE candidate-state ABI, not an atomic
checkpoint/cache API for external callers.
The private ABI is now version 2, with 26 geometry integers including group
count and selected-group count. Rebuild old libraries; version 1 is rejected
before any step call. Non-finite selection scores, router logits, softmax
inputs, RMS sums and SwiGLU projections are rejected rather than allowing
selection, normalization or clamping to hide overflow. Tensor names must be
strings and at most one explicit supported prefix is stripped.

## Provenance

The installed official Transformers distribution is 5.16.1 under
`D:/openagent/.tmp-native-decoder-oracle`. Config metadata was inspected first;
official Apache-2.0 modeling source was subsequently inspected with user
permission to resolve equations and layout. Neither source was copied or
line-by-line translated; no reference/colibri source was inspected.

- configuration_glm5_next.py SHA256:
  `B62936C9F5D74B09EA511F3211ABDE65F97A9BE1CB3C8F10FDAD7C879850FA72`
- modeling_glm5_next.py SHA256:
  `2092BBB4EFA2A8087B74F4A4DA37635C503FE1DF9AE73F1E6E8342AF8B4B8E8B`
- [Official Transformers model documentation](https://huggingface.co/docs/transformers/main/en/model_doc/glm5_next)
- [Kimi Linear/KDA paper](https://arxiv.org/html/2510.26692v1)
- [Official GLM-5.3-Flash documentation](https://docs.z.ai/guides/vlm/glm-5.3-flash)
- [Official model card](https://huggingface.co/zai-org/GLM-5.3-Flash)

Online main documentation is not the version-pinned numerical contract; the
local version and black-box oracle outputs determine this implementation's
scope. The KDA paper establishes the channel-wise delta-rule foundation;
GLM-specific gates, MHC layout, clamping and pooling are pinned to the local
official contract rather than inferred from component names alone.

## Verification And Numerical Scope

The owned tests compile the actual C file with MSVC /W4 /WX /O2 /fp:precise
on Windows or C99 -Wall -Wextra -Werror -O2 on POSIX. The exact oracle is
Transformers 5.16.1 with PyTorch 2.14.1+cpu. Tests include mixed dense/MoE and
KDA/DSA schedules; prefill, arbitrary chunks and single-token decode; expanded
KV, indexer, convolution and recurrent cache comparisons; multiple pool/head/
stream geometries; genuine-head conditional-generation logits; head absence;
shared DSA; safe/softplus gates; normalized/unnormalized routing; native
bit-exact chunk invariance; detached snapshots/reset; late-chunk rollback;
native tie, non-finite and invalid-geometry rejection; and disabled NumPy math
functions during execution.

Final owned Windows certification: 26 tests passed, zero failures, zero skips.
Compilation and execution were verified with MSVC; the POSIX build recipe is
provided but was not executed on this Windows host.
The seven additional tests cover grouped oracle outputs and all cache types,
group-boundary rollback, invalid group geometry and tensor names, late-token
normalization overflow rollback, SwiGLU overflow rejection, stale ABI rejection,
and grouped execution/chunk invariance with Python, Torch and NumPy fallback
paths disabled. Grouped oracle fixtures use the existing 1e-5 tolerances.

Most native oracle comparisons use atol=rtol=1e-5. The unbounded-softplus,
unnormalized-routing fixture alone uses atol=4e-5, rtol=2e-5 for comparison
against oracle chunk/recurrent paths. This is not an unexplained global
tolerance increase. On the seven-token fixture, seed 91, partition (1,4,2),
the measured absolute maxima are:

| Comparison | Hidden states | Indexer keys/gates |
| --- | ---: | ---: |
| Native versus full oracle | 2.741814e-6 | 4.529953e-6 |
| Native versus chunk oracle | 2.127886e-5 | 2.574921e-5 |
| Native versus single-token oracle | 1.621246e-5 | 1.949072e-5 |
| Full oracle versus chunk oracle | 1.853704e-5 | 2.121925e-5 |
| Chunk oracle versus single-token oracle | 5.066395e-6 | 6.258488e-6 |

Native versus full-oracle projected logits differed by 2.842397e-6. The
largest native-versus-chunk KDA-memory difference across the three linear
layers was 1.341105e-7; the oracle's own full-versus-chunk memory difference
was 1.862645e-7. A black-box hook measured layer-two output RMS normalization
gain up to 284.3632. This supports accumulation-order drift, amplified by
small-core normalization, as the explanation for downstream differences.
The diagnostic still asserts native/full agreement at 1e-5, and all native
chunk boundaries produce bit-identical outputs and cache state. These are
measured fixture bounds, not universal error guarantees over all weights or
sequence lengths. Selection ties are rejected instead of hidden by tolerance.

Run strict certification from D:/openagent:

```powershell
$env:PYTHONPATH='D:/openagent/.tmp-native-decoder-oracle;D:/openagent'
$env:GLM5_NEXT_CERTIFY='1'
$env:GLM5_NEXT_MEASURE='1' # optional reproducible numerical diagnostics
.accelerator-venv/Scripts/python.exe -m unittest discover -s tests -p test_native_glm5_next.py -v
```

Without certification mode, absent Torch/Transformers, an older Transformers
version, or an unavailable compiler cause clean optional skips. With
GLM5_NEXT_CERTIFY=1 those conditions are errors, not certification success.
The base .venv was checked to skip cleanly without the optional oracle.
No dependency installation was needed.

## Ownership

Only independent_glm5_next.py, native_glm5_next.py, native_glm5_next.c,
tests/test_native_glm5_next.py and this document were changed. The existing
optional-dependency/version and strict certification guards were
preserved. No changes to native_checkpoint.py, family_conformance,
exports, packaging, shared inventory, CI, or other dirty-tree files. No changes were pushed.

## Outstanding

This remains batch-one, unpadded, resident FP32 text execution. Tail-disabled
KPool, masks, caller positions, shared-indexer chains, production-size geometry,
vision/MTP, quantization, accelerator kernels and streamed experts remain
unsupported. No other GLM identity is accepted. Only Windows/MSVC CPU execution
was certified here; no POSIX or hardware-accelerator certification is claimed.
