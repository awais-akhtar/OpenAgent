# Exact GLM-MoE-DSA Decoder

`IndependentGlmDsaDecoder` in `independent_glm_dsa.py` is an independently
written NumPy equation baseline, not compiled native execution.
`NativeGlmDsaDecoder` in `native_glm_dsa.py` invokes the separate
`native_glm_dsa.c` implementation for all decoder arithmetic: embedding,
normalization, MLA, rotary transforms, sparse indexing, routing, dense/shared/
routed experts and logits. Production imports no Transformers or reference code.
The caller explicitly supplies the compiled shared library; there is no fallback.

## Scope

Only the exact `glm_moe_dsa` model type and its counted tensor contract are
accepted. This does not alias `glm5_moe`, registry `glm`, or `glm5_next*`.
Required schedules explicitly distinguish dense/sparse MLPs and full/shared
indexers. Supported math includes low-rank Q, asymmetric nonrotary/rotary/value
dimensions, default and static YaRN RoPE, grouped bias-corrected sigmoid routing,
optional top-k normalization, routed scaling, and multiple shared experts.
Indexer selections may be reused by subsequent shared-indexer layers.

Persistent C state is normalized KV latent, rotary key, and full-indexer key.
Expanded per-head K/V is reconstructed in tests only, not stored by the decoder.
C buffers carry element counts; geometry, inputs and weights are checked.
Forward holds a reentrant lock, checks its conservative byte budget before
default mask/position materialization, and commits pending state only after
the logits head succeeds. This is an allocation bound, not measured process RSS.
Weight construction/copying and caller-owned tensors are outside the per-forward
budget. Caller outputs retained across calls are likewise outside that budget.

`native_range_execution=True` denotes compiled resident decoder arithmetic.
`native_streamed_range_execution=False`: no leased-expert/Segment integration,
quantized scales, eviction/reload, or complete native-family parity is claimed.

## YaRN Contract

`rope_parameters` must explicitly specify `rope_type="yarn"`, `rope_theta>1`,
`factor>=1`, and positive integer `original_max_position_embeddings` (at most
1048576). Optional parameters are `beta_fast` (32), `beta_slow` (1), `truncate`
(true), `attention_factor` (null/derived), `mscale` (0), `mscale_all_dim` (0),
and `partial_rotary_factor` (only 1 is admitted). Beta values must be positive
with fast >= slow; magnitude parameters must be finite nonnegative numbers.
Positive numerical parameters are bounded to [1e-12,1e12]. Booleans are not
numbers in this contract. Unknown rotary extensions fail closed. A null factor,
partial rotary dimensions and other scaled RoPE variants are not admitted.
Explicit factors are honored even when they differ from the context ratio.

The independently derived static equations, for rotary width D, base theta,
original context L, scale F and frequency-pair index j, are:

```text
correction(beta) = D * log(L / (2*pi*beta)) / (2*log(theta))
low = max(floor(correction(beta_fast)), 0)
high = min(ceil(correction(beta_slow)), D-1)
# Without truncation, omit floor/ceil. If low == high, add .001 to high.
ramp_j = clamp((j-low)/(high-low), 0, 1)
frequency_j = (1-ramp_j)/theta^(2*j/D) + ramp_j/(F*theta^(2*j/D))
m(F,s) = 1                         if F <= 1
m(F,s) = 1 + .1*s*log(F)            otherwise
amplitude = explicit attention_factor, if supplied
amplitude = m(F,mscale)/m(F,mscale_all_dim), if both magnitudes are nonzero
amplitude = m(F,1), otherwise
MLA score multiplier = m(F,mscale_all_dim)^2
```

Adjacent real/imaginary input pairs are rotated into real-half/imaginary-half
storage using amplitude-scaled cos/sin. Both MLA queries/keys and indexer
queries/keys use this transform. The MLA score multiplier additionally scales
the entire nonrotary-plus-rotary dot product before softmax; it is not applied
to indexer scores. Native C derives frequencies, amplitudes and score scaling
from raw validated parameters. Python does not supply a frequency table or
execute rotary arithmetic for the native decoder. The independent NumPy
baseline separately evaluates these equations for diagnostics.

The counted private C ABI is version 2; rebuild older libraries. A version-1
library is rejected before geometry or layer execution. Cache layout is unchanged.

## Numerical Evidence

Locally compiled with MSVC `/W4 /WX /O2 /fp:precise`. Isolated Transformers
5.16.1 is an independently licensed black-box oracle used only in tests.
Existing dependency environments and pins were not upgraded.

- `tests/test_independent_glm_dsa.py`: five passing tests for the NumPy baseline.
- `tests/test_native_glm_dsa.py`: seven passing compiled tests. They compare
  logits, per-layer hidden states, attention, valid selected sparse key sets,
  reconstructed K/V and indexer caches, full versus chunked decode, padding,
  grouped routing variants, shared experts and greedy generation. Tests also
  reject bad counted head geometry, verify pre-allocation budget checks and
  late-failure state rollback, and prohibit NumPy forward fallback.

YaRN fixtures cover default-derived amplitude, magnitude ratios, explicit
amplitude, whole-score scaling, truncated/untruncated correction ranges,
factor-one and coincident-range boundaries, padding and caller positions beyond
the original context. Full and (3,1,4) chunk executions compare logits, all
hidden states/attention, valid sparse selection sets, expanded K/V and indexer
caches to the exact oracle. Native chunk outputs/caches are bit-identical.
Malformed Python configurations and native YaRN geometry fail closed; head
failure after layer/cache writes leaves committed position, mask and cache
unchanged. Stale ABI and Python rotary/frequency fallback are explicitly rejected.

Existing tolerances were not widened: logits/hidden/attention use atol=4e-6,
rtol=5e-5; native cache comparisons use atol=3e-6 with NumPy's default rtol=1e-7.
Baseline YaRN caches use atol=3e-6, rtol=5e-5, and its frequency comparison uses
atol=1e-8, rtol=2e-7. These are tiny-fixture bounds, not global guarantees.
Strict Windows certification: 12 passed, zero failures, zero skips.

```powershell
$env:PYTHONPATH='D:/openagent/.tmp-native-decoder-oracle;D:/openagent'
$env:GLM_DSA_CERTIFY='1'
.accelerator-venv/Scripts/python.exe -m unittest tests.test_independent_glm_dsa tests.test_native_glm_dsa -v
```

Masked-invalid top-k ties are compared through valid selected sets rather than
requiring an unspecified tie ordering. Routing is independently checked against
the oracle in the baseline tests and against that baseline in compiled variants.
Optional oracle tests skip when the pinned oracle is unavailable. With
`GLM_DSA_CERTIFY=1`, missing dependencies, a different Transformers version or
an unavailable compiler are errors rather than successful certification.

## Sources And Remaining Gaps

Architectural contracts were learned from the official
[GLM-5 report](https://arxiv.org/html/2602.15763v1), the index-score and sparse
selection equations in the [DeepSeek-V3.2 report](https://arxiv.org/html/2512.02556v1),
and the official [Transformers GLM-MoE-DSA documentation](https://huggingface.co/docs/transformers/main/model_doc/glm_moe_dsa).
Reference algorithms may be inspected, but no implementation was copied,
translated, bundled or imported into production.

The YaRN equations and GLM-specific score multiplier were resolved against the
installed official Transformers 5.16.1 `modeling_rope_utils.py` and
`models/glm_moe_dsa/modeling_glm_moe_dsa.py`. No reference/colibri source was
inspected, copied or translated. Production imports neither Transformers nor Torch.

Remaining GLM-DSA work: other scaled RoPE and partial rotary dimensions,
null/inferred YaRN factor, direct non-low-rank Q, quantized
weights, streamed experts, production-size memory/performance, accelerator
execution and vision wrappers. Tiny FP32 success does not establish these.

See [engine implementation boundaries](ENGINE_PARITY.md) for the separately
maintained cross-family inventory. This document describes only the achieved
and remaining `glm_moe_dsa` contracts; it makes no implementation-status claim
about other families. Distinct GLM, Qwen, Kimi, Inkling and DeepSeek identities
retain their own contracts and are not aliases to this decoder. No full-family
completion or complete streamed gate is claimed here.
