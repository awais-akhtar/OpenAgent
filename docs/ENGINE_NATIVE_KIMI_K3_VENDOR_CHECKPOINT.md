# K3 Publisher Checkpoint Investigation

## Status

Inspection only. No publisher checkpoint, quantized or unquantized tiny hybrid,
is admitted for inference. No decoder alias, vendor implementation copy or
inference fallback is provided. The inspection API is exported and its
offline tests are included in the native-engine CI workflow. The existing
`native_kimi_k3_checkpoint.py` equation-fixture loader stays separate.

`inspect_native_kimi_k3_vendor_checkpoint(directory, max_metadata_bytes=...)`
reads local config, optional shard index, and SafeTensors headers only. It checks
duplicate JSON keys/tensors, one-based schedule partitioning, tensor byte extents,
overlap/holes, local file containment, metadata budgets, and file identity changes.
It reports immutable tensor descriptors and typed `StructuralGap` values.
`admitted` and `published_checkpoint_supported` are always false. Malformed
metadata raises `ValueError`; unsupported valid structures return findings.
Tensor finiteness, numerical behavior, quantization decoding, and exact complete
publisher tensor inventory are deliberately NOT certified by this API.
The metadata budget bounds encoded metadata reads, not total Python process RSS.
File stamps detect ordinary mutation, not malicious same-stamp replacement.

## Pinned Evidence

Authoritative repository: [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3).
All investigation reads used revision
`f831ab66814297da540d832a5235f8e904f29d06`, not a floating branch.

| Artifact | SHA-256 |
| --- | --- |
| config.json | 9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213 |
| configuration_kimi_k3.py | 735eb9ebe593e17d231e08e1df7f7be9b5ee0e079f511aa201f9572077b416ae |
| modeling_kimi_linear.py | 9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a |

See pinned [config](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json),
[text source](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py),
and [tensor index](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/model.safetensors.index.json).
The config digest denotes exact original bytes, not canonicalized JSON and not
checkpoint authenticity. Caller-provided tensors cannot be authenticated by it.

## Geometry And Layout

Published config has wrapper `kimi_k3` / `KimiK3ForConditionalGeneration`, nested
`text_config.model_type=kimi_linear`, and a vision tower/projector. Native text
inspection does not silently unwrap or discard these components.

The 93-layer schedule is one-based: MLA at 4,8,...,92,93 (24 layers), KDA at
the remaining 69 layers. Final MLA is not simply every fourth layer.
`first_k_dense_replace=1` makes layer zero a dense MLP. The hybrid fixture now
supports explicit dense layer indices; its default remains all-MoE. This
structural extension does not establish exact publisher schedule parity.

Published hidden=7168, MLA heads=96, query rank=1536, KV rank=512,
nope/rope/value dimensions=128/64/128, vocabulary=163840, AttnRes block=12.
KDA has heads=96, head_dim=128, decay bottleneck=128, convolution kernel=4,
full-rank output gate, and gate lower bound=-5. Latent MoE has latent=3584,
expert intermediate=3072, experts=896, selected=16, shared=2. These exceed
the bounded fixture geometry (hidden<=64, MLA heads<=4, layers<=8,
MLA dimensions<=16, ranks<=64, vocabulary<=256, block<=8;
KDA heads<=8 and dimensions<=32; MoE experts<=64/intermediate<=128).

Unchanged tiny vendor construction confirmed KDA conv shape `(heads*dim,1,4)`
versus fixture `(heads,dim,4)`; `f_a_proj/f_b_proj/dt_bias/b_proj/o_norm/o_proj`
versus fixture `decay_down/decay_up/decay_bias/beta_proj/out_norm/out_proj`.
Names/shapes suggest possible reshape/rename operations, not proven admission.
Vendor residual queries are `(1,hidden)`; hybrid fixture queries are `(hidden,)`.
Dense-only fixture already uses rank-2 queries and is a separate contract.
Vendor experts have distinct `w1` (gate), `w3` (up), `w2` (down); no fused
gate/up tensor was established. Shared experts form one combined MLP with
intermediate width `shared*moe_intermediate`, unlike stacked fixture tensors.
Vendor latent normalization is applied after weighted expert aggregation, before
the latent up projection. Exact semantic mapping still needs whole-model parity.

## Published Quantization

Quantization lives in **text_config.quantization_config**, not only the wrapper.
It is compressed-tensors `mxfp4-pack-quantized`, group_size=32, num_bits=4,
float weights, uint8 scales, status=compressed. Attention, shared MLP, dense MLP,
LM head, vision and projector are excluded by publisher ignore expressions.
Do not interpret every tensor as FP4, or infer stored activation tensors from
the model card's MXFP8 activation description.

Range-only HTTP reads of shard 2's 818696-byte header established these actual
layer-1 expert-0 layouts (no tensor payload was downloaded):

| Tensor suffix | Dtype | Stored shape | Logical linear shape |
| --- | --- | --- | --- |
| w1.weight_packed / w3.weight_packed | U8 | (3072,1792) | (3072,3584) |
| w1.weight_scale / w3.weight_scale | U8 | (3072,112) | one scale per 32 inputs |
| w2.weight_packed | U8 | (3584,1536) | (3584,3072) |
| w2.weight_scale | U8 | (3584,96) | one scale per 32 inputs |

Prefix is `language_model.model.layers.1.block_sparse_moe.experts.0.`.
The same header shows BF16 latent down `(3584,7168)`, norm `(3584,)`, up
`(7168,3584)` and residual query `(1,7168)`. Two 4-bit values per U8 and
32-input scale grouping follow the observed extents/config; nibble ordering,
FP4 codebook, E8M0 exponent interpretation, and exceptional values are not
validated by header shape alone. Native FP32 checkpoint conversion does not
decode this format. An exact codec plus numerical oracle remains required.

## Unchanged Full Hybrid CPU Probe

Explicit probe in the new test module fetches only hash-verified vendor config
and text source into temporary storage; it never stubs FLA, replaces kernels,
rewrites source, or loads published weights. It builds four layers, KDA 1/2/3
and MLA 4, dense first layer, latent MoE, AttnRes, hidden=32, heads=2,
head_dim=16, experts=4, selected=2, shared=2, FP32 CPU. Public
`set_attn_implementation('eager')` selects MLA backend after construction.

Observed environment: Python 3.12, Torch 2.14.1+cpu, Transformers 4.57.6,
einops 0.8.1, fla-core 0.5.2, triton-windows 3.8.0.post29. Import and complete
model construction succeeded. First multi-token forward failed in
`ShortConvolution -> causal_conv1d -> Triton driver` with
`RuntimeError: 0 active drivers ([]). There should only be one.`
No full-hybrid logits or recurrent/cache parity were established. Eager MLA
selection does not make FLA KDA convolution CPU-capable. Dense-only oracle
results must not be presented as full-hybrid or published-checkpoint support.

Run offline tests:
`python -m unittest tests.test_native_kimi_k3_vendor_checkpoint -v`

Run explicit networked probe in an existing Torch/Transformers environment:
`python -m tests.test_native_kimi_k3_vendor_checkpoint --vendor-oracle`
On Windows, `--provision` installs only pinned FLA/einops/Triton into temporary
storage. This is a diagnostic command; a failure is not a passing test or skip.
No deployment dependencies or shared files are modified.
