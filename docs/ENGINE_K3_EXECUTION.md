# K3 Execution Contracts

This source implementation has not undergone tests, builds, linting, benchmarks,
or runtime verification. Exact publisher checkpoint support remains unverified.
The implementation scope covers `k3_execution`; shared integration is tracked
separately.

## Compatibility Follow-Up

The source compatibility review covered attention,
residual aggregation, routed/shared experts, positional handling, media merging,
and preprocessing/native resource admission. No validation imports, compilation,
tests, builds or benchmarks were run.

Implemented corrections:

- Media placeholder IDs/counts are checked before raw-image preprocessing;
  grid-derived expansion, cache capacity and position IDs are checked before
  tower execution. Invalid requests do not run the tower first.
- Preprocessing and tower execution use the remaining decoder transaction
  budget, rather than independently spending the entire executor budget.
  The estimate deliberately counts vision weight reservations conservatively;
  it is not an exact allocator/RSS accounting contract.
- Token IDs and explicit position IDs are detached from caller arrays before
  native execution/provider callbacks.
- The local routing fallback rejects nonfinite corrected/group scores and
  scaled routing weights. Native grouped-routing scratch admission uses
  overflow-safe byte arithmetic.

The initial source retrieval attempt was unsuccessful. The subsequent equation
review retrieved `modeling_kimi_linear.py` through approved read-only HTTP access.
Publisher source was inspected without execution, import, or copying into the
implementation.

Source-backed equation corrections at revision
`f831ab66814297da540d832a5235f8e904f29d06`:

- Lines 430-440 expand normalized KV latents and concatenate shared key
  coordinates before attention. `mla_mode="publisher-expanded"` is now the
  default: each latent cache page is expanded into FP32 K/V before native
  softmax, and values are aggregated after that projection. Expanded buffers
  are page-local, not persistent full-context K/V. The byte estimate includes
  this scratch. This removes the compressed mode's projection reassociation,
  not the remaining native/publisher numerical differences.
- Lines 1084-1085 multiply the residual norm gain and query in FP32 before
  scoring normalized residual values. `_residual` now forms that product first
  and passes unit gain to the existing native AttnRes operator, avoiding an
  extra gain-weighted-key rounding boundary. Native FP64 norm/score/softmax
  reductions still differ from the source's FP32 arithmetic.
- Lines 831-837 confirm routed aggregation, latent norm/up projection, then
  shared MLP addition on the original hidden input. Line 995 confirms the
  existing layer-index block boundary. Lines 359 and 403 confirm current MLA
  dimension scaling and `rotary_emb=None`. Those equations remain unchanged.

`mla_mode="compressed"` explicitly retains the previous reassociated execution
policy. Neither mode is publisher numerical certification. Vision source and
FLA kernel numerical contracts were not established by this text-source read;
the numerical/source blockers below remain open. No execution was performed.

## Authoritative Evidence

Publisher revision: `f831ab66814297da540d832a5235f8e904f29d06`.
The implementation is original code from pinned contracts, not imported vendor
code, a bounded hybrid fixture wrapper, or another-model alias.

- [Text source](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py)
- [MoonViT3d/projector source](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_k3.py)
- [Model config](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json)
- [Image processor](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/kimi_k3_vision_processing.py)
- [Media utilities](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/media_utils.py)
- [Preprocessing config](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/preprocessor_config.json)

## Facade And Libraries

```python
model = K3Execution(
    config, tensors, library=kimi_library, max_bytes=budget,
    operators_library=tensor_ops_library,
    attention_library=k3_attention_library,
    mxfp4_library=mxfp4_library, expert_provider=provider,
)
logits = model.forward(token_ids)  # numpy.float32 [N, vocab]
state = model.snapshot()
model.restore(state)
model.reset()
position = model.position
model.close()
```

Library paths are explicit; no JIT, build, remote-code execution or automatic
numerical fallback. Optional library paths default to `library`, permitting one
combined binary. Existing native Kimi matrix/attention/KDA ABI, shared
`NativeTensorOps`, and this directory's `attention.c` symbols are required.
Packed execution additionally requires `NativeMXFP4` ABI. `attention.c` supplies
compressed/page softmax, grouped routing, vision GELU and complex-pair RoPE.
No compiler or runtime has been invoked during this implementation update.

`K3Config` is an explicit equation schema, not permissive vendor config admission.
`tensor_shapes(config)` defines the exact publisher-name inventory. FP32 tensors
must be finite, contiguous and exact-shape. Packed expert tensors are U8
`.weight_packed`/`.weight_scale` pairs, not renamed FP32. Owned arrays are immutable
copies. `vision=K3VisionConfig(...)` adds the exact MoonViT3d/patchmergerv2 schema.
Unknown/extra tensors fail. Config and actual owned tensor bytes contribute to
snapshot identity; external experts additionally bind the provider identity.
Provider identity is its immutability contract, not a full-bank authenticity hash.

Input IDs are one-dimensional integers. 1..4096 is the per-call prefill chunk
limit, not the full context limit. Forward returns all logits; image expansion
changes N. Forward/restore stage state before atomic commit. Detached snapshots
contain all latent pages and KDA state. Reentrant public operations are rejected
before coercion/provider callbacks. Closed forward/reset/snapshot/restore/position
and encode access is rejected. `close()` is idempotent, releases owned CDLL,
KDA/operator/MXFP4 handles, and never closes caller-owned expert providers.
Budgeting estimates arrays/native scratch/leased experts, not total process RSS.

## Decoder Operations

- Explicit MLA/KDA schedule, final global MLA, dense-prefix replacement.
- Native row-tiled larger dense/shared/vocabulary projections.
- Compressed MLA query/KV projections, projection RMS norms, full output gate.
- Cache stores normalized latent plus shared key coordinates in 256-token pages:
  `[tokens,kvrank+rope]`, not expanded per-head K/V. No full-capacity allocation
  at reset. Transaction forks share immutable pages.
- Default `mla_mode="publisher-expanded"` expands each latent page through
  KV projection into FP32 K/V before scoring and value aggregation. This
  preserves the publisher projection graph while retaining latent-only caches.
- With explicit `mla_mode="compressed"`, NoPE query is projected through each head's key expansion transpose into
  latent space. Native online softmax accumulates weighted latents page-by-page;
  value projection happens afterward. Scratch is bounded by page size and
  heads times latent rank. This reassociates floating-point operations.
- Capacity up to 1,048,576 tokens, with allocations growing as used and subject
  to byte admission. Sequential exact-attention equation evaluation is not a
  claim of million-token throughput or production performance.
- Native KDA recurrence, publisher-shaped depthwise taps, lower-bound -5 decay,
  beta sigmoid, full output gate, dimension-based query scaling.
- Native SiTU gate/up saturation for dense, routed and combined shared MLPs.
- Publisher grouped selection: sum top-two corrected scores per group, choose
  groups, mask expert candidates, expert top-k; ORIGINAL sigmoid scores provide
  weights, renormalized for top-k >1 then scaled. Ties fail closed.
- Routed latent down projection, weighted expert aggregation, THEN latent RMS
  norm/up projection; one combined shared MLP in original hidden space.
- Rank-two `(1,hidden)` residual queries, block-boundary prefix capture, native
  AttnRes for attention/MLP/final aggregation.

`publisher-none` preserves the pinned text source's `rotary_emb=None`: its named
rope coordinates are not rotated. Optional `split-half` native rotation uses
explicit token positions/theta and is non-publisher mode. Matrix columns <=65536,
rows <=1048576; element-count bounds are avoided by row tiling. Attention heads
<=128, Q/K width <=512, value <=256; latent rank <=65536. KDA heads <=128,
dimension <=256, kernel <=64. Text norm/sigmoid/SiTU/AttnRes/optional rotation use
the shared dynamic native operators, not the bounded K3 fixture.

## Live Packed Experts

`expert_format="e2m1-low-first-e8m0-g32"` explicitly selects existing
`NativeMXFP4.matvec` rowwise E2M1, low-nibble-first, scale-per32 E8M0 convention.
Native reductions consume packed bytes without dense expansion. Selected experts
use K3 SiTU, not another model's expert activation. Shared/dense/attention/vision
arrays remain FP32. Packed scales 255 fail closed.

Without a provider, tables are owned resident copies. With `expert_provider`,
caller tensors MUST omit all routed expert bank entries. Provider supplies
`identity`, `max_expert_bytes`, exact full-name `tensor_shapes`, and
context-manager `lease(layer,expert)` returning six local U8 tables:
`w1/w3/w2.weight_packed` and `.weight_scale`. Tables must be read-only contiguous.
One selected expert is leased at a time; no bank copy/cache in model ownership.
Lease failures roll back the entire forward. Caller retains provider lifecycle.

`MappedMXFP4Experts(spans,max_expert_bytes=...)` is a concrete on-demand mmap
provider. `PackedTensorSpan(path,offset,shape)` describes absolute payload offsets
including any SafeTensors header. Checkpoint integration supplies parsed spans;
no automatic shard/header or codec admission. Constructor checks inventory and
file extents; selected tables map read-only on demand and unmap after matvecs.
Ordinary file mutation is checked before/after leases. File stamps do not prove
authenticity or detect malicious same-stamp changes. NativeMXFP4 packing is an
explicit numeric contract, not publisher parity inferred from header geometry.

## Vision And Merge

`K3VisionConfig` defaults to bias-free pinned MoonViT3d: hidden1024,
intermediate4096, QKV1536, heads12, layers27, patch14, spatial64x64/temporal4,
merge2x2, text7168. `vision_tensor_shapes` includes patch embed/proj/pos_emb,
encoder norm0/norm1/wqkv/wo/mlp.fc0/fc1, final norm and patchmergerv2
`mm_projector.proj.0/proj.2/post_norm`. No other tower is substituted.

Implemented operations: flattened native patch convolution; learned spatial
position embeddings with half-pixel edge-clamped bilinear interpolation; fixed
temporal sin/cos; native x/y-interleaved complex-pair 2D vision RoPE; noncausal
attention isolated per media; RMS pre-norm attention/MLP residual blocks; native
tanh GELU; final norm; spatial2x2 packing with all-frame temporal mean; bias-free
patchmergerv2 projection, native erf GELU, final projection and RMS post-norm.
Tower RMS epsilon reflects `nn.RMSNorm(eps=None)` for FP32, not text epsilon or
silently inferred BF16 finfo epsilon.

`encode_media(pixel_values,grid_thws)` returns projected per-media sequences,
without text cache mutation. Patches: finite contiguous FP32
`[sum(t*h*w),3,patch,patch]`; grids integer `[media,3]`. Time <=configured4,
height/width <=512, exactly divisible by spatial merger kernels.

`forward(ids,pixel_values=patches,grid_thws=grids,media_token_id=id)` runs tower
and projector, expanding each placeholder into its whole feature sequence.
One media item per placeholder in order; position IDs address EXPANDED tokens.
`forward(ids,images=[...],media_token_id=id)` additionally applies the pinned
still-image processor to PIL or uint8 RGB/RGBA inputs. Exported
`preprocess_images` implements bicubic resize, input/side limits, padding,
.5 mean/std, patch order and after-resize 8px chessboard transparency compositing
(255/180). This optional path requires Pillow; the executor does not install it.
`projected_media` remains a separate one-row-per-placeholder interface, not a
vision encoder. Missing, conflicting or mismatched media fails before commit.

## Exact Gaps And Source Blockers

No full-hybrid oracle is established. Native FP64 accumulation, optional
compressed-MLA reassociation, SiTU/AttnRes rounding, spatial interpolation, temporal pooling,
KDA gate rounding and FLA chunk/recurrent/state precision are not certified
identical to publisher BF16 execution. Exact FLA safe-gate formula/L2 epsilon
need a pinned numerical contract/oracle. MXFP4 nibble/exceptional scales and
accumulation convention need publisher parity; config/header shapes cannot
certify them. `require_publisher_exact=True` fails before execution and
`published_checkpoint_supported=False` remains unconditional.

The pinned Python image processor supports only `image` and explicitly raises
on other media types. Video FPS/timestamp/limits are in config, not implemented
in that executable processor path. Caller-prepared temporal patches are
supported, but no automatic video decoder/sampler is invented. Remaining code
boundaries: batch/padding/packed text masks, training, tied embeddings, alternate
activation/tower/projector families, nonexpert BF16 checkpoint conversion,
automatic checkpoint/tokenizer loading. Packed span/vision weight mapping is
owned by checkpoint integration, not this executor.
