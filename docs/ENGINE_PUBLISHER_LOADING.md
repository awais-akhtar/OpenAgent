# Publisher Checkpoint Loading

## Status

Code-first implementation; no tests, imports, compilation, lint, benchmarks or
verification were run during this implementation update. This service converts actual local tensor
payloads, not just metadata findings. It does not register a decoder, alias a
fixture, run remote code, fetch weights, or claim publisher inference support.
All reports and loaded payloads expose `inference_supported=False`.

## Public Facade

Import from `openagent.adaptive_engine.publisher_checkpoint`:

```python
checkpoint = PublisherCheckpoint.open(
    directory, family="kimi_k3", max_metadata_bytes=256 * 1024**2,
    max_tensors=1_000_000, max_shards=4096,
)
report = checkpoint.report
payload = checkpoint.load(
    targets=("k3.text.layers.1.moe.experts.0.gate",),
    max_weight_bytes=128 * 1024**2,
    output_encoding="mxfp4-e2m1-g32-ue8m0",
    max_chunk_bytes=1024**2,
)
for block in checkpoint.stream(
    targets=("k3.text.layers.1.moe.experts.0.gate",),
    output_encoding="float32", max_chunk_bytes=1024**2,
):
    stage(block.target, block.shape, block.offset, block.payload)
```

`PublisherCheckpointService.open` has the same signature. Its `load(directory,
family=..., max_weight_bytes=..., targets=..., output_encoding=...,
max_chunk_bytes=..., max_metadata_bytes=..., max_tensors=..., max_shards=...)`
opens and materializes in one call. No model factory or generation API exists.
Integration with the shared `NativeExecutionService` must preserve exact
support flags. Generic target payloads are not directly model factories; the K3
text-schema adapter below supplies the current equation executor's exact names.

`config` and `tensors` are immutable metadata views. `report.contracts` contains
exact sources, logical shape, conversion encoding, execution target and evidence
URLs. `report.unsupported` contains missing, mismatched and unmapped tensors.
`report.execution_gaps` retains coverage and executor limitations. Default
whole-inventory reads reject any mapping gaps before reading tensor payloads.
An explicit unique target sequence permits a supported subset without removing
other gaps. Unsupported configuration schemes raise typed
`UnsupportedPublisherContract` immediately. Malformed structures raise
`ValueError`; admission budgets raise `MemoryError`.

`ExecutionPayload.tensors` contains real owned, read-only NumPy arrays under the
explicit target names. Config digest is SHA-256 of local original config bytes,
not authentication or a claim of an immutable publisher revision match.

## Exact Implemented Tables

K3 requires outer `kimi_k3` / `KimiK3ForConditionalGeneration`, nested
`kimi_linear`, untied embeddings, explicit one-based hybrid partition with final
MLA, and a dense prefix. Rules derive shapes from configuration and retain full
publisher config rather than constructing fixture config:

- `language_model.model.embed_tokens.weight`, norm, output AttnRes norm/query;
  `language_model.lm_head.weight`.
- `language_model.model.layers.<i>.self_attn`: MLA query/KV low-rank weights,
  norms, output and optional gate; KDA Q/K/V, rank-3 convolution weights,
  `A_log`, `dt_bias`, decay projections, beta, gated norm and output gate/projection.
- Layer input/post-attention norms, `self_attention_res_norm/proj` and
  `mlp_res_norm/proj`. Residual queries retain `(1, hidden)`.
- Dense-prefix `mlp.gate_proj/up_proj/down_proj`.
- `block_sparse_moe.gate.weight/e_score_correction_bias`,
  `routed_expert_down_proj/up_proj/norm`, combined shared MLP, and individual
  `experts.<e>.w1/w3/w2` mapped explicitly to gate/up/down, never stacked or fused.
- Bias-free `vision_tower.patch_embed.proj.weight` and rank-3 spatial
  `pos_emb.weight`, encoder norms/QKV/output/MLPs/final norm, and
  `mm_projector.proj.0/proj.2/post_norm.weight`, retaining actual source names.

The authoritative basis is the publisher's
[text architecture](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py)
and [configuration](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json).
Previously recorded pinned header evidence is in
[the K3 investigation](ENGINE_NATIVE_KIMI_K3_VENDOR_CHECKPOINT.md).
Pinned config and index were read through read-only HTTP after the web tool's
pinned transport failures. Bounded index output established layers 0/1/3 names;
the existing investigation records packed shard-header shapes. The source model
was read on accessible main, without importing or executing it. The implementation
does not claim model-source hash attestation. Quantization ignore patterns are
matched as the exact publisher policy, without regex rewriting or arbitrary
pattern evaluation; JSON lists are frozen into tuples before planning.

V4.1 requires outer `deepseek_v41` / `DeepseekV41ForCausalLM`, nested
`deepseek_v41_text`, explicit compression ratios including MTP accounting,
source-layer inventories and FP8 32x32 UE8M0 / FP4 expert configuration:

- Exact published `embed.weight`, `norm.weight`, `head.weight`.
- `layers.<i>.attn.wq_a/wq_b/wkv/wo_a/wo_b.weight` with `.scale`, query/KV norms,
  attention sink, and layer `attn_norm/ffn_norm`.
- `layers.<i>.ffn.gate.weight/bias/bias_vl`, shared experts,
  and routed `experts.<e>.w1/w3/w2.weight` plus `.scale`.
- Per-sublayer hyper-connection projection, base and three learned scale values:
  `hc_attn_fn/base/scale` and `hc_ffn_fn/base/scale`.
- Compressor `wkv`, RMS norm and, for ratios greater than one, `wgate` only
  at configured KV source layers. Index query and head-weight projections only
  at configured index sources; index key projection and norm only at KV owners.
- Engram embedding weight/row scales, `wkv`, `q_weight` and `k_weight` only at
  configured Engram layers, with a separate configured row count per layer.

Rules use the pinned [converter](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/convert.py),
[inference architecture](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/model.py),
and [pinned index](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/model.safetensors.index.json).
Grouped `wo_a` retains its stored rank-2 block-diagonal matrix shape, not a full
attention output matrix. Other exported HF or tensor-parallel naming profiles
are not automatic aliases. A local index must supply exact matching names and
shapes. Pinned config and index reads established the wrapper architecture and
the published `layers/attn/ffn` profile. A bounded header-only HTTP range read of
shard 3 established layer-0 dense weights/scales, routed expert 0, shared experts,
norms and router biases. No weight payload was read. No complete header survey
across all layers or sharded deployment layout is claimed.

## Conversion And Bounds

F32/F16/BF16 map to finite FP32 in bounded flat chunks. K3 compressed-tensors
MXFP4 requires U8 packed `(rows, columns/2)` and U8 scales
`(rows, columns/32)`. V4.1 expert storage requires I8 packed bytes and
`F8_E8M0` scales with the same row groups. Nibbles are interpreted as adjacent
low/high E2M1 values; E8M0 scales use exponent bias 127. Code 255 and nonfinite
FP32 results are rejected. The K3 packing basis is the official
[MXFP4 compressor](https://github.com/vllm-project/compressed-tensors/blob/main/src/compressed_tensors/compressors/mxfp4/base.py)
and [packing helpers](https://github.com/vllm-project/compressed-tensors/blob/main/src/compressed_tensors/compressors/nvfp4/helpers.py).
The existing public independent MXFP4 numeric functions are reused.

FP8 linear weights use a UE8M0 scale per 32x32 tile. Engram tables instead use
one scale per row and group of 32 columns; they are not interpreted as tiled
linear weights. E4M3 subnormals and finite
top exponent values are decoded; NaNs are rejected. Optional packed output is
17-byte row-group records (scale followed by 16 adjacent-code bytes), compatible
with the local MXFP4 matrix payload format. This repacks stored bytes without
requantizing. It does not concatenate an entire expert or imply K3's Situ
activation is compatible with the existing SwiGLU expert backend.

Safe local opening, duplicate JSON detection and file identity checks reuse
native checkpoint helpers. Header extents, overlaps/holes, all shards and optional
index are checked before payload admission. Only one file descriptor is opened
per range; no mmap, whole shard reads or whole expert stack allocation occurs.
Metadata byte/tensor/shard limits and conversion chunk limits are separate.
Materialization budgets cover output arrays plus conservative conversion buffer
headroom, not Python metadata, allocator overhead or total process RSS.
Streaming blocks retained by callers are outside the service's chunk budget.
Packed weight/scale ranges are read once per bounded chunk. FP8 weight ranges
are also chunked; scale bytes use a chunk-local tile cache. This first pass is
still a Python group conversion path, not a throughput-optimized publisher loader.

Consumers must stage streamed blocks and publish only after iterator exhaustion;
late conversion/file changes can fail after earlier blocks have been yielded.
`load` returns no partial payload on failure. File stamps detect ordinary mutation,
not malicious same-stamp overwrites. Immutable checkpoint directories are required.

## Remaining Code Gaps

K3 vision/projector names and shapes are now mapped. Unexpected convolution bias,
new quantization fields, tensor aliases and different inventory profiles require
new explicit rules; they are not guessed. V4.1 now maps window-only DSpark stages
under `mtp.<stage>`, their separate routed expert count, first-stage main
projection/norm and final-stage norm, Markov embedding/head and confidence head.
Draft token embeddings and output head use the backbone at runtime; no duplicate
checkpoint tensors or aliases are invented. Vision patch/attention weights and
biases, RMS norms, bias-free SwiGLU MLPs, biased aligner and `image_start`,
`image_end`, `image_newline` are mapped from the exact pinned index and declarations.
Unknown hash auxiliaries remain explicit gaps, never inferred or discarded.
No complete V4.1 header survey or executor is established here.

## Selected Engram Rows

`checkpoint.read_embedding_rows(target, row_ids, max_output_bytes=...,
max_rows=4096, round_to_bfloat16=True)` reads only the requested rows of a
mapped row-scaled FP8 table. IDs must be Python integers inside its exact extent;
duplicates retain their order. Scale code 255 and nonfinite converted values
are rejected. The default rounds dequantized values to BF16 using round-to-nearest
ties-to-even, then returns them as a read-only FP32 array. Setting the option to
false retains the FP32 dequantization result. Neither mode quantizes activations
or implements the Engram hash lookup itself.

```text
Exact target and row IDs
          |
Row-count and conversion admission
          |
Selected weight bytes + selected row scales
          |
Dequantize -> optional BF16 rounding -> file-stamp recheck
          |
Owned read-only rows
```

The output budget includes conservative temporary-array headroom, not OS page
cache or process RSS. Rows remain staged until all reads succeed; no full table
is read or allocated. Immutable checkpoint files remain required.

## V4.1 Execution Profile

`v41_publisher_profile(checkpoint.config, capacity=...)` returns an immutable
backbone profile with actual attention/index head widths, low-rank projection
sizes, grouped output geometry, mHC streams, normalization constants, YaRN
parameters and per-layer cache owners. The shared profile also carries
actual routed/selected/shared expert counts, intermediate width, routed scale,
activation limit and validated scoring/routing/normalization policies.
Exported `V41EngramProfile` carries layer IDs and row counts, maximum n-gram
order, heads per order, head width, compressed/hash vocabulary and padding ID;
`projection_input` derives the declared concatenated lookup width.
Capacity must fit the publisher context.
Compressed layers require earlier or same-layer KV and index sources using the
same compression ratio. Candidate filtering starts after its configured source.
Pure window layers use the base rotary frequency without YaRN; compressed layers
use the compressed rotary base and original-context setting.

```text
Publisher config -> explicit tensor contracts -> source ownership checks
                                                   |
                         layer profiles + geometry + precision policies
                                                   |
                              source-name payload handoff (no executor)
```

Window KV uses FP8 E4M3 with UE8M0 group-32 scales; compressed KV uses FP4 E2M1
with E4M3 group-16 scales; index keys and queries use FP4 E2M1 with UE8M0
group-32 scales. These strings describe the publisher contract, not a claim
that the current private equation executor implements all three formats.
The profile remains separate from `V41Config`, and both support flags are false.

Next pass needs offline malformed-header/index/shape tests, BF16/FP8 edge cases,
MXFP4 nibble/scale/repacking vectors, mutation and streaming-budget tests, and
an unchanged numerical publisher oracle before executor admission. None were
executed during this implementation update. Shared exports, API integration and CI are tracked separately.

## K3 Execution Handshake

The loader's generic `k3.text.*` targets are a conversion schema, not what
`K3Execution` consumes. The concrete adapter now bridges that distinction:

```python
prepared = checkpoint.load_k3_execution_payload(
    capacity=4096, max_weight_bytes=weight_budget,
    expert_format="e2m1-low-first-e8m0-g32", max_chunk_bytes=1024**2,
)
# Parent or caller supplies explicit compiled libraries; loader opens none.
model = K3Execution(prepared.config, prepared.tensors,
                    library=library, require_publisher_exact=False,
                    mxfp4_library=mxfp4_library,
                    attention_library=attention_library,
                    operators_library=operators_library)
```

`checkpoint.k3_config(capacity=..., expert_format=..., include_vision=...)` also exposes config
construction without reading tensor payloads. The config explicitly maps hidden,
vocabulary, one-based hybrid schedule to runtime layer types, MLA/KDA geometry,
convolution width, dense prefix/intermediate, latent/intermediate expert widths,
expert count/top-k/shared count, AttnRes block, RMS epsilon, Situ parameters,
`routed_scale`, expert groups and selected groups. Capacity is caller-selected and
bounded by publisher context. Vision remains `None` by default for compatibility;
`include_vision=True` constructs the current worker's `K3VisionConfig` from
explicit publisher geometry and adds the full vision/projector inventory.
Projection epsilon, L2 epsilon and rotary mode/theta are exposed runtime equation
choices, not asserted publisher numerical kernel constants.

`PublisherCheckpointService.load_k3_execution_payload(directory, capacity=...,
max_weight_bytes=..., expert_format=..., ...)` is the one-call service facade.
It accepts the same metadata/chunk budgets and explicit equation policy options.

The adapter compares its complete selected destination table to the executor's
`tensor_shapes(config)` before allocating payload arrays. Changed runtime schemas
raise a typed gap rather than guessing. Float32 mode supplies actual
`language_model.*` runtime names; each packed expert is explicitly decoded to
its corresponding `w1/w3/w2.weight`. Packed mode retains separate U8
`.weight_packed` and `.weight_scale` arrays without requantization. All source
rules must pass metadata admission even though unrelated wrapper tensors remain
unsupported. Native whole-expert row-group interleaving is not passed as a
replacement for the runtime's separate arrays.

`K3ExecutionPayload` contains `family`, typed runtime `config`, source-name
`tensors`, untouched `publisher_config`, original `config_sha256`, contracts and
all remaining gaps. `inference_supported` and `published_checkpoint_supported`
remain false. No executor is constructed or certified by the loader. The caller
can supply a fresh model factory to `NativeExecutionService` using this prepared
payload only under the equation-execution policy, not exact publisher admission.
V4.1 does not coerce this profile into the private equation config. Instead,
`checkpoint.load_v41_execution_payload(capacity=..., max_weight_bytes=...,
max_chunk_bytes=...)` (also available on `PublisherCheckpointService` with a
directory argument) returns `V41ExecutionPayload` with the backbone profile,
untouched publisher config/digest, contracts, gaps and owned read-only FP32 arrays
under exact source weight names. Resident quantized weight/scale pairs become a
logical dequantized weight; no stale scales accompany it. Whole-inventory
admission is strict, including banks deferred to providers. No executor is constructed;
the profile describes the backbone, while draft/vision config remains intact in
`publisher_config`. The default `use_providers=True` excludes row-scaled Engram
embedding tables and routed FP4 expert banks (backbone and MTP) from resident
loading. Explicit `use_providers=False` retains full FP32 materialization and
can exceed laptop budgets. The resident attention/shared-expert/MTP/vision
weights can also exceed `max_weight_bytes`; provider mode is not a guarantee
that the remaining model fits a laptop.

## V4.1 Decoder Handoff Contract

The decoder workstream should consume these explicit file-level contracts,
without treating them as private equation-schema aliases:

```python
prepared = checkpoint.load_v41_execution_payload(
    capacity=4096, max_weight_bytes=resident_budget, use_providers=True,
    max_expert_bytes=64*1024**2, max_chunk_bytes=1024**2,
    max_engram_bytes=4*1024**2, max_engram_rows=4096,
)
with prepared.engram_provider.lease_rows(
        "layers.1.engram.embed.weight", selected_hash_rows) as tables:
    rows = tables["layers.1.engram.embed.weight"]
    # Consume rows within the lease; duplicates retain request ordering.
with prepared.expert_provider.lease(0, selected_expert) as tables:
    codes = tables[f"layers.0.ffn.experts.{selected_expert}.w1.weight"]
    scales = tables[f"layers.0.ffn.experts.{selected_expert}.w1.scale"]
    # Consume all six source-name U8 arrays within the lease.
with prepared.expert_provider.lease(0, draft_expert, namespace="mtp") as tables:
    pass  # Exact mtp.0.ffn.experts.<id>.w1/w3/w2.weight and .scale.
prepared.engram_provider.close()
prepared.expert_provider.close()
```

`V41EngramRowProvider.tensor_shapes` maps exact embedding weight names to logical
full-table shapes, without allocating them. `lease_rows(source, row_ids)` calls
the existing bounded selected-row reader. It yields a source-name mapping of
read-only contiguous FP32 rows, optionally BF16-rounded (default), then clears
that mapping on exit. No hashing or token compression is performed. Decoder
integration must supply the publisher's actual selected table addresses; the
private retrieved-Engram `fetch_rows` interface is not silently substituted.

`V41PackedExpertProvider.tensor_shapes` maps every actual expert weight/scale
name to its stored shape. `format` is `e2m1-low-first-e8m0-g32`.
`lease(layer, expert, namespace="layers")` yields exactly six full source-name
U8 arrays. Stored I8 weights retain their raw adjacent low/high nibble bytes;
E8M0 scales retain raw codes. Neither is dequantized or requantized. The provider
reuses `checkpoint.stream`'s admitted packed repacking and publishes only after
iterator exhaustion. It allocates one selected expert's six arrays, not an
entire expert bank or whole interleaved staging expert. `max_expert_bytes` bounds
those arrays; conversion chunk headroom is separate (allow at least twice
`max_chunk_bytes` in addition), as are resident weights and Engram row budgets.
`max_preparation_bytes` exposes that conservative expert-plus-chunk allowance.
There is no mmap reader, new packed-store format or persisted preparation file.
Every selected expert's extent is admitted against the budget during preparation.

Both providers have metadata/stamp-based `identity`, not content authentication
or publisher attestation; immutable checkpoint directories remain required.
Each rejects concurrent/nested leases and closing during a lease, and checks
file/control stamps before and after use. Arrays are owned bounded allocations
lent for the lease lifetime, not OS-mapped views; clearing mappings cannot
invalidate separately retained NumPy references. Callers must not retain them,
or that retained memory falls outside provider budgets. Keep providers alive
until all consuming models/factories finish. Full contracts remain in the
payload even when `tensors` contains only resident weights. Support flags stay
false. This source implementation update included no tests or validation imports.

The source review covered `inference/model.py`, `inference/vision.py`, `inference/convert.py`,
config and index at revision `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`
via read-only HTTP, without importing or executing publisher code. No new shard
header survey, numerical validation, tests, imports for validation, compilation,
builds or benchmarks were performed. K3 remained unchanged because the review
identified no additional demonstrated loading gap.

## Vision And Mmap Provider Handshake

Read-only header research at pinned K3 revision
`f831ab66814297da540d832a5235f8e904f29d06` established:

| Actual tensor | Dtype | Stored shape |
| --- | --- | --- |
| vision_tower.patch_embed.proj.weight | BF16 | (1024,3,14,14) |
| vision_tower.patch_embed.pos_emb.weight | BF16 | (64,64,1024) |
| encoder.blocks.0/26.wqkv.weight | BF16 | (4608,1024) |
| encoder.blocks.0/26.wo.weight | BF16 | (1024,1536) |
| encoder.blocks.0/26.mlp.fc0/fc1.weight | BF16 | (4096,1024) / (1024,4096) |
| encoder.blocks.0/26.norm0/norm1; final_layernorm | BF16 | (1024,) |
| mm_projector.proj.0.weight | BF16 | (4096,4096) |
| mm_projector.proj.2.weight | BF16 | (7168,4096) |
| mm_projector.post_norm.weight | BF16 | (7168,) |

Encoder rows use the full `vision_tower.encoder.` prefix. Projector headers came
from shard 95; vision headers from shard 96. No weight payload was downloaded.
The 27-layer inventory derives from pinned config and exact index names; local
header admission requires every configured layer to match its explicit shapes.
Metadata policy requires RMSNorm, bias-free projections, divided-fixed spatial
position storage with bilinear interpolation, sd2_tpool merging, and patchmergerv2
with GELU. Temporal positions are computed by the executor, not invented learned
checkpoint tensors. `vision_tower_epsilon` defaults to the FP32 equation policy
`2**-23`; this is not a claim of publisher BF16 RMSNorm equivalence.

The concrete handoff to the provider-aware executor is:

```python
prepared = PublisherCheckpointService.load_k3_execution_payload(
    directory, capacity=4096, max_weight_bytes=non_expert_weight_budget,
    expert_format="e2m1-low-first-e8m0-g32", include_vision=True,
    use_expert_provider=True, max_expert_bytes=selected_expert_budget,
    max_mapped_bytes=aligned_mapping_budget,
)
model = K3Execution(prepared.config, prepared.tensors,
                    library=library, attention_library=attention_library,
                    operators_library=operators_library,
                    mxfp4_library=mxfp4_library,
                    expert_provider=prepared.expert_provider,
                    require_publisher_exact=False)
# Retain provider until every model/factory using it has finished.
model.close()
prepared.expert_provider.close()
```

Preparation in provider mode allocates only non-expert tensors (including vision
when selected). `max_weight_bytes` budgets those FP32 tensors and conversion
buffers, not the expert bank. The complete packed metadata inventory must still
match `K3Execution.tensor_shapes`; no unknown expert name is inferred.

`K3MmapExpertProvider(checkpoint, max_expert_bytes=..., max_mapped_bytes=...)`
implements the actual interface: `identity` string, positive `max_expert_bytes`,
exact full-name `tensor_shapes`, and `lease(layer, expert)` yielding the six
read-only contiguous U8 arrays under `w1/w3/w2.weight_packed/weight_scale`.
It maps only selected tensor ranges, aligned down to OS allocation granularity,
and performs no whole-shard mmap or whole-bank materialization. The aligned-range
budget is separate from the sum of six tensor extents; its default is selected
expert budget plus six alignment units. All experts must fit both budgets before
any lease is admitted. Each lease checks shard stamps before mapping and after
use, checks control metadata stamps, and scans scales for code 255 in bounded
chunks. Identity hashes metadata/stamps, not tensor content or publisher identity.

The provider wraps `k3_execution.packed.MappedMXFP4Experts` using
admitted `PackedTensorSpan` descriptors; no duplicate mmap/packed reader remains.
Only one lease is admitted at once. Nesting/concurrent requests fail explicitly.
Borrowed arrays must not escape the context: the reader closes mappings on exit,
and this wrapper clears the borrowed dictionary before unmapping. Separately
retained array references cannot be protected and must never be accessed after
exit. `close()` rejects active leases and marks the provider closed; there is no
persistent mapped expert cache. Same-stamp hostile mutation,
concurrent truncation, OS page-cache RSS and numerical kernel parity are not
solved by mmap; immutable source directories remain a requirement.

Remaining K3 gaps are numerical BF16/FLA/Situ/routing/vision parity, publisher
processor/token-placeholder integration and any inventory outside the explicit
pinned text/vision profile. The loader preserves wrapper config but does not
perform preprocessing or forward calls. Those behaviors belong to the execution package.
V4.1 backbone mapping now includes compressor/indexer, mHC and Engram tensors,
based on the pinned inference declarations and index. The explicit source-name
handoff does not adapt tensors to the private equation decoder. All exact-support flags
remain false; no tests, compilation, lint, benchmarks or verification ran.

## Packed FP8 Projection Tiles

Provider-backed V4.1 preparation also exposes `fp8_provider`. FP8 projection
weights remain in their source E4M3 byte representation with UE8M0 scales,
instead of being materialized as FP32 matrices. `wo_a.weight` remains resident
because its grouped projection uses the publisher's explicit BF16 conversion.

`V41FP8ProjectionProvider.tile_rows(source)` gives the maximum admitted aligned
tile size. `lease(source, row_start=..., row_count=...)` yields the exact source
weight and scale names as read-only U8 arrays. Row starts must be multiples of
32, and only the final tile can contain a partial row block. The scale array is
`[ceil(tile_rows/32), columns/32]`, with offsets relative to the leased tile.
Unknown sources, incomplete blocks, budget overflow and scale code 255 fail
before arrays are published. File stamps are checked around each lease.

```text
Publisher FP8 matrix + tile-scale file regions
                    |
            aligned row-tile lease
                    |
        native packed projection -> release tile
                    |
            next output-row tile
```

`max_projection_bytes` defaults to 64 MiB and is exposed by both V4.1 loading
facades. It bounds the leased raw weight and scale payload, not process RSS,
retained caller views, numerical scratch or OS file caching. No full-shard read
or full FP32 projection allocation occurs in this provider. Source files must
remain immutable throughout preparation and execution. The arithmetic path and
numerical agreement still require subsequent execution checks.
