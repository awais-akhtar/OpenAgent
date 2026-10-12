# V4.1 Native Equation Composition

## Separate Publisher Text Path

`PublisherV41Execution` is a new, provider-bound publisher-shaped text
token-to-logit scheduler in `v41_execution/publisher.py`. It does not change
`V41Config`, relabel the square-projection graph, or claim numerical parity.
Both `inference_supported` and `published_checkpoint_supported` remain false
pending an independently executed oracle and integration acceptance.
No tests, validation imports, compilation, builds or benchmarks were run in
this source implementation update.

Contracts were read directly from the exact pinned
[model architecture](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/model.py)
and [arithmetic kernels](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/kernel.py).
The implementation is original scalar C arithmetic plus Python buffer
transport/scheduling, not a copied reference implementation.

New native operations 16-26 implement publisher-specific learned-scale mHC
(pre/post/comb source ordering, epsilon-offset pre, initial row softmax and
column normalization followed by fixed epsilon-denominator Sinkhorn sweeps),
adjacent trailing-channel YaRN with inverse output rotation, multihead shared
latent K=V attention with zero-value sinks, weighted rectified index scoring,
newest-block-pinned candidate selection, signed-square-root Engram gates,
sqrt-softplus routing and route-weight-before-down-projection clipped SwiGLU.
The private operations and version-2 ABI are preserved. Publisher constructors
require `oa_v41_publisher_contract_version() == 1` in an explicitly supplied
library; an older composition library cannot silently execute these operations.

The scheduler uses the shared `V41PublisherProfile`, low-rank Q,
block-diagonal grouped `wo_a`, delayed per-sublayer input mixing, learned final
FFN pre coefficients, configured KV/index ownership and complete compression
groups. Compressed KV and index keys rotate at the group's **first** position.
Each token gets its own causal local window and current complete compressed
entries; incomplete compression groups persist across calls. Index sources
publish selections to consumers within each token. Final RMSNorm and the
untied vocabulary head produce logits for each incoming token, without rerunning
history. Stable lower-index selection ties are a local deterministic policy,
not a demonstrated match to GPU `topk` ties.

`PublisherV41Execution.from_payload(payload, arithmetic, library=..., max_bytes=...)`
accepts `publisher_checkpoint.v41_adapter.V41ExecutionPayload`.
Contract targets resolve to the payload's exact original source names; no
tensor name guessing or private graph conversion occurs. `arithmetic` must
provide immutable `identity`, `linear(source_name, input)`, `cast(row)` and
`roundtrip(row, encoding)`, plus `engram_rows(layer, token_history)` when needed.
Alternatively the direct constructor accepts a target-name provider with
`tensor`, `linear`, `cast`, `roundtrip` and optional `engram_rows` methods.
All returned numerical buffers are detached and checked for finite values
and exact extents as part of runtime admission, not as a validation run here.

`NativePublisherV41Arithmetic(payload, execution_library=..., mxfp4_library=...,
tokenizer=... or compressed_ids=..., max_bytes=...)` supplies concrete transport
and arithmetic. It consumes `payload.expert_provider` six-array packed leases
through the existing `NativeMXFP4` matvec, applies group-32 FP8 activation
roundtrips before quantized projections, and rounds model boundaries to BF16.
Non-expert FP8 projections use aligned row leases through `payload.fp8_provider`
and the direct E4M3/UE8M0 matvec in `fp8_projection.c`. Supply `fp8_library`
explicitly, or include that ABI in the supplied composition library. Each tile
retains source bytes and tile scales; the kernel allocates output staging, not
a dense FP32 weight matrix. Its scalar native dot products accumulate in FP64,
not the publisher GPU kernel's FP32. Numerical equivalence remains unestablished.
`wo_a` is explicitly cast to the publisher's BF16 grouped representation.

Native operation 26 implements BF16 nearest-even casting, window group-32
FP8/UE8M0, compressed group-16 FP4/E4M3 and index group-32 FP4/UE8M0
roundtrips with the pinned scale floors. The private FP4 codec and block-128
FP8 backend have incompatible contracts, so they are not silently reused.
The new scalar operation has not been compiled or compared to GPU rounding.

Publisher Engram lookup is concrete: compressed tokenizer normalization uses
the pinned normalization sequence through `tokenizers`; explicit compressed
ID maps are also accepted. Hash metadata uses per-layer PCG64 odd multipliers
and non-reused ordered prime buckets (`sympy.nextprime`), with checked offsets.
The new native uint64 hash entry point XORs newest-first lookbacks and returns
bucket rows; `payload.engram_provider.lease_rows` reads only selected rows.
Neither the reference modules nor the private seed/chronological hash are used.
`tokenizers` and `sympy` must already be available for tokenizer-derived maps
and prime metadata; no packages were installed during this implementation update.

State commits only after the entire incoming call produces finite logits;
failed arithmetic leaves facade history and attention caches unchanged.
Providers must be immutable/stateless with respect to supplied token history.
Forward/reset/close are serialized and reentrant mutations are rejected.
Admission estimates include staged and committed packed cache rows and output
allocation, but are not an OS memory quota; provider weights, provider scratch,
Python objects and library allocation overhead require separate budgets.
Window, compressed and index rows are retained as `PackedPublisherRow` byte
buffers with explicit encoding and scale bytes. Attention decodes the selected
rows transiently. Pending incomplete compressor groups remain FP32. Peak memory
and numerical agreement have not been measured. GPU numerical parity,
full model-wide tensor parallelism and performance optimization remain gaps.
Image-aware forward, the image processor and DSpark speculative generation now
exist in the publisher execution modules and CLI. The optional distributed
projection bank is described in [the completion worklist](ENGINE_CODE_COMPLETION.md);
it does not itself provide sharded full-model loading or distributed sessions.

`snapshot()` returns `PublisherV41Snapshot` with immutable byte-backed rows;
`restore()` checks identity, token ranges, source-only global/index inventories,
local lengths and pending complete-group remainders before staging a replacement.
`reset()` and `close()` support the shared fresh-session execution factory.
The concrete arithmetic closes its own native handles but borrows loader
providers. The service factory must close model and arithmetic together.
`publisher_checkpoints.py` supplies a separate bounded disk codec; its integration
and current packed-state format are documented separately from the private
equation checkpoint files. It has not been executed during this implementation update.

`openagent.adaptive_engine.v41_execution.V41Execution` is a bounded,
explicit-library token-to-logit equation graph. It is not a V4 alias and does
not support publisher checkpoints. No Python numerical fallback is used.
This implementation has not been compiled or executed during this implementation update.

## Contract

Use `V41Config(schema='openagent.v41.composed-equations.v1')` and supply exactly
the inventory returned by `tensor_shapes(config)`. Tensors are finite,
contiguous FP32 arrays, matrices `[output,input]`, without implicit transpose,
fusion, padding, tying, bias inference or publisher name translation.
Hidden width is a multiple of 16, from 16 through 16384, including 5120.
Vocabulary is configurable through 1048576, intermediate width through 65536,
context capacity through 1048576, expert count through 4096, layers through 64, and residual streams
through 8. Individual native calls have a 268435456-element admission ceiling;
larger matrices use output-row tiling, with complete input-width dot products
and no split RMS/gating semantics. Inconsistent geometry is rejected before
model allocation. These are
admission ceilings, not a claim that every combination fits the byte budget.
Large widths require retrieved Engram; the resident-table primitive remains
limited to its actual width-256 contract. Every layer requires an
explicit native Engram descriptor with matching streams and width, including
all hash seeds, multipliers, table metadata, scales and projections. `compressed_ids`
is the explicit token-vocabulary-to-Engram-vocabulary mapping.

Supply `libraries` with `execution` and `quant` paths to
trusted independently built libraries. The execution library source is
`v41_execution/kernels.c`; the other libraries use existing standalone native
sources. There is no automatic build or shared API registration.
The composition ABI handshake requires version 2, including the retrieved
hash entry point; old version-1 composition libraries are rejected.
The `engram` library is required only if any layer uses a legacy descriptor.
An optional legacy `mhc` dictionary key is accepted for constructor
compatibility but is not loaded or used. Dynamic-width mHC is implemented in
the composition library: flattened full-stream RMS and affine H prediction,
sigmoid A/C, positive B with fixed Sinkhorn sweeps, full B-X mixing, and
delayed A reduction. It does not tile independent stream graphs or lift the
fixed-width external mHC ABI bounds. Attention scores, index selection and
routing allocate geometry-sized native scratch and report allocation failure.

The graph repeats embedding across residual streams, adds native gated Engram
branch deltas, predicts mHC coefficients, and evaluates attention on the
**previous** coefficients' stream mix. Commit applies current residual mixing
and injection and delays current input coefficients to the next block.
Attention and MoE are separate depth steps. mHC depth resets per token, not
across chunks. The final head has explicit final stream coefficients,
weighted RMSNorm, projection and bias.

Attention combines causal sliding-window local K/V and complete compressed
global entries under one softmax. Encoder Full layers project their own mixed
input; decoder Full projects the final encoder mixed representation. Reuse
keeps same-phase source and selection; Reindex refreshes selection against
that source. Decoder Reindex restricts selection to the Full layer's candidate
blocks. Entry scores and block maxima have stable low-index tie breaking.
Channel-wise compression softmax is over complete ratio blocks only.
Index keys derive from unrotated pooled entries. Local Q/K and global Q use
V4-style interleaved trailing-channel RoPE at token position; pooled global KV uses the
block's last-token position. These are declared equation-fixture conventions,
not claims about an unresolved publisher layout.

`fp4-all-projected-v1` stores local rotated K, local V and rotated pooled global
KV using the existing private E2M1/scale codec, and decodes for attention.
Global KV is both key and value in this explicit schema. Index keys, compression
inputs and compression gates remain FP32. Quantization happens when an entry
is created, never using future-token-dependent scales. This is not a publisher
mixed-channel cache format.
Large cache rows are transported in complete blocks through the existing
codec, in tiles of at most 512 channels. Scales stay per 16 channels; neither
RMS nor attention/mHC normalization is split into independent tiles.

MoE uses V4's declared sqrt-softplus affinity, corrected top-k selection,
uncorrected normalized route weights and explicit route scale. Routed and
shared experts use clipped SwiGLU, with explicit positive activation limit;
one shared expert is always added. Hash routing and group routing are not
silently inferred.

## Retrieved Engram

For each large-width layer, supply a descriptor with `kind='retrieved'`,
`dim`, `branches`, `vocabulary`, `pad`, `epsilon`, `heads`, `key`, `value`,
`query_scale`, `key_scale`, `provider`, and a nonempty immutable `revision`.
An optional `weight_budget` constrains resident Engram projection weights.
Projection shapes are `[branches,dim,memory]`, `[dim,memory]`, and
`[branches,dim]` for both scales; `memory` is the sum of head row widths.
These remain explicit immutable FP32 tensors, not inferred packed weights.

Each head has exactly `order`, `width`, `rows`, `seed`, `multipliers`,
`table_id`, and `row_format`. Orders 2, 3, and 4 must all be represented.
Multipliers are four explicit uint64 values. No full table is loaded.
The C hash contract is seed XOR the uint64-wrapping products of chronological
n-gram IDs and corresponding multipliers, modulo the declared row count.
Missing prefix tokens use the explicit compressed-vocabulary pad ID. The
session retains only the last three compressed IDs for hashing.

The provider implements `fetch_rows(requests)`. Requests are a tuple of
`EngramRowRequest(head, table_id, row, width, row_format)` produced from native
hash results. Return exactly one `RetrievedEngramRow(request, payload)` per
request, in the same order. The provider performs I/O only, never model math.
`row_format='fp32'` requires a contiguous FP32 `[width]` payload.
`row_format='local-fp4-block16-v1'` requires `PackedEngramRow(codes, scales)`:
`width/2` low-nibble-first E2M1 bytes and `width/16` unsigned E4M3 scale bytes.
Width must be divisible by 16 for packed rows; native decoding rejects
unsupported scale codes. This is the private local encoding, not an
interpretation of publisher embedding tables.

All hashing, packed-row decoding, key/value projections, per-branch full-width
RMS normalization, query/key scale multiplication, sigmoid gating and delta
generation run in C. Python detaches and transports selected row payloads.
Provider failure, wrong row identity/extent/encoding, or native arithmetic
failure prevents history commit. A provider must maintain immutable table
contents for its declared revision; that revision and table metadata enter
snapshot identity. The facade cannot certify an external provider's revision
honesty. Selected payloads are not stored in snapshots and are fetched when a
new token requests them. No publisher Engram table provider is supplied by this module.

## Facade And State

```python
model = V41Execution(config, tensors, libraries=libraries,
                     engram=layer_descriptors, compressed_ids=vocabulary_map,
                     max_bytes=64 * 1024**2)
logits = model.forward(token_ids)  # numpy.float32 [incoming_tokens, vocabulary]
saved = model.snapshot()          # V41Snapshot(identity, tokens, cache)
model.restore(saved)
model.reset()
model.close()
```

Only incoming tokens execute on each call. Cache lists and Engram sessions are
staged from the committed immutable state; previous tokens are not rerun.
Only successful execution and finite result allocation commit token history. Native
failure, iterator failure, invalid input and admission failure leave history
unchanged. Snapshots carry immutable token history and a digest covering
config, composition weights, compressed IDs and Engram weights/hash metadata
(including declared retrieved-table revision and identities).
Methods are serialized; reentrant mutation is rejected. Snapshots retain
immutable packed local/global rows, FP32 index and pending compression rows,
and Engram token state. Local caches retain only the sliding window; completed
compression blocks release their pending projection buffers. mHC remains
token-local. Restore checks layer inventories, row extents, numerical values,
prefix positions, and Engram history before committing. Fresh factories can
use this facade with `NativeExecutionService` and its bounded prompt chunks.

`save_v41_checkpoint` and `load_v41_checkpoint` provide explicit local recovery.
Files use bounded JSON metadata, typed row records, model/config identity, and
SHA-256, never pickle. Save fsyncs a temporary file before atomic replacement;
POSIX also fsyncs the containing directory. Load stages and validates the
complete file before restoring. The executor is locked for these operations.
The payload limit excludes filesystem overhead and is not a process RSS quota.

## Packed Experts

`V41PackedExperts` consumes explicit `PackedExpertBinding` records and a
caller-owned `PackedStore`. Each selected expert has six separately hashed
regions: gate/up/down codes and scales. The private contract is low-nibble-first
E2M1 with E8M0 group-32 scales. This is not automatic publisher admission.
Supplying `expert_provider` removes the resident routed FP32 expert banks from
the constructor inventory; shared experts and other projections remain FP32.
An explicit `mxfp4` library is required. Projection runs directly on leased
packed bytes; no full FP32 expert bank is decoded. Leases are released before
the next expert, and the store remains caller-owned.

Native index top-k uses a bounded worst-first heap and stable lower-index ties,
rather than sorting the full history by insertion. Native per-call element
ceilings remain enforced. Large vocabulary output projections tile output rows
while retaining complete dot products. Admission accounts for cache layout,
pending compression, index and attention working buffers, output rows, weights,
and selected experts. These estimates have not been measured and do not impose
an OS-level memory quota.

## Remaining Gaps

Authoritative exact publisher fused layouts, multi-head geometry, rotary
channels, Engram placement/token compression, learned final mixing and
mixed-precision packing remain unresolved. They require separate schemas
backed by authoritative contracts; other schema identifiers are rejected.
This code composes repository equation fixtures and does not establish
publisher numerical equivalence. Existing standalone CSA2 ownership and index
ABIs have width bounds below this graph, so it implements dynamic selection
and ownership directly instead of claiming incompatible ABI support. It uses
the same CED complete-block equation and same-phase ownership contract, not
the existing dense decoder wrapper. Width-5120 execution is implemented for
this schema, not demonstrated by an executed model. Full-model projection,
embedding, head, and shared MoE weights remain resident FP32. Packed routed
experts are optional and use an explicitly bound provider. Memory admission can
therefore reject a full large model despite supporting its hidden width.
No complete publisher runtime adapter, executed tests, or performance claim is
included. Incremental state and local checkpoint recovery are implemented but
remain unexecuted during this implementation update.

Publisher per-layer schedules and checkpoint configuration mapping are also
not supplied: `modes`, encoder split and all tensor shapes remain explicit.
The large-width path does not silently relabel this equation graph as an
exact published decoder.
