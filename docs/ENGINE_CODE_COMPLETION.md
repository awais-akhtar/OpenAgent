# Engine code completion worklist

### Qwen4 device checkpoints and packed experts

The concrete device decoder now admits complete local checkpoint inventories,
preserves exact integer PLE metadata, and can select non-expert tensors while
borrowing a family-specific packed expert provider. Provider geometry and
checkpoint fingerprint are checked; identity and storage/work charges are
included in execution, snapshots and cache transport. Selected expert products
execute on device through the shared bounded format decoder. Service/operator
construction remains to be connected. No tests, imports or builds ran.

### Complete external Qwen4 device cache transport

The Qwen4 device codec now carries exact layer state, lexical integer history,
lexical convolution, full text positions and masks in a versioned checksummed
frame. Restore admits identity, geometry, integer bounds and memory, stages
device banks, checks checksum/EOF and cancellation, then publishes atomically.
Inherited atomic file save/load now dispatch to the complete family codec.
Checkpoint construction, packed expert admission and service exposure remain.
No tests, imports, builds or device execution ran.

### Complete Qwen4 device rollback snapshots

The concrete device owner now issues family-specific owned snapshot receipts
containing every attention/recurrent bank, lexical token/convolution bank, full
text positions and occupancy mask. Immutable host copies are memory admitted;
restore stages all device banks and publishes together after cancellation checks.
Snapshot counts, byte charges, release and foreign/stale receipt rejection are
implemented. External cache transport, checkpoint and serving construction
remain outstanding. No imports, tests, builds or device execution ran.

### Concrete Qwen4 accelerator owner

`TorchQwen4Decoder` now owns exact FP32 model tensors and verified integer PLE
metadata on an explicitly requested CUDA/HIP or Metal device. Forward admits
aggregate resident, old/pending hybrid state, output and layer scratch; preserves
the committed mask and positions; stages all layer and lexical banks; and
publishes only after complete forward success and cancellation checks. Reset and
close include lexical and position ownership. Family snapshots/cache transport,
checkpoint constructor, packed-provider admission and service exposure remain
outstanding; inherited Qwen3.5 recovery is explicitly rejected pending the Qwen4
codec so lexical state cannot silently be lost. No execution or tests ran.

### Whole Qwen4 device forward composition

`qwen4_device_forward` composes embeddings, lexical features, attention and MoE
hyperconnections, recurrent or compressed-indexer attention, routed/shared
experts, final hyperconnection mixing and tiled logits. It admits complete
indexed continuation inventories and returns pending attention/recurrent/PLE
banks without publishing them. Qwen4 output-gate and route normalization policies
are retained. The concrete weight-owning device decoder, complete transport and
application construction remain outstanding. Source edits only; no imports,
tests, builds or hardware execution were run.

### Qwen4 device compressed-indexer sparse attention

`qwen4_device_qsa` implements device indexer Q/K projection and additive norms,
partial scaled RoPE, occupancy-based compression groups, stable block selection,
uncompressed tail inclusion, gated grouped-query attention and output projection.
It stages KV plus raw indexer continuation and computes selected attention one
query at a time with bounded online softmax. Explicit geometry, state/work
admission, finite checks and cancellation precede caller publication. Complete
Qwen4 device decoder ownership/recovery and serving still require composition.
No tests, imports, builds or hardware execution were run.

### Qwen4 device hyperconnections and lexical state

Family-specific device operations now cover hyper-channel RMS normalization,
low-rank mixing, block injection and lexical PLE features. PLE keeps exact
integer n-gram hash lanes, EOS boundaries, embedding projections, signed-root
gating, masked physical slots and dilated convolution continuation on device;
prior banks remain unchanged. Tensor geometry, scratch and cancellation are
admitted explicitly. Shared device delta and MoE operations now accept Qwen4's
sigmoid output-gate and optional top-k normalization policies. Full Qwen4 device
QSA, decoder ownership and serving integration remain under implementation.
No imports, tests, builds or device execution were performed.

### Packed Qwen3.5 device experts and gateway exposure

`TorchQwen35Experts` now admits the exact Qwen3.5 manifest, ordered layer
geometry, hashed borrowed store and persisted format contracts using the shared
bounded packed-device decoder. Qwen3.5 checkpoint loading can omit resident
expert banks after full source inventory admission. Routing executes selected
expert products on device, keeps the shared expert resident, pins provider
identity, and charges provider storage/work across forward, snapshots and cache
transport. Device service factories accept the provider. The native family
operator and existing gateway path now select Qwen3.5 device services explicitly,
including packed manifests without requiring a CPU expert kernel. Source-only
implementation; no tests, imports, builds or device execution ran.

### Qwen3.5 device recovery and service construction

The device decoder now transports complete KV, causal convolution, key-by-value
recurrent banks and full occupancy masks through a versioned checksummed frame.
Restore admits exact execution identity and geometry, stages device banks, and
publishes only after checksum/EOF and cancellation checks. Atomic cache file
helpers are wired. In-memory and local-checkpoint service factories own separate
mutable device decoders with aggregate executor admission and cleanup retry;
the common session adapter accepts both structured and direct logit results.
These connect the existing generation, rollback, service cache and batch owners.
Packed device experts and operator/gateway exposure remain under implementation.
No tests, imports, builds or device operations were run.

### Full Qwen3.5 device decoder composition

`TorchQwen35Decoder` now owns explicit CUDA/HIP or Metal FP32 weights and
composes embeddings, additive layer norms, full attention, gated-delta mixing,
routed/shared experts, final norm and a tiled vocabulary head. Whole-request
publication stages all KV, convolution and recurrent banks; cancellation or
failure preserves the committed continuation. Exact checkpoint inventory,
weight/work/state admission, pinned numerical identity, owned bounded snapshots,
restore, reset and close are implemented. Complete external cache transport,
packed expert streaming and application service integration remain outstanding
for this device family. This is source implementation only; no imports, builds,
tests or device execution were performed.

Qwen3.5 device composition now includes family-specific full attention and MoE
operations. Attention preserves per-head query gating, additive Q/K norm
parameters and partial text RoPE, stages complete KV and uses bounded causal
online softmax. MoE preserves stable top-k selection, normalized routes and the
sigmoid-gated shared SwiGLU expert. Both admit exact device weights and scratch,
check cancellation, and return pending results for whole-decoder publication.
Full device external recovery remains under implementation. No imports,
arithmetic, tests, compilation or builds ran.

### Device Qwen3.5 gated-delta operation

`qwen35_device_delta` now executes family-specific convolution and key-by-value
gated-delta continuation on the requested device: masked mixer inputs, physical
padding slots, causal convolution, normalized/repeated Q/K heads, beta/decay,
recurrent prediction/correction/readout, output norm/gate and projection. Exact
device tensor geometry and state/work bytes are admitted before allocation;
previous banks remain borrowed and unchanged, returned banks are pending until
decoder publication, and cancellation/finite checks precede return. Full Qwen3.5
device decoder ownership and attention/expert composition remain in the active
scope. No imports, device arithmetic, tests, compilation or builds ran.

### Bounded accelerator vocabulary projection

Device Mixtral now tiles final vocabulary projection across hidden rows with
explicit device scratch and host result admission. Each tile executes on device,
validates finite logits at the owned host boundary, and releases its device
result before the next tile. Cancellation checks surround tiles and complete
output publication still precedes KV commit. Attention/head scratch reservations
share a conservative peak budget and execution identity versions both policies.
No projection, transfers, imports, tests, compilation or builds ran.

### Bounded accelerator grouped-query attention

`device_attention` now implements device-resident online-softmax attention in
bounded query/key tiles. Grouped KV heads expand only per key tile. Running maxima,
normalizers and weighted values avoid full sequence-score allocation, preserve
absolute causal/padding/sliding-window masks, and retain zero outputs for wholly
masked rows. Tile/output bytes are admitted before allocation and cancellation
is checked between tiles. Device Mixtral uses this operation with an explicit
scratch reservation and versions its arithmetic identity. Block reassociation
does not claim numerical publisher equivalence. No device calls, imports,
attention execution, tests, compilation or builds ran.

### Window-bounded device Mixtral continuation

Device Mixtral now retains only the declared sliding-window KV suffix. Attention
uses absolute key/query positions across retained history plus pending prefill;
committed masks continue to admit the exact full logical prefix. Owned suffix
clones release full prefill storage instead of retaining oversized views.
Work/cache admission uses retained key extent. Snapshots and durable frames
derive window-specific KV dimensions while preserving absolute logical position
and full mask history; execution identity versions the layout change.
Full-attention configurations retain complete KV. No allocations, inference,
serialization, imports, tests, compilation or builds ran.

### Aggregate accelerator executor ownership

`BudgetedExecutorFactory` now reserves full declared model bytes before creating
each independently mutable executor. It enforces live count/aggregate limits,
rejects factory reuse/reentry, installs ownership before post-construction
validation and releases reservations only after successful close. Failed close
retains both executor and reservation, blocks new construction and supports
cleanup retry. Device Mixtral service factories now use this ledger, including
delayed checkpoint configuration checks under owned cleanup. Explicit aggregate
limits can reduce admission below max-sessions times the per-executor budget.
These are declared model reservations, not allocator RSS certification. No
constructors, imports, allocation, tests, builds or device probes ran.

Device Mixtral now exposes the execution service's cancellation-scope contract.
Cancellation is checked before device dispatch, between layers and selected
experts, before final output and before staged KV publication. Cancelled work
discards pending device continuation; active callbacks cannot reset or close the
owner. Cancellation scopes are thread-local and restore nested prior state.
No cancellation, device calls, imports, tests or builds ran.

### Explicit device Mixtral gateway selection

The native family operator now accepts an explicit accelerator device for exact
Mixtral and constructs the concrete full device service. Native library paths
remain required for the C path and become optional for this device path. Device
packed setup admits the local manifest/store, exact persisted lattice contracts
and selected-expert budgets, then constructs the actual device provider without
substituting a CPU expert kernel. Model residency identity includes device and
artifact admission. Existing gateway templates, streaming, decoding policies,
complete-cache recovery and owned service cleanup remain connected. Other
family accelerator operator paths remain in the full active scope. No device
selection, imports, payload reads, tests, probes, compilation or builds ran.

Device selected-expert staging now also admits the existing signed row INT2 and
grouped-G64 INT3 layouts. Decoding preserves each matrix's packed bit offsets,
cross-byte signed codes, exact trailing row/group scales and zero padding,
checks extent/finite values and reserves owned host matrices before device
products. No decoding, imports, device execution, tests or builds ran.

### Expanded device selected-expert formats

Device selected experts now use an explicit bounded staging decoder for row INT4,
grouped INT4, Q4/Q5/Q8 block formats, MXFP4 E2M1/UE8M0, block FP8, caller-contract
E8 and persisted-grid IQ3 in addition to FP32/FP16/row INT8. Existing owned matrix
decoders supply their exact formats; MXFP4/FP8/E8 staging checks record extents,
scales, padding and finite values, and IQ3 reuses the admitted grid/rotation
codec. Device lattice metadata must match the exact supplied persisted contract.
Only selected expert matrices are staged; expert products stay on the requested
accelerator. Direct low-bit device decoding and the remaining low-bit formats
remain in the active full scope. No payload decoding, imports, device operations,
tests, compilation or builds ran.

### Device Mixtral selected packed experts

`TorchMixtralExperts` now admits exact ordered Mixtral manifests and hashed disk
store regions, borrows the store, reserves selected host/device matrix and row
work bytes, and executes FP32/FP16/row-INT8 packed selected experts on the
decoder's explicit device. Hidden rows, routing scales and expert products stay
on device. One selected expert stages at a time; provider identity changes are
rejected. Wider packed codec coverage remains within the full implementation scope.

Device Mixtral construction, fused/split checkpoint selection, service factories,
snapshot admission and complete recovery now accept this exact borrowed provider
and omit resident expert banks after full source admission. Provider geometry,
whole-model fingerprint, identity and declared storage/work bytes participate in
admission and recovery identity. No imports, payload reads, transfers, inference,
tests, compilation or builds ran. Numerical/platform certification stays false.

### Device Mixtral checkpoint, recovery and service lifecycle

The device Mixtral loader now admits complete fused/split checkpoint inventories,
reserves conversion/source/device bytes, converts split experts to fused banks,
and constructs the actual device decoder. Complete device continuation transport
persists active KV banks and occupancy with exact execution/shape identity,
bounded host/device staging, checksum and EOF. Restore stages replacement device
banks before publishing; callbacks cannot reenter the owner. Atomic stable-file
save/load uses the shared executor recovery lifecycle.

Concrete in-memory and directory service factories now create independent device
executors, enforce prefill/snapshot/logit admission, retain configuration identity
across delayed checkpoint loading, and expose the existing transactional serving
and complete-cache recovery paths. Device source/library numerical policy stays
distinct from native C execution. No checkpoint reads, device transfers, imports,
tests, compilation or builds ran. Packed device experts and broader accelerator
family coverage remain in the full implementation scope.

### Full resident device Mixtral arithmetic

`TorchMixtralDecoder` now provides explicit CUDA/HIP or Metal resident FP32
token-to-logit execution: embeddings, RMS normalization, rotary grouped-query
attention with causal/padding/sliding-window masks, stable top-k routing,
normalized selected SwiGLU experts, residual updates and final norm/head. It
admits exact fused inventories, tied weights, model geometry and source/device/
continuation/work/snapshot bytes. Device KV is staged across all layers and
published only after finite output/state checks. Independent batch masks and
explicit rotary positions preserve continuation admission.

Owned snapshots retain active device KV on the host with identity/owner/handle
checks, retained-byte accounting, bounded device restoration and release.
Reset/close retain explicit ownership. No CPU fallback is selected and no device
or numerical certification is claimed. Selected checkpoint loading, durable
device codecs, packed-expert dispatch and serving registration remain parts of
this accelerator component. No Torch imports, allocations, device calls, tests,
compilation or builds were performed; full engine coding remains active.

### Durable complete native batch files

Native service batches now expose atomic complete-file save/load, recovery into
fresh private session slots and verified-file eviction. The file lifecycle uses
the existing flushed atomic writer and stable regular-file reader. Fresh-slot
recovery admits bounded frame inventory/extents before constructing sessions,
restores the complete batch codec and cleans every created slot on failure.
Eviction holds selected sessions, publishes and reads back a complete cache,
then attempts every slot close; failed cleanup retains the durable artifact and
service ownership. No file operations, recovery, imports, tests or builds ran.
The full engine goal remains active and incomplete.

### Complete native batch cache transport

Native service batches now serialize multiple complete family service frames
inside one bounded checksummed envelope. Selected session locks hold a coherent
capture; temporary files stage immutable frames before invoking the caller's
writer. Recovery checks envelope version, session count, extents, total staged
bytes, checksum and EOF before applying any nested cache. Each nested service
frame still enforces its own model/arithmetic/history identity. Full rollback
frames stage before application; failure attempts restoration for every slot
and quarantines on incomplete recovery. Session mapping is caller-supplied and
ordered, allowing recovery into fresh admitted slots. No serialization, restore,
imports, tests, compilation or builds ran; full engine completion remains active.

### Native service batch ownership and complete-state rollback

`NativeServiceBatch` now owns a concrete native execution service and coordinates
independent admitted sessions. It validates rectangular token inputs and binary
occupancy, removes padding before family execution, reserves result/event and
complete-cache rollback bytes, and locks selected sessions in stable order.
Complete family cache frames stage in bounded temporary files before execution.
Failure attempts restoration of every selected slot and quarantines the batch
if any rollback fails. Successful dispatch returns per-sequence generation
results and occupied indices, with one aggregate committed event. Cleanup stays
owned by the service. This executes independent slots rather than a fused device
batch. No batch calls, imports, tests, compilation or builds ran; the full engine
goal remains active.

### Native operator decoding and continuation controls

The native family operator now exposes explicit seeded temperature/top-k/top-p/
min-p and repetition/frequency/presence penalties through the existing bounded
service sampler. Policy admission precedes state reset/loading and generation;
returned metadata includes policy, finish reason and committed context extent.
Greedy remains the default. Requests can explicitly append to an owned session
or load a complete cache before generation, and atomically save the resulting
complete cache with a required transport byte budget. The operator also has a
direct module entry point with deterministic JSON output and guaranteed owned
service cleanup. No command, sampling, persistence, imports, tests or builds ran;
the full engine goal remains active.

### Durable application continuation catalog

`DurableContinuationStore` now manages explicitly keyed complete service caches
in a dedicated directory with entry/disk/frame admission and expiry. Unique
immutable cache filenames are flushed/read back before atomic catalog records
publish. Records carry model identity, cache checksum/extent and exact token
history receipts. Recovery verifies the durable frame, restores the service's
complete family state without token replay, and checks recovered history before
returning the session. Pruning expires/evicts catalog entries and reclaims orphan
cache files within the owned namespace. The store requires exclusive directory
management within one process; it does not implement multi-process coordination.

Native family operator sessions now expose store creation, save and replacement
recovery. A second service slot admits the restored session before the previous
session closes, so failed recovery retains the original owner. Failed old-session
cleanup remains tracked by the service. Directory/key/budget selection stays
explicit. No persistence operations, tests, imports, compilation, builds or
runtime probes ran; the complete engine goal remains active.

### Native gateway packed-expert configuration

The Mixtral/Qwen3.5/Qwen4 operator now admits an explicit local expert manifest,
compiled kernel selection, cache bytes and work bytes. It validates manifest
family/local storage, hashes all expert containers and lattice contracts into
checkpoint identity, creates the concrete compiled packed runtime with fallback
disabled, and constructs the exact family provider with complete layer mapping.
Public range sessions consume that provider through their native callbacks.

Registered callback owners retain packed runtime/backend resources for the ABI's
process lifetime; gateway service unload closes mutable sessions. Failed setup
closes unregistered resources, while partial registration retains the live owner
and can retry the missing edge registration without registering its segment
twice. Residency keys now include manifest/container/contract stamps so changed
packed artifacts trigger re-admission. No operator, packing, inference, imports,
tests, compilation, builds or hardware checks ran. The full goal remains active.

### Application gateway for native Mixtral and Qwen text families

The native publisher gateway now admits explicit `mixtral`,
`qwen3_5_moe_text` and `qwen4_exp_text` configurations and selects a concrete
local family operator. Existing template rendering, cancellation, bounded text
streaming, model identity checks and residency/unload policies apply to these
families. The operator admits explicit library/model/tokenizer paths, context,
native/staging/transfer/snapshot budgets, contiguous partitions and token inputs.
It hashes admitted checkpoint/library artifacts, rejects changing files and
escaping index shards, and constructs the actual family-owned native service.

Public registry owners are reused by complete configuration/artifact identity
and remain pinned as required by the ABI. Gateway sessions own their service
and close it on unload; requests reset complete state and use transactional
generation, stop IDs and native cancellation. Generation events are translated
to the gateway token stream. Family selection does not relabel another model or
enable numerical/platform certification. This operator connects resident text
execution; packed provider setup and multimodal family routing remain within
the active full engine goal. No CLI execution, imports, checkpoint reads, tests,
compilation, builds or runtime probes ran.

### Qwen4 packed selected-expert integration

Qwen4 now has an additive native selected-expert MoE ABI while retaining the
existing resident signature. The family kernel computes routing/top-k and its
optional route normalization, validates callback results, applies route scales,
and executes the unchanged shared-expert path. Native decoder/range construction
admits exact provider family/fingerprint/geometry, omits resident expert banks,
pins provider identity and reserves storage/work bytes. Expert callback failures
leave pending decoder continuation unpublished; callback reentry is rejected.

`Qwen4StreamedExperts` uses the bounded compiled selected-expert runtime with
fallback disabled. `pack_qwen4_experts` emits explicit fused expert slices with
hashes, family fingerprint, metadata/file guards and the existing admitted codec
contracts, including E8/IQ3. Full-inventory checkpoint admission precedes selected
range payload omission. Public sessions borrow the provider and preserve global
layer mapping; complete recovery identity and transport admission include the
provider. No conversions, imports, tests, compilation, builds, native execution
or hardware checks ran. The full engine implementation goal remains active.

### Qwen3.5 selected packed experts

The native Qwen3.5 layer now has an additive versioned selected-expert ABI.
Resident entry-point signatures remain unchanged. The streamed path rejects
resident expert banks, computes normalization/router/top-k/shared expert math
in the family kernel, requests unweighted selected SwiGLU vectors, validates
their outputs, and applies native routing scales. Callback failure discards
pending continuation. Decoder/range ownership checks exact provider family,
whole-model fingerprint and expert geometry, pins provider identity, reserves
its storage/work budget, and denies callback reentry during native calls.

`Qwen35StreamedExperts` borrows the existing bounded compiled packed-expert
runtime without fallback. `pack_qwen35_experts` converts explicit fused expert
slices with metadata/file guards, byte admission, hashes and existing explicit
codec contracts including E8/IQ3. Selected range loading admits the entire source
inventory before omitting expert payloads. Public adapter sessions borrow the
same provider, preserve original layer mappings, and include provider identity
in continuation recovery. This addition does not certify numerical/backend
equivalence. No packing, imports, tests, builds, native calls or hardware probes
were performed; full engine completion remains active.

### Qwen3.5 public native adapter and service composition

`NativeQwen35PublicAdapters` now registers the family-owned native range and
embedding/final-normalization/head implementations through public Segment/Edge
layouts. Family-specific activation/numeric identities, native range and complete
snapshot capabilities, row/context/activation capacities, memory budgets and
independent engine/session inventories are admitted explicitly. Session creation
retains original layer indices and complete native state. Engine cleanup requires
its sessions to close, with failed cleanup retained by the shared registry owner.

Public Qwen3.5 snapshot callbacks now use the complete checksummed native
KV/recurrent/mask format with bounded exact-count framing. The registered owner
constructs the native execution service from checkpoint identity and contiguous
family ranges, preserving existing transactional prefill/generation, cancellation,
cache persistence and serving cleanup. Optional local tokenizer callbacks use the
shared edge ABI. These are resident FP32 native adapters; streamed quantization
and broader accelerator scope remain in the full implementation scope. No tests, builds,
compilation, validation imports, checkpoint loading or runtime probes ran.

### Qwen3.5 selected native ranges and continuation ownership

`NativeQwen35Range` now owns only an exact original-index layer interval. Its
checkpoint loader admits the full Qwen3.5 source inventory before selecting
payloads and reserves selected conversion/ownership bytes. Native execution
accepts hidden rows, retains full-attention KV or linear recurrent/convolution
banks per selected layer, preserves mask and position admission, and commits
after all selected native kernels succeed. Global embeddings/head are excluded.
`NativeQwen35Edge` separately loads and owns the exact global tensors and
executes native embedding/final normalization/head with staging budgets.

Native Qwen3.5 owners now have pinned config/weight/library identities, explicit
close and library release, complete checksummed continuation serialization,
bounded array transport, and atomic stable-file save/load. Recovery persists
active KV prefixes, complete recurrent/convolution state and committed masks,
reconstructs native-capacity KV banks, checks identity/geometry/finite values,
checksum and EOF, and publishes all banks together. Original layer indices
remain in the schema; callback reentry is denied during transport. Public
Qwen3.5 adapter registration and streamed expert/accelerator extensions remain
within the full implementation scope. No tests, validation imports, compilation, builds,
checkpoint reads or runtime/hardware checks ran.

### Qwen4 family-owned native Segment/Edge composition

`NativeQwen4Range` now loads and owns only its selected original-index layer
interval after full source name/shape admission. The shared Qwen4 contract
derives shapes and PLE metadata without allocating model weights. Range
execution consumes all hyper-connection channels plus token IDs for PLE,
executes actual native linear/QSA/PLE/hyper/MoE primitives, and emits residual
hyper channels before the final mixer/head. It retains transactional recurrent,
convolution, sparse KV/indexer, PLE, mask and position state. Existing full
decoder construction and token-to-logit execution retain their default path.

`NativeQwen4Edge` owns only embeddings, the final hyper mixer and head, with
full-inventory admission followed by selected global reads. Embedding produces
the family hyper-channel activation width; final logits apply the native final
mixer and head with explicit row/staging bounds. Selected-layer recovery extends
the complete Qwen4 codec and uses atomic stable-file save/load for direct range
owners. Original layer indices remain encoded in continuation schemas.

`NativeQwen4PublicAdapters` registers family-specific activation/numeric classes,
advertises token-side input, checks range/session capacity and row budgets,
retains callback/resource cleanup ownership, and connects complete Qwen4 cache
frames to public transfer callbacks. Registered owners construct the existing
native execution service from explicit contiguous partitions and checkpoint
identity. This resident FP32 component does not close all quantized/accelerator
family scope. No tests, imports, compilation, builds, checkpoint reads, inference
or hardware probes were run; the full engine goal remains active.

### Mixtral public Segment/Edge and durable range recovery

`NativeMixtralPublicAdapters` now registers actual Mixtral hidden-range and
edge kernels through the public ABI. It admits explicit model/library paths,
engine/session memory, row/context and resource inventories, pins callbacks,
retains independent session continuations, and retains failed cleanup for
retry. Engine destruction requires sessions to close first. Dense selected
weights and caller-owned packed expert providers use the same Mixtral range
path. Edge embedding, final normalization, logits, greedy selection and optional
local tokenizer callbacks share the existing public ownership machinery while
advertising Mixtral-specific activation/numeric identities.

Native Mixtral ranges now expose bounded checksummed complete KV/mask transport
and atomic stable-file save/load. Recovery checks exact weight/provider/range,
capacity, batch and payload identity, finite cache state, mask semantics,
checksum and EOF before publishing. Callback reentry is denied during transport.
Public snapshot transfer callbacks use the same range codec with declared frame
extent. Cancellation is checked at the native operation boundary; native calls
retain atomic continuation publication. No imports, tests, compilation, builds,
checkpoint reads or runtime probes were performed. This addition closes this
Mixtral adapter code component; the full engine worklist remains active.

Registered Mixtral owners also construct the complete native execution service
from admitted family configuration, a checkpoint SHA-256 identity, explicit
contiguous layer partition size, context and native/transfer/snapshot budgets.
The composition reuses native edge/range capability agreement, transactional
prefill/generation, snapshot ownership, complete cache persistence and service
cleanup. The packed expert provider remains borrowed across the service lifetime.

The implementation scope prioritizes source development. Tests, validation imports,
compilation, builds, benchmarks and hardware probes are deferred. Numerical and
platform support flags remain evidence-gated. Runtime implementations remain
independent of reference code.

Older status tables lag current source. Image processing, image-aware V4.1
forward, DSpark speculative generation, the native publisher gateway, K3 FLA
contracts, FlashAttention and BF16 sequence arithmetic already exist. Extend
these paths rather than replacing them with new interfaces.

## Required closure

| Area | Remaining implementation scope |
| --- | --- |
| Model families | Complete exact publisher profiles, full geometry, wrappers, masks, positional variants and checkpoint loading beyond bounded fixtures |
| V4.1 | Audit existing image/DSpark integration, service recovery and supported decoding policies; complete model-wide tensor parallelism |
| K3 | Complete admitted batch/mask/input contracts and native/accelerator ownership; retain exact FLA continuation state |
| Quantization | Complete persisted model/selected-expert integration of the required formats, including E8/IQ3 transforms and codebook identity |
| Accelerators | Complete full-model operation coverage on the intended backends; expert kernels alone do not close decoder coverage |
| Segment/Edge | Complete family-owned range and edge adapters, activation schemas and all continuation state |
| Serving | Single-image K3/V4.1 gateway routing now implemented; extend model-family selection and remaining media/lifecycle coverage |
| Recovery | Complete family/layout serialization, alignment and scalable persistence without discarding recurrent/convolution/compression state |

No item is closed by a placeholder, an absent injected adapter, changing a false
support flag to true, or describing a tiny fixture as a full publisher decoder.

## Distributed projection addition

`tensor_parallel.py` now implements explicit row/column partition banks using a
caller-owned initialized Torch process group. Rank admission checks validate
partitions, collective schedule and replicated input hashes. Row shards gather
uneven output channels; column shards reduce partial products in FP32. Bank
identity includes all ranks' weight hashes. Local weights and operation buffers
have explicit admission budgets.

`load_projection_shard` streams a checkpoint tensor into a rank-local row bank
without a full matrix allocation. The converter currently reads the full source
sequentially on every rank, so source I/O is replicated.

V4.1 arithmetic accepts a borrowed `projection_backend`; named K3 scalar/batch
arithmetic dispatch accepts the same bank. Register shards before constructing
arithmetic; later bank changes invalidate its admitted identity. K3 nonrouter
projections require BF16, router projections FP32. Column reductions reassociate
floating-point dot products and remain numerically uncertified.

Callers must issue identical rank requests and operation order, configure
process-group timeouts, and manage worker/session lifecycle. Device/collective
failure can invalidate the process group. Local decoder rollback is not a
distributed recovery protocol. This addition does not launch workers, transport
requests, destroy groups or provide full model-wide parallelism. K3 still owns
its normal weight inventory; its projection hook does not shard model residency.

The overall goal remains incomplete. No tests, imports for validation, builds,
compilation, benchmarks or hardware verification ran for these additions.

## Ordered distributed sessions

`DistributedExecutionService` borrows a rank-local `NativeExecutionService` and
a separate caller-initialized Gloo control group. Rank zero opens sessions,
generates, resets and closes them; peer ranks run `serve()`. Logical handles map
to separate local session IDs. The coordinator admits root token commands,
checks matching execution limits, dispatches ordered commands, and compares
exact committed token trajectories, context lengths and finish reasons.

Use exclusive ownership of each wrapped service. Generation defaults to the
service's deterministic greedy policy and admits explicit seeded sampling.
Streaming callbacks and cooperative distributed cancellation now have the
boundary protocol described below. Durable distributed snapshots and worker
launch remain separate work. Rank errors poison the coordinator; partial device or
collective failure requires rank restart. Group timeouts bound lost-peer waits.
Caller-owned groups remain alive after coordinator shutdown.

## Native gateway image requests

The native publisher provider now implements the gateway's `complete_vision`
protocol. Configure `image_placeholder` as the exact single-token placeholder
string and `--image-token-id` in the existing operator arguments. K3 additionally
requires its existing `--include-vision`, explicit vision epsilon and image
budget. V4.1 requires a checkpoint containing its vision inventory.

The gateway inserts the marker into the latest user message, renders the normal
configured chat template, then requires exactly one matching token. It submits
the supplied image bytes to the existing processor without saving a temporary
image file. K3 accepts bytes as well as local file paths and admits encoded and
decoded buffers together. Output token limits are bounded by configured limits.
Auto vision routing can select this local provider before an external endpoint.
Configured vision is reported separately from numerical/platform certification.

## K3 padded sequences and batch sessions

K3 forward accepts a binary occupied-token mask and removes padding before
cache and recurrent arithmetic. Optional incoming positions are selected with
the mask. Empty occupied sequences leave state unchanged. K3BatchExecution
owns independent sequences with aggregate memory admission, per-slot
cancellation and transactional rollback of selected slots. Factory integration
uses the existing checkpoint/library admission. This is sequential independent
session batching; fused device batching and expert-union acceleration remain.

## Streamed native Mixtral

Native Mixtral can borrow MixtralStreamedExperts to execute selected packed
experts through the registered compiled backend while retaining routing and
route weighting in C. Selective checkpoint loading validates the complete
source tensor inventory but loads only resident dense/router banks. Split and
explicit fused projection ranges are convertible one matrix at a time.
The additive native input ABI accepts full-cache binary padding masks and
incoming rotary positions; transactional snapshots include committed masks.
Larger hidden/context geometry is admitted only with the new ABI. These paths
have not been compiled or executed. Published-checkpoint support remains false.

## Persisted caller E8 contracts

Expert manifests now admit e8-lattice-caller-v1 with exact contract metadata.
Packing requires an explicit LatticeContract and records its identity. The
contract writes an atomic bounded binary artifact containing its codebook,
orthogonal transform and SHA256 identity; reads require the manifest digest
and validate content. Runtime admission and each scalar/batch dispatch require
an explicitly selected matching backend with fallback disabled. Persist the
contract artifact separately alongside the manifest. This caller format does
not implement Colibri's upstream IQ3 byte contract.

Overall code completion remains in progress. Testing, imports for validation,
compilation and benchmarks are deferred under the code-first instruction.

## Native Mixtral durable cache transport

NativeMixtralDecoder.write_cache/restore_cache stream active KV prefixes and
padding masks through a bounded checksummed binary format. Restore validates
the full execution identity, physical position, exact geometry, finite arrays,
mask values, checksum and EOF before publishing staged state. Unused capacity
is zeroed, and both existing plus staged memory are admitted. Stream callbacks
cannot reenter model mutation during transport. The caller owns atomic file
publication and stream lifecycle. Explicit rotary positions are embodied in KV
keys; new positions retain the forward API's continuation rules. This closes
the Mixtral transport implementation gap, without execution certification or
claiming closure of all family recovery requirements.

Borrowed streamed-expert identity now separates live backend ownership from
persisted kernel identity. Independently implemented OpenAgent Numba backends
use canonical capability/contract metadata, a hash of engine Python sources,
dependency versions, compiler configuration and host architecture. New
instances with the same admitted artifacts can restore saved caches; replacing
a live registry binding is still rejected. External backends keep process-local
identity until they provide an equally explicit artifact contract. Deployment
sources must remain immutable while loaded. This portable identity is strict
and may reject unrelated source changes; numerical equivalence is untested.

## Streamed Mixtral serving factories

create_mixtral_execution_service accepts a borrowed explicit expert provider
with a resident dense/router tensor inventory. The new
create_streamed_mixtral_directory_service admits checkpoint configuration and
provider identity up front, then uses complete-header validation with selective
dense/router loading for each independent execution session. The shared
provider serializes selected expert calls and retains its explicit cache and
compiled-kernel policy. Factory cleanup retains failed decoder destruction for
retry during service close. The caller owns provider lifetime and conversion;
no quantizer is inferred or silently selected. Application-level model routing
and durable service token/logit metadata integration remain separate scope.

## Sparse metadata admission

SparseExpertManifest.read now accepts explicit byte, layer and total expert
limits, reads bounded bytes, and rejects duplicate fields and nonfinite JSON
constants. Manifest writes use unique temporary files with fsync and atomic
replacement. Route-profile reads are bounded and validate transition layer,
expert and route-width references before publishing the predictor. These
admission changes are source-only and have not been exercised.

The common load_native_text_decoder dispatcher now admits an explicit
Mixtral streamed provider and routes it through selective checkpoint loading.
Other family/provider combinations fail explicitly; they do not silently load
resident expert banks or delegate to a different family implementation.

## Qwen4 linear RoPE

Independent and native Qwen4 text paths now admit an explicit linear rotary
factor alongside default RoPE. Scaling applies to main QSA queries/keys and
indexer queries/compressed keys with their stored physical positions. The
original native function remains default-only; scaled execution requires the
new versioned additive function. Old libraries reject nonunit scaling. Dynamic
NTK, YaRN and multimodal position policies remain outside this added contract.
No native compilation or numerical checks have run.

## Qwen4 owned continuation snapshots

Qwen4 text decoders now expose owned opaque snapshot/restore/release handles.
Snapshots include recurrent and convolution caches, QSA keys/values/raw index
history, PLE convolution/token history, masks, explicit positions and physical
progress. Restore stages copies before publishing. Snapshot inventories and
bytes are admitted together with current state and forward work. Identity
binds configuration and resident weights; native identity also hashes the
explicit library artifact. These are in-memory session snapshots; portable
Qwen4 serialization and numerical execution checks remain outstanding.

create_qwen4_execution_service now connects the native text decoder to the
common generation lifecycle using its owned snapshots and independent mutable
state. Configuration, library, prefill, logit and snapshot budgets are explicit.
The factory accepts exact resident tensors and retains caller source ownership
until each decoder copies them. This does not complete the visual wrapper or
quantized full-model execution paths.

## Qwen4 durable continuation transport

Qwen4 write_cache/restore_cache now stream a versioned complete state inventory
containing linear recurrence/convolution, QSA KV/raw index history, PLE
tokens/convolution, masks and positions. Restore requires matching execution
identity and exact geometry, validates finite values and integer histories,
checks SHA256 and EOF, and publishes only after the full frame is admitted.
Transport blocks model mutation through reentrant stream callbacks. Buffers
are read/written in bounded chunks; restore also admits old and staged state
plus retained snapshots. The caller owns atomic destination-file publication.
The earlier portable Qwen4 serialization gap now has source implementation;
runtime and cross-process numerical checks remain deferred.

## Quantized conversion working-memory admission

The directory expert quantizer and Mixtral conversion wrapper now expose
max_conversion_bytes. Each mapped projection is admitted before payload read,
including conservative source/promotion/quantizer/output staging and E8
codebook work. Source matrix geometry and byte extent are checked against the
binding before allocation, so an oversized mismatched tensor cannot bypass the
declared conversion shape. The default working limit is 1 GiB and can be set
explicitly for larger publisher projections. No conversion was executed.

Expert conversion now records shard identity/extent/timestamps during header
admission, checks opened-file and path identity around each matrix read, and
rechecks all unique source shards before publication. SafeTensors headers also
reject duplicate fields, nonfinite constants and boolean shape dimensions.
This detects ordinary concurrent replacement/mutation; deployments still
require immutable source checkpoints throughout conversion. Source payload
authenticity remains separate from the converted region checksum.

Conversion writers use unique exclusively created temporary files, fsync
completed containers before replacement, and remove their own partial files
on cancellation as well as ordinary failures. Directory conversion rejects
destinations that would overwrite source shards. A multi-storage conversion
publishes separate files sequentially; the returned manifest must be published
last by the caller, and it is never returned after a partial publication error.

## Service-owned durable cache frames

NativeExecutionService.write_session_cache/restore_session_cache now transport
committed service token history together with an executor's complete cache
frame. Service metadata has bounded geometry/history and its own checksum;
the executor codec checks its state payload and EOF. Restoration requires
matching identities, stages native state, verifies restored physical position
against history, and publishes history under one session transaction. Failure
restores the prior owned snapshot, with quarantine if rollback fails. Executors
without a complete cache codec fail explicitly. The caller owns atomic file
publication and trusted association of history with state; checksums detect
corruption and do not authenticate a checkpoint. No recovery was executed.

save_session_cache/load_session_cache provide atomic file publication and
bounded regular-file admission for these frames. Writes use unique temporary
files, fsync, receipt/extent checks and replacement. Reads reject links/reparse
paths and check source identity/extent/timestamps at executor EOF, before its
transaction commits. recover_cached_session opens a fresh executor and retains
cleanup failure details if restoration fails. The service header converts JSON
history to the exact tuple contract before validation. These paths remain
unexecuted under the code-first instruction.

evict_cached_session now saves an idle session's complete state, reads back
the published file with bounded buffers, verifies its exact write-time SHA256
and extent, and only then closes the executor. Failure before cleanup retains
the live session; cleanup failure retains the durable frame and its path in
the error. File receipts include whole-frame hashes, so readback covers both
service history metadata and executor cache bytes. This lifecycle is source
implemented and remains untested.

## Distributed seeded sampling

The ordered coordinator now accepts an explicit SamplingPolicy, broadcasts its
bounded scalar/seed contract, and constructs the same request-local sampler on
every rank. Existing exact trajectory and committed-position comparisons still
apply. Complete serialized command bytes are admitted before broadcast; seeded
policy top-k and seed width are preflighted. This extends decoding policy
coverage without changing the current rank-restart failure protocol. Rank
numerical equality and stochastic behavior have not been executed or certified.

## Ordered distributed in-memory snapshots

The coordinator now maps logical snapshot handles to each rank's service-owned
snapshot. Snapshot, restore and release are ordered commands, with history
length/digest and restored-position consensus. Closing sessions invalidates
their handles. Root admission checks public snapshot inventory; generation
also admits response trajectory size against its control budget. Snapshot
history checks hash bounded chunks rather than gather entire histories.
These are rank-local in-memory snapshots; durable distributed recovery,
and worker launch remain outstanding. Failures still
poison the coordinator and require coordinated rank restart.

## Cooperative distributed generation cancellation

generate now accepts a root-owned cancellation event. Ranks compare generation
event signatures and synchronize cancellation before prefill and after each
prefill chunk/token. Native/backend polling reads a local latched event rather
than issuing collectives inside arithmetic. A cancellation with successful
rollback and matching restored context lengths advances the command sequence
and leaves peer loops available; root receives ExecutionCancelled. Quarantine,
cleanup failure, divergent event schedules or collective failure still require
rank restart. Cancellation latency is bounded by a model chunk/token plus
control collectives, subject to configured process-group timeouts. No ranks
were launched and this protocol has not been executed.

Distributed generate also accepts a rank-zero on_event handler. Events are
delivered after event/token schedule consensus and remain tentative until the
returned GenerationResult confirms commit. Peers synchronize callback failure
before continuing arithmetic; a failed callback rolls back local transactions
and poisons the coordinator under the current failure policy. Callbacks stay
local and are never serialized into commands. Streaming transport integration
can now use this hook; event delivery is unexecuted and unverified.

## Managed distributed worker launch

launch_distributed_execution now spawns owned ranks with operator-supplied
TCP/file rendezvous, timeout, Gloo/NCCL projection backend and explicit CUDA
device mapping. Each worker initializes a separate CPU control group and
projection group, constructs its rank-local service through an importable
factory, runs the root driver or peer control loop, and closes local services
and groups on exit. Root shutdown is ordered while the coordinator is healthy;
poisoned ranks shut down locally and worker failures propagate through the
spawn supervisor. Driver errors retain shutdown-failure notes. Call from a
guarded importable __main__; factories own their additional resources and
must implement their existing partition contracts. No workers were launched.
This implements process lifecycle, not missing model-wide sharded residency
or distributed durable cache recovery.

## Distributed durable checkpoint publication

The coordinator can save a complete session into a new generation directory
on an explicitly shared filesystem. Each rank writes and reads back its own
bounded complete service cache. Consensus checks history length and digest;
rank receipts include file extent, whole-frame SHA256 and execution identity.
Only after every rank succeeds does root atomically publish manifest.json as
the generation commit record. Partial generations remain inspectable without
a manifest. Coordinated restore and partial-restore rollback now have the
implementation described below. No files or ranks were generated
by executing this code.

load_session_cache admits bounded exact manifest fields, rank inventory,
filenames, histories and receipts before broadcasting restoration. Each rank
reads back the expected frame, checks its executor identity, retains an owned
rollback snapshot, restores its service cache and checks history against the
manifest. Rank consensus precedes releasing rollback state. A partial failure
restores ranks that acquired rollback snapshots, gathers cleanup failures,
and poisons the coordinator under its conservative restart policy. Recovery
requires the same rank count and explicit complete executor codecs; no missing
family codec, changed partition or different numerical artifact is substituted.
This distributed recovery code has not been executed.

## K3 and publisher V4.1 service cache integration

K3Execution and PublisherV41Execution now expose write_cache/restore_cache using
their existing complete checksummed checkpoint codecs. A bounded scratch-file
adapter transports those formats through service streams in 1 MiB chunks,
avoiding a second private cache schema and token replay. Restoration consumes
and checks source EOF before invoking the family loader; the family loader
still validates its full state before publishing. Model locks serialize the
operation and busy flags reject stream-callback mutation. Scratch files are
removed on all exits; scratch_directory is optional and otherwise uses the OS
temporary directory. Existing family payload/staging admission remains active;
transport adds bounded stream buffers and one frame of scratch disk storage.
These hooks allow applicable owned service adapters and distributed recovery
to retain K3 recurrent/FLA and V4.1 packed/pending cache state. No checkpoint
transport or numerical verification was executed.

## Public Segment/Edge cache transport

SegmentEdgeExecution now exposes complete service cache transport using opaque
adapter-owned snapshot blobs. The frame binds execution identity, context,
position and exact segment inventory, with each blob and the complete frame
checksummed. Restore admits old rollback plus incoming scratch storage,
validates the entire frame and EOF before native restore, and attempts rollback
of all segments if a native restore fails. A failed rollback leaves quarantine
active. Transport locks and guards reject stream-callback mutation. This
preserves each registered adapter's existing state contract; it does not supply
missing model-family range arithmetic or certify external adapters. No native
adapter, snapshot transport or test was executed.

## K3 sequence projection coverage

Source review found KDA/MLA occupied-prefill helpers and the batched vocabulary
head calling local matrix arithmetic directly. They now use the named sequence
projection dispatcher, so explicitly registered tensor-parallel banks also
cover those sequence paths. Local matrices remain the fallback for unselected
names. This closes a dispatch bypass; it does not remove K3's retained full
weight inventory or certify rank arithmetic. No sequence was executed.

## K3 rank-local projection residency

K3Execution now admits explicit sharded_projections with matching identified
BF16 banks. Selected admitted projection names are omitted from the resident
tensor inventory and represented by private shape references; dispatch must
resolve the same bank and cannot fall back to missing full weights. Identity
includes those references and bank identity. Memory admission reserves the
borrowed bank's explicit budget alongside local model state. Service and batch
factories filter selected matrices before decoder construction. Embeddings,
norms and residual mixing remain resident; scalar MLA dispatch is extended below.
The prepared checkpoint payload may still retain full source arrays, so
selective publisher payload loading remains required for full process-wide
sharded residency. No decoder or collective was executed.

Scalar publisher-expanded MLA now dispatches each active latent row through
its named kv_b_proj projection. This removes its direct full-matrix read under
BF16 arithmetic and allows kv_b_proj rank-local references as well as sequence
prefill. The non-BF16 compressed path retains its own resident matrix contract.
Embeddings, residual projections and selective source payload loading remain
separate residency work. No arithmetic or attention was executed.

## Selective K3 publisher payload preparation

load_k3 now accepts explicit sharded runtime projection names. It first checks
the complete mapped executor inventory, then omits admitted complete projection
rules from source conversion and retained arrays. The payload records its
omission contract; service/batch factories inherit that contract and reject
conflicting selections. Packed destination pairs cannot be partially omitted.
The executor still requires matching immutable BF16 banks for every omitted
matrix. This removes full selected-matrix staging in each publisher payload;
rank banks must be explicitly prepared from the same source contracts. Nonlinear
weights, embedding/residual ownership and broader V4.1 tensor parallelism remain
separate work. No checkpoint loading or rank execution was performed.

load_k3_projection_shard now resolves exact publisher rules to admitted runtime
projection names, streams one output-row shard, and returns that runtime name
for bank registration. This also handles the explicit packed-expert-to-FP32
runtime spelling without guessing an alias. Generic shard conversion now
closes checkpoint streams on failure as well as success. Source I/O remains
replicated across ranks; no shard conversion was executed.

### V4.1 grouped output projection code

The target publisher attention and DSpark/MTP paths dispatch `wo_a` through an explicit grouped arithmetic hook. The distributed bank selects each local weight row's group input and performs one gather or reduction, supporting either existing shard axis without materializing a full matrix on a rank. Rank agreement covers group counts, geometry, identity, input hashes, and admission failures before device arithmetic. Budgets include expanded group inputs and collective buffers. Resident paths preserve BF16 weight conversion and their existing scalar/batch contraction policies; older custom arithmetic retains the original resident target path.

V4.1 payload loading accepts an explicit unique `sharded_projections` tuple of exact checkpoint source names. Whole-inventory contracts are admitted before selected matrices are omitted from resident allocation. Only text/MTP projections dispatched through arithmetic hooks are eligible; embeddings, norms, direct tensor operands, and vision matrices remain resident or provider-owned. `load_v41_projection_shard` resolves one exact eligible source to its mapped target, streams an output-row shard, and preserves source naming for bank registration. Native arithmetic rejects omitted matrices unless its borrowed bank owns the exact shape and supports grouped output when required. The owned publisher service entry point forwards the selection and borrowed projection bank without taking bank ownership. Source I/O remains replicated across ranks. No tests, imports, compilation, or distributed execution were run. Whole-model partitioning and broader accelerator/family coverage remain open.

### Speculative rollback ownership

Speculative restore and failed generation transactions now attempt target, draft, and target RNG rollback independently, preserving the initiating error and annotating each rollback failure. Failed rollback quarantines generation, snapshot, and restore operations until a complete reset succeeds. Built-in autoregressive and DSpark proposers expose cache reset; recovery of an arbitrary stateful proposer requires its reset hook. A reset failure retains quarantine. Prefill rollback failure also quarantines the session. Complete durable compound speculative state serialization remains open; this change addresses failure ownership rather than claiming persistence completion. No runtime checks or tests ran.

Speculative snapshot capture, restore, reset, and transaction rollback-state capture now hold the decoder's active-operation guard across calls into target/proposer owners. Reentrant generation or state operations from custom owner callbacks are rejected; guards are released on every exit. Reset retains quarantine on any owner failure. These are source changes only; no execution or validation imports ran.

### DSpark/MTP durable draft state

`PublisherV41MTP.write_cache` and `restore_cache` now transport a complete versioned draft frame: exact provider identity, window geometry, position, all stage window bytes, and PCG64/PCG64DXSM generator state. The codec uses bounded duplicate-free JSON metadata and SHA-256 for corruption detection, validates complete extent/checksum/EOF before public restore, and rejects other RNG families explicitly. Writer progress and reader extents are bounded. Reentrant snapshot/reset/restore/forward during external stream callbacks is excluded. The caller serializes access to other owners and supplies transport budgets. This is the draft-state component; compound speculative persistence still requires target evidence, proposer RNG, and verifier logits/history wiring. No tests, imports, or compilation ran.

### DSpark target evidence recovery

`DSparkBackbone.write_cache` and `restore_cache` now store retained target hidden evidence and next logits alongside the existing complete publisher packed cache frame. Exact model identity, layer order, widths, context extent, nested frame size, finite evidence, whole-frame checksum, and EOF are admitted before applying recovery. Nested cache transport stages on a temporary disk file; evidence staging is charged against the backbone retention budget. Recovery checks evidence count against the restored target position and attempts target rollback on failure. Failed rollback quarantines the evidence owner until reset. External stream callbacks cannot reenter the wrapper or mutate the underlying publisher model during transport. The compound speculative codec still needs proposer and verifier-state composition. No tests, imports, compilation, or runtime checks ran.

### Compound DSpark speculative recovery

`SpeculativeDecoder.write_cache` and `restore_cache` now compose the built-in DSpark target evidence/cache and MTP draft window codecs with committed token history, verifier FP64 logits, verifier RNG, and proposer RNG. The versioned outer frame checks exact identities, policy, bounded metadata, nested frame sizes, token bounds, finite logits, checksum and EOF. Nested frames stage on temporary disk files and are individually admitted by their owners. Restore captures rollback states, applies both caches, checks exact target token history, restores RNGs, and publishes verifier history/logits only on success. Any application failure attempts all independent rollback owners; rollback failure quarantines generation. External stream callbacks are guarded against decoder, target, and draft reentry. PCG64/PCG64DXSM are explicit supported durable RNGs; arbitrary proposer compositions remain outside this codec. Byte budgets bound serialized/staged payloads rather than total process memory or Python object overhead. No execution, tests, imports, compilation, or certification was performed.

Speculative snapshots now preserve the proposer temperature policy, and both in-memory and durable recovery cross-check restored public target token history and retained logits against verifier state when available. Durable writes also check the association before serializing. Mismatched target evidence triggers the existing transactional rollback instead of publishing a mixed continuation. No runtime validation or tests ran.

### Atomic executor and speculative cache files

`save_executor_cache` and `load_executor_cache` reuse the complete service-cache file lifecycle for any executor-owned stream codec. A bounded counting writer measures actual frame bytes independently of codec return values, enforces the publication budget, and produces the existing byte/SHA-256 receipt. Publication uses a unique same-directory temporary file, flush/fsync and atomic replace; failed writes clean up the temporary file. Loading admits stable regular-file size and replacement stamps before codec commit. `SpeculativeDecoder.save_cache` and `load_cache` expose this lifecycle for the compound DSpark codec. These helpers do not add replay fallback or change family state schemas. No files were encoded/restored and no tests, imports, or builds ran.

### Native publisher vision streaming

`NativePublisherProvider.stream_complete_vision` now accepts the same bounded encoded image, MIME and output limit contract as non-streaming vision completion, then uses the shared bounded stream worker. Text and image streaming share queue backpressure, consumer-close cancellation, final metadata, and native session lifecycle. Vision runs retain the existing placeholder/token admission, configured media budgets, model selection, and tokenizer prefix consistency rules. Gateway routing to this new capability remains a separate integration item. No application requests, tests, imports, or builds ran.

### Gateway vision stream dispatch

`ModelGateway.stream_complete_vision` now uses the existing vision-provider admission and effective-model routing. Native publisher image streams retain image bytes and output limits, expose routing/modality/timing metadata, require an explicit completion event, and close the provider iterator when consumers disconnect. Providers without a native image stream use their existing image-aware completion protocol and then emit buffered chunks, marked `buffered-vision`; images are never dropped into text completion. External provider calls retain the existing egress approval context. HTTP/chat endpoint exposure is still to be connected where applicable. No requests, tests, imports, builds, or hardware checks ran.

### Chat image streaming integration

Chat streaming now resolves its current/historical image attachment and dispatches it through `ModelGateway.stream_complete_vision`. Text-only attachment completion retains its existing correction behavior. Image streams preserve extracted evidence in the prompt, persist the completed assistant message with provider metadata, and close the nested gateway iterator when the consumer exits so native cancellation reaches the worker. Image-provider failure remains explicit during streaming rather than switching an in-progress image request to text inference. No HTTP/chat requests, tests, imports, or builds ran.

Chat image attachment loading now admits a non-empty regular-file extent within the gateway's 256 MiB limit before allocating encoded bytes, uses a bounded read, and compares descriptor/path identity, size and timestamps after reading. Concurrent file growth/replacement is rejected instead of passing mismatched asset bytes to native inference. No attachment files were loaded and no tests or imports ran.

Compound speculative writing now subtracts the staged target-cache extent before admitting the draft writer, so simultaneous nested staging cannot reserve the entire outer transport budget twice. Final metadata/history/logit overhead still undergoes the complete frame-size admission before publication. No cache files or runtime codecs were executed.

### Exact IQ3 format-6 decode primitive

`iq3_lattice.py` independently decodes the reference's 98-byte/256-weight format-6 container: four-dimensional grid indices, seven-bit sign words with implied eighth sign, four-bit subscales, and FP16 super-scales. The caller supplies the exact doubled-magnitude uint8 grid with an explicit matching SHA-256; immutable backing pins grid bytes. Payload geometry, byte extent, finite scales, and allocation budgets are admitted. This format is distinct from caller E8. Decoded weights remain in rotated coordinates; the required model rotation, encoding, persisted manifest contract and compiled expert/backend dispatch still need integration. No reference implementation was imported, executed or copied; no tests, imports or builds ran.

The IQ3 primitive now supplies forward/inverse row rotation using the specified seeded xorshift64-star sign stream, low-bit power-of-two width decomposition capped at 32768, and normalized Hadamard butterflies. Weight rows and activation rows use the same forward transform; inverse rotation changes sign/transform order. The implementation constructs no dense rotation matrix and admits conversion/output/butterfly scratch before allocation. `ROTATION` names the precise deterministic contract. Persisted IQ3 encoding and compiled expert integration remain open. No rotation was executed and no tests, imports or builds ran.

`encode_iq3` now packs the format-6 layout independently: optional deterministic weight rotation, RMS/20 super-scale rounded to stored FP16, sixteen subscale candidates with nearest-grid search, smallest-magnitude sign correction for each eight-lane parity constraint, and exact index/sign/scale fields. Search uses FP64 and makes no reference-converter numerical agreement claim. Memory admission includes rotated matrix, payload, grid/sign and bounded per-superblock search buffers. Finite scales outside FP16 range are rejected; underflowed zero scales remain representable. Persisted expert manifest and compiled dispatch integration remain open. No weights were encoded and no tests, imports or builds ran.

IQ3 now integrates with expert manifest admission and SafeTensors packing under its distinct `iq3-format6-98-v1` dtype. Gate/up/down matrices are individually rotated/encoded into the 98-byte container without appended row scales. Manifests pin exact grid SHA-256 and deterministic rotation identity, admit both projection column alignments and complete expert byte extent, and require explicit matching kernel contracts with fallback disabled during execution. The packer takes `iq3_grid` separately from caller E8 contracts. A compiled IQ3 expert backend remains to be added; this manifest admission does not claim executable accelerator coverage. No expert conversion, tests, imports or builds ran.

### Compiled streamed IQ3 experts

`NativeIQ3ExpertBackend` implements selected leased format-6 gate/up/down experts with optional explicit Numba compilation and no fallback. It decodes weights inside dot products without allocating full decoded matrices, rotates hidden and SwiGLU activation vectors under the pinned deterministic transform, validates super-scale bytes and finite arithmetic, and applies route weights to outputs. Scalar/batch interfaces use the existing payload lease and route admission contracts, bound host workspace, and detach borrowed payload frames on failure. Backend metadata advertises the precise grid identity and IQ3 dtype for explicit manifest-matched registration. No kernel was imported, compiled or executed; numerical/platform coverage remains unverified and full-model accelerator work remains open.

IQ3 grid contracts now support fixed-size versioned atomic file publication and bounded reload with both stored and caller-expected SHA-256 admission. The Mixtral expert converter forwards an explicit `iq3_grid` through split/fused source mapping to the quantized packer, preserving exact grid/rotation manifest metadata. Existing explicit compiled-kernel registration and fallback-disabled sparse runtime rules apply to IQ3 streamed Mixtral experts. No grid files, expert conversion, inference, tests, imports or builds were executed.

IQ3 grid reload now checks fixed regular-file extent and descriptor/path stamps, rejecting source replacement during admission. Grid publication fsyncs its containing directory on supported POSIX hosts after atomic replacement. Decoder and compiled-dot scale admission now both reject sign-bit-set FP16 scales, including negative zero. Mixtral's existing portable OpenAgent/Numba artifact identity already covers the new IQ3 backend, so no separate process-local binding exception was introduced. No grid files, inference, tests, imports or builds ran.

### OLMoE persisted lattice expert conversion

`pack_olmoe_experts` now forwards explicit caller E8 and exact IQ3 grid contracts, together with the working-memory budget, to the common bounded SafeTensors converter. Its existing model metadata fingerprint and exact expert bindings remain authoritative; lattice manifest identities and strict backend matching are handled by the shared sparse runtime. This extends the actual OLMoE conversion API beyond format primitives without introducing a different payload schema. CLI configuration for these explicit lattice contracts remains to be connected. No conversion, tests, imports, builds or compilation ran.

OLMoE CLI packing now admits caller E8 and exact IQ3 dtype selections with explicit persisted lattice file and expected identity arguments, plus a conversion working-memory limit. The operator reads the corresponding bounded contract format before conversion and forwards it to the existing packer; lattice flags on unrelated dtypes are rejected. Format-specific container/manifest defaults are supplied. No CLI command, conversion, tests, imports or builds ran.

OLMoE runtime kernel selection now resolves explicit `iq3-format6-98-v1-<SHA256>-numba` and `e8-lattice-caller-v1-<SHA256>-numba` names to their admitted compiled backends. The runtime reads the corresponding `openagent-iq3-grid.bin` or `openagent-e8-contract.bin` artifact from the checkpoint directory with the exact expected identity, rejects unavailable implementations, and registers the actual backend name for manifest matching. Operators retain ownership of persisted contract placement; missing artifacts never trigger a substitute grid or fallback. No runtime commands, tests, imports or builds ran.

### Native Mixtral hidden range execution

The native Mixtral arithmetic now provides an additive versioned hidden-to-hidden range entry point. It uses the actual Mixtral attention/router/SwiGLU path with its own norms and sliding-window/mask/position semantics, supports resident or selected streamed experts, accepts staged KV banks and hidden rows, and returns post-layer hidden rows without loading embeddings or applying final norm/head. Existing full-decoder exports keep their signatures and select the original token-to-logit mode. Caller-owned staged cache commit remains required. Python range ownership/checkpoint loading and public Segment/Edge registration are the next parts of this component; the C entry point alone does not close family adapter scope. No compilation, imports, tests or execution ran.

`NativeMixtralRange` now owns an exact selected fused layer inventory and invokes the additive native hidden range API. It admits complete model configuration, layer bounds, batch/capacity and weight/cache/scratch budgets, copies weights into immutable backing, pins range execution identity, stages KV/mask continuation, supports explicit rotary positions and full-prefix padding, and commits only after native success. It exposes active-prefix snapshots with identity/extent/finite-state validation, reset and close, while borrowing the native library. Selected checkpoint loading, streamed range experts and public Segment/Edge registration remain parts of this component to finish. No imports, compilation, tests or execution ran.

Mixtral range checkpoint loading now admits every source tensor name and shape against either the exact fused or split family layout before selecting layer payloads. The range loader reads only its selected layer interval, excludes all global embeddings/head payloads, converts selected split expert banks into the native fused layout, and applies explicit source/resident/cache/conversion memory admission. It retains existing checkpoint metadata/file guards and cancellation completion handling. This completes the selected-range checkpoint path; streamed range providers and public Segment/Edge registration remain. No checkpoint was read, imported, tested or compiled.

Mixtral native ranges now borrow the complete admitted packed-expert provider. Dense-only selected loading retains full fused/split inventory admission, then omits resident expert matrices. Construction checks exact model fingerprint and full-model provider geometry, reserves provider storage/work budgets, pins provider identity, and installs a native selected-expert callback mapping local range indices to original checkpoint layers. Callback failure preserves staged continuation; provider weights remain caller-owned. The range executes resident and streamed modes through the same native attention/router path. Public Segment/Edge registration remains to be connected. No checkpoint loading, inference, compilation, imports or tests ran.

Mixtral now has additive native edge embedding and final-norm/head APIs, plus `NativeMixtralEdge` ownership. The edge admits only exact global tensors, handles tied-weight consistency, pins immutable weight identity, bounds embedding/logit staging, and performs numerical operations in the native kernel. Optional caller-owned tokenizers provide bounded tokenize/detokenize methods. It borrows library/tokenizer lifetime and closes owned weights explicitly. Selected edge checkpoint loading and public registration remain next; no decoder layer weights are required by this edge class. No native calls, compilation, imports or tests ran.

Mixtral edge checkpoint loading now validates every tensor name/shape against one complete fused or split family inventory before selecting global tensor payloads. The shared inventory contract preserves tied-head omission rules and exact attention/expert geometry. `NativeMixtralEdge.from_directory` reads only embedding/norm/head data, reserves conversion plus immutable ownership, retains existing file metadata/cancellation guards, and borrows its explicit tokenizer/library. It loads no decoder layer payloads. Public registration remains to be completed. No checkpoints, native operations, imports, tests or builds ran.
