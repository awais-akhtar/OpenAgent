# Generic Scalable Execution Infrastructure

This opt-in package lives in `openagent.adaptive_engine.scalable_execution`.
It does not change existing engines, native limits, model support declarations,
or the shared `execution_service.py`. No tests, compilation, lint,
benchmarks, or verification were run for this source implementation.

## Concrete Facade

A caller-supplied factory creates a fresh `ScalableSession`:

```python
session = ScalableSession(
    executor, session_id, owner, planner, vocab_size=vocab_size,
    create=True, prefix_id=None,
)
logits = session.forward(token_ids, cancelled=cancelled)  # numpy [N, vocab]
checkpoint = session.snapshot()                         # durable prefix handle
session.restore(checkpoint)                             # atomic session rollback
session.release_snapshot(checkpoint)                    # explicit pin release
session.reset()                                        # atomic empty state
session.close(preserve_durable_session=False)
```

`planner(tuple[int, ...]) -> ExecutionPlan(steps, finish, scratch_bytes)` must
provide real backend math. Each `DispatchStep(key, callback)` invokes
`callback(context, packed_memoryview)` under a checksum-checked lease. A `None`
key invokes a non-packed phase with `None` as its payload. `steps` can be a
generator that emits routed experts after an earlier callback computes routes.
`finish(context)` returns floating NumPy logits; shape/type are checked before
cache commit. NaN/Inf are rejected before commit using bounded chunks; the
scratch estimate must include finite-check buffers (up to 4096 values plus
their boolean mask). The facade's active guard covers iterator ingestion,
integer conversion, planning, callbacks, and output checks. There is no
fake/default model implementation.

Direct lower-level use:

```python
executor.dispatch(session_id, owner, token_ids, steps, finish,
                  scratch_bytes=scratch_bytes, cancelled=cancelled)
```

Callbacks use `context.state` for ephemeral backend scratch,
`context.cache.read_page(ordinal)` as a context manager, and
`context.cache.append_page(token_ids, payload)` for immutable new pages.
`context.cache.page_count()` includes staged pages. `context.position` is the
committed start position. Every admitted token must be appended exactly once,
in order, in groups of at most `page_tokens`; omission, extra tokens, wrong
byte counts, cancellation, or callback failure rolls back the entire dispatch.

All persistent adapter state must be serialized into these cache pages. Hidden
mutable backend state or external side effects cannot be rolled back by this
executor. The parent coordinator must release checkpoint handles after success
or after rollback. Unreleased handles intentionally pin their pages; hitting the
prefix quota is an explicit admission error, never silent eviction.

## State Codec Planner Hooks

`StateCodecPlanner` supplies concrete orchestration for injected backend hooks:

```python
codec = StateCodec(
    backend_identity=geometry.identity,
    cache_layouts=geometry.cache_layouts,
    create=create_request_local_state,        # (committed_position) -> state
    decode_page=decode_exact_cache_page,       # (state, readonly_view, metadata)
    encode_pages=encode_appended_state_pages, # (state, tokens, page_tokens)
    dispose=release_request_local_state,      # (state) -> None
)
planner = StateCodecPlanner(
    geometry, codec,
    phases=backend_phases,                    # (context, state) -> DispatchSteps
    logits=backend_owned_logits,             # (context, state) -> numpy[N,vocab]
    scratch_estimate=estimate_request_bytes,  # (tokens, max_context_tokens) -> int
)
```

Codec identity/layouts must exactly match the cache contract. On dispatch the
planner creates fresh request-local state and reconstructs it from leased,
checksum-checked pages. `CachePageMetadata` provides ordinal, causal position,
original token IDs, and payload byte count. It then streams backend phases,
obtains an owned logits array, and streams encoded appended pages into the
transaction. The scheduler closes phase/encode generators deterministically;
registered state cleanup runs before commit even on failure. Cleanup failure
rolls back. The logits array must remain valid after state disposal.

This is reusable state/planner binding, not itself a K3/V41 implementation. The
explicit K3 hybrid fixture binding below supplies a concrete native adapter;
V41 still needs exact serialization and native phases. No model's snapshots
are guessed or treated as a portable codec. Reconstruction may allocate a
context-sized state only if admitted by the supplied scratch estimate. A
failing `create` hook must clean up its own partial allocations.

## Concrete Native K3 Hybrid Binding

`K3HybridSnapshotCodec` binds the current exact `NativeKimiK3Hybrid` class,
not the dense-only decoder or publisher checkpoint inspection API:

```python
codec = K3HybridSnapshotCodec(
    admitted_native_hybrid,
    mla_layout="expanded-head-kv-v1",
    max_payload_bytes=4 * 1024 * 1024,
)
geometry = codec.geometry(max_batch_tokens=8)
# Construct cache/executor with this geometry. Each frame is one token;
# page_tokens=1 and a read_budget of at least 2*codec.step_bytes are sufficient.
session = bind_k3_session(executor, codec, session_id, owner, create=True)
logits = session.forward(token_ids, cancelled=cancelled)

payload = codec.encode_snapshot(admitted_native_hybrid.snapshot())
snapshot = codec.decode_snapshot(payload)
codec.restore(admitted_native_hybrid, payload)  # complete parse before mutation
```

The codec persists canonical JSON config plus a SHA-256 config fingerprint,
model weight identity, KDA component identities, serialization version, MLA
layout, and exact array descriptors (name, `<f4` dtype, shape, C layout, offset,
byte extent). Frames have explicit magic, exact admitted length, absolute
position, and SHA-256 checksum. There is no pickle, object dtype, arbitrary
array inventory, or payload-driven allocation shape. Full snapshots preserve
the native capacity buffers with zero unused MLA tails. Decode validates all
state before native atomic restore. A codec instance admits both its maximum
full-snapshot size and per-token frame size before encoding/decoding.

`K3HybridPlanner` reconstructs a native request-local cache from committed pages,
then executes the actual native hybrid forward method one token at a time.
Each page stores only that token's expanded MLA K/V rows, plus the KDA matrix
and all Q/K/V convolution histories at that position. All layers have matching
positions. Recurrent snapshots and per-token rows are typed and checksummed;
full MLA prefixes are not duplicated per durable token. Failure/cancellation
discards the candidate caches and rolls back the SQLite append transaction.
Shape/finite output checks and cleanup precede commit. The borrowed template's
state is not forwarded or mutated. Its immutable weights/native code are shared
by exact-class cache forks, and it must remain open while sessions are bound.

`scratch_required(tokens)` includes the native hybrid's current conservative
workspace formula, full snapshot reconstruction, codec frames, output and
finite-check buffers. The native engine still independently enforces its own
`max_bytes`, geometry and batch limits. This binding does not stream native
resident expert weights: current `NativeKimiK3Hybrid.forward` invokes its own
resident MoE objects, with no packed-expert callback ABI.

### Compressed MLA Incompatibility

The reviewed API exposes `KimiK3HybridSnapshot(identity, length,
keys, values, recurrent)`, where MLA keys are `[capacity, heads, nope+rope]`
and values are `[capacity, heads, value]`. Compressed MLA changes that contract
to latent/cache components with different shapes and/or fields. This adapter
requires explicit `expanded-head-kv-v1` and checks actual native array shapes;
it rejects compressed layouts instead of expanding, guessing, or reusing the
same durable identity. A compressed adapter needs the new snapshot fields,
exact latent/rotary serialization order and version, native restore invariants,
and a separate fingerprint/page contract. Publisher-scale support is not claimed.

`test_k3.py` contains serialization and actual native binding harness cases.
Native cases require an existing `OPENAGENT_K3_TEST_LIBRARY`; the harness never
compiles a library. None of these cases were executed during this implementation update.

## Geometry and Admission

`Geometry` requires backend identity, hidden/intermediate widths, layer count,
attention/KV head counts, head dimension, cache scalar width, context/batch
limits, supported packed formats, and one `CacheLayout` for each layer.
Attention projection size is independent of hidden size. These fields do not
assert that arbitrary native kernels accept that geometry.

`CacheLayout(kind, bytes_per_token, layout_identity)` explicitly declares each
layer's serialized bytes. Kinds are `dense-kv`, `latent-kv`, `recurrent-state`,
and `custom`. Only `CacheLayout.dense_kv(kv_heads, head_dim, scalar_bytes,
layout_identity=...)` uses the dense K+V formula. MLA, CSA, mixed layouts, and
recurrent models must declare their actual backend serialization. Layout
identity includes ordering, numeric format, positional rules, and version.
The cache contract persists every layer layout, not just an aggregate size.

Each token's page representation occupies the sum of the declared layer bytes.
The adapter defines the page's exact ordering. Recurrent-state pages represent
per-step snapshots by default; this is not a rolling-state allocator.
Opt-in `CacheLayout(..., variable_size=True)` declares an upper byte bound.
The generic cache admits nonempty payloads up to that bound; architecture
codecs enforce exact frame lengths. Default fixed layouts retain exact-length
admission and their original durable contract serialization.

Separate shared `Budget` pools cover mapped packed regions, cache read/bind
buffers, and dispatch scratch/output. A mapped lease charges alignment padding
and page-rounded extent. Cache read/bind admission requires twice one page's
bytes for transient copying. The caller's backend allocation for an append
page belongs in the scratch estimate. Pools fail fast and release on exceptions.
The executor also limits inflight requests and streamed step count.

These bounds are explicit allocations, not a hard process-RSS/OS page-cache
limit. Python objects, SQLite internals, third-party kernels, and OS caching
remain outside exact accounting. Trusted backend callbacks must honor their
scratch/output reservation; arbitrary native allocation cannot be intercepted.

## Packed Storage

`PackedStore(root, regions, mapped_budget=..., storage_roots=...)` accepts
`PackedRegion(path, offset, size, sha256, packed_format, storage="")` descriptors.
`PackedStore.from_sparse_manifest(...)` adapts existing sparse manifests without
modifying them. All regions require SHA-256. Named shard roots are containment
checked. The opened descriptor is checked against the original shard signature.
Only the requested aligned region is mapped; no entire shard is mapped or
decoded. SHA-256 is computed incrementally on every lease with cancellation
checks, and expert dispatch releases each mapping before the next step.

Files must be immutable while open. Stat/checksum checks are not protection
against a privileged concurrent writer changing mapped bytes after hashing.
Callbacks must not retain buffer views or native pointers. Escaped Python
buffer exports raise and keep their pool reservation until the views are
released and the pool reaps them. Close rejects unreleased exports. Untracked
raw native pointers remain the adapter's responsibility.

## Durable Cache and Recovery

`DurablePagedCache(directory, geometry, capacity_bytes=..., page_tokens=...,
read_budget=..., max_sessions=..., max_prefixes=..., journal_records=...,
sqlite_cache_bytes=...)` owns actual immutable payload BLOBs, not only metadata.
SQLite FULL-synchronous WAL atomically journals page data, session positions,
reference ownership, quota accounting, and a bounded audit journal. Payload
usage is a transactionally maintained counter. Indexed ref counts allow
unreferenced page deletion without scanning all session histories.

The cache directory is exclusively process-locked on Windows/POSIX. Threads
serialize cache write transactions, including their callbacks. Multiple
inflight requests may be admitted, but this implementation has one writer;
it does not claim parallel math execution or distributed scheduling.

`create_session(id, owner, prefix_id=None)` forks immutable prefix pages without
copying payloads. Owner checks gate mutation, checkpoint use, reset, and release.
Owner strings must come from the caller's authenticated authority; they are
not an authentication system. The prefix handle binds the entire causal chain,
not isolated matching token chunks. Appending allocates fresh immutable pages
instead of modifying a shared tail. Short pages are allowed; no tail coalescing
or automatic eviction of live sessions occurs.

Reopening the directory recovers committed sessions via SQLite WAL. Metadata,
shape, reference, and capacity checks run in the implemented recovery path;
payload and causal-chain hashes are checked on each page lease. This statement
describes source behavior; runtime verification remains outstanding.
`sessions(owner)` lists committed IDs, positions, and revisions. Construct a
facade with `create=False` to resume a recovered session.

`snapshot(destination)` uses SQLite's incremental backup API and fsync, then
atomic rename. Destination must be outside the live cache directory. POSIX
also fsyncs the containing directory; Windows directory-entry durability is
OS-owned. `restore_snapshot(source)` requires an empty inactive cache, checks
the layout contract and budgets, and copies records within one WAL transaction.
It never replaces an open database file. Session-level `restore_session` swaps
page references atomically to a pinned prefix, enabling generation rollback.

The payload quota is not a hard disk quota. SQLite B-tree overhead, WAL growth
during a transaction, free database pages, backups, and filesystem behavior are
additional storage costs. WAL auto-checkpoint and journal-size settings control
normal retention but cannot guarantee physical storage availability. I/O errors
propagate; applications must treat an OS/SQLite commit error as potentially
ambiguous and reopen/query durable revision before retrying non-idempotently.

## Remaining Integration Work

`K3CompressedStateCodec`, `K3CompressedPlanner`, and
`bind_compressed_k3_session` provide a separate concrete binding for the new
`K3Execution` compressed-MLA layout. Recovery frames bind configuration, weight
identity, occupied position, each MLA latent/rotary row, and complete KDA state
and convolution histories. The planner reconstructs actual native state and
performs token-to-logit calls before durable commit. A fresh executor factory is
required; its template and expert provider remain caller-owned. The older
`K3HybridSnapshotCodec` still targets expanded head KV and is not interchangeable.

`checkpoint_interval=1` retains the original fixed-frame codec and identity.
An interval greater than one enables a separate `OAK3C002` variable-frame
identity binding the interval. Each token keeps its MLA rows; complete KDA
matrices and Q/K/V histories are serialized at interval boundaries and always
on the final token of each dispatch. Recovery checks every frame's checksum,
position and exact extent and rejects a prefix lacking a terminal checkpoint.
Only whole committed dispatch endpoints are exposed as prefix handles, so no
token replay or guessed recurrent deltas are needed. Single-token dispatches
still serialize full recurrence; savings apply to multi-token dispatches.
Old/new contracts cannot share a cache directory. Admission/read budgets still
use the maximum full-frame size, while payload quota charges actual bytes.

`encode_model_step` borrows immutable MLA pages and recurrent arrays under the
exact decoder's private operation guard, returning owned bytes without making
a full-prefix snapshot per token. This is deliberately coupled to the current
`K3Execution` internals, not a portable codec ABI. Recovery still reconstructs
the complete occupied MLA prefix. Scratch admission reserves capacity-sized
reconstruction copies, full frames, logits and bounded finite-check buffers in
addition to caller-supplied native executor memory.

`K3PackedExpertProvider(config, store, max_expert_bytes=...)` implements the
existing decoder's selected-expert lease contract. Store region keys are exact
decoder expert tensor names from `tensor_shapes(config)`. Six read-only uint8
packed/scale tables are mapped per selected expert, checked against region
sizes/formats and checksums, and released after synchronous native use. Its
identity binds the tensor inventory and content digests; aligned simultaneous
map extents must fit the store budget. Pass this provider as `expert_provider`
to both the template and fresh factory's `K3Execution`; supply only non-expert
resident tensors. The provider/store remain caller-owned. Optional provider
`cancelled` checks occur during leases/hashing, not inside native kernels.
Escaped arrays retain map reservations and raise on lease release. Concurrent
map contention fails admission rather than waiting.

These edits have not been exercised. Publisher-exact FLA/BF16/quantization
semantics, native parity and crash recovery remain unverified. No throughput
or complete publisher-decoder support is claimed. Native reservations remain
caller-supplied; these pools do not enforce process RSS. No tests, validation
imports, compilation, builds or benchmarks ran in this code-only pass.

- Supply additional architecture-specific native planners and measured scratch estimators.
- Wire the execution factory to this opt-in facade; no shared API/CI edits here.
- Release coordinator snapshot pins explicitly; generation rollback is not
  automatic across separate successful `forward` calls without a checkpoint.
- Add optimized recurrent rolling state, architecture-specific sparse codecs,
  async/device transfer synchronization, and parallel/distributed scheduling.
- Add cancellation during SQLite backup and native/device work; cancellation
  currently occurs during cache-writer waits, at callback/stream/hash boundaries,
  and before commit, not inside a running native kernel or SQLite commit.
- Run the included `test_contracts.py` harness and add crash/power-loss,
  concurrency, native adapter, and allocation-pressure coverage in a later pass.
