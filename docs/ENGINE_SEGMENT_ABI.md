# Segment/Edge ABI v1

This is an OpenAgent clean-room ABI, **not upstream ABI compatible**. No
upstream compatibility has been verified. It was implemented from the local
`olmoe_segments.py` and `segment_snapshot.py` contracts, without copying an
upstream interface or implementation.

The compiled C bridge provides lifecycle and bounded transfers. Its callback
path does not implement native model kernels: attention, routing, expert math,
embeddings, final head, and snapshot validation still run in Python/NumPy.
`SegmentBridge.native_range_execution` and `upstream_abi_compatible` are false.
There is no speedup claim, automatic compilation, or silent Python fallback.

An independent native FP32 RMSNorm primitive is available in `segment_math.c`
and `.h` through `NativeSegmentMath`. It performs its own numerical computation
in C without Python callbacks. It is not wired into the callback model path
and is not a full native Segment/Edge decoder.

## Callback-Free OLMoE Execution

The separate `native/segment_executor.c` and `.h` implement OLMoE numerical
execution in C. `native_segment_executor.py` owns the ctypes descriptors;
`native_olmoe_segments.py` loads checkpoint-backed ranges and edges. This is
not the callback bridge described below and is not upstream ABI compatible.

```text
Token IDs -> C embeddings -> C layer range -> C final norm/head -> logits
                                |
                         staged KV cache
                                |
                       publish on success
```

The resident path decodes INT8-row experts to admitted FP32 banks at load
time. Its full-range decoder invokes one C transaction through the final
head. The leased path in `native_leased_olmoe_segments.py` retains packed
INT8-row or Q8_0 experts in bounded storage: Python coordinates leases, hashes and
transactions; C performs attention, routing, packed expert products and
residuals. No Python numerical callback supplies transformer results.

Leased execution copies KV state before work and publishes it only after all
layers, the head, and cancellation checks succeed. Cancellation is checked
before each layer and each expert lease, and before the final head. It cannot
interrupt a C kernel already running. Failures preserve the prior position
and KV state; expert leases are released on both success and failure.

Resident and leased range composition are available. The native interfaces
currently target OLMoE with INT8-row or Q8_0 storage, not every model or packed format.
The Python snapshot and lifecycle adapters remain Python code. Network edge
transport, upstream binary compatibility and non-Windows certification are
not provided by this implementation.

### Direct Q8_0 Execution

Q8_0 manifests select the additive v1 exports `oa_executor_swiglu_q8_0` and
`oa_executor_apply_q8_0`. Existing INT8 exports and defaults are unchanged.
Libraries without either Q8 export reject Q8 execution before checkpoint
loading; there is no decoded-bank or Python arithmetic fallback.

Each projection uses row-major 34-byte blocks: a little-endian binary16 scale
and 32 signed INT8 codes. Both projection widths must be multiples of 32.
C decodes each scale from its bit fields, performs block dot products directly,
and retains only activation/output scratch. Finite negative and zero scales,
subnormal scales and code -128 are accepted; nonfinite scales, wrong payload
lengths, invalid routes and FP32 overflow reject without publishing outputs.
There is no full-weight dequantization bank.

This path uses double block-dot accumulation with FP32 projection outputs and
SwiGLU intermediates. Its numeric class is
`c-fp32-dense-double-block-dot-native-q8_0-leased-v1`, distinct from INT8-row
execution. Range and decoder snapshots bind that class, exact selected
expert hashes and loaded dense-weight identity. Mixed-format compositions
are rejected by exact-manifest checks. This is not upstream quantizer byte
parity, a GGUF reader, or a kernel throughput claim.

Compiled tests compare direct blocks at H32/I32 and H64/I96 with an independent
scalar decoder, exercise signed/subnormal scales and malformed blocks, and
compare complete and composed Q8 OLMoE token-to-logit execution with the
independent Python model. They also cover binary snapshot resume and final-head
rollback. The operator CLI remains documented as INT8-only for now.

The local public Q8 manifest also completed a full 16-layer C leased run,
matching the independent BF16 continuation `[19, 5043, 374, 18207]`. It used
32-token capacity and a 256 MiB expert-cache budget. Observed load time was
2.456 s, first-token time after loading was 75.057 s, and total time including
loading was 90.730 s. Dense weights occupied 1,082,654,720 bytes and packed
cache occupancy was 267,386,880 bytes. This single short run establishes neither
broad quality nor controlled comparative performance; these are not RSS figures.

### Leased Composition Ownership

`NativeLeasedOlmoeSegmentEngine.compose([left, right, ...])` requires distinct,
ordered, adjacent, open leased engines with equal checkpoint configurations
and exact manifests. It shares their immutable dense buffers and pins their
expert stores; it does not copy expert banks or adopt existing source sessions.
Source close is rejected while a composition remains live. Close composed
sessions first, then the composition, then its sources. Nested compositions
follow the same ownership rules.

The default resident admission is the sum of shared dense weights. Native
transfer, staging and snapshot limits default to the smallest source limits;
optional overrides are validated explicitly. Context capacity cannot exceed
the smallest source capacity. Expert cache capacity is reported as the sum
of source caches, not a new allocation or a process-RSS limit.

```python
with NativeLeasedOlmoeSegmentEngine.compose([left, right]) as engine:
    with engine.create_decoder_session(edge) as session:
        logits = session.forward([1, 2, 3])
        saved = session.snapshot_state()
```

A decoder requires complete layer coverage and matching edge configuration.
Each call stages all composed KV buffers once, executes ranges in order, runs
the final head, then publishes position and KV together. A late store failure,
kernel error, head failure or cancellation leaves the previous state intact.
The composition identity hashes the ordered constituent execution identities;
decoder snapshots additionally bind the actual edge-weight identity. A
different partition has a different snapshot identity, even when it represents
the same model. No implicit snapshot migration is performed.

### Native-State Binary Snapshots

`native_segment_snapshot.py` provides `write_native_snapshot(session, stream)`
and `restore_native_snapshot(session, stream)` for resident or leased native
range sessions and full decoder sessions, including composed ranges. Python
serializes the owned native buffers; this is not a C serializer or an upstream
snapshot format.

```python
with open("run.snapshot", "wb") as output:
    write_native_snapshot(session, output)
with open("run.snapshot", "rb") as source:
    restore_native_snapshot(session, source)
```

The wire format is eight-byte `OANAT` version-1 magic, a little-endian uint32
JSON-header length, canonical JSON metadata, active-prefix little-endian FP32
keys followed by values, and SHA-256 over the preceding bytes. The header is
limited to 64 KiB and binds the numeric class, exact range identity, capacity
and position. Decoder metadata also binds the loaded edge-weight identity.
The checksum detects corruption; it is not authentication against an attacker
who can rewrite the snapshot and recompute its checksum.

Writing holds session ownership locks and streams active KV without calling
`snapshot_state` or copying the full cache. It validates finite values and
wire size before emitting data. Short writes are supported. Restore rejects
wrong identity, dimensions, position, nonfinite values, truncation, checksum
failure and trailing bytes before publishing state. It admits both the wire
length and active payload plus full-cache restore staging against `max_bytes`
and the engine snapshot budget. This bounds state buffers, not process RSS or
all temporary Python allocation. Restored inactive KV is zeroed.

Compiled tests exercise composed decoder resume, corrupt/truncated payloads,
identity mismatch, insufficient budgets, empty sessions and short-write
streams. Restoring a failed snapshot leaves the destination KV and position
unchanged. These snapshots are local persistence, not a network protocol.

Build the executor separately in an x64 Visual Studio developer terminal:

```powershell
cl /nologo /W4 /WX /LD openagent/adaptive_engine/native/segment_executor.c /link /OUT:segment_executor.dll
```

Run its differential and transaction tests:

```powershell
./.venv/Scripts/python.exe -m unittest tests.test_native_segment_executor tests.test_native_segment_adversarial tests.test_native_olmoe_segments tests.test_native_leased_olmoe_segments -v
```

These tests compile the actual C source and compare outputs and KV state with
independent NumPy paths. They also cover malformed descriptors, late failures,
checksum corruption, snapshot identity, cancellation and ownership. A missing
compiler skips compilation-dependent tests; a skip is not execution proof.

### Public Checkpoint Smoke Test

The integration validation run exercised the actual leased C execution path on the
public OLMoE checkpoint across all 16 layers, with a 256 MiB packed-expert
cache and a context capacity of 32 tokens. The four generated token IDs
`19, 5043, 374, 18207` matched the independent BF16 oracle continuation.

| Measurement | Observed value |
| --- | --- |
| Load time | 3.020 s |
| Total elapsed time, including load | 32.123 s |
| Resident dense-weight bytes | 1,082,654,720 |
| Packed cache bytes used | 264,929,280 |

This is one short checkpoint smoke test, not a broad model-quality evaluation
or a controlled speed comparison. The reported byte counts are dense weights
and packed cache occupancy, not total process memory. It validates the full
leased path; composed split ranges are covered separately by the compiled
differential and rollback tests above.

## Callback Bridge Build

Sources live in `openagent/adaptive_engine/native/segment_bridge.c` and `.h`;
the ctypes adapter is `openagent/adaptive_engine/segment_abi.py`.
Use a compiler matching the Python process architecture. No Python headers
or linking against Python are required.

In an x64 Visual Studio developer terminal:

```powershell
cl /nologo /W4 /WX /LD openagent/adaptive_engine/native/segment_bridge.c openagent/adaptive_engine/native/segment_math.c /link /OUT:segment_bridge.dll
```

On a POSIX host with GCC or Clang:

```sh
cc -std=c99 -Wall -Wextra -Werror -fPIC -shared \
  openagent/adaptive_engine/native/segment_bridge.c \
  openagent/adaptive_engine/native/segment_math.c -lm -o segment_bridge.so
```

The test module discovers an installed compiler (including MSVC through
`vswhere`), builds into a temporary directory, and tests the actual library:

```powershell
./.venv/Scripts/python.exe -m unittest tests.test_segment_abi tests.test_olmoe_segments tests.test_segment_snapshot -v
```

Missing compiler tools explicitly skip with "Native Segment ABI unavailable".
Compilation errors fail tests rather than masquerading as unavailability.
Windows test DLL temporary directories are retained because loaded DLLs
cannot be unlinked. They contain build artifacts, not model data.
Portable C source is provided; non-Windows execution remains unverified.

## C Contract

The header is authoritative. All exports use C linkage and Windows `cdecl`.
There are no exposed structs or compiler-sized enums in function signatures:
versions/operations/kinds use uint32, status uses int32, and handles, rows,
and byte lengths use uint64. Call `oa_segment_version()` before registration;
v1 registration requires exact version 1. No ABI negotiation is implied.

`oa_segment_register` binds a callback and borrowed context to a new handle.
Kinds are engine=1, session=2, edge=3. Only sessions have parents, and their
the runtime host must be a live engine. Registration does not construct model weights;
the Python adapter constructs sessions using the registered engine. It also
checks that a session belongs to that exact Python engine.

The library has 256 simultaneous slots. Nonzero IDs increase monotonically,
never recycle, and refuse allocation after uint64 exhaustion. IDs are local
to one loaded library lifetime; never store or transfer them between processes
or separate library instances. Release permits slot reuse, not ID reuse.

`oa_segment_call` accepts operation, row count, input pointer/byte length,
output pointer/capacity, and a required written-length pointer. Input and
output are independently bounded by the registered max_bytes and SIZE_MAX.
Zero-length pointers may be null. Buffer addresses must be valid for their
declared lengths and must not alias handle metadata or the written pointer.
Length checking cannot make arbitrary pointers safe; this is not a sandbox.

| Operation | Kind | Input | Output |
| --- | --- | --- | --- |
| forward=2 | session | rows x hidden_size LE float32 | same shape |
| embed=3 | edge | rows LE signed int64 token IDs | rows x hidden_size LE float32 |
| head=4 | edge | rows x hidden_size LE float32 | rows x vocab_size LE float32 |
| reset=5 | session | none, rows=0 | none, capacity=0 |
| snapshot=6 | session | none, rows=0 | existing OASEG v2 bytes |
| restore=7 | session | complete OASEG v2 bytes, rows=0 | none, capacity=0 |

Tensor buffers are tightly packed C-order; floats must be finite and token
IDs in vocabulary. These semantic checks are performed by the Python callback,
not native kernels. Tensor operations require positive rows and exact input
length. Output capacity must be sufficient before range math starts. There
is no size-query call which accidentally advances a session. Snapshot exports
are bounded while writing; undersized capacity returns LIMIT without mutation.
Native preflight rejects zero-row/empty-buffer tensor calls, nonempty reset
envelopes, snapshot rows/input, and restore rows/output or empty input before
invoking callbacks. Tensor width and snapshot content validation remain Python
callback responsibilities.

`oa_segment_release` invokes close=1 synchronously with empty buffers.
Live child handles prevent engine close. Failure keeps the handle and callback
alive for retry; successful release invalidates the handle and removes its
parent's child count. Repeat release returns HANDLE, not success.

| Status | Meaning |
| --- | --- |
| OK=0 | success |
| INVALID=1 | invalid arguments, operation/kind, or backend validation |
| VERSION=2 | registration version mismatch |
| HANDLE=3 | absent/stale handle or wrong session parent kind |
| BUSY=4 | reentrant handle use or engine has live children |
| LIMIT=5 | transfer capacity, registry capacity, or ID exhaustion |
| CALLBACK=6 | callback exception or invalid callback result |
| CANCELLED=7 | Python range cancellation |

Written length starts at zero and is published only on success. Callbacks
must report a known status, never unwind through C, and never claim more bytes
than capacity. On any failure the caller must ignore the output buffer. The
native bridge cannot undo side effects of a malicious/custom callback.

## Ownership And Serialization

Serialize **all** calls per loaded library, across threads, including register
and release. The C registry is intentionally not internally thread-safe.
Per-handle busy flags guard synchronous callback reentry, not concurrent calls.
The adapter shares an RLock keyed by library handle across its instances and
rejects callback reentry into active adapter calls. Raw ctypes users must take
`bridge.lock` too and must not release handles owned by the adapter directly.

Callbacks, context, and buffers are borrowed, synchronous, and never retained
by C after the call except the registered callback/context. The adapter pins
itself and its callbacks until every handle is successfully released, including
across garbage collection. Forgetting release therefore leaks resources by
design instead of leaving C with a dangling function pointer. Release sessions
before engines. Registration success transfers backend close responsibility;
failure leaves that responsibility with the caller. Do not register the same
backend in multiple adapters, create unregistered sessions, or mutate/close a
registered backend outside its adapter.
Duplicate backend registration is rejected across all adapters and library
instances in this Python process. Failed release retains that ownership;
successful release removes it. Direct raw-C registration cannot enforce Python
object ownership.

`bridge.close()` releases all registered sessions before engines and edges;
no manual handle bookkeeping is needed. Successful close is terminal and
idempotent: later registration or context entry is rejected. Individual
`release(handle)` remains available for earlier cleanup.

If release fails, close continues releasing unrelated handles. A parent with
a failed live child remains protected by the C child count (BUSY), without
invoking its backend close. Close raises `SegmentABICloseError`, a
`SegmentABIError` subclass whose `failures` contains `(handle, error)` pairs
and whose `remaining_handles` identifies all retained handles. The aggregate
status comes from the first failure; each failure retains its original cause.
Live callbacks remain pinned. Fix the cause and retry `bridge.close()`; a
failed attempt does not mark the bridge closed.

`with SegmentBridge(...) as bridge:` runs close on exit, including when setup
or the body raises. A cleanup failure without a body exception propagates.
With a body exception, exit preserves that same exception and adds a Python
exception note describing cleanup failure and retry; it does not suppress or
replace the original error. Retain the bridge reference to retry failed cleanup.

Python failures become SegmentABIError with a numeric status and the original
exception as its cause. Cancellation is supplied to `hidden(..., cancelled=)`
and checked at the existing range boundaries; it is not asynchronous native
interruption. Normal validation, kernel exceptions, and cancellation preserve
the existing transactional KV semantics. Process crashes and allocation failures
after backend commit are not a transaction rollback mechanism.

Snapshots reuse the existing bounded, checksummed format and identity checks,
including checkpoint fingerprint, layer range, context capacity, and NumPy
numeric class. Restore's max_bytes also bounds its existing KV staging check.
OASEG v2 also binds `execution_identity`, including expert-region SHA/size
and hashes of loaded dense-weight bytes, so matching metadata with different
packed expert or loaded dense weights cannot restore.
Old portable v1 snapshots are rejected. The C lifecycle ABI remains version 1;
its version is independent of the portable snapshot format.
The transfer bound is not a total process-memory or resident-weight budget:
ctypes buffers, returned bytes, and NumPy staging can each consume bounded
copies, while model/KV allocation is managed by the existing backend.

## Native RMSNorm

`oa_segment_rmsnorm_f32` accepts host-endian tightly packed FP32 input
`[rows, width]`, FP32 weight `[width]`, and a separate output of the input
shape. All buffers must be valid and mutually non-overlapping. Dimensions must
be positive; overflow-safe size checks bound each buffer by `max_bytes` and
`SIZE_MAX` before reading. Epsilon must be positive and finite. Non-finite
inputs/weights or FP32 output overflow return INVALID without writing output;
insufficient bounds return LIMIT. Buffers are borrowed only for the call.

The kernel accumulates squared values and computes the square root in double
precision, then rounds output to FP32. This avoids FP32 squared-input overflow
but is not bitwise equivalent to NumPy FP32 reduction. It preflights output
representability before a second computation pass writes results. No SIMD,
performance claim, model loading, attention, routing, or KV state is provided.

`NativeSegmentMath(library, max_bytes=...).rmsnorm(hidden, weight, epsilon=...)`
validates FP32 shapes, bounds before making contiguous copies, and allocates
separate output. C performs finite-value validation and the numerical work.
Missing exports fail explicitly; the wrapper does not fall back to NumPy.
The primitive is stateless and does not use the bridge registry or its lock.

Compiled tests compare varied dimensions and strided inputs against independent
NumPy double and FP32 references, exercise zeros and extreme finite values,
and verify unchanged output on raw-C invalid dimensions, overflow, invalid
epsilon, non-finite data, and insufficient capacity. Windows validation used
installed MSVC with `/W4 /WX`; non-Windows execution remains unverified.

## Python Example

```python
from openagent.adaptive_engine.olmoe_segments import OlmoeEdge, OlmoeSegmentEngine
from openagent.adaptive_engine.segment_abi import SegmentBridge

with SegmentBridge("segment_bridge.dll", max_bytes=64 * 1024 * 1024) as bridge:
    edge = bridge.register(OlmoeEdge(model_directory))
    engine = bridge.register(OlmoeSegmentEngine(
        model_directory, manifest, 0, 2, expert_cache_bytes=cache_budget))
    session = bridge.create_session(engine)
    hidden = bridge.embeddings(edge, [1, 2])
    hidden = bridge.hidden(session, hidden)
    logits = bridge.hidden(edge, hidden, final_head=True)
    snapshot = bridge.snapshot(session)
    bridge.reset(session)
    bridge.restore(session, snapshot)
```

The example range is illustrative and must fit the checkpoint; running the
final head after a partial range is not a complete model inference. Compose
all model ranges for full logits. Context exit cleans up already registered
handles if later construction fails. Registration failure still leaves cleanup
of the unregistered backend with its caller. Lifecycle convenience does not
change the bridge's scope: model math is Python/NumPy, not native kernels,
and upstream ABI compatibility remains unverified and explicitly false.
