# Phase 1 source implementation checklist

> Verification correction, 2026-10-19: a reported set of seven fixes was
> checked against Git index, working tree, stash, and live source. No changes
> are staged and none of the seven implementations is present. The universal
> engine and task-creation defects remain open. The tracked authoritative queue
> is `PHASE_1_ACTIVE_TODO.md`. This detailed checklist now lives in the tracked
> Phase 1 audit directory.

For the current remaining work, acceptance gates, and next pass, see
[PHASE_1_ACTIVE_TODO.md](../../../PHASE_1_ACTIVE_TODO.md). This file retains the detailed
requirements and audit history.

This checklist defines the completion criteria for the Colibri-inspired
OpenAgent engine source implementation. Existing implementations should be retained and
extended. An unchecked item means it needs source review and, where necessary,
implementation; it does not mean that its entire implementation is absent.

## Current deep audit — 2026-10-19

The current source was reconciled against the active queue across route
admission, inventories, media, speculation, distributed execution, K3,
formats/devices, public ABI, recovery, evidence tooling, serving, Colibri
restrictions, and the universal session layer. This was a source-only audit;
no tests, imports, builds, benchmarks, device probes, or model runs were used.

### Deep re-audit revision 2 — newest working tree

The latest working-tree delta was inspected file by file. The seven changed
implementation/check files add personal-assistant goal/task update and delete
workflows in the Python manager/routes, React UI, integration fixture, browser
smoke script, and focused checks. No file in `openagent/adaptive_engine` and no
adaptive-engine route changed. No untracked closure report, receipt, matrix,
or implementation file was added.

This means the new delta supplies no additional loader-to-logits trace,
inventory consumer, media/speculative/distributed closure, format/device
binding, ABI/recovery mapping, evidence receipt, or inference-serving lifecycle
closure. It is not credited to P02–P15.

Direct source inspection reconfirmed `phase1_closed=False`, false execution,
parity and platform flags, unverified/not-captured evidence with no receipt,
and fail-closed rejection of candidate/restricted/absent execution contracts.
The two previously claimed generated reports and `unified_execution.py` are
still absent; `universal_engine.py` is present. Accordingly, the P01–P15 table
below is unchanged and P15 remains OPEN.

### Confirmed bugs from the full static audit

The audit found source defects in addition to incomplete checklist evidence:

1. The universal engine is constructed, but no production family adapter is
   registered and no route/service caller opens a universal model. It is not a
   reachable end-to-end engine path.
2. Universal admission checks only checkpoint `model_type`; it does not bind
   the other five route fields to an admitted source-scope tuple. A supplied
   candidate backend/format route can reach an opener.
3. Warm-session keys omit opener options, allowing calls with different
   device, placement, cache, isolation, or family options to alias one owner.
4. Cancellation after `generate()` mutation and cancellation during streams
   have no universal rollback; streams also dispatch without a pre-cancel
   check. Warm state can advance for an unpublished request.
5. Universal snapshot receipts have no shared identity/epoch envelope and
   restore performs no stale/foreign validation.
6. Engine close paths remove sessions before owner close succeeds, while the
   session remains logically open on failure. Failed cleanup becomes
   unreachable through the engine instead of remaining retryable.
7. A stream holds the session lock until iterator teardown; an abandoned
   iterator has no engine-owned lease/timeout and can block later operations.
8. Outside the Phase 1 engine, assistant task creation accepts dangling or
   cross-project goal IDs because creation has no project-aware goal check or
   database foreign key; update validation is a separate transaction.
9. The new assistant mutations commit before audit persistence, so audit
   failure can produce a caller-visible error after the mutation committed.
10. Assistant mutation paths accept arbitrary goal status/horizon and task
    status/priority values; neither manager validation nor database checks
    preserve the state vocabulary used by queries and coaching logic.

The first seven defects directly reinforce the open P02/P05/P10/P12/P14/P15
statuses. The last three are product defects in the newest changed files and are
tracked separately from Phase 1 closure.

### Latest changed-tree delta

The newest uncommitted changes were reviewed separately. They implement
assistant goal/task update and deletion in `openagent/assistant.py` and
`openagent/routes/assistant_routes.py`, with frontend, smoke, integration
support, and route-test changes. They contain no Phase 1 adaptive-engine route,
loader, execution, tensor inventory, media, speculative, distributed, format,
device, ABI, recovery, evidence, or model-serving implementation. They
therefore do not change any P02–P15 status below.

The decisive gates remain unchanged in `source_scope.py` and
`wholemodel_validation/runner.py`: `phase1_closed=False`, execution and
platform certification flags are false, evidence is not captured, receipts
are absent, and candidate-unverified routes are rejected by evidence tooling.
The claimed generated closure reports remain absent. This delta was also
audited without executing tests, imports, builds, benchmarks, device probes,
compilers, or models.

| Requirement | Audited status | Remaining closure boundary |
| --- | --- | --- |
| P01 | DONE | None identified. |
| P02 | IN PROGRESS | Every required canonical tuple must have a complete loader-to-tooling trace or an enforced restriction/rejection. Candidate rows remain. |
| P03 | BOUNDED CONTRACT DONE; FAMILY CLOSURE OPEN | Finish all-family optional/shared/MTP/expert/media consumer ownership and bypass review. |
| P04 | IN PROGRESS | Complete every media/caller ordering, geometry, ownership, temporal, cancellation, recovery and cleanup chain. |
| P05 | IN PROGRESS | Complete all-caller speculative transition, dual-RNG, restore and publication guarantees. |
| P06 | IN PROGRESS | Complete all distributed banks, shard layouts, operation failures, process loss and recovery consensus. |
| P07 | IN PROGRESS | Complete K3 temporal/video, continuation and serving integration. |
| P08 | DONE (SOURCE SCHEMA) | Runtime observations remain P07/P13 work. |
| P09 | IN PROGRESS | Bind every admitted persisted format to its actual consumer and verify conversion/scale/rotation semantics. |
| P10 | IN PROGRESS | Produce actual placement and operation-level no-host-fallback evidence for admitted backends. |
| P11 | IN PROGRESS | Finish family ABI symbols/ranges/schemas/callback lifecycle and native compile/execute evidence. |
| P12 | IN PROGRESS | Finish all-family physical-state ownership, stale/foreign restore, cancellation/reentry and distributed rollback. |
| P13 | IN PROGRESS / A10 BLOCKED-ENV | Capture pinned runtime, cache, placement, native ABI and numerical comparison receipts. |
| P14 | IN PROGRESS | Finish caller-by-caller streaming, backpressure, stops, media, recovery, warm sessions and compatibility scope. |
| P15 | OPEN | Reconcile every applicable P02–P14 row and attach mandatory evidence before closure. |

The source remains intentionally fail-closed: candidate rows are not admitted;
evidence obligations declare `evidence_state='not-captured'`,
`completion_claim=False`, and `receipt_present=False`; the catalog declares
`phase1_closed=False`. The shared implementation present is
`universal_engine.py`; `unified_execution.py` is absent. The referenced
`phase1_closure_report.md` and `phase1_complete_route_summary.json` are also
absent from this checkout and cannot be counted as evidence.

**Audited conclusion:** Phase 1 and P15 remain **OPEN**. P01 and the P08 source
schema are done; P03 is done only for its bounded inventory contract; all
other P02–P14 packages remain partial or evidence-blocked as listed above.

## Current status update — 2026-10-18

Phase 1 and P15 remain **OPEN**. Recent changes added fail-closed P02/P03
boundaries, restricted unverified MTP/speculative paths, and tightened P04/P14
media staging cleanup in the publisher processor and native conditional CLI.
The latest full Python discovery on the current tree passes 1,682 tests with
95 skipped in 537.954 seconds. This run includes the final source regressions
and the `/api/agent-runs` route extraction fix. The immediately preceding full
run hit a Windows loopback `ConnectionAbortedError` in
`test_all_mutation_verbs_are_limited_before_dispatch`; its complete 18-test
module passed alone, and the subsequent full run passed. That transient socket
abort remains recorded as an environment/test-order observation.
A preceding discovery had one Windows loopback
`ConnectionAbortedError` in the commercial-security HTTP boundary test; the
test alone and its full 17-test module passed, and the clean full rerun passed.
This suite-level socket abort is recorded as a transient environment/test-order
observation, not silently treated as a code fix. The successful run includes K3 batch cleanup
hardening: rollback,
snapshot release, and batch close continue through all owned slots when a
cleanup exception cannot be formatted, while failed cleanup remains reachable
for retry. Three focused regressions cover these paths. The serving SSE writer
also now treats `ConnectionAbortedError` as a client disconnect and closes its
iterator without attempting a second error frame; its focused regression is
recorded separately. These are narrow bug fixes and do not close their broader
P12/P14 requirements.
Follow-up source fixes now reject configuration-only, unconsumed MTP claims in
the direct resident Qwen3.5/Qwen4/GLM5 inventory gate; the targeted regression
and all 58 source-scope inventory tests pass. The Qwen4 public native
Segment/Edge adapter and session paths now normalize cancellation through one
shared exact-0/1 guard before kernels/observers and before output/continuation
commit; its source-contract suite passes five tests. No C compiler is
available, so this ABI change has not been compiled. The successful full
discovery includes both follow-up regressions.
The post-change structural validator passes and classifies the 1,334
family/backend/format execution cases as 129 source-admitted, 797
candidate-unverified, 322 rejected, and 86 restricted across nine families.
It correctly reports `phase1_closed=false`, `executed=false`,
`numerical_parity_verified=false`, and `platform_certified=false`. The Python
sdist and wheel also build successfully with local installed build tools;
isolated build dependency resolution remains blocked because PyPI network access
is denied. Ruff F821 and scoped `git diff --check` pass for the changed Python,
C-source-contract, regression, and checklist files.
The selected-device E8 path also received a correctness fix: decoded weights
remain in the persisted codebook basis because the device consumer rotates
activations before both projections. This prevents a double transform; a
numerical regression covers the persisted-weight equation and signed finite
scales. CPU Torch tests pass, but this does not establish accelerator placement
or cross-platform model parity.
Inkling grouped continuation export/restore now forward cancellation through
component operations and roll back staged state on interrupted recovery.
The native publisher gateway now adopts its cancellation-aware detached K3
video frames directly, avoiding a second whole-frame copy in the public
borrowed-input constructor; the private adoption validates exact ownership,
immutability, geometry and timestamps.
The local Anthropic route now validates its required API-version header, and
trusted browser preflight admits the Anthropic auth/version headers only on
`/v1/messages`; its focused HTTP/route/lifecycle checks pass (20).
Publisher selected-video input, manifest loading, and processor metadata now
share overflow-safe FPS normalization; the media-input/route/lifecycle suites
pass (21).
Focused lifecycle/source-scope tests currently pass (28); distributed checkpoint,
tensor-parallel consensus, and continuation-lock tests pass (10); cache restore,
compound rollback and cleanup regressions pass (10); public factory cleanup
regressions pass (2); public-family checkpoint-binding regressions pass (4).
Combined focused lifecycle/source/inventory/cache/distributed/factory tests
previously passed (60). The focused source-inventory suite now passes (37
tests) after the latest route-map regressions; this is source regression
coverage, not end-to-end model
or platform evidence.
Latest P12 source work also protects K3 generation and generic file-cache
transport staging when initial descriptor close fails; the focused cleanup and
rollback suites pass 14 tests. These local-file fixes do not close distributed
transport loss or consensus recovery.
Latest P14 work adds bounded single-process FIFO admission for OpenAI/Anthropic
text JSON and SSE requests, overload/timeout 429 responses, a queue-wait
response header, provider-owned health/Prometheus counters, and a
service-unavailable response for unreadable server-side checkpoint
configuration. Local OpenAI and Anthropic SSE now emit protocol-specific
frames, cancellation and terminal/error events; OpenAI chat supports only
`stream_options.include_usage`. Priority scheduling, distributed admission,
and transport-loss recovery remain open. The capability and source-scope
catalogs were corrected on 2026-10-18 to match this limited source behavior;
this declaration does not establish full API compatibility.
Current exact matrix inspection materializes 1,334 family/backend/arithmetic/
source/persisted-format tuples: 800 are candidate-unverified, 322 are rejected,
86 are explicitly backend-restricted, and 126 exact local source routes are
source-admitted. These comprise 18 Mixtral dense FP32 routes (native, device,
and public-range across F32/F16/BF16 for each of its two catalogued model
variants), nine GLM-DSA dense FP32 routes (native, device, and public-range
across F32/F16/BF16), 21 Inkling publisher/text routes across native, device,
and public-range execution (publisher F8_E4M3 is scale-aware; text-only routes
admit only F32/F16/BF16), 24 OLMoE manifest-to-leased-consumer
routes (native INT8-row/Q4/Q5/Q8 and public-range INT8-row across F32/F16/BF16),
nine V4.1 BF16/FP8/FP4 local bank routes, and 42 Qwen3.5/Qwen4/GLM5 local
publisher-wrapper/text routes. Qwen publisher wrappers bind F32/F16/BF16 to
float32 and F8_E4M3 to scale-aware FP8 blocks; text-only paths bind only
F32/F16/BF16 to float32. Three additional GLM5 text routes bind those dense
formats to its public Segment/Edge service; Inkling's three corresponding text
routes are already included in its 21 rows. The public text ABI is explicitly
media-restricted, and `--public-runtime-library` selects it while pinning that
runtime with checkpoint artifacts. Other public-range Qwen/GLM rows and
generated tuples remain unverified and fail closed at exact route admission.
Source admission is not a runtime or parity claim.
The latest complete unittest discovery passes 1,645 tests with 95 optional
skips (562.345 seconds); the combined source-scope inventory, native checkpoint,
and family-source binding suites pass 97 tests with eight optional skips, and the Mixtral-native
regression group passes nine tests with eight optional skips. A fresh
structural audit validates all
nine families and all 1,334 rows, including source references; it continues to report
`phase1_closed=false`, `executed=false`, `numerical_parity_verified=false`,
and `platform_certified=false`.
The separate web pipeline also passes: 22 test files/142 tests, TypeScript
check, and production build. Vite reports a large Monaco code chunk (3.71 MB
minified); the build otherwise completes successfully. This frontend result is
recorded for repository-wide verification and does not count as engine P02–P15
closure or runtime evidence.
`validate_phase1_source` resolves declared owner references without importing
them. This is P02 source-path admission, not P13 runtime evidence or numerical
certification; other-family caller traces and all actual P13 receipts remain
open. A new AST regression confirms every public
execution-service factory in `execution_factories.py` reaches an exact
route/inventory gate or an explicit in-memory restriction through its local
wrapper chain; this does not cover every external CLI/gateway/recovery/tooling
caller or prove tensor consumers.

The 2026-10-18 P02 audit also found that exact local publisher/text source rows
for Qwen3.5, Qwen4, and GLM5 were missing from the matrix even though their
directory loaders and pre-payload inventory gates existed; valid local source
checkpoints could therefore fail closed before construction. Exact native and
device bindings now cover only those concrete loader format pairs. A related
P13 runner bug treated `verified-admitted` source rows as ineligible for
evidence binding; it now accepts source-admitted or candidate rows for review,
while rejecting restricted rows and continuing to require the row-specific
media adapter/report. No media evidence was added. Source scope, family binding,
ABI, and relevant format/admission regressions pass; Phase 1/P15 and runtime
evidence remain open.

The 2026-10-18 OLMoE route review found that the native leased CLI only gated
the public adapter path, and checkpoint preflight treated the source dtype as
the persisted expert format. Native leased construction now validates the
manifest-selected persisted format against exact P02 rows before opening its
kernel, retains the bounded checkpoint inventory, and rechecks it after weight
loading. The direct public adapter uses the same exact manifest-format binding
before ABI loading. Only the actual leased native formats and public INT8-row
path are source-admitted; candidate expert encodings remain unadmitted.

Every concrete Mixtral-derived public range adapter must declare its own exact
family/model identity; Qwen4 binds its pre-library gate to
`qwen4/qwen4_exp_text` rather than inheriting Mixtral's. Regressions cover the
concrete family map and reject subclasses with no explicit route.
The direct OLMoE public adapter now performs its own exact P02 checkpoint
inventory gate and verifies manifest architecture, source fingerprint, tensor
headers, layer geometry, and native INT8 payload schema before loading the ABI
library; it no longer relies on the CLI or deferred range-open path.
The direct K3/V4.1 publisher public adapter now independently enforces the
exact family/backend/arithmetic route and checkpoint-header inventory before
loading its native ABI, so callers cannot bypass the composition factory's
P02 gate. Its focused direct-construction regression and source-inventory
suite pass 49 tests together. This remains an admission boundary, not proof of
an admitted public route: the current matrix still has zero admitted tuples.
The Mixtral-derived public range adapters also retain that exact source
snapshot and recheck it around lazy Edge/range loads and service composition;
if a checkpoint changes during loading, the partial owner is closed or retained
for retry. K3/V4.1 payloads now carry their checkpoint root and immutable file
stamps; direct public composition compares those pins and configuration with
the requested model directory before ABI loading. Family-binding,
source-inventory, Qwen4 public-lifecycle and K3 focused suites pass 76 tests
with one optional skip. This narrows P02 source-binding gaps, not all loaders
or family paths.

Still outstanding: (1) exact per-family/backend/format/caller P02 traces and
family-wide P03 inventory-consumer reconciliation beyond the newly closed
resident-planner and selected-loader route-binding bypasses; (2) cross-family P04/P07
media/continuation and P14 caller lifecycle closure (the HTTP text queue is now
closed only for the documented synchronous local subset); (3) all-caller P05
speculative state/RNG/publication coverage; (4) every P06/P12 distributed bank,
operation, rollback and recovery-consensus path, with process/transport loss
still requiring group restart; (5) P09/P10 complete loader-consumer mapping,
converter coverage, runtime placement and no-fallback evidence; (6) P11
family-wide ABI and caller wiring review; and (7) P13 actual pinned observations,
cache/placement receipts and comparisons. Existing source catalogs, restrictions,
and focused tests do not satisfy those remaining obligations.

Selected-device E8 activation rotation has now been implemented for the shared
Mixtral/Qwen3.5/Qwen4 expert provider: exact persisted contract and aligned
geometry are checked, input and intermediate activations are transformed on the
current device with bounded scratch and cancellation checks, and the P02 format
route names both the decode consumer and rotation helper. Focused tests define
transform-orientation, budget-rejection, and pre-allocation-cancellation checks,
including NumPy-backed tensor math when Torch is unavailable. This
closes that specific P09/P10 implementation gap only; candidate routes remain
unadmitted, packed-manifest factory provenance restrictions remain, and no
accelerator placement or end-to-end model evidence is claimed.

The GLM5-next and Inkling multimodal evidence rows are also explicitly
restricted: no row-specific full-media observation adapter/report exists yet.
Their source rows now carry the concrete reason and the comparison runner rejects
those media tuples before adapter import. This is an enforced evidence-path
restriction, not media parity or a claim that the underlying serving media
execution is unsupported. P13/P15 remain open.

## Colibri reference comparison — source audit, 2026-10-16

**Phase 1 is not complete. Full Colibri engine equivalence is not implemented.**
There is substantial engine source, but component presence, a generated matrix,
or an explicit rejection does not establish a working end-to-end combination.
This section is the current comparison roll-up; dated notes below remain the
record of narrower changes. No engine execution or verification was performed
for this audit. Only this checklist is changed by this comparison.

### Reference boundary

- The local `reference/colibri` checkout reports revision
  `ce370e87d7b623d7759b52ec2007d75fc5b0e87e`. Its
  `c/family_registry.py` declares GLM, GLM53, Inkling, Kimi, OLMoE,
  Qwen36, Qwen38, DeepSeek V4 and DeepSeek V4.1 family descriptors.
  [Pinned upstream reference](https://github.com/JustVugg/colibri/tree/ce370e87d7b623d7759b52ec2007d75fc5b0e87e).
- The current upstream README, read separately on 2026-10-16, advertises
  thirteen engines, including MiMo, Qwen-Image-2.1, Laya and
  GLiNER2.5-Decide, plus Clef on the Qwen engine. These are an expanded
  comparison target, not automatically original Phase 1 requirements.
  Upstream `main` is mutable; this audit does not assign it an unverified
  commit hash. [Current upstream README](https://github.com/JustVugg/colibri/blob/main/README.md).
- Publisher display names are not architecture admission. OpenAgent's
  `qwen35`, `qwen4`, `glm5-next` and `glm-dsa` routes must be matched by exact
  model type, architecture, publisher revision, geometry and tensor contract;
  similarly named Colibri engines do not prove that mapping.
- Colibri itself has per-feature limits: its documented Segment/Edge adapters
  advertise CPU, require explicit registration, and are separate from ordinary
  serving binaries. Its HTTP API, media, tools and log-probability support also
  have family-specific limits. Compare each actual contract rather than
  assuming every engine exposes every feature.
  [Segment](https://github.com/JustVugg/colibri/blob/main/docs/segment-runtime.md),
  [Edge](https://github.com/JustVugg/colibri/blob/main/docs/edge-runtime.md),
  [HTTP API](https://github.com/JustVugg/colibri/blob/main/docs/api.md).

### Implemented source versus remaining engine work

Here **implemented source** means identifiable code exists, not that the whole
row is complete or numerically/platform verified. **Partial** means integration
or source-contract closure remains. **Restricted** means a concrete unavailable
route is rejected; it is not implementation of that feature.

| Area | Implemented source / evidence owners | Still missing or unresolved | Checklist |
| --- | --- | --- | --- |
| Family execution and publisher mapping | Nine-family catalog in `openagent/adaptive_engine/source_scope.py`; family-specific independent/native/device decoders, publisher contracts and resident inventory planners. OLMoE has its own native/leased route. | Complete exact family/model/backend/arithmetic/source-format/persisted-format chains through loader, cache, ABI, factory, CLI, gateway, recovery and tooling. Catalog rows remain candidates: `source_matrix_rows(admitted_only=True)` is designed to exclude them all until exact-row closure. This does not mean every existing local decoder is disabled. | P02/P03 partial |
| DeepSeek V4.1 | `v41_execution/` contains text, vision, publisher MTP, DSpark/speculative and device owners; `tensor_parallel.py`, BF16/packed banks and distributed service/recovery code exist. | Every speculative caller's accept/reject/bonus/stop/EOS/context/cancellation/publication lifecycle; each distributed bank's layout, failure and recovery consensus. Distributed V4.1 factory/CLI routes currently fail closed on unverified exact tuples. Local V4.1 remains a separate route. | P04–P06/P12 partial |
| K3 | `k3_execution/` has MLA/KDA/conv/media, temporal provenance, batch transactions with explicitly releasable snapshot receipts, and exact-admission code; `k3_generation_recovery.py` owns generation recovery. | Remaining temporal/media and serving integration, complete continuation ownership and source-bound operation observations. Metal and shared-executor FLA routes are explicitly rejected; specialized FLA code presence does not make that route admitted. Batch media and nonuniform video timestamps remain restricted. | P07 partial; P08 bounded source DONE |
| Other family media and auxiliary banks | Conditional loaders and Qwen3.5/Qwen4/GLM5/Inkling wrappers; selected image/video paths; Inkling MTP owner. | Cross-family placeholder ordering, geometry/positions, borrowed ownership, cancellation and recovery caller closure. Mixed-modality publisher requests remain restricted. Qwen3.5/Qwen4/GLM5 publisher MTP tensors and audio routes without an execution owner are rejected. | P03/P04 partial beyond bounded inventories |
| Checkpoints and packed streaming | `publisher_checkpoint/`, `resident_family_inventory.py`, `expert_packing.py`, `expert_runtime.py`, streamed projections and original/packed expert providers. Header, scale, padding, extent and publication cleanup checks exist. | Reconcile every persisted loader with the real family/backend arithmetic consumer, including scale conventions, packing and transformed activations. E8 selected-device rotation now exists for the shared Mixtral/Qwen3.5/Qwen4 provider, but tuple provenance and runtime placement/evidence remain open. Explicit format restrictions include `int4-rans256-g0` and `fp8-e4m3-b128-ue8m0`; similarly named owned formats are not Colibri container compatibility. | P09 partial |
| Accelerator decoding | Family Torch decoders, device expert/projection/head owners, precision/no-fallback policy and backend discovery/admission code. | Complete operation and retained-state chains for every advertised tuple; truthful factory placement and operation-level no-fallback observation contracts. Backend registration alone supplies no CUDA/HIP/Metal/Vulkan/OpenVINO equivalence or hardware certification. | P10 partial |
| Native Segment/Edge | `native/segment_edge_compat.*`, family adapters/ranges and public text owners; Qwen4 public C model/session/adapters; K3/V4.1 publisher text adapters. | Family-by-family symbols/version, range residency, activation/head/cache schemas, callbacks, reentry, ownership, close and caller composition. K3/V4.1 public rows are text-only. OpenAgent-owned ABI/snapshot formats are not established as interchangeable Colibri binaries or snapshots. | P11 partial |
| Cache, scheduler and recovery | Budget/admission/queue modules; `scalable_execution/scheduler.py`; durable/indexed continuation stores; cache/generation checkpoint codecs; distributed recovery launcher. | Complete family physical-cache/continuation and stale/foreign restore closure; serving publication ordering; process-loss/transport-stranding and consensus lifecycle. Existing distributed teardown/restart mechanisms do not establish transparent live recovery. | P12 partial; P01 budget fix DONE |
| Correctness/platform tooling | `wholemodel_validation/`, `platform_certification/`, source inventory, exact-row bindings, observation/cache schemas and explicit false certification flags. | Reconcile callable adapters and complete physical-state/observation schema coverage for every applicable row. Structural owner references are not comparison receipts. Actual numerical and placement evidence requires a later authorized execution phase, not more checklist assertions. | P13 partial |
| Serving | `openagent/routes/inference_routes.py`, `publisher_provider.py`, execution factories/services and gateway: bounded synchronous text compatibility plus local OpenAI chat/text and Anthropic message SSE event paths; opt-in bounded native request queue, queue wait/429 behavior, health/metrics, Brio/Jev adapters and capability disclosure. | Complete wider request semantics, sampling/logprobs/token-aware stop adaptation, family media/recovery/warm-session lifecycle and all serving-caller cleanup/backpressure. Implemented local text SSE is not full upstream or Colibri parity. See explicit feature gaps below. | P14 partial |

### Colibri functionality not fully implemented in OpenAgent

The following are **real feature gaps**, distinct from unfinished audit coverage.
Their present restrictions are visible in
`openagent/adaptive_engine/colibri_capabilities.py`,
`openagent/adaptive_engine/source_scope.py`, the compatibility HTTP handlers,
and publisher-provider guards. General OpenAgent features with similar names
must not be counted as these specific contracts.

- [ ] **Persistent Brio KV slots:** isolated cross-request `cache_slot`
  ownership, state-derived selection, bounded eviction and shared warm history.
  The current HTTP/provider paths reject `cache_slot`.
- [ ] **Brio clients and additional scoring owners:** Colibri-equivalent
  attached terminal/browser Brio clients; safe direct K3/V4.1 scoring lifecycle.
  Synchronous configured-native option/question/schema and Jev adapters exist.
  [Upstream Brio contract](https://github.com/JustVugg/colibri/blob/main/docs/brio.md).
- [ ] **Complete selected API contracts:** Local OpenAI chat/text and Anthropic
  message streaming now emit protocol-specific text deltas, terminal events and
  supported usage fields. Full upstream semantics remain incomplete: sampling
  controls, tokenizer-aware text stops, requested numeric logprobs, broader
  family prompt/continuation semantics, and exact Anthropic authentication
  policy. Unsupported controls remain rejected; optional local auth scopes are
  not identical to Colibri API-key policy.
- [ ] **Constrained generation and tools:** admitted grammar decoder and exact
  per-family tool-call rendering/parsing/streaming. Text-only response format
  and tool rejection currently enforce the boundary. Treat
  [the upstream grammar draft](https://github.com/JustVugg/colibri/blob/main/docs/grammar-draft.md)
  as a draft contract, not proof of universal upstream implementation.
- [ ] **Colibri engine wire protocol:** READY-gated mux/legacy stdin framing,
  capability/status/hardware/tier/expert-map reporting, request ordering and
  slot/error semantics. OpenAgent worker/JSONL protocols are separate.
  [Upstream serve protocol](https://github.com/JustVugg/colibri/blob/main/docs/serve_protocol.md).
- [ ] **Colibri queue/protocol parity:** OpenAgent's local `/v1` native serving
  subset now has bounded single-process FIFO admission, opt-in queue/timeouts,
  HTTP 429/Retry-After, queue-wait headers (including SSE), and health/Prometheus
  counters. Colibri READY/mux semantics, priority controls and any global or
  distributed scheduler contract remain unimplemented; local queue support is
  not evidence of those behaviors.
- [ ] **Routing persistence and visualization:** `.coli_usage`, compatible
  route traces and the expert atlas/dashboard behavior. Actual operand/routing
  observation hooks exist but do not provide this application lifecycle.
- [ ] **Optional cache-aware route substitution:** preserve top-J/substitute
  resident top-M with explicit family admission and agreement/KL/stat reporting.
  Ordinary exact-router expert caching is already a different feature.
  [Upstream cache-route contract](https://github.com/JustVugg/colibri/blob/main/docs/CACHE_ROUTE.md).
- [ ] **Mirrors and partial mirrors:** replica identity, placement planning,
  selection/fallback and error policy. Explicit named OLMoE storage roots and
  striped pack destinations exist; they are not mirrors.
  [Upstream multi-disk contract](https://github.com/JustVugg/colibri/blob/main/docs/multidisk.md).
- [ ] **Current-upstream expansion:** MiMo engines, Qwen-Image generation,
  Laya/GLiNER decision engines and Clef-specific contracts have no corresponding
  admitted row in OpenAgent's nine-family Phase 1 catalog. No dedicated engine
  implementation for those names was identified in the scoped adaptive-engine
  source search. Do not silently add these to the original pinned scope or
  describe the existing general-purpose project as implementing them.
- [ ] **Current-upstream installation/product parity:** model recommendation,
  resumable acquisition, backend/prebuilt selection, launcher/service management
  and upstream setup MCP/dashboard flows need a separate explicit contract map.
  This engine-focused audit does not establish their absence across all of
  OpenAgent, nor establish compatibility from generic tooling alone.

### What must close before calling Phase 1 source-complete

1. Finish exact-row producer-to-consumer mapping and propagate the selected tuple
   through every applicable entrypoint. Record a concrete enforced restriction
   for intentionally unsupported combinations; do not count a restriction as
   implementation of a requested capability.
2. Close the remaining media, speculation, distributed, format/device, ABI,
   scheduler/cache/recovery and serving source obligations in P02–P14. For each
   closure name the real caller, state owner, failure path and cleanup owner.
3. Complete row-specific observation/comparison/physical-cache and platform
   tooling source bindings. Keep evidence capture and numerical/platform
   certification separate and unclaimed while execution remains prohibited.
4. Reconcile P15 against those exact source obligations. Keep full-Colibri
   feature gaps visible even if a deliberately narrower Phase 1 scope closes;
   no completion percentage follows from file counts or rejected routes.

**Current task labels:** P01 DONE; P03 DONE only for its bounded inventory
contract, with family closure open; P08 DONE only for the exact-admission source
contract; P02/P04–P07/P09–P14 IN PROGRESS; P15 OPEN. The late note below that
groups P03 with partial media work does not revoke its bounded inventory status.
No tests, validation imports, builds, compilation, benchmarks, comparison runs,
hardware verification, commits or pushes were performed for this audit.

## Current progress — 2026-10-16

The status rows summarize completed implementation work and remaining source
requirements, with dated implementation notes appended below. No completion
percentage or execution certificate is asserted. Existing implementations are
retained; the open tasks below include integration review, not necessarily missing
implementations.

| Task | Current status | Completed source work | Remaining closure work |
| --- | --- | --- | --- |
| P01 | DONE | Input recovery and optional output publication budgets are separate; recovery without a sink passes zero publication budget. | None for P01-F01. |
| P02 | IN PROGRESS | Nine-family candidate matrix and exact-tuple fail-closed admission helper exist; conformance separates catalogued from admitted identities. Public K3/V4.1 Segment/Edge rows are text-only. Both V4.1 distributed entrypoints—the CLI session and direct service factory—check enabled BF16/FP8/FP4 tuples before checkpoint opening. Direct native Qwen4 service construction rejects a declared architecture other than `Qwen4ExpForCausalLM`; the directory-backed device Qwen4 factory now repeats the exact `model_type`/architecture gate at service admission before inventory planning. K3 Metal is explicitly rejected at CLI and shared K3 executor boundaries. K3 FLA is now also rejected at the shared `K3Execution` constructor and classified as a backend restriction in the source matrix: its specialized mixer does not prove an exact admitted source-format/arithmetic/continuation/tooling tuple. The raw multimodal checkpoint reader rejects Qwen4/Inkling/GLM5 identities unless their dedicated conditional inventory planners are used. The device expert factory checks manifest architecture against its exact family route and now loads the required E8 lattice transform for the shared Mixtral/Qwen3.5/Qwen4 device consumer; v1 packed-manifest routes remain rejected for missing source-dtype/exact-tuple provenance. `SparseExpertRuntime` now resolves each layer format against the selected/fallback kernel policy before opening expert storage. The native-family CLI now rejects v1 packed-expert manifests before resolving expert payload files because the manifest has no source-dtype/arithmetic provenance to bind an exact P02 row. The OLMoE public CLI binds checkpoint identity to the canonical parsed expert manifest and attempts execution and adapter teardown independently, preserving primary generation failures. Independent Qwen3.5/Qwen4 text decoders reject present `architectures` metadata that does not match their admitted causal-LM class. The GLM-DSA independent text contract now rejects any present `architectures` declaration because no architecture-class identity is pinned for that route; GLM5 text now likewise rejects every present `architectures` declaration because its accepted architecture class is not pinned. Publisher Qwen3.5/Qwen4/GLM5 MTP tensor routes are now explicitly rejected before conditional-loader payload reads because only Inkling has a concrete admitted MTP consumer. GLM-DSA device source-format candidates now match the resident reader's F32/F16/BF16 admission; E4M3 is no longer advertised on that route because no DSA scale-aware conversion consumer exists. V4.1 distributed candidate rows include their specialized backend dispatch owner instead of inheriting only the generic device consumer. Candidate case IDs and family/backend format inspection have explicit non-authorizing helpers; source catalog inspection is bounded. Exact format-consumer lookup now rejects backend names not declared for that family and explicitly restricted backends rather than falling through to a generic device consumer. Public generic native text reads/loads now reject decoder identities without a complete resident-family planner (currently `deepseek_v4` and `kimi_linear`) rather than treating registry membership as inventory admission. The direct public GLM5/Inkling Segment/Edge service factory now repeats exact native family/model/source-format admission against the named checkpoint headers before registration. The direct K3 session factory now rejects caller-assembled `K3ExecutionPayload` objects; only the publisher checkpoint adapter marks a payload after conversion and source snapshot checks complete. | Other factories/gateway still lack truthful full-tuple propagation. All current V4.1 distributed tuples and generated combinations remain unverified; reconcile remaining exact chains or add similarly precise route-level restrictions where closure is absent. |
| P02 | IN PROGRESS | Nine-family candidate matrix and exact-tuple fail-closed admission helper exist; conformance separates catalogued from admitted identities. Public K3/V4.1 Segment/Edge rows are text-only. Both V4.1 distributed entrypoints—the CLI session and direct service factory—check enabled BF16/FP8/FP4 tuples before checkpoint opening. Direct native Qwen4 service construction rejects a declared architecture other than `Qwen4ExpForCausalLM`; the directory-backed device Qwen4 factory now repeats the exact `model_type`/architecture gate at service admission before inventory planning. K3 Metal is explicitly rejected at CLI and shared K3 executor boundaries. K3 FLA is now also rejected at the shared `K3Execution` constructor and classified as a backend restriction in the source matrix: its specialized mixer does not prove an exact admitted source-format/arithmetic/continuation/tooling tuple. The raw multimodal checkpoint reader rejects Qwen4/Inkling/GLM5 identities unless their dedicated conditional inventory planners are used. The device expert factory checks manifest architecture against its exact family route and now loads the exact E8 transform required by the shared Mixtral/Qwen3.5/Qwen4 device consumer; packed-manifest routes remain rejected for missing source-dtype/exact-tuple provenance. `SparseExpertRuntime` now resolves each layer format against the selected/fallback kernel policy before opening expert storage. The native-family CLI now rejects v1 packed-expert manifests before resolving expert payload files because the manifest has no source-dtype/arithmetic provenance to bind an exact P02 row. The OLMoE public CLI binds checkpoint identity to the canonical parsed expert manifest and attempts execution and adapter teardown independently, preserving primary generation failures. Independent Qwen3.5/Qwen4 text decoders reject present `architectures` metadata that does not match their admitted causal-LM class. The GLM-DSA independent text contract now rejects any present `architectures` declaration because no architecture-class identity is pinned for that route; GLM5 text now likewise rejects every present `architectures` declaration because its accepted architecture class is not pinned. Publisher Qwen3.5/Qwen4/GLM5 MTP tensor routes are now explicitly rejected before conditional-loader payload reads because only Inkling has a concrete admitted MTP consumer. GLM-DSA device source-format candidates now match the resident reader's F32/F16/BF16 admission; E4M3 is no longer advertised on that route because no DSA scale-aware conversion consumer exists. V4.1 distributed candidate rows include their specialized backend dispatch owner instead of inheriting only the generic device consumer. Candidate case IDs and family/backend format inspection have explicit non-authorizing helpers; source catalog inspection is bounded. Exact format-consumer lookup now rejects backend names not declared for that family and explicitly restricted backends rather than falling through to a generic device consumer. Public generic native text reads/loads now reject decoder identities without a complete resident-family planner (currently `deepseek_v4` and `kimi_linear`) rather than treating registry membership as inventory admission. The direct public GLM5/Inkling Segment/Edge service factory now repeats exact native family/model/source-format admission against the named checkpoint headers. The direct K3 session factory now rejects caller-assembled `K3ExecutionPayload` objects; only the publisher checkpoint adapter marks a payload after conversion and source snapshot checks complete. The legacy generic V4.1 in-memory session factory now fails closed because it does not bind publisher source dtype, persisted encoding, or checkpoint-content provenance. | Other factories/gateway still lack truthful full-tuple propagation. All current V4.1 distributed tuples and generated combinations remain unverified; reconcile remaining exact chains or add similarly precise route-level restrictions where closure is absent. |
| P03 | DONE (bounded inventory contracts; family closure still open) | Complete resident header admission; mapping geometry preflight; root/wrapper namespace collision rejection; shared complete GLM5 inventory planner; GLM5 resident versus streamed expert selection; Inkling MTP header admission before component payload reads, including a new FP32 resident lower-bound check against the MTP-specific total budget; publisher Inkling rejects present MTP headers before payload reads unless explicit MTP execution budgets activate the draft consumer; publisher Qwen3.5/Qwen4/GLM5 MTP routes now fail closed before conditional-loader payload reads, and direct Qwen-family wrapper partitioning rejects opaque MTP tensors unless the Inkling consumer owns them; explicit optional vision/audio ownership, tied/shared head ownership, MTP/auxiliary ownership, expert-bank ownership, preserved source-text versus execution-arithmetic dtype metadata; direct native Qwen3.5 tiny-wrapper config plus complete text and exact vision tensor header inventory preflight before payload materialization, now rejecting audio config/tower headers before payload reads; Qwen4 and GLM5 publisher inventory admission now likewise reject declared audio configs and audio tensor headers before payload materialization because their current wrappers have no audio execution owner; direct native Qwen4/Inkling wrapper routes and raw-reader APIs explicitly reject before payload loading and retain the separate publisher conditional loader path; Qwen3.5 and Qwen4 require `tie_word_embeddings` to be a real boolean because it changes whether `lm_head.weight` is part of the admitted inventory. Native Qwen4 range and edge loading now preflight source dtypes for the complete inventory, requiring I64 for integer PLE metadata before reading selected-layer payloads. The shared resident planner enforces exact `config.model_type` ↔ family identity; GLM5's standalone inventory planner now also binds the supplied component contract to the exact config captured by that inventory snapshot. | Verify no factory/CLI bypasses the family-conditional planners. All-family producer/consumer and caller closure remains open. |
| P04 | IN PROGRESS | Qwen4, Qwen3.5, GLM5, Inkling, V4.1 and K3 media paths have source-traced portions through expansion/dispatch to text logits. Cancellation-aware staging, temporary-owner cleanup, placeholder/order checks and interrupted-construction cleanup are implemented for selected callers. Publisher mixed-modality requests are explicitly rejected until placeholder ordering is defined. `PublisherProvider.stream_complete_media` eagerly detaches caller buffers before returning its iterator, preserving stable ownership after stream creation. `K3PreparedMedia` now rejects pixel/grid arrays that alias caller-owned or borrowed storage instead of merely marking that storage read-only. | Finish cross-family media order/position/geometry and borrowed-owner reconciliation, plus remaining K3 temporal semantics and caller-by-caller cancellation/recovery/cleanup paths. Unsupported audio or unplanned media routes remain restricted; decode and external processor calls are not interruptible mid-operation. |
| P05 | IN PROGRESS | Target/draft/logit/RNG/stop state and proposer snapshots are transactional; stop-policy acquisition rollback and DSpark CLI/resume caller traces exist. Rejected or unused draft suffixes now restore the independently owned proposer RNG as well as draft state, so retrying cannot inherit randomness consumed by discarded proposals. `SpeculativeDecoder.restore()` now requires saved proposer-RNG state to match the current proposer's RNG ownership, preventing snapshots without RNG evidence from silently preserving unrelated live randomness; in-memory speculative snapshots are now accepted only as exact receipt objects issued by that decoder (the exposed owner marker alone can no longer be copied into a forged receipt), with at most 64 live receipt capabilities. Snapshot-capable V4.1 proposers must return a non-null snapshot receipt at capture and step boundaries, preventing rollback from silently omitting draft restoration. Successful speculative steps now compare exposed target history/logits against the would-be committed boundary before publishing logical state; a mismatch enters target/draft/both-RNG rollback and quarantine handling. `SpeculativeDecoder.reset()` and the V4.1 request reset both attempt target and draft reset independently, preserving the first failure and retaining quarantine before teardown/retry. V4.1 CLI callback/checkpoint/cancellation/partial-restore failures preserve primary errors and quarantine failed resets. Fresh DSpark CLI passes request cancellation into decoder prefill, allowing cancellation during prompt evaluation to roll back target state rather than waiting until prefill completes. The V4.1 CLI rechecks cancellation after checkpoint publication and immediately before token-event callbacks in fresh and resumed generation, and before dispatching the resumed `recovered` event after restore validation. Fresh and resumed DSpark callers now reject using a separate stream callback together with a checkpoint sink because those independent owners cannot atomically publish one committed speculative boundary. | Reconcile accept/reject/bonus/EOS/context/cancel/repeated restore and cleanup across every serving caller; compatible atomic stream-plus-checkpoint publication is unavailable and is explicitly restricted. |
| P06 | IN PROGRESS | V4.1 preflights bank/provider/device policy before format registration, synchronizes setup failures, and quarantines after transport errors; dense TP `linear` cancellation remains `InterruptedError`. `TensorParallelProjections.local_stage` synchronizes rank-local `BaseException` failures before leaving a staged collective boundary and preserves unanimous `InterruptedError` and `KeyboardInterrupt` after consensus rather than downgrading Ctrl+C to `ValueError`. Original BF16 source registration, projection preflight, and provider-row evaluation also synchronize rank-local `BaseException` failures before peers enter row-gather collectives. Rank projection schedule/policy disagreement now closes and poisons the affected tensor-parallel bank, preventing a retry from entering a divergent follow-on collective. Agreement descriptors now enforce bounded integer scalars (maximum 256 bits) as well as bounded tuple structure and strings; malformed peer payloads quarantine the bank. V4.1 `agreed_stage` preserves unanimous `InterruptedError` from rank-local resource construction after cleanup and consensus. V4.1 MTP validates complete BF16/FP8/FP4 provider inventories before registration. Distributed coordinator construction now rank-agrees service type, execution limits, and both coordinator budgets before publishing an instance, so a rank-local constructor admission error is reported by all participants rather than leaving peers waiting at the admission collective. Distributed `reset` retains per-rank rollback snapshots through result agreement and restores successful ranks plus generation-boundary metadata if any rank fails or disagrees. Failed/disagreeing distributed `close_session` and whole-service `close` make rank-agreed best-effort cleanup attempts; failed/disagreeing snapshot release now closes the affected session on each surviving rank to release remaining local snapshot owners before poisoning the group. Failed/disagreeing distributed snapshot creation now releases receipts created on successful ranks before poisoning the group, and tears down the owning session if receipt release itself fails. Restore/reset/cache-load failure cleanup attempts rollback-snapshot retirement independently of physical rollback and now closes the session on every surviving rank after any failed/disagreeing collective, even when rollback succeeded, because the poisoned coordinator cannot issue a later cleanup request. Distributed session open now requires a bounded stable model identity and agrees identity plus executor implementation across ranks before publishing the handle; disagreement enters the existing best-effort open rollback and poisons the group. Distributed cache-manifest admission rejects empty/NUL model and rank-receipt identities before collective restore. Distributed cancellation consensus now compares the digest of each rank's post-cancellation token history as well as context length, detecting same-length rollback divergence before returning a canceled result. | Map all distributed banks; close operation-level failures for every consumer, process loss, rank layout and recovery consensus. Transport failure/process loss can still strand peers and requires group restart; reset rollback is best-effort and release/close cleanup is destructive cleanup, not rollback or general bank-level consensus. |
| P07 | IN PROGRESS | K3 validates media rotary bounds; FLA cache ownership, cancellation polls and staged-fork restore cleanup are transactional. K3 video rejects nonuniform timestamps because temporal RoPE uses frame indices; timestamp values are safely normalized before interval arithmetic, including rejection of integers too large for finite float representation. Raw processor pixel/grid callers are restricted to temporal extent one; multi-frame media requires the explicit `K3VideoInput` frame-selection/timestamp route. Batch media remains restricted. K3 CLI recovery forwards cancellation through bounded read/cache staging/restore, and generation-boundary admission validates ordered per-item temporal provenance before publication or accepting a restored boundary. Batch snapshot restore now forwards cancellation through each executor's staged native/FLA restore and rolls every selected slot back on cancellation. Batch reset now checks cancellation at slot boundaries and restores every selected slot before surfacing cancellation. Partial batch-snapshot capture and transaction receipt-release failures are retained for retry; the batch blocks further operations until private cleanup succeeds and close releases receipts before executors. The incomplete Metal route is explicitly rejected at CLI and shared executor boundaries. K3 media patch construction now deterministically closes temporary PIL decoder/resizer objects across cancellation and failure. | Complete remaining temporal/video semantics, serving callers and native/FLA continuation integration; per-slot reset itself remains atomic and cannot be interrupted mid-operation. |
| P08 | DONE | Exact-admission source contract is complete for v2 text, v3 single-media and v4 unique multi-case evidence, including strict case selection, physical-state binding, continuation persistence and factory/CLI wiring. | P07/P13 retain their own integration and observation-coverage work; no runtime evidence was generated. |
| P09 | IN PROGRESS | Format routes record packing/geometry/consumer candidates and explicit rejected-unadmitted formats. The shared selected-device expert provider validates exact E8 contract/geometry and applies bounded, cancellable device activation transforms for input and intermediate rows; source routes name both decode and rotation owners for Mixtral, Qwen3.5 and Qwen4. Block128 FP8 consumers reject negative/nonfinite scales and allow zero scales only for all-zero code tiles; the selected-device FP8 decoder preflights every matrix and scale grid before allocating decoded matrices. Selected-device MXFP4 decoding now preflights every scale, decoded FP32 value and partial-block padding before allocating matrices, rejecting NaN scale code 255, finite-scale multiplication overflow, and nonzero padded nibbles. Selected-device INT2/INT3 decoding requires strictly positive finite row/group scales, matching the writer contract (zero blocks are persisted with scale 1). IQ3 selected-device geometry is checked at construction; the shared IQ3 decoder now scans all matrix-block FP16 scales and reserved bits before allocating the decoded matrix, rejecting negative scales and malformed late-block metadata without partial output allocation. Selected-device Q4_0/Q4_1 now preflight exact payload extent and finite half-precision block metadata before allocating any decoded projection; signed finite deltas/minima permitted by the format remain accepted. Selected-device Q8_1 now likewise validates exact payload extent and every block's half-precision delta/sum before allocating decoded projections. Q5_0/Q5_1 scalar, native, and full-decode consumers now reject negative block deltas; full decoders scan all metadata before allocating projection arrays. Original native projection row reads now reject zero/negative FP8 scales consistently with streamed projection and the publisher tensor reader. V4.1 packed FP4/FP8 matvec validates exact payload/scale-grid extents and finite codes/scales before device copies. IQ3 grid publication now preserves primary write/fsync/rename failures while annotating descriptor or temporary-file cleanup failures. `DiskExpertStore` now explicitly releases failed partial read/checksum payloads and preserves primary read/cache-admission/prefetch errors if cleanup also fails. Expert-container conversion now closes every successfully opened staging writer after partial-open, conversion, write, progress-callback, or flush/fsync failure, preserving the primary exception and attaching close failures as notes. Streamed publisher projections now reject non-finite inputs/biases and invalid arithmetic blocks before publishing output, and perform a final cancellation check after the last row block. | Verify each family/backend persisted-loader and execution consumer pair, including E8/IQ3 rotation, scale geometry, packing order and converter publication cleanup. |
| P10 | IN PROGRESS | Device text owners preserve no-fallback/precision policies and V4.1 codec admits the device executor. K3 precision policy is rechecked at operation entry. Selected-device FP8 scale/code representation is preflighted before decoded matrix allocation; selected-device Q4_0/Q4_1 metadata is checked for exact extent and finiteness before decode allocation, preserving signed values allowed by the format. The shared Mixtral/Qwen3.5/Qwen4 selected-device expert path validates and applies bounded, cancellable E8 activation transforms on the current device. V4.1 packed FP4/FP8 inputs are validated before device allocation. GLM-DSA device-owner identity and status now distinguish CUDA from HIP using the linked Torch runtime version tuple; the shared `cuda:N` device name alone had conflated those execution identities for cache/recovery attribution. Adaptive backend discovery advertises only registered text prefill/decode frontends; the device autotuner now rejects embedding tuning before benchmark dispatch because no embedding frontend is admitted. Shared selected-device expert execution rejects non-finite FP32 activation rows before acquiring persisted payloads or staging device weights. OpenVINO GPU/NPU resolution requires an exact provider ID instead of synthesizing a kind-level target, and target-to-profile attribution no longer falls back to a GPU/NPU family prefix, preventing ambiguous device ownership. | Prove complete operation chains and retained-state transport for every admitted backend; reconcile factory tuple placement receipts and operation-level no-fallback evidence. Registry phase restrictions do not prove backend compilation, actual device placement, or runtime no-fallback. |
| P11 | IN PROGRESS | Public K3/V4.1 text owners exist; derived identities and snapshot handles enforce ownership; cancellable ABI forwards roll back/quarantine; Mixtral construction retains failed cleanup and Qwen4 engine/session closes are idempotent. Qwen4 rejects final-hidden native entry reentered from its synchronous routing observer callback. GLM-DSA Segment session run, snapshot, restore and close serialize on one per-session lock and reject same-thread reentry from synchronous callbacks. K3/V4.1 publisher ABI session run, snapshot, restore and close now reject same-thread reentry while retaining cross-thread serialization, so cancellation and cache callbacks cannot close or mutate the active session on the callback stack. Mixtral public range sessions now likewise serialize run/snapshot/restore/close and reject same-thread callback reentry. GLM-DSA Segment open now closes its newly-created range engine if capability-record construction fails before public-handle adoption; failed cleanup remains reachable in the adapter's pending-resource inventory. The shared Segment/Edge callback guard clears constructor handles and complete capability records before admission, then clears capability records again if any later open step fails, including after family callbacks populated them but before handle publication. Segment/Edge tokenizer and detokenizer ABI callbacks now zero output count/extent before handle lookup or decoding, while preserving the required-size result for undersized-buffer sizing calls. The Segment callback now rejects a null written-length out-pointer before dereferencing it and returns an ABI callback error with diagnostic state retained. `SegmentBridge.call` now rejects successful fixed-tensor callbacks that report a short/long written extent; if a session forward has already run, the affected handle is quarantined rather than allowing continuation from potentially advanced KV. Qwen3.5 Segment run now validates the exact public run-record size, rejects nonzero token/reserved fields for its hidden-activation wire, requires exact input/output extents, and rejects nonfinite activation input before advancing KV state. Mixtral Segment run now enforces the exact `_Run` record size, rejects unused token/reserved fields, and requires exact hidden-activation input/output extents rather than accepting oversized buffers. | Review all family ABI ranges, hidden/head schemas, callback ownership/reentry across all adapters, registration, continuation, factory/CLI wiring and close order; audit analogous out-parameter behavior across other family adapters. |
| P12 | IN PROGRESS | Direct Segment ABI cancellation rolls back or quarantines; direct K3 generation publication/restore is lock-scoped; direct V4.1 publication acquires the model or complete DSpark owner lock chain. Distributed cache-load rollback restores prior generation-boundary metadata with physical state, and failed/disagreeing distributed session and whole-service close attempts retry cleanup on retained rank-local owners before leaving the group poisoned. Service cache restore performs file-backed reader final-completeness checks before committing history. OLMoE Segment forward/reset/close and portable snapshot read/write operations now serialize on a per-session reentrant lock. Durable `recover_into` snapshots and serializes the existing caller-owned session, restoring its prior physical/logical state on post-load mismatch rather than closing that existing session; if rollback itself fails, the session is quarantined as closed. Durable `recover` serializes session-registry admission, verifies the returned session is the sole newly created owner, and cleans only registry entries proven to have been created by a malformed/raising recovery attempt, preventing orphan leaks and avoiding cleanup of pre-existing caller sessions. K3 generation-boundary rollback failure now quarantines the executor while preserving `close()` for resource release. Distributed rank cache restore re-admits the shared manifest on each rank immediately before consuming receipts and after physical install; a changed manifest therefore enters the existing collective rollback path. DSpark snapshot capture checks hidden/logit evidence against the captured backbone token history and releases the nested backbone receipt if validation fails, quarantining on cleanup failure. Distributed cache/generation/reset/restore disagreements now close each affected session even after a successful rollback, since the poisoned coordinator cannot dispatch subsequent session cleanup. | Close V4.1 restore and other-family physical state, serving order, stale/foreign restore and distributed failure/consensus lifecycle; distributed close retry is best effort, not rollback or process-loss recovery. |
| P13 | IN PROGRESS | `source_scope.py` attaches validated evidence obligations to generated rows, now naming function-level observation/cache-schema owners, report schemas, artifact-pin and placement references; absent media adapters are explicit and structurally checked. Each obligation carries an exact execution-row adapter binding while labeling its owner family-schema-reference-only and requiring a separate receipt. Whole-model comparison inputs bind an exact family/model/backend/arithmetic/source/persisted/media row to both declared module:callable factories; the runner joins that binding to the generated source catalog and rejects absent or restricted tuples before importing adapters, then requires both adapters to echo the row. It also rejects media-row bindings without catalogued media adapter/report schema. The runner now rejects any whole-model manifest missing the exact row/artifact-pin binding and resolves catalog source owners against this checkout before importing adapters; the catalog validator requires a comparison report schema on every obligation. P13 owner resolution now parses source syntax and requires the referenced symbol to be declared, rather than accepting a matching word in a comment/docstring/string. Every generated obligation explicitly says `evidence_state=not-captured` and `completion_claim=False`; source-scope validation rejects attempts to promote those generated obligations or the catalog itself to complete/certified. K3 has pinned exact-media fixture and comparison inputs; its comparator rejects absent source-bound per-operation placement observations. Structural validators retain `receipt_present=False`, `executed=False`, and false certification flags. | Add and reconcile exact bindings on every applicable row; capture/pin row-specific observations, cache leaves and runtime artifacts; bind real per-operation placement/comparison receipts. Catalog joins, syntax-tree/filesystem owner checks and execution bindings are structural source contracts, not proof of adapter correctness or produced evidence. No run may use an unbound manifest, but existing configs must be regenerated with exact bindings. K3 media comparison remains restricted pending placement observations for media encode, prefill, continuation decode, and restore. Keep unsupported rows restricted. Direct K3/V4.1 scoring and remaining Colibri surfaces remain separately open. |
| P14 | IN PROGRESS | Colibri-specific capability status is exposed; `/v1/messages` `x-api-key` is route-scoped and works through opt-in auth scopes. Local OpenAI chat/text and Anthropic messages now support text SSE with protocol-specific terminal/error events, cancellation, and iterator cleanup; chat streaming supports `stream_options.include_usage`. The native publisher bounded FIFO queue supports opt-in queue/timeouts, pre-header 429/Retry-After, health/Prometheus counters, and `X-Queue-Wait-Ms` on JSON and SSE responses. The OpenAI text-completion-only `echo` option is rejected on chat-completion requests; nonempty text stop sequences, Anthropic stream options, tools and unwired controls remain rejected; OpenAI `user` attribution is rejected because no caller-attribution sink is connected. Escaped invalid-Unicode message/stop text returns HTTP request validation errors, and direct native publisher complete/stream/Brio scoring boundaries convert lone-surrogate UTF-8 failures into `ProviderConfigurationError` before session scoring/execution. `/v1/brio` rejects `cache_slot` at the HTTP route boundary; persistent isolated history remains unsupported and the provider also guards non-route callers. The Colibri READY/mux/priority/global queue protocol remains explicitly unavailable. Model-gateway upstream streaming threads consumer-disconnect cancellation into bounded egress reads, closes the active upstream connection if cancellation arrives while blocked in a line read, and preserves cancellation as `InterruptedError` rather than rewrapping it as a generic transport error; it stops dispatching queued lines after cancellation and closes each provider iterator on completion, failure, or consumer disconnect while preserving the primary error. The native publisher vision-stream gateway now explicitly closes its provider iterator on response setup failure, normal exhaustion, consumer cancellation and iteration failure while retaining the primary provider/consumer exception. The local `ChatService.stream` likewise preserves its primary provider/consumer failure when iterator close fails, while surfacing close errors after otherwise-normal exhaustion. V4.1 CLI generation and resume callers recheck cancellation after checkpoint publication and immediately before token-event callback dispatch. Runtime SSE delivery closes its provider iterator even if response/header setup fails, and writes SSE error payloads only after headers are sent. The codebase-run SSE route resolves run existence/ownership before it sends HTTP 200 headers; its lazy generator cannot turn a missing/foreign run into an already-committed SSE error instead of an HTTP 404. The direct hardware streaming client now bounds its local event handoff to 16 events, applying pipe-consumer backpressure; abandoning the iterator signals cancellation and releases a blocked event publisher before terminating an unfinished worker. GLM5 publisher construction and owner close now attempt decoder, expert-provider and projection-provider cleanup independently, retain failed owners for retry, and preserve the first cleanup failure. | Close caller-by-caller stream/backpressure, stop, recovery, warm-session, media and cleanup/error paths; full API/READY/route-telemetry/cache-routing/mirror parity remains unavailable or restricted. Cancellation-triggered socket close is best effort and does not establish device-side cancellation or end-to-end stream lifecycle behavior. |
| P15 | OPEN | Prohibitions respected; the checklist distinguishes structural catalog coverage and narrow source fixes from implemented consumer closure. | Reconcile every admitted matrix cell and remaining P04–P14 finding before completion; no percentage or completion certificate is claimed. |

### P02 OLMoE exact config architecture admission — 2026-10-15

`OlmoeConfig.read()` now requires the exact singleton
`architectures: ["OlmoeForCausalLM"]` declaration in addition to
`model_type: "olmoe"`. The shared reader is used by the native OLMoE CLI
loader path and the Segment/Edge public-range adapter path, so a checkpoint
with a mismatched, missing, or ambiguous architecture declaration cannot
enter those OLMoE routes. This closes only the OLMoE config-identity gap;
source dtype/format evidence, execution parity, and the full factory/gateway/
recovery/tooling chain remain open. P02 remains IN PROGRESS and P15 remains
OPEN. No tests, imports, builds, benchmarks, or device probes were run.

### Source closure update — 2026-10-15

- P02: bounded candidate-row inspection and explicit non-authorizing case-id
  parsing are available. Exact format-consumer lookup rejects backend names
  not declared for the selected family and rejects restricted backend paths.
  These helpers do not turn catalog candidates into admitted routes.
- P03: the shared resident planner checks the exact config family identity
  before applying that family's tensor plan, and the standalone GLM5 inventory
  planner binds the component contract to the snapshot's exact source config.
  The GLM5 device directory selector also rejects unsupported or integer
  tensor dtypes against the complete exact inventory before selected payloads
  are read; the Inkling device selector now applies the corresponding
  F32/F16/BF16 header check before payload reads as well. The Qwen4 device
  selector validates float versus integer metadata dtypes across its full
  inventory and binds streamed projections to the same checkpoint config and
  tensor geometry before reading selected payloads. The Qwen3.5 device
  selector now preflights full-inventory F32/F16/BF16 dtypes and binds its
  optional streamed projection bank to the same frozen config and admitted
  tensor geometry before payload reads. The publisher conditional loader
  rejects non-Inkling auxiliary/MTP configuration even when the corresponding
  MTP tensors are absent, preventing an unsupported draft-bank declaration
  from being silently ignored. For tied publishers without a streamed
  projection provider, physical heads are compared to the admitted embedding
  and normalized to the same shared owner before wrapper construction.
  Generic native full-text reads and
  loads now also reject `deepseek_v4` and `kimi_linear`, which have no complete
  resident-family planner; their dedicated readers remain separate.
- P04/P14: publisher decoded-media array staging propagates cancellation in
  bounded chunks; processor/tokenizer initialization failures now close the
  newly-created processor and preserve the primary error.
- P05: V4.1 speculative capture and step require real proposer snapshot
  receipts. Rollback compares observable target history/logits with the
  pre-transaction boundary and quarantines on mismatch.
- P06: distributed V4.1 bank policy construction reports rank-local
  `InterruptedError`/`KeyboardInterrupt` through the first rank agreement and
  preserves cooperative cancellation as interruption on all participating
  ranks, instead of letting peers enter the collective while one rank exits
  locally. Malformed rank policy or
  error records are validated before formatting diagnostics, so corrupt
  agreement payloads fail closed rather than raising during local decoding.
- P09: streamed publisher projections reject nonfinite inputs/biases and
  malformed/nonfinite arithmetic blocks before publishing output, and poll
  cancellation after the final block.
- P10: OpenVINO target resolution now requires exact provider IDs for GPU/NPU
  and exact target-to-profile attribution; it no longer fabricates a kind-level
  device identity or guesses ownership from a backend-prefix match.
- P11: OLMoE Segment/Edge constructors validate the capabilities out-pointer
  and clear the complete output record before fallible work, complementing the
  existing cleared-handle behavior on failed construction.
- P13: whole-model observation reports now bind the observation archive hash to
  the exact manifest digest and execution-row/artifact-pin binding digest.
  This unsigned integrity association is not certification or placement proof.
- P14: corrected the Colibri source-scope declaration to match the implemented
  route-scoped Anthropic `x-api-key` handling; requests containing both bearer
  and `x-api-key` credentials now fail with HTTP 400 rather than selecting one
  silently. At that audit point Anthropic streaming remained unavailable.

These are narrow source-boundary changes. P02/P03 family-wide closure,
P04/P05/P06/P09 integration, P07/P10/P11/P12, actual P13 runtime evidence,
P14 caller parity, and P15 reconciliation remain open. No tests, imports,
builds, benchmarks, device probes, commits or pushes were run.

P13/P15 source gate update (2026-10-14): each generated evidence obligation now states `evidence_state=not-captured` and `completion_claim=False`, and the generated source catalog carries `phase1_closed=False`. The structural validator rejects changes promoting those claims, in addition to requiring exact row bindings, owners, pin classes, cache/placement requirements and false execution/parity/certification fields. Owner references are now checked by parsing Python syntax and confirming the referenced leaf is declared; this is static parsing, not import or execution. This prevents source catalog structure or an absent receipt from being labeled captured/complete/certified. It does not manufacture or validate runtime receipts, establish adapter correctness, device placement, observed execution, parity, or certification. Actual row-bound observations and receipts remain outstanding; P13 is IN PROGRESS and P15 OPEN.

P13 source-contract audit (2026-10-14): the whole-model runner now joins a bound seven-field execution contract to the generated source catalog before importing adapters. It rejects absent tuples and any tuple whose matching catalog rows are not all `candidate-unverified` in both admission and combination status; adapter metadata echo and artifact-pin integrity checks remain additional gates. This prevents a manifest and two adapters from self-consistently naming an absent or explicitly restricted tuple. It remains a structural catalog check: it does not prove adapter correctness, device placement, observed execution, parity, or certification, and it is intentionally not a runtime receipt.

P02 static admission audit (2026-10-13): `source_matrix_rows(admitted_only=True)` admits only `verified-admitted` rows (currently none); `require_source_matrix_admission` rejects candidate tuples, and `validate_source_scope` requires the eleven ordered dimensions and canonical chain ownership while allowing only candidate/rejected classifications. V4.1 distributed `PublisherSession` and the direct `open_publisher_v41_service` factory now form the enabled BF16/FP8/FP4 bank tuples from explicit distributed flags and invoke the admission guard before checkpoint opening; all such rows currently fail closed, intentionally disabling that unverified route through both entrypoints. Other factories/loaders still do not carry a truthful complete six-field tuple, so route-specific guard wiring remains open; global wiring would reject all currently catalogued source routes.

P02/P03 source audit update (2026-10-14): K3 FLA and V4.1 distributed candidate
rows now include the specialized backend dispatch owners in their ordered
execution references instead of inheriting only a generic device owner. These
rows remain candidate-unverified; no admission status changed. The audit also
found that `NativeQwen4Range.from_directory` and `NativeQwen4Edge.from_directory`
checked complete inventory names and shapes but not source dtypes before
selected payload reads. They now require I64 on declared integer metadata
tables and F32/F16/BF16 on resident weights. This closes those two range/edge
header gaps only. P02 remains IN PROGRESS, P03
remains bounded-contract DONE with family closure open, and P15 remains OPEN.
No tests, imports, builds, benchmarks, device probes, commits or pushes were
run.

### P06 distributed projection identity-preparation agreement — 2026-10-16

The generic bank's `linear` and `grouped_linear` methods assembled their
rank-agreement descriptors after leaving the local validation `try` block.
The descriptor includes a computed bank identity and optional observer policy;
rank-local failure while preparing those values could bypass the first
collective agreement. Descriptor preparation is now caught with input
validation, and failed ranks send a restricted fallback descriptor together
with the error rather than an untrusted caller identifier. As with `local_stage`
and `_collective`, an agreement or tensor-transport exception quarantines the
bank against further operation calls. Surviving ranks cannot guarantee a
return from a collective after another process/transport is lost; this is not
recoverable in-band and the caller must terminate/restart the whole rank
group. Do not retry a poisoned coordinator/bank or advertise process-loss
recovery. P06/P12 remain IN PROGRESS and P15 remains OPEN. No tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

### P14 gateway refuses provider EOF without terminal event — 2026-10-16

The shared `ModelGateway.stream_complete` caller now requires an explicit
`StreamChunk.done is True` event from the selected provider, and independently
from any fallback provider. Previously a provider iterator that exhausted
without a terminal event could be treated as a complete response by callers
such as `/api/chat/stream`, whose HTTP SSE wrapper synthesizes a terminal
frame for ordinary iterable exhaustion. Missing terminal events now become
provider failures; fallback remains allowed only before caller-visible
output, and existing provider iterator cleanup still runs. This complements
upstream SSE `[DONE]`/JSONL terminal enforcement, but does not prove device
cancellation or cover non-gateway event streams. P14 remains IN PROGRESS and
P15 remains OPEN. No tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P06 bounded tensor-parallel scalar descriptors — 2026-10-17

The distributed projection agreement envelope now bounds integer scalar size
to 256 bits in addition to its existing node, depth, tuple-width, and string
limits. This prevents an otherwise-valid Python integer from making peer
descriptor comparisons unbounded. A focused fault-injection test covers both
unsupported mutable values and an oversized integer, and confirms the bank is
quarantined after either malformed record. This remains a source-level
control-envelope guarantee only: rank/process loss, transport failures, and
the incomplete map of bank consumers/recovery still require broader closure.
P06/P12 remain IN PROGRESS and P15 remains OPEN.

### P06 tensor-parallel agreement descriptor validation — 2026-10-17

Tensor-parallel rank-agreement records are now bounded to immutable scalar and
tuple descriptors, finite floats, and bounded error strings. Malformed nested
objects are rejected after the agreement collective and poison/close the local
projection bank before descriptor comparison; equality failures are also
caught and quarantine the bank. A fake-rank regression injects a NumPy array
inside a peer descriptor and confirms the bank is quarantined. This does not
solve process loss or transport failure during a collective; surviving peers
may still require group restart. The focused consensus test passes. P06 remains
IN PROGRESS and P15 remains OPEN.

### P02 source-reference audit accepts the repository root — 2026-10-17

The filesystem reference audit previously accepted either the
`adaptive_engine/` directory itself or a repository root containing a direct
`adaptive_engine/`, but not this repository's `openagent/adaptive_engine/`
layout. Passing the documented repository root therefore reported hundreds of
existing owner symbols as unresolved. The resolver now recognizes both root
layouts, and a regression calls the full source validator with the repository
root. `tests.test_source_scope_inventory` passes (36 tests). This repairs an
audit-root defect only: validation remains structural, all 758 generated
execution tuples remain candidate-unverified, 362 tuples remain rejected, and
there are still zero admitted tuples. P02/P15 remain OPEN.

### P06 tensor-parallel exception agreement formatting — 2026-10-17

Rank-local agreement paths in the generic projection, BF16 row, packed FP8/
FP4 banks, V4.1 bank setup, distributed execution coordinator, and collective
generation-checkpoint reservation/write/rollback publication no longer call
arbitrary exception `__str__` methods before peers reach consensus. Error
envelopes use bounded exception type summaries and preserve cooperative
interruption classification where required; a custom or broken stringifier
can no longer escape before these owners' control collectives. Regression
coverage verifies the projection boundary, coordinator summary helper, and a
fault-injected collective generation-directory reservation failure.
This closes exception-formatting failure seams only; rank/process loss,
transport errors, and complete bank-consumer/recovery coverage remain
unresolved. P06/P12 remain IN PROGRESS and P15 remains OPEN.

### P06 distributed peer-record bounds — 2026-10-17

`DistributedExecutionService` result summaries now reject integers above the
signed 64-bit wire range before cross-rank equality checks. V4.1 bank
preflight validates nested policy descriptors with bounded immutable-value
rules before comparing inventories, rejects empty policy/error envelopes,
and caps setup error-field lengths. Generation-checkpoint publication now
validates every rank's reservation envelope, bounds failure strings, validates
cleanup/error records, rejects unbounded per-rank frame-byte integers, and
requires failure rows to carry no success digest or receipt. It removes the
root's newly reserved directory before poisoning the owner on malformed
reservation consensus. This prevents
malformed nested values and huge scalar comparisons from escaping unanimous
admission. Focused tests cover oversized coordinator summary/frame integers,
unprintable exception reporting, and malformed reservation cleanup; the
changed sources parse and the consensus/checkpoint test group passes. This
does not close all collective record schemas or transport/process-loss
recovery. P06/P12 remain IN PROGRESS and P15 remains OPEN.

### P05 generation commit cancellation gate — 2026-10-17

`NativeExecutionService.generate` now polls cancellation after final executor
position validation and immediately before publishing logical session history.
Previously cancellation arriving during that final validation could be missed
and the request would commit despite having been cancelled. A deterministic
regression sets cancellation from the executor position property at precisely
that boundary and verifies the service raises `ExecutionCancelled` and rolls
physical state/history back. This closes one cancellation interleaving in the
generic generation transaction; speculative target/draft RNG reconciliation
and cross-caller behavior remain open. The focused durable-eviction suite
passes 5 tests. P05/P14 remain IN PROGRESS and P15 remains OPEN.

### P05 autoregressive proposer failed-reset quarantine — 2026-10-17

`AutoregressiveDraftProposer.reset()` now marks its physical/logical state
unusable before invoking the underlying model reset. If native reset partially
mutates cache and then raises, later proposals reject until a subsequent reset
succeeds; logical history/logits are cleared only after successful physical
reset. A fault-injection test exercises partial mutation, verifies proposal
rejection, then confirms a successful retry restores usability. This closes
the proposer-local reset failure path, not cross-caller acceptance/RNG/restore
coverage. P05 remains IN PROGRESS and P15 remains OPEN.

### P14 model gateway text-stream iterator teardown — 2026-10-17

The general text streaming route now explicitly closes both the primary
provider iterator and a pre-output fallback iterator in `finally` blocks.
Closing the gateway generator on a client disconnect therefore closes the
wrapped upstream stream deterministically instead of relying on loop-frame
destruction. A focused regression test retains the provider iterator and
asserts its close hook runs after the consumer closes the gateway stream.
This covers the general text gateway only: vision has its own explicit close
path, while native CLI/speculative lifecycle and HTTP event-stream caller
coverage still require the P14 caller-by-caller audit. P14 remains IN PROGRESS
and P15 remains OPEN. The focused test passes; no builds, benchmarks, or device
probes were run.

### P02 publisher direct-factory admission boundaries — 2026-10-17

The sealed K3 payload proved loader ownership but did not prove an exact
family/backend/arithmetic/source-format/persisted-format route. The K3 shared
factory now requires every format row for its exact route to be
verified-admitted. The direct K3 CLI and public-range CLI invoke the same
family/backend check before payload weights or expert providers are loaded;
the public factory repeats it at construction. Since the K3 route rows are
currently candidates or restricted, these caller paths fail closed instead
of treating the payload seal as Phase 1 admission. The whole-model evidence
runner remains separate so it can collect the receipts needed to change that
status.

The same family/backend gate now covers V4.1's in-memory arithmetic factory,
public-range composition factory, and DSpark session factory. Those APIs accept
sealed payloads and arbitrary arithmetic factories but cannot otherwise prove
one exact source/backend/format row. Their native/public-range route rows are
currently unverified, so they reject before opening runtime libraries or
constructing execution owners.

This closes the identified K3 and V4.1 direct-factory/CLI admission bypasses
by explicit restrictions, not K3 or V4.1 execution parity. K3 media,
continuation, FLA/native, rollback and serving paths remain open, as do V4.1
speculative, distributed and recovery lifecycles. P02/P04–P07/P12/P13 remain
IN PROGRESS and P15 remains OPEN. The full source-inventory suite passes 5
tests, including native/public-range K3 and V4.1 factory boundaries; no build,
benchmark, or device probe was run.

### P02 public Segment/Edge direct-route admission — 2026-10-17

The generic `create_segment_edge_execution_service` now requires an explicit
family, model-type, and arithmetic identity and applies the exact
family/backend admission gate and bounded checkpoint-header inventory for
`public-range`, preventing its caller-supplied family label from standing in
for the actual persisted source encodings. Mixtral, GLM-DSA, Qwen3.5,
and Qwen4 service adapters supply their declared identities. The Mixtral base
registration preflights the persisted SafeTensors inventory before opening
native libraries, using exact subclass route identities for DSA and Qwen3.5.
The GLM5/Inkling and Qwen4 model owners likewise preflight before native
registration. OLMoE public CLI now performs the same inventory admission
before constructing its registered adapter. Public K3/V4.1 composition now
also requires its exact family/backend/arithmetic route, covering the V4.1
open-service composition path as well as the named K3/V4.1 factories. These
checks deliberately fail closed for tuples whose exact format rows remain
unverified; they do not establish model/backend parity. Remaining indirect
factory, CLI and gateway callers still require a caller-by-caller audit.
The conditional directory-backed service now forwards its selected
arithmetic into inventory admission rather than silently using the helper's
`fp32` default.
P02/P03 remain IN PROGRESS; P11, P13 and P15 remain OPEN. The combined
source-inventory and family-conformance unittest suites pass 18 tests with 1
skip, including a regression asserting the generic Segment/Edge admission
and checkpoint-inventory gates run before range/runtime construction. The base
Python had no NumPy and could not run the suite; the workspace virtualenv
provided the dependencies. Changed Python sources parsed and scoped
whitespace validation passed. No builds, benchmarks, or device probes were
run.

### P14 serving owner cleanup retry — 2026-10-17

The conditional publisher and native-family `PublisherSession.close()` owners
are now unusable as soon as teardown begins, preventing new requests against
partially released state while retaining failed owners for a `close()` retry.
K3 warm-session close likewise rejects new work before releasing its retained
dependency stack, and still retries pending callbacks on repeated close.
V4.1 close now retains the shared dependency stack until every warm draft and
vision borrower closes, retrying failed borrower teardown before provider
release.
Conditional shutdown attempts native service and media processor cleanup
independently. `PublisherMediaProcessor.close()` invokes its underlying
processor close hook and retains the processor/tokenizer on failure for retry;
post-construction cleanup failures attach a retry callback to the primary
construction error. The focused regression
`tests/test_conditional_session_cleanup.py` covers independent cleanup and
retry of retained owners and V4.1 borrower-before-provider ordering across
these CLI paths. It also verifies malformed conditional stop-prefix input is
rejected before the warm session is reset; conditional stop-token IDs and the
generation budget now receive the same pre-mutation validation. This closes
only these CLI teardown seams. `tests/test_inference_routes.py` now also pins
the current OpenAI/Anthropic route exclusions for streaming, queue controls,
tool calls, and text stop sequences, including the Anthropic error envelope.
Caller-wide serving lifecycle, streaming/backpressure, cancellation, and
recovery remain open. P14 remains IN PROGRESS and P15 remains OPEN.

### P12 durable recovery lock-order admission — 2026-10-17

`DurableContinuationStore.save` and `recover_into` now acquire the existing
session lock non-blockingly before trying the store lock, and reject if either
owner is busy. This removes the wait cycle where a synchronous session
callback held the session lock while reentering the store as another store
operation held the store lock and waited for that session. Once admitted,
recovery retains its prior snapshot/rollback, foreign-handle check, and
failed-rollback quarantine behavior. `tests/test_continuation_store_lock_order.py`
passed under `.venv`, covering both public operations against a contended store
lock. This addresses one in-process recovery ordering race, not distributed
consensus, transport loss, or all-family physical-state coverage. P12 remains
IN PROGRESS and P15 OPEN. No broader tests, builds, benchmarks, device probes,
commits, or pushes were run.

### P13 local evidence inventory check — 2026-10-17

The current workspace `models/` directory contains an OLMoE checkpoint and
OpenVINO data, but no local Qwen3.5-MoE, Qwen4-Exp, GLM5/GLM-DSA, or DeepSeek
V4.1 checkpoint artifacts. Therefore this workspace cannot produce pinned
whole-model, cache, media, or device-placement receipts for those target-family
rows from its current model stash. This filesystem observation is not evidence
that such artifacts are unavailable elsewhere and does not replace the
required receipts. P13 remains IN PROGRESS and P15 OPEN.

### P02 reject unhandled floating dtypes in checkpoint preflight — 2026-10-17

`require_checkpoint_inventory_admission` now rejects every floating SafeTensors
dtype outside F32/F16/BF16/F8_E4M3 before splitting the inventory into resident
and FP8 tuple checks. Previously, an unsupported float such as F8_E5M2 could
be silently omitted when the checkpoint also had a supported ordinary float,
so preflight only checked a subset. The mixed-dtype regression
`tests/test_source_scope_inventory.py` passed under `.venv`. This does not
replace family tensor planners or close the route matrix; P02 remains IN
PROGRESS and P15 OPEN. No broader tests, builds, benchmarks, device probes,
commits, or pushes were run.

### P11 Qwen4 direct C constructor output ownership — 2026-10-17

The Qwen4 public model and session constructors, plus native original-expert
projection, row-table, and expert-bank constructors, now clear a non-null
output slot before validating descriptors, layer ranges, or retaining an
owner. Previously, an early admission failure could leave a caller's stale
handle in the output parameter, despite some Python adapters defensively
initializing their callback outputs. This closes that direct-C stale-output
failure class for these constructors only; it is not a family-wide ABI review
or proof of all callback ownership/reentry paths.
P11 remains IN PROGRESS and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

The Qwen4 public session's aggregate `q4_add` admission helper now rejects a
null accounting pointer, zero element width, and an already-over-limit running
total before subtracting the remaining budget. This prevents zero-width
division and unsigned underflow from turning invalid accounting state into an
apparently available allocation budget. This source guard was not separately
compiled or runtime-tested; P11 remains IN PROGRESS and P15 OPEN.

### P04/P14 conditional media executor close serialization — 2026-10-17

`MultimodalSessionExecutor` now holds a lifecycle lock across media validation,
bounded detachment of borrowed processor arrays, decoder execution and logits
validation. A close from another thread waits for the active operation; close
or nested operation reentry from the active synchronous callback is rejected.
Decoder close failures leave the wrapper open so cleanup can be retried. Media
failure cleanup clears both the owned staging map and the last loop-local array
and iterator references, so a retained decoder exception traceback does not
pin the final large staging buffer. This closes the wrapper's close-versus-
forward race, that traceback-retention seam, and late cancellation publication
for text/media logits (cancellation is rechecked after decoder execution and
result validation). It does not close cross-family processor ordering/geometry
or every serving caller's cancellation and recovery paths.
The focused regression in `tests/test_speculative_cancellation.py` also covers
late text and media cancellation for the wrapper and passed under `.venv`.
No broader tests, builds, benchmarks, device probes, commits, or pushes were
run. P04/P14 remain IN PROGRESS and P15 remains OPEN.

### P05 autoregressive draft cancellation propagation — 2026-10-17

The built-in autoregressive draft proposer now participates in the speculative
decoder's request-scoped cancellation scope and forwards that scope to its
underlying model when supported. It checks cancellation before and after each
draft model forward and between proposal tokens; cancellation therefore enters
the existing target/draft/RNG rollback path instead of returning a proposal
after cancellation was observed. Models without an interruptible forward still
finish their in-flight call before the post-forward check, so this is
cooperative rollback, not guaranteed device interruption. P05 remains IN
PROGRESS pending caller-wide acceptance/rejection/bonus/EOS/context/restore
reconciliation. The focused standard-library regression
`tests/test_speculative_cancellation.py` passed under the repository `.venv`.
No broader tests, builds, benchmarks, device probes, commits, or pushes were
run. P15 remains OPEN.

### P06/P12 distributed root interrupt during preflight — 2026-10-17

The root-only distributed request preflight guard now catches process-level
interrupts as well as ordinary validation errors and sends the same ordered
rejection envelope to waiting peers when no request broadcast has occurred.
It then re-raises `KeyboardInterrupt`, `SystemExit`, or `GeneratorExit` on the
root after synchronization, rather than leaving peers blocked before the next
collective. This does not recover a rank that exits or loses transport during
the collective itself; that still requires whole-group restart. P06/P12 remain
IN PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P05/P14 streamed token publication is explicitly provisional — 2026-10-16

Native conditional/family CLIs, the V4.1 CLI, and K3 fresh/recovered streams
now include `committed: false` on streamed token events; the Colibri native
provider preserves that on token deltas and marks successful final/terminal
events committed. The execution-service transaction
can restore model/session state after a later callback or checkpoint-sink
failure, but cannot retract output already accepted by an arbitrary caller.
This makes the callback contract truthful; it does not provide atomic
stream-plus-checkpoint publication or remove provisional-output semantics.
P05/P14 remain IN PROGRESS and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P14 native CLI callback reentry is rejected — 2026-10-16

Native family and conditional publisher sessions now guard run, resume,
continuation, recovery and close entrypoints against same-thread reentry while
a public operation is active. Their reentrant lock previously let a streaming
or checkpoint callback call back into the same owner; notably `close()` could
request teardown while generation still held the callback stack. Other threads
remain serialized by the session lock. This is a focused lifecycle guard, not
closure of every gateway/client caller, network backpressure, or all callback
ownership paths. P14 remains IN PROGRESS and P15 remains OPEN. No tests,
imports, builds, benchmarks, device probes, commits, or pushes were run.

### P02 K3/V4.1 gateway residency admission — 2026-10-17

The native provider's warm-residency key and `available()` status now apply
the same source-route preflight to K3 and V4.1 before reporting readiness,
artifact fingerprinting, or session reuse. K3 selects its
native/device/FLA tuple from the parsed arithmetic options; V4.1 checks each
required BF16, FP8, and FP4 bank row for its selected local backend. This
duplicates the CLI/service boundary intentionally so gateway warm reuse cannot
be treated as route admission. Regressions prove an unverified tuple aborts
before artifact fingerprinting, and readiness reports false for an unverified
route; `tests.test_source_scope_inventory` and
`tests.test_conditional_session_cleanup` passed together (20 tests).

P02 remains IN PROGRESS: this does not audit every gateway route or prove
runtime consumers for admitted tuples. P13 runtime observations and P15 final
reconciliation remain open. No builds, benchmarks, device probes, commits, or
pushes were run.

### P02 Mixtral streamed factory inventory preflight — 2026-10-17

The exported `create_streamed_mixtral_directory_service` now applies the
central exact Mixtral/native checkpoint-inventory gate before inspecting its
borrowed expert provider or reaching library/model loading. Its regression
asserts the admission call and verifies an unverified source tuple stops the
factory at that boundary. This closes the missing central preflight on this
entrypoint; it does not make the Mixtral streamed provider route admitted or
resolve its separate source-content identity restriction. P02/P03 remain IN
PROGRESS and P15 remains OPEN. The focused source and V4.1 lifecycle suites
passed together (17 tests). No builds, benchmarks, device probes, commits, or
pushes were run.

### P07 K3 batch private snapshot cleanup remains retryable — 2026-10-17

K3 batch capture/transaction cleanup now retains a retry closure for every
slot snapshot whose release fails. The batch owner rejects further operations
while private cleanup is pending, and `close()` retries those releases before
closing their executors; otherwise the executor's closed-state guard could
make an outstanding receipt permanently unreleasable. A failed retry keeps the
batch owner and model references available for another close attempt. This is
limited to deterministic surviving-process cleanup; transport/process loss,
complete K3 media/serving integration, and runtime parity remain open. P07 and
P15 remain IN PROGRESS/OPEN. No tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P04 Inkling provider cleanup on pre-construction admission failure — 2026-10-16

`InklingMultimodalDecoder` now retires providers whose ownership was explicitly
transferred to it when publisher MTP/config/tensor inventory admission fails
before media or text children are constructed. Previously those validation
paths sat outside the constructor's cleanup guard and could leak an owned
expert/projection provider. Cleanup failures are attached to the primary
admission error with a retryable `cleanup` handle, consistent with later
construction failures.

This closes only this Inkling constructor failure window; it does not prove
processor frame/audio ordering, all-family borrowed-buffer lifetime, or serving
caller cleanup/recovery. P04/P14 remain IN PROGRESS and P15 remains OPEN. No
tests, imports, builds, benchmarks, device probes, commits, or pushes were run.

### P11 Segment/Edge greedy selection inherits scoped cancellation — 2026-10-16

`SegmentEdgeExecutor.select()` now uses the active thread-local cancellation
event when its caller omits the explicit `cancel` argument, matching the
existing `forward()` and `embed_tokens()` behavior. Previously a final-head
selection performed inside `cancellation_scope()` silently constructed a
native callback that could not observe that scope, leaving a cancellation gap
at the public ABI's selection boundary. The per-call explicit event still
takes precedence.

This is a narrow callback propagation fix, not completion of the P11 family
ABI map, callback ownership/reentry audit, caller wiring, or native cancellation
evidence. P11 remains IN PROGRESS and P15 remains OPEN. No tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

### P04 GLM5 cancellation during media ordering and video expansion — 2026-10-16

GLM5's multimodal caller now rechecks its cancellation callback every 1,024
tokens while classifying video spans and every 1,024 expanded frames while
materializing the video grid sequence. Previously these bounded but linear
loops could continue through a full prompt or selected-frame set after the
serving caller had cancelled; cancellation was only observed at adjacent
embedding/tower boundaries. This does not change the processor's declared
ordering or grid mapping and is not evidence that processor ordering matches
publisher behavior.

P04/P14 remain partial: GLM5/Inkling processor-to-placeholder parity and all
caller cleanup/recovery paths are unverified, as are cross-family temporal
semantics and runtime observations. P04/P07/P14 remain IN PROGRESS and P15
remains OPEN. No tests, imports, builds, benchmarks, device probes, commits,
or pushes were run.

### P12 Inkling MTP partial snapshot cleanup retention — 2026-10-16

If staged snapshot creation fails and releasing one of the already-created
stage receipts also fails, `InklingMTP` now retains the exact stage/receipt
pairs. `close()` retries those releases before closing stages and leaves the
owner intact for another cleanup attempt if a retry fails. This closes the
identified orphan-receipt path only. MTP snapshot now captures and validates
the shared logical position before acquiring any stage receipts, and retains
receipts if publication into its owner inventory fails. MTP restore now
attempts rollback on every stage after a restore error, preserves the primary
restore exception, and quarantines the owner if any physical rollback fails;
snapshot cleanup and close remain available after quarantine. CPU/device MTP
draft and reset now use the same all-stage rollback/quarantine rule; failed
private rollback-receipt cleanup is retained for close retry, and close refuses
to invalidate outstanding caller receipts. Broader physical-state ownership,
cross-family restore and distributed consensus remain open. P12 remains IN
PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P05/P14 V4.1 dual-sink publication restriction — 2026-10-16

V4.1 generation and resume now reject simultaneous arbitrary `emit` and
checkpoint sinks for both speculative and ordinary decoding. These sinks do
not share an atomic commit protocol: checkpoint-first can persist a boundary
before event publication fails, while event-first can publish tokens before
checkpoint persistence fails. Rejecting the combination prevents the API from
claiming exactly-once streaming/recovery semantics it cannot enforce. This is
a concrete restriction, not atomic sink support; V4.1 stream-with-checkpoint
must use one publication owner or separate operations. P05/P14 remain IN
PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P03 conditional publisher component inventory ordering — 2026-10-16

The conditional publisher loader checked `components.auxiliary_config` before
constructing `components`, so Qwen3.5, Qwen4 and GLM5 metadata admission could
raise an unbound-local error before reaching the intended optional/MTP-bank
contract (Inkling short-circuited that check but constructed the same contract
later). The parsed `PublisherComponents` contract is now built immediately
after source/header admission and before auxiliary configuration or tensor-bank
ownership checks; the later duplicate construction was removed. This closes a
loader ordering defect only; it does not complete family-wide tensor-to-consumer
or caller wiring. P03 remains IN PROGRESS and P15 remains OPEN. No tests,
imports, builds, benchmarks, device probes, commits, or pushes were run.

### P06 root preflight rejection enters the collective schedule — 2026-10-16

Root-side validation in distributed public APIs could raise before the normal
command broadcast, leaving peer `serve()` loops blocked waiting for a command.
Public root request methods now send a bounded `reject` control command when a
preflight fails before the operation sequence advances. Every rank consumes
the rejection and advances the same sequence; root then receives the original
validation outcome without poisoning an otherwise unchanged group. Failures
after a collective starts still use the existing rollback/poison/restart path.
This cannot recover a peer that has exited or a broken collective transport,
and it does not certify every distributed consumer or shard operation. P06/P12
remain partial and P15 remains OPEN. No tests, imports, builds, benchmarks,
device probes, commits, or pushes were run.

### P06 distributed result-record schema agreement — 2026-10-16

`DistributedExecutionService._execute` now validates rank-gathered result
summaries against an action-specific schema before comparing them. Summary
fields are restricted to exact builtin primitives and bounded tuples, avoiding
rank-local exceptions from malformed nested objects during equality checks.
The save-cache receipt path also validates each rank number, canonical filename,
byte extent, SHA-256 and model identity before constructing the committed
manifest. A malformed or missing successful receipt is normalized into the
existing unanimous operation-failure path, so affected sessions are rolled
back/closed as appropriate and the coordinator is poisoned instead of a rank
escaping before peer cleanup.

This protects surviving ranks after a collective returns malformed records; it
does not recover a process that dies or a transport collective that fails
before all participants return. Those conditions can strand peers and still
require external group restart. It also does not add consensus to raw bank
callers outside this coordinator or prove physical rollback of opaque native
executor state. P06/P12 remain IN PROGRESS and P15 remains OPEN. No tests,
imports, builds, benchmarks, device probes, commits, or pushes were run.

### P12 Qwen4 streamed continuation cancellation boundary — 2026-10-16

`restore_qwen4_cache` now accepts an optional cancellation callback and checks
it before/after bounded header and payload reads, before each staged array
allocation, and immediately before the single physical-state publication.
Arrays remain private staging until checksum, EOF, and cancellation admission
all succeed, so cooperative cancellation during a restore leaves the existing
Qwen4 cache, PLE, mask, positions, and logical position unchanged. The direct
Qwen4 text API and public text wrapper forward the callback. The Qwen4
multimodal wrapper checks cancellation while validating its media-cache header
and forwards it into the nested text restore; it deliberately has no
post-text-commit cancellation point, avoiding a split text-cache/rope-delta
boundary.

This is cooperative between reads; it cannot interrupt a blocked reader or a
large single allocation, and legacy public callback transports that do not
provide a cancellation owner remain non-cancellable. This does not close other
family restore paths, caller-wide cancellation ordering, distributed consensus,
or process/transport-loss handling; those failures still require group restart
where agreement is lost. P12 remains IN PROGRESS and P15 remains OPEN. No
tests, imports, builds, benchmarks, device probes, commits, or pushes were run.

### P14 gateway streaming terminal-order enforcement — 2026-10-16

The generic text gateway validates each provider event's required field types
and rejects duplicate or post-terminal events for both the selected provider
and its pre-publication fallback. A malformed event before anything is yielded
can still use configured fallback; after a yielded event, existing logic
refuses provider splicing. The gateway vision path now validates the same event
shape and terminates its upstream iterator at the terminal event. This does not
implement OpenAI/Anthropic SSE semantics or close all endpoint-specific
stream/backpressure/cancellation callers. P14 remains IN PROGRESS and P15
remains OPEN. No tests, imports, builds, benchmarks, device probes, commits, or
pushes were run.

### P07/P12 K3 batch snapshot release ownership — 2026-10-16

K3 batch transactions called `K3Execution.release_snapshot()` while that API
was absent, so their `finally` cleanup could fail after both successful work
and rollback. K3 now issues one-shot lifetime tokens with in-memory snapshots;
release validates originating-executor ownership, invalidates the receipt, and drops its
detached MLA/KDA host or device state. Restore rejects a released receipt while
checkpoint-decoded snapshots without a local release token remain restorable.
Batch snapshots now expose a matching release operation with per-slot retry
tracking, reject restore after release, and prevent closing a session still
referenced by a retained receipt. Explicit release drops the retained-byte
charge, partial release blocks restore and remains retryable, and batch close
releases owned receipts before tearing down their executors.

This repairs the source-level K3 batch snapshot lifecycle but is not execution
evidence and does not close all K3 continuation, media, serving, or recovery
paths. P07/P12 remain partial and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P10 GLM-DSA CUDA/HIP device identity distinction — 2026-10-16

`TorchGlmDsaDecoder.identity` previously included Torch version and a device
string such as `cuda:0`, which is shared by CUDA and ROCm/HIP builds. The
decoder identity now also binds Torch's linked CUDA/HIP/MPS runtime version
tuple, and the owner status reports that tuple for diagnostics. This prevents
cross-runtime cache/recovery identities from aliasing at the source-contract
level. It is not runtime placement evidence, hardware certification, or proof
that every operator remained on device. Other device-family identities and
the full P10 placement/no-fallback evidence review remain open. No tests,
imports, builds, benchmarks, device probes, commits, or pushes were run.

### P02 legacy generic V4.1 in-memory factory restriction — 2026-10-16

`create_v41_execution_service(config, tensors, ...)` accepted caller-supplied
equation tensors without a publisher checkpoint adapter, exact persisted
source encoding, or checkpoint-content identity. That generic constructor is
not the source-bound Phase 1 V4.1 route; it now fails closed before provider,
library, or executor construction. Publisher-backed `open_publisher_v41_service`
and the corresponding public-adapter path remain distinct and are not changed
by this restriction. This closes only that direct in-memory factory bypass;
other V4.1 callers and exact-row evidence remain open. No tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

### P02 public-family ABI factory checkpoint re-admission — 2026-10-16

`create_native_family_public_execution_service` accepts an already materialized
GLM5 or Inkling decoder, which by itself does not identify the persisted source
format or prove that its tensors came from the checkpoint directory supplied
for public registration. The factory now accepts only the exact GLM5/Inkling
decoder classes and performs complete checkpoint-header admission for the
corresponding native family/model tuple before constructing the public adapter
or loading/using the public runtime. This closes that specific prebuilt-decoder
factory bypass; it does not authenticate that the caller's in-memory decoder
weights are byte-identical to the admitted checkpoint. P02 remains IN PROGRESS
because the whole matrix and other factory/CLI/gateway/recovery/tooling routes
are not yet reconciled. P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P02 K3 direct-factory payload provenance gate — 2026-10-16

`create_k3_execution_service` previously checked the public payload dataclass
type and geometry, but the dataclass itself was caller-constructible; a caller
could assemble a payload and bypass the checkpoint adapter's conversion and
source-snapshot checks. Payloads produced by `publisher_checkpoint.k3_adapter`
now carry a private in-process admission seal only after their complete
conversion and snapshot-integrity path succeeds, and the direct session
factory requires that seal. This is an API provenance guard, not a
cryptographic boundary against arbitrary in-process code. Other K3 route
dimensions and families remain under the wider P02 audit; P02/P15 stay open.
No tests, imports, builds, benchmarks, device probes, commits, or pushes were
run.

### P12 Qwen4 cancellation-safe continuation restore — 2026-10-16

Qwen4's bounded text continuation reader now accepts a cancellation callback,
checks it during bounded reads and chunk staging, and performs a final check
immediately before publishing the staged cache, PLE, mask, position and extent.
The independent decoder, native public-text wrapper and Qwen4 multimodal frame
forward that callback; the multimodal owner commits its rotary delta only after
the nested text restore succeeds and does not add a cancellation point after
that commit. This closes one local Qwen4 restore-cancellation path only. A
blocked source read cannot be interrupted, and other families, distributed
consensus and caller-by-caller recovery remain open. P12 remains IN PROGRESS
and P15 remains OPEN. No tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P02 source matrix generation and static owner resolution — 2026-10-17

`source_execution_cases()` now creates its execution cell before backend
classification, normalizes scalar format-owner references as one owner instead
of a tuple of characters, and preserves distinct pinned Mixtral variants while
requiring their shared six-field execution contract to agree. The six-field
resolver refuses variants whose executable contracts diverge. Restricted K3
FLA/Metal rows now bind to their actual rejecting/dispatch source owners, and
the source validator does not require an executable format consumer for those
restricted rows. Format catalog validation now distinguishes source-dtype
labels from persisted formats and excludes narrative contract labels from
filesystem owner resolution; explicitly rejected formats carry visible
non-resolvable restriction sentinels.

`tests/test_source_scope_inventory.py` now materializes and validates the
entire generated source catalog and resolves every declared static owner. The
current catalog has 1,120 rows: 758 candidate-unverified, 362 rejected, and
zero verified-admitted. This closes catalog-generation/structural-validation
defects, not the end-to-end source audit or route admission. Runtime evidence,
family/backend/format consumer tracing, and P13 receipts remain open; P02/P13
remain IN PROGRESS and P15 remains OPEN.

### P06/P12 launcher teardown diagnostics preserve primary failures — 2026-10-17

Distributed worker teardown no longer calls arbitrary exception `__str__`
methods while aggregating service/process-group cleanup failures or attaching a
secondary close failure to the driver's primary exception. It uses the
coordinator's bounded type-only failure summary, so hostile exception
formatting cannot mask the operation failure after cleanup. A focused
regression covers exceptions whose `__str__` raises. This protects surviving
rank cleanup reporting only; it does not recover a lost process or repair a
failed collective. P06/P12 remain IN PROGRESS and P15 remains OPEN.

### P13 native comparison state protocol and source pin coverage — 2026-10-17

The shared native whole-model observation adapter now accepts either a matched
`snapshot`/`restore` or `snapshot_state`/`restore_state` pair. Cache observation
uses the same explicit state API when no separate cache getter is supplied,
allowing stateful native owners such as the OLMoE decoder to expose a single
complete physical boundary for comparison and rollback. A regression verifies
observed cache state, restore to a prior boundary, subsequent continuation, and
owner close. The certification source inventory now pins both shared
whole-model adapter and cache-reader modules under correctness tooling. The
focused lifecycle/source/inventory/cache/distributed/factory suite passes
(57 tests). This wires a generic adapter protocol; it does not instantiate an
OLMoE row-specific adapter or produce pinned model observations, cache receipts,
placement evidence, or comparison reports. P13 and P15 remain OPEN.

### P14 Brio/Jev disconnect-cancellation restriction — 2026-10-16

Reconciled the Brio/Jev serving declarations with their actual synchronous
HTTP request path: those scoring handlers do not receive a caller-disconnect
signal and cannot cancel an in-flight native score. Capability entries now
state that limitation explicitly rather than allowing synchronous endpoint
availability to imply cancellation parity. Queue controls remain rejected on
compatibility routes, persistent `cache_slot` remains rejected on Brio, and
native scoring ownership still serializes warm-session use. No code change
was warranted in this bounded pass; adding true disconnect cancellation
requires threading a request-lifetime signal from the HTTP handler through
the scorer and native owner. P14 remains IN PROGRESS and P15 remains OPEN. No
tests, imports, builds, benchmarks, device probes, commits, or pushes were
run.

### P11 Mixtral/GLM-DSA activation callback and output contracts — 2026-10-16

The public Mixtral and GLM-DSA range activation paths now require cancellation
callbacks to return exactly 0 or 1, rather than accepting arbitrary truthy
integers. Both validate staged activation output finiteness before copying it
to the caller-provided destination; GLM-DSA already validates finite input,
and this pass keeps publication transactional on invalid output.

These guards close two family-specific activation callback/output checks only.
They do not close other family ABI layouts, request and capability schemas,
callback lifetime/reentry contracts, factory/CLI/service wiring, or execution
evidence. P11 remains IN PROGRESS and P15 remains OPEN. No tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

### P02 native checkpoint reader backend call-site audit — 2026-10-16

Every production `read_native_text_checkpoint()` call site under
`openagent/adaptive_engine` was enumerated. Whole native reads now identify
`native`; Torch accelerator readers identify `device`; Mixtral/Qwen3.5/Qwen4/
GLM-DSA public range and edge readers identify `public-range`; streamed
Mixtral/GLM-DSA source validation identifies its native consumer. The API now
defaults only to native for legacy whole-reader callers and rejects an
explicit unknown backend. The bounded Qwen3.5 multimodal reader also binds its
complete headers to the native P02 tuple before weight payload reads. These
rows remain candidate/unverified and fail closed. The audit does not prove
whole-model numerical execution, external callers outside this package, or
family service/recovery/tooling completion. P02/P15 remain OPEN. No runtime
checks or commits were performed.

P02 direct-service bypass closure (2026-10-14): `open_publisher_v41_service`
now applies the same distributed bank-tuple guard as the CLI session before
checkpoint open, so callers constructing the service directly cannot bypass
the CLI-only restriction. The guard remains conservative: no distributed
V4.1 tuple is verified-admitted, and any tuple without an exact declared row
also fails closed. This closes only the service-factory bypass for that
distributed route; P02 remains IN PROGRESS. No tests, imports, builds,
benchmarks, device probes, commits or pushes were run.

P02 Qwen4 directory-factory admission alignment (2026-10-14): the public
`open_device_qwen4_directory_service` boundary now explicitly checks exact
`qwen4_exp_text` identity and the supported causal-LM architecture declaration
before invoking inventory planning. This duplicates the decoder's downstream
guard intentionally, keeping the service factory's admitted family/backend
route visible at the entrypoint. It does not provide complete tuple propagation
or certify loader/execution/cache/ABI/gateway/tooling closure; P02 remains IN
PROGRESS and P15 OPEN. No tests, imports, builds, benchmarks, device probes,
commits or pushes were run.

P09/P10 selected-device consumer audit (2026-10-14): the shared Mixtral/Qwen3.5/Qwen4 selected-device expert provider now rejects non-finite FP32 activation rows before opening the selected persisted expert payload or allocating device weight tensors. This closes only the consumer input-admission boundary; it does not certify packed-format parity, actual device placement, full backend chains, or no-fallback behavior. P09/P10 remain IN PROGRESS and P15 remains OPEN.

P09/P10 selected-device format admission update (2026-10-14): MXFP4 expert decoding now validates the complete encoded scales, FP32-decodable values, and canonical partial-block padding before allocating any decoded projection matrices. This rejects NaN UE8M0 scale codes, finite scale/code products that overflow FP32, and nonzero padded nibbles at the persisted-consumer boundary. This does not close other family/backend format routes, prove placement, or provide runtime evidence; P09/P10 remain IN PROGRESS and P15 remains OPEN.

### P02 native-family packed-manifest tuple restriction — 2026-10-14

`native_family_cli.PublisherSession` now rejects a v1 packed-expert manifest
after bounded metadata parsing and family-identity matching, but before
resolving expert payload paths, hashing payload files, opening `DiskExpertStore`,
or constructing a streamed/device provider. The manifest records packed output
format but not source dtype/arithmetic provenance, so this route cannot select
or prove an exact six-field P02 row; it is explicitly restricted rather than
inferring an admitted tuple from family name. This does not close unpacked
native routes or other factories, and P02 remains IN PROGRESS. P15 remains OPEN
because other P02 chains and P04–P14 findings remain. No tests, imports, builds,
benchmarks, device probes, commits or pushes were run.

### P06 distributed snapshot-release failure cleanup — 2026-10-14

If rank-local snapshot release fails or ranks disagree on its result,
`DistributedExecutionService` now attempts to close the affected session on
every surviving rank and drops its coordinator-side generation-boundary and
snapshot handles before poisoning the group. This drains remaining physical
snapshot owners where possible, but is deliberately destructive to that
session, is best effort, and cannot recover transport/process loss or provide
rollback/consensus. P06 remains IN PROGRESS and P15 OPEN. No tests, imports,
builds, benchmarks, device probes, commits or pushes were run.

### P05 in-memory snapshot owner binding — 2026-10-14

`SpeculativeSnapshot` now records a process-local owner capability supplied by
its `SpeculativeDecoder`; `restore()` rejects a snapshot created by another
decoder even when both have equal generation policies and compatible RNG
types. This covers in-memory receipt substitution, not durable DSpark cache
identity, arbitrary external state mutation, or full serving-caller recovery
closure. P05 remains IN PROGRESS and P15 OPEN. No tests, imports, builds,
benchmarks, device probes, commits or pushes were run.

### P06 independent rollback-snapshot retirement — 2026-10-14

Distributed `reset`, `restore`, `restore_generation`, and `load_cache` failure
cleanup now attempts to release the temporary rollback snapshot even if
physical rollback itself raises. When rollback or receipt retirement fails,
each surviving rank also makes a best-effort close of the affected session and
clears coordinator-side handles before the existing collective failure poisons
the group. This prevents a single cleanup exception from skipping an
independent cleanup action, but does not make rollback or teardown reliable
under arbitrary rank/process loss. P06 remains IN PROGRESS and P15 OPEN. No
tests, imports, builds, benchmarks, device probes, commits or pushes were run.

### P06/P12 poisoned-session teardown after collective disagreement — 2026-10-14

For distributed `reset`, `restore`, `restore_generation`, and `load_cache`,
rollback success previously left the local session and caller snapshots alive
even though the rank result disagreement poisoned the coordinator. Since
`_request` refuses all later operations on a poisoned coordinator, callers
could not close those still-owned sessions. The failure path now closes the
affected session on every surviving rank after any failed/disagreeing
collective, independently of whether physical rollback and rollback-snapshot
retirement succeeded. This is fail-closed cleanup, not recovery consensus;
transport/process loss and teardown failure remain unresolved. P06/P12 remain
IN PROGRESS and P15 OPEN. No tests, imports, builds, benchmarks, device probes,
commits or pushes were run.

### Implemented source changes

- [x] Native Segment/Edge token-to-logit rollback failure now quarantines the
  affected session before attempting release. If native release also fails,
  the retained handle remains available for cleanup retry but cannot execute
  against physical cache state whose rollback was not established:
  `native_segment_bridge.NativeSegmentBridge.token_logits`.
- [ ] P11/P12 residual: audit equivalent rollback-failure quarantine behavior
  across other native family/public adapters and validate teardown/restore
  ordering for every ABI family. This narrow fix does not close family-wide
  ownership, stale/foreign restore, or recovery-consensus requirements.

- [x] Added a code-based eight-family source owner catalog without importing
  decoders or claiming backend execution: `source_scope.py`, `family_conformance.py`.
- [x] Added structural Phase 1 source-scope validation and exposed each family’s
  backend, format, media, evidence, and owner-module classifications through
  `family_conformance.conformance_matrix`.
- [x] Added a second exact-tuple check at the native device expert factory:
  `_DeviceOwner` verifies packed-manifest architecture against its family route
  and rejects families without a concrete packed-device provider before opening
  the expert store. This duplicates the CLI admission boundary intentionally;
  it does not establish a complete family/backend/format chain.
- [x] Made source-scope validation accept both native tuple records and
  JSON-decoded list records, with non-empty string classification checks.
- [x] Added the existing OLMoE Segment/Edge source owners to the conformance
  catalog so the public registry cannot reference an unclassified family.
- [x] Added explicit positional/admission variant classifications for every
  catalog family and structural validation for those variants.
- [x] Added pure complete Inkling MTP header admission and shared canonical names:
  `inkling_mtp.admit_mtp_inventory`, conditional loader and multimodal constructor.
- [x] Retained K3 batch failed factory cleanup and checked declared slot budgets:
  `k3_execution.batch.K3BatchExecution.open_session`.
- [x] Made failed native public kernel finalizer release retryable:
  `native_library_identity.release_finalizer`, GLM/Inkling and Qwen4 public owners.
- [x] Checked K3 checkpoint restore staging before detached array allocation;
  bounded chunks, checked original paths and descriptor/path stability, added
  cancellation/write-progress checks and retained staging cleanup:
  `k3_execution.checkpoints`.
- [x] Admitted the concrete V4.1 device executor in the complete packed checkpoint
  codec; added original-path and descriptor/path checks and retained publication
  staging cleanup: `v41_execution.publisher_checkpoints`.
- [x] Matched the shared file transport cancellation call to the actual K3 and
  V4.1 methods, retained partial staging and enforced binary write progress:
  `file_cache_transport.transport_cache`.
- [x] Strengthened original file/config/tokenizer/media identities and warm reuse:
  `native_checkpoint`, `native_library_identity`, `publisher_media_files`,
  `publisher_media_processor`, family/conditional/K3/V4.1 CLIs and publisher gateway.
- [x] Retained failed gateway construction cleanup before admitting a replacement
  session: `publisher_provider._construct_session` / `_release_session`.
- [x] Strengthened continuation receipt/cache filename ownership, application-key
  binding, database ownership, bounded scans and durable deletion order:
  `continuation_store`, `indexed_continuation_store`, `directory_ownership`.
- [x] Strengthened scalable transaction quarantine and cleanup retries, cancellation
  classification, queue startup and deadline error preservation, restored metadata
  types/reference checks and bounded page/token checksum reads:
  `scalable_execution`, `execution_queue`, `execution_service`, `executor_budget`.
- [x] Strengthened rank checkpoint/source manifests, boundary directory uniqueness,
  primary-error preservation and failed factory cleanup before group destruction:
  `distributed_generation_checkpoint`, `distributed_execution`,
  `distributed_recovery`, `distributed_launcher`.
- [x] Added bounded closed-set continuation likelihood scoring with rollback to
  the original executor boundary, service-backed native/conditional session
  wrappers, native-only `/v1/brio` one-question/batch/schema routes with common
  prompt-prefix sharing, and `/v1/systemone` noul/choice/score translation.
  Added Jev 255-label propagation, an 8 MiB rendered batch-prompt bound,
  schema choice/prompt aggregate bounds, and a
  single warm-session lifecycle lock and normal load/use timestamps across
  dependent schema fields. Their shared token prefix is prefetched once and the
  original service session snapshot is restored; schema prompts and candidate
  token reads have aggregate bounds. Direct K3/V4.1 adapters remain open.
- [x] Added a cumulative 1 MiB text-admission budget to `/v1/systemone` before
  concatenating noul/choice/score adapter prompts; bound question-name metadata.

### Remaining work, in dependency order — rechecked 2026-10-15

The following roll-up is conservative: narrow changes recorded below close only
their named source boundary. They do not promote a parent phase or establish
runtime correctness.

1. **P02 — exact source-combination admission:** Reconcile every family ×
   backend × arithmetic × persisted-format tuple through configuration, tensor
   inventory, loader, execution consumer, cache/tensor-parallel owner, ABI,
   factory, CLI, gateway, recovery and tooling. Generated matrix rows remain
   candidates; `admitted_only=True` currently yields no verified rows. Exact-tuple
   helpers reject candidates. The V4.1 distributed session factory now
   constructs the exact bank tuples from its explicit BF16/FP8/FP4 flags and
   calls the admission helper before opening the checkpoint; because no such
   row is currently verified, distributed V4.1 construction is explicitly
   fail-closed. Other factories/loaders still do not carry a truthful complete
   tuple. Add that data flow or enforce route-specific restrictions. Global
   guard wiring without the tuple would block all catalogued routes. Resolve
   the additional `_DeviceOwner` family/manifest guard with the remaining
   factory and gateway identity gaps; it only closes packed-device manifest
   admission and does not bind arithmetic or source tensor formats. Resolve
   public Segment/Edge composition separately from internal counted kernels.
   The V4.1 distributed session is specifically disabled by this guard until
   its bank tuples are verified; local V4.1 is not blocked by that restriction.
2. **P03/P04 — inventory and media:** Keep P03 limited to its bounded resident
   header-inventory contract until family producer/consumer closure is mapped.
   Connect optional/shared/tied/MTP/expert/media banks to execution, and trace
   processor ordering, multi-image/audio/video expansion, positional geometry,
   logits, borrowed ownership, cancellation, recovery and constructor-failure
   cleanup for every admitted family. Qwen4/Qwen3.5/GLM5/Inkling/V4.1 have
   bounded traces; V4.1 preprocessing now closes converted image intermediates
   when resize/pad construction fails. Qwen3.5 audio and unplanned multi-chunk
   paths are restricted; K3 and remaining family/caller variants still need
   closure.
3. **P05/P06 — speculation and distributed execution:** Speculative iteration
   now raises `ExecutionCancelled` between yielded transactions; partial
   distributed session-open consensus rolls back successful rank-local opens;
   enabled V4.1 distributed bank routes now fail closed before checkpoint open
   because their exact tuples remain unverified; tensor-parallel local-stage
   `BaseException` errors reach the rank agreement boundary. Original BF16
   registration, projection preflight and provider-row failures also synchronize
   before peers enter row-gather operations, preserving cooperative interruption.
   Speculation restores the proposer's separately owned RNG when a rejected or
   unused draft suffix is discarded.
   Still reconcile every serving caller and accepted/rejected/bonus/EOS/stop/
   context/cancellation/repeated-restore boundary, both RNGs, cache state and
   publication sinks. For every distributed bank/consumer, close shard-layout,
   operation-level collective failure, process loss, cancellation, rollback,
   failed-open internals and recovery-consensus ordering. V4.1 bank preflight
   and MTP inventory checks do not establish later operation guarantees.
4. **P07 — K3 integration:** Complete temporal/media semantics and caller
   ordering, batch rollback, and native/FLA continuation/recovery across serving
   routes. Nonuniform video timestamps and batch media are explicitly rejected;
   generation-boundary admission validates ordered per-item temporal
   provenance before publication or recovery. Device precision checks, FLA
   transactional cleanup, K3 resume cancellation propagation into the recovery
   codec, and batch reset cancellation/rollback between atomic slot resets are
   narrow closures. A single slot reset remains non-interruptible.
   Metal remains unadmitted. Keep P08's exact-case evidence contract distinct
   from actual evidence capture.
5. **P09/P10 — formats and devices:** Map each admitted persisted format to its
   actual loader and arithmetic consumer, including packing, scales/zero points,
   geometry, E8/IQ3 transformations, conversion cleanup, placement receipts,
   device admission and operation-level no-fallback. E8 is rejected on shared
   Mixtral/Qwen3.5/Qwen4 selected-device expert paths until activation rotation
   exists. FP8 zero-scale rules and selected-device representation preflight,
   V4.1 packed FP4/FP8 pre-allocation validation, IQ3 geometry admission and
   cancellation-safe expert staging, IQ3 grid writer cleanup preservation, and
   partial `DiskExpertStore` payload cleanup are narrow fixes, not full matrix
   closure. Backend discovery advertises only
   prefill/decode for registered text frontends; it does not establish actual
   backend compilation, placement, or no-fallback execution.
6. **P11/P12 — ABI and recovery:** Produce a family-by-family public ABI map
   covering symbols, versions/ranges, tensor/head schemas, callbacks, ownership,
   reentry/thread rules, continuation, close order and factory/CLI/service
   wiring. Exact snapshot-handle identity is enforced for SegmentEdge and
   Mixtral device snapshots; direct Segment cancellation rollback, K3 codec
   publication/restore locking, and V4.1 publication locking cover only named
   paths. GLM-DSA ABI run/snapshot/restore/session-close operations now share a
   per-session lock, preventing cache serialization from interleaving with
   model forward or teardown, and reject same-thread public-session reentry
   through synchronous callbacks. This is a GLM-DSA-only boundary; finish every
   family's physical state, stale/foreign restore,
   cancellation/reentry order, cleanup retry, and distributed consensus rollback.
   Durable continuation `recover_into` now snapshots and serializes the existing
   caller-owned session and attempts rollback on post-load validation failure;
   distributed rank-cache restore re-admits its shared manifest before and
   after rank installation and uses collective rollback on change. These
   bounded paths do not close the family recovery matrix.
7. **P13 — evidence/tooling:** Bind each applicable matrix row to callable
   evidence adapters and schemas, artifact pins, cache leaves, placement and
   comparison inputs. The static source-reference audit now resolves P13
   observation, cache, comparison, media, artifact-pin and placement owners to
   repository files/symbols. Obligations validate exact execution-row adapter
   bindings but explicitly identify only family-schema-reference scope and still
   require separate receipts. Source obligations and the K3 media placement gate
   are not receipts; no runtime evidence or certification is claimed. Capture actual
   row-specific observations and per-operation placement, or keep rows explicitly
   restricted. Direct K3/V4.1 scoring remains restricted pending safe lifecycle
   composition.
8. **P14 — serving and Colibri features:** Audit caller-by-caller media,
   streaming/backpressure, stop handling, warm sessions, cancellation, recovery,
   auth, cleanup and error propagation. Route-scoped `/v1/messages` `x-api-key`,
   rejection of unsupported controls and stream-only options on non-streaming
   requests, explicit `/v1/brio` `cache_slot` rejection, provider-iterator
   cleanup around SSE header setup and gateway stream teardown, plus primary-
   error preservation during local `ChatService.stream` iterator cleanup; the
   direct OpenVINO route rejects constrained-generation fields it cannot honor;
   the chat `echo` restriction and upstream cancellation-triggered socket close
   are partial closures only; socket close is best-effort and does not prove
   provider/device execution cancellation. Persistent
   isolated `cache_slot`, terminal/browser
   clients, full OpenAI/Anthropic streaming and request semantics, READY/mux,
   GBNF/tool protocols, HTTP queue admission and Colibri queue metrics,
   `.coli_usage`/route traces, cache-aware expert routing, and multi-disk mirror
   planning remain absent or explicitly restricted.
9. **P15 — final reconciliation:** Revisit every P02–P14 row and source
   restriction. P15 stays OPEN until each applicable path is implemented and
   source-evidenced, or rejected with a concrete enforced reason, and no
   applicable requirement remains unresolved.

Tests, imports, builds, compilation, benchmarks, device probes, comparison runs,
commits and pushes: **not run**. Runtime/numerical/platform verification remains
outside this Phase 1 source scope and is not counted as remaining source work.

### Deep source audit — 2026-10-13

This audit was performed by static source inspection only. It did not import
modules, execute decoders, open model artifacts, run tests/builds/benchmarks or
probe devices. The following items are evidenced as implemented source changes:

- P02: `source_scope.py` contains the nine-family matrix schema, all eleven
  dependency-chain dimensions, persisted-format route decisions, explicit
  rejected-format sentinels, unique callable-chain validation, format-owner and
  execution-consumer inclusion checks, and filesystem reference validation. P02
  is closed at the source-matrix contract level; downstream consumer closure
  remains tracked under P03/P09/P10/P11/P13/P14.
- P03: header admission, namespace/geometry checks, optional and tied-head
  ownership, MTP/auxiliary/media ownership, expert-bank ownership, and separate
  source/execution dtype metadata are present. P03 remains DONE at the inventory
  contract level; downstream consumer closure is tracked under P04/P09/P10/P13.
- P04/P07/P14: media order is frozen before preprocessing; typed publisher video
  metadata now matches the exact three-field processor contract; audio ownership
  no longer overwrites image cleanup state; K3 and publisher video inputs detach
  caller-owned frames. End-to-end multi-image position/type expansion, dispatch,
  streaming, repeated recovery and warm-session closure remain open.
- P05/P06/P12: snapshot/RNG/rollback, rank manifests, cancellation consensus,
  transactional restore and retained cleanup paths exist. Caller-by-caller
  acceptance-boundary, collective failure-ordering, direct-codec cancellation,
  stale/foreign restore and physical cache ownership closure remain open.
- P08: exact-admission source contracts and case persistence remain DONE; no
  runtime evidence is asserted.
- P09: format-specific packers/readers and bounded device decoders are present;
  the FP8 decoder now handles valid zero-scale blocks explicitly. E8/IQ3
  rotation/geometry, all family/backend consumer bindings, omitted-bank
  residency and conversion cleanup remain open.
- P10/P11/P13/P14/P15: source owners, ABI lifecycle paths, conformance/tooling,
  gateway routing and prohibition tracking exist, but the checklist’s required
  family-by-family callable chains, ABI adapters, evidence bindings, public
  option/lifecycle matrix and final cross-area closure are not yet proven.

No parent status is promoted by this audit. P15 remains OPEN until every open
matrix cell and cross-area source requirement has concrete evidence or an
explicit truthful restriction.

### Deep audit recheck — 2026-10-13

The source tree and checklist were re-audited after the latest P02/P03, media,
recovery, and format-path changes. This recheck is static only: no tests,
imports, builds, benchmarks, device probes, or runtime execution were used.

- **P02:** the eleven required matrix dimensions and persisted-format routes are
  present. Selected `format_owner` references are now copied into each row's
  loading chain, and validation rejects rows that omit them. This fixes a
  matrix-consistency defect but does not close P02; every family/backend/format
  row still needs callable-chain reconciliation through config, inventory,
  loader, execution, cache/TP, ABI, factory, CLI, gateway, recovery, and
  tooling.
- **P03:** optional, tied/shared, MTP/auxiliary, media, expert-bank, and
  source/execution-dtype ownership remain represented in the source contract.
  The publisher conditional loader now value-checks physical tied heads against
  the admitted embedding, then binds the canonical head to that exact shared
  owner before wrapper construction when no streamed projection provider owns
  it. Remaining: the custom-selector families bypass the generic resident
  planner by design, so their per-family inventory decisions still require a
  complete caller-by-caller audit; downstream consumers remain open in
  P04/P06/P07/P09/P10/P13/P14.
- **P04/P07/P14:** frozen media ordering, typed video metadata, audio waveform
  ownership, detached batch frames, and lifecycle guards are present. Complete
  multi-media/temporal expansion, dispatch, backpressure, prefix-stop,
  repeated recovery, and warm-session composition are not source-closed.
- **P05/P06/P12:** rollback quarantine, RNG/target-draft state, rank manifests,
  cancellation, replacement orchestration, stale-restore checks, and cache
  cleanup paths exist. Full caller coverage, collective failure ordering,
  direct-codec versus serving ownership, and publication consensus remain open.
- **P08/P09:** P08 remains source-contract DONE. P09 has the zero-scale FP8
  admission correction. The block128-FP8 native Numba consumer now enforces
  nonnegative finite scales and the canonical all-zero-tile rule for zero
  scales, in agreement with `device_packed_decode.py`. This closes that narrow
  consumer-validation mismatch only; the family/backend map, E8/IQ3 routes,
  scale and packing geometry, conversion cleanup, and all other execution
  consumers remain open.
- **P10/P11/P13/P15:** source owners and partial lifecycle/tooling paths exist,
  but full operation chains, ABI range/callback/reentry closure, physical-cache
  observation coverage, and final cross-area reconciliation remain open.

The earlier P02 source-matrix-contract promotion was overstated: generated
family/backend/source-dtype/persisted-format rows copied generic owner references
and labeled them `implemented-source` without proving that the exact tuple
reaches those owners. P02 is IN PROGRESS, not DONE. P01, P03, and P08 remain
task-level DONE entries; P04-P14 remain IN PROGRESS, and P15 remains OPEN. No
100% or Phase 1 completion claim is supported by this audit.

## Operating constraints

P05/P06/P08/P10/P12 source update: K3 tensor-parallel correctness capture now
observes the actual converted Torch operands before row/column shard selection.
Observation policy participates in rank agreement, capture errors use the
collective local-stage failure path, and complete assembled outputs remain
observed by the arithmetic owner. Alternate streamed BF16/packed banks keep
their separate capabilities. FP32 family/bank boundaries reject ambient
autocast and TF32; BF16 resident collectives require full-precision reduction.
Speculation preserves primary snapshot/rollback failures while retaining failed
releases. Scalable dispatch retains failed iterator/reservation cleanup and
keeps its admission quarantined until disposal succeeds. Indexed continuation
construction and index rollback retain primary failures. Tokenizer generation
identity checks descriptor and path stability. No execution checks were run.

P07/P08/P10 source update: exact K3 media admission filters recorded token/grid/
dtype/extent candidates before hashing source arrays, bounds input inventories,
requires explicit exact-mode selection and captures immutable buffers shared by
admission and execution. Animated containers cannot enter still-image or selected
frame preprocessing. Source receipt assembly preserves primary cleanup failures.
The six complete FP32 Torch family owners now reject implicit Metal CPU fallback
and undeclared fast math before construction and forward execution; common dense
projection and final-head boundaries retain the same policy. This policy reads
process settings only and grants no observed placement or platform certificate.
No runtime, imports, builds or tests were invoked.

P02/P03 scope source: original Mixtral configuration references are
[8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/41c1fe324a57f555d8df74b85a1353a583746532/config.json)
and [8x22B](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/blob/76fa59362cd95eb633a4e419c24364c90674966c/config.json).
Both pinned releases use unscaled rotary frequencies. Their metadata now has
an explicit offline source admission API and appears in family conformance;
existing linear/padding/selected-expert APIs are retained. No model was loaded.

P05/P08/P13 source update: autoregressive speculation drafts now preserve
primary proposal failures, retain failed snapshot releases, quarantine failed
rollback and stage restored logical buffers before changing the model. K3
media exact evidence requires every declared route/arithmetic leaf across
prefill, decode and both restoration trajectories with the same text/media
schemas. Exact evidence and pinned source reads reject linked/reparse paths
and descriptor identity changes. Qwen3.5 device cache restore enforces physical
capacity. Qwen3.5/Qwen4 projector captures observe the actual insertion call,
without a separate vision execution. No tests, imports or comparisons ran.

P09/P11/P12/P13 source update: generation/cache recovery manifest admission
now compares open descriptor identities before and after reads as well as the
path identity. Native DSA cache/execution identity includes the admitted
prebuilt kernel content; public Mixtral/Qwen3.5 and GLM/Inkling registrations
pin runtime/kernel sources before load and registration. Source inventory
rejects Windows reparse traversal alongside ordinary symbolic links. Expert
publication syncs parent directories for single-container commits and retains
primary failures with concrete staging cleanup paths. No tooling was invoked.

P03/P10/P13/P14 source update: Qwen3.5 physical context capacity now remains
separate from the publisher's semantic rotary-position bound in both native
and device owners. An additive versioned native context ABI keeps the original
counted-buffer layout and existing exports unchanged. Conditional loading and
wrapper construction retain the original text config for expert fingerprints;
cache allocation, occupied-token admission and recovery bind physical capacity.
Qwen3.5/Qwen4 media comparison source now compares actual full and chunked
continuation calls from occupied media snapshots, including complete terminal
physical caches and executed routing evidence. No comparison or import was run.

- Preserve unrelated changes. Do not reset, revert or remove existing work.
- Do not run tests, validation imports, builds, compilation, benchmarks or
  hardware verification. Reading source and editing files are permitted.
- Do not commit or push.
- Do not manufacture comparison evidence or enable numerical/platform support
  flags without evidence. Their remaining false is not a Phase 1 source gap.
- Do not use placeholders, injected-but-absent adapters, silent fallback,
  source-file presence or historical status prose as proof of completion.

## 1. Establish the exact source scope

- [x] Map all nine requested areas to their actual implementation, callers,
  configuration admission, ownership and public entry points.
- [x] Record each supported publisher family/variant and its pinned source
  contract. Identify required positional, media and quantization variants from
  those contracts rather than assuming every possible model option is required.
- [x] Reconcile historical completion notes with the current source. Explicitly
  distinguish missing implementation, deliberate contract restrictions and
  deferred execution evidence.

Done when every requirement below has identifiable source and no applicable
requirement remains classified as unknown.

## 2. Model-family execution and publisher mappings

- [x] Review complete inventory, tensor names, shapes, source dtypes, wrapper
  components, tied embeddings, routing, masks and positional contracts for
  Mixtral, Qwen3.5, Qwen4, GLM5-next, Inkling, GLM-DSA, K3 and V4.1.
- [x] Complete missing execution/mapping paths for the admitted publisher
  variants, preserving original model identities and explicit arithmetic policy.
- [x] Complete any additional Mixtral RoPE policies required by the admitted
  publisher contracts; default and linear policies already have source paths.
- [x] Ensure resident and selected/streamed loading reject incomplete or
  inconsistent whole inventories before reading selected payloads.

Done when every admitted family can be constructed through a concrete public
entry point with complete required source tensors and operations.

## 3. V4.1 vision, MTP, speculation and tensor parallelism

- [x] Trace image preprocessing, expansion, vision projection and text forward
  through native and device arithmetic; close missing operation dispatch.
- [x] Bind V4.1 expanded image token types to the aligner's downsampled output
  grid while retaining the full-resolution patch grid for vision encoding;
  validate exact row/newline geometry both at image-span construction and
  vision dispatch. Previously non-unit `downsample_ratio` produced raw-patch
  token spans that could not match aligner output, so those inputs failed.
- [x] Trace MTP proposal construction, verification, acceptance/rejection,
  stop handling, cancellation and rollback through concrete owners.
- [x] Review distributed BF16, FP8 and FP4 bank coverage across required text,
  expert, shared-expert and auxiliary projections; implement missing coverage.
- [x] Ensure all ranks agree on geometry, policy, source identities, operation
  order and continuation; retain explicit collective allocation budgets.
- [x] Complete persistence of target caches, draft caches, retained evidence,
  next logits, both RNG states and request token/media layout.

Done when these features compose through the same admitted serving/session
paths, including failure cleanup and continuation recovery.

## 4. K3 contracts, media and exact admission

- [x] Review occupied-token masks, positions, independent batch slots, media
  expansion and capacity admission across native, FLA and Metal execution.
- [x] Preserve complete MLA, KDA recurrent and convolution state across fork,
  rollback, snapshot, persistence and restore.
- [x] Review v2 text and v3 media evidence admission for complete observations,
  source/runtime pins, geometry, trajectories and bounded call admission.
- [x] Extend the single-fixture media evidence contract to the required set of
  image/video/layout cases with explicit case selection and per-case bounds.
  Unobserved cases must remain rejected; no evidence is generated in Phase 1.
- [x] Wire applicable admission options through factories and serving entry
  points, with consistent ownership and failure behavior.

Done when the required K3 contracts have concrete source paths and evidence
admission represents the required media cases without making parity claims.

## 5. Checkpoint and packed/quantized streaming

- [x] Review required format contracts, scale layouts, packing order, alignment,
  codebook/lattice/rotation identities and finite-value admission.
- [x] Complete required persisted model and selected-expert integration for
  native and device paths, including E8 and IQ3 transforms.
- [x] Ensure conversion/loading admits complete original inventory and binds
  config, weights and provider identity without materializing omitted banks.
- [x] Review bounded row/expert leases, cancellation, atomic publication and
  cleanup after partial reads, writes or conversion failure.

Done when each required format is connected to actual loading and execution,
not only a standalone decoder or conversion helper.

## 6. Accelerator decoder coverage

- [x] Trace each admitted CUDA/HIP/Metal decoder's complete token-to-logit path,
  including attention, routing, dense/shared/selected experts and final head.
- [x] Close required device operations and verify by source inspection that
  host work is explicit control/transport rather than implicit neural fallback.
- [x] Review device/version admission, dtype boundaries, bounded allocations,
  cancellation, transactional state updates and cleanup.
- [x] Update source capability mappings to point to concrete complete decoders
  while retaining false numerical/platform certification flags.

Done when backend coverage is supported by complete source call paths for the
admitted contracts. Real-machine execution remains deferred.

## 7. Native Segment/Edge ABI

- [x] Review family-owned adapters, registration and native capability/version
  agreement for all required families.
- [x] Review counted buffers, activation schemas, explicit masks/positions,
  embedding/head operations and hidden range boundaries.
- [x] Ensure every range carries all required continuation state and that
  transfer callbacks, cancellation, reentry and destruction honor ownership.
- [x] Complete any missing ABI source declarations/bindings and complete-cache
  transport without changing existing ABI layouts inadvertently.

Done when each required family has concrete compatible source on both sides
of the ABI and a public service construction path.

## 8. Serving integration

- [x] Fix V4.1 gateway recovery without a checkpoint sink: previously the gateway
  passes a positive publication budget that the session rejects when no sink is
  present. Keep recovery and publication budgets separate.
- [x] Review checkpoint capture/resume for every routed family, including token
  prefix semantics, sampling state, stopped boundaries and repeated recovery.
- [x] Extend decoded-media gateway routing to the media-capable family paths
  required by their contracts; currently decoded mixed media is K3-only.
- [x] Review warm-session reuse, configuration changes, cancellation,
  streaming/backpressure, tokenizer identity, error propagation and cleanup.
- [x] Ensure CLI, factory and gateway interfaces expose the implemented source
  features consistently and reject incompatible combinations before mutation.

Done when supported features are reachable end to end through serving, without
known argument mismatches or disconnected adapters.

## 9. Scalable cache, scheduling and recovery

- [x] Review cache paging/storage, aggregate budgets, leases, durable indexing,
  admission scheduling, eviction, fairness and cancellation under source-level
  ownership rules.
- [x] Preserve all family-specific physical state and alignment in recovery;
  do not substitute token replay for complete continuation.
- [x] Complete the rank-failure recovery orchestration required by the scope:
  explicit replacement of poisoned groups/rank owners, identity agreement and
  restoration from committed rank checkpoints before accepting new requests.
  Collective failure currently requires externally restarting ranks.
- [x] Review rollback consensus, quarantine, retryable cleanup, atomic storage
  publication and stale/corrupt/foreign checkpoint rejection.

Done when scalable continuation and failure recovery have concrete bounded
source paths, including the stated distributed lifecycle.

## 10. Source-only correctness and platform tooling

- [x] Update source inventory/capability mappings for every completed path.
- [x] Complete later-phase comparison adapters and canonical physical-cache
  schemas for the admitted families, media and quantization contracts.
- [x] Preserve explicit artifact pins, actual-observation requirements,
  placement contracts and rejection of missing/projected/empty evidence.
- [x] Keep tooling opt-in: Phase 1 must not trigger compilation, decoder imports,
  tests, benchmarks or hardware probes.

Done when the tooling source represents the implemented contracts and can
support a separately authorized verification phase without certifying them now.

## Final closure

P03/P11/P14 source update: native Qwen3.5 now separates an explicitly bounded
physical cache capacity from the original publisher maximum context. Range/Edge
loading, public session construction, CLI wiring, state allocation, cache restore
and executor identity agree on that capacity without changing expert-store config
fingerprints. Generic native loading exposes the capacity and retained-snapshot
budget. `resident_family_inventory.py` supplies complete pure header admission
for Mixtral, Qwen3.5, Qwen4, GLM-DSA, GLM5 and Inkling before resident payload reads.
Integer PLE descriptors remain separate from floating tensors; GLM5 source prefix
normalization rejects canonical collisions. No decoder was constructed or imported.

P13 source addition: Qwen4 media tooling now exports full QSA indexers, PLE token
and convolution histories, hybrid attention caches, actual masks/three-axis
positions and wrapper rotary continuation. Public native range frames export and
cross-check the same auxiliary state across every range. The Qwen4 observation
session shares only fixture/transaction/comparison transport with Qwen3.5 and
retains its own wrapper, reader, geometry and identity. Its pinned configuration
CLI compares projector outputs, complete continuation, routing where the owner
exports it and both restore trajectories. Oracle auxiliary access must be explicit
and observed. No imports, comparison, compilation or device checks were run.

P09 source update: expert packers now admit every mapped original layer, expert,
matrix view, dtype, lattice alignment and conversion budget before reading selected
payloads. Complete SafeTensors headers reject overlapping/missing ranges, invalid
byte geometry and changed source files. Floating container copies validate finite
values in bounded chunks, and all writes enforce actual progress. Alignment
staging is explicitly bounded. Single-source publication rechecks source identity.
Multi-container publication requires unused immutable names, publishes each member
exclusively and returns a manifest only after the full set is present. Failed-set
cleanup checks the created inode identity before removing owned members; existing
containers cannot be overwritten by a partially published set.
No packing operation, import or numerical verification was executed.

P06 source finding: aligned FP8 `wo_a` tensors could overwrite their logical
BF16 grouped bank assignment during V4.1 distributed registration. Packed FP8
registration now excludes `wo_a` consistently, retaining its original decoded
BF16 group-local execution through the BF16 bank or admitted resident owner.
The source byte encoding no longer changes that projection's arithmetic boundary.
BF16 collectives now agree on a separate arithmetic-policy identity that includes
the executed source/kernel policy, dependency identity and work admission while
excluding rank-local device ordinals. This closes the case where different rank
arithmetic policies could share only an empty device-library tuple.
No distributed or numerical execution was performed.

P08/P13 source addition: K3 has bounded opt-in capture of actual router logits,
selected IDs/weights and named local BF16 projection operands/results. Its
recorder captures values used by execution without recomputing routes or casting
fixtures. Text/media comparison factories require matching explicit observation
schemas and policies; media prefill, continuation and restoration compare those
leaves alongside projector outputs and complete physical caches. Unobserved or
borrowed-bank operands are rejected rather than represented by a local substitute.
Original K3 dense-row banks observe the first actual BF16 input shared by their
output-row tiles and the complete assembled boundary result through their bound
arithmetic owner. Banks lacking an executed operand hook remain rejected.
Recorder memory is included in the executor's transaction admission. No evidence,
comparison, import or device operation was executed in Phase 1.

P08/P13 source addition: `wholemodel_validation/k3.py` now exports physical FLA
FP32 recurrence and all original BF16 convolution taps alongside raw compressed
MLA pages. No latent projection, inversion or dropped history is used. Text and
media candidate factories select this layout explicitly; unchanged authoritative
readers must declare the same physical FLA policy. Structural empty declarations
and media observation budgets distinguish native kernel-minus-one histories from
full FLA convolution storage. The prior projected adapter remains a separate
comparison interface, outside exact physical-cache admission.
The same physical reader and candidate factories now admit an explicit Metal
recurrence policy with matching arithmetic/routing ownership; FLA and Metal
policies are mutually exclusive. No device probe or comparison was run.

P12 source addition: `scalable_execution/k3_compressed.py` now owns distinct
native/FLA versioned frames, preserving original BF16 convolution bits and FP32
recurrent tensors. Restore reconstructs each FLA tensor on its admitted device
and binds the full arithmetic/evidence policy. Media-bound codecs retain their
selected exact fixture in the layout identity and terminal snapshot. Explicit
`install_prefix` atomically seeds a real executed expanded prefix using observed
MLA rows and its complete terminal recurrence, without replay or invented prior
recurrent states. Seed and page budgets remain explicit. No execution occurred.

P12 source update: distributed generation restore retains a service rollback
snapshot until all ranks agree on restoration and complete logical boundary
identity. Failed or divergent restoration restores prior cache state and the
prior retained generation boundary before releasing rollback ownership.
Recovery manifest admission now rejects linked/reparse parent paths and bounds
rank/count geometry consistently. No distributed execution was performed.

P13 source addition: Qwen3.5 media tooling now has immutable original expanded
processor fixtures, complete wrapper sessions, projector and executed routing
observations, physical KV/convolution/recurrent exports and observed rotary delta.
The oracle delta requires an explicit pinned accessor. The comparison runner
covers full media prefill, greedy continuation and both restore trajectories;
its CLI requires a pinned configuration and publishes a report only on explicit
later-phase invocation. All original fixture buffers and adapter/loader sources
must be pinned. Numerical and platform certification remain false. This closes
the previously absent Qwen3.5 media observation path; exhaustive P13/P15 source
closure is still outstanding.

Current source change: `publisher_conditional_loader.py` now admits the complete
Qwen3.5, Qwen4 and Inkling text/media inventories, including omitted original
expert banks, streamed projections, lexical tables and generated integer
metadata, before reading payloads. Original PLE metadata and tied embedding/head
comparisons are deferred until that admission succeeds. `inkling_media.py` shares
one pure tower-shape planner between checkpoint admission and execution ownership.
This closes the discovered admission-order defect; exhaustive P03/P15 closure
remains open.

P13 source addition: `wholemodel_validation/qwen35_publisher_caches.py` exports
every occupied full-attention KV and linear convolution/recurrent leaf from
native/device owners or the pinned publisher cache. It checks batches, shapes,
finite values, explicit recurrent orientation, export budgets and the candidate's
unmasked text contract. `publisher_family_configuration.py`,
`publisher_family_candidate.py` and `publisher_family_caches.py` connect its
schema, reader and loader adapter to the opt-in comparison runner. No comparison
or import was executed. Media comparison coverage and final closure remain open.

P11 source update: `qwen35_public_adapters.py` registers independent Qwen3.5
hidden ranges and native embedding/head operations through the public dispatcher.
It reuses transport registration and the aggregate resource ledger, with separate
family wire and numeric identities. Each session retains complete KV, recurrent,
convolution and mask state; the bounded callback reader frames complete Qwen3.5
cache transport. `create_execution_service` connects registered ranges to serving.
This public wire is unmasked contiguous text; three-axis media retains the
wrapper's separate composition interface. Native ABI execution remains deferred.

P03/P10 source finding: native Qwen3.5 used a 67,108,864-element fixture ceiling
for immutable weights, which excluded the pinned publisher vocabulary matrix
and resident expert banks. `native_qwen35.py` and `native_qwen35.c` now use
address-space bounds for admitted immutable tensors, while retaining the separate
scratch allocation ceiling and explicit Python owner budgets. Embedding/head
output extents also reject address-space overflow. No native compilation or
execution was performed.

P10/P13 source update: Qwen3.5 device MoE now optionally exports the router logits
and selected weights actually used by each layer, with aggregate observation
admission and cancellation before state commit. Native execution exposes its
existing executed route traces through the same opt-in policy. The Qwen3.5
canonical reader and configuration connect those observations to complete routing
schemas. `qwen35_routing_oracle.py` uses explicitly pinned publisher module paths,
classes and returned-field indices; it never recomputes routes. Nonzero wrapper
media rotary continuation is rejected by the text-only cache comparison schema.

P08/P13 source finding: exact receipt readers previously accepted tensor
descriptors without checking occupied geometry, and media receipts could carry
projected cache metadata. `k3_execution/exact_admission.py` now requires complete
canonical numeric descriptors, permits zero extents only for explicitly declared
structural cache leaves, rejects projected caches, checks original RGB patch/grid
geometry, requires full expanded-prefix/decode vocabulary logits and accounts for
every projector row in the media expansion. `wholemodel_validation/k3_media.py`
rejects projected cache adapters before comparison. Existing v2/v3/v4 source
schemas remain recognized; incomplete receipts fail closed.

P10/P15 source update: native Qwen3.5 pins a stable prebuilt library before load,
checks that pin after binding/identity construction and on use, and still permits
cleanup after a source change. The tiny vision fixture's transactional text copy
now borrows the base library owner instead of sharing its destruction finalizer,
and its identity includes the actual merged embedding bytes. Kernel/source
changes and different fixture embeddings cannot share continuation identities.

- [x] Read the changed source and its callers for incompatible arguments,
  missing symbols, unsupported required branches, incomplete transport state,
  ownership leaks and accidental fallback. Fix findings without execution.
- [x] Mark each checklist item complete with concrete source references and a
  short explanation of the implemented behavior or confirmed existing path.
- [x] Confirm that no required implementation item remains open or unknown.
- [x] Report Phase 1 as source implementation complete only after those steps.
  Tests, numerical parity, compilation and platform/hardware certification are
  separate deferred phases and must remain explicitly unclaimed.

## Implementation breakdown and execution order

Use the task IDs below when resuming work. These tasks expand the requirements
above; they do not replace or narrow them. Start with the known integration
defect, then complete the scope matrix, implement missing paths and finish the
cross-area review. Keep existing implementations instead of rewriting an area
merely because its box is unchecked.

### Tracking rules

- Every task starts OPEN. Use IN PROGRESS while working and DONE only after its
  source completion criterion is satisfied. Identify blockers precisely without
  converting them into completion.
- For each task, record the concrete implementation files, callers inspected,
  changes made, source findings resolved and any remaining dependencies.
- Review both producer and consumer whenever a schema, function argument,
  native symbol, configuration field or ownership rule changes.
- Inspect ordinary, empty/first-call, continuation, stop, cancellation, failure
  and close paths by reading code. Do not execute those paths in Phase 1.
- A task that proves an existing implementation sufficient may be marked DONE
  with source references and reasoning; a new edit is not mandatory.
- Do not silently remove required variants, weaken admission, erase missing
  cache leaves or relabel a missing implementation as out of scope. Resolve
  uncertain requirements against the user's scope and pinned source contracts.
- The remaining-work list must include findings discovered during implementation,
  even when they belong to an earlier task.

### P01 — Repair the known V4.1 recovery argument mismatch

- [x] Trace `publisher_provider.py:resume_checkpoint` into
  `publisher_v41_cli.py:resume_generation` and `_checkpoint_options`.
- [x] Separate the input frame recovery budget from the optional output frame
  publication budget. A missing sink must not prevent bounded recovery.
- [x] Inspect capture-only, resume-only and resume-plus-publication callers,
  including CLI file handling and streaming recovery.
- [x] Keep invalid sink/budget combinations rejected before state mutation.

Closure: every caller passes compatible arguments for all three modes, and
resume-only has a complete source path. Dependency: none; implement first.

### P02 — Build a complete family/feature scope matrix

- [x] Create one entry for each family listed in section 2, including required
  wrapper/text identities and pinned publisher variants.
- [x] For each entry identify config admission, full inventory admission,
  resident loading, streamed loading, decoder, device decoder, range adapter,
  edge adapter, factory, CLI, gateway and cache codec.
- [x] List required attention/position variants, source dtypes, expert formats,
  media components, optional heads and actual runtime arithmetic policies.
- [ ] Classify each exact cell as IMPLEMENTED, MISSING, CANDIDATE-UNVERIFIED
  or CONTRACT RESTRICTION with source references. A family/backend/format
  Cartesian product and a resolvable owner symbol are not proof of a callable
  path; unknown exact tuples stay candidate/unverified until traced.

Closure: OPEN. The matrix enumerates candidate tuples and explicit
`REJECTED-UNADMITTED` persisted formats, but does not prove all family/backend/
format chains. Each admitted-looking tuple must be traced or given a concrete
enforced restriction. Dependencies: none.

### P03 — Close model-family loading and mapping gaps

- [x] Read `publisher_family_contracts.py`, `publisher_weight_mapping.py`,
  `publisher_conditional_loader.py`, `publisher_glm5_loader.py`,
  `publisher_checkpoint/` and each family loader/decoder selected by P02.
- [x] Trace original source names into runtime tensors, including fused/split
  experts, transposes, scale tensors, tied heads and optional component ownership.
- [x] Ensure every original tensor is admitted or explicitly owned by a required
  component; do not allow selected loading to hide bad omitted tensors.
- [x] Preserve checkpoint source dtype separately from execution dtype and bind
  both policies into the appropriate identities.
- [x] Implement required positional variants consistently in native/device
  decoders, hidden ranges, config fingerprints and recovery identities.

Closure: each P02 loading/mapping/position cell is implemented or is a justified
restriction of the actual publisher contract. Optional heads, tied/shared heads,
MTP/auxiliary namespaces, vision/audio/video ownership, expert-bank ownership,
and source-versus-execution dtype are explicitly admitted or rejected before
payload conversion. Dependency: P02.

### P04 — Complete V4.1 image and auxiliary execution

- [x] Trace `image_input.py` through `publisher_vision.py` / `device_vision.py`,
  `publisher_auxiliary.py` and `publisher.py` into final token logits.
- [x] Inspect multiple-image ordering, expanded token types, image position
  accounting, patch/grid geometry and workspace admission.
- [x] Trace `publisher_mtp.py` and `dspark.py` for target evidence capture,
  draft seeding, proposal positions and native/device arithmetic dispatch.
- [x] Close missing dispatch for operations actually reached by those paths;
  preserve the intended numerical boundaries rather than bypassing them.
- [x] Inspect reset, borrowed projection owners, auxiliary lifecycle and cleanup
  after an interrupted image or MTP request.
- [x] Retire an inner text snapshot if Qwen4 media snapshot publication fails;
  retain a failed retirement for owner close cleanup.

Closure: a concrete Qwen4-family caller chain is present: conditional factory
construction returns `MultimodalSessionExecutor`; `NativeExecutionService`
admits the media prompt as one nonempty prefill operation, captures the model
snapshot, dispatches `forward_media`, and restores that snapshot on any
exception/cancellation before committing token history; the executor bounds
and owns staged arrays; `Qwen4MultimodalDecoder.forward` checks placeholder
types against processor types, pairs image/video arrays with grids, computes
merged-grid positions, scatters vision features into token embeddings, and
passes embeddings plus position IDs to text forward for logits. This closes
that source trace only. The analogous per-family producer-to-final-logit
trace, exact processor geometry/order contracts, cancellation/cleanup through
every public caller, and borrowed-owner behavior for all admitted families
remain unverified; P04 remains IN PROGRESS.
Dependencies: P02, P03.

### P03 bounded direct-wrapper header admission — 2026-10-13

`native_checkpoint.load_native_multimodal_decoder(..., model_type="qwen3_5_moe")`
passes a metadata-only admission callback into `_read_native_checkpoint`.
Before any tensor payload is read, the callback checks the supported tiny
Qwen3.5 wrapper geometry and requires the exact vision tensor-name/shape set
for its declared depth. Header byte ranges and limits continue to be enforced
by the existing reader. The publisher conditional loader is a separate path
with its own family inventory planner and was not changed. The direct native
Qwen4 and Inkling wrapper choices now fail before checkpoint payload loading;
they must use that conditional loader. This is a concrete direct-path
restriction, not a claim that those families are unsupported overall. The
Qwen3.5 direct preflight does not cover its text tensor namespace; family
execution/caller closure remains tracked under P04/P09/P10/P14.

### P05 — Complete V4.1 speculation and durable generation

- [x] Read `speculative.py`, `dspark_cache.py`, `mtp_cache.py`,
  `speculative_cache.py`, `publisher_checkpoints.py` and
  `v41_generation_recovery.py` together with their session callers.
- [x] Trace accepted proposals, rejection/replacement, bonus tokens, EOS,
  context limits and cancellation to committed target/draft state.
- [x] Bind history, token types, next logits, verifier RNG, proposal RNG, target
  evidence, draft window and generation policy to the complete persisted frame.
- [x] Inspect frame limits, checksums, exact EOF, tokenizer identity and physical
  cache agreement before publishing recovered continuation.
- [x] Preserve synchronous single-use checkpoint writers and independent
  ownership of caller streams/storage.

- [ ] Verify the serving-caller acceptance boundary and cancellation/reentry
  ordering across checkpoint restore followed by continuation; core speculative
  state primitives alone do not close this integration.

P05 serving-caller audit (2026-10-13): the only direct DSpark serving entrypoints
found in Python source are the V4.1 publisher CLI/session `run` and
`PublisherSession.resume_generation`; no gateway/provider route currently admits
speculation. Fresh requests use independent target/proposal RNG streams, prefill
before generation, and synchronously checkpoint committed decoder boundaries
before token events. The fresh CLI now passes its cancellation predicate into
`SpeculativeDecoder.prefill`; previously cancellation was checked only before
and after prompt evaluation, so a long/interruptible target forward could finish
before observing request cancellation. Prefill's transactional target snapshot
now handles that predicate and rolls back on cancellation. Resume restores both RNG states and the compound cache,
checks history/logits/stopped-boundary agreement, and reuses the checkpointed
EOS/stop-prefix policy rather than caller-supplied replacements. The decoder
transaction bounds proposals by remaining context, restores both owners on
rejection/early stop, and rolls back on cancellation/errors. This is source
tracing, not executed evidence. Remaining: source-level failure injection and
cleanup review around failed checkpoint writes, event callbacks, and partial
restore; audit any future gateway caller before exposing DSpark there. P05 stays
IN PROGRESS; do not infer all-caller closure from the CLI path.

Closure: core ordinary/speculative capture and rollback primitives are present;
caller-by-caller capture/resume and cancellation closure remains open without
replay or discarded auxiliary state. Dependencies: P01, P04.

### P06 — Complete model-wide tensor parallel composition

- [x] Map every required projection to a concrete bank or justified replicated
  operation, including expert/shared-expert, grouped attention and auxiliary paths.
- [x] Review `tensor_parallel.py`, `bf16_tensor_parallel.py`,
  `packed_tensor_parallel.py`, `projection_dispatch.py` and
  `v41_distributed_session.py` for full coverage of that map.
- [x] Inspect uneven partitions, empty local shards where applicable, format
  alignment, input agreement, rank-independent global identity and local placement.
- [x] Keep collective schedule, cancellation, work budgets and result assembly
  consistent across ranks; close missing service/launcher interfaces.
- [x] Record cache ownership and whether each state leaf is replicated or sharded
  so P12 can restore the same rank layout.

`v41_distributed_session.make_projection_backend` now adds unanimous control
gates around rank-local BF16/FP8/FP4 bank setup, FP4 provider construction,
packed-kernel creation, and final dispatch construction. Local setup errors are
gathered before peers can enter the next bank collective; locally-created
resources are closed on a peer failure. Per-source registration still performs
its existing rank agreement before use. This closes setup-exception ordering
only. `TensorParallelProjections` and the packed/BF16 subclasses now quarantine
a bank if a control or result transport collective raises, so uncertain rank
state cannot be reused after a reported transport failure. Process loss that
leaves a collective blocked, transport recovery, rank restart, and service-level
consensus rollback remain outside this source contract and P06 stays IN PROGRESS.
Focused operation trace (2026-10-13): the dense row-sharded `linear` path in
`TensorParallelProjections.linear` validates replicated-input digests and
registered shard identity on every rank, synchronizes local product/allocation
failures before `all_gather`, gathers padded row extents, trims each rank to
deterministic `shard_bounds`, synchronizes row assembly, checks finite output
with a collective minimum, and synchronizes host-export failures. Cancellation
is checked at synchronized local-stage boundaries and now remains an
`InterruptedError` when propagated to every rank; the bank is quarantined after
that failed stage. Control/result transport exceptions also quarantine the
bank. This confirms ordered source-level failure
boundaries for this one dense row-projection operation only; it does not close
all formats/banks or lost-process/blocked-transport recovery. The focused trace
did not justify changing the collective implementation. P06 inventory,
cancellation, callback, process-loss and recovery findings remain open.
Dependencies: P02–P05.

### P07 — Complete K3 input, batch and media contracts

- [x] Trace `k3_execution/executor.py`, `batch.py`, `preprocessing.py`,
  `video_input.py`, vision code and factory/session callers.
- [x] Inspect occupied-token masking, signed/unsigned IDs, explicit positions,
  placeholder ordering, image/video expansion and expanded capacity accounting.
- [x] Preserve per-slot atomicity and independent state ownership; do not call
  independent slot batching a fused accelerator batch.
- [x] Review native recurrence, `fla_mixer.py` and `metal_mixer.py` for compatible
  full recurrent/convolution snapshots, forks, restore and cancellation.

The reviewed primitives have explicit admission and rollback paths, but this is
not complete caller closure. The executor now closes replaced FLA mixer caches
when reset, restore or forward commits a forked recurrent state. If retirement
fails, the mixer stays owned in a retry list and the executor is closed to new
operations until close retries cleanup. Forward computation now retires its
staged FLA forks on execution failure/cancellation; reset does the same if
staging fails. Restore now stages each FLA fork into owned transaction state
before applying its snapshot and retires all staged forks if inventory
validation, cancellation, or mixer restore fails. Cleanup errors are attached
as notes to the original exception; failed mixer closes remain owned for
executor-close retry and quarantine new operations. FLA cancellation is polled between
convolution calls and after recurrence (kernels already dispatched cannot be
interrupted). `K3BatchExecution.forward` is now explicitly
text-only and rejects media expansion; media must use an individual K3 slot.
The native conditional service also rejects empty or multi-chunk media prompts:
without a media-span planner it cannot safely replay one processor bundle over
multiple text chunks. K3 expanded positions are now validated before narrowing
to a tensor integer type, preventing uint8/int8 wraparound from turning an
out-of-range media expansion into an apparently admissible rotary coordinate.
K3 video temporal positions are currently frame-index based; elapsed timestamp
values are not consumed by the tower. `video_input.py` therefore rejects
nonuniform timestamp intervals both when decoded video is admitted and when
prepared-media metadata is validated. This makes uniformly sampled frames the
only admitted video timing contract, with timestamps retained as provenance;
it does not claim elapsed-time temporal rotary positioning. End-to-end K3
media callers, broader temporal semantics, native/FLA continuation and
serving caller reconciliation remain open. P07 remains IN PROGRESS.
Dependencies: P02, P03.

### P08 — Complete K3 exact-admission source interfaces

- [x] Review `exact_admission.py` with `wholemodel_validation/k3_media.py`,
  K3 canonical cache readers and the executor/factory consumers.
- [x] Define the required multi-case evidence schema without manufacturing any
  receipt: stable case IDs, processor fixture identities, original tokens,
  expanded positions, grids, media selection and per-case continuation bounds.
- [x] Bind every admitted case to recorded complete logits, projector outputs,
  physical cache leaves, routing/quantization observations required by its
  contract, generation and continuation restoration.
- [x] Keep v2/v3 compatibility explicit or provide a versioned migration path;
  reject ambiguous, duplicate, foreign, partial or unobserved cases.
- [x] Expose case selection and exact policy through the actual factory and
  serving paths, with no global claim beyond the admitted evidence.

Closure: DONE for the P08 source contract. Exact admission can represent and
select all required evidence cases; execution remains evidence-gated. P07 and
P13 remain owners of their separate integration and observation-coverage work.

Source closure references: `K3ExactEvidence` admits v2 text, v3 single media and
v4 unique multiple cases with complete descriptor/check coverage; selectors bind
immutable original typed buffers and contiguous occupied positions before actual
execution. `K3Execution` carries selected evidence through snapshots, complete
cache/generation recovery and compressed scalable terminal state. Factories/CLI
require explicit mode and pinned evidence. Actual native/FLA/Metal cache readers
and local/distributed operand hooks produce the required comparison interfaces.
`assemble_media_evidence` combines already recorded reports only. All runtime
verification and evidence creation remain deferred; those deferred checks do
not reopen the implemented P08 source schema or admission contract.

### P09 — Complete packed and quantized checkpoint integration

- [x] Enumerate required formats from P02 and map each to packing, persisted
  metadata, native decode, device decode, projection/expert execution and loading.
- [x] Read `expert_packing.py`, `expert_store.py`, `publisher_tensor_reader.py`,
  original row/expert providers, `device_packed_decode.py`,
  `device_iq3_rotation.py`, E8/IQ3 contracts and family streamed providers.
- [x] Inspect byte extents, row/group ordering, scales, zero/nonfinite codes,
  lattice/codebook/rotation pins and format-specific alignment.
- [x] Ensure omitted expert banks remain absent from model residency while
  original whole-inventory/config identity is still admitted.
- [x] Inspect lease release, partial conversion cleanup, atomic manifest/container
  publication and changed-source rejection.

Closure: every required format has a complete bounded persisted-to-execution
source path in each applicable family/backend. Dependencies: P02, P03.

Concrete enforced restriction (2026-10-13): `TorchMixtralExperts.__init__`
rejects `e8-lattice-caller-v1` before admitting the manifest/provider. The
device decoder transforms E8 codebook rows into the rotated basis, while the
Mixtral device consumer only applies activation rotation for IQ3; allowing E8
would therefore multiply incompatible bases. E8 remains unsupported for this
device consumer until activation rotation, bounded scratch, cancellation, and
failure cleanup are implemented as one path. This restriction does not close
P09/P10 for other family/backend/format rows.

### P10 — Close accelerator operation and ownership gaps

- [x] For every admitted family/backend, write its token-to-logit operation chain
  from embedding through final normalization/head, including all cache updates.
- [x] Read the relevant `torch_*_decoder.py`, device expert/projection owners,
  K3 BF16/Metal paths and V4.1 device arithmetic against that chain.
- [x] Inspect runtime version/device/dtype admission and reject implicit fallback
  or mismatched placement before neural operations.
- [x] Close required operations and retained-state transport; bound weights,
  attention, selected experts, output rows and snapshot staging.
- [x] Review cancellation, failure-before-commit and cleanup-after-failure source
  paths without loading a runtime or probing a device.

Closure: every required accelerator chain is complete in source, including
continuation and ownership. Dependencies: P03, P04, P07, P09.

### P11 — Complete native Segment/Edge composition

- [x] Map native declarations to Python bindings and registrations in
  `native/`, `segment_edge_runtime.py` and each family public adapter.
- [x] Match version/capability checks, counted buffers, dimensions, dtypes,
  masks, positions, hidden schemas and optional expert callbacks.
- [x] Inspect embedding/head edges and hidden ranges for original layer indexing
  and all required state, including index-sharing range boundaries.
- [x] Inspect callback lifetime, owner-thread/reentry rules, snapshot transfer,
  cancellation, failed construction and session-before-engine destruction.
- [x] Wire each required adapter into a concrete service factory and CLI path.

Closure: every required family has complete source on both sides of its ABI and
an integrated service path. Dependencies: P03, P07, P09.

### P12 — Complete scalable cache, scheduling and recovery

- [x] Read `scalable_execution/`, `admission_scheduler.py`, `executor_budget.py`,
  continuation stores, family cache codecs and `distributed_execution.py`.
- [x] Inspect aggregate reservations, cache pages, active leases, queue order,
  cancellation, eviction and durable index mutation for bounded ownership.
- [x] Bind complete physical state, layout, capacity, identity and occupancy to
  serialization; retain recurrent, convolution, compressed and pending state.
- [x] Implement the missing rank-recovery lifecycle with explicit caller-owned
  replacement resources, checkpoint agreement and restored-rank consensus.
- [x] Keep poisoned groups unusable, reject stale/foreign checkpoints, and admit
  new requests only after the replacement execution set is complete.
- [x] Inspect rollback/quarantine and retained cleanup retries across storage,
  sessions, adapters and distributed owners.
- [x] Preserve the primary generation-recovery failure while attempting both
  staged-cache close and rollback-snapshot release in
  `execution_service.NativeExecutionService.restore_generation_checkpoint`.
- [x] K3 generation codec direct publication now holds `model._lock` across
  boundary validation against the live position and physical cache streaming;
  direct restore holds the same owner lock across pre-restore snapshot,
  cache replacement, boundary-position validation and rollback. Both codec
  operations accept cooperative cancellation, with restore cancellation after
  mutation taking the transactional rollback path.
- [ ] Verify cancellation/reentry ordering and physical cache ownership across
  every family codec, serving caller and distributed replacement path.

Closure: transactional cache/recovery primitives and bounded rank replacement
exist; family/caller ownership, cancellation ordering and distributed failure
closure remain open. Dependencies: P05–P07, P09–P11.

### P13 — Complete source-only correctness and platform tooling

- [ ] Map every family/backend/media/format contract to its later-phase adapter,
  canonical observation/cache schema and explicit placement requirements.
- [x] Review `wholemodel_validation/` and `platform_certification/` for missing
  comparison bindings, state leaves, artifact pins and candidate entry points.
- [x] Update tooling source for changed evidence/ABI/cache schemas while keeping
  schema readers strict and version compatibility explicit.
- [x] Keep source inventory and capability mappings accurate without treating
  file existence as execution evidence.
- [x] Preserve opt-in execution boundaries and false certification flags; do not
  invoke this tooling while implementing Phase 1.

Closure: source tooling covers the required contracts and cannot manufacture
success from missing or projected evidence. Dependencies: P02; revisit after
P03–P12 and P14. P08's schema design may proceed alongside this task.

### P14 — Complete serving feature routing and lifecycle

- [ ] Trace configuration → parser → loader → factory/session → generation →
  stream/checkpoint/recovery → cleanup for each P02 family entry.
- [x] Extend required decoded image/video/media routing beyond the current K3
  restriction using each family's actual processor and decoder contract.
- [x] Preserve caller media ordering, placeholder ownership and expanded limits;
  reject genuinely incompatible media rather than silently converting it.
- [x] Inspect warm reuse identities, changed config/artifacts, idle/lifetime
  release, cancellation, bounded streaming queues and tokenizer prefix handling.
- [ ] Reconcile all option names/defaults across CLI, session, factory and gateway,
  especially recovery-only versus recovery-plus-publication.

Closure: every required feature is reachable through its serving entry point
with consistent lifecycle and argument contracts. Dependencies: P01, P03–P12;
feed new schema/entry-point requirements back into P13.

### P15 — Final source closure; only then mark Phase 1 complete

- [ ] Revisit every top-level requirement and P02 matrix cell; resolve all MISSING
  and unknown entries that belong to the requested scope.
- [ ] Review the changed implementation and callers for stale symbols, argument
  mismatches, missing required branches and schema/version incompatibilities.
- [ ] Review persistent state inventories against forward state updates and
  restore consumers so no continuation leaf is discarded.
- [ ] Review allocation admission, cancellation, rollback, reentry and close
  paths for every newly composed owner.
- [ ] Update every completed task with concrete source evidence; leave no
  unsupported completion percentage, placeholder or unexplained exclusion.
- [ ] Confirm the user prohibitions were respected and unrelated changes remain
  preserved. Do not use tests/imports/builds to close source tasks.

Closure: P01–P14 and all original scope requirements are DONE with source
evidence, no required implementation gap remains, and certification is still
explicitly deferred. This is the sole point at which Phase 1 source completion
may be reported as 100%.

## Work log format

### Implementation sequence

This document serves as the implementation ledger. Completion requires all
applicable Phase 1 source requirements to be implemented and reviewed.
Planning and documentation alone do not establish completion.

1. Read the latest work log and preserve completed changes. Do not infer DONE
   from informal progress reports, file existence, or a historical status table.
2. Start with P01, then P02. Resolve dependencies before changing downstream
   interfaces. Work through P03–P12, integrate P14, reconcile P13, then P15.
3. For the active task, identify the exact missing behavior and trace its callers
   before editing. Record the concrete change needed, not just "review this".
4. Implement it completely: admission, operation, state updates, public wiring,
   cancellation, rollback, persistence and close paths where applicable.
5. Read both sides of every changed interface. If that reveals another gap,
   add it to this ledger and implement it under its owning task.
6. Close the task with source references and remaining dependencies. Continue
   with the next dependency-ready task rather than ending at a partial result.
7. Do not report 100% until P15 closes every required matrix cell and finding.
   Never execute prohibited checks to obtain a completion mark.

### Concrete source map for each task

All implementation paths below are relative to `D:\openagent\openagent\adaptive_engine`.
These are starting points for source inspection, not an exhaustive list or proof
that the task is complete. Follow imports and callers to find additional owners.

| Task | Implementation starting points | Consumers that must be reconciled |
| --- | --- | --- |
| P01 | `publisher_v41_cli.py`, `v41_generation_recovery.py` | `publisher_provider.py`, `generation_checkpoint_files.py` |
| P02–P03 | `publisher_family_contracts.py`, `publisher_weight_mapping.py`, `publisher_conditional_loader.py`, `publisher_glm5_loader.py`, `native_checkpoint.py`, `publisher_checkpoint/`, `mixtral_inventory.py` | Family constructors, selected providers, `native_family_cli.py`, `execution_factories.py` |
| P04 | `v41_execution/image_input.py`, `publisher_vision.py`, `device_vision.py`, `publisher_auxiliary.py`, `publisher_mtp.py`, `dspark.py` | `v41_execution/publisher.py`, `device_execution.py`, `publisher_v41_cli.py` |
| P05 | `v41_execution/speculative.py`, `speculative_cache.py`, `dspark_cache.py`, `mtp_cache.py`, `publisher_checkpoints.py`, `v41_generation_recovery.py` | Session run/resume methods, gateway streams, atomic checkpoint writer/readers |
| P06 | `tensor_parallel.py`, `bf16_tensor_parallel.py`, `packed_tensor_parallel.py`, `projection_dispatch.py`, `v41_distributed_session.py` | Native/device arithmetic, grouped projections, auxiliary providers, `distributed_execution.py`, `distributed_launcher.py` |
| P07–P08 | `k3_execution/executor.py`, `batch.py`, `preprocessing.py`, `video_input.py`, `fla_mixer.py`, `metal_mixer.py`, `exact_admission.py`, `checkpoints.py` | `publisher_k3_cli.py`, `execution_factories.py`, `wholemodel_validation/k3_media.py`, cache readers |
| P09 | `expert_packing.py`, `expert_store.py`, family streamed expert providers, device packed decoding/rotation, `publisher_tensor_reader.py`, `publisher_checkpoint/` | Decoder projections/experts, provider identities, conversion entry points, cache restoration |
| P10 | `torch_mixtral_decoder.py`, `torch_qwen35_decoder.py`, `torch_qwen4_decoder.py`, `torch_glm5_decoder.py`, `torch_glm_dsa_decoder.py`, `torch_inkling_decoder.py`, K3 BF16/Metal code, V4.1 device code | Device factories, provider selection, packed providers, decoder capability mappings |
| P11 | `native_family_public_adapters.py`, family public adapters and ranges, `segment_edge_runtime.py`, `native/segment_edge_compat.c`, `native/segment_family_models.c` | Edge/range registration, service factories, callbacks, complete snapshot transport |
| P12 | `scalable_execution/`, `admission_scheduler.py`, `executor_budget.py`, continuation stores, family cache codecs, `distributed_execution.py` | `execution_service.py`, launcher/replacement owners, durable checkpoint publication and recovery |
| P13 | `wholemodel_validation/`, `platform_certification/`, `family_conformance.py` | Changed decoder/evidence/cache/ABI contracts and source inventory |
| P14 | `publisher_provider.py`, family CLIs, media processors, `execution_factories.py`, `execution_service.py` | Model gateway interfaces, checkpoint streams, warm sessions, lifecycle cleanup |
| P15 | Every changed implementation and its reachable callers | Entire family/feature matrix, unresolved findings and task closure log |

### P02 matrix: one row per required family/variant/backend combination

Create separate rows when wrappers, source formats or backend policies change
the implementation contract. Never collapse a missing path into another family's
working path. Use the following columns and fill every cell with a source symbol
and status; a blank cell remains unresolved.

```text
Family / wrapper / publisher revision:
Backend and explicit arithmetic policy:
Configuration and required feature variants:
Complete tensor inventory and source dtype admission:
Resident loader:
Selected/streamed loader and required packed formats:
Embedding, attention, routing, experts, normalization and head:
Required vision/audio/video/MTP components:
Tensor-parallel projection and cache ownership:
Native Segment / Edge registration and bindings:
Factory / CLI / gateway entry points:
Snapshot, complete physical cache and generation recovery:
Source correctness / platform tooling adapters:
Remaining source gaps and owning task IDs:
Status and concrete closure references:
```

Start with Mixtral, Qwen3.5, Qwen4, GLM5-next, Inkling, GLM-DSA, K3 and V4.1.

The following source rows identify the concrete owners to trace. They remain
IN PROGRESS until the detailed cases and finding ledger below are reconciled;
these entries do not promote backend or numerical certification.

| Family / contract | Inventory and loading | Native / device execution and selected storage | Media / auxiliary ownership | Public integration and recovery | Source correctness tooling |
| --- | --- | --- | --- | --- | --- |
| Mixtral, exact fused/split F32/F16/BF16; default/linear RoPE | `mixtral_inventory.admit_mixtral_inventory`, `native_checkpoint.read_native_text_checkpoint` | `native_mixtral_decoder.NativeMixtralDecoder`, `torch_mixtral_decoder.TorchMixtralDecoder`, `mixtral_streamed_experts`, `torch_mixtral_experts` | Text contract; pinned original 8x7B/8x22B use default RoPE, P03-F01 closed for those sources | `mixtral_public_adapters.NativeMixtralPublicAdapters`, `native_mixtral_ranges`, `mixtral_cache_recovery`, `torch_mixtral_cache`, `execution_factories` | `wholemodel_validation.runner`, `platform_certification.decoder_matrix`, `family_conformance` |
| Qwen3.5 publisher revision in `PUBLISHER_VARIANTS`; text and wrapper identities separate | `resident_family_inventory`, `publisher_family_contracts.component_contract`, `publisher_conditional_loader` | `native_qwen35.NativeQwen35TextDecoder`, `torch_qwen35_decoder.TorchQwen35Decoder`, `qwen35_publisher_experts`, `qwen35_streamed_experts`, `device_publisher_projections` | `qwen35_multimodal.Qwen35MultimodalDecoder`; variable-grid vision, interleaved MRoPE, occupied mask, separate physical capacity; original auxiliary tensors retained | `qwen35_public_adapters` text ranges/Edge, `native_qwen35_ranges`, `qwen35_cache_recovery`, `torch_qwen35_cache`, wrapper cache framing, conditional CLI/factory | `wholemodel_validation.qwen35_publisher_caches`, `qwen35_routing_oracle`, `qwen35_media`, `qwen35_media_cli` |
| Qwen4 publisher revision in `PUBLISHER_VARIANTS`; linear/QSA/PLE | `native_qwen4_ranges.qwen4_tensor_inventory`, `publisher_conditional_loader`, `publisher_weight_mapping` | `native_qwen4.NativeQwen4TextDecoder`, `torch_qwen4_decoder.TorchQwen4Decoder`, original/packed expert owners, `qwen4_ple_store`, device projection bank | `qwen4_multimodal.Qwen4MultimodalDecoder`, `qwen4_vision`, native composition, three-axis positions and lexical histories | `native_qwen4_public_adapters`, `native_qwen4_public_text`, native counted public sessions, conditional CLI/factory, full QSA/PLE cache framing | `wholemodel_validation.qwen4_publisher_caches`, `native_qwen4_caches`, `qwen4_media`, `qwen4_media_cli` |
| GLM5-next publisher revision in `PUBLISHER_VARIANTS`; KDA/MLA/KPool/MHC | `resident_family_inventory`, `publisher_glm5_config`, `publisher_glm5_loader` | `native_glm5_next.NativeGlm5NextTextDecoder`, `torch_glm5_decoder.TorchGlm5NextDecoder`, native/device publisher expert owners | `glm5_multimodal.Glm5MultimodalDecoder`, own vision and temporal packing | `native_family_public_adapters`, `native_family_public_text`, native range APIs, `execution_factories`, conditional CLI and physical cache transport | `wholemodel_validation.publisher_family_configuration`, `publisher_family_candidate`, `publisher_family_caches` |
| Inkling publisher revision in `PUBLISHER_VARIANTS`; hybrid/sliding/FIR | `resident_family_inventory`, `publisher_conditional_loader`, `inkling_media.media_shapes` | `native_inkling.NativeInklingDecoder`, `torch_inkling_decoder.TorchInklingDecoder`, native/device original publisher experts | `inkling_multimodal.InklingMultimodalDecoder`, image/audio owners, explicit bounded MTP draft owner | `native_family_public_adapters`, `native_family_public_text`, native range APIs, conditional CLI/factory, owned complete continuation | `wholemodel_validation.publisher_family_configuration`, `publisher_family_candidate`, `publisher_family_caches` |
| GLM-DSA exact `glm_moe_dsa`; default/static YaRN | `resident_family_inventory`, `native_glm_dsa_ranges.admit_inventory`, selected checkpoint callbacks | `native_glm_dsa.NativeGlmDsaDecoder`, `torch_glm_dsa_decoder.TorchGlmDsaDecoder`, native/device original/packed selected expert owners | Family text contract, latent/rotary/indexer cache, independent occupied masks | `glm_dsa_public_adapters`, `native_glm_dsa_ranges`, `native_glm_dsa_cache`, `cache_file_recovery`, family CLI/factories | `wholemodel_validation.glm_dsa_caches`, full candidate/oracle runner |
| K3 pinned original profile, explicit native/FLA/Metal policy and exact evidence | `publisher_checkpoint.k3_adapter`, `k3_execution.config`, `publisher_k3_cli` | `k3_execution.K3Execution`, BF16/device arithmetic, native/FLA/Metal recurrence, leased packed/original projections | Own RGB/video expansion, MLA/KDA state, selected observed media case, v2/v3/v4 evidence admission | Execution factory/gateway, full cache/generation transport, `scalable_execution.k3_compressed`, atomic executed-prefix installation | `wholemodel_validation.k3`, `k3_media`, actual route/arithmetic recorder, `exact_evidence_files` |
| V4.1 pinned original profile; BF16/FP8/FP4 logical execution boundaries | `publisher_checkpoint.v41_adapter`, `v41_profile`, publisher tensor providers | `v41_execution.PublisherV41Execution`, `DevicePublisherV41Execution`, BF16/packed tensor-parallel banks, original selected experts | Own vision, Engram/attention state, publisher MTP blocks, DSpark target/draft and verification | `publisher_provider`, V4.1 CLI/gateway, complete generation frames, distributed boundary consensus and fresh-rank replacement recovery | `wholemodel_validation.v41_publisher`, `v41`, command/placement tooling with false certification flags |
For a CONTRACT RESTRICTION, cite the actual publisher contract and explain why
the excluded feature is not required. An implementation rejecting a feature is
not sufficient evidence that the feature is outside the requested scope.

### Cross-area cases that must be inspected in source

For each relevant P01–P14 task, record which cases below apply and where their
behavior is implemented. These are source-reading cases, not instructions to
run tests, instantiate decoders, import validation modules or probe hardware.

| Case | Source closure requirement |
| --- | --- |
| Initial request | Inputs, artifacts and budgets are admitted before mutation; required owners exist. |
| Continuation | Token positions, mask occupancy, all state leaves and next-logit policy stay consistent. |
| Boundary input | Empty/padded inputs, capacity limits and geometry edges either have defined behavior or reject explicitly. |
| Media expansion | Ordering, placeholders, processor layout, expanded positions and capacity agree across producer and decoder. |
| Packed projection | Format, scales, codebook/rotation, row/expert selection and work budgets agree with the execution consumer. |
| Speculative rejection | Target, draft, retained evidence and RNG rollback preserve the same committed boundary. |
| Stop boundary | EOS/stop tokens, cached token history and saved next-token state agree; resume does not generate past an already stopped boundary. |
| Cancellation | Work stops at explicit boundaries; published state remains consistent and owned resources remain releasable. |
| Read/write failure | Partial frames and leases cannot become live state; staging resources are released or retained for explicit cleanup retry. |
| Corrupt/foreign recovery | Complete frame, identity, policy, layout, checksums and EOF are admitted before live restoration. |
| Distributed mismatch | Rank identities, operation sequence and restored continuation agree; poisoned resources cannot accept requests. |
| Failed construction | Every resource created before failure is released or attached to a retryable cleanup owner. |
| Reentry and close | Active operations prevent conflicting mutation/destruction; borrowed resources are not destroyed by borrowers. |

### Finding ledger and completion evidence

Use stable finding IDs such as `P05-F01`. Each finding must record its concrete
symptom, source location, required behavior, implementation change, consumers
affected and closure references. Findings discovered later remain open until
resolved even if their parent task was previously marked DONE; reopen that task.

The following known items seed the ledger and remain unclosed here:

| Finding | Known behavior / required next action | Owner |
| --- | --- | --- |
| P01-F01 | CLOSED: `publisher_provider.resume_checkpoint` passes zero output publication budget without a sink while retaining the explicit input recovery budget. | P01 |
| P14-F01 | IMPLEMENTED, closure review pending: decoded still images reach V4.1/Qwen3.5/Qwen4/Inkling/GLM5 owners; explicit frame-index/FPS videos reach Qwen3.5/Qwen4/GLM5; K3 retains its own temporal packing contract. | P02, P14 |
| P03-F01 | CLOSED for the pinned original Mixtral publisher scope: 8x7B revision `41c1fe324a57f555d8df74b85a1353a583746532` and 8x22B revision `76fa59362cd95eb633a4e419c24364c90674966c` both declare default RoPE with theta 1e6. `mixtral_inventory.MIXTRAL_PUBLISHER_VARIANTS` records exact geometry and source pins; `publisher_contract` admits explicit offline source selection. Existing broader default/linear bounded contracts remain available. These publisher pins do not require another RoPE equation. | P02, P03 |
| P08-F01 | CLOSED for P08 source: v4 supports unique pinned cases, individual observed bounds, owned case continuation and persistence; the receipt assembler preserves existing observations and strict pins; factory/media/CLI paths require explicit pinned evidence and case selection. | P07, P13 (remaining integration owners) |
| P12-F01 | IMPLEMENTED, closure review pending: replacement owners restore pinned cache or generation rank sets before root work; retries use bounded explicitly supplied fresh rendezvous epochs. | P06, P12 |
| P15-F01 | The complete nine-area source closure review has not been established; fill all matrix cells and resolve subsequent findings. | P02, P15 |
| P02-F02 | IMPLEMENTED, closure review pending: `source_scope.py` uses the `publisher_checkpoint.v41_profile` namespace and now expands detailed backend, arithmetic, source-format, publisher-pin and persisted-route cases with structural policy agreement checks. Remaining work is cross-reference review, not the missing namespace/catalog branch. | P02, P13 |
| P12-F03 | IN PROGRESS: K3 restore now preflights detached staging and V4.1 packed checkpoint admission includes the concrete device executor. Finish direct-codec lock/reentry/cancellation ordering and review these changes through serving, file transport and distributed consumers before closure. | P05, P10, P12 |
| P03-F02 | IMPLEMENTED, closure review pending: separate `Qwen35MultimodalDecoder` uses pinned Qwen3.5 wrapper mapping, variable-grid native vision, family-specific native/device text with interleaved MRoPE and embedding insertion, selected original experts, snapshots and distinct media cache framing; loader/factory/CLI/gateway paths are wired. The tiny fixture remains separate. | P02, P03, P10, P14 |
| P12-F02 | IMPLEMENTED, closure review pending: distributed generation now publishes complete rank cache/logits/RNG frames with logical-boundary consensus, immutable rank manifests and bounded resume without prompt replay. | P06, P12 |
| P14-F02 | IMPLEMENTED, closure review pending: K3 CLI/gateway argv accepts explicitly pinned v2/v3/v4 exact evidence and passes it to the concrete executor; exact mode without observed evidence remains rejected. | P08, P14 |

These findings are not an exhaustive inventory. Source inspection of P02–P14
must add every further required omission rather than assuming this list is all
that remains.

For a task to be DONE, its log must contain all of the following:

- Concrete implementation symbols and caller/consumer references.
- The required behavior implemented, or existing source behavior established.
- Applicable cross-area cases above accounted for by source reasoning.
- Every associated finding closed, with dependencies resolved.
- Public wiring and persistence updated wherever the change requires them.
- No placeholder, missing required branch, unexplained exclusion or fabricated
  execution evidence used to justify closure.

P15 must account for every top-level checkbox, P01–P14 task, matrix row and
finding. A parent checkbox cannot be complete while a required child is open.
No percentage is assigned from counting files, edits or checked boxes alone.

Append one entry whenever a task is completed or its remaining scope changes:

```text
Task ID:
Status: OPEN / IN PROGRESS / DONE
Requirement addressed:
Implementation files and symbols:
Callers/consumers inspected:
Implemented behavior or confirmed existing source path:
Failure, cancellation, ownership and recovery findings:
Remaining implementation/dependencies:
Source-only closure evidence:
Execution checks: not run, as instructed
```

The checklist provides the persistent implementation record. Work proceeds
from the first open task with satisfied dependencies. Completed source is retained,
and newly identified gaps are assigned to the appropriate task. Documentation
updates alone do not establish implementation completion.

Completing every applicable item above establishes 100% completion of the
defined Phase 1 source scope. It does not establish runtime correctness or
numerical/platform certification. No completion percentage is asserted yet.

### Source implementation record

Task ID: P01
Status: DONE
Requirement addressed: distinct input recovery and output publication budgets.
Implementation: `publisher_provider.resume_checkpoint` retains `max_bytes` for
reading and passes zero `max_checkpoint_bytes` when no output sink exists.
Consumers inspected: V4.1 `_checkpoint_options`, `resume_generation`, CLI
capture/resume branches, gateway file-source and streaming-source owners.
Invalid sink/budget combinations remain rejected before restore; file sources
remain scoped to their owning reader context. P01-F01 is closed.
Execution checks: not run, as instructed.

Task IDs: P03, P10, P14
Status: IN PROGRESS
Implementation: `qwen35_multimodal`, `qwen35_publisher_experts`, native additive
MRoPE ABI, native bounded owned snapshots, device MRoPE and embedding inputs,
publisher component/conditional mappings, factories, CLI and decoded gateway
media entry points. Qwen3.5 original full text/vision inventory is admitted
before mapped payload conversion. Original selected expert providers preserve
the Qwen3.5 fingerprint and family text kernels. Tiny fixture vision stays a
separate API. Native snapshot and cache budgets include retained snapshot bytes.
Required follow-up: finish source closure review of wrapper optional heads,
source namespace/dtype cases, native public composition and remaining matrix
cells. Original device dense projection streaming now has concrete bindings,
bounded original row transfers, tied-source admission and vocabulary/shared
expert projection dispatch through the Qwen3.5 decoder. Cache and snapshot
budgets include the borrowed projection owner; wrapper cleanup owns it explicitly.
Execution checks: not run, as instructed.

Task IDs: P07, P08, P14
Status: IN PROGRESS
Implementation: K3 v4 evidence admits multiple uniquely identified recorded
fixtures with individual observed continuation bounds; unknown fixtures remain
rejected. CLI/gateway argv accepts explicit evidence file, SHA256 and byte
admission. Decoded image paths cover V4.1/Qwen3.5/Qwen4/Inkling/GLM5;
selected-frame publisher videos carry explicit FPS and frame indices. Gateway
rejects caller-supplied placeholder collisions and animation before detachment.
Per-frame and aggregate admission remain distinct.
Selected fixture and evidence identity now persist in snapshots and exact-mode
version-three cache frames. The comparison receipt assembler combines explicitly
pinned existing reports and creates no observations. The compressed native
text codec binds evidence identity; media/BF16 use complete checkpoint transport.
Required follow-up: full cross-area ownership and required public wiring review.
Execution checks: not run, as instructed.

Task IDs: P06, P12
Status: IN PROGRESS
Implementation: `distributed_generation_checkpoint` publishes immutable rank
frames and commits manifests only after logical boundary consensus. Frames
retain physical cache, logits, sampler and stopping policy. Coordinator supports
restore-only, owned-boundary resume and file-manifest resume. Retained boundary
bytes have explicit admission, propagated through the launcher. Replacement
rank orchestration accepts cache or generation checkpoints with explicit pins,
fresh rendezvous epochs and application identities. Failed collectives poison
the owner and never reuse the old process group.
Required follow-up: source closure of rank failure ordering, root callback
failure, cancellation, publication failure and source/runtime identity cases.
Execution checks: not run, as instructed.

### Current source update — 2026-10-12

Task IDs: P02, P03, P04, P05, P06, P09, P11, P12, P14, P15

Status: IN PROGRESS

Completed source changes since the prior ledger entry: the source catalog now
expands concrete family/backend/arithmetic/format/publisher-pin cases and
validates backend-policy agreement; tied publisher heads require exact shared
embedding ownership; speculative target/draft owners, DSpark/MTP/vision
auxiliaries and V4.1 session shutdown have retryable close/closed-state paths;
distributed grouped projection returns the correct unbatched shape; accelerator
process clearing is serialized while preserving cancellation of an in-flight
request; packed device decoding admits bytes-like payloads and grouped-g64
format routing.

Remaining source closure: complete the cross-area P02–P14 caller review,
reconcile every matrix cell and public ABI path, finish direct codec/serving
lock and cancellation review, and resolve the remaining acceptance, media,
distributed, cache, streaming and evidence-gate findings before P15 can close.
No runtime execution, imports, tests, builds, compilation, benchmarks, device
probes, comparisons, commits or pushes were performed.

### Deep source completion audit — 2026-10-12

Audit basis: current source files and their statically reachable callers were
read from the working tree. No module was imported and no runtime, numerical,
platform, hardware, comparison, build or test check was used as evidence.
The audit does not treat a source reference, an existing rejection, a matrix
row, or a historical progress note as proof that the corresponding end-to-end
requirement is complete.

#### Verified closed source scope

| Task | Audit result | Primary closure evidence |
| --- | --- | --- |
| P01 | DONE | `publisher_provider.resume_checkpoint`, `publisher_v41_cli.PublisherSession.resume_generation`, and `execution_service` keep input recovery and optional output publication budgets separate and reject invalid sink/budget combinations before mutation. |
| P08 | DONE for the exact-admission source contract | `k3_execution.exact_admission.K3ExactEvidence`, `wholemodel_validation.k3_media`, K3 factories/CLI, and cache/snapshot consumers bind explicit v2/v3/v4 case identity, observed leaves, physical cache state, bounds, and pinned selection; unobserved cases remain rejected. |

#### Implemented but not yet closed end-to-end

| Task | Implemented source verified in audit | Remaining closure required |
| --- | --- | --- |
| P02 | CLOSED for the source-matrix contract: `source_scope.py` v3 expands family, wrapper identity, backend, arithmetic, source-dtype, persisted-format policy, positional, media, public-range, cache/TP, factory, gateway, recovery and tooling records; `validate_source_scope` checks structural agreement and truthful rejected routes. | No remaining P02 matrix-contract work. P03/P09/P10/P11/P13/P14 retain downstream implementation, caller and runtime/evidence closure against these rows. |
| P03 | Whole-inventory admission, tied-head ownership, PLE geometry, source dtype preservation, conditional loading, GLM5 prefix normalization, and family-specific mappings have source implementations. | Reconcile every original tensor namespace and optional bank with its consumer, especially auxiliary/MTP/vision/audio/video ownership, source-vs-execution dtype identity, tied heads, and positional variants. Confirm no admitted row relies on an unreviewed selected-loader omission. |
| P04 | V4.1 image planning, immutable media reads, native/device vision, auxiliary/MTP ownership, DSpark target evidence, and retryable lifecycle paths exist. | Close the complete composition review for multiple-image ordering, expanded token types/positions, native/device dispatch, borrowed projection owners, interrupted construction, reset, and auxiliary shutdown. |
| P05 | Speculative acceptance/rejection, target/draft snapshots, RNG state, cache frames, checksum/EOF checks, stop boundaries, and retryable cleanup are implemented. | Reconcile every accepted, rejected, bonus, EOS, context-limit, cancellation, repeated-restore, and gateway-stream path. Confirm target/draft physical evidence and both RNGs remain aligned through every serving consumer. |
| P06 | Distributed projection policies, rank agreement, replacement-rank orchestration, fresh rendezvous epochs, and poisoned-group handling have source paths. | Finish complete BF16/FP8/FP4 bank mapping for text, shared, expert, grouped-attention and auxiliary projections; review uneven/empty shard policy, collective failure ordering, root callback failure, cancellation, cache ownership and restore consensus. |
| P07 | K3 occupied-token admission, immutable media inputs, native/FLA policy separation, batch cleanup, slot-budget checks, and recurrence owners exist. The incomplete Metal route is now rejected at both publisher CLI and shared executor entrypoints. | Complete occupied/padded/signed input, explicit-position, media-expansion, independent-slot rollback, fork/restore, convolution-history, mixer continuation, and all factory/session caller reviews. |
| P09 | Format-specific packers/readers, FP8 scale admission, lattice/IQ3 contracts, bounded leases, device decoding, source identity checks, and atomic publication paths exist. | Map every P02 format to actual persisted loading and every native/device execution consumer; close E8/IQ3 geometry, packing order, scale/codebook/rotation, omitted-bank residency, conversion failure cleanup, and all family streamed-provider consumers. |
| P10 | Device arithmetic/projection/vision owners, no-fallback policy, bounded allocations, concrete V4.1 device checkpoint admission, and accelerator-process cleanup exist. | Complete each admitted family/backend token-to-logit chain and source capability mapping; reconcile all device expert/projection owners, cache restoration, cancellation, failed construction, runtime/version/dtype admission, and absence of implicit host neural fallback. |
| P11 | Public Segment/Edge declarations, version checks, counted-buffer checks, callback pinning, range/session lifecycle and retryable destruction paths exist. | Finish family-by-family ABI map, hidden/shared-index range boundaries, embedding/head schemas, continuation-state coverage, callback reentry/owner-thread rules, native symbol compatibility, and concrete service/factory/CLI wiring. |
| P12 | Scalable cache/page/lease/quarantine paths, durable receipt handling, direct codec checks, immutable rank manifests, consensus rollback, and replacement recovery exist. | Finish direct-codec versus serving lock/cancellation ordering, stale/foreign/failed rank restore consensus, cache page ownership, eviction/fairness, complete physical-state recovery for every family, and retained cleanup retry paths. |
| P13 | Family conformance, K3/Qwen3.5/Qwen4/V4.1 observation schemas, media/cache readers, artifact pins, and false certification flags exist. | Reconcile every family/backend/media/format matrix cell with a later-phase adapter and physical-cache leaf inventory; close missing source references, placement requirements, comparison bindings, and changed ABI/cache schema consumers. |
| P14 | Family CLIs, gateway checkpoint budgets, decoded media routing, warm artifact identity, cancellation and construction cleanup have source paths. | Complete end-to-end option/default reconciliation, streaming/backpressure, token-prefix semantics, repeated/stopped recovery, warm-session configuration changes, all media-capable gateway routes, and lifecycle/error propagation. |
| P15 | Prohibitions have been respected and the source ledger contains the major findings. | Revisit every top-level checkbox, P02 matrix row, cross-area case and finding; resolve all remaining integration gaps, then mark parent tasks and final closure only when no source requirement is unknown or open. |

#### Current completion determination

Phase 1 source implementation is **not yet 100% complete** according to the
checklist definition. The authoritative remaining work is the end-to-end
caller/consumer reconciliation above, not runtime verification. In particular,
the source matrix is not yet fully classified at the required operation,
public ABI, gateway, recovery and tooling granularity, and P15 cannot close
until those cells and findings are individually evidenced.

#### Required next closure order

1. Finish P02 matrix rows and classify every cell with concrete symbols.
2. Resolve P03/P04/P07/P09/P10/P11 operation and ownership gaps found by that
   row review.
3. Reconcile P05/P06/P12 physical-state, distributed, rollback and cancellation
   consumers.
4. Reconcile P13/P14 evidence/tooling and serving option/media paths.
5. Perform the P15 changed-source/caller audit, update every task ledger entry,
   and only then mark the applicable parent checkboxes complete.

### Full project source audit — 2026-10-13

Audit scope covered the top-level P02–P15 status table, the detailed task
checkboxes, the source matrix, all changed adaptive-engine callers, serving
queues, family CLIs, media processors, checkpoint/recovery codecs, cache and
distributed boundaries, ABI adapters, and lifecycle owners. This is a static
source audit only; it is not execution or certification evidence.

Verified at source level:

- P01 recovery/publication budget separation remains implemented.
- P02 has a concrete eleven-dimension source matrix contract with callable
  owner references and explicit rejected routes.
- P03 has resident inventory, optional/tied/shared/MTP/auxiliary/media and
  source-dtype ownership contracts.
- P08 has strict K3 evidence/admission schemas and explicit case selection.
- P14 now carries validated stop-prefixes through the bounded generation
  queue, accounts for their storage, gives multimodal sessions close and
  cancellation guards, and keeps one owned V4.1 auxiliary session for warm
  vision/MTP reuse.
- The project preserves explicit no-certification flags and the requested
  source-only execution boundary.

Remaining source closure, preventing a 100% claim:

- P04: complete V4.1 image ordering/position/type composition and verify all
  native/device/borrowed-owner/reset/interrupted-construction callers.
- P05: reconcile speculative acceptance, rejection, bonus/EOS/context-limit,
  cancellation, repeated restore and both RNG states through every gateway.
- P06: finish concrete BF16/FP8/FP4 text/shared/expert/MTP mapping, uneven or
  empty shards, collective/root failure ordering and restore consensus.
- P07: finish K3 occupied/padded/signed input, temporal media expansion,
  independent slot rollback and mixer continuation callers.
- P09/P10: map every admitted persisted format to every actual loader and
  execution consumer, including scale/packing/codebook/rotation and device
  no-host-fallback ownership.
- P11: finish family-by-family ABI symbols, range/embedding/head schemas,
  continuation leaves, callbacks, owner-thread/reentry and destruction order.
- P12: finish physical cache leaf inventories, page ownership/eviction,
  direct-codec versus serving cancellation, stale/foreign restore consensus
  and retained cleanup retry semantics.
- P13: bind every family/backend/media/format row to concrete evidence,
  placement, cache leaves, source references and schema consumers.
- P14: finish every family gateway's media ordering, streaming/backpressure,
  stop-prefix/stopped recovery, warm option identity and lifecycle/error
  propagation path. Queue-level support does not prove adapter-wide closure.
- P15: reconcile every unchecked detailed requirement and matrix row before
  changing any parent status to DONE or claiming Phase 1 completion.

The repository contains extensive pre-existing and ongoing uncommitted source
changes; they were preserved. Static whitespace validation only was used.
Tests, imports, builds, benchmarks, device probes, commits and pushes were
not performed.

### P02 closure update — 2026-10-12

P02 is now marked DONE for the source-matrix contract. The v3 matrix has one
row per admitted family identity/backend/source-dtype case, expands pinned
publisher variants, carries the persisted-format policy, and requires concrete
owners for configuration, tensor inventory, resident/streamed loading,
execution, cache/tensor parallelism, ABI, factory, CLI, gateway, recovery,
tooling and comparison. `validate_source_scope` rejects missing dimensions,
unadmitted cases, policy drift, and persisted routes that claim consumers they
do not have. This closes the matrix-shape contract only; it does not close the
full P02 callable-chain requirement. P02 remains IN PROGRESS until each
family/backend/format row is traced through its concrete loader, execution,
cache/TP, ABI, factory, CLI, gateway, recovery and tooling callers.

### P02 deep callable-reference audit correction — 2026-10-12

A static source scan of the declared owner strings found no missing Python
module prefix among the admitted family rows. The two `unadmitted.*` format
owners are intentional sentinel rejections, not missing modules. This verifies
catalog reachability only; it does not verify that each dotted symbol is the
correct consumer for every family/backend/format combination or that the full
chain reaches all serving and recovery callers. No imports or runtime
construction were used for this audit.

Source implementation added: `source_scope.validate_source_references` now
performs the missing static closure step when called by source tooling. It
resolves declared Python module prefixes and native slash-paths under the
adaptive-engine source root, checks declared dotted symbols by source text,
and preserves `unadmitted.*` as required rejection sentinels. It deliberately
does not import or execute owners and does not certify numerical, hardware or
runtime behavior. P02 still requires the resulting row-level callable-chain
review and truthful restriction decisions before final closure.

Additional P02 source implementation: `_P02_CHAIN_DIMENSIONS` is now the single
ordered contract for `configuration → inventory → loading → execution →
cache/TP → ABI → factory → CLI → gateway → recovery → tooling`; every expanded
execution case carries and validates that exact chain. This closes the former
duplicated/implicit dependency-order definition, but does not falsely convert
source references into runtime or numerical certification.

The same ordered chain is now exposed at matrix and per-family summary level,
and `validate_source_scope` rejects catalogs that omit or reorder it. This is
the code-level P02 dependency contract; row-specific consumer reconciliation
continues until every declared owner is either source-verified or explicitly
restricted/rejected.

P02 format closure was strengthened: each expanded row now carries an explicit
`format_route` decision containing the exact persisted-format names and their
native/device consumers from `source_format_routes()`. Validation rejects a
row whose format decisions diverge from the authoritative route table, so a
generic `packed-expert` label cannot silently stand in for a format-specific
consumer.

P06/P12 source closure was strengthened in `distributed_execution` and
`distributed_recovery`: distributed cache manifests now carry one bounded
model identity agreed by all rank receipts; restore admission validates that
identity against the target session before dispatch and each worker validates
both the receipt identity and manifest identity before mutating its cache.

`scalable_execution.DurablePagedCache` now exposes bounded owner-scoped prefix
eviction with deterministic candidate ordering, explicit protected checkpoints,
reference-triggered page reclamation and durable eviction journaling. It never
guesses which live continuation is safe to discard.

### P03 closure update — 2026-10-12

P03 is now marked DONE for the tensor-inventory contract. `component_contract`
preserves `source_text_dtype` separately from `execution_arithmetic`; source
identity is never silently replaced by runtime dtype metadata. `partition_weights`
owns every tensor exactly once across text, vision, audio and auxiliary/MTP
namespaces, rejects unadmitted media, and binds tied output heads to the exact
embedding owner. `publisher_conditional_loader` admits complete original text,
vision/audio, PLE, MTP and expert inventories before conversion, validates
fused/split expert geometry and source dtypes, preserves shared/tied aliases,
and rejects unowned scales or omitted component tensors. `publisher_glm5_loader`
and `inkling_mtp.admit_mtp_inventory` provide the corresponding complete GLM5
and Inkling ownership checks. Downstream P04/P09/P10/P13/P14 work remains
execution, ABI, media-composition and evidence closure rather than P03
inventory admission.

### P04/P07/P14 media and continuation update — 2026-10-12

Source implementation extended in `publisher_provider._residency_key`: warm
session reuse now fingerprints construction and continuation policy fields
(context, cache/projection/expert/work budgets, media token identities,
segment policy, arithmetic/device composition and speculative settings) in
addition to checkpoint/tokenizer/library artifacts. A session is therefore
reconstructed when a warm request changes a state-layout or ownership policy.

Existing source paths retain ordered multi-image markers, explicit expanded
media ownership, selected-frame video indices/FPS, K3 temporal grids, bounded
stream queues with cancellation/backpressure, stop-token persistence and
repeated recovery prefix checks. These paths still require the remaining
cross-family caller review before P04/P07/P14 can be marked fully closed.

### P14 source audit correction — 2026-10-13

The P14 source changes were re-audited against their direct callers. The
following items are verified source implementation, not runtime certification:

| Area | Verified directly in source | Still partial or remaining |
| --- | --- | --- |
| Queue admission and stop prefixes | `execution_queue.NativeExecutionQueue.submit` admits bounded unique stop-prefixes, includes their token cost in pending-storage admission, persists them in `_Request`, and forwards them to `NativeExecutionService.generate`. | Media-bearing queued requests and prefix options still need reconciliation through every higher-level queue/CLI/distributed adapter. |
| Multimodal session lifecycle | `multimodal_session.MultimodalSessionExecutor` has explicit closed-state checks, idempotent close, cancellation validation, bounded numeric media-array copying, and family-specific media field allowlists. | Every factory-created conditional decoder still needs a caller-by-caller close/error/retry audit, including media preprocessing cancellation and decoder cleanup after partial construction. |
| V4.1 warm ownership | `publisher_v41_cli.PublisherSession` creates one auxiliary owner, registers it for cleanup, reuses it for vision/MTP, and closes replaced children when budgets change. | Full warm-session option/default identity coverage across V4.1, K3, Qwen3.5, Qwen4, GLM5 and Inkling remains open; changed media processors, tokenizer/configuration, checkpoint artifacts and distributed policies must be checked at each gateway. |
| Recovery and streaming | Native service checkpoint budgets, transactional rollback, synchronous event callbacks, scheduler admission and repeated recovery paths exist in source. | P14 is not closed: no complete source map yet proves streaming/backpressure, stop-prefix persistence, stopped-boundary replay, repeated restore, media placeholder ordering and error propagation for every P02 family route. |

P14 therefore remains **IN PROGRESS**. No tests, imports, builds,
benchmarks, device probes, commits or pushes were performed for this audit.

### P05/P06/P12 speculative, distributed and cache update — 2026-10-12

Source implementation strengthened in `distributed_generation_checkpoint`:
failed rank-frame capture and failed manifest publication now coordinate a
root-owned removal of only the newly reserved, uncommitted boundary directory;
cleanup failures are gathered and retained in the collective error. Rank
restore now validates the command identity, recovery budget, world geometry,
committed directory containment, rank receipt and generation boundary before
opening a frame, rejecting stale or foreign restore commands before service
mutation.

`v41_execution.speculative` snapshots target, proposer and sampler RNG state,
restores all independent owners on rejection/cancellation, quarantines partial
rollback, and retains retryable snapshot cleanup. `cache_file_recovery`, K3
checkpoint transport and V4.1 generation recovery retain complete-frame,
identity, checksum/EOF and ownership checks. Remaining P05/P06/P12 work is
cross-consumer closure of these source contracts through every serving and
replacement-rank caller; no runtime or collective execution evidence was used.

### P05/P06/P12 strict source audit correction — 2026-10-12

The preceding update was rechecked against the concrete callers and is not a
100% closure claim. The following distinctions are authoritative for this
audit:

| Area | Verified directly in source | Still partial or remaining |
| --- | --- | --- |
| P05 | `v41_execution.speculative` snapshots target, proposer and decoder RNG; rejection restores target state; exception paths attempt rollback of both owners; transport frames bind target/draft identity, policy, history, logits, stop state, both RNG states, sizes, checksum and EOF; busy/lock/cancellation gates exist. | No source audit has yet proven that every serving gateway, cancellation owner, EOS/bonus/context-limit branch and repeated restore caller preserves the same acceptance boundary and both RNG states. The transport codec does not itself prove public stream/recovery integration. P05 remains IN PROGRESS. |
| P06 | Distributed generation capture gathers rank-frame errors and logical signatures, publishes only a complete manifest, validates world geometry and rank receipts, and now quarantines only the newly reserved boundary on capture/publication failure. Replacement recovery admits pinned immutable manifests and opens fresh sessions before invoking the driver. | The distributed cache manifest is bound to executor identity/history but has no explicit application-identity field; generation restore carries application identity separately. Root callback failure, cancellation ordering, poisoned collective behavior, and every BF16/FP8/FP4 text/shared/expert/grouped/auxiliary mapping still require caller-by-caller source closure. P06 remains IN PROGRESS. |
| P12 | Native service cache restore is transactional and identity/geometry/history/checksum bounded; failed restore attempts model rollback and quarantine on rollback failure. Generation checkpoint restoration now preserves the primary restore/cancellation error while attempting both staged-cache close and rollback-snapshot release. Scalable cache code has page/lease/owner/quarantine paths. Replacement recovery re-admits the manifest before and after restore and closes opened sessions on failure. | The full family-by-family physical cache state, page ownership/eviction fairness, stale/foreign rank restore consensus, direct-codec versus serving cancellation/re-entry ordering, and retained cleanup retry semantics are not yet proven across all consumers. P12 remains IN PROGRESS. |

Additional correction: `restore_rank` now validates the rank-list shape and the
current receipt before resolving or opening the frame, including lower-case
digest shape, rank, filename, byte bound and identity. This closes a concrete
pre-index validation gap, but it does not by itself close the end-to-end P06 or
P12 requirements above. No tests, imports, builds, benchmarks, device probes,
commits or pushes were performed.

### Full Colibri-surpass source audit — 2026-10-12

Audit method: read the Colibri reference notes under `reference/colibri/docs`,
the Phase 1 requirements, the v3 source matrix, and the statically reachable
OpenAgent owners/callers. A named symbol, capability flag, source row,
documentation statement, or rejection path is not counted as implementation
unless its consumer and failure/lifecycle path are also present. No imports,
tests, builds, benchmarks, numerical comparisons, hardware probes, or runtime
requests were used.

#### Verified source capabilities

| Capability | Verified source evidence | Status |
| --- | --- | --- |
| Family/source scope | `source_scope.py` enumerates the admitted family/backend/source-dtype/format axes, closure owners, and explicit rejected formats. | Source contract verified; consumer closure still required. |
| Tensor ownership | Conditional loaders and inventory planners cover complete text, optional media, MTP/auxiliary, expert and tied/shared-head ownership with source dtype kept separate from execution arithmetic. | P03 source contract verified. |
| Durable native/service cache | Service restore is transactional, bounded, identity/geometry/history/checksum checked, and attempts model rollback; file transport verifies complete frames and receipts. | Core source path verified; all family codecs/callers not closed. |
| Speculation | V4.1 speculative code retains target/draft snapshots, decoder/proposer RNG, policy/history/logits/stop state, checksum/EOF transport, busy gates and rollback quarantine. | Core source path verified; serving closure not verified. |
| Distributed publication/replacement | Rank capture has logical-boundary consensus and immutable receipts; replacement recovery pins manifests, re-admits before/after restore, uses fresh sessions and closes them on failure. | Core source path verified; full collective/family mapping not closed. |
| Media routing | Provider source validates image/video placeholders, family restrictions, frame metadata and K3 temporal ownership; warm-session residency fingerprints media and execution policy. | Routing source verified; all family execution/streaming closure not verified. |
| Native public ownership | Public adapters, ranges, callback/resource release and several family library identities exist. | Partial; ABI and session/reentry matrix remains open. |

#### Remaining implementation required for a truthful Colibri-surpass claim

| Area | Concrete remaining source work | Checklist owner |
| --- | --- | --- |
| Matrix truth | Expand every admitted P02 row from named owner to a verified callable chain: config → exact tensor inventory → loader/format reader → execution → cache/TP ownership → ABI → factory → CLI → gateway → recovery → tooling. Remove or explicitly restrict rows whose consumer is only a placeholder, generic backend, or source note. | P02/P15 |
| Formats and quantization | Reconcile each matrix format with a real decoder and family consumer, including E8/IQ3 rotation and codebooks, FP8 scale encoding, MXFP4 backend limits, packed expert geometry/order, conversion cleanup, and explicit offline-only `int4-rans256-g0` / refused UE8M0 handling. | P09/P10 |
| Full family execution | Prove source-complete token-to-logit chains for Mixtral, OLMoE, Qwen3.5, Qwen4, GLM5, Inkling, GLM-DSA, Kimi K3 and DeepSeek V4.1 for every admitted backend. Operation kernels or factories alone are insufficient. | P03/P09/P10/P13 |
| Media and continuation | Finish multi-image ordering and position/type expansion, K3 video temporal packing, publisher frame-index video, MTP/auxiliary continuation, stop-prefix semantics, stream backpressure, cancellation, repeated recovery and warm-session option changes for every media-capable gateway route. | P04/P07/P14 |
| Speculation | Reconcile accepted/rejected/bonus/EOS/context-limit branches, rollback boundaries, decoder and proposer RNG alignment, target/draft physical cache state, cancellation, repeated restore, stream publication and non-speculative fallback consumers. | P05 |
| Distributed execution | Implement or explicitly restrict concrete expert/shard placement and collective execution for each distributed family; finish BF16/FP8/FP4 shared/expert/MTP bank mapping, uneven/empty shards, root callback failure, collective cancellation, poisoned-group teardown, cache ownership and restore consensus. Rank orchestration alone is not parity. | P06/P12 |
| K3 limitations | Preserve truthful restrictions: source notes state K3 is single-token, has no speculative/MTP path, and CPU plus optional Vulkan rather than CUDA/Metal. Any broader OpenAgent matrix row must either implement the missing route or reject it. | P07/P10/P15 |
| Native ABI | Complete family-by-family counted-buffer, range, hidden/shared-index, embedding/head schema, callback pinning, owner-thread/reentry, registration-before-lookup, concurrent-session and close-order review; distinguish kernel existence from loadable runtime dispatch. | P11 |
| Cache/scheduler | Close direct codec versus serving lock/cancellation ordering, page ownership, eviction/fairness, stale/foreign rank restore, physical-state recovery for every family, publication pointer cleanup and retryable cleanup after failed close/recovery. | P12 |
| Gateway/API | Reconcile READY gating, request ordering, image placeholder accounting, unsupported-tool rejection, auth, static/path safety, streaming terminal/error events, token-prefix and stopped/repeated recovery behavior, and all CLI/provider defaults. | P14 |
| Colibri-specific surfaces | Add or explicitly scope Brio closed-set scoring, grammar/tool-call protocol, routing telemetry/expert atlas, multi-disk partial mirror behavior, cache-route semantics, and the documented family-specific APIs. These are not implied by ordinary text generation. | P13/P14/P15 |
| Evidence/tooling | Bind every family/backend/media/format row to source-only observation schemas, physical-cache leaves, artifact pins, comparison inputs, changed ABI/cache schemas and truthful unsupported/restricted outcomes. | P13/P15 |

#### Reference limitations that must not be overstated as completed

The Colibri notes explicitly describe several capabilities as restricted or
unfinished: Qwen3.6 Phase 1 is attention-only/identity-substituted for much of
the hybrid stack and omits later ViT/MTP work; DeepSeek V4 multi-GPU expert
parallel is drafted rather than implemented in the reference; K3 documents no
MTP/speculation and no CUDA/Metal route; Metal Kimi/DSA notes contain stubs or
planned kernels; Windows HIP notes do not establish runtime DLL dispatch; and
format notes distinguish offline repack, rotated E8/IQ3, Vulkan-only MXFP4 and
refused UE8M0 routes. OpenAgent must implement the stronger capability or mark
the corresponding row restricted. Reference documentation is not evidence that
OpenAgent has implemented any of these paths.

#### Audit conclusion

The repository is **not 100% Colibri-surpass complete** from source evidence.
P02/P03/P08 have bounded source contracts, and several core cache,
speculation, distributed-publication and media-routing primitives are present.
The remaining work is the concrete cross-product and caller closure listed
above, especially P09/P10/P11/P13/P14 plus the unresolved P05/P06/P12 paths.
P15 must remain OPEN until every admitted row is either source-complete or
explicitly rejected/restricted with a concrete reason and every remaining
finding is resolved. This audit updates the remaining-work definition; it does
not claim runtime correctness or performance parity.

### Colibri reference-notes reconciliation — 2026-10-12

The repository also contains `reference/colibri`, including the format
registry, backend notes, family phase plans and gap analyses. These documents
were read as comparative requirements for “Colibri-surpass”; they are not
treated as proof of OpenAgent implementation. The following concrete notes add
requirements or expose limitations that were not explicit enough in the prior
ledger.

#### Newly recorded missing or restricted capabilities

| Colibri source note | Requirement or limitation found | OpenAgent Phase 1 consequence |
| --- | --- | --- |
| `reference/colibri/docs/FORMATS.md` | `int4-rans256-g0` is an offline-only format in the reference engine: no decode branch, no public ordinal, and stamped dispatch is required before inference. | P09 must explicitly classify this format as **MISSING/NOT ADMITTED** unless OpenAgent adds a complete stamp-gated decoder, byte-budget admission, loader, and execution consumer. It must not be counted as supported merely because general quantized loading exists. |
| `reference/colibri/docs/FORMATS.md` | E8/IQ3 stores rotated weights and requires activation rotation; MXFP4 is Vulkan-only and is not a CPU/QT/Metal format; FP8 UE8M0 is recognized but refused. | P09/P10 must record per-backend format capability, activation rotation, scale encoding, and refusal behavior. A format row is incomplete until every admitted backend has a real consumer or an explicit rejection. |
| `reference/colibri/docs/deepseek-v4.md` | Multi-GPU expert-parallel execution is explicitly drafted, not implemented in the reference engine. | OpenAgent’s distributed P06/P12 work must not claim parity from rank orchestration alone; it needs a concrete expert-bank/shard mapping and collective execution consumer, or an explicit single-device restriction. |
| `reference/colibri/docs/qwen36-phase01.md` | Qwen3.6 Phase 1 runs only 25% gated-attention layers; 75% DeltaNet layers are identity; ViT and MTP are not implemented until later phases. | Any Qwen3.6-like row must be marked **partial/restricted**, not full model support. P02/P03/P04/P10/P13 must distinguish attention-only, identity-substituted, vision, MTP and full-model rows. |
| `reference/colibri/docs/metal_implementation.md` | `kimima_forward_metal` is a deliberate monolithic stub; DSA indexer Metal kernels are planned/not implemented; multi-token regression and later optimization remain open in the notes. | P10/P11 must distinguish granular implemented Metal operations from a complete family forward path. A backend capability flag cannot advertise full Kimi/DSA Metal execution while the monolithic/indexer route is absent. |
| `reference/colibri/docs/windows.md` | HIP DLL runtime loading is explicitly not implemented; a built DLL does not engage the GPU and execution remains CPU. | P10/P11/P14 must verify that OpenAgent’s HIP capability means actual runtime loading and dispatch, not only build/configuration declarations. Otherwise the route must be marked unavailable/restricted. |
| `reference/colibri/docs/segment-runtime.md` | Registration must be explicit before lookup; sessions cannot receive concurrent calls; close must fail while sessions are alive; range identity, cancellation and fixture non-advertisement are contractual. | P11/P12/P14 need explicit caller review for registration ordering, session serialization, close/reentry, half-open ranges, cancellation publication and prevention of tiny-fixture support claims. |
| `reference/colibri/docs/serve_protocol.md` and `api.md` | READY gating, request-id ordering, image-placeholder accounting, unsupported-tool rejection, path-safe static fallback, auth and streaming protocol semantics are required public behavior. | P14 must classify each gateway/server route for readiness, ordering, media placeholders, auth, streaming errors and unsupported capability rejection; model execution alone is insufficient. |

#### Newly identified checklist actions

1. Add explicit format rows for `int4-rans256-g0`, E8/IQ3 rotation, MXFP4
   backend scope and FP8 scale encodings to the P02/P09/P10 matrix.
2. Add a Qwen3.6-style partial-model classification rule: attention-only or
   identity-substituted layers cannot be marked as full-model completion.
3. Add a backend distinction between “operation kernels exist”, “family
   forward path exists” and “runtime library is actually loadable and
   dispatching”. This is required for Metal, HIP and native Segment/Edge.
4. Add explicit P11/P14 rows for READY/auth/request ordering/media placeholder
   and concurrent-session contracts from the Colibri protocol notes.
5. Add an explicit single-device-versus-expert-parallel verdict for every
   distributed family; rank checkpoint/replacement support alone is not enough.

These additions are now part of the remaining 100% definition. They do not
claim that the referenced Colibri limitations are OpenAgent bugs; they identify
where OpenAgent must either implement the stronger capability or document and
enforce a truthful restriction before P15 can close.

### P02 implementation update — 2026-10-12

Status: IN PROGRESS

Implemented source change: `source_scope.source_format_routes` now records
Colibri's `int4-rans256-g0` as an explicit `rejected-unadmitted` route with no
native or device consumer and a mandatory reason. `validate_source_scope`
rejects incomplete classifications and rejects any route that claims consumers
for this offline-only format. This prevents the matrix from inheriting an
ordinary int4 decoder merely because the repack payload contains reconstructed
int4 bytes. The same explicit rejection now covers the recognized-but-unadmitted
FP8 E4M3 block `UE8M0` scale encoding, which has no admitted decoder.

Remaining P02 work: continue classifying every family/backend/source-format row
with concrete IMPLEMENTED, CONTRACT RESTRICTION or REJECTED-UNADMITTED status;
then inspect each referenced loader, execution owner, ABI, factory, CLI,
gateway, recovery and tooling consumer before P02 can close.

Additional P02 implementation: `source_scope.py` now has an explicit
`_FAMILY_CLOSURE` record for all nine admitted families. Each expanded matrix
case carries concrete `cache_tp`, `factory`, `gateway`, `recovery` and
`tooling` owners in addition to configuration, inventory, loading, execution,
streaming, ABI, cache, CLI and comparison owners. `validate_source_scope` requires these
dimensions in every family row, so a decoder-only catalog cannot pass the P02
shape contract. The catalog schema is now `openagent.phase1-source-scope.v3` to
make the expanded ownership contract explicit to downstream consumers. The
matrix also carries an explicit persisted-format policy per family/backend row
instead of treating the generic `packed-expert` label as support for every
packer; `validate_source_scope` requires that policy on every expanded case.

Execution checks: not run, as instructed. Commits and pushes: not performed.

### Colibri whole-project functionality comparison — 2026-10-13

This is the latest comparative source audit, based on static reading of the
Colibri docs and OpenAgent source/search results. It does not claim that every
Colibri source file was executed or that behavior is equivalent. “Not found”
means no implementation/call chain was located in the searched OpenAgent
product source; it is a source gap to resolve, not a runtime verdict. Several
Colibri docs describe capabilities that are themselves experimental, optional,
family-specific, or explicitly unfinished. Parity must be decided per
capability and family, not inferred from the Colibri name or from generic
OpenAgent features.

| Colibri functionality/surface | OpenAgent source finding | Classification and remaining decision |
| --- | --- | --- |
| Closed-set Brio and Jev scoring (`docs/brio.md`, `c/openai_server.py`) | `NativeExecutionService` has transactional prefill and grouped continuation scoring; service-backed native/conditional owners expose text and tokenized option scoring plus a prefix transaction that restores the original session. Provider methods implement one-question, batched-question, and dependent schema-fill forms; schema fills prefill the common token prefix once while retaining earlier selected values. `ModelGateway` routes only to the configured native provider; `/v1/brio` validates all forms; `/v1/systemone` adapts noul/choice/score with aggregate-only audit metadata and Jev's 255-label choice cap. Direct K3/V4.1 sessions remain explicitly rejected pending lifecycle composition. | **PARTIALLY IMPLEMENTED.** Continue with direct K3/V4.1 adapters, response/error compatibility and additional client entrypoints; keep P13/P14 open. |
| Grammar-constrained generation and tool-call protocol (`docs/grammar-draft.md`, `docs/api.md`) | Catalog marks grammar/tool-call protocol restricted/not admitted. OpenAgent has general tool execution APIs and family-specific decoding, but no matching grammar/GBNF constrained-token decoding path was located. | **MISSING / NOT ADMITTED for Colibri parity.** Generic tools are not equivalent to constrained generation or Colibri’s per-family tool syntax and rejection rules. Decide/implement the required family/API subset. |
| Route trace, `.coli_usage`, expert atlas (`docs/routing-telemetry.md`, API docs) | OpenAgent has local system, scheduler, and expert-tier telemetry, but catalog says Colibri route telemetry/expert atlas is source-observation-only. No matching shared route-trace persistence/format and atlas consumer was located. | **PARTIAL, NOT PARITY-CLOSED.** Map existing local metrics to required route-level data, lifecycle, format and atlas workflow; otherwise record an explicit scope restriction. |
| Cache-aware expert route substitution (`docs/CACHE_ROUTE.md`) | Catalog labels cache-route semantics source-observation-only. Existing expert-tier residency/promotion and routing components do not establish Colibri’s top-K-preserving route substitution and its accuracy/counter semantics. | **PARTIAL / UNVERIFIED.** Establish whether an equivalent selector exists and trace it through each relevant MoE family, or explicitly restrict this capability. |
| Multi-disk mirrors and partial-mirror planning (`docs/multidisk.md`) | Catalog marks multi-disk partial-mirror restricted/not admitted. Generic disk streaming, cache directories, or expert storage are not evidence of mirror identity, fallback, weighting, or partial shard planning. | **MISSING / NOT ADMITTED.** Implement if in parity scope; otherwise keep the limitation explicit and reject any matrix claim that implies it. |
| Colibri family-specific API/gateway protocol (`docs/api.md`, `docs/serve_protocol.md`) | OpenAgent has its own family entrypoints and publisher gateway. The catalog records explicit-family-entrypoints-only; that does not establish wire/API compatibility. Existing protocol audit already identifies READY gating, request ordering, media placeholders, auth, unsupported-tool rejection, streaming terminal/error events, and recovery as caller-level obligations. | **PARTIAL / NOT EQUIVALENT BY DEFAULT.** Inventory each OpenAgent endpoint/CLI against only the selected Colibri contract; implement or explicitly exclude each protocol behavior. |
| Family/model/backend capabilities | The OpenAgent nine-family source catalog and many decoder/loader paths exist, but matrix rows still require per-format and per-backend consumer closure. The reference itself limits some paths (e.g. Qwen3.6 staged hybrid, K3 no MTP/speculation and constrained backends, DeepSeek V4 expert parallel drafted, Metal/HIP limitations). | **PARTIAL.** Do not count reference limitations as OpenAgent implementations. For each family/variant/backend, either prove the callable path or enforce a restriction; see P02–P11 and the reference reconciliation above. |
| Ordinary generation, media, state/recovery, distributed and operational lifecycle | OpenAgent has substantial implementations in these areas, recorded under P03–P12 and P14; this review did not establish complete per-family/backend/public-caller equivalence. | **PARTIAL / CLOSURE OPEN.** Continue the row-by-row producer-to-consumer and failure/lifecycle review already specified in P02–P14. |

#### Required continuation work

1. Decide which Colibri application surfaces are in the intended “same
   functionalities” scope. Do not silently treat a source-scope label as an
   implementation or as an accepted product exclusion.
2. For each in-scope surface above, add a concrete owner and callable path to
   the P02/P13/P14 matrices, including input contract, output/wire contract,
   unsupported cases, lifecycle/error behavior and evidence owner. Mark it
   IMPLEMENTED only when that path exists; otherwise mark MISSING or an
   enforced CONTRACT RESTRICTION with a concrete reason.
   For Brio, continue from `NativeExecutionService.score_continuation_groups`,
   the service-backed `PublisherSession.score_option_groups` mixin,
   `NativePublisherProvider.score_brio`, `ModelGateway.score_brio` and
   `POST /v1/brio`; multi-question forms share their token prefix, and schema
   fields prefill their shared token prefix once then restore the original service
   boundary; direct K3/V4.1 scoring remains open.
3. Reconcile per-family tool support and documented API behavior independently;
   generic tool execution, telemetry, routing, disk storage, or generation is
   not equivalent evidence.
4. Continue the existing backend/format/media/state closure and final P15 review.

#### Current conclusion

The source evidence does **not** support saying the whole OpenAgent project
implements Colibri’s complete functionality set. Some broad engine and serving
primitives exist, while multiple Colibri-specific application surfaces are
explicitly restricted or only observational, and the core family/backend
callable-chain review remains open. The checklist records both classes of
remaining work so later implementation can proceed without conflating generic
similarity, explicit exclusions, or unfinished Colibri reference features.

This audit used static source inspection/search only. No tests, imports, builds,
compilation, benchmarks, device probes, comparison runs, commits or pushes were
performed.

### Deep source recheck — 2026-10-13

This is the authoritative current status. It supersedes any earlier status
table or completion statement that conflicts with it. The recheck read source
text only; it did not import modules or run tests, builds, benchmarks, device
probes, commits or pushes.

| Task | Status | Verified source work | Remaining source closure |
| --- | --- | --- | --- |
| P02 | IN PROGRESS | Nine-family catalog, eleven-chain schema and rejected-format entries exist. Public K3/V4.1 rows are truthfully text-only. The source matrix's restricted K3 Metal row is enforced before CLI checkpoint loading and at the shared K3 executor boundary, so direct factory/validation callers cannot activate the incomplete route. | Trace every remaining admitted backend/format row to a concrete consumer; a catalog reference is not closure. |
| P03 | DONE (inventory contract) | Inventory/mapping, optional/tied ownership and dtype identity admission are represented before conversion. | Downstream consumers remain owned by P04/P09/P10/P13/P14. |
| P04 | IN PROGRESS | Qwen4 and Qwen3.5 media paths are traced through staging/vision expansion and final text logits; single-prefill restrictions and cancellation-aware owned staging are explicit. | Prove remaining-family media ordering/position/geometry and failure-lifetime paths through final logits, including public callers and borrowed-owner cleanup. |
| P05 | IN PROGRESS | Target/draft/RNG/stop state and proposer snapshots are transactional; discarded rejected/unused draft suffixes now rewind the proposer RNG even when that RNG is external to the model snapshot. Stop-policy acquisition rollback and DSpark CLI/resume caller traces are recorded. V4.1 request reset now attempts target and draft reset independently, preserving the first failure while attempting both owners before session teardown; callback/checkpoint errors still retain precedence over cleanup errors. | Reconcile accept/reject/bonus/EOS/context/cancel/repeated restore across all serving callers, and verify checkpoint publication and rollback behavior beyond the V4.1 CLI/session path. |
| P06 | IN PROGRESS | Rank frames/replacement exist; V4.1 preflights all selector/provider/device policy before format-specific registration, synchronizes bank setup failures, and quarantines banks after transport errors. Failed/disagreeing distributed snapshot release now closes the affected session on each surviving rank and clears coordinator-side handles before group poisoning. | Map all distributed banks and operation-level collective failure, cancellation, process-loss and recovery-consensus behavior to concrete consumers. |
| P07 | IN PROGRESS | K3 validates input/media positions and slot rollback; FLA cache ownership, cancellation polls and staged-fork restore cleanup are transactional. | Finish temporal/video semantics and serving-consumer integration; preserve text-only batch and unsupported-backend restrictions. |
| P08 | DONE (source schema) | Exact v2/v3/v4 evidence selection and persistence are implemented. | P07/P13 retain execution and observation coverage. |
| P09 | IN PROGRESS | Format route catalog, bounded decode paths and rejected formats exist. The shared selected-device expert provider validates the E8 contract and aligned geometry, and applies bounded/cancellable activation rotation to input and intermediate rows for Mixtral, Qwen3.5 and Qwen4. IQ3 requires its exact persisted grid contract, aligned 256-weight hidden/intermediate dimensions at provider construction, and bounded device activation rotation. Device weight staging checks cancellation between each host-to-device matrix copy and clears partial staged tensors on exception. V4.1 checkpoint contracts separately admit only its exact FP8/FP4 publisher encodings, not generic E8/IQ3 expert packs. Shared Mixtral/Qwen3.5/Qwen4 selected-device execution now rejects non-finite FP32 activation rows before acquiring persisted payloads or staging decoded weights. | Verify every family/backend loader/consumer compatibility, scale geometry, packing order and conversion cleanup; keep unsupported format/backend tuples rejected until their exact consumer is implemented. |
| P10 | IN PROGRESS | Device admission and several text/cache owners exist; Mixtral, Qwen3.5 and Qwen4 share a selected-device expert constructor with bounded/cancellable E8 activation transforms. Selected-device expert tensor staging observes cancellation between copies and clears partial staged state on failure. Shared expert execution rejects non-finite FP32 activation rows before persisted-payload acquisition and device weight staging. Adaptive backend registry no longer advertises embedding/vision/audio execution for text-only frontends or any phases for unknown backends; the device autotuner rejects embedding-phase tuning before invoking the benchmark backend. | Complete source operation chains and no-fallback placement closure for every admitted backend. Registry capabilities remain candidates, not proof of compilation or actual placement. |
| P11 | IN PROGRESS | Public K3/V4.1 text ABI owners exist; derived identities are immutable, snapshot handles require exact object ownership, and cancellable stateful ABI forwards roll back or quarantine. The shared Segment/Edge callback guard clears the opaque handle and complete capability output before open admission, and re-clears capabilities if a family callback fails after populating them but before handle publication. | Review every family range/callback/reentry/close contract and distinguish internal kernels from public ABI support. |
| P12 | IN PROGRESS | Transactional restore/quarantine/rank recovery exist; direct ABI cancellation rollback, locked/cancellable K3 generation publication/restore, and session-locked OLMoE Segment forward/reset/snapshot/restore are implemented. | Close V4.1 restore/other-family physical-state, codec-vs-serving cancellation, stale/foreign restore and distributed consensus coverage. |
| P13 | IN PROGRESS | Source tooling, false certification gates, Brio question/schema HTTP forms and Jev systemone adapters exist; multi-question and dependent schema scoring share prefetched token prefixes, restore the original service boundary, propagate Jev label limits, and hold a single provider lifecycle lock with aggregate input and token-read bounds. Every matrix row carries a validated exact-tuple obligation with named observation/cache schema owners, artifact-pin classes, placement owner, comparison adapter and report schema; missing media adapters are explicit. | Instantiate and reconcile these owners against each tuple, then produce pinned observations, complete physical-cache comparisons and actual placement receipts. Generic adapter references do not establish backend/format coverage; media adapters are not produced evidence. Finish direct K3/V4.1 scoring only if safe, and other Colibri gaps. |
| P14 | IN PROGRESS | Media and queue/stop-prefix primitives exist. `/v1/messages` accepts Anthropic `x-api-key` through the same local token/scope checks, including opt-in scope enforcement; the header remains route-scoped. The local `ChatService.stream` closes its provider iterator without masking an active primary failure. The compatibility route rejects malformed UTF-8 request text/stops as client errors rather than leaking `UnicodeEncodeError`; OpenAI `user` attribution remains explicitly unsupported. | Close all family gateway/CLI stream, recovery, warm-session and error-cleanup callers; reconcile auth when enforcement is disabled and full OpenAI/Anthropic request/streaming semantics. |
| P15 | OPEN | Prohibitions were respected and the unsupported completion claim was removed. | Cannot close until P02 and P04–P14 are source-closed. |

The checked child items under P04–P15 record work performed, not task closure.
Their task status is controlled by this table until their remaining rows have
concrete source evidence.

### P13/P14 serving auth update — 2026-10-13

`routes/runtime.py` now recognizes `x-api-key` only on `POST /v1/messages` when
the global auth gate is enabled. It resolves the value through the existing
local bearer-token store and applies the same `local_ai:admin` write scope as
other compatibility POSTs; a supplied Authorization bearer token retains
precedence, and an Anthropic header cannot authorize unrelated routes. This
closes the authenticated Anthropic-header mapping path, not full Anthropic
authentication parity: enforcement remains operator-opt-in, so an unauthenticated
local configuration still does not require a key. Streaming events, cancellation,
request semantics and full response/error compatibility remain open. No tests,
imports, builds, benchmarks or device probes were run.

### P14 stream disconnect cleanup update — 2026-10-14

`routes/runtime.py:send_event_stream` now includes response-header setup inside
the stream cleanup `try/finally`. A client disconnect or header-write failure
therefore closes the provider iterator as well as a disconnect during event
delivery; the native publisher generator's finalizer sets its cancellation
event and waits briefly for its worker. Error events are only written after SSE
headers were successfully sent, avoiding an attempted SSE body on a failed
response setup. This closes the HTTP iterator-ownership gap, not provider
decoder interruption guarantees or per-family caller coverage. No tests,
imports, builds, benchmarks or device probes were run.

### P14 gateway provider-iterator ownership update — 2026-10-14

`model_gateway.py:stream_complete` now wraps both primary and fallback provider
iterators with deterministic close-on-unwind handling. This includes consumer
disconnect while a chunk is yielded, so upstream providers can release response
and socket resources instead of relying on garbage collection. If iterator
cleanup fails while another error or cancellation is already unwinding, the
cleanup detail is attached as a note without replacing that primary failure.
This is a gateway ownership fix, not full P14 backpressure or caller lifecycle
closure. No tests, imports, builds, benchmarks or device probes were run.

### Source-closure correction — 2026-10-13

The earlier claim that P02–P15 were source-complete is withdrawn. It confused
catalogued owners and structural validation with a completed caller-by-caller
implementation review.

Concrete correction: `publisher_public_cli.PublisherSession` rejects
`include_vision`, `image`, `dspark` and `media_items`, so the K3/V4.1
`public-range` matrix rows are text-only. `source_scope.py` now marks their
media cells as `contract-restriction`; they must not inherit the ordinary
publisher media capability.

Implementation follow-up: the public CLI now also rejects the separate K3
`image_token_id`, `image_mib` and `vision_tower_epsilon` switches. This closes
the public-boundary option leak: no media/auxiliary option can be silently
accepted by a text-only Segment/Edge construction path.

The matrix validator now enforces that restriction for every generated
K3/V4.1 public-range row. A copied or regenerated row that advertises the
ordinary family media contract through the public text ABI fails source-scope
validation before it can be exposed to a caller.

It also classifies Segment/Edge ownership per backend: only an explicit
`public-range` row may advertise the public wire ABI. Native, device, FLA,
Metal and distributed rows retain `private-family-abi`, preventing a family
public adapter from being mistaken for a consumer of every backend.

Task ID: P02–P15
Status: IN PROGRESS; P15 OPEN
Verified source work: the catalog, explicit rejected format routes, public
K3/V4.1 text ABI route, transactional cache/recovery primitives, speculative
state capture and media routing primitives all exist.
Remaining source work: reconcile each admitted row with its real loader,
format consumer, operation chain, cache/TP owner, ABI, factory, CLI, gateway,
recovery and tooling consumer. The task-level remaining work is recorded in
the current-progress table and the detailed audit below.
Source-only evidence: source-text inspection only; no execution/certification
claim is made.
Execution checks: not run, as instructed.

Execution checks: not run, as instructed. Commits and pushes: not performed.

### Colibri-surpass gap audit — 2026-10-12 (second deep pass)

This audit is the current source-of-truth for what is and is not implemented.
It was performed by reading source and statically following callers only. No
imports, tests, builds, compiled libraries, benchmarks, device probes, runtime
requests, numerical comparisons, commits or pushes were used.

#### Confirmed implementation, not a remaining source gap

The following findings were checked rather than inferred from filenames or
progress notes:

| Area | Source conclusion |
| --- | --- |
| Exception and cleanup `pass` statements | Not missing implementations. They are empty exception subclasses, expected `FileNotFoundError`/`OSError` cleanup suppression, or cleanup-only branches. |
| `segment_snapshot.py` validation loop | Not missing implementation. The first pass validates every chunk and the second pass performs the actual digest/write; the loop is intentionally side-effect free. |
| Low-bit/Q5/Q8 `_check_available` methods | Not missing implementations for the scalar backends. These providers are deliberately always available and execute through their checked scalar kernels; compiled/native availability is enforced separately by the native backend descriptors. |
| GLM5 production execution | Implemented through `native_glm5_next.NativeGlm5NextTextDecoder`, `publisher_glm5_loader`, `flat_native_session`, `torch_glm5_decoder`, device forward, factories, CLI and gateway paths. |
| P01 and P08 | Closed at source-contract level as recorded above. |

#### Explicit restriction that must not be mistaken for 100% completion

`independent_glm5_next.IndependentGlm5NextTextDecoder.forward` deliberately
raises `NotImplementedError`. That class is a contract/layout, ownership and
buffer oracle, not an arithmetic backend; its module explicitly requires the
compiled native decoder for arithmetic. This is acceptable only for the
independent oracle path. It is not evidence that GLM5 is fully closed: the
native symbol/ABI, compiled-library artifact, runtime capability and every
public caller still require the P02/P10/P11/P13/P14 source reconciliation
listed below. A source-only audit cannot certify the external compiled library.

#### Remaining work that prevents a truthful 100% claim

These are implementation-closure gaps, not test requests:

1. **P02 source matrix is not fully classified.** Every admitted row still
   needs a concrete source verdict for configuration, complete tensor
   inventory, resident and streamed loading, full token-to-logit operation,
   tensor-parallel/cache ownership, native/device ABI, factory, CLI, gateway,
   recovery and tooling. The catalog records candidates; it does not prove
   that every row has a complete consumer chain.
2. **P03 inventory-to-consumer reconciliation remains open.** Optional heads,
   shared/tied heads, MTP/auxiliary banks, vision/audio/video tensors,
   source-vs-execution dtype identity, positional variants and omitted expert
   banks must each be traced from admitted namespace to the execution owner.
   Qwen3.5 text configuration now rejects non-boolean `tie_word_embeddings`
   before inventory planning, preventing a truthy malformed value from
   silently switching the required-head inventory and execution alias policy;
   this closes only that configuration/admission consistency defect.
3. **P04/P07/P14 media and continuation composition remains open.** Multiple
   media ordering, expanded positions/types, native/device dispatch, borrowed
   projection ownership, K3 temporal packing, warm-session configuration,
   streaming/backpressure, stop-prefix and repeated recovery callers still
   need row-level closure evidence. Qwen3.5 publisher audio is now rejected
   before tensor payload reads, and Qwen4/Qwen3.5/GLM5 reject orphan media
   payloads before vision execution; these restrictions do not close the
   remaining family/caller matrix. Unsupported OpenAI streaming controls and
   `user` attribution without a connected sink are rejected, not implemented.
4. **P05/P06/P12 state and distributed closure remains open.** Accepted,
   rejected, bonus, EOS, context-limit and cancelled speculative paths must be
   reconciled with target/draft physical state and both RNGs. Distributed
   BF16/FP8/FP4 bank mapping, uneven/empty shards, root callback failure,
   collective failure ordering, cache ownership, stale/foreign restore and
   consensus rollback remain open source-review items. Rank schedule mismatch
   now poisons the affected projection bank, and failed distributed close
   retries cleanup on retained rank handles, but transport/process loss can
   still strand a group and close cleanup is not rollback or consensus.
5. **P09/P10 format and execution coverage remains open.** Every admitted
   source format still needs a concrete persisted-loader and execution-consumer
   mapping, including FP8 scales, E8/IQ3 geometry, codebook/rotation, omitted
   bank residency, conversion cleanup, device capability/version/dtype
   admission and proof that no admitted row silently falls back to host neural
   execution.
6. **P11 public ABI closure remains open.** Family-by-family Segment/Edge
   symbols, range boundaries, embedding/head schemas, continuation state,
   callback pinning/reentry/owner-thread rules and concrete service/factory/CLI
   wiring are not yet checked as one complete map.
7. **P13 evidence/tooling closure remains open.** Every family/backend/media/
   format cell needs a later-phase adapter, physical-cache leaf inventory,
   placement requirement, comparison binding and changed ABI/cache-schema
   consumer review. The opt-in comparison manifest now hashes a canonical
   binding of the exact execution row, both factory identities, and each
   role's complete artifact-pin records; the runner verifies that binding
   against the manifest before importing either adapter. This is structural
   tamper detection, not a signature, source-owner verification, adapter
   correctness proof, produced observation, cache receipt, or placement
   receipt. Bound comparison tuples now join the source catalog before
   adapters load, in addition to digest/pin/metadata checks. These structural
   gates are not produced observations, physical-cache receipts, or actual
   placement evidence; existing evidence objects do not close an unreviewed
   cell.
8. **P15 remains open by definition.** It cannot close while any matrix cell,
   cross-area case, finding, public route, recovery consumer or source
   restriction lacks an explicit IMPLEMENTED, CONTRACT RESTRICTION or MISSING
   classification with concrete symbols.

#### Final status after this audit

Phase 1 is **not 100% source-complete**. No additional hidden `pass`-based
implementation hole was found, but the required full source closure is still
missing in the matrix and cross-consumer reviews above. The largest concrete
technical limitation is that GLM5 arithmetic depends on an external compiled
native implementation; the Python independent decoder intentionally cannot
replace it. The remaining work is therefore to complete and, where needed,
implement the row-level consumer chains and ABI/format/media/recovery wiring,
then update each parent task and P15 only after every applicable requirement
has a concrete source result.

Execution checks: not run, as instructed. Commits and pushes: not performed.

### Latest checklist authority — 2026-10-13

This final whole-project comparison supersedes older audit text above that
called itself the current source of truth. Read it with the opening progress
table: P13/P14 remain IN PROGRESS, P15 remains OPEN, and complete Colibri
functionality parity is not established. No tests, imports, builds, benchmarks,
device probes, commits or pushes were run/performed.

### Whole-project Colibri contract recheck — 2026-10-13 (superseded by implementation checkpoint below)

This recheck tightens the preceding comparison using the upstream README and
the `docs/brio.md`, `docs/api.md`, `docs/grammar-draft.md`,
`docs/serve_protocol.md`, `docs/routing-telemetry.md`, `docs/CACHE_ROUTE.md`,
and `docs/multidisk.md` contracts, then tracing local route declarations and
handlers, gateway scoring, provider session ownership, and source-scope
classification. It is a source comparison only; it does not claim an executed
Colibri build or exhaustive equivalence for undocumented behavior. The
reference contracts can be inspected at
[Colibri README](https://github.com/JustVugg/colibri/blob/main/README.md),
[Brio/Jev](https://github.com/JustVugg/colibri/blob/main/docs/brio.md),
[API](https://github.com/JustVugg/colibri/blob/main/docs/api.md),
[grammar draft](https://github.com/JustVugg/colibri/blob/main/docs/grammar-draft.md),
[serve protocol](https://github.com/JustVugg/colibri/blob/main/docs/serve_protocol.md),
[routing telemetry](https://github.com/JustVugg/colibri/blob/main/docs/routing-telemetry.md),
[cache route](https://github.com/JustVugg/colibri/blob/main/docs/CACHE_ROUTE.md),
and [multi-disk](https://github.com/JustVugg/colibri/blob/main/docs/multidisk.md).

| Contract slice | Verified OpenAgent source state | Remaining source work |
| --- | --- | --- |
| Brio HTTP scoring | `/v1/brio` exposes options, question batches and dependent schema forms; session-backed scoring shares prefixes and restores the original boundary. It is explicitly routed to configured native publisher only; `cache_slot` is now explicitly rejected at the route boundary and guarded by the provider for non-route callers. | Compare exact Brio request/response/error fields; implement persistent isolated `cache_slot` history only with a bounded multi-session owner and durable snapshots, or retain the enforced restriction; add terminal/browser clients after the HTTP contract is stable. Direct K3/V4.1 remain rejected. |
| Jev `/v1/systemone` | Local adapter supports `noul`, `choice` and `score`, 1 MiB aggregate question text, up to 255 choice labels, JSON object/array state and instructions, arbitrary nonempty Jev alias input, and served-model output. | Reconcile exact per-field serialization, normalization, usage/rounding and validation/error envelope. |
| OpenAI-compatible serving API | Added local model listing, chat completion and legacy text completion routes, native-publisher-only; basic text validation, stop limits and explicit tool/logprob/grammar rejection. | Synchronous subset only. SSE, exact controls/continuation/usage/finish semantics, complete legacy semantics, grammar/tools and per-family support/rejection matrix remain open. |
| Anthropic API | Added synchronous `/v1/messages` text-only route, basic system/text-block handling and explicit unsupported-tool/block rejection. | `x-api-key`, Anthropic error envelope, SSE event lifecycle/pings, cancellation and full request/stop behavior remain open. |
| Grammar and tools | Source-scope marks constrained grammar and Colibri tool-call protocol restricted/not admitted; generic tool execution and family decoding are separate. | Implement the documented grammar subset/response-format path and per-family tool protocol, or keep each unavailable route/feature explicitly rejected. |
| Native serve protocol | OpenAgent has worker/engine command protocols, but the Colibri READY-gated mux/legacy stdin contract with CAPS/STAT/HWINFO/TIERS/EMAP was not found as a compatible surface. | Map request framing, readiness, forward-compatible unknown lines, slot bounds, errors and capability reporting; do not treat internal worker messages as wire parity. |
| Scheduling and operations | `/health` and Prometheus `/metrics` now expose actual native execution-scheduler active/waiting/capacity/byte gauges when the service owner provides a snapshot; missing snapshots are reported as unavailable instead of fabricated zeroes. The capability endpoint now explicitly restricts Colibri request-queue semantics. | No compatible Colibri request queue admission, timeout/429 or wait headers; no completed/rejected/timeout counters or histograms; auth-protected operational semantics and caller-level lifecycle remain open. |
| Routing telemetry and expert atlas | OpenAgent has expert tier and system telemetry; no common `.coli_usage` format, shared route-trace persistence, or matching atlas workflow was located. | Implement route-level record lifecycle and consumers, or record an explicit product exclusion. Generic counters do not close it. |
| Cache-aware routing | OpenAgent has expert residency/promotion mechanisms; no Colibri top-J preservation/top-M resident substitution and agreement/KL/stat semantics were found. | Implement the family-limited selector and observability if in scope, or explicitly restrict it; residency alone is not cache-route parity. |
| Multi-disk mirror | Streamed expert stores and cache directories exist; mirror identity/header agreement, read-error fallback, weighted striping and partial-mirror planner were not found. | Implement or enforce rejection for mirrored and partial layouts, including fallback and planning contracts. |
| Model-family behavior | OpenAgent has its own nine-family catalog and significant native/device/media/state code. | Continue P02–P14 row/caller closure; family name or broad feature overlap does not prove loader-to-logits, backend, media, recovery, API, and operational parity. |

These distinctions are now encoded in `_COMMON['colibri_surfaces']` in
`openagent/adaptive_engine/source_scope.py`. The labels are explicit
source-scope classifications, not validation of implementations or runtime
certificates. In particular, “missing” does not silently become a supported
matrix capability, and the Brio/Jev entry records only the implemented
subset.

#### Updated remaining-work order

1. Close core Phase 1 P02–P12 source matrices and public caller chains as
   described in the status table; these block P15 regardless of Colibri parity.
2. For P13/P14, choose and implement the intended Brio/Jev wire subset,
   cross-request cache/snapshot behavior and clients; add or reject the
   OpenAI/Anthropic APIs and native serve protocol; close grammar/tool protocol,
   queue/health/metrics, route telemetry/cache-route and mirror behavior.
3. For each feature, trace concrete route → validation → provider/engine →
   output/error/lifecycle owners and every applicable family/backend. Mark
   IMPLEMENTED only with a complete path, or record an enforced contract
   restriction with a specific reason.
4. Reconcile any reference behavior that is optional, experimental, family-
   limited or drafted instead of assuming all Colibri docs represent a
   universally enabled production feature.
5. Keep P15 OPEN until these applicable source closures and the existing P02–
   P14 work are actually complete.

No execution was used for this update: tests, imports, builds, compilation,
benchmarks, device probes, runtime comparisons, commits and pushes were not
performed.

### Source implementation checkpoint — 2026-10-13

#### Superseding source-audit correction

The checked boxes and closure paragraphs in the original implementation plan
below are historical implementation notes, not proof that the current Phase 1
requirements are closed. The current audit remains authoritative: P02–P07,
P09–P12 and P14 are still open where the top-level status table or deep
callable-path findings say `IN PROGRESS`/`OPEN`. In particular:

- P02 matrix rows are candidate source references, not verified combinations.
  The generator expands family/backend/source-dtype/persisted-format products
  and attaches family-level owners; this does not prove each exact tuple reaches
  the matching loader, arithmetic consumer, factory, CLI, gateway, recovery,
  and tooling path. Do not use `implemented-source` on a tuple until that exact
  route has been traced; otherwise apply an enforced restriction or mark it
  unverified.
- P03 inventory and P04 media composition are not closed by admission schemas
  alone. Preserve each family’s tensor producer-to-logits and media
  ordering/position/ownership/cleanup trace as open until callers are covered.
- P05 has strong speculative snapshot/rollback primitives, but caller-level
  agreement across acceptance, rejection, bonus, EOS, context limit,
  cancellation and repeated restore remains unproven.
- P06 is open: rank orchestration does not establish per-bank collective
  consumers, all layouts, failure/cancellation ordering, or consensus rollback.
- P07 is open: K3 primitive input/media/batch/FLA paths do not establish all
  serving routes. K3 Metal, speculation/MTP, video policy, and text-only public
  adapters remain explicitly restricted where applicable.
- P09–P11 and P12 remain open for exact format/backend consumers, device
  admission/no-fallback, complete public ABI maps, and family physical-state
  recovery through callers. P13/P14 remain open as stated above.

This correction supersedes any earlier `Closure: DONE` sentence or checked
checkbox in the historical sections when it conflicts with the top-level status
table or this audit. It does not erase evidence of implemented primitives.

This checkpoint records implementation work after the recheck above. It does
not promote the Phase 1 parent tasks or certify Colibri parity. The worktree
already contained extensive edits in the adaptive-engine source; those edits
were preserved. The following additional source changes were made in this
turn:

| Area | Source work completed in this iteration | Remaining / status |
| --- | --- | --- |
| Jev `/v1/systemone` | State and instructions now accept text, JSON objects, or arrays serialized with bounded canonical JSON; arbitrary nonempty Jev model aliases are accepted and the configured served model is returned; 255-label choice and aggregate text bounds retained. | Exact per-field Jev normalization, response usage/rounding and validation/error-envelope parity still need comparison and correction. |
| Local OpenAI-compatible surface | Added `/v1/models`, `/v1/models/{model}`, `/v1/chat/completions`, and legacy `/v1/completions` routes, native-publisher-only; validates text request shapes, limits/stops and explicitly rejects unsupported tools/logprobs/grammar formats. The capability response now calls out that non-empty text `stop` sequences are rejected pending tokenizer-aware stopping. | Synchronous subset only: SSE, exact finish/usage semantics, continuation behavior, complete OpenAI controls, grammar, tools, per-family support matrices, queue and rate policies remain open. The legacy `echo` shape is a partial adapter. |
| Local Anthropic surface | Added synchronous `/v1/messages`; accepts text-only string/text-block content and system text, emits a basic message response, returns an Anthropic-shaped error envelope on handled route errors, and explicitly rejects tools/thinking/unsupported blocks and unwired sampling/metadata controls. The capability response now calls out that non-empty `stop_sequences` are rejected pending tokenizer-aware stopping. | Exact event stream lifecycle/pings and complete request fields remain unsupported. Streaming/tool use is not implemented; `x-api-key` is supported only through OpenAgent’s route-scoped auth path, not as full Colibri credential parity. |
| Operations | Added `/health` and Prometheus `text/plain; version=0.0.4` `/metrics`; when available they include low-cardinality native-provider and real execution-scheduler gauges, with unavailable queue values left null/omitted. The capability contract explicitly restricts Colibri HTTP queue semantics. | Metrics do not include Colibri completed/rejected/timeout counters or histograms; HTTP queue admission/429/wait headers are not implemented and remain an explicit restriction, not an implied feature. Auth policy and caller-level operational behavior need closure. |
| Auth | New `/v1/*` model and inference paths are covered by the existing optional OpenAgent local-AI scopes. On `/v1/messages`, requests that provide both a bearer credential and `x-api-key` now fail with HTTP 400 rather than silently selecting one credential. | Not Colibri `COLI_API_KEY`/`x-api-key` authentication parity; credentials still use OpenAgent's local-token and scope model. |
| Brio `cache_slot` | The route explicitly rejects the field with HTTP 422 before provider invocation; the provider also rejects it for non-route callers instead of silently ignoring it. | Persistent isolated KV slots and state-derived slot selection are not implemented. The current native publisher resets its session cache per request; snapshot APIs alone do not supply the Colibri serving lifecycle. |
| Colibri-specific restrictions | Current provider continues to reject direct K3/V4.1 closed-set scoring, constrained GBNF/tool protocols, and uncomposed families. | GBNF draft-source verification, per-family tool calling, READY/mux/legacy stdin protocol, route trace/`.coli_usage`, cache-route substitution and mirror/partial-mirror planning remain missing/restricted. |
| Colibri capability boundary | Added `GET /v1/colibri/capabilities`, backed by `adaptive_engine.colibri_capabilities.colibri_capabilities`, which reports each Colibri-only surface as partial, restricted, or not admitted, with its actual entrypoint and narrow scope. Existing request guards remain authoritative (including `cache_slot`, grammar, and tool rejection); absent protocol/storage routes remain unavailable rather than implied by ordinary generation. `colibri_queue_contract` is now `restricted`: the native scheduler remains internal, while HTTP queue admission, 429/Retry-After, and queue-wait headers are unavailable. | This is explicit scope disclosure, not compatibility implementation or evidence. Durable `cache_slot`, clients, full OpenAI/Anthropic contracts, READY/mux, GBNF/tool protocols, `.coli_usage`/atlas, cache-route substitution, and multi-disk mirror planning remain open or restricted. |
| Multi-disk mirror admission audit (2026-10-14) | The Colibri capability catalog has `multi_disk_mirrors` as `not_admitted` with no entrypoint. No Colibri route/provider accepts a mirror or partial-mirror configuration. OpenAgent's separate OLMoE operator accepts explicit `--stripe NAME=PATH` pack destinations and explicit `--storage-root NAME=PATH` execution roots; `SparseExpertManifest` assigns each region one storage name, and `DiskExpertStore` resolves that region against exactly one declared root. This is named placement/striping, not mirror identity, fallback, replica selection, or partial-mirror planning. | No speculative rejection was added: the actual OpenAgent multi-root route has explicit single-root region selection, and the Colibri mirror feature has no admitted API/configuration entrypoint to reject. Keep Colibri mirrors restricted; do not describe explicit OLMoE roots or striped packing as Colibri mirror parity. |
| P02–P12 source closure | No parent status changed in this iteration; checklist owner/callable-chain rows remain the governing work items. | Family/backend/format loader-to-logits and ABI/factory/CLI/gateway/cache/recovery/tooling closure is still outstanding per the P02–P12 table and deep audits. |

#### Follow-up source corrections in this iteration

- Compatibility API responses now bypass OpenAgent's default response envelope;
  handled `/v1/messages` failures use the Anthropic error envelope. This remains
  a synchronous local subset and does not claim full wire compatibility.
- OpenAI/Anthropic sampling controls, OpenAI seed and nonzero penalties, and
  unsupported text stop sequences are rejected when the native provider cannot
  honor them. `/v1/systemone` scoring output indentation was corrected.
- Removed unreachable Brio cache-slot snapshot bookkeeping. `cache_slot` still
  receives an explicit unsupported error; no persistent cache-slot feature is
  claimed.
- Source-scope Cartesian rows now carry `candidate-unverified` rather than
  `implemented-source`/`admitted` status, and select only the format consumer
  for the row's backend. P02 remains open until exact tuple chains are traced
  or concretely restricted.
- `MultimodalSessionExecutor.forward_media` now checks cancellation during
  media-field admission and immediately before/after each owned array copy;
  canceled or partially staged inputs are not dispatched to the decoder. This
  closes that staging boundary only, not each family media caller or all
  interrupted-construction cleanup.
- Native generation-cache restore now closes its staged cache frame before
  publishing restored service history, and attempts staged-frame plus rollback
  snapshot cleanup without replacing the primary restore/cancellation error.
  P12 remains open for cross-family physical-state and caller-ordering review.
- `/health` and `/metrics` now read actual execution-scheduler active/waiting,
  capacity and byte-reservation gauges from the native service when available;
  unavailable snapshots are not replaced with zero-valued queue claims. This
  does not implement Colibri timeout/429/wait-header behavior or counters and
  histograms, so P14 remains open.
- Anthropic-compatible authentication now rejects requests that supply both
  `Authorization: Bearer` and `x-api-key`; the route previously chose bearer
  implicitly. This prevents ambiguous credential selection only. It does not
  implement Colibri API-key semantics or close P14.

No tests, imports, builds, benchmarks, device probes, commits or pushes were
performed. Changes were reviewed by source inspection only.
| P13/P14/P15 | Partial source APIs and ops surfaces added; no phase completion claim. | Evidence/cache/placement bindings and media/stream/backpressure/stop/recovery/warm-session/error cleanup closure remain. P15 remains OPEN. |

#### Remaining source work — do not lose

1. Repair and then source-review the new compatibility handlers for exact
   route contracts, bounds, response envelopes, token accounting, unsupported
   values, cancellation and cleanup. Streaming is not implemented, so do not
   advertise it as supported. The compatibility handlers currently require
   synchronous requests (`stream=false`); JSON-schema/GBNF is deliberately
   rejected.
2. Close Brio schema per-field serialization/normalization and exact response
   fields. Implement cross-request `cache_slot` only together with a bounded,
   isolated multi-session native owner and durable slot history; otherwise
   keep returning an explicit unsupported error. Add the terminal and browser
   clients only after the HTTP contract is stable.
3. Add or explicitly restrict local API key semantics. OpenAgent scoped bearer
   auth is not interchangeable with Colibri bearer/API-key or Anthropic
   `x-api-key` authentication. Add Anthropic SSE only with correct event order,
   disconnect cancellation, terminal events and errors-before-headers behavior.
4. Implement the GBNF byte-subset parser/walker as a verified draft source
   integrated with an actual target-forward speculative verification owner;
   grammar must never constrain sampling. Implement family-specific tool
   parsers/serializers or reject unsupported families explicitly.
5. Implement or explicitly exclude the byte-counted READY-gated mux and legacy
   engine protocol. Complete queue admission, timeout/429, queue-wait headers,
   `/health` scheduler counters and protected Prometheus metrics from a single
   real scheduling owner rather than synthetic counters.
6. Implement or exclude `.coli_usage`/route-trace and expert atlas; implement or
   exclude the documented family-limited cache-route selector and its agreement
   telemetry; implement or exclude mirror validation/fallback/weighted reads
   and partial-mirror planning. Generic caches, telemetry or streamed storage
   do not close these rows.
7. Continue all P02–P12 family/backend/format source matrices and P13/P14 caller
   chains in the current progress and deep-audit sections. Do not mark P15 done
   until every applicable row has concrete implemented source evidence or a
   specific enforced restriction.

No tests, imports, builds, compilation, benchmarks, device probes, runtime
comparisons, commits or pushes were performed for this checkpoint. A static
`git diff --check` was used only as whitespace hygiene; it is not a source or
runtime correctness certificate.

### P04/P07 media-span restrictions — 2026-10-13

`NativeExecutionService.generate` admits a media-bearing prompt only when the
prompt is nonempty and fits one admitted prefill operation. This prevents the
current service from forwarding the same unplanned media tensors for each text
chunk, which could duplicate media expansion and corrupt placeholder/position
alignment. `K3BatchExecution.forward` accepts an explicit `media` argument only
to reject it with a precise text-only restriction; media expansion remains an
individual-slot operation. These restrictions do not close P04/P07: per-family
media-span planning, full K3 temporal/FLA serving composition and caller-level
recovery remain open. No tests, imports, builds, benchmarks, device probes,
commits or pushes were run for this source change.

### P04 K3 publisher media position extent — 2026-10-13

The public `PublisherSession._generate` caller now passes rotary positions for
the original token chunk into `K3Execution.forward`, even when the caller has
already computed an expected expanded output count for its shape check.
`K3Execution` performs placeholder-to-feature expansion and therefore must
derive the expanded rotary positions itself. Previously the CLI passed the
expanded count as the position-vector extent, while the executor validates
positions against the unexpanded token input before expanding them; any media
prompt whose projected feature count differs from its placeholder count was
rejected before vision dispatch. The caller still uses executor output rows and
committed cache position to validate the expanded count, and its request-level
exception path resets cache state on interruption/failure. This closes that
specific K3 image/video caller wiring defect only: it does not establish
processor parity, all media routes, K3 serving integration, or P04/P07/P14
completion. No tests, imports, builds, benchmarks, device probes, commits or
pushes were run.

### P02 Qwen4 in-memory device factory identity admission — 2026-10-15

`create_device_qwen4_execution_service` now repeats the exact
`qwen4_exp_text` model identity and supported causal-LM architecture checks
before constructing the device service, matching the directory-backed
factory. A tensor map with compatible geometry can no longer enter this
public route under a different or unpinned architecture declaration. This
closes one in-memory factory admission seam only; full tuple propagation and
family/backend/format chain tracing remain open. P02 remains IN PROGRESS and
P15 remains OPEN. No tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P06 activation-fingerprint staging across projection banks — 2026-10-16

The generic, BF16, and packed FP8/MXFP4 banks fingerprint replicated inputs
before comparing rank digests. Building the byte representation for
`sha256(vector.tobytes())` allocates, and was previously outside a local-stage
agreement; a rank-local allocation failure could make peers wait at the next
digest collective. Each bank now prepares the activation digest inside its
local-stage agreement before comparing digests. Projection weights remain
rank-local immutable copies, with registration fingerprints and provider pins
bound to the bank identity; these operations do not mutate a persistent KV or
session cache and have no cache rollback requirement. Failures inside the
transport collective itself, including rank/process loss or a broken
transport, remain uncertain and require group restart; no process-loss
recovery is claimed. P06/P12 remain IN PROGRESS and P15 remains OPEN. No
tests, imports, builds, benchmarks, device probes, commits, or pushes were
run.

### P05 resumed stop-prefix boundary uses complete decoder history — 2026-10-16

The DSpark decoder tests stop prefixes against its full retained history, but
the V4.1 resume caller validated `decoder._stopped` and its own terminal
decision against generated tokens alone. A prefix spanning the prompt/generated
boundary could therefore be rejected as an inconsistent restored snapshot or
be treated as nonterminal by the caller. Resume validation and its terminal
decision now use `prompt_ids + generated_ids`, matching the core decoder's
established semantics. This is limited to resumed DSpark stop-prefix parity;
other caller-specific stop rules, accept/reject equivalence, opaque restore
contracts, and sink atomicity remain open. P05 remains IN PROGRESS and P15
remains OPEN. No tests, imports, builds, benchmarks, device probes, commits,
or pushes were run.

### P05 publisher stop-prefix policy aligned across speculative and plain routes — 2026-10-16

After aligning DSpark resume checks with its decoder's complete retained
history, the plain V4.1 sampling loops still matched stop prefixes against
generated IDs only. Fresh and resumed plain routes now use the same
`prompt_ids + generated_ids` suffix policy as speculative decoding, so the
acceptance mode does not change stop-prefix behavior at a prompt boundary.
This does not prove that every gateway or family implements matching stop
semantics, nor does it close target/draft acceptance parity, cancellation
coverage, opaque restore contracts, or external sink atomicity. P05 remains IN
PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P05 direct proposer reset and owner-separation contract — 2026-10-16

`SpeculativeDecoder` now rejects a proposer that aliases the target or lacks a
`reset()` operation in addition to its paired snapshot/restore methods. A
rollback failure quarantines the decoder and disables receipt restore, so a
proposer without reset had no supported way to recover both physical owners;
resetting only the target could also silently reuse stale draft state. The
top-level owner separation guard prevents the same object from being restored
twice to incompatible target and proposer receipts. Built-in DSpark and
autoregressive proposers implement reset. An exposed proposer RNG must also
provide both `choice` and restorable bit-generator state, preventing malformed
public RNG ownership from being silently omitted from rollback. This does not
inspect or authenticate opaque nested owner state, prove private RNG inclusion
in custom snapshots, or
provide output-sink atomicity; P05 remains IN PROGRESS and P15 remains OPEN.
No tests, imports, builds, benchmarks, device probes, commits, or pushes were
run.

### P11 Qwen4 routing callback cleanup error preservation — 2026-10-16

`NativeQwen4PublicAdapters.routing_scope()` now preserves a primary observer or
body failure if native observer-scope release also fails. It retains the
callback and owning-thread cleanup record for retry and attaches the release
failure as a note instead of replacing the original error. A release failure
without a prior error still surfaces normally. This closes one Qwen4 callback
ownership/error-ordering seam only; arbitrary native callback safety and the
full family ABI map remain open. P11 remains IN PROGRESS and P15 remains OPEN.
No tests, imports, builds, benchmarks, device probes, commits, or pushes were
run.

### P11 Segment/Edge fixed capability output extent — 2026-10-16

The native C `coli_segment_engine_capabilities` and
`coli_edge_engine_capabilities` functions now clear only the fixed known
capability-record size before copying the record. They no longer use the
caller-declared `caps->size` as a `memset` extent; that field is a capacity claim,
not proof that the pointed-to allocation is that large. This prevents a
malformed oversized capacity from causing an out-of-bounds write past the
known ABI record. It does not make arbitrary pointers safe or close the wider
family ABI/schema and caller review. P11 remains IN PROGRESS and P15 remains
OPEN. No tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P03 Expert-provider identity preflight and residual content binding — 2026-10-16

The Mixtral, Qwen3.5 and Qwen4 Torch directory selectors now validate the
provider's exact family, manifest architecture/config fingerprint, and full
expert geometry against the checkpoint configuration before the expert tensor
payloads are omitted from the read selection. The GLM-DSA device selector
likewise checks its packed-manifest/config fingerprint (or original publisher
provider fingerprint), in addition to the previously recorded owner type,
contract, sparse-layer coverage, and geometry checks. This closes late
constructor-only rejection for mismatched family/config manifests and prevents
their geometry from being used to authorize selected reads.

The separate packed-expert manifest fingerprint is currently a hash of the
configuration contract, not a digest of the checkpoint's expert tensor bytes.
Therefore two checkpoints with identical configuration but different expert
weights can still share the same config fingerprint; packed expert payloads
are internally hashed, but that does not prove they came from the selected
checkpoint. Fail-closed restrictions now reject those mixed-source providers
at the Mixtral native/device decoder constructors and streamed service path,
the Qwen3.5/Qwen4 directory text-device paths, and GLM-DSA native/device
decoder paths, before expert tensors are omitted or consumed. Qwen3.5/Qwen4
external projection-provider paths are likewise rejected; Qwen4's internally
created projection owner remains allowed because it is constructed from the
same directory snapshot and rechecks that snapshot before use. Publisher
conditional loaders retain their internally constructed expert/projection
owners from the same inventory snapshot. Re-enabling external packed experts
requires a versioned manifest carrying a full source-content digest and a
loader check against the exact checkpoint files; legacy manifests have no such
proof and remain rejected on the affected routes. This restriction does not
establish general artifact-to-source binding for every provider family or
consumer. P03/P15 remain IN PROGRESS/OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P09/P10 target-family conditional device-wrapper admission — 2026-10-16

Added private publisher-admission tokens to the device-backed Qwen3.5, Qwen4,
and Inkling multimodal wrapper constructors. The metadata-planned
`publisher_conditional_loader` supplies the token only after its family
inventory/format admission; native checkpoint wrapper paths remain unchanged
because they select CPU/native text execution. Direct construction of those
three multimodal wrappers with `text_device` now fails closed before tensor
partitioning or decoder construction, preventing arbitrary pre-materialized
arrays from bypassing the publisher route's format admission. GLM5 conditional
construction performs its own directory inventory admission, and direct
in-memory family factory routes for these families remain rejected where
source provenance cannot be established.

Residual: low-level exported Torch decoder constructors still accept exact
FP32 in-memory inventories directly; callers outside these guarded wrappers
must be separately reviewed or restricted. This change does not prove runtime
device placement, per-operation no-fallback, or arithmetic correctness. GLM-DSA
direct device construction and all actual placement receipts remain open.
P09/P10 remain IN PROGRESS and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P09/P10 Mixtral direct device-constructor source gate — 2026-10-16

`TorchMixtralDecoder` can now be constructed only through its metadata-admitted
`from_directory` path. The raw in-memory constructor previously accepted a
shape-compatible tensor mapping without the checkpoint header/source-format
admission performed by `read_native_text_checkpoint`; its low-level device
class was therefore an alternate route around the family policy. The
directory reader supplies a private admission token only after the full
metadata route has succeeded. This closes the direct Mixtral low-level device
constructor bypass only. Equivalent low-level constructor paths for other
device families remain to be reconciled. `admit_no_fallback` remains a source
policy, not proof that every device operation stayed on-device; placement and
runtime receipts are still outstanding. P09/P10 remain IN PROGRESS and P15
remains OPEN. No tests, imports, builds, benchmarks, device probes, commits,
or pushes were run.

### P04 V4.1 batched image-span geometry admission — 2026-10-16

`PublisherV41Vision.merge_image_embeddings()` now budgets its per-sample
occupancy maps across the full batch. It also validates aligned image-grid
geometry directly against the bounded borrowed token-type span instead of
allocating a second expected-type array sized from image geometry metadata.
This avoids an unbudgeted geometry-derived allocation and closes the batch
mask undercount in this merge helper. It does not establish all-family media
ordering/position behavior or caller lifecycle closure: Qwen, GLM, Inkling,
K3, V4.1, audio, and video routes still need per-caller source/evidence review.
P04/P07 remain IN PROGRESS and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P04 Qwen3.5/Qwen4 pre-tower context admission — 2026-10-16

The shared Qwen multimodal forward now checks current text position and the
family decoder's physical context capacity immediately after validating the
expanded prompt IDs, before embedding lookup or vision execution. The text
decoder retains its own mutation-boundary check. This avoids spending a full
media transaction on a prompt that cannot fit, for both Qwen3.5 and Qwen4
wrappers using the shared path. It does not close processor timestamp/frame
alignment, cross-family media evidence, or serving cancellation and retry
behavior. P04/P14 remain IN PROGRESS and P15 remains OPEN. No tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

### P14 queue/health/metrics and native warm-session audit — 2026-10-16

Reconciled Colibri queue capability disclosure with route behavior: the three
compatibility generation endpoints reject all listed queue/wait/priority
keys based on presence, including null-valued controls; `/v1/brio` rejects
`cache_slot` before scoring, and the provider independently rejects it for
non-route callers. `/health` reports application health and only labels
queueing metrics available when the native runtime supplies a scheduler
snapshot; `/metrics` emits actual provider/scheduler gauges and omits
unavailable scheduler samples. Capability wording now records these exact
boundaries. Reviewed native publisher warm-session expiry/cleanup: it is
explicitly lazy on request/status, retains failed cleanup ownership for retry,
and surfaces `cleanup_pending`; no additional defect was established in this
bounded source review, and this is not a runtime expiry/cleanup receipt.
P14 remains IN PROGRESS and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P02/P11 family-conformance registry and capability reconciliation — 2026-10-17

The family conformance report previously mixed the nine identities in the
local Colibri registry with OpenAgent-only model identities without exposing
that distinction. It now reports the exact Colibri registry separately and
marks each engine row's membership; `mixtral` and legacy `inkling_text` are
explicit engine extensions rather than claimed Colibri registry entries. The
Qwen4 independent implementation contract now matches its decoder's actual
default-or-linear text contract instead of the older default-only label.

The first full conformance run exposed stale expected counts, identity order,
and capability lists (the engine catalog has 19 model identities, while the
Colibri registry has 17). After the report and assertions were reconciled,
private per-process snapshots now avoid re-expanding/revalidating the same
catalog for every conformance report; public report and execution-case APIs
still return detached copies. The complete `tests/test_family_conformance.py`
suite passes: 11 tests, 1 skipped, in 3.5 seconds (previously 569 seconds).
`tests/test_source_scope_inventory.py` passes 4 tests. These are source/tooling
results only: they do not admit any P02 route, prove decoder parity, or supply
P13 runtime evidence. P02/P11 remain IN PROGRESS and P13/P15 remain
IN PROGRESS/OPEN.

### P12 K3 recovery staging descriptor handoff — 2026-10-17

K3 generation checkpoint read and write previously created a temporary staging
file and closed its initial descriptor before entering the cleanup-protected
block. If descriptor close failed, the staging path could remain behind. Both
paths now use a shared handoff helper that removes the staging file before
propagating a close failure. The injected close-failure regression and generic
checkpoint recovery suite pass 8 tests; `git diff --check` is clean. This
source-only inventory run reports `inventory_sha256=6e2210b39d01bbb726e401ae9dd4676cba9749210addd4265f43d6cb9963f6cc`,
`source_only=true`, `executed=false`. This closes only the K3 temporary-file
handoff edge; distributed consensus,
transport/process-loss recovery, and family-wide physical-state ownership
remain open. P12 remains IN PROGRESS and P15 remains OPEN.

### P12 File-cache transport staging descriptor handoff — 2026-10-17

The generic file-backed cache transport had the same pre-cleanup descriptor
close edge: `mkstemp` succeeded, but the staging path was assigned only after
`os.close`, so a close exception could leave the temporary file behind. It now
uses a cleanup-protected staging-path helper. Its injected close-failure test,
the K3 equivalent, and cross-family cache restore/rollback regressions pass 14
tests. Source inventory reports
`inventory_sha256=f92d6ec358136a5bd4d223a2f0bad2485f95e34d122e1cce75baeedf20816f4d`,
`source_only=true`, `executed=false`. This closes only local staging cleanup;
it does not provide distributed transport recovery, consensus, or process-loss
handling. P12 remains IN PROGRESS and P15 remains OPEN.

### P04/P14 Publisher NumPy image wrapper ownership — 2026-10-17

`PublisherMediaProcessor._image` created a temporary PIL image wrapper around a
borrowed NumPy array and converted it without explicitly closing the wrapper.
The conversion now uses the wrapper's context manager, closing it on both
success and exceptions while returning the detached RGB image to the caller.
The new success/failure ownership regressions plus conditional-session, media
input and ordering suites pass 20 tests. Source inventory reports
`inventory_sha256=d2c10810613fe0244cb98dc66ca281276b4bb9915663bb2eb65336615ed67a44`,
`source_only=true`, `executed=false`. This fixes one preprocessing ownership
edge only; cross-family media order, temporal dispatch and caller lifecycle
closure remain open. P04/P14 remain IN PROGRESS and P15 remains OPEN.

### P14 Publisher selected-video cancellation-aware ownership transfer — 2026-10-17

The streaming gateway detached each admitted `PublisherVideoInput` frame with
cancellation checks, then called the public constructor, which copied every
frame again without cancellation. A private invariant-checking adoption path
now accepts only immutable detached uint8 RGB arrays with ordered frame
metadata, consistent geometry, bounded extent and normalized FPS. The gateway
uses it to transfer its own already-detached frames, removing that second
uncancellable full-frame copy while preserving caller-buffer isolation. The
stream-cancellation, selected-video, PIL-wrapper, and conditional-session
suites pass 22 tests. Source inventory reports
`inventory_sha256=c217199cd49c6b8c3cf34d15def642c60d464f1f9ea8613a28109b72a9440522`,
`source_only=true`, `executed=false`. This closes only this publisher video
copy/ownership seam; cross-family temporal paths and caller lifecycle remain
open. P14 remains IN PROGRESS and P15 remains OPEN.

### P05 speculative accept/reject, bonus, EOS and context boundaries — 2026-10-17

Added deterministic transaction-level regressions over the real V4.1
`SpeculativeDecoder` with independent target/proposer state owners. They cover
fully accepted proposals plus the target bonus token, EOS and context-limit
suppression of the bonus, and rejection followed by target replay and proposer
RNG rollback. Sampling temperature conversion is now centralized and overflow-
safe across proposer/decoder constructors, V4.1 request admission and sampling;
huge integers, booleans and non-finite values fail as `ValueError`. The focused
transaction path now rejects null target/draft rollback receipts before
prefill, speculative mutation, or restore; regressions confirm owners remain
unchanged when receipts are missing. The transaction and cancellation suites
pass 16 tests. Snapshot restore now also checks target/draft histories and both
RNG states against the captured boundary. Seeded stochastic cases check target RNG draws for full accept
plus bonus and rejection plus residual sampling, while confirming the draft
RNG rewinds on rejection. New cancellation cases inject interruption after proposer state
and RNG mutation and after target verification advances its cache; both target
and draft histories plus both RNG streams return to their pre-step boundary.
This validates these core decoder boundaries only; stochastic multi-token
parity, cancellation/restore across all serving callers, and atomic event
publication remain open. P05/P14/P15 remain IN PROGRESS/OPEN.

### P14 Anthropic browser preflight headers — 2026-10-17

The local `/v1/messages` auth gate accepts Anthropic's `x-api-key`, but trusted
browser preflight did not advertise that header or `anthropic-version`. The
preflight response now allows both only for `/v1/messages`; other route groups
retain the prior header allowlist. The runtime forwards the version header into
the route, which rejects missing or invalid date-form values before invoking a
provider. Regressions cover trusted/untrusted origins, endpoint-scoped preflight,
valid usage, route-level missing/invalid versions, and runtime header forwarding
through the HTTP handler. The focused HTTP-boundary, inference-route, and
conditional-session suites pass 20 tests. This closes only
the Anthropic header/auth wiring seam; streaming, full Anthropic semantics, and
other P14 lifecycle requirements remain open.

### P14 V4.1 compatibility usage and terminal reason — 2026-10-17

The V4.1 publisher owner returned `prompt_ids` and `generated_ids` but omitted
prompt usage and finish-reason metadata. The OpenAI/Anthropic compatibility
route therefore reported zero prompt tokens for V4.1 and treated length-limited
output as a normal stop. V4.1 initial and resumed generation now return exact
prompt/expanded token counts and terminal reason (`length`, `eos`, or
`stop_prefix`); the route also derives prompt usage from `prompt_ids` for
compatible runtime records without the new counters. Regressions verify both
the Anthropic response mapping and producer-side exact prompt count plus EOS
reason. The inference-route and conditional lifecycle suites pass 17 tests.
Other family/caller compatibility semantics remain open; P14/P15 remain IN
PROGRESS/OPEN.

### P13 evidence runner enforces dependency artifact pins — 2026-10-17

The comparison configuration builder already required each oracle and
candidate to pin source, config, weights, tokenizer, and dependency artifacts,
but the runner's final `pinned_artifacts` validation accepted manifests
missing the dependency category. The runner now enforces the same five-kind
set at execution time, so hand-edited manifests cannot bypass dependency
artifact pinning. A regression creates and hashes the four other artifact
kinds and verifies the runner rejects the incomplete pin set. The source-scope,
native-state-protocol, and OLMoE adapter suites pass 32 tests. This is a
structural pin-completeness fix, not dependency authority review or runtime
evidence; P13/P15 remain IN PROGRESS/OPEN.

### P14 compatibility route rejects cross-endpoint ignored fields — 2026-10-16

The shared OpenAI request allowlist intentionally covers both text-completion
and chat-completion endpoints, so endpoint-specific fields need an additional
boundary check. `/v1/completions` now rejects `messages` and
`max_completion_tokens`; `/v1/chat/completions` now rejects `prompt`. These
fields were previously accepted by the shared allowlist and silently ignored
on the other endpoint. Capability disclosure now also states that the
synchronous compatibility routes do not implement request-disconnect
cancellation. Anthropic auth continues to use route-scoped `x-api-key` mapped
to OpenAgent local tokens/scopes, not Colibri API-key parity. P14 remains IN
PROGRESS and P15 remains OPEN; full request semantics, streaming, stop
adaptation and cancellation remain incomplete. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P04 Inkling media prompt admission — 2026-10-16

`InklingMultimodalDecoder.forward()` now rejects prompts exceeding the
configured physical context before text embedding or image/audio encoding, and
validates image/audio placeholder IDs as distinct in-vocabulary integers
before dispatch. This closes avoidable media work and ambiguity at this
caller boundary; it does not establish audio frame-to-placeholder temporal
semantics or close serving cancellation/retry behavior. P04/P14 remain IN
PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P04 family media-route boundary audit — 2026-10-16

Source trace for this pass: the public conditional factory admits Qwen3.5,
Qwen4, GLM5-next and Inkling only; the processor assigns image/video outputs
only to Qwen3.5/Qwen4/GLM5 and audio only to Inkling. GLM-MoE-DSA has no
multimodal decoder route in this factory and remains text-only/restricted
there. Qwen3.5/Qwen4 wrappers reject audio inventories; Inkling has no video
processor route. The processor explicitly rejects mixed image/audio/video
requests because its upstream processor API does not preserve cross-modality
placeholder ordering. GLM5's wrapper checks complete video spans and maps
selected grids to frame rows; Qwen wrappers require modality placeholders to
match supplied grids. These are source-level admission boundaries, not
verified processor parity or runtime evidence. Timestamp-to-frame embedding
alignment, Inkling audio row-to-placeholder mapping, and caller retry/cleanup
remain open. P04/P14 remain IN PROGRESS and P15 remains OPEN. No tests,
imports, builds, benchmarks, device probes, commits, or pushes were run.

### P04/P14 V4.1 multi-image aggregate staging admission — 2026-10-16

`PublisherImageProcessor.expand()` now passes the remaining aggregate byte
budget into each image preparation. Previously every `prepare()` call could
individually consume the processor's full limit, and the combined prepared
patch/type arrays were rejected only after the additional allocation had
already occurred. Per-image encoded/decoded and resize workspace admission now
uses the remaining budget while earlier prepared spans are still owned. This
closes one multi-image peak-memory admission gap in the V4.1 image caller; it
does not prove processor geometry parity, complete cross-caller token ordering,
or service recovery/cleanup closure. P04/P14 remain partial and P15 remains
OPEN. No tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P04/P14 Qwen4 and Qwen3.5 media-position cancellation — 2026-10-16

Qwen4's shared expanded-position planner, also used by Qwen3.5, now polls the
request cancellation callback between text/media spans and after constructing
the media-span meshgrid, then checks once more before dispatching to the text
decoder. Previously a large but admitted media span could keep constructing
position arrays after request cancellation, even though the vision tower and
text decoder themselves were cancellation-aware. This is cooperative
boundary-level cancellation only; NumPy meshgrid itself cannot be interrupted,
and full geometry/parity and all-caller recovery remain unverified. P04/P14
remain partial and P15 remains OPEN. No tests, imports, builds, benchmarks,
device probes, commits, or pushes were run.

### P06/P12 distributed-bank and recovery audit — 2026-10-16

Re-read the distributed control owner and each concrete projection-bank
implementation rather than treating the V4.1 coordinator as the only bank:
`TensorParallelProjections` owns dense local shards and row-gather/column-sum
collectives; `BF16TensorParallelProjections` owns publisher BF16 streamed rows;
`PackedFP8TensorParallelProjections` owns leased tiled FP8 rows;
`PackedMXFP4TensorParallelProjections` specializes the packed bank for FP4;
and `ProjectionDispatch` owns the format-to-bank mapping. The common projection
path agrees shard geometry, checkpoint/provider/kernel identity, arithmetic
policy, replicated input digest, local staging failures, collective failures,
row assembly, finite-result admission, and host export. Registration pins the
per-rank shard/provider fingerprints before publication. The V4.1 distributed
session separately agrees resource construction and publishes dispatch only
after bank registration. Distributed execution snapshots before destructive
reset, cache load, restore and generation/resume; on surviving-rank failure it
attempts physical rollback, retires temporary snapshots, reconciles retained
generation-boundary metadata, closes unusable sessions, and poisons the
coordinator. Rank cache/generation files and manifests are byte-receipt pinned
and re-admitted around restore.

This audit found no additional safe generic collective patch: local bank
`close()` must remain usable by rank-local setup rollback, so making it an
unconditional collective would deadlock precisely when construction succeeded
on only a subset of ranks. Coordinated service shutdown already retries and
agrees cleanup on surviving ranks. Remaining P06/P12 closure is therefore
explicit: direct-bank callers must prove identical operation/cleanup schedules;
family consumers outside the V4.1 distributed session need an exact bank and
physical-state owner map; per-family restore/reentry/cancellation paths still
need caller-specific review; and actual sharded arithmetic/cache ownership
requires row-bound execution evidence. A lost process or broken transport can
prevent every surviving rank from reaching consensus and remains a documented
group-restart condition, not transparent recovery. No change in this audit
proves general rollback of opaque executor/native owner state. P06/P12 remain
IN PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P09 E8 contract writer failure preservation — 2026-10-16

`LatticeContract.write` now preserves the original staging/write/fsync/replace
exception if removing its temporary artifact also fails, attaching cleanup
failure context instead of masking the publication error. This aligns the E8
contract artifact writer with the already-reviewed IQ3 grid writer behavior.
It does not add E8 activation rotation or device execution; Mixtral, Qwen3.5,
and Qwen4 selected-device E8 paths remain explicitly rejected. Other format
converter lifecycle seams and family/backend loader-to-consumer mappings remain
open, and no actual device placement or execution evidence is claimed. P09/P10
remain IN PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks,
device probes, commits, or pushes were run.

### P05 V4.1 target/proposer RNG alias guard and caller map — 2026-10-16

`SpeculativeDecoder` now rejects a caller-supplied target RNG that is either
the same Generator as the proposer's RNG or wraps the same BitGenerator. Such
aliasing would couple target sampling to proposal count and make rollback
states represent one stream twice. The V4.1 fresh CLI/session and
`PublisherSession.resume_generation` remain the only integrated DSpark
serving callers found in the Python source; both seed and restore distinct
target/proposal PCG64 streams. No gateway route admits DSpark, and other
families remain outside this decoder path rather than being counted as
verified speculative callers. The V4.1 caller rejects stop-prefix streaming
and separate streaming-plus-checkpoint sinks where publication cannot be
atomic, and resets/quarantines target and draft owners after callback,
checkpoint, cancellation, and restore failures.

This closes the direct-API RNG alias case and records the current bounded
caller map; it is not evidence from execution and does not prove opaque custom
owner snapshots restore physical state. Arbitrary external consumers of the
public decoder can still publish events non-atomically; caller-level rollback
cannot retract output already accepted by such sinks. P05 remains IN PROGRESS
until source failure-injection or equivalent execution evidence covers every
accepted/rejected/bonus/EOS/context/cancellation/resume boundary and durable
sink behavior. P15 remains OPEN. No tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P04/P07 K3 timestamp contract shared by video execution and recovery — 2026-10-16

K3 decoded-video admission and generation-boundary restore now call the same
`validate_frame_timestamps` implementation. Restore also reconstructs JSON
arrays as the exact tuple-shaped in-memory media records before admission;
previously every serialized media provenance record failed tuple-type checks,
so mixed-media generation boundaries could not be resumed. The shared check
prevents drift on finite/nonnegative, strictly-increasing and uniform-interval
rules (K3 temporal encoding remains frame-index-based, not elapsed-time RoPE).
This is a source-level contract fix only: it does not prove timestamp-to-pixel
provenance, close cross-family media/serving paths, or provide execution
evidence. P04/P07 remain IN PROGRESS and P15 remains OPEN. No tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

### P04 GLM5 token-ID media dispatch admission — 2026-10-16

`Glm5MultimodalDecoder.forward()` now validates the configured image
placeholder and video span delimiters as distinct IDs within the text
vocabulary before embedding lookup or media dispatch. Previously equal or
out-of-range IDs could make the same input token both open/close a video span
and denote an image placeholder, silently corrupting modality-to-placeholder
selection. This closes that local GLM5 dispatch ambiguity only; it does not
prove video-token mapping, processor ordering, temporal geometry, full media
parity, or all serving caller behavior. P04/P14 remain partial and P15 remains
OPEN. No tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P02 service and gateway entry preflight — 2026-10-16

Added a metadata-only exact tuple preflight for directory-backed service
factories: conditional publisher, flat native, GLM5 publisher, GLM-DSA native,
and Mixtral/Qwen3.5/Qwen4/GLM5/Inkling/GLM-DSA device services. The native
family and conditional CLI sessions now apply the same gate immediately
after model identity parsing, before tokenizer/media/runtime artifact opening.
The local publisher gateway residency key also gates source inventory before
artifact stamping or warm-session reuse, preventing a cached session from
bypassing current tuple admission. Raw in-memory Mixtral native/device, Qwen4
native/device, Qwen3.5 device, and Inkling device factories now reject because
materialized arrays do not preserve source-dtype/persisted-format provenance.
All these current rows remain candidate/unverified and therefore fail closed;
no tuple is promoted. K3/V4.1 retain separate explicit gates. Remaining P02
work is exhaustive caller/factory/API discovery and actual verified evidence;
this pass does not close recovery/tooling paths or P15. P02/P15 remain OPEN.
No tests, imports, builds, benchmarks, device probes, commits, or pushes were
run.

### P05 V4.1 streaming stop-prefix restriction — 2026-10-16

The V4.1 CLI emits token callbacks without a token-boundary withholding filter.
A multi-token stop prefix could span callback events, so earlier prefix tokens
could already be visible before a later token or speculative transaction
completed the stop. Fresh V4.1 streaming now rejects stop-prefix requests
before prefill, and resumed V4.1 streaming rejects the saved stop-prefix policy
before cache restore/publication, for both plain and DSpark routes. Non-streamed
V4.1 generation remains admitted. This is an explicit caller restriction, not
a general streaming implementation; external sink atomicity, all-caller stop
parity, and recovery behavior outside the V4.1 publisher remain open. P05/P14
remain IN PROGRESS and P15 remains OPEN. No
tests, imports, builds, benchmarks, device probes, commits, or pushes were
run.

### P11 Qwen4 cancellation callback status-width normalization — 2026-10-16

`NativeQwen4PublicAdapters.final_hidden()` now requires an optional
cancellation observer to be callable and converts its result to a strict 0/1
before returning through the native `c_int` callback ABI. Previously it passed
the observer's arbitrary integer conversion through a `c_int` return boundary,
so out-of-range values depended on ctypes conversion behavior. Callback
exceptions still become cancellation and are surfaced after native return.
This closes one Qwen4 cancellation status-width/API-shape gap only; other
family callback status and ownership contracts remain open. P11 remains IN
PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P02 Qwen4 packed-provider direct factory restriction — 2026-10-16

The Qwen4 in-memory device factory and directory-backed device service now
reject `TorchQwen4Experts` before checkpoint reads or lazy executor creation.
The generic packed manifest does not bind source dtype and the complete P02
tuple, while the native-family CLI already rejects that artifact class. Direct
factory callers could previously pass it to the decoder's shape/config checks
without the CLI provenance restriction. Other projection providers are not
conflated with this manifest route. This closes only the Qwen4 packed-manifest
factory seam; other family/backend/format tuples and full caller chains remain
unverified. P02/P15 remain OPEN. No tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P14 ModelGateway cancellation does not trigger provider fallback — 2026-10-16

`ModelGateway.stream_complete()` now re-raises `InterruptedError`, native
`ExecutionCancelled`, and internal model-stream cancellation before its generic
provider-failure fallback branch. Previously cancellation from a native stream
could be classified as provider unavailability and transparently replayed
against another provider, continuing work after the caller had cancelled.
Ordinary provider failures retain configured fallback behavior. This closes
the shared streaming fallback/cancellation classification seam only; it does
not provide a cancellation argument to every provider or guarantee device-side
interruption. P14 remains IN PROGRESS and P15 remains OPEN. No tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

### P14 OpenAI-compatible upstream stream terminal enforcement — 2026-10-16

The shared `stream_sse_json` parser now ignores SSE comment/control lines,
accepts both `data: [DONE]` and `data:[DONE]`, and raises
`ProviderUnavailableError` if the upstream closes without `[DONE]`. Its
existing bare-JSONL compatibility remains available only when a parsed object
contains `done: true`. This covers the OpenAI-compatible and llama.cpp stream
clients that share the parser: truncated transport EOF is no longer presented
as normal completion. It does not implement OpenAI/Anthropic downstream SSE
contracts, multiline SSE data folding, or provider cancellation on remote
disconnect beyond the existing iterator-close path. P14 remains IN PROGRESS
and P15 remains OPEN. No tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P02/P03 V4.1 execution payload loader provenance — 2026-10-16

`V41ExecutionPayload` is publicly constructible, so type-checking it did not
prove that its tensors came through `load_v41`'s exact checkpoint inventory
admission. The loader now stamps materialized payloads with a private source-
admission identity, and `PublisherV41Execution.from_payload` rejects payloads
without that loader provenance before selecting execution owners. This closes
the direct in-memory payload-construction bypass only; source-weight identity
and exact row closure for every factory/CLI/gateway/recovery/tooling route still
require reconciliation. P02/P03 remain IN PROGRESS and P15 remains OPEN. No
tests, imports, builds, benchmarks, device probes, commits, or pushes were run.

### P14 isolated accelerator stream callback failure cleanup — 2026-10-16

`DirectHardwareClient.request` now retires the active worker if a streaming
event callback raises. The worker may still be generating and emitting JSONL
events; previously the serialized request lock could be released while that
old command continued, leaving stale events in the shared response queue and
causing a later request to hit a request-ID ordering failure. The callback's
original exception remains primary, and worker termination cleanup failures
are attached as diagnostic notes. This fail-closed cleanup prevents reuse of
that worker after a consumer callback failure; it does not add in-process
generation cancellation, avoid process restart on callback failure, or close
other device/session recovery paths. P14 remains IN PROGRESS and P15 remains
OPEN. No tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P14 gateway streaming does not fall back after caller-visible output — 2026-10-16

`ModelGateway.stream_complete` now permits provider fallback only when the
primary provider fails before yielding any event to its caller. Previously a
provider could emit text and then fail, after which the gateway would start a
fallback stream; the already-delivered prefix could not be retracted, so one
response could combine partial output from two different providers. After
the first visible event, failure now propagates and the active upstream
iterator is closed by the existing stream ownership wrapper. This bounds the
fix to text gateway streams; vision streams, native session recovery,
provider-specific cancellation behavior, and the overall caller matrix remain
separate P14 work. P14 remains IN PROGRESS and P15 remains OPEN. No tests,
imports, builds, benchmarks, device probes, commits, or pushes were run.

### P14 compatibility response and READY/mux contract audit — 2026-10-16

Audited the implemented Brio/Jev and synchronous OpenAI/Anthropic handlers
against their emitted response and error shapes. Brio/Jev scoring maintains
bounded input schemas, explicit normalization, token usage, and endpoint-local
validation/unavailable errors. The compatibility handlers expose only text
responses, reject non-empty text stop strings until tokenizer-aware stop
adaptation exists, and map native `length` versus EOS/stop completion outcomes
to their respective OpenAI and Anthropic terminal reasons. Anthropic
`x-api-key` remains accepted only on `/v1/messages` through OpenAgent's local
auth/scope path; this is not Colibri credential parity. Clarified the
capability record that internal worker JSONL and HTTP routes are not the
Colibri READY-gated mux/legacy stdin protocol and do not expose its CAPS,
STAT, HWINFO, TIERS, EMAP, or byte-counted framing. No further source change
was justified by this bounded inspection. Full Brio/Jev compatibility,
Anthropic request/error/cancellation parity, OpenAI streaming and stop
semantics, and the READY/mux protocol remain open or explicitly restricted;
P14 remains IN PROGRESS and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P14 Brio text-message shape rejects ignored fields — 2026-10-16

`_brio_messages_to_state()` now rejects message keys beyond `role` and
`content`, and content blocks must have exactly `type="text"` plus a string
`text` value. Previously fields such as names/tool-call metadata and blocks
with unrelated types or extra multimodal payload could be silently discarded
while scoring a different textual state. The capability report now states the
text-only shape. This does not provide Brio response/normalization parity,
persistent cache history, media scoring, or client coverage. P14 remains IN
PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P06 distributed shard registration publication agreement — 2026-10-16

After shard fingerprint gathering, `DistributedProjectionBank.register()` now
publishes the local `ProjectionShard`, pin record and resident-byte accounting
inside a second local-stage agreement. If a rank cannot publish its local
registry entry, all surviving ranks learn of the failure; each removes any
partial entry, corrects its resident-byte count and quarantines the bank
instead of leaving successful peers with a registered shard. This complements
the earlier pre-fingerprint staging agreement but does not make process loss
or transport collectives recoverable; those still require group restart. Other
bank/consumer ownership remains open. P06 remains IN PROGRESS and P15 remains
OPEN. No tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P02 remaining reader backend attribution — 2026-10-16

The selected checkpoint admission audit corrected the remaining concrete
backend labels: Native Mixtral and GLM-DSA full/streamed loaders now request
`native`; the six Torch directory readers request `device`; and native
selected range/edge readers retain the `public-range` default. Publisher
conditional admission now checks the complete original checkpoint header
against wrapper identity plus the selected `native`, `device`, or
`public-range` backend before tensor payload materialization. The GLM5
publisher text inventory gate now receives its actual backend instead of
assuming native execution. These rows remain candidate/unverified and are
therefore rejected by exact admission; no route is promoted to verified.
This does not complete factory/CLI/gateway/recovery/tooling chain review or
runtime evidence. P02/P15 remain OPEN. No tests, imports, builds, benchmarks,
device probes, commits, or pushes were run.

### P14 compatibility routes reject malformed/unsupported tool controls — 2026-10-16

The OpenAI and Anthropic compatibility handlers now require any non-null
`tools` value to be an array; non-empty arrays remain explicitly unsupported.
Non-null `tool_choice` is rejected even when it is an empty/falsey object that
previous truthiness checks silently ignored. Anthropic non-null `thinking` is
likewise rejected even when falsey, and `stream_options` is no longer admitted
on `/v1/messages` (it was an OpenAI-specific no-op when empty). Empty OpenAI or
Anthropic tool arrays remain accepted because they define no tools. Capability
metadata now states those precise boundaries. Full OpenAI/Anthropic schemas,
streaming, tool protocols, stop adaptation and auth semantics remain open. P14
remains IN PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks,
device probes, commits, or pushes were run.

### P14 OpenVINO direct stream terminal and iterator ownership — 2026-10-16

`OpenVINODirectProvider.stream_complete()` now treats the worker's `done`
event as terminal and stops dispatching any trailing event records. It
explicitly closes the worker event iterator on completion, downstream close,
or iteration failure; close errors remain visible after successful streaming
and are attached to a primary stream failure otherwise. This closes one direct
hardware provider iterator-ownership/terminal seam. Worker/device interruption,
llama.cpp lifecycle, stop-token adaptation, recovery, and cross-provider
backpressure parity remain open. P14 remains IN PROGRESS and P15 remains OPEN.
No tests, imports, builds, benchmarks, device probes, commits, or pushes were
run.

### P07/P12/P15 K3 checkpoint staging cleanup — 2026-10-16

Both K3 generation-boundary checkpoint publication and recovery stage the
physical cache in a private temporary file. Their unconditional `unlink` in a
`finally` could replace the primary serialization, cancellation, integrity,
or restore exception if cleanup itself failed, obscuring the failure that
determines whether the caller may continue. Staging removal now preserves the
primary exception and adds cleanup failure as a note; on otherwise successful
completion, a staging cleanup failure remains visible to the caller. This is a
narrow K3 checkpoint error-reporting/ownership fix only. It does not close K3
media-to-logits, FLA/native continuation parity, batch rollback, all serving
callers, distributed recovery, or P13 runtime evidence. P07/P12 remain partial
and P15 remains OPEN. No tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P14 Ollama stream terminal and early-close handling — 2026-10-16

`OllamaProvider.stream_complete()` now stops dispatching records at Ollama's
first `done` record and closes the JSONL response iterator on normal
completion, parsing/provider failure, or downstream generator close. EOF before
that required protocol terminal is now a provider failure rather than a
fabricated successful terminal chunk. Cleanup errors are surfaced after
otherwise-normal completion and attached to an active primary failure. This is
Ollama-specific; llama.cpp, OpenVINO, network-disconnect cancellation,
stop-token adaptation, recovery, and cross-provider lifecycle parity remain
open. P14 remains IN PROGRESS and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P14 llama.cpp stream terminal and early-close handling — 2026-10-16

`LlamaCppProvider.stream_complete()` now stops at the SSE `[DONE]` marker,
closes its parsed response iterator on completion, provider failure, or
downstream close, and reports an upstream EOF without `[DONE]` as an incomplete
stream instead of silent success. Cleanup failure is surfaced after successful
streaming or attached to the active primary exception. This is limited to the
llama.cpp streaming adapter; request-level cancellation propagation,
stop-token semantics, recovery, and other provider lifecycle paths remain
open. P14 remains IN PROGRESS and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P02 whole-model native loader exact-row boundary — 2026-10-16

The shared `read_native_text_checkpoint` complete-header admission callback now
binds actual floating tensor encodings to the exact family/model/native/FP32/
`float32` source-matrix row for resident Mixtral, Qwen3.5, Qwen4, GLM5-Next,
Inkling, and GLM-DSA whole-model loads. Family geometry and memory admission
still run first; exact-row admission then runs before any tensor payload is
materialized. Since these rows remain candidate-unverified, the corresponding
whole-model native routes now fail closed rather than treating a valid family
inventory as full P02 admission. Selected-range/device paths have their own
metadata callbacks and are not all covered by this boundary. P02/P15 remain
OPEN; factory/CLI/gateway/recovery/tooling and non-native tuple review remains
incomplete. No tests, imports, builds, benchmarks, device probes, commits, or
pushes were run.

### P02/P03 resident declared source dtype binding — 2026-10-16

`resident_family_inventory.admit_resident_inventory` now compares optional
`dtype`/`torch_dtype` declarations with every floating tensor's persisted
header encoding before family geometry admission. It rejects conflicting or
unsupported declarations and mixed floating encodings when the config claims a
single source dtype; integer metadata tensors remain governed by their family
schema. Mixtral retains its dedicated equivalent check. This closes a bounded
configuration-to-inventory mismatch for resident-planner families. It does not
trace every CLI/gateway/recovery/ABI/tooling route, establish format consumers,
or provide exact P02 tuple evidence. P02/P03 remain partial and P15 remains
OPEN. No tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P11/P03 OLMoE native Segment run envelope — 2026-10-16

The native OLMoE Segment run callback now rejects nonzero token-count and
reserved fields on its activation wire, and rejects nonfinite activation input
before entering the session execution path. This closes an envelope-admission
gap only; it does not establish end-to-end OLMoE Segment activation-consumer
compatibility, nor family-wide run-record validation, ownership, or caller
wiring. Those must remain separately audited under P03/P11. P11 remains IN
PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P02 Qwen4/GLM directory header-to-tuple admission — 2026-10-16

Added `require_source_inventory_admission`, which binds actual floating
SafeTensors header encodings to exact family/model/backend/arithmetic/persisted
format rows during metadata planning, before payload materialization. Qwen4
and GLM5 device directory readers invoke it from their full-inventory selector;
the Qwen4 optional streamed-projection path checks its snapshot header before
constructing the projection owner. Qwen4 and GLM5 native resident readers and
the GLM-DSA native resident planner bind their native rows at complete-header
admission; GLM-DSA device selection binds the same way. The GLM5 publisher
loader checks ordinary float rows and scaled FP8 rows separately. Current
catalog rows remain candidate-unverified, so these paths fail closed before
weight payload reads rather than inferring admission from geometry. This is
also why the GLM-DSA directory service now rejects a generic packed expert
provider whose manifest has no source-format provenance. This is
not complete family route closure: other families, public ABI/CLI/gateway
composition, recovery/tooling and runtime evidence remain unresolved. P02 and
P15 remain OPEN. No tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P14 native publisher cancellation error identity — 2026-10-16

The native publisher's fresh-generation and checkpoint-resume callers previously
caught `ExecutionCancelled` through its `RuntimeError` base class and converted
it to `ProviderUnavailableError`. Cancellation during session construction was
also reclassified by those callers. They now re-raise the cancellation type
before generic runtime-error cleanup/wrapping, preserving the request's
cancellation outcome and leaving transaction rollback semantics to the session
owner. This is limited to the two native publisher caller paths; it does not
close cancellation behavior for other families/providers, guarantee device-side
interruption, or complete P14 stream/backpressure, stop, recovery, warm-session,
media, and cleanup parity. P14 remains IN PROGRESS and P15 remains OPEN. Only
scoped `git diff --check` was run; no tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P14 native publisher stream cancellation-callback error delivery — 2026-10-16

`NativePublisherProvider._stream_run()` now reports a worker-side exception
without re-invoking the caller's cancellation predicate during error delivery.
Previously, if that predicate raised while the worker was publishing a token,
the worker's error publication called the same predicate again; the exception
could escape the daemon thread and the consumer would observe only a generic
"ended without a completion event" failure. Error delivery remains bounded by
the existing event queue and stops when the consumer closes. This closes one
native publisher stream error-propagation seam, not cross-provider cancellation
or device interruption. P14 remains IN PROGRESS and P15 remains OPEN. No tests,
imports, builds, benchmarks, device probes, commits, or pushes were run.

### P06/P12 distributed reset handle type admission — 2026-10-16

`DistributedExecutionService.reset_session` now requires an exact `int` session
handle before dictionary lookup, matching the other public distributed
operations. Python booleans compare equal to integer keys, so previously
`reset_session(True)` could resolve session handle `1` and trigger a destructive
reset despite not being a valid handle value. This closes that local API
admission defect only; it does not close distributed operation-failure
consensus, rollback under transport/process loss, all-bank/shard coverage, or
cross-family recovery ownership. P06/P12 remain IN PROGRESS and P15 remains
OPEN. No tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P04 Qwen4 media-to-logits caller trace — 2026-10-13

Source inspection traced the Qwen4 composition through
`execution_factories.conditional_execution_factory` ->
`MultimodalSessionExecutor` -> `NativeExecutionService.generate` ->
`Qwen4MultimodalDecoder.forward` -> `Qwen4VisionEncoder.forward` and the
Qwen4 text decoder. The service snapshot brackets media prefill and the
existing exception path restores it before token history commit. Qwen4 checks
expanded placeholder/type agreement, image/video grid pairing, projected
feature count, merged-grid position construction, then supplies the scattered
embeddings and 3-axis positions to text forward. No code change was needed for
this already-wired path. This evidence does not establish every family path,
processor ordering/geometry, public caller cleanup, or borrowed-owner cleanup;
those remain P04 work. No tests, imports, builds, benchmarks, device probes,
commits or pushes were run.

### P04 Qwen4 snapshot publication cleanup — 2026-10-13

`Qwen4MultimodalDecoder.snapshot` previously acquired the inner text snapshot
before allocating/publishing the wrapper receipt. If wrapper receipt creation or
registry insertion failed, the inner snapshot could remain live without a
wrapper handle. It now retires that inner snapshot on publication failure and
records failed retirement for a later close attempt. The change closes this
specific partial-construction leak only; family-wide snapshot publication,
media processor cleanup, borrowed-owner behavior, and serving-caller rollback
remain open. No tests, imports, builds, benchmarks, device probes, commits or
pushes were run.

### P04 Qwen3.5 media-to-logits caller trace — 2026-10-13

Source inspection traced the Qwen3.5 publisher path through
`publisher_conditional_loader` -> `Qwen35MultimodalDecoder` ->
`MultimodalSessionExecutor.forward_media` -> `NativeExecutionService.generate`
-> `Qwen35MultimodalDecoder.forward` (the family subclass of the Qwen4 media
transport) -> `Qwen35VisionEncoder` (family-tagged Qwen4 variable-grid vision
implementation) -> `NativeQwen35TextDecoder` or `TorchQwen35Decoder` -> logits.
The composition pairs processor pixels with image/video grids, checks actual
placeholder token types, validates patch geometry in the vision owner, scatters
projected features in token order, computes merged-grid 3-axis positions and
passes both embeddings and positions into the Qwen3.5 text consumer. The
service's prefill snapshot provides caller rollback; the media session stages
detached arrays with cancellation boundaries. Qwen3.5 single-image grids are
restricted to temporal extent one by its `_positions` override. Audio tensors
are now rejected at the conditional loader's metadata boundary when either
`audio_config` or an audio-tower tensor namespace is present, before creating a
tensor reader or materializing checkpoint payloads. The local publisher media
processor also rejects nonempty Qwen3.5 audio before calling Transformers; the
CLI already rejects `--audio-wav` for non-Inkling families before reading audio.
Thus Qwen3.5 audio and chunked-audio modes are explicitly unavailable, not
silently accepted. `NativeExecutionService.generate` separately rejects any
media-bearing prompt larger than one admitted prefill operation before model
snapshot or media dispatch, so chunked/multi-operation media prompts (including
Qwen3.5 images/video) are unavailable until a span planner preserves expanded
placeholder/position ownership. This trace does not
close independent direct-call ownership/failure cleanup, all processor ordering
variants, Inkling/GLM5/K3/V4.1 callers, or the remaining P04 cross-family audit.
No tests, imports, builds, benchmarks, device probes, commits or pushes were
run.

### P11 public family identity boundary — 2026-10-13

`NativeFamilyPublicTextDecoder.identity` previously exposed a writable proxy to
the `SegmentEdgeExecution` recovery digest. A caller could replace the digest
after construction, weakening the model/cache identity boundary used by
snapshots and restores. The adapter now permits only assigning the already
derived value and rejects overrides; the digest remains bound to the admitted
runtime, checkpoint, family adapter and cache schema. This closes that specific
identity-mutability gap only. The family-by-family public symbol, counted-range,
embedding/head, continuation, callback, reentry, close-order and factory/CLI
review remains open, so P11/P15 remain IN PROGRESS/OPEN. No tests, imports,
builds, benchmarks, device probes, commits or pushes were run.

### P11 Segment/Edge constructor output-handle failure contract — 2026-10-14

The shared Segment/Edge callback guard now validates and zeroes constructor
out-handles before resource admission or callback execution. This applies to
segment open, edge open, and session create, so any failure path cannot leave
the caller's preexisting pointer value appearing to identify a newly published
resource. Successful publication still writes the handle only through the
existing resource ledger. This closes the stale-output-handle failure contract
for those shared constructors only; it does not establish the full family ABI,
range/head/continuation schemas, thread-affinity guarantees, or factory/CLI
wiring. P11 remains IN PROGRESS and P15 OPEN. No tests, imports, builds,
benchmarks, device probes, commits or pushes were run.

### P05 V4.1 resumed-generation callback cancellation boundary — 2026-10-14

`PublisherSession.resume_generation` now checks request cancellation after
restoring and validating the recovered target/draft boundary, immediately
before dispatching the `recovered` callback. This closes a narrow race where a
caller disconnecting during cache restore/validation could still receive a
post-cancellation recovery event. It does not change the committed restored
state or make callbacks atomic with asynchronous cancellation; exceptions still
flow through the existing request rollback/reset path. Other serving callers,
speculative accept/reject/bonus/EOS/context equivalence, and durable sink
atomicity remain open. P05 remains IN PROGRESS and P15 OPEN. No tests, imports,
builds, benchmarks, device probes, commits or pushes were run.

### P11 Mixtral public engine construction rollback — 2026-10-13

`NativeMixtralPublicAdapters` now closes the newly opened native range template
if capability-record preparation fails before the engine is handed to the
public resource ledger. Once publication begins, `_publish` remains the single
owner of failure cleanup and retains a failed close for retry. This closes the
pre-publication construction leak for the Mixtral Segment engine callback;
the family-wide ABI map, range/head schemas, callback reentry/thread rules,
continuation coverage, and factory/CLI/service wiring remain open. No tests,
imports, builds, benchmarks, device probes, commits or pushes were run.

### P07 K3 temporal provenance at continuation boundary — 2026-10-14

`GenerationBoundary.admit` now requires each recovered K3 media provenance
record to have the exact `timestamps`/`selection_identity` fields, a bounded
identity string, and (for video) one-to-four finite, increasing timestamps
with uniform intervals consistent with the existing frame-index temporal
encoding. The same admission runs before checkpoint publication and after
checkpoint decode, so malformed or nonuniform temporal provenance cannot be
written and later surfaced as a valid restored boundary. This validates
metadata shape/order only: timestamps are still provenance rather than
physical-time RoPE inputs, and it does not close media-to-serving composition,
all temporal semantics, or native/FLA caller coverage. P07 remains IN PROGRESS
and P15 OPEN. No tests, imports, builds, benchmarks, device probes, commits or
pushes were run.

### P09/P10 selected-device Q8_1 metadata preflight — 2026-10-14

The explicit selected-device Q8_1 decode path now validates exact payload
extent and scans every block's binary16 delta and sum before calling the
full-projection decoder. This prevents malformed metadata in a later matrix
from failing after earlier decoded projection arrays have already been
allocated. This closes only the selected-device Q8_1 metadata admission
boundary; other family/backend loader-consumer chains, placement/no-fallback
evidence, and format-wide parity remain open. P09/P10 remain IN PROGRESS and
P15 OPEN. No tests, imports, builds, benchmarks, device probes, commits or
pushes were run.

### P03 Inkling MTP schedule inventory strictness — 2026-10-14

`inkling_mtp.admit_mtp_inventory` no longer treats false-y malformed
`mtp_local_layer_ids` values (for example integer `0` or an empty string) as
the valid empty sliding-window schedule. The schedule remains checked against
the declared MTP depth before component loading; malformed config now fails
closed instead of silently selecting a different stage inventory/attention
policy. This is a narrow pre-payload config-admission fix only: it does not
close Inkling shared-weight payload equality, every family optional/MTP bank,
or P03 producer-to-execution closure. P03 remains DONE only for bounded
inventory contracts; family closure and P15 remain open. No tests, imports,
builds, benchmarks, device probes, commits or pushes were run.

### P07 K3 timestamp numeric-boundary validation — 2026-10-15

`K3VideoInput` timestamp validation now converts accepted Python integers and
floats to finite binary64 values before ordering/uniform-interval arithmetic.
Extremely large Python integers therefore fail as ordinary invalid timestamp
metadata instead of leaking `OverflowError` during admission. This closes only
the timestamp numeric-boundary edge; frame-index temporal RoPE still does not
consume physical elapsed time, and video cadence semantics, serving callers,
native/FLA continuation and full P07/P15 remain open. No tests, imports, builds,
benchmarks, device probes, commits or pushes were run.

### P07 K3 raw temporal-grid admission restriction — 2026-10-15

`K3Execution.forward` now rejects processor-supplied pixel/grid inputs whose
temporal extent exceeds one unless they came through the explicit
`media_items` route. This prevents raw temporal grids from bypassing the
selected-frame identity and timestamp checks; the pinned processor contract
documents still images, while the typed `K3VideoInput` path is the only route
that currently carries explicit video selection/timing provenance. This is an
admission restriction, not elapsed-time temporal RoPE support or end-to-end
video parity. P04/P07/P15 remain open. No tests, imports, builds, benchmarks,
device probes, commits or pushes were run.

### P04 K3 decoded-media preflight before detach — 2026-10-14

`k3_execution.video_input.prepare_media` now walks the bounded still/video
inventory and checks decoded-frame byte admission and cancellation before making
detached copies of caller-owned buffers. It checks cancellation again between
frame copies, preventing an over-budget request from first duplicating the full
media input. The later preprocessing budget remains in force. This closes only
the K3 preprocessing allocation/order boundary; processor equivalence, all
caller positions/geometry, media-to-logits, and broader cancellation/recovery
remain open. P04/P07/P15 remain open. No tests, imports, builds, benchmarks,
device probes, commits or pushes were run.

### P13 media-row adapter obligation gate — 2026-10-14

The whole-model runner now rejects an exact execution-row binding for a media
case unless its source-catalog obligation explicitly names a media adapter and
comparison report schema and does not mark the media adapter missing. This
prevents a generic text adapter binding from being presented as a structurally
complete media comparison plan. It is only a catalog/manifest gate: it does not
execute the adapter, establish correct ordering or geometry, generate receipts,
prove placement, or certify numerical parity. P13 remains IN PROGRESS and P15
OPEN. No tests, imports, builds, benchmarks, device probes, commits or pushes
were run.

### P03 unsupported audio component admission — 2026-10-14

The Qwen4 conditional publisher loader now rejects `audio_config` and known
audio tensor namespaces during checkpoint-header inspection, before creating a
payload reader. Its wrapper previously rejected audio tensors only after
loading; it has no audio execution owner. The shared GLM5 publisher inventory
planner now rejects the same declared or present audio component before any
payload materialization, and the GLM5 multimodal owner no longer classifies
MTP namespaces as retained auxiliary tensors (its inventory planner already
rejects them because no GLM5 draft consumer exists). These are explicit family
restrictions, not new audio/MTP implementations. P03 remains DONE only for the
bounded inventory contract, with producer-to-consumer closure open; P02 and
P15 remain IN PROGRESS/OPEN. No tests, imports, builds, benchmarks, device
probes, commits or pushes were run.

### P05 speculative receipt identity hardening — 2026-10-14

`SpeculativeDecoder` now records each exact `SpeculativeSnapshot` object it
issued and accepts restore/release only for that live object. Previously the
snapshot exposed its decoder owner marker, so a caller could wrap substituted
target/draft physical receipts in a new dataclass instance while copying the
marker from a legitimate snapshot. The live issued-receipt inventory is
bounded to 64; a release attempt retires the capability even if physical
release fails, while failed resource owners remain in retryable cleanup state.
This closes that receipt-wrapper substitution path only. Cross-serving-caller
accept/reject/bonus/EOS/context/cancel equivalence, arbitrary sink atomicity,
and the rest of P05 remain open. P05 remains IN PROGRESS and P15 OPEN. No tests,
imports, builds, benchmarks, device probes, commits or pushes were run.

### P04/P07 K3 nested selected-video frame cleanup — 2026-10-14

`prepare_media` now records each copied PIL frame nested in a `K3VideoInput`
as soon as the detached copy is made. Its existing all-exits cleanup therefore
closes those temporary frame owners after packing and when cancellation or a
later copy fails, without closing the caller-owned source frames. This is a
bounded staging-resource fix; temporal semantics, other family media callers,
and end-to-end serving lifecycle reconciliation remain open. P04/P07/P14 stay
IN PROGRESS and P15 OPEN. No tests, imports, builds, benchmarks, device probes,
commits or pushes were run.

### P02 direct native Qwen4 factory architecture identity — 2026-10-14

`create_qwen4_execution_service` now validates an optional `architectures`
declaration before constructing the native decoder or admitting its
caller-supplied tensor mapping. It accepts only the admitted
`Qwen4ExpForCausalLM` identity (or no declaration for legacy direct configs),
matching the independent Qwen4 config gate. This closes only the direct native
factory identity bypass; it does not establish exact backend/format tuple
admission or complete loader-to-tooling closure. P02 remains IN PROGRESS and
P15 OPEN. No tests, imports, builds, benchmarks, device probes, commits or
pushes were run.

### P06 tensor-parallel interruption propagation — 2026-10-14

`TensorParallelProjections.local_stage` already synchronized rank-local
`BaseException` failures, but only tagged `InterruptedError` for collective
classification. A unanimous `KeyboardInterrupt` was therefore downgraded to
`ValueError` after every rank reached agreement. The stage now tags this
interrupt and `_agree` re-raises `KeyboardInterrupt` collectively, preserving
the caller's cancellation/interrupt cleanup path. Other rank-local failures
retain the stable `ValueError` envelope. This applies only to synchronized
local stages; it cannot recover a transport failure, lost process, or peer
that never reaches the agreement collective. P06 remains IN PROGRESS and P15
OPEN. No tests, imports, builds, benchmarks, device probes, commits or pushes
were run.

### P11 Segment/Edge tokenizer output-count failure contract — 2026-10-14

The Segment/Edge tokenizer and detokenizer callbacks now require and clear
their output count/extent before adapter-handle lookup, admission, text decode,
or tokenizer work. A failed call can no longer leave a caller's unrelated
stale extent appearing valid. When output capacity is too small, the callback
still publishes the required extent so callers can use the established
size-query/retry pattern. This is limited to these two Edge callbacks; it does
not establish equivalent behavior for every family ABI out-parameter or close
the wider P11 schema, callback, and factory/CLI map. P11 remains IN PROGRESS and
P15 OPEN. No tests, imports, builds, benchmarks, device probes, commits or pushes
were run.

### P02/P09 packed manifest consumer admission before expert-store open — 2026-10-14

`SparseExpertRuntime` now resolves each admitted manifest layer's exact
persisted dtype against the selected kernel registry and fallback policy
before constructing `DiskExpertStore`. Previously the early manifest pass
checked lattice contract metadata only; a well-formed non-lattice format with
no compatible selected consumer could cause expert files to be opened before
the later routed-expert provider rejected it. The native family CLI passes its
selected kernel registry and `allow_kernel_fallback=False` into this runtime,
so the CLI's admitted file-format row now fails before expert-store opening if
the exact kernel cannot consume it. This does not establish format/packing
parity for formats that do resolve, add compatibility across other factories,
or close the wider family/backend/format chain and evidence matrix. P02/P09
remain IN PROGRESS and P15 OPEN. No tests, imports, builds, benchmarks, device
probes, commits or pushes were run.

### P02/P03 publisher MTP consumer closure restriction — 2026-10-14

Header-only admission in `load_publisher_conditional_decoder()` now rejects
MTP tensor namespaces for Qwen3.5, Qwen4 and GLM5 before any model tensor
payload is read. Those paths previously accepted opaque MTP arrays as
auxiliary ownership without an admitted draft executor. Direct
`partition_weights()` wrapper construction now enforces the same rule for
Qwen3.5/Qwen4 callers that bypass directory loading, while retaining Inkling's
budgeted `InklingMTP` route. The independent GLM5 inventory planner applies the
restriction too, covering its direct multimodal construction path. This is a
concrete exclusion, not implementation of Qwen3.5/Qwen4/GLM5 draft decoding;
it does not close remaining family/backend/format tracing or all factory
bypasses. P02 remains IN PROGRESS, P03 remains bounded-DONE, and P15 remains
OPEN. No tests, imports, builds, benchmarks, device probes, commits or pushes
were run.

### P06 distributed model identity agreement at session admission — 2026-10-14

Before a distributed session handle is published, each rank now reports its
executor's bounded stable `identity` and implementation type through the
existing result-agreement record. An absent/malformed identity or any
cross-rank identity/implementation mismatch fails the open transaction and
uses the existing best-effort rank-local session cleanup before poisoning the
coordinator. This prevents a mixed checkpoint/model implementation from
reaching generation, snapshot, or cache collectives merely because command
schedules match. The identity contract does not prove tensor-shard layout or
provider configuration is encoded in every model identity, and it cannot
recover a rank lost during the collective; those P06/P12 gaps remain open.
P06/P12 remain IN PROGRESS and P15 OPEN. No tests, imports, builds,
benchmarks, device probes, commits or pushes were run.

### P04 Inkling waveform cancellation staging — 2026-10-14

`PublisherMediaProcessor.prepare()` now polls cancellation between 262,144-sample
chunks while converting and checking each admitted Inkling waveform. The prior
single full-array conversion and finite scan could delay cancellation across a
large waveform before reaching the processor boundary. This only closes
publisher waveform staging responsiveness; NumPy allocation, source decoding,
and the downstream processor call remain non-interruptible operations, and
other audio/media callers remain open. P04/P14 remain IN PROGRESS and P15 OPEN.
No tests, imports, builds, benchmarks, device probes, commits or pushes were
run.

### P02 GLM-DSA architecture declaration restriction — 2026-10-14

`IndependentGlmDsaDecoder` previously admitted `model_type="glm_moe_dsa"`
while ignoring any declared Transformers architecture class. Because this
route has no pinned architecture-class mapping, it now rejects every present
`architectures` field before tensor inventory/execution; architecture-less
configs still require the exact model type and existing geometry/schedule
contracts. This is an explicit restriction, not proof that the family route is
fully traced. P02 remains IN PROGRESS; factory/CLI/gateway/tooling tuple
closure and generated candidates remain open. P15 remains OPEN. No tests,
imports, builds, benchmarks, device probes, commits or pushes were run.

### P09 Q5 block-delta admission — 2026-10-14

Q5_0/Q5_1 scalar execution, optional native kernels, and full-weight decode
now reject negative block deltas in addition to non-finite metadata. Full
decoders validate metadata across the complete payload before allocating
projection matrices. The packers emit non-negative deltas, so accepting a
negative delta was inconsistent with the persisted source contract. This
closes only this malformed-metadata seam for Q5_0/Q5_1 consumers; it does not
establish all-family loader/consumer parity, other format geometry, or runtime
placement/evidence. P09 remains IN PROGRESS and P15 OPEN. No tests, imports,
builds, benchmarks, device probes, commits or pushes were run.

### P13 evidence-obligation shape validation — 2026-10-14

The source catalog validator now rejects evidence rows whose artifact-pin
classes are missing, duplicated, malformed, or not exactly config,
weights-or-shards, tokenizer, and runtime-dependencies; it also requires the
explicit per-operation placement and complete-unprojected physical-cache
obligations. This catches weakened or malformed structural declarations
before audit consumers use them. It does not generate or verify any artifact
hash, capture placement/cache observations, validate adapter behavior, or
create runtime evidence. P13 remains IN PROGRESS and P15 OPEN. No tests,
imports, builds, benchmarks, device probes, commits or pushes were run.

### P02 E8 device family/format admission — 2026-10-14

`native_family_cli._DeviceOwner` now rejects `e8-lattice-caller-v1` expert
manifests for Mixtral, Qwen3.5 and Qwen4 before constructing `DiskExpertStore`.
Those device families lack the matching activation rotation, so this exact
device-family/format route is concretely restricted at admission rather than
waiting for a later provider failure. GLM-DSA E8 remains separately admitted
by its own provider contract. This closes only that restriction point: the
broader P02 exact tuple chains, truthful tuple propagation, and all generated
candidate classifications remain open. No tests, imports, builds, benchmarks,
device probes, commits or pushes were performed.

### P04 reject orphan processor payload before tower dispatch — 2026-10-14

Qwen4 (and inherited Qwen3.5) plus GLM5 conditional forwards now reject
nonempty image/video processor payloads when the expanded token sequence has no
matching modality placeholders, before calling the vision encoder. Previously
the encoder could allocate and compute a complete feature set only for the
subsequent feature-count check to reject it. This closes one bounded dispatch
ordering/admission case; it does not prove processor ordering parity, complete
media geometry/position handling, borrowed-owner behavior, or caller-level
rollback across families. P04 remains IN PROGRESS and P15 OPEN. No tests,
imports, builds, benchmarks, device probes, commits or pushes were run.

### P04 Inkling orphan media dispatch admission — 2026-10-14

Inkling conditional forward now rejects image/audio payloads without an
expanded-token placeholder before calling the corresponding encoder, matching
the bounded Qwen4/GLM5 dispatch guard. A standalone audio frame mask without
audio IDs is rejected instead of being ignored. This only prevents orphan
payload work and detached-mask ambiguity; it does not establish cross-modal
ordering, geometry/position parity, or caller rollback. P04 remains IN
PROGRESS and P15 OPEN. No tests, imports, builds, benchmarks, device probes,
commits or pushes were run.

### P13 exact execution-row adapter binding — 2026-10-14

Whole-model validation configuration can now carry a strict execution contract
covering family, model type, backend, arithmetic, source format, persisted
format, and media applicability, bound to the exact oracle and candidate
`module:callable` factories. The runner rejects a bound case unless both
loaded adapters' metadata echoes the identical contract and records the
binding with receipt/execution/certification false. This is an opt-in
structural guard: generic and historical manifests may remain unbound, adapter
metadata remains an assertion, and no runtime observations or placement
receipts are created. Wiring and reconciling every applicable matrix row,
checking adapter implementations, and capturing evidence remain open; P13 is
IN PROGRESS and P15 OPEN. No tests, imports, builds, benchmarks, device probes,
commits or pushes were run.

### P12 distributed rank-cache manifest re-admission — 2026-10-14

`DistributedExecutionService._execute` now reopens and validates the committed
distributed cache manifest on each rank immediately before reading that rank's
receipt and again after cache installation/history verification. The parsed
manifest must still equal the exact manifest admitted by the root command. A
replacement or mutation detected after installation raises inside the existing
collective operation; the already-held physical snapshot and generation-boundary
metadata rollback path then restores prior state on each rank. This closes the
manifest replacement window for this shared-filesystem cache restore path only.
It does not close rank process loss/transport failure, other family codecs, or
distributed recovery consensus beyond this operation. P12 remains IN PROGRESS
and P15 OPEN. No tests, imports, builds, benchmarks, device probes, commits or
pushes were run.

### P05/P14 V4.1 stream cancellation before event dispatch — 2026-10-14

Fresh V4.1 CLI generation and checkpoint-resume now recheck the request's
cancellation predicate after checkpoint publication and immediately before
calling the token-event sink. This closes the source-level window where a
slow/reentrant checkpoint sink could allow a token callback after consumer
cancellation had already become observable. The model step/checkpoint may
already be committed when cancellation is detected; the existing request
failure path resets or quarantines the session, while an arbitrary checkpoint
sink still owns atomic publication semantics. A cancellation racing after the
final predicate check cannot be made atomic with arbitrary synchronous callback
code. Caller-wide streaming/backpressure, durable sink, stop, recovery and warm
session semantics remain open; P05/P14 remain IN PROGRESS and P15 OPEN. No
tests, imports, builds, benchmarks, device probes, commits or pushes were run.

### P11 Qwen4 routing-observer reentry guard — 2026-10-14

`NativeQwen4PublicAdapters.final_hidden` now rejects calls made while the
owner's synchronous native routing observer is active. The observer executes
user code from a native callback under a reentrant Python lock; without the
explicit guard, that callback could recursively enter the same native model
through `final_hidden`. This closes only that callback-to-model reentry edge.
The wider family ABI symbol/range/head schema, callback thread/ownership,
registration, continuation, close-order, and factory/CLI/service map remains
incomplete; P11 remains IN PROGRESS and P15 OPEN. No tests, imports, builds,
benchmarks, device probes, commits or pushes were run.

### P02 exact Qwen4 native/F32/float32 tuple trace — 2026-10-13

Traced the relevant source owners: `native_family_cli.PublisherSession`
admits `qwen4_exp_text` and checkpoint shard artifacts; the conditional
publisher loader builds a complete Qwen4 text/media inventory and materializes
source tensors before constructing `Qwen4MultimodalDecoder`; its native text
consumer is `NativeQwen4TextDecoder`. The Qwen4 packed-expert converter,
`Qwen4StreamedExperts`, and native expert callback path are also present.
However, `F32` names a checkpoint source tensor dtype while persisted
`float32` names a packed-expert bank encoding. No exact admission receipt or
callable binding joins the checkpoint's full header inventory to that packed
bank identity for the requested tuple. Therefore
`qwen4/qwen4_exp_text/native/fp32/F32/float32` remains
`candidate-unverified`; do not promote it based on the independent existence
of the loader and expert consumer. This audit adds no broader Qwen4 restriction
and does not close the ABI/factory/gateway/recovery/tooling tuple chain or P02.
No tests, imports, builds, benchmarks, device probes, commits or pushes were
run.

### P11 Qwen4 Segment/Edge close idempotency — 2026-10-13

Static source review of `native_qwen4_public_text.py`,
`qwen4_public_adapters.py`, `native_qwen4_public_adapters.py`,
`segment_edge_adapters.py`, and the Qwen4 conditional wrapper confirms the
current public chain uses batch-one FP32 activation rows of `hc_count *
hidden_size`, PLE token-side rows, a complete ordered contiguous layer
partition, three-axis text positions, and native embedding/final-head
operations. The public text owner serializes stateful calls with a reentrant
lock and explicitly rejects callback reentry; registry destroy callbacks claim
resources before closing and retain failed closes for retry. The Qwen4 engine
close is now idempotent, and its range-session close is idempotent while
preserving retry when model cleanup fails. The existing conditional factory
only wraps native CPU text when a Segment runtime library is supplied, and the
adapter requires both Segment and Edge registration before service creation.

This closes the repeated-close edge in the Qwen4 engine/session lifecycle only.
P11 remains IN PROGRESS: the checklist still lacks a verified complete ABI map
for every family, per-symbol range/head/continuation schemas, callback
thread-affinity guarantees across all adapters, and all factory/CLI/service
construction routes. This static source-only update ran no tests, imports,
builds, benchmarks, device probes, commits or pushes.

### P04 GLM5 conditional media cleanup trace — 2026-10-13

The GLM5 conditional path composes `publisher_glm5_loader` inventory/weight
admission -> `Glm5MultimodalDecoder` -> `Glm5VisionEncoder` grid/patch geometry,
per-frame spatial positions and projected feature grouping -> ordered image or
video placeholder scatter -> GLM5 text `forward(inputs_embeds=...)` logits.
Video grids are expanded into one temporal-one vision grid per frame while the
opening/closing video token span selects video placeholders; incomplete or
nested spans and patch/placeholder count disagreement are rejected. Static
inspection found `Glm5MultimodalDecoder.close` short-circuited after a text
owner close failure, potentially leaving its independent vision native owner
unclosed. Cleanup now attempts the continuation and vision owners even when a
prior close fails, retains failed owners for retry, and only marks the wrapper
closed after all cleanup succeeds. This does not prove processor ordering or
caller rollback for every GLM5/Inkling/K3/V4.1 caller; P04 remains IN PROGRESS.
No tests, imports, builds, benchmarks, device probes, commits or pushes were
run.

P11 follow-up: `NativeQwen4PublicTextDecoder.identity` had the same writable
recovery-digest proxy. Repository caller search found no post-construction
identity rebinding; construction passes the registered owner digest directly
to `SegmentEdgeExecution`. The Qwen4 adapter now permits only the unchanged
derived identity and rejects overrides. This closes only the identity
mutability gap for this adapter; broader P11/P15 review remains open. No tests,
imports, builds, benchmarks, device probes, commits or pushes were run.

### P05 speculative proposer transaction boundary — 2026-10-13

`SpeculativeDecoder` now requires every draft proposer to expose paired
`snapshot()` and `restore()` methods at construction. Previously an arbitrary
callable without recoverable state could mutate its draft state during proposal,
then survive a target rejection, cancellation, or downstream error while the
target and target RNG rolled back. Such a proposer is now rejected before use,
so a speculative transaction cannot claim rollback while omitting the proposal
owner. This is a concrete API restriction for custom/stateless callables as well
as a source-level closure of the missing-proposer-snapshot admission gap. It does
not certify every accept/reject/bonus/EOS/context/cancellation interaction,
distributed collective failure ordering, or serving recovery caller; P05/P06
remain in progress and P15 remains open. No tests, imports, builds, benchmarks,
device probes, commits or pushes were run for this source change.

Follow-up transaction audit found that `step()` installed the caller's new
stop-prefix policy before checking a saved stop boundary and before acquiring
target/draft snapshots. The saved-stop check now occurs before mutation, and if
snapshot acquisition fails, the early-exit path restores the prior stop-prefix
policy before releasing any partially acquired owner state.
This closes that narrow pre-transaction leak only; serving-caller boundary and
accept/reject/bonus/EOS/context/cancellation coverage remain open. No tests,
imports, builds, benchmarks, device probes, commits or pushes were run.

### P05 V4.1 checkpoint/callback cleanup boundary — 2026-10-13

The `PublisherSession` generation and resume callers now preserve the primary
exception when synchronous event callbacks, checkpoint sinks, cancellation, or
generation operations fail and the subsequent request reset also fails. The
secondary reset error is recorded as an exception note; a failed reset marks
the session closed and attempts owned-resource cleanup without masking the
reset error. Resume now applies the same primary-error rule when closing a
staged checkpoint frame after partial restore validation or cache installation.
This closes only exception precedence and request-owner cleanup in the
V4.1 CLI/session path. Atomic file sinks already stage the replacement frame
and preserve the prior committed file on failed publication; arbitrary caller
sinks remain responsible for their own atomicity. P05 remains IN PROGRESS:
caller-wide speculative accept/reject/bonus/EOS/context/cancellation and
repeated-restore state equivalence are not established by this change. No
tests, imports, builds, benchmarks, device probes, commits or pushes were run.

P05 iterator cancellation follow-up: `SpeculativeDecoder.generate()` previously
returned normally when cancellation was observed between yielded `step()`
transactions, making a direct iterator consumer unable to distinguish a
cancelled partial stream from successful early completion. It now raises the
same `ExecutionCancelled` used by `step()` at that boundary. This closes only
the iterator's inter-transaction cancellation signal; it does not verify every
caller’s cancellation propagation, durable checkpoint publication, or all
accept/reject/bonus/EOS/context/repeated-restore equivalence. P05 remains IN
PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks, device
probes, commits or pushes were run.

P05 serving-owner reset follow-up: `PublisherSession._reset_request()` previously
called target reset followed by draft reset in one short-circuiting `try`, so a
target reset exception skipped cleanup of the independent draft history. It now
attempts both resets, retains the first reset failure with subsequent reset
failures attached as notes, and closes/marks the session unusable if either
owner cannot reset. This covers the V4.1 CLI/session request teardown path only;
it does not prove target/draft physical equivalence, arbitrary stream-sink
atomicity, or caller-wide accept/reject/bonus/EOS/context/cancellation behavior.
P05 remains IN PROGRESS and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits or pushes were run.

### P06 distributed reset rank-failure rollback — 2026-10-14

`DistributedExecutionService._execute` now snapshots each rank's session before
reset, retains that receipt until rank result agreement, and restores the
pre-reset physical session plus prior generation-boundary metadata if any rank
reports failure or a result mismatch. Snapshot release and rollback errors are
included in the existing collective cleanup agreement; a failed cleanup still
poisons the distributed coordinator and requires group restart. This closes
only reset's local-success/peer-failure split when the executor's snapshot and
restore contract can recover it. It does not make process loss or a blocked
collective recoverable, nor establish all-bank operation ordering or general
distributed recovery consensus. P06 remains IN PROGRESS and P15 OPEN. No tests,
imports, builds, benchmarks, device probes, commits or pushes were run.

### P04/P07 K3 patch preprocessing resource cleanup — 2026-10-14

`k3_execution.preprocessing.preprocess_images` now closes its intermediate
resized and color-converted PIL images on success, cancellation, and
preprocessing failure, and also closes the temporary PIL source created from a
NumPy frame. Cleanup attempts all owned objects and annotates rather than
replaces an active processing error. Caller-owned PIL inputs remain borrowed
and are not closed. Previously these intermediate image resources were left to
garbage-collection timing on cancellation or an exception. This is a bounded
K3 media-staging cleanup improvement only; it
does not close video temporal semantics, all media callers, native/FLA
continuation integration, or P04/P07/P15. No tests, imports, builds,
benchmarks, device probes, commits or pushes were run.

### P06 V4.1 distributed bank preflight agreement — 2026-10-13

`make_projection_backend` now performs one common control-group agreement before
any BF16/FP8/FP4-specific collective registration. It compares the per-rank
bank flags, checkpoint/config identity, transport/device policy and budgets,
selected V4.1 projection shard inventory, and exact BF16/FP8 provider source
inventories. Device admission and collective/arithmetic device matching also
run inside the guarded preflight; their normalized device identity participates
in rank agreement. Per-rank metadata, inventory or device-admission errors are
collected at this same boundary so peers do not proceed into format-specific
collectives while another rank rejects locally. This closes that constructor
ordering gap only; it does not solve process loss, later operation-level
collective failure, cross-family bank coverage, or recovery consensus. P06
remains IN PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks,
device probes, commits or pushes were run.

P06 V4.1 MTP bank coverage is now explicitly checked against the admitted
checkpoint inventory before format-specific registration. `_shard_rule` is the
source selector for 2-D `v41.mtp.*` matrices; BF16 covers source-BF16 rows and
the publisher's grouped `wo_a` boundary, FP8 covers tiled32 projections other
than `wo_a`, and FP4 covers every admitted `v41-fp4-g32` projection. The new
preflight rejects missing or unexpected MTP sources before ranks enter those
bank collectives (and synchronizes FP4-provider construction/validation
failure). The execution consumer is `PublisherV41Auxiliary.mtp` borrowing the
same arithmetic as the target decoder; its provider's `linear` and grouped
`wo_a` calls dispatch through the same `ProjectionDispatch`. BF16/FP8/FP4 bank
projection operations retain their own rank agreement and cancellation scopes.
MTP KV windows and RNG remain per-rank auxiliary state captured by the MTP
snapshot/cache path, not by projection banks; generation recovery must therefore
restore each rank's MTP frame along with its target/cache frame. This closes
only source-inventory completeness for this V4.1 MTP bank slice; it does not
close operation-level process loss, blocked collectives, cancellation rollback
across the generation caller, or distributed MTP restore consensus. P06 stays
IN PROGRESS. No tests, imports, builds, benchmarks, device probes, commits or
pushes were run.

### P14 OpenAI echo request-surface restriction — 2026-10-13

The text-completion `echo` option is accepted only on `/v1/completions`; a
`/v1/chat/completions` request that supplies it is rejected as unsupported
instead of silently ignoring it. This closes that request-surface ambiguity,
not the OpenAI compatibility contract: streaming, stop adaptation, sampling,
tool use and other unsupported semantics remain explicitly restricted.

### P06 Distributed session-open rollback — 2026-10-13

`DistributedExecutionService._execute` now treats distributed `open` as a
transaction across rank-local execution services. If rank results contain an
error or disagree after the result gather, each rank that successfully created
the requested local session attempts `close_session`, removes the handle only
after that close succeeds, and joins a cleanup-error gather before the
coordinator reports failure. Any cleanup error is included in the distributed
failure and the coordinator remains poisoned for restart. This closes the
specific partial-open session leak when rank-local open succeeds on only a
subset of ranks. It does not prove that a failing `open_session` implementation
cleans its own partially constructed internals, nor does it close process loss,
transport/blocked-collective failure, other operation rollback, bank coverage,
or distributed recovery consensus. P06 remains IN PROGRESS and P15 remains
OPEN. No tests, imports, builds, benchmarks, device probes, commits or pushes
were run.

### P07 K3 batch reset cancellation rollback — 2026-10-14

`k3_execution.batch.K3BatchExecutor.reset` now accepts an optional cancellation
callback, checks it before capture, between slot resets, and before commit, and
uses the pre-reset snapshots to roll all selected slots back if cancellation
arrives. Each individual slot reset remains an atomic operation and is not
interruptible mid-reset. This closes the batch-reset cancellation gap only;
remaining K3 temporal/media semantics and serving/native/FLA continuation
integration remain open. P07 remains IN PROGRESS and P15 OPEN. No tests,
imports, builds, benchmarks, device probes, commits or pushes were run.

### P04 V4.1 image resize failure cleanup — 2026-10-14

`v41_execution.image_input.PublisherImageProcessor.prepare` now starts its
cleanup scope before resize/pad construction and closes both the resized image
when created and the converted intermediate image if preprocessing raises.
Cleanup attempts both owners and attaches cleanup failures without replacing an
active preprocessing failure. Previously a resize/pad exception occurred
before the cleanup `try`, leaking the converted image object. This is a bounded
local resource-lifetime fix only;
it does not establish processor parity, media ordering/position correctness,
borrowed-owner cleanup, serving/recovery closure, or family-wide media-to-logits
coverage. P04 remains IN PROGRESS and P15 OPEN. No tests, imports, builds,
benchmarks, device probes, commits or pushes were run.

### P11 K3/V4.1 public session callback reentry guard — 2026-10-14

`NativePublisherPublicAdapters` routes run, snapshot, restore, and close
through a per-session operation guard. It preserves serialization between
threads while rejecting same-thread callback reentry, which otherwise could
use the session's `RLock` to close or mutate a K3/V4.1 model during an active
native ABI operation. This closes only that K3/V4.1 publisher-session reentry
boundary; the family-wide symbol/range/schema/callback/factory map remains
incomplete. Static source inspection only; no tests, imports, builds,
benchmarks, device probes, commits or pushes were run.

### P12 durable continuation recovery-into ownership — 2026-10-14

`DurableContinuationStore.recover_into` now serializes recovery with the
existing service session and captures an owned rollback snapshot before cache
loading. If the loaded handle or restored history fails catalog validation,
the prior physical cache and token history are restored; the caller-owned
session is no longer closed as if it had been newly created by `recover()`.
This covers only this durable continuation recovery caller; remaining family
physical-state ownership and distributed recovery consensus stay open. P12
remains IN PROGRESS and P15 OPEN. No tests, imports, builds, benchmarks,
device probes, commits or pushes were run.

### P09/P10 V4.1 packed-projection admission before device copies — 2026-10-14

`v41_execution.device_arithmetic._packed_matvec_open` now checks the selected
FP4/FP8 encoding's exact packed extent and scale-grid extent, finite UE8M0
scale codes, finite FP8 weight codes, and finite contiguous FP32 activation
vector before materializing device tensors. Unsupported encodings and malformed
payload geometry are rejected before device allocation instead of after its
copies. This closes only this V4.1 packed projection boundary; it does not
validate every persisted-format route, establish scale/packing parity with
other family consumers, or close device placement/no-fallback receipts and
conversion cleanup elsewhere. P09/P10 remain IN PROGRESS and P15 OPEN. No
tests, imports, builds, benchmarks, device probes, commits or pushes were run.

### P06 V4.1 distributed construction cancellation classification — 2026-10-14

`v41_distributed_session.make_projection_backend`'s `agreed_stage` already
waited for every rank to report local construction outcome and attempted
rank-local cleanup before returning a construction failure. It now re-raises
`InterruptedError` when every rank reports cooperative interruption, instead
of wrapping that unanimous stop in `RuntimeError`. This lets callers retain
cancellation-specific cleanup/rollback semantics after collective agreement.
Mixed-rank failures remain construction errors; transport/process loss can
still strand agreement and requires group restart. This does not complete
distributed bank mapping, operation-level failure behavior for every consumer,
shard-layout coverage, or recovery consensus. P06 remains IN PROGRESS and P15
OPEN. No tests, imports, builds, benchmarks, device probes, commits or pushes
were run.

### P09/P10 native FP8 row-scale parity — 2026-10-14

`native/original_experts.c` now rejects nonpositive FP8 block scales in the
original native projection row-table read path. Previously this path checked
only finiteness, while streamed projection evaluation and
`PublisherTensorReader.read` reject zero and negative factors. This aligns the
row-table consumer's scale admission with the already implemented FP8 source
contract; it does not establish all-family format/backend compatibility,
conversion cleanup, geometry receipts, or device placement/no-fallback
coverage. P09/P10 remain IN PROGRESS and P15 OPEN. No tests, imports, builds,
benchmarks, device probes, commits or pushes were run.

### P07 K3 batch continuation cancellation — 2026-10-14

`K3BatchExecution.restore` accepted no cancellation predicate and called each
executor's public non-cancellable restore in sequence. It now validates and
checks a caller predicate between slots and inside each executor's staged
native/FLA restore, while holding the normal executor operation lease. Any
cancellation follows the existing batch rollback path to the pre-restore
snapshots; rollback itself is deliberately not cancellable. This closes only
the batch snapshot-restore cancellation gap. Batch reset cancellation,
caller-wide continuation integration, temporal/video semantics and full P07
remain open. No tests, imports, builds, benchmarks, device probes, commits or
pushes were run.

### P12 distributed cache-restore metadata rollback — 2026-10-13

When collective `load_cache` fails or rank summaries disagree after local cache
load, the coordinator now restores the prior `_generation_boundaries` entry
alongside the rank-local service snapshot, including removing a newly introduced
boundary when none existed before. Previously this metadata repair was limited
to `restore` and `restore_generation`, allowing stale continuation metadata to
survive a failed cache replacement even after physical session rollback. This
closes only that coordinator metadata/physical rollback mismatch; process loss,
blocked collectives, rollback-failure consensus, other family codecs and full
distributed recovery remain open. P12/P15 remain OPEN. No tests, imports,
builds, benchmarks, device probes, commits or pushes were run.

### P12 service cache final-admission rollback — 2026-10-14

`load_service_cache` already rechecked the stable descriptor/path after service
restore returned, but that post-return check occurred after
`restore_session_cache` had committed the restored token history. A late
replacement/extent failure could therefore report a failed restore while the
session retained the newly installed state. `restore_session_cache` now invokes
the stable reader's optional `require_complete()` inside its rollback-protected
transaction, after exact frame consumption and model-position validation but
before publishing new service history. The outer check remains as defense in
depth. This covers the file-backed service-cache path only; arbitrary readers
without this capability, other family codecs and distributed failure consensus
remain open. P12/P15 remain OPEN. No tests, imports, builds, benchmarks, device
probes, commits or pushes were run.

### P07 K3 CLI recovery cancellation propagation — 2026-10-13

`PublisherSession.resume_generation` already wrapped checkpoint reads with
cancellation checks, but did not pass the same predicate into
`restore_generation_boundary`; cancellation could therefore be delayed through
its bounded cache staging and locked physical restore section. The CLI now
passes that predicate into the codec, which checks cancellation during staging,
before the restore lock, and around cache installation. This closes cancellation
propagation for this K3 resume caller only; other direct codecs and family
serving callers remain to be reconciled. P07/P12 remain IN PROGRESS and P15
OPEN. No tests, imports, builds, benchmarks, device probes, commits or pushes
were run.

### P02 source-combination classification correction — 2026-10-13

P12 distributed cache-restore metadata rollback follow-up: when a rank-collective
`load_cache` operation fails or rank summaries disagree after local cache load,
the coordinator now restores the prior `_generation_boundaries` entry together
with the rank-local service snapshot, including removing a newly introduced
boundary when none existed before. Previously this metadata repair was limited
to `restore` and `restore_generation`, so a failed cache replacement could leave
stale continuation metadata beside rolled-back physical state. The fix closes
that coordinator metadata/physical rollback mismatch only; process loss,
blocked collectives, rollback failure consensus, other family codecs and full
distributed recovery remain open. P12/P15 remain OPEN. No tests, imports,
builds, benchmarks, device probes, commits or pushes were run.

The earlier source matrix generated Cartesian combinations of family, backend,
source dtype and persisted format, then labeled each populated owner cell
`implemented-source`. A resolvable owner symbol and a route entry do not prove
that this exact tuple reaches the corresponding loader or execution consumer.
`source_scope.py` now labels generated cells `candidate-unverified`, exposes a
separate `combination_status`, and records only the consumer selected by the
row's backend (native/public-range versus device). The validator checks that
candidate rows retain the selected consumer reference without promoting them
to implementation. Existing K3 Metal and rejected-format rows retain their
explicit restriction/rejection states. This corrects status reporting; it does
not close P02 or certify any combination.

The P02 detailed classification checkbox and closure statement above are
therefore open. Each exact tuple still needs source tracing or a concrete,
enforced restriction. No tests, imports, builds, benchmarks, device probes,
commits or pushes were run for this correction.

### P02 fail-closed admission helper and conformance-report correction — 2026-10-13

`source_scope.require_source_matrix_admission` now resolves a complete exact
family/identity/backend/arithmetic/source-format/persisted-format key and
rejects every row except `verified-admitted`; it also refuses an admitted
label if the row lacks a complete callable-chain record. The helper is
exported for construction adapters. `family_conformance.py` now distinguishes
`phase1_identity_catalogued` from `phase1_identity_admitted`; catalog presence
no longer appears as load authorization, and each model's report carries only
its own tuple records. At this audit point, P02's matrix has no verified
admitted tuples, so this helper rejects all exact tuples.

Important remaining limit: production family factories and gateway
construction still do not generally supply the full six-field tuple to this
helper. V4.1 distributed `PublisherSession` is now a narrow exception: it
derives three truthful bank tuples from the explicit BF16/FP8/FP4 flags and
rejects before checkpoint opening because those rows are not verified-admitted.
This is an enforced restriction, not distributed-path completion. Native
loader signatures elsewhere do not carry all six dimensions, so silently
inferring them would create false admission. P02 remains open until each other
factory or gateway either passes a truthful exact tuple through this gate or
explicitly rejects that route before construction. No tests, imports, builds,
benchmarks, device probes, commits or pushes were run for this change.

### P04/P14 media staging cancellation checkpoint — 2026-10-13

`MultimodalSessionExecutor.forward_media` now observes the service cancellation
signal during field admission and immediately before and after each detached
array copy. If cancellation arrives while a copy is in progress, the next
boundary raises `ExecutionCancelled` before the decoder receives a partial
media bundle; the enclosing `NativeExecutionService.generate` transaction then
restores the pre-generation model snapshot. This closes cancellation observation
for the bounded media-copy stage only. It does not make an individual NumPy
copy interruptible, nor close per-family media ordering/geometry, processor
cleanup, or the full P04/P14 caller lifecycle. Those requirements and P15
remain open. No tests, imports, builds, benchmarks, device probes, commits or
pushes were run for this change.

### P11/P12 native snapshot-handle ownership — 2026-10-13

`SegmentEdgeSnapshot` is a frozen dataclass, but callers can construct another
value with copied fields. The executor previously used dataclass equality for
snapshot lookup, allowing such a forged equal value to select or release the
issued native-state files. `SegmentEdgeExecution` now resolves snapshot state
by object identity for both restore and release. This closes that specific
public snapshot-capability ownership gap only; family-by-family ABI and
recovery/reentry review remains open. No tests, imports, builds, benchmarks,
device probes, commits or pushes were run for this source change.

P12 direct ABI cancellation follow-up: `SegmentBridge.call` now snapshots a
stateful session before a cancellable `FORWARD` call and restores that state if
the native callback reports cancellation. If rollback fails, the handle is
quarantined from further execution but remains releasable for cleanup. The ABI
also rejects cancellation owners on non-execution operations rather than
silently ignoring cancellation. This closes only direct SegmentBridge session
forward cancellation rollback; it does not establish serving caller ordering,
all-family physical-state coverage, distributed consensus rollback, or stale/
foreign restore closure. P12 remains IN PROGRESS and P15 OPEN. No tests,
imports, builds, benchmarks, device probes, commits or pushes were run.

P11/P12 Mixtral device snapshot-capability follow-up: `TorchMixtralDecoder`
restore and release now require the exact issued `TorchMixtralSnapshot` object
by identity (and exact handle type), rather than dataclass equality. This
prevents a caller-created equal-valued handle from selecting or releasing
device KV snapshot state. It closes only this concrete device decoder snapshot
lookup boundary; family ABI ranges/head schemas, callback ownership,
reentry/thread rules, service/factory/CLI registration, and remaining recovery
paths stay open. P11/P12 remain IN PROGRESS and P15 OPEN. No tests, imports,
builds, benchmarks, device probes, commits or pushes were run.

P12 direct K3 codec follow-up: `k3_generation_recovery.write_generation_boundary`
and `restore_generation_boundary` now acquire the K3 model's reentrant owner
lock around the multi-step cache/sampler boundary, rather than relying on the
docstring's caller-held request lock. Optional cooperative cancellation is
checked before work, during cache staging/streaming, and before state mutation;
cancellation during restore uses the same rollback path as other restore
failures. Scope is only this direct K3 generation codec: the V4.1 compound
generation codec, all other family codec/caller combinations, and distributed
operation-failure consensus remain unaudited/unclosed. P12 remains IN PROGRESS.

### P12 OLMoE Segment snapshot owner serialization — 2026-10-14

`OlmoeSegmentSession` now owns a reentrant lock shared by engine/session
forward, reset, close, state snapshot/restore, and the entire portable segment
snapshot read/write operation. This prevents a direct checkpoint codec from
capturing or publishing KV while an in-process forward/reset concurrently
changes the same session. It closes only the Segment session's direct
in-process mutation race; ABI caller cancellation/reentry, other family
physical-state ownership, and distributed consensus remain open. P12 remains
IN PROGRESS. No tests, imports, builds, benchmarks, device probes, commits or
pushes were run.

### P12 direct V4.1 generation publication owner serialization — 2026-10-13

`v41_generation_recovery.publish` is a direct codec entrypoint as well as a
serving callback. It now acquires the V4.1 model's reentrant lock for ordinary
generation publication, or the full owned DSpark decoder/proposer/target/model/
draft lock chain for compound publication, before checking the live cache and
verifier boundary and keeps those locks through synchronous frame writing.
The codec rejects incomplete or aliased lock chains, and observes cancellation
before lock acquisition, after acquisition, and at its existing bounded frame
transfer checkpoints. `PublisherSession.resume_generation` already serializes
restore/continuation under its non-reentrant serving lease; this source change
closes the direct publisher callback race only. It does not prove cancellation
of a blocked sink callback, every family codec/caller ordering, distributed
operation-failure consensus, or complete physical-state recovery. P12 remains
IN PROGRESS and P15 OPEN. No tests, imports, builds, benchmarks, device probes,
commits or pushes were run.

P14 auth follow-up: `_require_auth` now derives the current request path before
resolving its credential, and `_enforce_auth_gate` uses the same route-scoped
credential resolver. This makes the Anthropic `/v1/messages` `x-api-key`
fallback work through both ordinary authentication and opt-in scope enforcement
while leaving other routes bearer-only. Streaming, cancellation, full request
semantics and remaining serving callers are still open. No tests, imports,
builds, benchmarks, device probes, commits or pushes were run.

### P03 resident inventory budget recheck — 2026-10-13

`resident_family_inventory.admit_resident_inventory` now validates header
record names, shapes and source dtypes itself, computes the FP32-loaded plus
source-conversion staging bound (retaining I64 metadata width), and rejects an
inventory exceeding its explicit byte admission before family-specific shape
checks. The native checkpoint reader already applies a related whole-checkpoint
budget; this makes the family admission helper safe as an independent source
contract too. This does not close wrapper/multimodal inventory-to-consumer
paths or any P04/P09/P10 consumer mapping. P03 remains DONE only for the
resident inventory contract; P02 remains candidate/unverified until exact
family/backend/format callable chains are traced. No tests, imports, builds,
benchmarks, device probes, commits or pushes were run for this source change.

### P09/P10 selected-device packed expert staging — 2026-10-13

`TorchMixtralExperts.execute` now checks cancellation between each decoded
gate/up/down host-to-device tensor copy and clears already-staged tensors if
any later copy or cancellation fails. The provider does not return or retain a
partially staged expert. This closes only the selected-device packed-expert
staging cleanup boundary; it does not establish per-family format/backend
coverage, full converter cleanup, device placement closure, or E8 execution.
E8 remains explicitly rejected for Mixtral/Qwen3.5/Qwen4 selected-device
experts pending compatible bounded activation rotation. P09/P10 remain open.
No tests, imports, builds, benchmarks, device probes, commits or pushes were
run for this source change.

### P13 K3 media placement evidence gate — 2026-10-13

The K3 media adapter already binds source/checkpoint/library pins, the pinned
processor fixture identity, the expanded-prefix logits and vision-feature
observations, complete physical cache leaves, and an explicit candidate/oracle
comparison path. Static inspection found no source-bound observer that emits
actual placement receipts for each media encode, media prefill, continuation
decode, and restore operation. `K3MediaAdapter` now declares this missing
operation-placement schema explicitly, and `run_media_comparison` rejects the
K3 media comparison unless both adapters provide that exact per-operation
schema. Configured device strings are not accepted as placement evidence. This
is a concrete fail-closed restriction, not a produced receipt or runtime
comparison; P13 remains IN PROGRESS and P15 remains OPEN. No tests, imports,
builds, benchmarks, device probes, commits or pushes were run.
### P04 Inkling media-to-logits and interrupted cleanup trace — 2026-10-13

Static source tracing covers publisher processor output ->
`MultimodalSessionExecutor.forward_media` detached bounded staging ->
`InklingMultimodalDecoder.forward` placeholder accounting and ordered scatter ->
Inkling HMLP image or discrete dMel audio encoding -> native/device text
`forward(inputs_embeds=...)` -> logits. Image patches are consumed in processor
array order; audio frames retain row-major order after the optional binary
frame mask. Both feature counts must exactly match their expanded placeholder
counts, geometry and codebook bounds are checked, and text positions follow the
expanded unpadded processor token sequence. The service currently rejects a
media prompt larger than one admitted prefill operation, so it cannot duplicate
media dispatch across chunks. Cancellation is checked before/after owned
processor-array staging and between HMLP layers; native/device owners receive
the cancellation callback during text execution.

Inkling wrapper cleanup now independently attempts MTP, text/session, media,
and owned projection/expert provider release. It retains failed owner references
for retry and defers closing callback providers while MTP/text may still use
them. Constructor rollback uses the same cleanup path. This closes the observed
short-circuit cleanup gap only; it does not certify every processor ordering,
direct caller, recovery trajectory, cancellation race, or borrowed-owner
lifecycle. Inkling P04/P14 and overall P04 remain in progress. No tests,
imports, builds, benchmarks, device probes, commits or pushes were run.

### P04/P14 native media stream lazy staging — 2026-10-14 (superseded)

`PublisherProvider.stream_complete_media` previously validated and copied all
decoded frame inputs synchronously before returning the stream iterator. It
now returns a lazy event iterator that performs admission and detached-frame
staging on first consumption, so creating and abandoning an unconsumed media
stream does not copy media or enter the warm-session request path. This does
not make the copy interruptible after iteration begins, and does not close
other serving media entry points or ordering/geometry/cleanup lifecycle gaps.
P04/P14 remain IN PROGRESS and P15 OPEN. No tests, imports, builds, benchmarks,
device probes, commits or pushes were run.

This lazy detachment behavior was superseded on 2026-10-15 by the stable
ownership correction below: current `stream_complete_media` eagerly detaches
caller buffers before returning its iterator.

### P14 upstream stream cancellation propagation — 2026-10-13

`EgressClient.fetch` now accepts an optional callable cancellation check and
checks it between streamed response-line reads and before line dispatch. The
model gateway passes its consumer-closed event into that egress path, so queued
backpressure publication and subsequent reads stop once the downstream
iterator closes. This does not interrupt a socket already blocked in
`readline`; cancellation latency is still bounded by the configured network
timeout. It also does not close local native-provider stream lifecycles or
complete OpenAI/Anthropic SSE semantics, stop propagation, warm-session/media
cleanup or full P14 caller coverage. P14 remains IN PROGRESS and P15 OPEN. No
tests, imports, builds, benchmarks, device probes, commits or pushes were run.

### P09/P10 selected-device FP8 representation preflight — 2026-10-13

`device_packed_decode.decode_device_expert` now validates every FP8 expert
matrix's E4M3 code bytes and block128 F32 scale grid before allocating decoded
matrices. It rejects non-finite codes/scales, negative scales, and zero-scale
tiles containing any nonzero-magnitude code; thus a malformed later matrix
cannot leave earlier converted matrices allocated during unwinding. The check
is limited to the selected-device FP8 conversion path and its existing bounded
host decode budget. Other format/backend consumers, converter lifecycle
cleanup outside this path, device placement/no-fallback evidence, and complete
family coverage remain open; P09/P10 remain IN PROGRESS and P15 OPEN. No tests,
imports, builds, benchmarks, device probes, commits or pushes were run.

### P04/P07 K3 decoded-media staging admission — 2026-10-14

`k3_execution.video_input.prepare_media` now measures the complete decoded
still/video-frame inventory and enforces its decoded-media budget before
detaching caller buffers. It checks cancellation while preflighting and between
frame copies, preventing oversized media from first allocating an unbudgeted
duplicate and making cancellation responsive during ownership transfer. This
closes only the pre-copy admission/cancellation gap in this K3 preparation
caller. It does not close placeholder ordering across callers, processor
parity, temporal position semantics, execution/serving integration, restore
ownership, or other-family media cleanup; P04/P07 remain IN PROGRESS and P15
OPEN. No tests, imports, builds, benchmarks, device probes, commits or pushes
were run.

### P04/P07/P14 K3 owned PIL media cleanup — 2026-10-14

`k3_execution.video_input.prepare_media` now retains ownership of copied PIL
still-image staging objects and closes them on successful packing, preprocessing
failure, or cancellation. Cleanup errors are attached to the primary failure;
without a primary failure, cleanup failure is reported. Caller-owned source
images remain untouched. This fixes only copied PIL staging lifetime in this
K3 media path. It does not establish all-family processor cleanup, ordering and
position parity, gateway lifecycle, media-to-logits coverage, or restore and
borrowed-owner closure. P04/P07/P14 remain IN PROGRESS and P15 OPEN. No tests,
imports, builds, benchmarks, device probes, commits or pushes were run.

### P02/P03 Qwen3.5 multimodal reader inventory gate — 2026-10-14

The raw multimodal checkpoint reader now admits only the explicitly bounded
Qwen3.5 native wrapper; Qwen4, Inkling, and GLM5 must use their dedicated
conditional inventory planners. Before materializing any Qwen3.5 wrapper
payload, the shared header admission validates both the complete text tensor
inventory and the depth-dependent vision bank, including canonical namespace,
shape, source dtype, tied-head omission policy, and total resident budget. The
direct decoder path and raw reader share this check, preventing a low-level
reader call from bypassing the text decoder's post-materialization name check.
This is a bounded route fix, not all-family inventory/consumer closure or a
truthful family/backend/format matrix; P02 remains IN PROGRESS, P03 remains
bounded-DONE, and P15 remains OPEN. No tests, imports, builds, benchmarks,
device probes, commits or pushes were run.

### P09 IQ3 grid publication cleanup — 2026-10-14

`IQ3Grid.write` now keeps the original publication failure primary if closing
the raw or directory descriptor, or removing its temporary file also fails;
cleanup failures are attached as notes instead of replacing the format
conversion/publication error. This is limited to the IQ3 grid artifact writer.
It does not close
expert packer transaction cleanup, other persisted-format converters, family/
backend loader-consumer coverage, or placement/evidence obligations. P09
remains IN PROGRESS and P15 OPEN. No tests, imports, builds, benchmarks,
device probes, commits or pushes were run.

### P03 Qwen4 tied-head inventory configuration strictness — 2026-10-14

The Qwen4 publisher component contract now rejects a present non-boolean
`tie_word_embeddings` value before conditional inventory planning or payload
reads. The direct `IndependentQwen4TextDecoder` boundary enforces the same
contract for non-publisher callers. This matters because the flag changes
whether `lm_head.weight` is separately admitted or aliased to the embedding;
truthiness coercion could otherwise select inconsistent inventory and
execution ownership. This closes only that Qwen4 configuration/admission
consistency defect. It does not close family-wide optional/MTP/expert consumer
tracing or factory/caller closure. P03 remains DONE only for bounded inventory
contracts; P15 remains OPEN. No tests, imports, builds, benchmarks, device
probes, commits or pushes were run.

### P12 DSpark captured physical-state alignment — 2026-10-14

`DSparkBackbone.snapshot` now checks that retained target hidden rows and
logits have valid finite geometry and correspond to the nested backbone's
captured token-history length before publishing a composite receipt. If that
check fails, it releases the nested backbone snapshot; failed cleanup
quarantines the DSpark owner and is attached to the primary validation error.
This catches inconsistent physical-state capture when a caller bypasses the
DSpark owner and mutates its wrapped backbone. It does not establish
cross-family physical-state ownership, distributed restore consensus, or
serving caller ordering. P12 remains IN PROGRESS and P15 OPEN. No tests,
imports, builds, benchmarks, device probes, commits or pushes were run.

### P12 durable continuation rollback quarantine — 2026-10-14

If `DurableContinuationStore.recover_into` cannot restore the existing
session's rollback snapshot after a failed cache load or catalog check, it now
marks that session closed before propagating the primary failure. The owner can
no longer continue serving from physical state that may be partially restored;
the cleanup failure is still attached to the primary exception. This is a
single-process service-session safeguard, not distributed consensus or a
cross-family recovery guarantee. P12 remains IN PROGRESS and P15 OPEN. No
tests, imports, builds, benchmarks, device probes, commits or pushes were run.

### P14 constrained-generation rejection on direct route — 2026-10-14

The `/api/adaptive-engine/direct/generate` route now rejects `grammar`,
`gbnf`, `guided_decoding`, `guided_json`, `json_schema`, and `response_format`
inside its `generation` object before dispatch. The direct OpenVINO adapter
does not implement a grammar-constrained decoder or target-forward constraint
verification; previously these unknown values could be discarded and produce
unconstrained output. This restriction applies to the direct adapter only and
does not close the separate Colibri compatibility contract, family-specific
tool protocols, or GBNF draft-source integration. P14/P15 remain IN PROGRESS/
OPEN. No tests, imports, builds, benchmarks, device probes, commits or pushes
were run.

### P09 expert-store partial payload cleanup — 2026-10-14

`DiskExpertStore._read_payload` now releases a partially materialized mmap view
when cancellation, a short read, or checksum validation fails, and cleanup
failures are attached to the primary error. Demand-cache admission also
releases a payload if entry construction/insertion fails, and asynchronous
prefetch releases successfully read payloads if cache admission raises before
ownership transfers. This covers the expert-store read/cache staging boundary;
it does not complete packer publication transactions, other converter cleanup,
format/backend loader-consumer chains, E8/IQ3 rotation, or placement evidence.
P09 remains IN PROGRESS and P15 OPEN. No tests, imports, builds, benchmarks,
device probes, commits or pushes were run.

### P09 selected-device low-bit scale admission — 2026-10-14

`decode_device_expert` now rejects non-finite, zero, or negative INT2/INT3
row/group scales before allocating decoded matrices. The source packer always
emits a positive scale (including 1.0 for all-zero blocks), so this closes a
consumer-side admission mismatch for those two persisted formats. This is
limited to selected-device low-bit decoding; family/backend loader-consumer
coverage, other format geometry and rotation parity, converter cleanup, and
placement evidence remain open. P09 remains IN PROGRESS and P15 OPEN. No tests,
imports, builds, benchmarks, device probes, commits or pushes were run.

### P10 embedding autotuning admission restriction — 2026-10-14

`DeviceAutoTuner.tune` now accepts only `prefill` and `decode`. It rejects
`embedding` before selecting a representative dimension or invoking the
OpenVINO benchmark adapter, because the adaptive backend registry has no
registered embedding frontend; this prevents synthetic benchmark evidence
from implying an executable embedding path. This does not prove model
compilation, actual placement, precision preservation, or runtime no-fallback
for the admitted text phases. P10 remains IN PROGRESS and P15 OPEN. No tests,
imports, builds, benchmarks, device probes, commits or pushes were run.

### P13 exact-row and artifact-pin manifest join — 2026-10-14

For opt-in whole-model comparisons, `comparison_configuration` now hashes a
canonical payload containing the exact family/model/backend/arithmetic/
source-format/persisted-format/media row, both module:callable identities, and
the complete oracle and candidate artifact-pin records. The runner recomputes
the digest and checks those records still match the manifest after validating
the pinned files but before importing adapter code. Adapter metadata must
still echo the exact row, while the report retains `receipt_present=False`,
`executed=False` and `certification=False` for the binding itself. This closes
the manifest-level join only; it is not a signature or evidence that adapters
are correct, nor does it produce row-specific observations, physical-cache
receipts, media ordering evidence, placement receipts, or coverage for
unbound historical manifests. P13 remains IN PROGRESS and P15 OPEN. No tests,
imports, builds, benchmarks, device probes, commits or pushes were run.

### P06 collective schedule disagreement quarantine — 2026-10-14

`TensorParallelProjections._agree` now quarantines the bank when ranks
successfully reach agreement but report different collective schedule or
policy descriptors. The mismatch is already unanimous at that control
boundary, so keeping the bank open would allow a retry to select different
next tensor collectives (for example, all-reduce versus all-gather). The
guard prevents unsafe bank reuse after the detected mismatch; it does not
recover blocked collectives, process loss, transport failure, or establish
coverage for every distributed bank/consumer. P06 remains IN PROGRESS and P15
OPEN. No tests, imports, builds, benchmarks, device probes, commits or pushes
were run.

### P03 Inkling MTP consumer-budget admission — 2026-10-14

`load_publisher_conditional_decoder` now collects Inkling MTP tensor headers
and rejects a present MTP bank before any payload read unless both the
configured MTP component and work budgets admit construction of the actual
`InklingMTP` draft consumer. When enabled, the existing complete MTP header
planner still validates declared depth, local-layer schedule, shared weights,
and every stage shape before materialization; the wrapper then constructs the
draft owner. This closes only the no-consumer/no-budget admission seam for
publisher Inkling. It does not complete all-family optional-bank consumer
mapping, prove draft usage from a serving caller, or close P04/P05 lifecycle
semantics. P03 remains DONE only at the bounded inventory-contract level;
P05 remains IN PROGRESS and P15 OPEN. No tests, imports, builds, benchmarks,
device probes, commits or pushes were run.

### P06 failed distributed snapshot creation cleanup — 2026-10-14

When a rank-agreed `snapshot` operation fails or reports different logical
boundaries, ranks that successfully created a service snapshot now release that
new receipt before the coordinator is poisoned. If releasing the receipt
fails, that rank attempts to close the owning session and removes coordinator
references only after close succeeds. Cleanup outcomes are gathered across the
control group so the failure remains rank-agreed. This prevents successful
ranks from retaining newly-created receipts behind a failed snapshot request.
It does not make snapshot creation rollbackable after process loss or a broken
collective: transport failure can still strand peers, and the group still
requires restart. P06/P12 remain IN PROGRESS and P15 OPEN. No tests, imports,
builds, benchmarks, device probes, commits or pushes were run.

### P03 Inkling MTP budget preflight — 2026-10-14

`admit_mtp_inventory` can now receive the MTP total ownership budget and
computes a conservative lower bound for all non-shared MTP stage/projection
tensors at FP32 width. Both the publisher conditional loader and direct
multimodal constructor pass that bound while admitting headers, before payload
materialization. Shared embedding/head tensors are excluded because they are
checked as aliases against the target backbone rather than retained as MTP
copies. This prevents a declared MTP bank that cannot fit its own ownership
budget from being staged under only the larger model-wide weight budget. It
does not prove peak staging/workspace fit or close other family inventories;
P03 remains bounded-DONE and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits or pushes were run.

### P11 Segment callback null out-pointer guard — 2026-10-14

The Segment callback dispatcher now checks the required `written` length
out-pointer before clearing or writing it. A null pointer returns
`CALLBACK_ERROR` and retains a diagnostic callback error instead of attempting
to dereference a null ctypes pointer before the dispatcher exception boundary.
This protects the null case only; arbitrary invalid non-null pointers cannot
be proven safe in Python and remain a native caller contract. Other family ABI
schemas, registration paths and callback wiring remain under review. P11
remains IN PROGRESS and P15 OPEN. No tests, imports, builds, benchmarks,
device probes, commits or pushes were run.

### P04 Qwen3.5 audio admission restriction — 2026-10-14

The publisher conditional loader now rejects a Qwen3.5 checkpoint that
declares `audio_config` or audio-tower tensor headers immediately after
configuration and resident inventory admission, before a tensor reader is
constructed or checkpoint payloads are read. The existing Qwen3.5 wrapper
also rejects audio tensors before vision/text dispatch. This explicitly
restricts audio rather than treating optional inventory as an executable
media path. Video/image ordering, positions, geometry, other family audio
paths, and caller cleanup remain open; P04 remains IN PROGRESS and P15 OPEN.
No tests, imports, builds, benchmarks, device probes, commits or pushes were
run.

### P09 expert-container staging writer cleanup — 2026-10-14

The expert-container conversion path now owns staging writer closure explicitly
instead of delegating it to `ExitStack`. If opening a later destination,
conversion, writing, the progress callback, or flush/fsync fails, all previously
opened staging writers are closed; close failures are attached to the original
exception rather than replacing it. On a successful conversion, all writers
are still closed and the first close failure is raised after attempting the
rest. This is limited to staging writer lifecycle in
`expert_packing.pack_experts`; it does not complete multi-container publish
rollback, all other format converters, loader/consumer parity, or placement
receipts. P09 remains IN PROGRESS and P15 OPEN. No tests, imports, builds,
benchmarks, device probes, commits or pushes were run.

### P06/P12 distributed close failure cleanup — 2026-10-14

If distributed `close_session` succeeded on some ranks but failed or disagreed
on another, the coordinator now makes a second rank-agreed cleanup attempt for
the same local session on every rank that still retains its handle. Successful
cleanup also drops that session's retained generation boundary and distributed
snapshot references. The collective remains poisoned and requires group/rank
restart after any close disagreement; this is best-effort cleanup, not
distributed close rollback or general process-loss/transport recovery. P06 and
P12 remain IN PROGRESS and P15 OPEN. No tests, imports, builds, benchmarks,
device probes, commits or pushes were run.

The same retry/agreement boundary now covers whole-service `close`: after an
initial rank failure or result disagreement, each rank whose service close did
not complete retries its retained service cleanup before leaving the control
loop. This can drain owners left by a partial service close, but cannot restore
already closed ranks or recover a lost transport/process; the coordinator stays
poisoned and requires group restart.

### P14 OpenAI user-attribution restriction — 2026-10-14

The local non-streaming OpenAI-compatible route now rejects any supplied
`user` field with an explicit unsupported-feature error. The route has no
durable caller identity or attribution/abuse-monitoring sink, so it no longer
silently accepts and discards that identifier. This is a route-scoped
restriction only; it does not establish general OpenAI request compatibility
or close caller lifecycle coverage. P14 remains IN PROGRESS and P15 OPEN. No
tests, imports, builds, benchmarks, device probes, commits or pushes were run.

### P14 malformed-Unicode request rejection — 2026-10-14

The local OpenAI/Anthropic compatibility handler now treats escaped lone
surrogates in stop strings and message text as invalid request data. It checks
UTF-8 encodability before applying byte limits, returning the route's normal
client-error envelope instead of allowing `UnicodeEncodeError` to escape into
generic server-error handling. This is input validation only; it does not add
streaming, stop-token adaptation, complete API semantics, or caller lifecycle
coverage. P14 remains IN PROGRESS and P15 OPEN. No tests, imports, builds,
benchmarks, device probes, commits or pushes were run.

### P05 speculative reset attempts both physical owners — 2026-10-14

`SpeculativeDecoder.reset()` previously stopped after a target reset exception,
leaving a stateful draft proposer unattempted. It now attempts target and draft
reset independently, preserves the first error and annotates additional reset
failures, and only clears logical history/quarantine after both owners succeed.
This closes the independent-owner attempt gap for this decoder reset method;
failed owners remain quarantined for a later retry. It does not establish
rollback to the previous physical boundary, nor reconcile every generation,
restore, stream, or serving caller. P05 remains IN PROGRESS and P15 OPEN. No
tests, imports, builds, benchmarks, device probes, commits or pushes were run.

### P04/P03 Qwen3.5 native audio admission restriction — 2026-10-14

The raw native Qwen3.5 multimodal checkpoint reader now rejects a non-null
`audio_config` or audio/audio-tower tensor headers during metadata admission,
before any checkpoint payload is read. Its native wrapper implements image and
video only, so accepting an audio declaration would have admitted inventory
without an execution consumer. The publisher conditional route already has an
equivalent early rejection. This closes the raw-reader admission seam only;
other family media routes, producer-to-consumer closure, and caller lifecycle
remain open. P03 stays DONE only for bounded inventory contracts, P04 remains
IN PROGRESS, and P15 remains OPEN. No tests, imports, builds, benchmarks,
device probes, commits or pushes were run.

### P12 K3 generation-recovery rollback quarantine — 2026-10-14

If restoring a K3 generation checkpoint fails after physical cache mutation
and restoring the pre-operation snapshot also fails, the recovery path now
marks the K3 executor closed under its operation lock. The normal K3 operation
gate therefore prevents further inference from potentially inconsistent cache
state, while `close()` remains available to release retained resources. This
is a single-process K3 codec safeguard, not a guarantee that rollback succeeds,
not distributed consensus, and not cross-family recovery ownership closure.
P12 remains IN PROGRESS and P15 OPEN. No tests, imports, builds, benchmarks,
device probes, commits or pushes were run.

### P11/P12 Segment ABI fixed-result extent enforcement — 2026-10-14

`SegmentBridge.call` now checks the exact byte count for successful FORWARD,
EMBED and HEAD calls, and requires zero result bytes from RESET/RESTORE while
requiring a nonempty snapshot result. The native C bridge only enforces
`written <= capacity`; previously a backend could report success with a
truncated fixed-shape tensor and the public ABI would return it as valid. A
short/long session FORWARD is quarantined because physical KV may already have
advanced. Remaining P11 work is family-wide ABI symbol/range/schema/callback
and factory/CLI/service wiring review. Remaining P12 work is cross-family
physical-state ownership, stale/foreign restore and cancellation ordering, and
distributed rollback/consensus; this check is not a general recovery guarantee.
P11/P12 remain IN PROGRESS and P15 OPEN. No tests, imports, builds, benchmarks,
device probes, commits or pushes were run.

### P02/P03 Qwen text architecture identity gate — 2026-10-14

The independent Qwen3.5 and Qwen4 text decoders now reject a present
`architectures` declaration unless it names the exact admitted text class
(`Qwen3_5MoeForCausalLM` or `Qwen4ExpForCausalLM`). Missing declarations remain
accepted for legacy configs; publisher wrappers continue to stamp the exact
architecture during component mapping. This prevents a mismatched family
architecture from silently borrowing the decoder selected only by
`model_type`. It closes only this configuration-identity seam, not complete
family/backend/format tracing or all factory bypasses. P02 remains IN PROGRESS,
P03 remains bounded-DONE, and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits or pushes were run.

### P05 speculative restore proposer-RNG ownership — 2026-10-14

`SpeculativeDecoder.restore()` now rejects a saved snapshot when the snapshot's
proposer-RNG state presence does not match whether the current proposer owns a
NumPy bit-generator. Without this check, a snapshot lacking proposer-RNG
evidence could restore model/draft state while silently retaining randomness
consumed by a different live proposer history. Existing rollback/quarantine
still governs failures after mutation. This closes only this ownership
consistency seam; it does not close acceptance-branch or serving-caller parity.
P05 remains IN PROGRESS and P15 OPEN. No tests, imports, builds, benchmarks,
device probes, commits or pushes were run.

### P09/P10 selected-device Q4 block metadata preflight — 2026-10-14

The explicit selected-device Q4_0/Q4_1 decode path now checks the complete
payload extent and scans every half-precision block metadata value before it
calls the projection decoders. A non-finite value in a late block therefore
fails before any decoded projection array is allocated. This closes only the
Q4_0/Q4_1 selected-device metadata admission boundary; Q5/Q8 and other
family/backend format consumers still need their own path review, and this is
not placement or execution evidence. P09/P10 remain IN PROGRESS and P15 OPEN.
No tests, imports, builds, benchmarks, device probes, commits or pushes were
run.

### P04 publisher media image cleanup aggregation — 2026-10-14

`PublisherMediaProcessor.prepare()` now attempts to close every owned decoded
PIL image even if an earlier close fails. A preprocessing/processor exception
remains primary and receives notes for cleanup failures; if processing
succeeded but cleanup failed, the first cleanup failure is raised and later
cleanup failures are attached. This closes the multi-image cleanup masking and
short-circuit seam for this publisher processor only; it does not close other
family media owners, temporal/position parity, or caller lifecycle coverage.
P04 remains IN PROGRESS and P15 OPEN. No tests, imports, builds, benchmarks,
device probes, commits or pushes were run.

### P07/P14 direct hardware stream backpressure — 2026-10-14

`DirectHardwareClient.stream_generate()` previously used an unbounded local
queue between the device-worker request thread and its iterator consumer. A
slow consumer could therefore buffer an unbounded number of generated events.
The handoff now has a 16-event bound, so a slow consumer backpressures pipe
response dispatch; closing/abandoning the iterator signals cancellation so a
blocked callback stops publishing before an unfinished worker is terminated.
This closes only the local handoff queue bound and abandonment seam. It does
not establish device-side cancellation semantics, bounded buffering in other
providers/routes, or complete stop/recovery/media lifecycle parity. P07/P14
remain IN PROGRESS and P15 OPEN. No tests, imports, builds, benchmarks, device
probes, commits or pushes were run.

### P04 publisher/K3 decoded-frame conversion ownership — 2026-10-14

`K3VideoInput`, `PublisherVideoInput`, and the publisher provider's asynchronous
decoded-media freezer convert caller PIL images to detached RGB arrays. Each
path now closes the temporary PIL image returned by `convert('RGB')` after the
array copy. If copying fails, temporary-image cleanup failure is attached to
the primary copy failure instead of replacing it. This closes temporary
conversion-image ownership at these three admission/freezing boundaries only;
it does not close family-wide media ordering, geometry, caller cancellation,
or serving lifecycle. P04/P07/P14 remain IN PROGRESS and P15 OPEN. No tests,
imports, builds, benchmarks, device probes, commits or pushes were run.

The typed K3 and publisher selected-video constructors also account the
provider's existing 256 MiB decoded-media charge before eagerly detaching any
frames. Thus direct construction cannot first allocate an over-budget copied
frame inventory and only fail later at the provider boundary. This remains a
bounded-constructor admission improvement, not cross-family media closure.

### P14 direct native publisher UTF-8 request boundary — 2026-10-14

`NativePublisherProvider._run()`, its streamed caller, and direct Brio scoring
boundaries now size user text, choices, and rendered prompts through the same
UTF-8 boundary helper. Lone surrogates are converted to
`ProviderConfigurationError` instead of leaking `UnicodeEncodeError` from
those direct provider paths. Other provider boundaries, full API semantics,
and the caller-by-caller lifecycle audit remain open. P14 remains IN PROGRESS
and P15 OPEN. No tests, imports, builds, benchmarks, device probes, commits or
pushes were run.

### P14 outbound model-stream cancellation while socket-read blocked — 2026-10-14

The `model_gateway._stream_model_lines` caller already supplies a cancellation
predicate to `EgressClient.fetch`, but `EgressClient._request_once` previously
checked it only between `HTTPResponse.readline()` calls. A disconnected client
could therefore leave the daemon fetch thread blocked on an idle upstream socket
until its read timeout. Streaming reads now have a small cancellation watcher
that closes the active connection when the consumer's predicate becomes true;
the reader translates the resulting socket failure back to `InterruptedError`
when cancellation caused it. The outer egress boundary preserves this
`InterruptedError` rather than wrapping it as a generic transport failure. This
preserves bounded queue backpressure and the existing response-size cap. It
closes this upstream idle-read cancellation gap; it does not provide
device-side cancellation, full provider stop semantics, or
caller-wide P04/P07/P14 closure. P14 remains IN PROGRESS and P15 OPEN. No tests,
imports, builds, benchmarks, or device probes were run.

### P13 required whole-model binding and owner preflight — 2026-10-14

The whole-model comparison runner now rejects manifests without an exact
execution-row/artifact-pin binding. Before importing either adapter it
validates the generated source catalog and resolves its source-bearing owners
against this checkout. The source catalog validator also rejects an evidence
obligation lacking a comparison report schema. The runner continues to emit
`receipt_present=False`, `executed=False`, and `certification=False`: this
change enforces structural prerequisites only and does not create or claim
runtime observations, cache receipts, placement receipts, comparison evidence,
or numerical parity. Existing unbound manifests must be regenerated with an
exact binding. P13 remains IN PROGRESS and P15 OPEN. No tests, imports, builds,
benchmarks, device probes, commits or pushes were run.

### P12 durable recovery fresh-owner cleanup guard — 2026-10-14

`DurableContinuationStore.recover` now holds the execution-service session
registry lock across cache-session creation and verifies that the returned
identifier is a newly registered session before history validation or cleanup.
This prevents a malformed cache recovery adapter from returning a pre-existing
caller session and having failure cleanup close that unrelated owner. This
closes one durable-recovery handle-ownership edge only; family physical-state
and distributed recovery closure remain open. P12 remains IN PROGRESS and P15
OPEN. No tests, imports, builds, benchmarks, device probes, commits or pushes
were run.
### P09/P10 selected-device grouped INT4 scale preflight — 2026-10-14

`device_packed_decode.decode_device_expert` now validates the exact g64 grouped
INT4 payload extent and preflights all persisted FP32 scales before allocating
any decoded projection. Scale magnitudes are bounded so the format's minimum
signed code (-8) cannot overflow during FP32 reconstruction. The general
grouped decoder still validates scale finiteness and odd-width nibble padding
before its own allocations. This closes only the selected-device grouped INT4
admission gap; P09/P10 family/backend loader-consumer mapping, other format
geometry and execution/placement evidence remain open. P09/P10 remain IN
PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P11 OLMoE Edge reserved request fields — 2026-10-15

The OLMoE public adapter now rejects nonzero reserved words in `embed`,
`logits`, and `select` request records before using payloads. Its guarded
constructor callbacks clear caller-provided handle/capability outputs before
admission and clear capabilities again if open fails. Tokenize/detokenize clear
their output extents before handle lookup or decoding, preserving only the
intentional required-size result on capacity negotiation. This closes several
stale-out-parameter/undefined-reserved-field seams for this adapter only; it
does not close analogous validation across every family or the wider
Segment/Edge schema, continuation, registration, callback ownership, and
caller-wiring review. P11 remains IN PROGRESS and P15 remains OPEN. Only scoped
`git diff --check` was run; no tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P04 K3 prepared-media buffer ownership — 2026-10-15

`K3PreparedMedia` now rejects pixel/grid arrays that are views or otherwise do
not own their backing buffers. Previously `np.asarray` could preserve a
borrowed view and setting it read-only did not prevent the original owner from
changing bytes (or restore its writeable flag). The internal `prepare_media`
producer constructs detached owning arrays, so this is a fail-closed boundary
for direct construction without another full media copy. This closes only the
prepared-record aliasing seam; family-wide ordering/position/geometry,
processor ownership, and caller lifecycle remain open. P04 remains IN PROGRESS
and P15 remains OPEN. Only `git diff --check` was run; no tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

### P07/P12 K3 recovery timestamp numeric admission — 2026-10-16

K3 generation-boundary recovery now converts timestamp values to finite FP64
inside a guarded conversion before ordering and uniform-interval checks. A
JSON integer too large for FP64 previously made `math.isfinite()` raise
`OverflowError` directly instead of being rejected as malformed temporal
metadata, unlike the bounded media-input validator. Interval checks now use
the already-admitted normalized values, avoiding a second unchecked
conversion. This closes only malformed timestamp handling in the recovery
envelope; it does not establish temporal encoding parity or bind provenance to
pixels. FLA/native continuation parity, full media caller coverage, and batch
rollback failure recovery remain open. P07/P12 remain partial and P15 remains
OPEN. No tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P09/P10 selected-device FP8 decoded-weight overflow preflight — 2026-10-16

The selected-device FP8 block decoder now checks each block's actual maximum
finite E4M3 code multiplied by that block's persisted FP32 scale against the
FP32 representable limit before allocating any decoded projection. Previously
finite scales and codes could individually pass admission while their product
overflowed only during matrix materialization. The check is tile-exact, so it
does not impose a conservative global scale limit on blocks whose codes are
smaller. This closes that selected-device decode admission/cleanup gap only;
it does not establish family/backend format closure, runtime placement, device
no-fallback behavior, or numerical parity. P09/P10 remain IN PROGRESS and P15
remains OPEN. No tests, imports, builds, benchmarks, device probes, commits, or
pushes were run.

### P09/P10 Q4_0 conversion failure cleanup — 2026-10-16

Q4_0 conversion now preserves the original write/fsync/publication exception
when temporary-file removal also fails, adding cleanup failure as diagnostic
context instead of replacing the primary error. A cleanup failure after
successful publication still surfaces to the caller. This closes one offline
conversion cleanup path only; it does not establish general conversion
rollback, persisted loader/execution parity, device placement, or no-fallback
coverage. P09/P10 remain IN PROGRESS and P15 remains OPEN. No tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

### P09/P10 selected-device FP8 decoded-weight overflow preflight — 2026-10-16

The selected-device FP8 block decoder now checks each block's actual maximum
finite E4M3 code multiplied by that block's persisted FP32 scale against the
FP32 representable limit before allocating any decoded projection. Previously
finite scales and codes could individually pass admission while their product
overflowed only during matrix materialization. The check is tile-exact, so it
does not impose a conservative global scale limit on blocks whose codes are
smaller. This closes that selected-device decode admission/cleanup gap only;
it does not establish family/backend format closure, runtime placement, device
no-fallback behavior, or numerical parity. P09/P10 remain IN PROGRESS and P15
remains OPEN. No tests, imports, builds, benchmarks, device probes, commits, or
pushes were run.

### P05 speculative successful-boundary verifier check — 2026-10-15

Before publishing a successful speculative step's logical history and logits,
`SpeculativeDecoder.step()` now compares the exposed target history/logits (when
available) with that would-be committed boundary. If an accepted-forward or
rejection-repair path returns normally but leaves exposed verifier state
behind or divergent, the existing failure handler rolls target/draft/both RNG
owners back and quarantines if rollback cannot be established. This is one
local verifier-boundary guard; it does not close every accepted/rejected/bonus/
EOS/context/cancellation/repeated-restore path or every serving caller. P05
remains IN PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks,
device probes, commits, or pushes were run.

### P07/P14 K3 batch retained-continuation admission — 2026-10-16

`K3BatchExecution` now weakly tracks externally retained `K3BatchSnapshot`
receipts and includes their detached physical-state bytes when admitting any
new snapshot or transaction snapshot for forward/restore/reset. Previously
`max_snapshot_bytes` bounded each capture independently, so callers could keep
unbounded copies of recurrent/attention state while continuing to execute
transactions whose additional snapshots each fit the same full budget. Weak
tracking does not retain snapshots itself; dropping a receipt releases its
charge on the next admission. This bounds source-level continuation ownership
but does not make a failed slot restore recoverable: if atomic rollback itself
fails, the existing behavior quarantines the whole batch. FLA/native numerical
continuation parity, media batching, and device/process-loss recovery remain
open. P07/P14 remain partial and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P09/P10 direct selected-device provider format guard — 2026-10-16

`TorchMixtralExperts` now resolves every layer's persisted format against its
exact family/device capability row during provider admission. This also covers
the inherited Qwen3.5, Qwen4 and GLM-DSA selected-device providers, preventing
direct provider/decoder construction from using the shared decoder's broader
`FORMATS` set to bypass the family-specific allowlist. The native-family CLI
has the matching pre-store guard. These source checks restrict unsupported
formats but do not prove an admitted device operation stays on-device, certify
the selected hardware, or supply placement receipts/runtime parity evidence.
Other direct factory and loader paths still require reconciliation. P09/P10
remain IN PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks,
device probes, commits, or pushes were run.

### P14 OpenAI best_of scalar validation — 2026-10-16

`_handle_compat_completion()` now requires `best_of` to have exact JSON
integer type before accepting the sole supported value `1`. Python's equality
would otherwise treat JSON `true` as integer `1`, silently accepting a
malformed field in a route whose candidate-count semantics are restricted.
The capability report records the exact scalar restriction. This does not add
best-of sampling or change valid `best_of=1` behavior; other request
normalization and OpenAI/Anthropic parity obligations remain open. P14 remains
IN PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P14 runtime SSE closes the resource-owning iterator — 2026-10-15

The runtime SSE sender now closes the concrete iterator returned by
`iter(events)` when that iterator exposes `close()`, falling back to the
original iterable's close hook otherwise. This covers iterable wrappers whose
resource-owning generator is distinct from the wrapper and ensures the actual
stream owner is released after exhaustion or client disconnect. It closes one
SSE cleanup seam only; caller-by-caller serving lifecycle closure remains
incomplete. P14 remains IN PROGRESS and P15 remains OPEN. Only
`git diff --check` was run; no tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P04 publisher cross-modality placeholder ordering restriction — 2026-10-15

`PublisherMediaProcessor.prepare` now rejects requests that combine images,
audio, or videos across modality arguments. The upstream processor interface
accepts these as separate arrays and does not provide a verified mapping from
their concatenated outputs back to interleaved prompt placeholders; accepting
such a request could silently misbind media features. Ordering within a single
modality remains preserved. This is an explicit source restriction, not
cross-family media-order support, and direct family executors/callers still
need separate review. P04/P07/P14 remain IN PROGRESS and P15 remains OPEN. No
tests, imports, builds, benchmarks, device probes, commits, or pushes were run.

### P11 Qwen4 routing-observer reentry stability — 2026-10-16

The native Qwen4 session run now pins the active thread-local routing observer,
context, and registration generation for the whole staged append. If an
observer callback ends/replaces its scope reentrantly, the append fails before
publishing output or continuation state. This prevents mixed-observer routing
telemetry within one model run while preserving transactional session state.
This is a narrow Qwen4 callback-scope guard, not completion of Qwen4 or the
family-wide ABI/reentry review. Qwen3.5, GLM5, GLM-DSA and V4.1 still require
their callback/session direct-path maps; P11 remains IN PROGRESS and P15 OPEN.
No tests, imports, builds, benchmarks, device probes, commits, or pushes were
run.

### P09/P10 sparse-runtime family/native format gate — 2026-10-16

`SparseExpertRuntime` now resolves each Mixtral and OLMoE layer's persisted
format against that family's native consumer row before constructing
`DiskExpertStore`. The generic kernel registry remains broader; previously a
format present in that registry could reach artifact opening even when the
family/backend source matrix did not admit it. This closes the known Mixtral
and OLMoE generic-runtime admission boundary only. Other direct family loaders,
factories, and tooling paths still need exact source-to-consumer tracing, and
actual accelerator placement/no-fallback receipts and runtime comparisons are
still required for P10 evidence. P09/P10 remain IN PROGRESS and P15 remains
OPEN. No tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P11 Qwen4 direct Edge owner validation — 2026-10-16

The Qwen4 C Edge `embed`, `logits`, and `select` adapter callbacks now reject a
null owner before reading its model pointer. Previously those three functions
validated request records but dereferenced `owner` during local initialization,
unlike the GLM5/Inkling callbacks which validate both owner and request first.
The shared registry normally supplies a live owner, but direct callback safety
now matches the family adapters' own boundary. No other Qwen4 ABI contract or
family-wide P11 map is closed; P11 remains IN PROGRESS and P15 OPEN. No tests,
imports, builds, benchmarks, device probes, commits, or pushes were run.

### P06 distributed setup agreement malformed-record cleanup — 2026-10-15

`v41_distributed_session.agreed_stage` now validates every gathered setup
agreement record before formatting rank diagnostics. A malformed record is
reported as an agreed setup failure and follows the same local resource cleanup
path as ordinary remote construction failures, instead of raising while
indexing the bad record and leaking locally-created provider/kernel/bank
objects. InterruptedError is still propagated only when every rank supplied a
valid cooperative-interruption record. This is a local cleanup safeguard; it
cannot restore agreement after a broken collective or recover a lost process.
P06/P12 remain IN PROGRESS and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P06/P12 V4.1 projection dispatch publication agreement — 2026-10-16

The V4.1 family bank factory previously performed a rank-local `assignments`
dictionary insertion immediately after each bank's collective source
registration. A local allocation/interruption failure could therefore make
that rank unwind while peers entered the next source-registration collective.
Each BF16, FP8, FP4, and non-BF16 provider dispatch insertion now has its own
rank agreement boundary; failed publication is unanimous, and cooperative
interruptions retain an interruption-specific exception. This does not make
the process group resilient to a rank or transport disappearing during that
agreement; survivors can still block and require whole-group restart.

Restore-path source map for this pass: distributed cached rank files are
manifest-pinned and independently receipt-checked by `load_session_cache`;
generation recovery additionally pins `application_identity` and compares
manifest content before/after restore. K3 recovery scopes bind media
provenance, V4.1 restore validates private checkpoint identity and now rolls
back if cancellation arrives during publication, while durable continuation
pins its cache receipt. Qwen3.5 segment restore has no cancellation member in
its snapshot ABI, so cancellation during that direct restore is not promised;
Mixtral and Qwen3.5/Qwen4 direct restore APIs also lack a generic cross-family
operator receipt interface. Those are explicit caller/API scope limitations,
not evidence of uniform family recovery closure. P06/P12 remain IN PROGRESS
and P15 remains OPEN. No tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P02 Qwen3.5 direct packed-provider factory restriction — 2026-10-16

The Qwen3.5 in-memory device factory and directory-backed device service now
reject `TorchQwen35Experts` before checkpoint reads or lazy executor creation.
The generic packed manifest exposes destination format/geometry but not source
dtype and the complete exact P02 tuple; the native-family CLI already rejects
this artifact class, while direct factory callers could previously bypass
that restriction. Original-publisher expert providers remain on their separate
family fingerprint path. This closes only the Qwen3.5 packed-manifest direct
factory seam; other family/backend/format tuples and route chains remain
candidate or unverified. P02/P15 remain OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P09/P10 selected-device row-INT8 effective-weight range preflight — 2026-10-16

The selected-device `int8-row` decoder now checks the actual per-row maximum
signed code times its persisted row scale against the FP32 weight range before
returning any buffers for device staging. This catches finite positive scales
that would otherwise overflow only when dequantized in the device projection,
without imposing a conservative global-scale bound on rows whose codes are
smaller. It closes only this selected-device row-format admission boundary;
other loader/consumer tuples, runtime placement, operation-level no-fallback,
and numerical parity remain unproven. P09/P10 remain IN PROGRESS and P15
remains OPEN. No tests, imports, builds, benchmarks, device probes, commits, or
pushes were run.

### P06/P12 distributed cancellation history agreement — 2026-10-15

When a distributed generation reports cancellation, each surviving rank now
includes the digest of its post-cancellation token history in the collective
result summary, in addition to context length. Same-length but divergent
logical rollback histories therefore fail result agreement and poison the
coordinator instead of being reported as a unanimous cancellation. This
checks logical token history only; it does not establish physical KV equality,
cover every distributed bank/consumer, or recover ranks lost during transport
failure. P06/P12 remain IN PROGRESS and P15 remains OPEN. Only
`git diff --check` was run; no tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P14 native publisher cancellation before buffered event dispatch — 2026-10-15

The native publisher stream now checks caller cancellation after dequeuing an
event and before yielding it. Previously cancellation was observed while
waiting on an empty queue and by the worker, but already-buffered token events
could still be delivered after cancellation. The media freeze path also accepts
the caller cancellation predicate and checks it between bounded ndarray row
copies; direct and checkpointed media completion/stream callers propagate that
predicate into generation and stream-worker cancellation. This closes only
host-side staging and queued-event publication races; the PIL conversion,
external processor, and in-flight native/device operation remain non-
interruptible. P04/P14 remain IN PROGRESS and P15 remains OPEN. No tests,
imports, builds, benchmarks, device probes, commits, or pushes were run.

### P02 GLM5 declared architecture restriction — 2026-10-14

`Glm5NextTextContract.parse` now rejects any supplied `architectures` field.
The specialized `glm5_next_text` contract previously treated that field as
ignorable metadata despite having no pinned Transformers class mapping. This
fail-closed restriction applies to every caller of the shared parser, including
device directory/in-memory factories and native GLM5 loading. It intentionally
does not claim architecture-bearing GLM5 checkpoint support; that requires a
reviewed exact class identity. Other family/backend/format and caller chains
remain open, so P02 remains IN PROGRESS and P15 OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P04 publisher processor interrupted-construction cleanup — 2026-10-15

`PublisherMediaProcessor` now treats post-load tokenizer binding and artifact
revalidation as part of processor construction ownership. If either fails, it
closes the just-created processor when that owner exposes `close()`, attaches a
cleanup failure to the primary construction error, clears partial instance
references, and re-raises the original failure. This closes only the shared
publisher processor's post-load construction unwind; it does not address
mid-call processor interruption or cross-family media/serving lifecycle
closure. P04/P14 remain IN PROGRESS and P15 OPEN. No tests, imports, builds,
benchmarks, or device probes were run.

### P02/P03 generic native reader family-planner gate — 2026-10-15

`read_native_text_checkpoint` and `load_native_text_decoder` now reject the
registered `deepseek_v4` and `kimi_linear` identities because they are not in
the complete resident-family planner set; registry membership alone no longer
admits these generic full-text read/load routes. Their dedicated fixture or
family loaders are unaffected. For planned families, the existing complete
resident inventory admission remains mandatory on full-model reads. Selected
range/edge reads retain their family-specific full-header selectors, since
their admitted source can exceed the selected-residency byte budget (for
example, streamed experts). This closes only those two generic route bypasses;
selected-loader auditing and complete P02/P03 family/caller reconciliation
remain open. P02 remains IN PROGRESS, P03 remains bounded-DONE, and P15 remains
OPEN. No tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P06/P12 distributed cache identity admission — 2026-10-15

The root-side distributed cache manifest admission now rejects embedded NULs
in both the shared model identity and each rank receipt identity, matching the
identity hygiene already enforced for distributed generation recovery. This
closes malformed identity admission before the collective cache-restore
request; it does not establish all-rank manifest pinning/consensus, operation
rollback, transport/process-loss recovery, or cross-family state ownership.
P06/P12 remain IN PROGRESS and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P02 OLMoE public CLI cleanup attempts are independent — 2026-10-15

The public OLMoE Segment/Edge CLI now attempts adapter cleanup even when
execution teardown fails, and preserves all cleanup failures as notes when a
generation error is already primary. With no primary error, the first teardown
failure is raised after every owned cleanup attempt, with later failures
attached. This closes one concrete public caller teardown gap; it does not
complete the OLMoE route or wider P02/P03 matrices. P02 remains IN PROGRESS,
P03 remains bounded-DONE, and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P05 speculative rollback boundary verification — 2026-10-15

`SpeculativeDecoder._rollback` now checks the target's exposed retained history
and next-logit evidence after restoring the pre-transaction target receipt.
If a restore callback returns without re-establishing that observable boundary,
the rollback is treated as failed and the decoder is quarantined instead of
allowing another speculative step from split physical/logical state. Targets
that expose no such evidence still rely on their restore contract. This is a
single speculative-target rollback guard; it does not close proposer state,
all accept/reject/bonus/EOS/context/cancellation callers, external sink
atomicity, distributed consensus, or cross-family recovery. P05 remains IN
PROGRESS, P06/P12 remain IN PROGRESS, and P15 remains OPEN. No tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

### P13 observation archive-to-run integrity binding — 2026-10-15

The whole-model runner now adds a canonical SHA-256 binding for each emitted
observation archive, joining the archive digest to the exact manifest digest
and execution-row/artifact-pin binding digest. This reduces the risk of
cross-run archive/report mixups. The binding is an unsigned integrity
association only: it does not certify adapter correctness, produce placement
evidence, or turn source obligations into runtime receipts. P13 remains IN
PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P03 Inkling direct MTP constructor schedule strictness — 2026-10-15

`InklingMTP` direct construction no longer normalizes false-y malformed
`mtp_local_layer_ids` values (for example `0` or an empty string) to an empty
schedule. The constructor now preserves the exact value for its existing
type/range validation, matching the pre-payload inventory planner and closing
this direct-construction validation bypass. It does not prove every MTP tensor
is consumed through every factory/CLI or close P03's family-wide consumer map.
P03 remains DONE only for bounded inventory contracts; P05/P15 remain open. No
tests, imports, builds, benchmarks, device probes, commits, or pushes were run.

### P05 speculative draft restore boundary verification — 2026-10-15

`DraftProposer.restore` now compares the restored model's exposed history and
next-logit vector with the requested draft receipt before publishing that
receipt into the proposer wrapper. A callback that returns successfully while
leaving observably divergent physical draft state now fails and quarantines
the proposer through its existing restore-failure path. Models exposing no
state evidence still rely on their restore contract; this does not complete
both-RNG behavior or every speculative serving edge/caller. P05 remains IN
PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P09 selected-device E8 decode preflight — 2026-10-15

The selected-device E8 decoder now scans the complete packed expert before
allocating any decoded projection, validating every codebook index, finite and
nonnegative scale, and the transformed FP32 weight vector for overflow. A
malformed late record therefore fails before partial decoded matrices are
allocated. This is format-input admission/cleanup hardening only; it does not
close E8 activation rotation for selected-device execution, family/backend
format coverage, conversion parity, or device-placement evidence. E8 remains
restricted for the Mixtral/Qwen3.5/Qwen4 selected-device expert providers.
P09/P10 remain IN PROGRESS and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P11 Qwen3.5 public run-record contract — 2026-10-15

The Qwen3.5 public Segment run callback now validates the exact `_Run` ABI
record size, rejects token and reserved fields because that wire carries hidden
activations only, requires exact input and output byte extents, and rejects
nonfinite activations before calling the stateful forward path. This closes
one family-specific caller/schema mismatch only; it does not complete the
family-wide ABI, head schema, continuation, registration, or service wiring
review. P11 remains IN PROGRESS and P15 remains OPEN. No tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

### P13 Published observation archive digest check — 2026-10-15

After atomically renaming the temporary NPZ archive into its published output
path, the whole-model runner now hashes the published path and requires it to
match the digest computed before rename. The report's observation receipt
association uses this post-publication digest. This detects a publication-path
substitution or mutation before receipt assembly; it is not a signature,
does not prevent later mutation, and does not establish adapter correctness,
placement, or runtime evidence. P13 remains IN PROGRESS and P15 remains OPEN.
No tests, imports, builds, benchmarks, device probes, commits, or pushes were
run.

### P09/P10/P15 selected-device low-bit padding cursor — 2026-10-16

The selected-device INT2/INT3 preflight now initializes the packed-code cursor
before validating tail padding in each projection. Previously, widths with
unused tail bits could raise `UnboundLocalError` before the intended padding
check, instead of deterministically admitting valid payloads or rejecting
noncanonical padding. This fixes that narrow source path only; no device
execution, format-consumer matrix, family admission, or runtime placement is
established. P09/P10 remain IN PROGRESS and P15 remains OPEN. No tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

### P03/P11 Qwen device projection-provider factory wiring — 2026-10-16

The in-memory Qwen3.5 and Qwen4 device-service factories now forward an
explicit `DevicePublisherProjections` owner into their decoder constructors.
The Qwen3.5 directory-backed service factory likewise forwards its optional
projection provider to `TorchQwen35Decoder.from_directory`. These decoder
paths already validate projection identity, device, and dense tensor binding;
the service factories had silently omitted that supported consumer, leaving
factory-created Qwen device services unable to use the persisted projection
route. This closes only that factory-wiring omission. It does not prove every
family tensor consumer, projection artifact lifecycle/receipt, ABI or CLI path,
numerical parity, or runtime device placement. P02/P03/P11 remain IN PROGRESS
and P15 remains OPEN. No tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P03 Qwen4 PLE prime metadata range and cost — 2026-10-16

Qwen4 PLE prime-lane generation now uses a deterministic Miller–Rabin test
over its signed-int64 metadata domain instead of trial division up to the
square root of every candidate. It rejects prime bases or generated offset
totals that cannot be represented by the persisted int64 metadata arrays,
before NumPy conversion or cumulative-offset overflow. This closes a bounded
metadata-generation and representation gap only; it does not complete Qwen4
inventory/consumer tracing, PLE parity evidence, or any backend/runtime route.
P03 remains bounded-DONE with family closure open and P15 stays OPEN. No tests,
imports, builds, benchmarks, device probes, commits, or pushes were run.

### P02/P14 GLM route and cleanup alignment — 2026-10-16

The GLM-DSA device source-dtype policy now matches its actual resident
checkpoint reader: F32/F16/BF16 are admitted, while E4M3 is not advertised
without a DSA scale-aware conversion consumer. The GLM5 publisher owner and
failed-construction cleanup now attempt decoder, expert-provider and
projection-provider teardown independently, retain each failed owner for a
retry, and annotate secondary failures without replacing the first cleanup
error. These are bounded admission and cleanup fixes; they do not complete the
GLM family tuple/caller matrix, architecture-bearing GLM5 restrictions, ABI
review, or runtime evidence. P02/P14 remain IN PROGRESS and P15 remains OPEN.
No tests, imports, builds, benchmarks, device probes, commits, or pushes were
run.

### P04/P14 publisher media stream ownership correction — 2026-10-15

The earlier lazy-staging behavior recorded under the 2026-10-14 entry has been
superseded. `PublisherProvider.stream_complete_media` now detaches validated
caller-owned frames before returning its iterator, so callers may safely reuse
or mutate their buffers after stream creation. The tradeoff is that an
unconsumed stream still incurs bounded media staging; cancellation is polled
during ndarray copies and continues into generation. This corrects the prior
ownership ambiguity only, not remaining serving lifecycle gaps. P04/P14 remain
IN PROGRESS and P15 OPEN. No tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P12 distributed generation recovery requires a concrete application identity — 2026-10-15

`RankGenerationRecoveryCheckpoint.admit` now rejects an empty or NUL-containing
application identity instead of letting an empty caller value disable the
manifest-to-application identity comparison. The pinned manifest still has to
match the operator SHA-256 and all rank receipts, and restore still verifies
the committed boundary. This closes only the missing-identity admission case;
it does not establish expected model identity for legacy cache checkpoints or
close distributed process-loss/consensus behavior. P12 remains IN PROGRESS and
P15 remains OPEN. No tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P02 OLMoE public CLI manifest identity binding — 2026-10-15

The public Segment/Edge OLMoE CLI previously accepted any syntactically valid
64-character checkpoint identity, even though its adapter opened the expert
manifest separately. It now parses the bounded manifest once, computes the
SHA-256 of its canonical serialized contents, requires the supplied identity
to match, and passes that same parsed manifest object to the adapter. This
binds the caller's route identity to the manifest actually used by the
adapter. It does not close the wider OLMoE loader/backend/ABI/gateway/recovery
tuple or prove the checkpoint shards beyond the manifest fingerprint. P02
remains IN PROGRESS and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P14 codebase run late-cancel terminal guard — 2026-10-16

`CodebaseService.cancel_run()` now applies cancellation with one conditional
database update restricted to nonterminal states while holding the run-event
ownership lock. If completion/failure wins between the caller's initial read
and the update, the cancel call returns the current run without changing its
terminal status, completion time, cancellation event, or run event log. This
closes late-cancel mutation for completed/failed/cancelled/blocked runs; it
does not make all run-result publication and cancellation transitions one
transaction, nor interrupt an active tool/process immediately. P14 remains IN
PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P11 OLMoE public session operation serialization — 2026-10-16

Native OLMoE Segment sessions now have a per-handle operation guard. Run
position admission and the forward execute under that guard; snapshot, restore,
and destroy/close serialize against the same active operation, and same-thread
callback reentry is rejected. This prevents concurrent calls from both
admitting the same stale position and racing session mutation. It closes only
the OLMoE Python Segment-handle ownership seam; other family callbacks and
distributed/caller-level ordering remain open. P11 remains IN PROGRESS and
P15 remains OPEN. No tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P05 separate stream and checkpoint sinks explicitly restricted — 2026-10-15

The fresh V4.1 request path and resumed-generation path reject DSpark when a
caller supplies both a token-event callback and a separate checkpoint sink.
Checkpoint publication commits the speculative boundary before callback
delivery; a callback can fail after commit, while reversing the order would
expose tokens before a failed durable write. The current API has no shared
atomic publication owner, so this combination cannot claim exactly-once
publication. Non-streaming checkpoint use and streaming without a checkpoint
sink remain admitted. This is an enforced restriction, not atomic stream
semantics; other P05 acceptance/rejection/bonus/EOS/context/cancellation and
caller-closure work remains open. P05 remains IN PROGRESS and P15 OPEN. No
tests, imports, builds, benchmarks, device probes, commits, or pushes were run.

### P05 public DSpark factory cleanup ownership — 2026-10-15

`publisher_v41_dspark_session` now registers its created MTP draft with the
owning `ExitStack` before constructing the speculative decoder, then registers
`decoder.close()` after successful construction. LIFO cleanup therefore
attempts decoder/proposer/backbone cleanup before the target model and
arithmetic are closed; if decoder construction itself fails, the draft is
still closed by the already registered fallback. This addresses that public
factory's missed teardown path only; it does not establish speculative
acceptance/rejection parity or cleanup for every custom proposer/caller. P05
remains IN PROGRESS and P15 OPEN. No tests, imports, builds, benchmarks,
device probes, commits, or pushes were run.

### P11 GLM-DSA Segment activation ABI extent — 2026-10-15

The GLM-DSA public Segment run callback now requires input and output byte
extents to equal the declared `rows * hidden_size * sizeof(float32)` tensor
size exactly. Previously, oversized buffers were accepted as if this fixed
shape ABI were a capacity-based interface, which could hide a caller/schema
mismatch. This narrows only the GLM-DSA activation transfer contract; the
family-wide ABI symbol/range/schema/callback and service wiring review remains
open. P11 remains IN PROGRESS and P15 OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P12 durable continuation orphan-owner cleanup — 2026-10-15

`DurableContinuationStore.recover()` now records the service-session registry
delta while holding its admission lock. If a custom cache recoverer raises
after publishing a session, returns a pre-existing/unknown handle, or creates
multiple sessions, recovery rejects the result and attempts cleanup only for
the newly-added registry entries proven to belong to that call. Cleanup errors
are attached to the primary recovery failure. This does not change existing
owner rollback semantics in `recover_into()` or close cross-family recovery
and distributed-consensus gaps. P12 remains IN PROGRESS and P15 remains OPEN.
No tests, imports, builds, benchmarks, device probes, commits, or pushes were
run.

### P14 vision-stream upstream cleanup preserves primary failures — 2026-10-15

`ModelGateway.stream_complete_vision` now routes the provider iterator through
the existing `_closing_provider_stream` owner. On normal completion, closing
failures still surface; when provider iteration or downstream consumption is
already failing, upstream close errors are attached as notes instead of
masking that primary failure. Closing the gateway vision iterator also
deterministically closes the provider iterator. This fixes one vision caller's
cleanup/error-ordering path only; other streaming, stop, recovery, backpressure,
and serving-caller lifecycle gaps remain open. P14 remains IN PROGRESS and P15
OPEN. No tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P03 Mixtral tied-head metadata admission — 2026-10-15

The exact Mixtral resident inventory planner now rejects a declared tied
embedding/output head when its source shape or dtype differs from the input
embedding, before checkpoint payload materialization. If the tied output head
is omitted, the existing runtime aliases it to the input embedding; if it is
present, the decoder still checks actual value equality after load. This closes
one metadata-level inventory bypass only and does not close family-wide
optional/shared/MTP/expert/media consumer reconciliation or prove every
factory/CLI uses its planner. P03 remains bounded-DONE with family closure
open, P02 remains IN PROGRESS, and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P14 vision-stream consumer-close error ordering — 2026-10-15

The outer `ModelGateway.stream_complete_vision` iterator now preserves
`GeneratorExit`, cancellation, and provider/consumer failures if its explicit
upstream close also fails. Normal-completion close failures still surface.
This prevents cleanup during consumer disconnect from replacing the control
flow or primary exception; it does not provide device-side cancellation or
close other gateway callers. P14 remains IN PROGRESS and P15 remains OPEN.
No tests, imports, builds, benchmarks, device probes, commits, or pushes were
run.

### P09 IQ3 all-block metadata preflight — 2026-10-15

`decode_iq3` now validates every block's FP16 super-scale and reserved high
bit before allocating its decoded matrix. Negative scales and malformed late
blocks therefore fail before partial output allocation. This closes one
format-consumer admission seam only; it does not establish all-family loader /
execution parity, IQ3 activation-rotation closure, or runtime placement and
evidence. P09/P10 remain IN PROGRESS and P15 remains OPEN. Only
`git diff --check` was run; no tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P11 Mixtral Segment fixed activation wire admission — 2026-10-15

The Mixtral Segment run callback now requires the exact public `_Run` record
size, rejects token/count and reserved fields because this callback consumes
hidden activations only, and requires input/output byte lengths to equal the
declared fixed activation tensor extent. Previously it accepted oversized
buffers as capacities, unlike the shared Segment fixed-tensor contract. This
closes one Mixtral ABI boundary only; family-wide ABI and caller wiring review
remains open. P11 remains IN PROGRESS and P15 remains OPEN. Only
`git diff --check` was run; no tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P11 Qwen4 Segment fixed activation wire admission — 2026-10-15

The Qwen4 Segment run callback now requires the exact public `_Run` record
size, zero reserved fields, the exact PLE token-side row count, and exact
input/output activation byte extents. The prior callback accepted oversized
buffers as capacity-based even though neighboring fixed-shape family ABIs use
exact tensor extents. This narrows only the Qwen4 Segment transfer contract;
family-wide ABI/schema/callback/registration review remains open. P11 and P15
remain OPEN. No tests, imports, builds, benchmarks, device probes, commits, or
pushes were run.

### P11/P12 publisher Segment serialized position admission — 2026-10-15

The publisher Segment run callback now checks `_Run` record size and zero
reserved fields, and rechecks the request position only after acquiring the
session operation lease. Concurrent calls can no longer both validate one
old position and then advance sequentially using stale continuation metadata.
This closes one publisher callback boundary only; complete ABI and recovery
ownership reviews remain open. P11/P12/P15 remain IN PROGRESS/OPEN. No tests,
imports, builds, benchmarks, device probes, commits, or pushes were run.

### P11/P12 Qwen4 Segment operation ownership serialization — 2026-10-15

Qwen4 Segment run, snapshot, restore, and close now use the session's shared
operation guard. Concurrent teardown waits for the active operation, and
same-thread callback reentry into another session operation is rejected; the
Qwen4 close path remains idempotent after successful teardown. This closes
one Qwen4 public-session ordering gap only; stale/foreign restores and
cross-family physical-state/recovery consensus remain open. P11/P12 and P15
remain IN PROGRESS/OPEN. No tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P03 Mixtral streamed loader inventory-plan enforcement — 2026-10-15

The Mixtral streamed checkpoint loader now consults the shared family inventory
planner before selecting tensors and rejects a fused/split layout that disagrees
with the plan. This closes that loader's layout-to-inventory consistency gap;
it does not establish consumer links for every optional/shared/MTP/expert/media
tensor or cover other families and entrypoints. P03 and P15 remain open.

### P11/P12 GLM-DSA and Qwen3.5 public session boundary hardening — 2026-10-15

GLM-DSA now enforces the exact public run-record size, rejects token and
reserved fields for its hidden-activation protocol, and rejects non-finite
activations before advancing. Qwen3.5 now serializes run/snapshot/restore/close
with a per-session operation guard and rejects same-thread reentry, preventing
cache and close operations from racing model mutation. Broader family ABI and
physical recovery reviews remain open; P11/P12/P15 are not closed. No tests,
imports, builds, benchmarks, device probes, commits, or pushes were run.

### P02 K3 FLA candidate route restriction — 2026-10-15

The source catalog classifies K3 FLA backend tuples as candidate-unverified:
the presence of `FLAK3Mixer` does not bind source dtype, BF16 arithmetic,
continuation/cache ownership, serving callers, and tooling into one exact P02
tuple. The catalog now marks `('kimi-k3', 'fla')` restricted, and the shared
`K3Execution` constructor rejects `FLAK3Policy` so direct factories and other
constructor callers cannot bypass that decision. K3 native routes without
FLA and the separately restricted Metal route are unchanged. This is an
enforced limitation, not FLA completion; P02/P07 remain IN PROGRESS and P15
remains OPEN. Only `git diff --check` was run; no tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P14 checkpointed stream media ownership — 2026-10-15

`NativePublisherProvider.stream_complete_checkpointed` contained `yield from`,
so Python deferred the function body—including decoded-media detachment—until
the caller's first iterator pull. There are no in-repository call sites, so no
caller-side stable-buffer guarantee could be confirmed. The method now stages
and detaches media before returning the stream iterator, matching
`stream_complete_media` ownership semantics; cancellation remains checked
during the bounded copy and is forwarded into generation. This may stage media
for an iterator that is never consumed, but avoids binding an asynchronous
request to caller-mutable buffers. Other media callers and full stream/stop/
recovery lifecycle parity remain open. P14/P15 remain IN PROGRESS/OPEN. Only
`git diff --check` was run; no tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P14 model-gateway JSON/SSE parser cancellation cleanup — 2026-10-15

The JSON-lines and SSE parsers now explicitly close their underlying
`_stream_model_lines` iterator on parse failure, consumer disconnect, and
normal early termination (including the SSE `[DONE]` marker). This makes the
existing cancellation event and active HTTP-read cleanup deterministic rather
than relying on generator finalization/garbage collection. Cleanup failures
are surfaced on normal completion or attached to the primary parser/consumer
failure. This closes only the model-gateway parser-to-egress cleanup seam; it
does not establish device-side cancellation or caller-wide stream/stop/recovery
parity. P14 remains IN PROGRESS and P15 remains OPEN. Only `git diff --check`
was run; no tests, imports, builds, benchmarks, device probes, commits, or
pushes were run.

### P06/P12 distributed recovery identity admission — 2026-10-15

Distributed cache recovery now rejects embedded NULs in the manifest model
identity and every rank receipt identity, matching the service-side cache
restore admission. Distributed generation checkpoint creation, manifest
admission, and direct rank restore likewise reject NUL-bearing execution/rank
identities. This prevents malformed identity strings from reaching recovery
comparison and worker restore paths; it does not close process-loss recovery,
all-bank consensus, or cross-family physical-state ownership. P06/P12 remain
IN PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P14 OLMoE JSONL event-consumer failure cancellation — 2026-10-15

`JsonlServer._events()` now cancels its request handle if delivering an event
to the JSONL output fails, so a closed/broken stdout consumer does not leave
the decoder generating an undeliverable stream. Cancellation cleanup failures
are attached to the original output failure, which remains visible to the
reader thread. This closes one local JSONL producer/consumer disconnect seam;
it does not establish network-serving cancellation, all-caller backpressure,
device interruption, or recovery parity. P14 remains IN PROGRESS and P15
remains OPEN. Only `git diff --check` was run; no tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P03 Mixtral streamed-loader planner binding — 2026-10-15

`NativeMixtralDecoder.from_streamed_directory` now routes its complete source
inventory through `admit_mixtral_inventory` before selecting dense tensors for
materialization, and checks that the loader's fused/split selection agrees
with the shared planner. This prevents the streamed loader's duplicated local
shape/dtype checks from silently diverging from the family inventory contract
(including tied-head rules). It closes one Mixtral factory/planner bypass;
all-family producer-to-consumer closure and verification that every factory
and CLI uses its correct family planner remain open. P03 remains DONE only for
bounded inventory contracts, and P15 remains OPEN. Only `git diff --check` was
run; no tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P09/P10 selected-device Q4 block metadata preflight — 2026-10-15

Selected-device Q4_0/Q4_1 admission preflights the full encoded extent and
requires block metadata to be finite before allocating decoded projection
matrices. Signed finite Q4_0 deltas and signed finite Q4_1 deltas/minima remain
accepted, as permitted by these formats. It does not close other format
loader/consumer pairs, runtime placement evidence, or device-wide no-fallback
proof. P09/P10 remain IN PROGRESS and P15 remains OPEN. No tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

Correction to the earlier P09/P10 summary rows and historical Q4 claim: signed
finite Q4_0 deltas and Q4_1 deltas/minima are permitted by the format contract.
The source now preflights finiteness and exact extent only; it does not reject
negative Q4 metadata. This correction supersedes earlier wording that described
negative Q4 deltas as invalid. Q5 negative-delta admission remains a separate
format-specific contract. P09/P10 and P15 remain open.

### P05 speculative target prefill boundary verification — 2026-10-15

`SpeculativeDecoder.prefill()` now checks exposed target history and retained
logits against the completed prompt boundary before publishing decoder history
and next-logits state. A target that returns finite prompt logits while
retaining a different observable physical boundary is rejected and enters the
existing prefill rollback path. This closes the prefill analogue of the
successful-step boundary check only; opaque target state, rollback evidence,
caller-wide cancellation/restore behavior, and atomic stream publication remain
open. P05 remains IN PROGRESS and P15 remains OPEN. Only `git diff --check` was
run; no tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P06/P12 distributed manifest-resume rollback ownership — 2026-10-15

Manifest-backed `DistributedExecutionService.resume_generation` now captures a
rank-local service snapshot before any rank restores its generation frame and
keeps it through rank agreement and continuation. On rank failure or result
disagreement, the existing restore cleanup path rolls back from that snapshot,
retires it independently, and tears down the affected session because the
coordinator is poisoned. Previously only the explicit `restore_generation`
action retained such a snapshot; a manifest-backed resume could restore one
rank successfully while another rejected its frame, leaving the surviving
rank at a partially installed generation boundary. This closes that
manifest-resume recovery ordering gap only. It does not establish consensus
under process/transport loss, close failures in opaque executor rollback, or
prove family MTP/auxiliary physical-state ownership. P06/P12 remain IN PROGRESS
and P15 remains OPEN. No tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P13 finalized report-to-observation integrity binding — 2026-10-15

The whole-model validation runner now emits an unsigned report binding when
an observation archive is published. The binding hashes the finalized report
payload (excluding the binding field itself) and joins that digest to the
archive SHA-256, exact manifest digest, and execution-row/artifact-pin binding;
`validate_report_receipt_binding` can detect accidental cross-run report or
archive substitution. This is an integrity association only: it is not signed,
does not attest adapter correctness, does not establish device placement, and
does not create runtime evidence. P13 remains IN PROGRESS and P15 remains
OPEN. Actual pinned observations, complete physical-cache receipts, per-op
placement receipts, and validated comparisons remain outstanding. Only scoped
`git diff --check` was run; no tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P04 Qwen3.5 candidate decoder failure cleanup — 2026-10-15

The Qwen3.5 image path executes against a private decoder candidate and
publishes it only after the multimodal prefill succeeds. If candidate forward
fails after allocating partial cache state, the candidate is now closed while
the still-current base decoder remains untouched; cleanup errors are attached
to the primary forward error. This closes only candidate-cache cleanup for
that image-prefill path. Broader media ordering/geometry, other family owners,
and caller lifecycle remain open. P04 remains IN PROGRESS and P15 remains
OPEN. No tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P14 compatibility endpoint queue-control restriction — 2026-10-16

The OpenAI-compatible `/v1/chat/completions` and `/v1/completions` routes and
Anthropic-compatible `/v1/messages` route now explicitly reject request-body
queue controls `queue`, `queue_timeout`, `wait_timeout`, `wait_for_slot`, and
`priority` with an unsupported-feature response. These endpoints do not route
through an admission scheduler and do not emit queue-wait headers or define a
429/Retry-After queue policy; scheduler wiring was not inferred from unrelated
native runtime gauges. The capability report records the exact restriction.
This does not implement HTTP queue admission, timeout or capacity responses,
wait headers, or queue telemetry. P14 remains IN PROGRESS and P15 remains OPEN.
No tests, imports, builds, benchmarks, device probes, commits, or pushes were
run.

### P14 bounded HTTP request admission and queue telemetry — 2026-10-17

Replaced the prior all-controls-rejected restriction on the local non-streaming
OpenAI/Anthropic text compatibility endpoints with a bounded native-provider
admission contract. Requests default to immediate admission; overload returns
HTTP 429 with `Retry-After: 1`. Callers can opt into FIFO waiting with
`queue=true` and an optional `queue_timeout` (default 30 seconds, capped at 300).
The provider allows one active native session request and at most 64 waiting
requests. Cancellation removes a waiter; close wakes queued requests. Successful
compatibility responses include `X-Queue-Wait-Ms`. `/health` exposes the
provider-owned request-queue snapshot and `/metrics` exports active/waiting,
admission/rejection/timeout/cancellation counters, cumulative wait, and a wait
histogram. `priority`, `wait_timeout`, and `wait_for_slot` remain rejected/not
admitted; streaming is still explicitly restricted. Capability and source-scope
catalog entries now describe this partial implementation.

Focused scheduler and inference-route tests pass (9 tests). This closes only
single-process local text request admission and its basic observability. It does
not implement Colibri's complete queue contract, streaming/backpressure,
priority scheduling, cross-process coordination, or execution/transport-loss
recovery. P14 remains IN PROGRESS and P15 remains OPEN.

### P14 OLMoE submit-acknowledgement failure cancels unobserved request — 2026-10-15

`JsonlServer.dispatch()` now cancels a submitted request and removes its
unstarted event-reader entry if writing/flushing the submit acknowledgement
fails, or if starting the reader fails. This preserves response-before-event
ordering while preventing a request from generating without an attached event
consumer. Cancellation cleanup failure is attached to the original dispatch
failure. This closes one JSONL submit/ack handoff seam only; broader
caller-by-caller stop, backpressure, recovery, warm-session, and media lifecycle
coverage remains open. P14 remains IN PROGRESS and P15 remains OPEN. Only
scoped `git diff --check` is permitted; no tests, imports, builds, benchmarks,
device probes, commits, or pushes were run.

### P14 OLMoE token publication ordered against cancellation — 2026-10-16

`OlmoeService._execute()` now serializes token event publication with request
cancellation under the service lifecycle condition. If cancellation or the
request deadline is observed before publication acquires that boundary, the
token is not appended to the request's visible event stream or result buffer;
after cancellation is recorded, no later token can overtake its cancellation
event. Tokens already published remain observable and are not retractable, and
decoder interruption remains cooperative. This closes one JSONL/native service
token-versus-cancel ordering seam only; streaming/backpressure, stop, recovery,
warm-session, media, and other serving callers still require review. P14 stays
IN PROGRESS and P15 OPEN. No tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P12 distributed recovery checkpoint key admission — 2026-10-16

Both cache-backed and generation-backed distributed recovery checkpoint
descriptors now reject embedded NUL characters in their logical checkpoint
keys, in addition to the existing nonempty/length bounds. This prevents an
ambiguous key from entering the recovered-session mapping and makes the two
checkpoint admission paths consistent with the manifest identity checks.
This closes only descriptor-key validation; it does not resolve worker loss,
transport stranding, operation-level collective failure, or family physical
state consensus. P06/P12 remain IN PROGRESS and P15 remains OPEN. No tests,
imports, builds, benchmarks, device probes, commits, or pushes were run.

### P02/P03 Mixtral declared source-dtype consistency — 2026-10-16

The Mixtral whole-inventory planner now binds tensor-header dtypes to an
explicit `dtype` or `torch_dtype` declaration when the checkpoint supplies
one; conflicting `dtype`/`torch_dtype` fields, non-string declarations, and
unsupported declared types fail admission. It still permits mixed supported
floating dtypes when neither field is declared, preserving checkpoints whose
config omits source-dtype provenance. This prevents selected-payload conversion
from silently accepting headers that contradict declared checkpoint metadata.
It does not close Mixtral factory bypass review or the family/backend/format
loader-to-tooling chain; P02 stays IN PROGRESS, P03 remains bounded-DONE with
family closure open, and P15 stays OPEN. No tests, imports, builds, benchmarks,
device probes, commits, or pushes were run.

### P09/P10 selected-device INT2/INT3 projection preflight — 2026-10-16

Selected-device low-bit decoding now validates scale finiteness/positivity,
worst-case signed-code FP32 representability, and packed row padding for all
three projections before allocating any decoded projection. A malformed late
projection can no longer leave earlier decoded matrices resident during
admission. This closes only the selected INT2/INT3 pre-allocation validation
gap; it does not establish full format/backend consumer coverage or runtime
device placement. P09/P10 remain IN PROGRESS and P15 remains OPEN. No tests,
imports, builds, benchmarks, or device probes were run.

### P02/P03 Qwen3.5 selected-range dtype admission — 2026-10-16

The Qwen3.5 native range/edge directory planner now validates each source
tensor header's dtype as F32, F16, or BF16 while checking the complete tensor
inventory, before selecting or materializing payloads. Previously it checked
names and shapes only, allowing an unsupported persisted encoding to reach
conversion. This closes that header-admission gap only; Qwen3.5 factory
bypass review, all-family consumer closure, and runtime backend evidence remain
open. P02 stays IN PROGRESS, P03 remains bounded-DONE with family closure
open, and P15 stays OPEN. No tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P03/P04/P09 V4.1 borrowed-array extent admission — 2026-10-16

The DeepSeek V4.1 publisher text row, MTP activation, and vision tensor
adapters now inspect borrowed provider/patch array extents and floating dtype
before making detached FP32 copies, validate finiteness before conversion,
and reject values that overflow FP32. Vision image-span type metadata is read
without an eager copy; existing prompt-span bounds then constrain its use.
This closes only a pre-allocation input validation/resource gap in these
adapters. It does not establish publisher numerical parity, complete image
ordering/position semantics, family inventory-to-consumer closure, or runtime
backend/device evidence. P03/P04/P09 remain IN PROGRESS and P15 remains OPEN.
No tests, imports, builds, benchmarks, device probes, commits, or pushes were
run.

### P04 GLM5 multimodal context pre-admission — 2026-10-16

`Glm5MultimodalDecoder.forward` now checks that the expanded token sequence
fits the text decoder's remaining context before embedding lookup or vision
execution. Previously an over-capacity media request could run the full image
or video tower before the text decoder rejected the sequence. The text decoder
still validates its own mutation boundary. This closes only the early resource
and ordering failure for this GLM5 owner; media geometry/order parity, other
callers, temporal semantics, and interruption behavior remain open. P04 remains
IN PROGRESS and P15 remains OPEN. Only scoped `git diff --check` was run; no
tests, imports, builds, benchmarks, device probes, commits, or pushes were run.

### P13 command-adapter evidence publication boundary — 2026-10-16

`platform_certification/command_adapter.py` now publishes a forward placement
receipt only after its referenced NPZ is confined to the fresh session
directory, matches the declared digest, and contains exactly the configured
logit/cache/evidence arrays. Previously the adapter appended runtime placement
evidence before validating the observation artifact, so a failed forward could
leave a success-shaped evidence row in adapter metadata. This aligns source
receipt publication with the completed observation boundary; it does not
authenticate operator-supplied runner claims, produce actual execution
evidence, or close family/backend comparison bindings. P13 remains IN PROGRESS
and P15 remains OPEN. Only scoped `git diff --check` was run; no tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

### P15 targeted family admission reconciliation — 2026-10-16

Source review of Qwen3.5, Qwen4-Exp, GLM5-Next, and GLM-MoE-DSA found
enforced, bounded admission in their current direct/device text paths:
Qwen3.5 and Qwen4 bind the exact `model_type`, reject unsupported declared
architectures, and plan complete tensor inventories before directory payload
selection; Qwen3.5 range/edge inventory also checks source dtype. GLM5 text
uses `Glm5NextTextContract` and an exact canonical tensor-layout selector,
rejecting declared architectures pending a pinned mapping. GLM-DSA rejects
declared architectures and its device directory planner requires exact tensor
names/shapes and F32/F16/BF16 headers. These are code-backed restrictions for
the inspected entrypoints, not proof that every factory, CLI, gateway,
recovery, ABI, or tooling route uses them. The targeted review found no
additional source-only patch that could honestly close the remaining rows.
Family/backend/format route tracing, optional-bank and media consumer closure,
cross-caller lifecycle/recovery, family ABI wiring, numerical parity, and
actual P13 runtime evidence remain unverified or partial. Therefore P02-P14
remain open where already marked open, and P15 remains OPEN; no completion is
claimed. This review ran no tests, imports, builds, benchmarks, device probes,
commits, or pushes.

### P12/P15 V4.1 private checkpoint cancellation boundary — 2026-10-16

The private `V41Execution` checkpoint loader previously called a nonexistent
`model._check_cancelled()` after reading and hashing the checkpoint, so a
valid private checkpoint could fail at the final validation boundary before
restore. `load_v41_checkpoint` now accepts an optional callable cancellation
predicate, checks it while reading bounded records and immediately before
model mutation, and raises `InterruptedError` without publishing staged state
when cancellation is requested. Existing callers remain source-compatible.
This closes only that private V4.1 checkpoint caller's missing cancellation
hook and removes the invalid method call; publisher checkpoints and distributed
recovery have separate owners and remain open. P12/P15 remain IN PROGRESS/OPEN.
No tests, imports, builds, benchmarks, device probes, commits, or pushes were
run.

### P11/P15 GLM final-hidden public activation finite boundary — 2026-10-16

`NativeFamilyPublicAdapters.final_hidden` now rejects non-finite FP32 stream
activations before native dispatch and rejects non-finite native results before
returning them to the public caller. This prevents NaN/Inf values from being
treated as a valid fixed-shape hidden-state response at this GLM ABI boundary.
It does not close the remaining family ABI symbol/schema/callback map, service
wiring, recovery ownership, numerical parity, or runtime evidence. P11/P12
remain IN PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks,
device probes, commits, or pushes were run.

### P02/P03 direct-factory route audit — 2026-10-16

Read-only source tracing revisited the direct text/device factory boundaries in
`execution_factories.py` and their family contract/planner owners. The inspected
routes delegate configuration identity and tensor layout to family-specific
owners: Qwen3.5/Qwen4 device factories use their independent inventory
contracts; GLM5 device and publisher services use `Glm5NextTextContract` or
`component_contract`; GLM-DSA device construction uses
`IndependentGlmDsaDecoder` and its directory reader; the conditional publisher
service checks exact `model_type` before building the component contract; and
flat-native services restrict the accepted family set and compare the local
config identity. `admit_resident_inventory` independently binds each supported
resident planner to its exact config identity and checks complete names, shape,
dtype, and resident conversion budget (with Mixtral delegated to its dedicated
whole-inventory planner).

This trace does **not** show that every CLI, gateway, recovery, public ABI, or
tooling caller traverses the same factory/planner; nor does it bind every such
call to the complete ordered P02 tuple. The generated source matrix remains
candidate-unverified, and no route becomes verified-admitted from this source
review. Optional/shared/MTP/expert/media inventory-to-consumer links also remain
family- and caller-specific and incomplete. No safe blanket guard was added:
the exact-row admission catalog currently has no verified-admitted row, so
applying it indiscriminately at all existing factories would disable all
catalogued routes rather than selectively restricting only proven-incomplete
tuples. P02/P03 remain partial and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P05 speculative transaction source audit — 2026-10-16

Re-read `v41_execution/speculative.py` from proposal acquisition through
commit/rollback, snapshot restore, iterator cancellation, and reset. The
ordinary step path captures target and proposer receipts plus verifier and
proposer-owned RNG states before proposal; rejection/early stop restores the
target and draft boundary and rewinds a separately exposed proposer RNG; a
failure restores all owners independently and quarantines on rollback errors.
Committed history/logits are published only after the target boundary check,
and cancellation is checked before publication. This pass found no additional
narrow defect that could be safely fixed from source evidence alone, so no
speculative behavior was changed.

This bounded review is not proof of accept/reject/bonus/EOS/context equivalence
for every serving caller. Opaque custom proposers still rely on their paired
snapshot/restore contract when they do not expose independently checkable
physical history/logits; durable sinks cannot be assumed atomic; and caller,
distributed, and recovery coverage remain incomplete. P05 remains IN PROGRESS
and P15 remains OPEN. No tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P05 close refuses live public snapshot receipts — 2026-10-16

`SpeculativeDecoder.close()` now checks for outstanding decoder-issued
snapshot capabilities before closing the target or proposer. A caller must
release those receipts and retry teardown; this prevents a successful close
from silently invalidating a still-restorable continuation boundary or
leaving owner-side pinned snapshot resources unreachable. The check occurs
before either physical owner is touched, so close remains retryable. This is a
bounded resource-lifecycle safeguard, not cross-caller speculative parity:
opaque proposer restore contracts, durable sink atomicity, and comprehensive
accept/reject/bonus/EOS/context/cancellation coverage remain open. P05 remains
IN PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks,
device probes, commits, or pushes were run.

### P11 Qwen4 public hidden activation finite/cancellation ABI — 2026-10-16

The registered Qwen4 public hidden-state entrypoint now rejects non-finite
input activations before invoking its borrowed kernel and scans staged outputs
for non-finite values before publishing them to caller memory. Its cancellation
callback is wrapped to require the ABI's strict integer 0/1 contract; an
out-of-contract result forces failure and leaves the caller output unchanged.
The existing private staging buffer preserves transactional publication.

This closes one Qwen4 public hidden ABI input/output/callback boundary only.
It does not establish the full family-wide symbol/schema/range map, all callback
ownership and reentry rules, service/factory/CLI wiring, or runtime numerical
behavior. P11 remains IN PROGRESS and P15 remains OPEN. No tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

### P09/P10 Q8_0 converter failure cleanup — 2026-10-16

The Q8_0 manifest converter now preserves its original packing, write, fsync,
validation or publication exception if temporary-file removal also fails;
cleanup failure is attached as a diagnostic note. After successful hard-link
publication, an unlink failure is still surfaced. This closes only Q8_0
converter cleanup. Grouped-format converter cleanup, family/backend
loader-to-consumer coverage, device placement/no-fallback and runtime evidence
remain open. P09/P10 remain IN PROGRESS and P15 remains OPEN. No tests,
imports, builds, benchmarks, device probes, commits, or pushes were run.

### P05 cancellation check before custom proposer dispatch — 2026-10-16

`SpeculativeDecoder.step()` now rechecks cancellation immediately after
capturing target/draft and RNG rollback receipts, before calling the proposer.
This avoids launching an expensive custom proposal after cancellation arrived
during receipt acquisition, including proposers that do not expose their own
cancellation scope. The existing transaction exception path restores both
owners and RNG state and quarantines if rollback fails. This does not establish
cross-caller accept/reject/bonus/EOS/context equivalence or atomic output-sink
publication; those remain open. P05 remains IN PROGRESS and P15 remains OPEN.
No tests, imports, builds, benchmarks, device probes, commits, or pushes were
run.

### P05 issued snapshot receipt mutation guard — 2026-10-16

`SpeculativeSnapshot` is a frozen dataclass but its instance dictionary and
nested RNG state dictionaries remain mutable. The decoder now keeps a private
seal for each issued snapshot, validates its immutable metadata and both RNG
receipts before restore, and releases the original target/draft receipts even
if a caller rewrites public receipt fields. This prevents mutating an exact
issued capability in place to restore unrelated state or leak the original
owner receipts. It does not authenticate opaque mutable state inside
owner-provided target/draft receipts, prove caller-wide speculative parity, or
provide atomic durable-sink publication. P05 remains IN PROGRESS and P15
remains OPEN. No tests, imports, builds, benchmarks, device probes, commits,
or pushes were run.

### P05 durable speculative restore rewinds proposer RNG on failure — 2026-10-16

The compound DSpark cache restore path captured the decoder RNG before
publishing restored target/draft caches, but its exception rollback omitted
the separate proposer RNG. A failure after assigning either saved RNG could
therefore leave the proposer random stream at the rejected cache boundary
while target/draft state rolled back. The restore transaction now captures the
old proposer RNG state before mutation and passes it through the shared
rollback routine, which attempts to restore both streams independently and
quarantines on rollback failure. This closes that DSpark cache-restore RNG
rollback path only; caller-wide acceptance semantics, opaque proposer state,
and durable output-sink atomicity remain open. P05 remains IN PROGRESS and P15
remains OPEN. No tests, imports, builds, benchmarks, device probes, commits,
or pushes were run.

### P14 research SSE completion terminal recognition — 2026-10-16

The research lifecycle event source emits `type="complete"` when its durable
session enters a terminal state. The shared runtime SSE writer now recognizes
that marker as terminal, alongside codebase `stream.closed` and generic
`done`/`error`/`terminal` markers, so it does not append a second synthetic
`done` event after the research completion event. This closes one research SSE
terminal-protocol mismatch; research heartbeat timeout still ends its iterator
without a terminal marker and therefore retains the generic `done` fallback.
Other route-specific terminal, cancellation, backpressure, recovery, and media
semantics remain open. P14 remains IN PROGRESS and P15 remains OPEN. No tests,
imports, builds, benchmarks, device probes, commits, or pushes were run.

### P13 command-adapter consume-the-captured-artifact boundary — 2026-10-16

`CommandSession.forward` now opens the audit-captured NPZ after checking its
digest, rather than reopening the runner-owned original path. Previously a
replacement between digest verification and `np.load` could make the arrays
used by the comparison differ from the bytes copied and recorded as the
observation receipt. This closes that local artifact TOCTOU gap only; it does
not authenticate the operator-supplied runner, establish numerical correctness,
or provide actual pinned runtime observations, cache/placement receipts, and
comparisons. P13 remains IN PROGRESS and P15 remains OPEN. Only scoped
`git diff --check` was run; no tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P02 direct local V4.1 factory admission — 2026-10-16

`open_publisher_v41_service` now requires exact source-matrix admission for
the native or device BF16/FP8/FP4 bank tuples before resolving the checkpoint
directory, opening artifacts, reserving resources, or constructing providers.
The distributed path already applied the analogous per-enabled-bank check.
The current matrix marks these local rows candidate-unverified, so direct API
callers now fail closed instead of bypassing the CLI's distributed-only gate.
This intentionally restricts local V4.1 execution until exact rows gain
reviewed admission; it does not implement an alternate loader path or close
other families' factory/CLI/gateway/recovery/tooling seams. P02 and P15 remain
OPEN. No tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P06/P12 tensor-parallel agreement and interrupt boundary — 2026-10-16

`TensorParallelProjections._agree` now validates every gathered control record
before indexing or comparing it. A malformed peer record quarantines the bank
instead of raising while leaving it apparently reusable. Registration and
linear/grouped-linear preflight now capture `BaseException` long enough to
enter the rank agreement boundary; `InterruptedError` and `KeyboardInterrupt`
are tagged so unanimous interruption remains interruption instead of a
rank-local escape that could leave peers waiting for agreement. This closes
only those in-process control-record and local-preflight boundaries. It does
not make a failed/lost process-group transport recoverable; transport loss,
peer death, and retrying rank collectives still require a fresh group. Generic
tensor-parallel construction and all family operation schedules have not been
proven to traverse one admission agreement, and distributed consensus remains
partial. P06/P12 remain IN PROGRESS and P15 remains OPEN. No tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

### P11 native family Edge callback null-envelope guards — 2026-10-16

The GLM5/Inkling native Edge adapter now validates `out`, capability storage,
options, model pointer, owner and request pointers before dereferencing them.
Failed open starts with a null handle and zeroed capabilities; embed/logits/
selection callbacks return a bounded error for null envelopes instead of
crashing across the C ABI. This only hardens these native Edge callback
boundaries. It does not close the family-wide ABI symbol/schema/range,
reentry/ownership, factory/CLI/service wiring or numerical contract review.
P11 remains IN PROGRESS and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P13 comparison report producer binding — 2026-10-16

Every execution-row evidence obligation now declares
`comparison_report_owner=wholemodel_validation.runner.report_receipt_binding`;
structural source validation requires that exact owner, and the P13
source-reference audit resolves the concrete receipt-binding symbol, not just
its containing module. This supplements each row's comparison report schema
label with a resolvable source producer. It does not attest that the runner
executed, that a report or observation receipt exists, or that report contents
are valid runtime evidence. Each row also binds the expected
`openagent.wholemodel-report-binding.v1` receipt schema, which structural
validation now requires. That schema declaration is not a generated receipt.
P13 remains IN PROGRESS and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P11 Qwen3.5 public range capability-failure cleanup retry — 2026-10-16

When Qwen3.5 Segment range construction failed after the native template was
created (for example while validating or copying capability metadata), cleanup
failure previously replaced the construction exception and left the template
unreachable for retry. The adapter now retains that unadopted template in its
cleanup-pending inventory and preserves the primary failure with a cleanup
note. This closes only the pre-publication owner-retention seam; family-wide
ABI schema, callback, factory/CLI/service wiring and all-caller lifecycle
coverage remain incomplete. P11/P14 remain partial and P15 remains OPEN. No
tests, imports, builds, benchmarks, device probes, commits, or pushes were run.

### P13 artifact-pin owner resolved to executable source tooling — 2026-10-16

Every family P13 evidence obligation now names
`wholemodel_validation.runner.pinned_artifacts` as the artifact-pin source
owner, replacing the unresolved label `publisher_artifact_manifest`. The
existing source-reference audit can now bind each matrix row's pin obligation
to the implementation that hashes and rechecks the selected artifact files.
This is a source/catalog binding only; it does not prove that any artifact was
pinned or that execution, numerical parity, or device placement evidence
exists. P13 remains IN PROGRESS and P15 remains OPEN. No tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

### P04/P14 Qwen4 media-owner retryable child cleanup — 2026-10-16

`Qwen4MultimodalDecoder.close()` now retires successfully closed text/vision
owners immediately; after the text borrower is closed, it independently
attempts both native bank wrappers even if an independent snapshot/vision
cleanup failed. It retains the shared native-store runtime while either
wrapper still borrows it, and only closes each owned publisher provider after
its dependent text/bank owner is released. A failed text close deliberately
keeps those dependent banks, runtime and providers reachable for retry. This
is a bounded cleanup ownership fix, not proof of Qwen4 media ordering/position
parity or all caller lifecycle closure. P04/P14 remain partial and P15 remains
OPEN. No tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P11 Qwen4 native routing callback pointer admission — 2026-10-16

The Python callback installed into the Qwen4 native routing observer now rejects
null raw-logit, selected-expert, or scale-array pointers before constructing
NumPy views. Previously a malformed native callback envelope could reach
`np.ctypeslib.as_array` with a null pointer while the callback exception was
the only containment boundary. This closes one callback-input ABI check only;
it does not validate arbitrary pointer accessibility or finish the family-wide
ABI symbols, ranges, tensor/head schemas, callback reentry/ownership, and caller
wiring review. P11 remains IN PROGRESS and P15 remains OPEN. No tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

### P11 public range open-record ABI admission — 2026-10-16

The shared native Segment/Edge adapter open path now requires an exact
`_SegmentOptions` or `_EdgeOptions` record size and zero reserved fields before
resolving the checkpoint path or admitting backend/plan values. Mixtral,
GLM-DSA, Qwen3.5, Qwen4, publisher-family, and OLMoE adapter overrides now pass
the complete record through this common admission boundary. This closes only
the shared open-record size/reserved-field gap; create/session records, all
other family ABIs, ownership/reentry, and family wiring still require audit.
P11 remains IN PROGRESS and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P07/P14 K3 checkpoint cancellation at serving sink — 2026-10-16

The K3 publisher CLI already accepts cancellation during cache-frame
publication in `write_generation_boundary`, but its generation and resume sink
callbacks did not pass the request cancellation predicate to that API. Both
paths now thread cancellation through the checkpoint writer, which checks it
while serializing bounded cache chunks and before publication. Previously a
cancelled request could remain inside durable-boundary staging until the full
cache frame finished, even though cancellation was checked immediately after
the sink returned. This is a caller-wiring fix only; sink atomicity, complete
K3 serving/media parity, and P13 execution evidence remain open. P07/P14
remain partial and P15 remains OPEN. No tests, imports, builds, benchmarks,
device probes, commits, or pushes were run.

### P06/P12 distributed generation result rollback and cleanup agreement — 2026-10-16

`DistributedExecutionService` now retains a per-rank physical snapshot before
ordinary generate/resume execution and agrees on snapshot acquisition before
any rank enters generation. If surviving ranks disagree or report a failure
at the result boundary, each rank restores the prior executor state
before the existing owner teardown and coordinator poisoning path. Successful
operations retire the temporary snapshot. If snapshot retirement itself
disagrees, every surviving rank now tears down the owning session and agrees
on teardown errors, preventing a subset of generated or snapshot state from
remaining reachable behind a failed distributed handle. This is bounded to
the distributed service's generate/resume result and temporary-snapshot
cleanup boundary; failures inside a process/transport collective can still
strand peers and require group restart, and this does not prove every bank or
family consumer's physical state is covered. P06/P12 remain IN PROGRESS and
P15 remains OPEN. No tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P09 Q4_1 conversion temporary cleanup ordering — 2026-10-16

The Q4_1 expert converter now removes its same-directory staging file on
conversion or publication failure without replacing the primary exception;
cleanup failure is attached as a note. After successful hard-link
publication, temporary unlink failure is surfaced rather than hidden. This
closes only the Q4_1 converter cleanup seam; Q5/Q8/grouped converters and the
full loader-to-consumer format map still require review. P09 remains IN
PROGRESS and P15 remains OPEN. No runtime validation, tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P04 cancellable processor-array ownership staging — 2026-10-16

`MultimodalSessionExecutor.forward_media` now detaches borrowed processor
arrays with bounded NumPy iterator chunks, checking request cancellation between
chunks instead of performing one potentially large non-interruptible
`tobytes()` copy. If staging or cancellation fails, already-created owned
buffers are dropped before propagating the primary failure; the staged arrays
are also released immediately after synchronous decoder dispatch, including
when that dispatch raises. This closes only the conditional-session array
staging boundary; external processor/decode calls, family-specific media
ordering/geometry, and remaining caller cleanup paths remain open. P04 remains
IN PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P06 tensor-parallel projection agreement caller-field admission — 2026-10-16

`DistributedProjectionBank.linear()` and `grouped_linear()` now require an
exact non-empty string projection name and encode only admitted scalar name
and group fields in cross-rank schedule descriptors. Previously a malformed
caller identifier could be embedded in a descriptor after local lookup failed;
array-like objects can make tuple equality raise during rank agreement rather
than yielding a stable schedule mismatch. This closes only caller-field
admission for these two bank operations. Rank process loss, transport failure,
and unbounded/unavailable collectives still require group restart; other bank
and consumer contracts remain open. P06 remains IN PROGRESS and P15 remains
OPEN. No tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P09/P10 Q5_0 converter failure cleanup — 2026-10-16

The Q5_0 expert converter now preserves its original write/fsync/manifest or
publication failure if removing the temporary container also fails, attaching
the cleanup error as a diagnostic note. After successful hard-link
publication, temporary-file removal failure remains visible to the caller.
This closes only the Q5_0 conversion cleanup seam; Q5_1, Q8_0 and grouped
converters still require individual review, and loader-to-consumer parity,
actual device placement, operation-level no-fallback and runtime evidence
remain open. P09/P10 remain IN PROGRESS and P15 remains OPEN. No tests,
imports, builds, benchmarks, device probes, commits, or pushes were run.

### P09/P10 selected-device family-format admission before store open — 2026-10-16

The native family CLI's selected-device expert owner now resolves every
manifest layer's persisted format through `require_format_execution_consumer`
for the exact family/device pair before constructing `DiskExpertStore` or
loading lattice metadata. This prevents the shared Mixtral-derived device
provider's broader decoder registry from implicitly admitting formats absent
from a family capability policy; the existing specific E8 activation-rotation
restriction still runs first. The source-scope helper documentation now names
this enforced admission boundary. This closes only the CLI selected-device
manifest gate; other direct factories, loader-to-consumer chains, selected
operation no-fallback, and actual runtime placement remain unverified. P09/P10
remain IN PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks,
device probes, commits, or pushes were run.

### P14 research synthesis provider-stream cleanup — 2026-10-16

`ResearchEngine._stream_model_result()` now explicitly closes both a returned
iterable and its distinct iterator on exhaustion or failure, rather than
depending on generator finalization after its `for` loop. Cleanup attempts are
independent: cleanup errors are attached to an active provider/consumer failure,
and otherwise the first cleanup failure is surfaced with later cleanup errors
noted. This protects provider stream/session ownership if research chunk
processing fails or iteration is interrupted. It does not close interruption
inside the provider operation or establish recovery/backpressure parity for
other callers. P14 remains IN PROGRESS and P15 remains OPEN. No tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

### P09/P10 Q5_1 converter failure cleanup — 2026-10-16

The Q5_1 expert converter now preserves its primary conversion/publication
exception if removal of its same-directory temporary file also fails,
attaching the cleanup failure as a note. A temporary unlink failure after
successful hard-link publication remains visible. This closes only Q5_1
converter cleanup; Q8_0 and grouped INT4 conversion cleanup still need review.
Family/backend loader-to-execution-consumer coverage, exact scale/geometry
parity across admitted tuples, selected-device no-fallback behavior, actual
placement receipts and runtime numerical evidence remain open. P09/P10 remain
IN PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks,
device probes, commits, or pushes were run.

### P06/P12 distributed cache manifest publication rollback — 2026-10-16

After rank cache files are agreed, root now removes only that operation's
fresh generation directory if manifest encoding, size admission, staging, or
atomic rename fails before publication. Staging-file cleanup errors are
attached to the primary exception. If rename succeeded but a later directory
fsync fails, the generation is retained because publication durability is
uncertain. This closes the local uncommitted-artifact leak for the distributed
cache publisher; it does not recover rank files after process/transport loss,
or establish cross-rank cleanup after a failed transport collective. P06/P12
remain IN PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks,
device probes, commits, or pushes were run.

### P14 codebase-run SSE terminal-event recognition — 2026-10-16

`send_event_stream()` now treats the codebase run stream's explicit
`stream.closed` event as terminal, alongside `done`, `error`, and `terminal`.
The codebase event source emits `stream.closed` after a final run status; the
generic SSE caller previously appended a second synthetic `done` event when
that iterator exhausted. The explicit terminal event is now preserved as the
single protocol boundary, while iterator close on completion/disconnect/error
continues through the existing finalizer. This is one SSE protocol/caller fix;
other route-specific terminal, cancellation, backpressure, recovery, and media
semantics remain to be reconciled. P14 remains IN PROGRESS and P15 remains
OPEN. No tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P09/P10 grouped INT4 converter failure cleanup — 2026-10-16

The grouped INT4 expert converter now preserves the primary packing/write/fsync
or publication failure when temporary-file removal also fails, attaching the
cleanup failure as a note. Cleanup failure after successful hard-link
publication remains visible. This closes only grouped INT4 conversion cleanup;
other grouped/low-bit format consumers, family/backend loader-to-consumer
coverage, actual device placement, operation-level no-fallback and runtime
evidence remain open. P09/P10 remain IN PROGRESS and P15 remains OPEN. No
tests, imports, builds, benchmarks, device probes, commits, or pushes were run.

### P11 public session-create record admission — 2026-10-16

The shared Segment/Edge constructor guard now validates the session-create
record's exact `_SessionOptions` size and requires all reserved words to be
zero before reading its memory budget or reserving resources. This covers
family callbacks using the common `_CREATE` ABI boundary, including OLMoE,
Mixtral, GLM-DSA, Qwen3.5, Qwen4, and publisher-family adapters. Per-family
context/budget semantics and other public records remain separately reviewed;
this does not close the family-wide ABI map. P11 remains IN PROGRESS and P15
remains OPEN. No tests, imports, builds, benchmarks, device probes, commits,
or pushes were run.

### P02 GLM in-memory factory source-provenance restrictions — 2026-10-16

The direct in-memory GLM5 device, GLM-DSA device, and GLM-DSA native service
factories now fail closed at entry. These APIs receive already-materialized
arrays without authenticated source dtype/persisted-format provenance, so
family geometry validation alone cannot select or prove an exact P02 tuple.
The guard runs before tensor copying, decoder/factory construction, or any
checkpoint artifact access. Directory-backed paths still have family-specific
inventory planners, but their exact family/backend/format rows and caller
chains remain unverified. This is a narrow restriction, not GLM route closure;
P02/P15 remain OPEN. No tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P06/P12 distributed cache receipt pinned during physical restore — 2026-10-16

Distributed cache restore previously verified each rank file's digest, then
opened the path again for physical installation. A shared-filesystem swap in
that interval could install a different valid cache with matching logical
history while the distributed manifest's exact byte receipt was never checked
against the bytes actually consumed. The stable service-cache reader now
optionally hashes and counts the same descriptor stream used by the executor,
and distributed restore supplies its admitted rank receipt to that read. The
receipt check occurs before the service transaction publishes token history;
the existing rollback path handles any mismatch. This closes the receipt
path-swap window for distributed cache loading only. Process/transport loss
still requires group restart; other recovery formats and consumers remain
open. P06/P12 remain IN PROGRESS and P15 remains OPEN. No tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

### P09/P10 grouped and low-bit converter cleanup review — 2026-10-16

Reviewed the remaining grouped/low-bit conversion cleanup seams in
`expert_packing.py` and `grouped_int4_conversion.py`. `expert_packing` already
closes staging-writer failures independently, attaches cleanup errors to the
primary exception, removes pending staging paths, and rolls back published
multi-container links; no additional concrete error-masking defect was found
in that path. The standalone grouped INT4 converter's masking defect was
closed in the adjacent P09/P10 entry. This is not a full audit of every format
converter or family/backend consumer: exact loader-to-execution mapping,
scale/geometry parity, selected-device no-fallback, actual placement receipts
and runtime evidence remain open. P09/P10 remain IN PROGRESS and P15 remains
OPEN. No tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P07/P12 K3 recovery scope-to-media provenance binding — 2026-10-16

`GenerationBoundary.admit()` now rejects a text-only or ordinary image-text
checkpoint carrying temporal media records, and rejects a mixed-media boundary
that omits its ordered media provenance. The K3 publisher CLI produces these
scope/metadata combinations consistently: file/processor still-image prompts
have no temporal records, while explicitly detached media items carry one
record per item. Previously the recovery envelope validated each record's
shape and timestamps but allowed a forged or malformed envelope to report a
scope inconsistent with those records after restoring a physical cache. This
binds only the K3 generation metadata contract; it does not replay media or
prove the provenance corresponds to the source pixels, nor close FLA/native
continuation parity, batch rollback, all serving callers, or P13 runtime
evidence. P07/P12 remain partial and P15 remains OPEN. No tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

### P14 hardware worker stream-consumer failure stops generation — 2026-10-16

`HardwareWorker.generate_stream()` now catches event-emitter failure inside
the OpenVINO streamer callback, returns the callback's stop signal, and then
re-raises the original sink failure after `pipeline.generate()` returns (or
adds any engine-stop exception as a note). The JSONL worker exits quietly on a
broken output pipe instead of trying to serialize an error onto the same
disconnected stream. This closes one direct hardware stream consumer-failure
path; it does not establish provider-wide cancellation, device-side immediate
interruption, or other workers' stop/backpressure semantics. P14 remains IN
PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P12 durable continuation cache receipt pinned during restore — 2026-10-16

`DurableContinuationStore.recover()` and `recover_into()` now pass the catalog
bytes/SHA receipt into the native service-cache reader. The stable descriptor
used for physical cache installation hashes/counts the same bytes the executor
consumes, closing the verify-then-reopen path-swap window where a different
valid cache with matching token history could otherwise be installed. Custom
recovery-service adapters retain their legacy call contract and remain
responsible for equivalent receipt binding. The existing owned-session rollback
and quarantine behavior handles receipt mismatch. This is a single durable
continuation path fix, not cross-family recovery closure; process/transport
failure remains outside scope. P12 remains IN PROGRESS and P15 remains OPEN.
No tests, imports, builds, benchmarks, device probes, commits, or pushes were
run.

### P11 GLM-DSA range preflight before checkpoint tensor access — 2026-10-16

The direct GLM-DSA public Segment open callback now checks its requested
`begin < end` range against the family's 4096-layer admission ceiling before
calling the directory loader. Its existing `_allowed(options)` backend and
root gate therefore runs before this range preflight and before checkpoint
tensor access; malformed empty/reversed/out-of-ceiling intervals no longer
reach the metadata/tensor loader. The shared Segment/Edge service factory also
preflights known backend bits, contiguous ordered ranges, and a common
checkpoint root before constructing the executor or loading its native
runtime. This cannot establish that `end` is within the concrete model's layer
count until configuration admission, nor that every external family adapter
uses this factory. Other-family direct-open ordering and full ABI route tracing
remain open. P02/P11 remain partial and P15 remains OPEN. No tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

### P02 Mixtral packed-provider direct factory restriction — 2026-10-16

The Mixtral in-memory device factory and directory-backed device service now
reject `TorchMixtralExperts` at entry, before input copying or checkpoint
reads. The CLI already rejects generic packed manifests because they do not
bind the source dtype and full family/backend/format tuple; direct device
factory callers could previously bypass that restriction with a provider
whose geometry/fingerprint alone passed decoder checks. This closes only the
Mixtral packed-manifest direct-device seam; ordinary native/device tensor
routes and other families still need exact-row and caller-chain review. P02
and P15 remain OPEN. No tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P03 GLM-DSA device streamed-expert selector ownership preflight — 2026-10-16

The GLM-DSA device directory selector now validates an optional streamed
expert provider before it removes the checkpoint's resident expert tensors
from the selected payload set. Only the exact device/Torch GLM-DSA provider
types are accepted, and the provider must be live, have the same complete
family contract, cover every sparse layer, and match hidden/intermediate,
expert-count, and top-k geometry. This prevents a generic or foreign-family
provider from bypassing the inventory-to-consumer boundary simply by exposing
`max_bytes`. The family selector audit in this pass covered this GLM-DSA
device directory path only; it does not certify other GLM-DSA entrypoints or
the remaining family selectors/factories. P03 remains IN PROGRESS and P15
remains OPEN. No tests, imports, builds, benchmarks, device probes, commits,
or pushes were run.

### P03 Publisher optional-bank consumer audit — 2026-10-16

Audited the publisher conditional path across Qwen3.5, Qwen4, GLM5, and
Inkling: `component_contract()` extracts the declared auxiliary fields;
`partition_weights()` rejects MTP payloads outside Inkling and rejects
unowned media/audio namespaces; the conditional loader rejects unsupported
Qwen audio declarations/tensors and requires exact whole text/media inventories
against each family's planner; Inkling headers are routed through
`admit_mtp_inventory()` before payload materialization. That MTP admission now
also rejects auxiliary configuration keys that have no admitted Inkling MTP
consumer, instead of silently retaining unknown draft semantics; the component
contract also rejects malformed non-object `mtp_config` values rather than
coercing false-y declarations such as `0` or `''` into an empty bank. The resident
family planner audit covered Mixtral, Qwen3.5, Qwen4, GLM-DSA, GLM5, and
Inkling text contracts; those planners require exact config-family identity,
complete names/shapes, and permitted source dtypes, with expert banks removed
only where their consuming provider is separately admitted. The preceding
GLM-DSA device directory selector check separately binds its streamed owner
before selected payload reads. This is source-path inspection, not a runtime
or exhaustive caller-coverage claim: public/native/device factories, all
streamed-expert packers, and every media invocation still require end-to-end
caller and ownership reconciliation. P03 remains IN PROGRESS and P15 remains
OPEN. No tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P03 Non-publisher selector and optional-bank audit — 2026-10-16

Source review extended to the alternate non-publisher selectors. The shared
resident planner (Mixtral, Qwen3.5, Qwen4, GLM-DSA, GLM5, Inkling) requires the
family's exact model identity and full planner-derived header inventory;
Mixtral's planner requires exactly one complete fused or split expert layout,
and the other resident planners bind optional integer metadata or tied-head
aliases explicitly. Device directory selectors for Mixtral, Qwen3.5 and Qwen4
validate the complete source headers before removing expert banks for a
separate provider; Qwen3.5/Qwen4 projection providers are also checked against
the same frozen checkpoint config and inventory geometry. GLM-DSA native
range/edge and streamed-expert selectors call its complete inventory planner;
the device streamed-expert selector additionally preflights provider family,
config, sparse-layer coverage and geometry before selecting payloads. The K3
hybrid and V4.1 CED native loaders admit their exact shape/dtype inventories
before selected checkpoint payload handling; no optional expert/MTP/media bank
is present in those exact tensor-shape contracts. Publisher paths reviewed in
the previous note cover Qwen3.5/Qwen4/GLM5/Inkling media and MTP partitioning.

This pass found no additional safe code change in those inspected exact-header
selectors. This is not closure of all factories or execution callers: native
range composition, device/provider artifact-to-source content identity,
runtime ownership through all factory/service/CLI entrypoints, and all media
callers still need separate end-to-end reconciliation. In particular, matching
provider configuration/geometry does not prove packed expert weights came
from the same checkpoint content. P03 remains IN PROGRESS and P15 remains
OPEN. No tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P06 distributed shard fingerprint staging agreement — 2026-10-16

`DistributedProjectionBank.register()` now computes the local shard digest and
allocates the rank fingerprint receive list inside a local-stage agreement
boundary before entering `all_gather_object`. Previously a rank-local
allocation or digest preparation failure could escape after shard metadata
agreement while peers entered the fingerprint collective. This closes that
pre-collective staging failure seam only. Process loss or transport failure
can still strand a collective and requires group restart; post-collective
registry commit and every other bank/consumer remain separately open. P06
remains IN PROGRESS and P15 remains OPEN. No tests, imports, builds,
benchmarks, device probes, commits, or pushes were run.

### P12 V4.1 checkpoint restore post-publication cancellation rollback — 2026-10-16

`load_v41_checkpoint()` now takes a lightweight prior immutable snapshot before
publishing the fully staged checkpoint and rechecks cancellation immediately
after `V41Execution.restore()`. If cancellation arrives during that final
publication window, it restores the prior physical boundary and reports the
cancellation instead of returning success with newly installed state. The
checkpoint parser continues to verify complete file integrity and identity
before mutation. This closes only the V4.1 local file-checkpoint publication
race; other family recovery APIs and distributed consensus remain open, and
process/transport loss still requires group restart. P12 remains IN PROGRESS
and P15 remains OPEN. No tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P02 selected-range source tuple gate — 2026-10-16

The shared `read_native_text_checkpoint()` boundary now applies exact
family/backend/arithmetic/source-encoding/persisted-format admission to the
complete header map for selected-range and edge reads, before the selector can
materialize tensor payloads. This covers Mixtral, Qwen3.5 text, Qwen4 text, and
GLM-DSA public-range loaders; because their current public-range rows are not
verified-admitted, those direct loader paths now fail closed rather than
becoming callable from geometry validation alone. Whole-model native-loader
gating remains as previously recorded. This does not trace or prove device,
conditional, media, factory, CLI, gateway, recovery, or tooling routes, and
does not promote any tuple to verified. P02 and P15 remain OPEN. No tests,
imports, builds, benchmarks, device probes, commits, or pushes were run.

Follow-up clarification: the same header boundary now accepts an explicit
source backend, and the Qwen3.5, Qwen4, Mixtral, GLM5, Inkling, and GLM-DSA
Torch `from_directory` readers identify themselves as `device`. Their
complete header maps are therefore checked against the device tuple rows,
instead of being mislabeled as public-range routes. These device rows are also
not verified-admitted, so those direct device loaders fail closed pre-payload.
The service factories/CLI/gateway still lack route-wide matrix reconciliation;
this gate alone does not close their full callable chains. P02/P15 remain
OPEN. No tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P11 publisher-family Edge request record admission — 2026-10-16

The publisher-family public Edge `embed` and `logits` callbacks now require
their exact `_Embed`/`_Logits` record sizes and zero reserved words before
reading token or activation payloads. Previously these two callbacks checked
row and payload extents but ignored versioned record size and reserved fields,
unlike the shared OLMoE Edge callbacks. This closes those K3/V4.1 Edge request
envelopes only; Segment family schemas, callback/ownership rules, and complete
factory/CLI/service wiring remain under P11 review. P11 remains IN PROGRESS
and P15 remains OPEN. No tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P09/P10 selected-device provider subclass rejection — 2026-10-16

The direct Qwen3.5, Qwen4 and Mixtral selected-device factory restrictions now
use `isinstance` for their packed-manifest provider families. Previously an
unverified subclass could bypass an exact-type rejection while inheriting the
same generic manifest format/geometry contract. The selected-device provider
itself validates family-format policy before execution, and the CLI validates
it before store construction. This closes the simple subclass bypass only;
wrappers outside those class hierarchies and other direct factory/loader
routes still need review. It does not establish device placement or no-fallback
evidence. P09/P10 remain IN PROGRESS and P15 remains OPEN. No tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

### P13/P15 local evidence inventory and execution prerequisites — 2026-10-16

Read-only inventory found no whole-model observation `.npz`, row-bound
placement/observation receipt, or decoder execution manifest under the inspected
`.qa`, `.runtime`, `.local-data`, `.tmp-engine-validation`, and validation
benchmark directories. The only model roots found in `models/` were
`models/olmoe-1b-7b-instruct` and
`models/openvino/qwen2.5-1.5b-instruct-int4-ov`. The former has an OLMoE config,
tokenizer, three safetensors shards/index, local partial shard tails, and
OpenAgent expert artifacts; the latter is Qwen2.5 (`model_type=qwen2`) OpenVINO
IR, not Qwen3.5-MoE, Qwen4-Exp, GLM5-Next, GLM-MoE-DSA, or DeepSeek V4.1. These
files therefore do not bind the requested target-family P13 rows. They are not
runtime observations or placement/cache receipts.

The whole-model runner's current binding gate joins a manifest to the exact
catalog tuple and permits only rows still marked `candidate-unverified`; it
does not promote those rows or certify execution. Currently generated rows are
not `verified-admitted`, and restricted/absent rows are rejected. The local
OLMoE checkpoint has config/tokenizer/index and three full-size shards whose
lengths match its local recovery metadata; bounded safetensors-header reads
found BF16 for all 3,219 tensor entries. However the generated OLMoE backend
policy currently lists only `float32` and `float16` source formats, so this
local BF16 checkpoint cannot honestly bind to an exact generated OLMoE row.
The other available model is Qwen2.5 (`model_type=qwen2`) OpenVINO IR, not an
admitted Phase 1 family. No matching decoder manifest, runner receipt or
observation artifact was found, and no P13 execution was attempted. For the
requested Qwen3.5-MoE, Qwen4-Exp, GLM5-Next, GLM-MoE-DSA and DeepSeek V4.1 rows,
the required local checkpoints are absent. A future valid run needs (1) an
exact P02 tuple with reviewed oracle/candidate factories, (2) complete pinned
source/config/all-weight-shard/tokenizer/dependency files, (3) fresh offline
sessions that return the exact row metadata, (4) exact complete cache/evidence
schemas and per-operation placement observations, and (5) produced digested
observation artifacts, cache/placement receipts and comparator report. No
source or local artifact inspected here substitutes for those executed
receipts. P13 remains IN PROGRESS and P15 remains OPEN. No tests, imports,
builds, benchmarks, device probes, commits, or pushes were run.

### P05 greedy speculative bonus preserves target RNG — 2026-10-16

At temperature zero, speculative acceptance and rejection already use
deterministic argmax decisions, matching the plain V4.1 sampler. The bonus
token path nevertheless called `Generator.choice` on a one-hot distribution,
advancing target RNG state despite greedy sampling. `SpeculativeDecoder._sample`
now returns argmax directly at zero temperature, keeping the checkpointed
target RNG boundary consistent with the plain caller. This closes only that
greedy bonus RNG mismatch; all-caller accept/reject behavior, external sink
atomicity, and opaque proposer restore guarantees remain open. P05 remains IN
PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P06 distributed projection post-collective host-failure agreement — 2026-10-16

`DistributedProjectionBank.linear()` now agrees on host synchronization of the
reduced finite flag before host export. `grouped_linear()` performs that finite
flag read and output-shape construction inside its next local-stage agreement.
Previously a rank-local device synchronization or shape failure could escape
after the shared all-reduce while peers entered the following host-export
agreement. This closes those post-collective local-failure seams only. A
collective/process/transport failure itself remains non-recoverable and needs a
group restart; remaining operations and every consuming model family still
require their own closure. P06 remains IN PROGRESS and P15 remains OPEN. No
tests, imports, builds, benchmarks, device probes, commits, or pushes were run.

### P06 BF16/FP8/FP4 bank registration and interruption agreement — 2026-10-16

The source map confirms V4.1 BF16, leased FP8, and MXFP4 tensor-parallel banks
are selected by `v41_distributed_session.make_projection_backend`; generic
projection shards also feed publisher K3/V4.1 checkpoint adapters, while
`DistributedExecutionService` coordinates rank-local native services and
replacement recovery is orchestrated by `distributed_recovery`. For the
BF16/packed banks, source identifiers are now exact non-empty strings, pin
buffer allocation and registry publication are rank-agreed, and packed local
work propagates `KeyboardInterrupt`/`InterruptedError` through agreement
instead of escaping before peers. This closes those bank registration and
cooperative interruption seams, not every projection/consumer mapping. A
process or transport loss can still strand peers at a collective and requires
group restart. P06/P12 remain IN PROGRESS and P15 remains OPEN. No tests,
imports, builds, benchmarks, device probes, commits, or pushes were run.

### P06/P12 distributed latest-pointer uncertain-commit retention — 2026-10-16

`CollectiveGenerationSink` no longer removes a generation after manifest/latest
publication has begun. `_atomic_manifest()` may complete the latest-pointer
rename and then fail its directory fsync; deleting that generation would leave
`latest.json` pointing to missing recovery state. On uncertain post-manifest
failure the generation is retained (it may be orphaned if pointer publication
never completed), preferring recoverable extra files over a stale committed
pointer. Failures before any manifest exists still remove only the fresh
reservation. This does not recover peers after process/transport loss, which
still requires group restart. P06/P12 remain IN PROGRESS and P15 remains OPEN.
No tests, imports, builds, benchmarks, device probes, commits, or pushes were
run.

### P09/P10 family format-route reconciliation snapshot — 2026-10-16

Source-level route-to-admission map reviewed for this pass:

- Mixtral and OLMoE source inventory readers bind floating source encodings to
  `source_scope.require_source_inventory_admission`; their generic native
  `SparseExpertRuntime` now checks every layer against the family/native
  persisted-format consumer row before opening `DiskExpertStore`.
- Qwen3.5, Qwen4, GLM5-next, Inkling and GLM-DSA directory/publisher paths use
  metadata-only family inventory admission before payload materialization;
  selected-device generic packed providers are restricted at family policy,
  CLI, and direct factory boundaries where source provenance is absent.
- K3 and V4.1 continue on dedicated publisher format/projection owners rather
  than generic expert manifests; candidate formats do not thereby become
  admitted generic decoder routes.

This is a source-map reconciliation, not proof that every third-party caller
uses those entrypoints. Exact per-checkpoint loader/conversion/execution tuples
and all alternate callers remain to be reconciled. P10 still lacks operation-
level no-fallback observations and real device-placement receipts for each
applicable family/backend/format row; source admission and declared device
names are not placement evidence. P09/P10 remain IN PROGRESS and P15 remains
OPEN. No tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P09/P10 low-level selected-device constructor provenance gates — 2026-10-16

Guarded direct device constructors for Torch Inkling, Qwen3.5, Qwen4, GLM5-next,
and GLM-MoE-DSA with module-private source-admission tokens. The corresponding
directory readers supply these tokens only after metadata inventory and
`source_scope` persisted-format admission; the publisher conditional wrappers
pass them only after their own inventory admission. GLM5's publisher loader
passes the token after `admit_publisher_glm5_inventory`. Inkling's staged MTP
device decoders inherit the token from the already admitted backbone path.
Existing direct in-memory service factories remain fail-closed before these
constructors where source provenance is not available.

This is a source API provenance barrier, not a security boundary against
callers deliberately importing private module state. It does not establish
actual device placement or per-operation no-fallback. Direct GLM-DSA device
construction is now limited to its metadata-admitted directory reader; exact
runtime placement, operation-level fallback observations, numerical evidence,
and broader per-format execution receipts remain open. P09/P10 remain IN
PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks, device
probes, commits, or pushes were run.

### P02/P03 GLM5 device provider source-content restriction — 2026-10-16

The GLM5 device directory factory and `TorchGlm5NextDecoder.from_directory`
now reject externally constructed expert or projection providers. Their
configuration and tensor geometry could match while their payloads came from
another checkpoint revision; the provider owners and directory loader do not
yet share one immutable source-content identity. This closes that specific
provider-substitution bypass by restriction, not the GLM5 provider execution
path. The directory reader still admits the complete local tensor header
against the exact device/source-format row before payload selection, and the
direct decoder constructor requires its private metadata-admission token.

This does not complete P02 route tracing across all admitted
family/backend/format/caller/tooling tuples or P03's family-wide inventory to
consumer map. Other factories, CLIs, and third-party callers still need
reconciliation, and source admission is not runtime evidence. P02/P03 remain
IN PROGRESS and P15 remains OPEN. No tests, imports, builds, benchmarks,
device probes, commits, or pushes were run.

### P03 Inkling device provider source-content restriction — 2026-10-16

The Inkling device directory decoder and service factory now reject external
expert and projection providers before checkpoint reads or service publication.
Their configuration and geometry contract did not prove that omitted provider
weights belonged to the same checkpoint revision as the resident tensors. This
is a fail-closed restriction until provider manifests and resident loaders share
an immutable source-content identity; the Inkling MTP/media tensor inventory
and its consumer behavior are not changed by this restriction.

This closes only the Inkling device provider substitution path. P03 still lacks
complete family-wide tracing from every admitted optional/shared/MTP/expert/
media tensor through every factory, CLI, and execution caller, and does not
provide runtime execution evidence. P03 remains IN PROGRESS and P15 remains
OPEN. No tests, imports, builds, benchmarks, device probes, commits, or pushes
were run.

### P04 Qwen contiguous media-run positional splitting — 2026-10-16

Qwen4's MRoPE position builder previously treated every contiguous run of one
placeholder ID as exactly one media item. Processor output can concatenate
adjacent expanded image placeholders without an intervening text token, so a
prompt containing two adjacent images could be rejected against the first
image's grid even though the ordered grids and total placeholder count agreed.
The position builder now consumes each contiguous run as the ordered sequence
of grid extents, assigning each item's positions and rotary cursor separately.
Qwen3.5 inherits the corrected position builder, while retaining its stricter
single-temporal-grid validation for still images. Cancellation remains checked
between media items.

This is a bounded Qwen positional/order fix, not proof of processor parity or
complete P04/P07/P14 closure. GLM5 and Inkling still lack equivalent verified
processor-to-placeholder ordering evidence; V4.1 and K3 have separate media
owners whose lifecycle and caller integration remain partial. Temporal/video
semantics, all-family geometry, ownership under every serving caller, recovery,
and actual runtime observations remain open. P04/P07/P14 remain IN PROGRESS
and P15 remains OPEN. No tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P14 synchronous inference disconnect cancellation propagation — 2026-10-16

The local HTTP handler now watches the connection for peer EOF during Brio,
Jev/SystemOne, OpenAI-compatible and Anthropic-compatible synchronous requests.
Its request-scoped cancellation callback is passed through route dependencies
to native scoring or generation; cancellation is returned as a request-cancelled
outcome. These handlers call the configured native owner directly, so they do
not replay canceled work through a gateway fallback provider. The capability record now describes this narrow behavior as
cooperative cancellation on detected disconnect. This does not add streaming
semantics, prove cancellation for a socket failure the watcher cannot observe,
or cover other route groups and non-HTTP callers. P14 remains IN PROGRESS and
P15 remains OPEN. No tests, imports, builds, benchmarks, device probes,
commits, or pushes were run.

### P02 direct in-memory factory admission regression — 2026-10-17

Added a route-wide regression over the direct in-memory execution-service
factories for Mixtral, GLM-DSA, Qwen4, Qwen3.5, Inkling, and GLM5-next. Each
must fail with the explicit `in-memory factory is restricted` admission error
before attempting to open a runtime library or probe a device. The focused
`tests.test_source_scope_inventory` suite passed (8 tests).

This verifies those fail-closed boundaries; it does not prove the alternate
directory/publisher routes are complete or that every gateway, CLI, recovery,
and tooling caller carries the same full family/backend/format tuple. P02/P03
remain IN PROGRESS and P15 remains OPEN. No imports, builds, benchmarks,
device probes, commits, or pushes were run.

### P05/P14 V4.1 speculative stream publication restriction — 2026-10-17

V4.1 generation and recovery now reject speculative (`dspark`) streaming before
publishing events unless a transactional event-sink protocol exists. The
current callback API cannot retract provisional events if later cancellation,
callback failure, or request rollback resets target and draft state. The
regressions verify regular generation rejects this mode before touching the
session or emitting an event, and recovery rejects it after envelope admission
but before cache restore or event publication. `tests.test_conditional_session_cleanup`
passed (8 tests).

This is a concrete restriction for V4.1 speculative streaming, not completion of
speculative behavior across all serving callers. Ordinary V4.1 streaming,
checkpoint sink composition, other families, and end-to-end output atomicity
remain separately scoped. P05/P14 remain IN PROGRESS and P15 remains OPEN. No
imports, builds, benchmarks, device probes, commits, or pushes were run.

### P03 direct-wrapper configuration-only MTP rejection — 2026-10-17

Shared `partition_weights` now rejects MTP/auxiliary configuration for every
publisher wrapper except Inkling, even when the checkpoint contains no MTP
tensors. This aligns direct Qwen3.5/Qwen4 wrapper construction with the
conditional loader's existing pre-payload restriction. GLM5's independent
inventory planner now rejects its configuration-only MTP claim as well as
physical MTP tensors, because no GLM5 draft consumer is wired. Regressions
cover config-only and tensor-backed claims for Qwen3.5, Qwen4, and GLM5, plus
the GLM5 planner boundary. The focused source-scope and lifecycle suites passed
together (23 tests).

Inkling's supported MTP path is unchanged; full stage execution and serving
caller coverage still require separate reconciliation. P03 remains IN
PROGRESS and P15 remains OPEN. No builds, benchmarks, device probes, commits,
or pushes were run.

### P03/P14 conditional-service MTP caller restriction — 2026-10-17

`open_native_conditional_directory_service` now rejects nonzero MTP budgets
before checkpoint admission or construction. It previously could load the
Inkling MTP owner, but the service generation path did not call `draft()` or
verify those draft tokens against target logits. Lower-level `InklingMTP` and
`InklingMultimodalDecoder.draft` remain available to explicit callers; ordinary
CLI/service generation cannot imply that the MTP bank participates. A focused
regression verifies rejection before opening the checkpoint. The source-scope
and lifecycle suites passed together (24 tests).

Integrating a target-verified MTP proposal loop and validating its rollback/RNG
behavior remains open. P03/P05/P14 and P15 remain IN PROGRESS/OPEN. No builds,
benchmarks, device probes, commits, or pushes were run.

### P04/P14 processor media staging release on success and failure — 2026-10-17

`PublisherMediaProcessor.prepare` now clears the processor-call keyword staging
and its owned float32 waveform staging list in `finally`, including when
preprocessing raises or cancellation interrupts it. It also drops references
from its owned-waveform tracking list, last waveform slice, decoded frame
containers, and local caller-sequence snapshots; it closes/clears decoded image
owners as before. The caller-provided waveform is not mutated. A focused
failure-injection regression verifies the processor's retained audio list is
empty after the exception and that the original input remains unchanged.
Processor result arrays are also dropped if post-call validation, cancellation,
or image-owner cleanup fails; successful result arrays remain with the caller.
The combined
`tests.test_conditional_session_cleanup` and `tests.test_source_scope_inventory`
run passed (25 tests).

This closes only the processor-local staging lifetime; it does not establish
ordering/geometry parity, every caller's media cleanup, or full P04/P14
coverage. Cross-family media/continuation and caller lifecycle paths remain
open. P04/P14 remain IN PROGRESS and P15 remains OPEN. No builds, benchmarks,
device probes, commits, or pushes were run.

### P04/P14 conditional CLI releases decoded request buffers on every exit — 2026-10-17

`PublisherSession.run` now validates sampling, stop controls, token budget, and
cache budget before media preprocessing. It clears request-local image, audio,
video, and video-metadata lists in a `finally` immediately after
`PublisherMediaProcessor` returns or raises, and drops local references to the
last decoded file payload/waveform/frame sequence when present. The service
receives the processor's independent model-input result, while caller-owned
media objects are not mutated. Failure-injection regressions confirm both that
processor-retained inputs are empty after preprocessing failure and that
prepared tensors are cleared after a service-generation failure. The guarded
session lifecycle also invokes request-media cleanup on every exit, including
audio/video decode or later admission errors. A weak-reference regression
confirms decoded audio is collectible after a post-decode admission failure.
The source-scope and lifecycle suites passed (28 tests); with the three
distributed regression suites the combined run passed (38 tests).

This closes request-local media-reference cleanup only for the native
conditional CLI lifecycle. Media ordering and geometry parity, other caller
lifecycles, and full P04/P14 closure remain open.
P04/P14 remain IN PROGRESS and P15 remains OPEN. No builds, benchmarks, device
probes, commits, or pushes were run.

### P06 distributed protocol regression verification — 2026-10-17

Re-ran `tests.test_distributed_checkpoint_errors`,
`tests.test_tensor_parallel_consensus`, and
`tests.test_continuation_store_lock_order` against the current distributed
coordination and receipt-validation changes. All 10 focused tests passed. This
is source-level regression evidence only; it does not cover every distributed
bank/operation, multiprocess transport loss, or runtime failure consensus.
P06/P12 remain IN PROGRESS and P15 remains OPEN. No builds, benchmarks, device
probes, commits, or pushes were run.

### P12 GLM-DSA and Mixtral rejected-cache staging cleanup — 2026-10-17

The GLM-DSA and Mixtral physical-cache readers already delayed state publication
until identity, geometry, checksum, EOF, and cancellation checks completed, but
large staged arrays could remain referenced by exception tracebacks after any
of those checks rejected a restore. Both readers now clear partially built
array containers, masks, and last-chunk views on every uncommitted exit; a
successful restore still transfers those arrays to the model. New malformed-
checksum regressions capture weak references to the allocated arrays and prove
they are collectible while the model's prior cache/position remain unchanged.
`tests.test_cache_restore_staging_cleanup` passed (2 tests).

This closes rejected-restore staging cleanup for these two native family codecs
only. Other family cache readers, all-family physical-state ownership, stale or
foreign restore behavior, distributed rollback/consensus, and runtime restore
evidence remain open. P12 remains IN PROGRESS and P15 remains OPEN. No builds,
benchmarks, device probes, commits, or pushes were run.

### P12 Qwen4 and Qwen3.5 rejected-cache staging cleanup — 2026-10-17

Qwen4 native and Qwen3.5 cache restore also delayed publication correctly but
held staged arrays in traceback frames after malformed checksum, mask data,
geometry or cancellation errors. Their restore paths now clear staging maps,
per-layer banks, masks, and last chunk/iterator views whenever publication was
not reached. Qwen3.5 explicitly closes its buffered NumPy iterator on error.
Four checksum-failure regressions across GLM-DSA, Mixtral, Qwen4, and Qwen3.5
use weak references to prove staged arrays become collectible and prior cache
state remains unchanged. `tests.test_cache_restore_staging_cleanup` passed (4
tests); the combined focused source/lifecycle/distributed suites passed (42).

This remains a local file-cache cleanup audit, not closure of all device,
Inkling, GLM5, K3, V4.1, batch, or distributed restore lifecycles. P12 remains
IN PROGRESS and P15 remains OPEN. No builds, benchmarks, device probes, commits,
or pushes were run.

### P12 device-backed rejected-cache staging cleanup — 2026-10-17

Extended restore-failure cleanup to Torch Qwen3.5, Qwen4, and Mixtral cache
codecs. A checksum/EOF rejection now drops staged host arrays, views, decoded
buffers, masks, and newly created device tensors before the exception escapes,
without publishing or replacing prior cache state. Added weak-reference
regressions for all three codecs. `tests.test_cache_restore_staging_cleanup`
passes (7 tests total). This closes only rejected-restore staging ownership for
these codecs; family-wide recovery, distributed consensus, and serving recovery
callers remain open. P12 remains IN PROGRESS and P15 remains OPEN.

### P12 Inkling device restore cancellation cleanup — 2026-10-17

The Torch Inkling cache reader now clears partially decoded host arrays and
candidate device tensors if validation, checksum, cancellation, or publication
fails before commit. A regression cancels after candidate tensors are created,
proves the temporary host/device arrays are collectible, and confirms the
previous cache and position remain unchanged. The focused restore cleanup suite
passes (8 tests total). This does not close Inkling MTP, other recovery owners,
or cross-family restore consensus; P12 remains IN PROGRESS and P15 remains
OPEN.

The in-memory Torch Inkling snapshot restore path now also drops candidate
device tensors if cancellation arrives after staging but before publication.
Its regression confirms the prior cache boundary is unchanged and the live
snapshot's owned host state remains retained while temporary device candidates
are released. The focused restore-staging suite now passes (9 tests). This is
not a substitute for the broader P12 family and distributed recovery review.

### P12 compound MTP restore EOF and rollback — 2026-10-17

The shared compound cache reader incorrectly used its exact-length read helper
for the final one-byte EOF probe; a valid frame ending normally therefore
raised a truncation error. The reader now performs a guarded one-byte probe and
rejects only non-EOF trailing data, with cancellation checked around that probe.
If a component restore fails after mutating state, already-snapshotted
components are rolled back in reverse publication order while attempting every
rollback. Added a compound-cache regression that authenticates a real framed
cache, forces the second component to fail after mutation, and verifies reverse
rollback restores both prior states. Focused cache recovery tests pass (10 in
the staging suite). P12 remains IN PROGRESS pending family-wide and distributed
restore closure.

### P14 public ABI composition failure cleanup — 2026-10-17

Public K3 and V4.1 factory composition now closes the already-created backing
execution service if public Segment/Edge composition fails. The directory-backed
public V4.1 factory also covers payload extraction/argument preparation failures
before delegation. If backing close itself fails, the primary composition error
is preserved and a retry cleanup callable is attached. Added regressions for
primary-error preservation and retryable cleanup failure. The focused factory
cleanup/source-inventory suite passes (18 tests); the combined focused set passes
(50 tests). This closes only those factory-construction cleanup paths, not the
full P14 caller lifecycle. P14 and P15 remain open.

### P02/P11 public family ABI checkpoint provenance — 2026-10-17

The public GLM5/Inkling Segment/Edge adapter previously validated the named
checkpoint's family headers but could not prove that the already-materialized
decoder had been loaded from those exact files. Source-backed GLM5 and
conditional Inkling native text loaders now attach a private receipt containing
the canonical checkpoint root, config digest, and complete file-stamp inventory.
Public ABI registration requires an unchanged matching receipt before opening
the native runtime; directly constructed or mismatched decoders fail closed.
The source-scope callable chain now includes the binding issuer on GLM5/Inkling
load paths and the verifier on their public ABI path. Four focused regressions
cover successful exact binding, changed inventory,
missing loader provenance, and refusal before ABI loading. The
binding/source-inventory/public-factory suite passed (22 tests); the combined
focused suite passed (54). This closes only the identified public-family
composition provenance seam, not the family-wide P02/P11 maps. P02/P11 remain
IN PROGRESS and P15 remains OPEN.

### P13 source inventory includes family provenance gate — 2026-10-17

The certification source inventory's `model-families` and `segment-edge`
areas now include `family_source_binding.py`. This pins the loader-issued
checkpoint provenance gate alongside the family loaders and public ABI that
consume it, preventing an area-scoped source receipt from silently omitting
that admission boundary. A focused regression asserts both bindings. The
source-scope/provenance suite passes (21 tests). This is source inventory
coverage only; it is not a pinned model observation, physical-cache receipt,
device-placement receipt, or numerical comparison. P13 remains IN PROGRESS
and P15 remains OPEN.

### P12/P13 Generation checkpoint descriptor handoff cleanup — 2026-10-17

The generation-checkpoint store had three raw-descriptor handoffs (writer,
leased source reader, and detached scratch capture) that called `os.close`
directly when `os.fdopen` failed. A close error could replace the original
failure without recording cleanup diagnostics. These sites now share an
ownership helper that closes the descriptor and attaches close failures to the
primary exception, and the pre-wrap source validation path preserves its own
failure similarly. Fault injection verifies both descriptor closure and
primary-error preservation when close itself fails; the two new tests and
cache-restore cleanup suite pass 14 tests. This covers descriptor handoff at
these generation-checkpoint sites only; remaining temporary-file paths and
distributed process/transport loss remain open. P12/P13 remain IN PROGRESS and
P15 remains OPEN.

### P02/P03 direct OLMoE public-ABI inventory and manifest binding — 2026-10-17

`NativeOlmoePublicAdapters` now applies the same exact `olmoe/olmoe`,
`public-range/fp32` checkpoint inventory gate as the CLI before loading its
native ABI. It binds the expert manifest to the source metadata fingerprint,
validates the full dense tensor header inventory, and checks ordered layer
geometry, expert IDs, hashes, and the INT8 payload sizes actually consumed by
the native range engine. Direct adapter construction can no longer defer a
foreign-manifest or wrong-family failure until after library loading or session
creation. Focused family-binding and source-inventory suites pass 47 tests.
All catalog tuples remain candidate-unverified; this closes one OLMoE ABI
entrypoint only, so P02/P03 remain IN PROGRESS and P15 remains OPEN.

### P03 resident inventory planner binds its own P02 route — 2026-10-17

`admit_resident_inventory` previously returned successfully from its Mixtral
family branch after proving tensor shapes and conversion budget, while exact
backend/source-format/persisted-format binding happened only later in the
checkpoint reader. Direct callers could therefore mistake a complete shape
plan for P02 tuple admission. The planner now performs the exact source-row
check before returning; the checkpoint reader passes its selected backend
through the planner, and the same backend is used by the other family-specific
source checks. A regression confirms direct Mixtral device planning rejects
the currently unverified tuple. The focused source-scope/provenance suite
passes (22 tests). This closes that planner bypass only; family-wide optional,
shared, MTP, expert and media consumer reconciliation remains open. P02/P03
remain IN PROGRESS and P15 remains OPEN.

### P13 row-specific native OLMoE observation adapter — 2026-10-17

Added `wholemodel_validation.olmoe_native` as a candidate-side adapter for the
local native leased OLMoE decoder. It requires runner-provided pins for the
packed expert manifest and existing native library, validates the complete
per-layer physical KV schema, maps observed key/value arrays without
reconstruction, and owns decoder/edge/engine teardown in retryable order. The
P13 OLMoE obligation now names this row-specific cache reader/schema, and the
certification source inventory includes the adapter. Tests cover complete and
malformed cache rows, exact artifact-pin enforcement, native snapshot-state
rollback, and source owner resolution. The focused suite passes (60 tests).
This makes a pinned local OLMoE comparison runnable in source; no model
observation, numerical comparison, cache receipt, or placement receipt has yet
been produced. Other families still use their own evidence obligations. P13
and P15 remain OPEN.

### P13 GLM5-next/Inkling media evidence restriction — 2026-10-17

The GLM5-next and Inkling source media rows previously only exposed a boolean
`media_adapter_missing` marker. They now carry an explicit reason stating that
no row-specific full-media observation adapter and comparison report are
implemented and that `validate_execution_binding` rejects the tuple before
candidate adapter import. Source validation checks the reason is present only
for media rows lacking an adapter; a focused regression inspects both families.
This restriction closes ambiguity in the evidence contract only: it does not
implement or numerically validate media parity, and the model-serving media
paths remain separately applicable. P13/P15 remain OPEN.

The restriction is now compared against the canonical generated row contract,
not accepted by prefix match, so a modified/attenuated reason cannot validate.
The whole-model binding failure now includes the row's concrete restriction
reason, so callers do not receive an unexplained generic media-evidence error.
The focused source-inventory suite now passes (23 tests), including catalog tamper,
runner rejection, selected-loader, and canonical-identity checks; this suite does not close the wider model/runtime
evidence requirements.

### P02 selected-loader format gate covers GLM5-next and Inkling — 2026-10-17

The shared selected-tensor branch of `read_native_text_checkpoint` omitted the
GLM5-next and Inkling family-to-catalog mapping, even though their complete
header map is available before the selected payload is read. The branch now
requires exact source-format/persisted-format admission for both families, just
as the full-resident branch does. A regression invokes the real loader boundary
with a captured metadata callback and proves both unverified public-range routes
fail before payload loading. This does not close each family’s full P02 route
trace or ABI/tooling map. P02/P15 remain open.

### P02 canonical source identity lookup for loaders and CLIs — 2026-10-17

The source catalog now exposes `source_family_for_identity`, resolving an
exact model type through the catalog's declared family identities and rejecting
unknown or ambiguous identities. The native checkpoint loader, native family
CLI, conditional CLI, conditional/flat service factories, and conditional
publisher loader use this lookup instead of separate identity maps. A
regression checks every declared family identity is unique and resolves back to
its exact source family. The gateway retains its deliberately broader
application-family dispatch and is not yet fully reconciled against this
catalog. The combined source-inventory, provenance, conditional-construction,
and public-factory cleanup suites pass (41 tests). P02/P15 remain open.

### P02 Kimi Linear direct checkpoint loader restriction — 2026-10-17

`NativeKimiLinearDecoder.from_checkpoint` used a private SafeTensors reader and
could bypass the source-matrix gate even though the catalog's exact Kimi Linear
native FP32 row remains candidate-unverified and no complete serving/ABI route
is admitted. The loader now performs complete checkpoint-header admission for
the exact `kimi_linear` identity before opening its own payload files; the
current tuple fails closed. A source regression proves the family/backend/
format row remains unadmitted and the loader does not call its local file
opener when the gate rejects. Existing native parser/conversion tests retain
their coverage behind a mocked successful admission gate. This intentionally
restricts checkpoint loading; it does not close Kimi Linear parity or P02 maps.
P02/P15 remain open.

### P03 direct Qwen auxiliary-bank construction restriction — 2026-10-17

The directory-backed conditional loader already rejected Qwen3.5/Qwen4 MTP
headers and auxiliary configuration because neither family has an admitted
draft consumer. Direct `Qwen35MultimodalDecoder` and
`Qwen4MultimodalDecoder` construction nevertheless partitioned and retained
those auxiliary tensors, bypassing the loader's inventory-to-consumer rule.
Both constructors now reject either physical auxiliary/MTP tensors or an
auxiliary configuration before staging model components. Inkling remains the
only conditional wrapper accepting this bank, with its explicit MTP executor.
This closes that direct-constructor bypass only; all-family inventory mapping
and consumer reconciliation remain open. P03/P15 remain IN PROGRESS/OPEN.

### P04/P12 Inkling grouped continuation cancellation — 2026-10-17

`InklingMultimodalDecoder.restore_cache` did not accept or forward a caller
cancellation callback, even though its grouped continuation codec can check
cancellation and its MTP restore path supports it. Grouped import now forwards
cancellation to component implementations that declare a `cancelled`
parameter (while retaining compatibility with components that do not), and
the Inkling wrapper validates and propagates the callback. Export now does the
same for cancellable writers and adapts the callback to event-style session
scopes. The Inkling wrapper marks export and restore busy for the complete
operation, so a cancellation callback cannot re-enter and mutate the owner; a
regression covers both callback reentry boundaries. The group remains spooled before publication; a regression cancels a
component export and verifies no partial group frame reached the output. A
restore regression cancels after the first component stages state and verifies
rollback of every component. The cache-restore/source-inventory/conditional-
lifecycle suites pass 51 tests. This closes the Inkling grouped-cache cancellation seam
only; broader family recovery and serving-caller cancellation/consensus remain
open. P04/P12/P15 remain IN PROGRESS/OPEN.

### P02 checkpoint preflight rejects unadmitted non-float encodings — 2026-10-17

`require_checkpoint_inventory_admission` rejected unsupported floating
encodings but had no equivalent rejection for unexpected non-floating tensor
encodings, so a checkpoint could carry an unclassified I32/U8/BOOL-like bank
alongside a supported floating route. Preflight now accepts only declared
floating source types and I64 family metadata; other dtypes fail before
checkpoint payload loading. A regression covers I32, U8, BOOL, and a missing
dtype. The source-inventory, cache-restore, and conditional-lifecycle suites
pass 51 tests. This tightens checkpoint admission only; exact route closure and
consumer mapping remain open. P02/P15 remain IN PROGRESS/OPEN.

### P02 OLMoE source-encoding axis matches checkpoint headers — 2026-10-17

The OLMoE backend catalog used lowercase `float32`/`float16` in its
source-format axis even though SafeTensors headers expose `F32`/`F16` and the
family contract already declared those values. Consequently, checkpoint
preflight could not resolve an OLMoE header into its catalog row. The native
and public-range source rows now use `F32`/`F16`, while persisted encodings
remain a separate format axis. A regression confirms an F32 OLMoE header maps
to its exact candidate-unverified row and fails closed with the intended P02
admission reason. The shared tiny OLMoE fixture now declares the exact
architecture required by the existing config reader. Source-inventory and
native/leased OLMoE suites pass 64 tests. P02 route admission remains
unverified pending complete consumer/evidence closure; P02/P15 remain
IN PROGRESS/OPEN.

### P04/P14 publisher video FPS overflow validation — 2026-10-17

`PublisherVideoInput`, selected-video manifest loading, and publisher processor
metadata validated caller FPS through separate paths; the first could raise
`OverflowError` for a huge integer, while other paths used NumPy finite checks
with inconsistent errors. These now share overflow-safe normalization to a
finite positive float before frame opening/copying; valid integer FPS is stored
canonically. Regressions cover huge integers across both input and manifest
paths, non-finite/non-positive values, and normalized timestamp projection.
The publisher-media-input, inference-route, and conditional-lifecycle suites
pass 21 tests. Cross-family video ordering/temporal semantics and full caller
media lifecycle remain open; P04/P14/P15 remain IN PROGRESS/OPEN.

### P04 K3 video geometry preflight before frame copies — 2026-10-17

`K3VideoInput` validated each frame's dimensions and copied it before checking
that all selected frames shared one geometry. A mismatch late in the sequence
therefore triggered avoidable full-resolution copies before rejection. The
constructor now validates every frame and aggregate detached-byte budget, then
checks constant geometry before copying any frame. A regression patches the
copy operation and proves a geometry mismatch is rejected without detaching
the first frame. The K3 video-input and publisher video-input suites pass 4
tests. This closes only pre-copy geometry admission; K3 temporal serving and
cross-family media integration remain open. P04/P07/P14/P15 remain
IN PROGRESS/OPEN.

### P06/P12 distributed rollback snapshot receipt admission — 2026-10-17

Distributed restore, reset, generation, and cache-recovery paths stored each
rank's local rollback snapshot without rejecting a `None` result. A later rank
failure or consensus mismatch could therefore leave one physical executor
without a rollback receipt while surviving peers restored theirs. Every
distributed operation now requires a non-null `ExecutionSessionSnapshot`
bound to that rank's exact session before proceeding to mutation, recovery, or
logical snapshot registration. Regressions cover missing, wrong-type, and
foreign-session receipts; the distributed
checkpoint, tensor-parallel consensus, and speculative transaction/cancellation
suites pass 22 tests. A rank-writer failure regression additionally verifies
that an uncommitted generation directory and latest pointer are absent after
quarantine. Process/transport failure handling and end-to-end multi-rank
consensus remain open; P06/P12/P15 remain IN PROGRESS/OPEN.

### P03 Qwen4 public ABI checkpoint-to-decoder binding — 2026-10-17

The Qwen4 publisher conditional loader now passes its admitted immutable
checkpoint inventory into public Segment/Edge construction. The Qwen4 wrapper
requires that receipt to identify the same checkpoint directory and binds the
materialized native text decoder to its exact unchanged file snapshot before
public registration. The public adapter independently re-admits the persisted
headers and requires the decoder's loader receipt to match the new snapshot,
so a same-geometry or header-only unrelated in-memory decoder cannot be
registered against the public checkpoint. A focused regression verifies
rejection before native ABI loading; `tests.test_family_source_binding` passes
5 tests, and the wrapper-level guard plus related family/source inventory and
checkpoint suites pass 41 tests (2 skipped). This closes only the Qwen4
public native text ABI provenance seam;
other Qwen4 consumers, family routes and inventory consumers remain open.
P03/P11 remain IN PROGRESS and P15 remains OPEN.

### P13 Conditional-media source artifact pin coverage — 2026-10-17

The source-only artifact inventory omitted the Qwen3.5, Qwen4, GLM5 and
Inkling conditional wrappers and the shared media-session owner from both the
family and serving pin groups; it also omitted the publisher conditional
loader from the serving group. Those files are now covered by the designated
artifact groups, with a structural regression requiring every wrapper to be
present in both groups. This improves source artifact pin completeness only:
it creates no artifact pin, runtime observation, cache/placement receipt, or
comparison evidence. The related source-inventory and Qwen4 provenance suites
pass 41 tests (2 skipped). P13 remains IN PROGRESS and P15 remains OPEN.

### P13 Windows source-inventory receipt portability — 2026-10-17

The source inventory compared `st_ctime_ns` from `Path.stat()` and `os.fstat()`
as if both had POSIX semantics. On Windows these differ (path stat reports
creation time while descriptor stat reports last-write time), so the inventory
rejected an unchanged source file and could not emit a source receipt. `_stamp`
now uses `st_birthtime_ns` when available and retains POSIX ctime otherwise;
a regression covers both shapes. The complete source inventory now runs on
this checkout with zero missing entries across every declared area, emits
`source_only=true` and `executed=false`, and returns a transient digest
`18e06e72cbd64ef552d58c72f5b750f173336c1fd4070790735bdfd623bddf8e`.
That receipt is not persisted/pinned and does not establish runtime or numeric
evidence. The shared native-library stamp used by whole-model artifact hashing
now uses the same portable birth-time normalization; a real temporary-file
digest regression passes. The source-inventory and related Qwen4/source-scope suites pass 41
tests (2 skipped). P13 remains IN PROGRESS and P15 remains OPEN.

### P02 Exhaustive candidate-route fail-closed check — 2026-10-17

The generated source matrix currently contains 1,120 catalog rows representing
1,048 distinct exact six-field family/model/backend/arithmetic/source/persisted
tuples across 50 family/backend groups. None is verified-admitted. An exhaustive
regression now submits every distinct tuple and group to the runtime admission
helpers and requires rejection; it also proves a mutable exact-row lookup
cannot change the cached catalog's admission state. The catalog snapshot and
exact-tuple resolver are cached/indexed internally, reducing the exhaustive
check to a fraction of a second while public results remain detached. This
confirms an enforced restriction on all currently generated candidates, not a
complete end-to-end trace or runnable support route. P02 remains IN PROGRESS;
P15 remains OPEN. The source-scope, route-binding, Qwen4 and K3-focused suites
pass 64 tests (3 skipped).

### P11/P14 Qwen4 native routing-observer lifecycle regression — 2026-10-17

Added callback-level regressions for the public Qwen4 routing observer. An
observer that attempts to reenter native hidden-state execution is rejected,
the callback failure is propagated as an observation failure, and the scope
is released. If native observer teardown reports failure, the callback remains
retained, new observation scopes are denied, and cleanup can be retried on its
owning thread before unregistering the native model. The lifecycle, source
binding and related Qwen4 suites pass 10 tests (2 skipped). This covers this
public callback seam only; family-wide ABI/reentry and serving-caller closure
remain open. P11/P14 remain IN PROGRESS and P15 remains OPEN.

### P14 Multimodal service checkpoint lifecycle ownership — 2026-10-17

`MultimodalSessionExecutor` previously serialized text/media forwards but let
the service call decoder snapshot/restore/release/reset and cache export/import
through `__getattr__`, outside its lifecycle owner. Those checkpoint operations
now run under the same operation guard: close waits for active cache I/O and
same-thread callback reentry is rejected. Cache cancellation callbacks are
forwarded only to codecs that declare support; requesting cooperative
cancellation from a codec without it fails explicitly instead of being
silently dropped. The lifecycle, conditional cleanup, and generation-admission
suites pass 22 tests. Source inventory remains zero-missing/source-only and
emits transient digest `8679ba274a539bb4602f4fe65d65a1eebf3bb8ee0c8ebe581c6a2777c42147a8`.
This closes only the multimodal service owner's checkpoint-lifecycle seam;
family-specific recovery, complete caller cancellation and atomic publication
remain open. A failed decoder close now first marks the wrapper unavailable,
but a separate completion flag permits cleanup retry; a regression verifies
requests are rejected until retry succeeds. P14 remains IN PROGRESS and P15
remains OPEN.

### P14 Publisher stream cancellation callback ownership — 2026-10-17

The publisher stream previously polled the caller's cancellation callback from
both the generation worker and iterator consumer. A stateful or thread-affine
callback could therefore be entered concurrently, even though cancellation
events were serialized. The iterator consumer now exclusively polls the user
callback and signals the worker through the stream stop event; buffered events
are still rejected before dispatch when cancellation is observed. Focused
stream, multimodal lifecycle and conditional cleanup regressions pass 20 tests.
The source inventory has zero missing declared area entries and reports
`inventory_sha256=9c432cc7bfd5c51e0ef2a4c5032f5ccfb3c2ebe59d6f8be744dfff9b151722af`,
`source_only=true`, `executed=false`; that digest is transient source inventory,
not runtime evidence. This closes only the publisher stream cancellation
callback ownership seam, not cancellation of an in-flight native kernel or
other serving callers. P14 remains IN PROGRESS and P15 remains OPEN.

### P04 Publisher media placeholder admission — 2026-10-17

The publisher preflight rejected caller-inserted image markers and rejected
video markers when the request also contained a video attachment, but a
configured video control token could be placed in plain user text when no
video was attached. That could create a prompt/media ordering mismatch in
families that interpret the token specially. The preflight now rejects either
configured media control token in caller message text regardless of attached
media kind; only the gateway-appended markers may represent media. A focused
provider-preflight regression plus publisher media/stream and gateway suites
pass 22 tests. The source inventory has no missing declared areas and reports
`inventory_sha256=be46a5613bc5c2e0bab63da8c6b42014bdf164f379e21d90f7e5d98f05c207fa`,
`source_only=true`, `executed=false`. This closes that publisher token-spoofing
case, not arbitrary in-message media placement or cross-family temporal/media
parity. P04/P14 remain IN PROGRESS and P15 remains OPEN.

### P13 Local OLMoE execution-evidence prerequisite audit — 2026-10-17

The checkout contains the three OLMoE model shards, index/config/tokenizer,
and expert manifests, but this environment has no installed Torch,
Transformers or Safetensors package and no discoverable C/C++ build tool
(`cmake`, `ninja`, `cl`, `gcc`, `clang` or `cc`). No prebuilt native library
is present beside that checkpoint. The row-specific OLMoE candidate adapter
requires that existing library as a pinned dependency, and a numerical report
also requires an independent unchanged oracle; therefore the local checkpoint
files alone cannot produce a valid observation or placement receipt. No
dependency was installed, code was built, or model/device execution attempted
in this audit. This is a verified prerequisite gap, not a source defect and not
evidence that other model rows have equivalent prerequisites. P13/P15 remain
OPEN until a permitted environment supplies the pinned runtime/library and
independent oracle and produces actual receipts.

### P09 Packed conversion cleanup and Q5 signed-scale consumers — 2026-10-17

The grouped-INT4, Q5_0 and Q5_1 converters created a raw `mkstemp` descriptor
then could enter cleanup after `os.fdopen` failed without closing it. They now
retain descriptor ownership until `fdopen` succeeds, close it on that failure
path, and preserve cleanup errors while removing staging files. The Q5_0/Q5_1
decoders and scalar/native kernels also rejected finite negative FP16 deltas,
contradicting the format regression fixtures and the signed-scale execution
equation. They now reject nonfinite metadata but preserve signed deltas and
affine minima through decode and direct-byte execution. Grouped INT4, Q5_0,
Q5_1 and injected descriptor-failure suites pass 22 tests, including actual
compiled Numba Q5_0 and Q5_1 kernel execution on signed finite scale payloads
and numeric comparison to each format's scalar consumer. This is CPU-kernel
format verification, not a complete family loader/consumer chain or model
comparison. The source inventory
reports no missing declared areas,
`inventory_sha256=deb62999a2eacba38a9b8c9e2969ae4368b4af654f2318739327e49616c4b9e4`,
`source_only=true`, `executed=false`. This closes these specific converter and
Q5 format-consumer seams; complete P09 loader/consumer mapping, other converter
paths, placement and no-fallback evidence remain open. P09/P10/P15 remain
IN PROGRESS/OPEN.

### P06/P12 Distributed recovery driver ordering and cleanup — 2026-10-17

Added focused fault-injection coverage for the rank replacement `_RecoveryDriver`:
application work is invoked only after each committed checkpoint is admitted,
restored, and re-admitted; a later-rank restore failure closes every opened
session in reverse order; and a close failure is attached to the original
restore failure without skipping cleanup of remaining sessions. The new driver
tests plus distributed checkpoint-error and tensor-parallel consensus suites
pass 14 tests. This validates coordinator behavior for surviving workers only;
transport/process loss still poisons the group and requires fresh-rendezvous
restart. No process-spawn recovery was exercised. P06/P12 remain IN PROGRESS
and P15 remains OPEN.

### P13 Whole-model runtime metadata is mandatory — 2026-10-17

The generic whole-model runner previously allowed an adapter with no
`runtime_metadata()` method to produce a numerically passed report with an
empty runtime section. The runner now requires a callable method returning
nonempty, failure-checked runtime details, and reports
`runtime_evidence_verified=false` explicitly. A focused contract regression
covers missing, empty, and populated runtime metadata. This closes only the
source-side omission path: adapter-supplied runtime details are not an
independently verified device-placement receipt, nor a signed attestation.
Pinned observations, cache/placement receipts and comparisons from a permitted
real model/runtime environment are still absent, so P13 remains IN PROGRESS
and P15 remains OPEN.

### P02/P03 Public service-factory gate call-chain regression — 2026-10-17

An AST-based source regression now enumerates every top-level public
`create_*`/`open_*` execution-service factory in
`execution_factories.py` and requires its local call chain to reach exact
source-matrix admission, checkpoint/header inventory admission, or an explicit
unverified in-memory route restriction. This also covers wrappers that defer
admission to their guarded backing factory. The focused regression passes.
It proves only factory-module gate reachability; it does not prove that every
optional/shared/MTP/expert/media tensor reaches the correct runtime consumer,
nor does it cover every CLI, gateway, recovery, or tooling caller. P02/P03
remain IN PROGRESS and P15 remains OPEN.

### P04/P14 Shared conditional media rejects nonfinite processor arrays — 2026-10-17

The shared multimodal service owner now checks floating processor-array chunks
for NaN/Inf while it performs its existing bounded, cancellation-aware detach.
Invalid arrays are rejected before family decoder dispatch, and the existing
staging exception path releases any partially copied inputs. A regression
proves NaN and positive infinity never reach the decoder. The focused
multimodal-lifecycle, conditional-cleanup, and publisher-media-input suites
pass 22 tests. This is shared-boundary numeric preflight only; it does not
replace each family’s geometry/placeholder validation or close temporal media,
caller ordering, serving recovery, or atomic publication. P04/P14 remain IN
PROGRESS and P15 remains OPEN.

### P13 Comparison configuration descriptor cleanup — 2026-10-17

`wholemodel_validation.configuration.write_configuration` previously leaked
its raw `mkstemp` descriptor if wrapping it with `os.fdopen` failed. The
failure path now closes the descriptor while preserving the primary error and
records a retryable cleanup callback if close itself fails; the staging file
is still removed by the existing cleanup path. Fault injection verifies the
descriptor is closed and neither staging nor destination remains. The focused
regression passes. This closes one tooling cleanup path only; actual artifact
pins, observations, cache/placement receipts, and model comparisons remain
runtime-evidence requirements. P13 remains IN PROGRESS and P15 remains OPEN.

### P05/P14 Final generation cancellation restores physical state — 2026-10-17

`NativeExecutionService.generate` previously performed its final cancellation
check in the `try` statement's `else` clause. If cancellation arrived during
the final executor-position validation, that exception bypassed the handler
that restores the transaction snapshot: logical history stayed unchanged but
the physical model cache retained generated tokens. The final cancellation
check now runs inside the rollback-protected `try` block. The regression
injects cancellation from the final position query and asserts both logical
history and physical executor tokens return to the prior state; it passes.
This closes that shared generation boundary only, not all speculative callers,
stream publication atomicity, or distributed failure behavior. P05/P14 remain
IN PROGRESS and P15 remains OPEN.

### P12/P13 Shared owned-descriptor transfer for cache and evidence files — 2026-10-17

Additional raw descriptor handoffs in replay-checkpoint save/load, service-cache
save/load, source-inventory hashing/report publication, and the K3 media
comparison report used bare `os.close` in failure branches; a close error
could mask the original validation or wrapping failure. These sites now use a
shared owned-descriptor transfer helper, while pre-wrap file validation closes
through the same primary-error-preserving cleanup policy. Fault-injection tests
cover descriptor closure and close-error notes. The focused descriptor,
source-inventory, cache restore, durable eviction, and prior descriptor
cleanup suites pass 57 tests. This closes only the audited call sites; other
module-specific raw descriptors remain to audit, and no model/runtime evidence
is produced. P12/P13 remain IN PROGRESS and P15 remains OPEN.

The expanded native-checkpoint suite was also attempted after this change; its
checkpoint cases currently fail before exercising I/O because the active P02
matrix rejects the Qwen4 FP32/F32 route as `candidate-unverified`. This is a
pre-existing scope/admission mismatch, not evidence that the descriptor edits
failed. Related continuation, distributed-manifest, media-input, and descriptor
ownership suites pass (14 tests). The candidate route remains restricted until
its source path is actually admitted or its tests use an admitted tuple.

### P12/P13 Remaining recovery and publisher descriptor transfers — 2026-10-17

Extended the primary-error-preserving owned-descriptor helper to batch cache
recovery, distributed generation manifests, K3 and V4.1 publisher checkpoints,
native-library identity reads, native checkpoint bundles, continuation
catalogs, publisher media, and publisher artifact hashing. Failed wrapping now
closes the raw descriptor without replacing the original error; pre-wrap
validation failures use the same cleanup policy. Focused ownership, generation
checkpoint, source inventory, cache restore, and durable-eviction suites pass
(56 tests). A static scan now leaves only the helper itself and converter
callers with dedicated cleanup logic; those still require call-site review.
This is not distributed process-loss recovery or actual pinned placement/cache
evidence. P12/P13 remain IN PROGRESS and P15 remains OPEN.

### P09/P10 selected-device E8 activation transform — 2026-10-17

Implemented `device_e8_rotation.rotate_device_e8` and connected it to the
shared selected-device expert consumer inherited by Mixtral, Qwen3.5 and Qwen4.
The consumer requires exact persisted lattice metadata and aligned dimensions,
applies the forward transform to hidden input and intermediate activation before
the corresponding packed projections, checks cancellation/finite values, and
accounts for bounded same-device scratch. The source route now binds the E8
decoder and activation helper as separate owners. Added focused tests for
transform orientation, malformed transforms, work-budget rejection and
pre-allocation cancellation. The E8, source-scope and distributed teardown
focused suites ran 46 tests successfully, with 3 actual-Torch numerical tests
skipped because Torch is not installed in this environment; two NumPy-backed
transform/budget/cancellation tests and the source-route checks ran. Thus the
implementation is source-complete for
this narrow seam, but numerical Torch execution, accelerator placement and
end-to-end model behavior remain unverified. Packed-manifest factory routes
still fail closed for missing source-dtype/exact-tuple provenance; P02/P09/P10
remain IN PROGRESS and P15 remains OPEN.

### P04/P14 K3 video async-freeze adoption — 2026-10-17

Added a private `K3VideoInput._from_owned_frames` path for the native serving
gateway. It validates detached owning immutable RGB arrays, frame count,
geometry, timestamps and aggregate budget, then adopts those exact arrays.
`NativePublisherProvider._freeze_decoded_media` uses it for frames already
copied with chunk-level cancellation checks, preventing the public
caller-ownership constructor from making a second opaque full-frame copy.
Regression coverage proves the original caller frame is detached and the
public constructor is not re-entered. The focused media, ordering, source-scope,
E8 and distributed-cleanup selection passes 55 tests (3 optional actual-Torch
cases skipped). This closes one shared async-staging allocation path only;
other caller/family media lifecycle, placeholder ordering and recovery remain
open. P04/P14 remain IN PROGRESS and P15 remains OPEN.

### P02 Qwen4 public-range source identity — 2026-10-17

`NativeQwen4PublicAdapters` reused the Mixtral ABI transport subclass but had
inherited `('mixtral', 'mixtral')` for its source-admission identity. It now
declares the exact `('qwen4', 'qwen4_exp_text')` row before the inherited
constructor performs checkpoint admission. The Qwen4 source-binding,
public-lifecycle and source-catalog suites pass 47 tests. This closes one
family-identity mismatch only; the route remains candidate-unverified and
family/backend/format caller tracing remains open. P02 remains IN PROGRESS and
P15 remains OPEN.

### P02 direct publisher public-adapter admission — 2026-10-17

`NativePublisherPublicAdapters` can be constructed directly, independently of
`compose_public_publisher_service`. It now requires its exact K3 or DeepSeek
V4.1 public-range family/backend/arithmetic admission and a bounded checkpoint
header inventory before reading the native library identity or loading the
ABI. Regression coverage proves an unadmitted direct route fails before the
native-library read; a second regression proves malformed/unadmitted checkpoint
headers fail before the native-library read. The combined family-source-binding
and source-inventory suites pass 49 tests. This closes one factory-bypass seam only; the public
routes remain unadmitted in the current matrix and P02/P15 remain open.

### P02 lazy public-range checkpoint binding — 2026-10-17

The Mixtral public-range base and its Qwen3.5/GLM-DSA subclasses now retain
the admitted checkpoint snapshot and assert it is unchanged before and after
lazy Edge and Segment range construction, and before service composition. If
the source changes during construction, the partial owner is closed; a failed
close remains in the adapter's retry inventory with the original source error
preserved. Focused family-binding, source-inventory, Qwen4 public-lifecycle and
K3 suites pass 76 tests with one optional skip. K3/V4.1 payload records also
carry their source root and immutable checkpoint file stamps; the public
adapter requires those pins and payload geometry to match its requested model
directory before loading the ABI. A regression rejects a K3 payload paired
with a different shard stamp before the native library read. These changes
close source-identity races and a payload/directory mismatch seam for these
specific public entrypoints only. Other model loaders and callers, runtime
evidence, and all remaining P02/P03 closure remain open; P15 stays OPEN.

### Workspace verification pass — 2026-10-17

Ran the complete Python discovery suite after repairing the FP8 findings from
the prior run: `1578 tests` passed in 596.856 seconds, with 95 environment or
optional-dependency skips. The fixes bind OLMoE's already implemented FP8
block-128 FP32-scale format into its family persisted-format policy and align
the native FP8 reader with the reference decoder's finite signed/zero FP32
scale contract. The focused FP8/OLMoE/source-inventory selection passed 45
tests. Frontend TypeScript checking passed; Vitest passed 140 tests in 22
files; Playwright passed all 112 browser tests against an isolated temporary
port; and the guarded production build succeeded. The build reported large
Monaco worker/chunk-size warnings, but verified that the shared `web` bundle
was unchanged and demo fixtures were excluded. `validate_phase1_source` also
returned true on the final tree; this is structural source-catalog validation,
not runtime evidence.

This pass does not close the wider Phase 1 obligations. Optional skips mean
some dependency-backed numerical/device cases did not execute; no device
placement probe or benchmark was performed, and P13 row-bound observations,
cache/placement receipts and comparisons remain uncaptured. P02/P03 family
route and inventory closure, P04–P14 caller/runtime obligations, and P15 final
reconciliation remain open. Do not treat a green test suite or structural
validator as Phase 1 completion.

### P14 research event-stream missing-session cleanup — 2026-10-18

The `/api/research/sessions/{id}/stream` handler previously passed a lazy
generator directly to the generic SSE writer and caught `KeyError` only around
generator construction. A nonexistent session therefore failed after HTTP
200/SSE headers, producing an unhandled `event stream producer failed`
traceback. The route now checks session existence before sending headers and
returns the normal 404 envelope; if deletion races with stream iteration, the
route emits one terminal SSE error event instead of logging a producer failure.
Regressions cover both paths. Research lifecycle tests pass 9 tests; the
affected server, research, chat-research and route suites pass 29 tests.

The full browser suite intermittently failed Vault modal/demo-timer cases under
eight-worker CPU load, while the focused Vault set passed with two workers. CI
Playwright concurrency is now capped at four workers; the full isolated-browser
suite then passed all 117 tests, and frontend type-checking passed. This is a
test-runner reliability adjustment, not a broader P14 lifecycle closure. Other
serving callers, streaming semantics, cancellation and P14/P15 reconciliation
remain open; no Phase 1 completion claim is made.

### P12 distributed recovery cleanup diagnostics — 2026-10-18

`_RecoveryDriver` previously called `repr(cleanup_exception)` while trying to
close every opened replacement-rank session after a restore/driver failure. An
exception with a failing `__repr__` could interrupt the cleanup loop, skip the
remaining sessions, and replace the primary failure. Cleanup diagnostics now
use a bounded type-only summary and note attachment is guarded so teardown
continues. A regression injects an unprintable cleanup exception and verifies
remaining sessions are still attempted and the original restore failure remains
primary. Distributed recovery/checkpoint and research lifecycle focused suites
pass 20 tests. This closes one recovery error-cleanup defect; P06/P12 consensus,
process-loss behavior, family physical-state closure and P15 remain open.

### P14 inference disconnect watcher starts after body parsing — 2026-10-18

The synchronous inference route started its cancellation watcher before
`read_json()` consumed the request body. Because the watcher peeks the same
HTTP socket as the handler, it could race body consumption and issue a blocking
peek on an empty persistent connection. The watcher now starts only after a
successful JSON body read. An HTTP-level regression records actual body parsing
before watcher startup. The full server, inference-route and commercial
security suites pass 31 tests; the focused distributed recovery/checkpoint and
research lifecycle set also passes 22 tests. This closes the shared synchronous
inference watcher ordering race only; other streaming, caller lifecycle and
P14/P15 requirements remain open.

### P14 HTTP request-body/disconnect ordering and suite resource cleanup — 2026-10-18

The full Python suite first exposed a test-fixture resource leak: auth HTTP tests
shut down server threads but left listening sockets open. Under full discovery,
the backup-restore unauthorized request intermittently ended in a Windows
connection abort, and the assertion then cascaded because it expected an
`HTTPError`. Those fixtures now call `server_close()` after shutdown/join. The
same pass also closed the inference watcher/body-read race recorded above.
Focused auth/server/inference/security tests pass 34 cases. After the fixture
fix, full Python discovery passes 1,582 tests with 95 skips in 529.733 seconds.
This is source/test reliability evidence only: remaining socket ResourceWarnings
in other fixtures and optional skips remain, no device placement/runtime
evidence was obtained, and the Phase 1 P02–P15 requirements remain open.

Follow-up fixture audit closed listening sockets in the calendar-sync,
communication-resilience, document, local-AI lifecycle and response-envelope
HTTP tests as well. The nine affected HTTP/auth modules pass 66 tests. The
full-suite result above was collected after the auth fixture fix and before
these additional server-close-only test changes; no second full discovery run
was needed to validate their scoped behavior.

### P02 direct Qwen3.5 multimodal loader source gate — 2026-10-18

`load_native_multimodal_decoder()` called the private SafeTensors reader with a
family geometry callback but omitted the P02 source-format admission used by
`read_native_multimodal_checkpoint()`. Direct callers could therefore reach
payload loading for a source tuple still marked candidate-unverified. Both
entrypoints now share one callback that validates the family inventory and
requires the exact source-format row before tensor reads. A regression invokes
the direct loader callback against the live source catalog and verifies the
candidate tuple is rejected. The multimodal and source-scope suites pass 41
tests with one optional-oracle skip. This closes one direct-loader admission
bypass only; P02 family route tracing, P03 consumers and P15 remain open.

### P02 optional Transformers/Accelerate MoE execution restriction — 2026-10-18

The HF MoE model-type allowlist and provider artifact scan did not establish a
complete P02 family/backend/format route. `HFMoeAdapter.load()` now requires an
exact source-scope family/backend admission before importing Transformers or
reading checkpoint tensors. `HFMoeProvider.available()` applies the same gate
before scanning tensor files, and the CLI rejects the route before loading a
tokenizer. Manifest and placement metadata remain inspectable; runtime
execution is restricted until a loader-to-logits route is catalogued and
reviewed. Differential runtime tests are explicitly skipped while this route
is restricted. The focused HF MoE, provider, CLI, family-conformance and
source-inventory suites pass 60 tests with 2 skips. This is a concrete
restriction for this optional backend only; the broader P02 matrix and P15
remain open.

The V4.1 serving session also applies the safe reporting policy when target
and draft resets both fail and owner cleanup fails during a request boundary.
A regression uses secondary exceptions whose `str()` and `repr()` raise and
confirms the first reset error remains surfaced. The durable-eviction,
generation-admission, conditional-session, streaming-cancellation and
speculative-boundary focused suites pass 40 tests after this serving-path
addition.

The same cleanup-reporting flaw was closed at public executor construction and
ownership-transfer boundaries across the shared family factory module. This
includes native, conditional, publisher, Mixtral, DSA and checkpoint-backed
construction cleanup. A regression confirms an unprintable cleanup exception
does not replace the original public-factory error. The factory, family-binding,
durable-eviction, conditional-session and speculative-boundary focused suites
pass 54 tests after this change; a fresh full-suite rerun is pending.

### Cross-project regression and build verification — 2026-10-18

Latest Python discovery passes 1,587 tests with 95 skips. Frontend-second
passes 141 tests, its TypeScript check passes, and its production build
completes. The legacy frontend suite passes all 162 tests when run serially;
an initial run overlapped the second frontend test, typecheck and build and
reported 12 chat-test timeouts, which did not reproduce once resource-heavy
jobs were serialized. The frontend/API coverage audit currently catalogs 654
routes across 49 groups (249 primary and 257 second-frontend references) and
writes its two gap reports under `frontend-second/`. Build output reports the
large Monaco editor chunk (3.71 MB), a bundle-splitting/performance follow-up,
not a build failure. Python discovery still emits several `ResourceWarning`s
for unclosed HTTPError response objects in test fixtures; their allocation
sites are being traced before deciding the narrow cleanup. A CUDA
probe-scoped certification attempt recorded `unavailable` at preflight because
this host is Windows and the runner requires Linux; no CUDA device commands
ran. OpenVINO device enumeration also could not run because `openvino` is not
installed. No runtime placement evidence or model benchmark was gathered.
These results do not close the
remaining P02–P15 source requirements or P15.

The full-project defect audit separately found that the legacy `/api/audit`
route was absent from the enforced-auth scope map while its unfiltered query
can return all projects. It now requires `security:admin`; an HTTP regression
checks 401 for anonymous, 403 for `security:read`, and successful global feed
access only for `security:admin`. The focused auth/security set passes 9 tests.
This fix is outside Phase 1 engine closure and does not define project-level
membership isolation.

After the serving rollback-diagnostic changes, full Python discovery completed
with 1,592 tests passing and 95 skipped (`.venv\\Scripts\\python.exe -m unittest
discover -s tests`). It emitted the known HTTPError response `ResourceWarning`s
and a dependency deprecation warning, but no test failures. Runtime model/device
evidence was not part of this run; P02–P15 and P15 remain open.

### P04/P05/P12 cleanup diagnostics preserve primary failures — 2026-10-18

Speculative target/draft rollback, reset, DSpark evidence/proposal rollback,
DSpark cache recovery, V4.1 checkpoint restore and MTP RNG rollback attempted
to format secondary cleanup exceptions with arbitrary `str()`/`repr()` calls.
The V4.1 image processor had the same issue while closing resized and converted
image wrappers. An unprintable secondary exception could replace the original
cancellation, processing or state-mutation error. These paths now share a
bounded exception-type-only summary and a best-effort built-in `add_note`,
without invoking exception formatting and without allowing note failures to
escape. The native execution service now applies the same policy to executor
cleanup, session restore, prefill/scoring rollback, generation/cache recovery,
replay and snapshot cleanup. Regressions cover cancellation plus target-restore
failure, dual reset failure, multiple image-close failures, and an unprintable
executor rollback failure while preserving their primary errors. The combined
speculative, serving, durable-eviction, generation-admission, V4.1 image,
checkpoint-cleanup, V4.1 checkpoint and distributed-checkpoint suites pass 42
tests. This closes diagnostic masking on the audited paths only; other
`add_note` cleanup sites remain to be reviewed, P04/P05 caller-by-caller
lifecycle and state/RNG reconciliation, P12 distributed recovery, and P15
remain open.

### Follow-up full suite after factory cleanup hardening — 2026-10-18

After extending the safe diagnostic helper through shared public executor
factories and ownership transfer, full Python discovery completed with 1,593
tests passing and 95 skipped in 594.118 seconds. Expected HTTPError response
`ResourceWarning`s and the Chroma deprecation warning remain. The focused
factory/ABI/session/speculation set passed 54 tests. No runtime model, device
placement, or benchmark evidence was produced; remaining P02–P15 items and P15
remain open.

The repeated full run exposed leaked `urllib` HTTPError response handles in
auth-enforcement, document, calendar-sync and local-AI lifecycle tests. Those
fixtures now close error responses (and auth-enforcement success responses) on
all asserted paths. Their combined warning-enabled subset passes 23 tests with
no `ResourceWarning`; the full 1,593-test result above predates only these
test-fixture lifetime edits, while no production code changed after it.

Additional provider, public-publisher, K3, media-input/processor, batch, and
JSONL-serving cleanup notes now use the same safe failure reporter. Their
focused media, streaming, factory, conditional-session, Olmoe and K3 staging
tests pass 38 tests. The prior full-suite run predates these production edits;
a final full discovery rerun is required before reporting suite status.

The following full discovery rerun passed 1,593 tests with 95 skips in 537.019
seconds after the production cleanup-reporting changes. It exposed one
intermittent Windows loopback abort in an auth-rejection fixture; the test
passed alone, and that helper now retries once only on a connection abort for
pre-dispatch rejected requests. The next full run passed. Remaining HTTPError
ResourceWarnings were traced to the auth, communication, research, integration,
document, calendar and local-AI test fixtures; these response handles now close
on the error paths. A 45-test warning-enabled subset passes with no
ResourceWarning. No full-suite rerun was performed after these test-only
resource-lifetime changes.

Distributed execution, grouped continuation, paged cache, durable continuation,
K3 preprocessing, Inkling MTP, expert packing and scalable scheduler rollback
diagnostics were then migrated to type-only safe notes. The focused distributed,
cache, K3 and checkpoint tests pass 53 tests with 8 optional skips; the MTP,
scheduler and expert subset passes 46 with one optional skip. A subsequent full
run passed 1,593 tests with 95 skips in 558.046 seconds; the one HTTPError 404
warning was traced to `test_response_envelope`, which now closes its response in
`finally`. Its warning-enabled four-test module passes cleanly. No full rerun
was performed after that test-only response-handle fix. The source audit still
finds 106 formatted-exception `add_note` sites elsewhere in the adaptive engine
for review; the edits here do not close all cleanup-diagnostic paths or P02–P15.

### P14 OpenAI/Anthropic local SSE streaming path — 2026-10-18

The local compatibility routes now accept `stream=true` for chat completions,
text completions, and Anthropic messages. Native provider streams receive the
same bounded queue timeout and disconnect cancellation signal as non-streaming
requests. The route primes one event so admission and pre-first-token failures
still return normal JSON errors/429 responses before SSE headers. OpenAI
responses emit delta/final chunks and `[DONE]`; chat `stream_options` supports
only boolean `include_usage`. Anthropic responses emit the named message and
content-block event sequence with input/output usage and finish reason. The
HTTP writer preserves protocol-specific frames and does not append its generic
JSON terminal event to these streams; SSE responses include the bounded queue-wait
header derived from the first native event. Provider iterators are closed on normal
completion, failure, or disconnect. Regressions cover both protocols, usage,
queue-timeout forwarding, pre-header overload behavior, and live HTTP SSE
serialization. The focused inference, publisher stream-cancellation, model
gateway, and server serialization tests pass (35 tests across the runs in this
change). Text stop sequences, tool calls, Anthropic stream options, broader
request semantics, and runtime-model evidence remain restricted or unverified;
this narrows but does not close P14, P13, or P15.

The native first-token events now carry the request's measured queue wait, and
the local OpenAI/Anthropic SSE response includes the same bounded
`X-Queue-Wait-Ms` header as the synchronous JSON response. The event-stream
writer only forwards the whitelisted queue/retry headers. A real HTTP server
regression checks SSE content type, queue-wait header, protocol frame bytes,
and absence of the generic JSON terminal event. The latest focused
inference/provider/gateway/server/security run passes 45 tests after this
addition. A subsequent full discovery after this header change passes 1,599
tests with 95 skipped in 591.439 seconds. This verifies source/test integration,
not model runtime, placement or broader caller closure; P14/P15 remain open.

### P02/P03 raw checkpoint reader cannot omit inventory admission — 2026-10-18

The private `_read_native_checkpoint` primitive previously treated its
metadata-admission callback as optional and could parse/materialize tensors
without a family inventory or explicit fixture contract. It now rejects a
missing callback before opening checkpoint files or reading payloads. All six
production call sites in the adaptive engine pass a metadata gate; a static
AST regression fails if a future call site omits it. Parser-only unit fixtures
pass an explicit isolated callback rather than silently exercising the
production helper ungated. The source-scope, native checkpoint, multimodal
checkpoint and V4.1 fixture suites pass 70 tests with 9 optional skips. This
closes an internal raw-reader bypass only; source rows remain unadmitted,
P02/P03 family route/consumer reconciliation and P15 remain open. A full
Python discovery run after this change passes 1,601 tests with 95 skipped in
597.212 seconds; that verifies regression integration, not runtime model,
device placement or P13 evidence.

Earlier full discovery after a truncated-stream guard and a Windows loopback
test fixture retry passed 1,599 tests with 95 skipped in 563.339 seconds; this
result is superseded by the 1,601-test full run above. The retry
is limited to WinError 10053 on an authorization request rejected before
protected handler dispatch. Compatibility-preserving empty `stream_options`,
defensive close of an empty iterator, and propagation of queue-wait metadata to
SSE response headers are covered by a final focused 45-test inference/provider/
gateway/server/security run; route lint and the scoped whitespace check pass.
A broader lint invocation still reports
style/import findings in already-massively-edited provider/runtime files and
is not a clean-project lint result. The suite does not provide model runtime,
device placement, benchmark, or pinned P13 evidence; P02–P15 are not thereby
closed.

### P14 capability audit corrected to match streaming routes — 2026-10-18

The public Colibri capability map and structural source catalog still described
OpenAI/Anthropic compatibility as synchronous-only and the provider queue as
non-streaming-only after the local SSE routes landed. That stale audit data is
now corrected: it declares only the implemented text SSE subset, OpenAI chat's
`stream_options.include_usage` subset, route-scoped Anthropic API key support,
and bounded JSON/SSE queue health. Unsupported controls, distributed queue
semantics and full protocol parity remain explicitly partial or unavailable.
The new capability regression and source-scope repository validation pass (3
tests); `family_conformance` reports nine source families and retains
`phase1_closed=false`, `executed=false`, `numerical_parity_verified=false`, and
`platform_certified=false`. Full discovery after this catalog/test correction
passes 1,602 tests with 95 skipped in 581.259 seconds. No model runtime,
placement, numerical parity, benchmark, or P13 evidence is implied. Python
`compileall` and explicit imports of the capability/source-scope/conditional
loader modules pass. `npm run build:web` passes (Vite reports oversized
chunks); a device-enumeration probe was attempted but OpenVINO is not installed
in this environment, so no device placement or benchmark can be claimed. The
final source-scope suite passes 39 tests; scoped Ruff and whitespace checks
pass for the capability/catalog files. A whole-worktree whitespace check still
finds trailing spaces in the unrelated, pre-existing `web/app.js` changes.

### P02 V4.1 bank route binding and factory provenance — 2026-10-18

The V4.1 provider and factories required exact `BF16→bf16`, `FP8→fp8`, and
`FP4→fp4` tuples that the family persisted-format policy did not declare, so
the matrix lookup failed before source admission. Those three formats are now
in the V4.1 policy. Nine exact local source tuples (three bank routes for
native, device, and public-range execution) carry an explicit P02 source-chain
binding. V4.1 distributed routes now reject with the concrete outstanding
process-loss/cancellation/consensus reason. `require_checkpoint_inventory_admission`
uses the full V4.1 publisher contract inventory rather than the generic dense
float-loader route. Public factories also validate the private `load_v41`
provenance token before resolving/opening a runtime library; this fixed a real
wrong-type/provenance ordering bug exposed when the valid local P02 tuples were
admitted. P13 evidence fields remain not captured and execution/parity/platform
flags remain false.

The exact source matrix now reports 1,300 rows: 887 candidate-unverified, 404
rejected, nine source-admitted, and 86 backend-restricted combinations. Source
scope, V4.1 fixture, family binding, and ABI checks pass (84 focused tests,
including checkpoint-preflight and native/device gateway route regressions;
the source-scope module passes 42 tests). Full discovery after the V4.1 gate
changes passes 1,604 tests with 95 skipped in 570.295 seconds; the added
gateway-only regression passes separately. The
source-scope CLI audit reports nine families, 1,300 rows and still reports
`phase1_closed=false`, `executed=false`, `numerical_parity_verified=false`, and
`platform_certified=false`. Scoped Ruff and whitespace checks pass for the
source catalog/test edits. These checks establish source admission only, not
model runtime correctness, hardware placement, benchmark performance, or P13
receipts; P02/P06/P13/P15 remain open overall.

The latest Mixtral route review adds exact source admission for local dense FP32
execution only: nine tuples cover native, explicit-device, and public-range
readers for F32/F16/BF16 tensors. Packed Mixtral experts remain restricted
because the manifest binds architecture metadata but not exact source tensor
content. Public-range and directory factories now reject packed providers
before library or payload work; the unreachable streamed-factory construction
body was removed. The exact matrix now has 1,334 rows: 824
candidate-unverified, 322 rejected, 86 backend-restricted, and 102
source-admitted. The updated source-matrix and fail-closed provider regressions
pass for Mixtral and GLM-DSA. The latest full discovery passes 1,615 tests with
95 skipped in 602.793 seconds.
Structural audit reports `phase1_closed=false`, `executed=false`,
`numerical_parity_verified=false`, and `platform_certified=false`; runtime/P13
evidence and P15 remain open. Scoped `git diff --check` is clean. Ruff reports
95 E701/E702 compact-statement style findings across the selected legacy-heavy
modules; no broad formatting rewrite was applied as part of these route fixes.

The follow-up GLM-DSA audit admits only its exact dense F32/F16/BF16-to-FP32
native, device, and public-range directory paths (nine matrix rows). Its
packed-provider paths now reject before directory payload/runtime work in the
native/device factories, device loader, range loader, and public adapter,
because the available provider identity is configuration-bound but not proven
against source tensor content. The source-scope inventory suite passes 50
tests and the public-family binding suite passes 17. The refreshed full test
run passes 1,615 tests with 95 skipped; source matrix and phase-1 source
validators pass. This closes only those local dense GLM-DSA P02 tuples; the
broader family/backend map, P03–P14 obligations, runtime evidence, and P15
remain open.

The Inkling P02 follow-up binds 21 exact local source routes: publisher
wrapper and independent text identities across native, device, and
public-range execution. The wrapper binds F32/F16/BF16 to FP32 and F8_E4M3
through its scale-aware FP8 block conversion; text-only routes bind only
F32/F16/BF16. This follows the conditional loader's complete text/media
inventory and optional Inkling MTP planner, plus its native, device, and
Segment/Edge callers; source admission does not claim media parity or full MTP
serving closure. New preflight regressions cover all six identity/backend
pairs, and the source-scope suite passes 51 tests. Full unittest discovery
passes 1,622 tests with 95 skipped in 584.959 seconds. The 1,334-row matrix
validates with 126 source-admitted, 800 candidate-unverified, 322 rejected,
and 86 backend-restricted rows; `phase1_closed`, execution, parity, and platform
certification remain false. P03–P15 and actual P13 evidence remain open.

The public Segment/Edge service factory had an over-broad second admission
check: after reading the checkpoint's exact bounded header inventory and
admitting each dtype/format tuple, it also required every declared family-wide
public-range format row to be verified. This incorrectly blocked an exact
admitted checkpoint because unrelated family formats remained candidates. The
redundant family gate is removed at this caller; the exact inventory check
still rejects unsupported or unadmitted encodings before executor/runtime
construction. Focused regressions pass for both an unadmitted exact tuple and
the exact-inventory factory boundary. Full unittest discovery passes 1,622
tests with 95 skipped in 584.959 seconds. Structural validators pass for the
1,334-row matrix (126 source-admitted, 800 candidate-unverified, 322 rejected,
86 backend-restricted); Phase 1 closure, execution, parity, and platform flags
remain false. The checklist is locally ignored by Git, so the update is present
in the workspace but does not appear in normal Git status output.

The GLM5/Inkling public family adapter already required a decoder receipt
matching an immutable checkpoint snapshot, but the ordinary native text loader
never issued that receipt and the native-family CLI had no selector for this
ABI. The loader now admits an explicit `public-range` backend, checks the
checkpoint snapshot after loading, and seals GLM5/Inkling text decoders to that
exact inventory. A directory factory preflights and owns decoder cleanup. The
CLI exposes `--public-runtime-library` for those CPU text identities only and
pins it in the artifact set. Matrix rows name the concrete factory, adapter,
and C callback source; identity-specific restrictions keep the text ABI from
advertising media support. This adds three GLM5 public text tuples; Inkling's
three matching tuples were already among its 21 local routes. Focused
source-scope/native-checkpoint/family-binding suites pass 97 tests with eight
optional skips. Full discovery passes 1,622 tests with 95 skipped. Structural validation now
reports 126 source-admitted, 800 candidate-unverified, 322 rejected, and 86
backend-restricted rows. Native ABI build/runtime verification is unavailable
because this environment has no CMake, Ninja, or C compiler. Other P02 paths,
P03–P15, and actual P13/runtime evidence remain open.

### Final verification and bug-fix pass — 2026-10-18

This pass fixed concrete source defects found during the audit. The P02
checkpoint header gate now admits E8M0 only when it is the convention-matched
scale companion of an F8_E4M3 tensor; unrelated/orphan E8M0 remains rejected
for family-specific geometry/value admission. Speculative prefill now verifies
that a failed/cancelled target restore returned exposed physical history/logits
to the unpublished boundary and quarantines on mismatch. Qwen4/Qwen3.5
multimodal constructor failures now retire explicitly owned providers; Qwen3.5
initializes the inherited snapshot-cleanup inventory. Public Qwen4 checkpoint
path preflight accepts concrete pathlib subclasses and has a direct regression
test. Whole-model report validation now checks the nested observation receipt
schema/flags and recomputes both observation and execution binding digests.

A full-project audit also found intermittent Windows image false rejection:
path `ctime` can differ slightly from the still-open handle despite matching
bytes and identity. Chat image attachment loading now validates the stored
asset SHA-256 alongside stable file identity, size and mtime. Fifteen repeated
image-load regressions passed. The Qwen4 missing `Path` import was corrected;
the Mixtral device staging cleanup no longer deletes a binding that Ruff
cannot prove live.

Final Python discovery before the latest E8 correction passed 1,638 tests with 95 optional skips in 613.884
seconds. Focused chat/Qwen4 regressions pass (11); Segment ABI tests pass (22);
the P02 route/preflight, P04 media cleanup, P05/P06/P12 recovery/speculation,
and P13 whole-model evidence tests pass in their scoped runs. Frontend checks
pass: 142 tests, TypeScript check, and production build (large Monaco chunk
warning only). Python sdist and wheel build successfully. `ruff --select F821`
passes across engine and tests; scoped diff whitespace checks pass. Structural
source validation resolves nine families, 19 registry identities and 1,334
tuples: 126 source-admitted, 800 candidate-unverified, 322 rejected, 86
backend-restricted.

This host has no CMake, Ninja, or C compiler for the native CMake build. The
accelerator environment reports Torch CPU-only, CUDA unavailable, and zero CUDA
devices; Windows video-controller enumeration is denied. No model checkpoint
comparison, pinned P13 observations/cache/placement receipts, numerical parity,
or platform certification was produced. Therefore P02–P15 remain only
partially closed and P15 remains OPEN; the structural source audit continues to
report `phase1_closed=false`, `executed=false`, `numerical_parity_verified=false`,
and `platform_certified=false`. The checklist itself is locally ignored by Git,
but this update is saved in the workspace.

The follow-on P11 lifecycle pass found that the public Qwen4 hidden-export ABI
holds a reentrant Python lock while invoking its cancellation callback. A
cancellation callback could recursively invoke hidden export or close/unregister
the model during native execution. The owner now tracks active native operations
and rejects callback reentry, routing-scope entry, and close until the operation
returns. The regression exercises both forbidden callback actions and verifies
the model is still unregistered exactly once afterward. Focused lifecycle tests
(3) and Ruff/whitespace checks pass. The later combined full Python discovery
run after the Qwen4 lifecycle and K3 media unwind fixes passes 1,638 tests with
95 skips in 613.884 seconds.

The P04/P07 K3 follow-up also hardens timestamp normalization against huge
integers that otherwise raise `OverflowError`, avoids detaching video frames
until aggregate geometry and byte admission pass, closes temporary PIL RGB
conversions, and releases partial patch banks plus detached frame locals when
packing is cancelled or fails (including when a traceback is retained). A
weak-reference regression verifies a partially built frame bank is freed. The
P02/P03 route audit found no factory bypass: Qwen4 conditional routes repeat
exact inventory admission at the loader, while its unverified public-range
tuple is rejected before runtime construction. The final P05/P06/P12 caller
audit found no additional safe local defect; speculative streaming without an
atomic sink and process/transport-loss recovery remain restricted/unsupported.
Combined K3/media/publisher/Qwen4 lifecycle focused tests pass (14). The
complete Python discovery run on the combined changes, including subsequent
P11/P14 fixes, passes 1,638 tests with 95 skips in 613.884 seconds. Ruff F821
and scoped whitespace checks pass, and the Python sdist and wheel rebuild
successfully after the final source changes. The structural validator still
resolves 9 families, 19 identities, and 1,334
tuples; the matrix still contains 126 source-admitted, 800 candidate-unverified,
322 rejected, and 86 backend-restricted rows. P15 remains OPEN because no native
compiler/runtime/device or real-checkpoint evidence was available.

After that run, the P14 stream audit found that a provider iterator could remain
open when its first `next()` failed before SSE response ownership transferred.
The route now closes that iterator while retaining the original mapped error;
the focused inference-route suite passes 13 tests. The P11 callback audit found
that shared Segment/Edge, Qwen3.5/Qwen4, and publisher adapters did not
consistently reject cancellation values outside the C ABI's exact 0/1 contract.
Those ABI routes now validate exact integer results; the Python text-cancellation
wrapper normalizes its higher-level truthy predicate before invoking the ABI
validator. Regressions cover malformed ABI values and Python predicate
normalization. Combined focused ABI/serving checks pass 39 tests. The final
complete discovery run after these changes passes 1,638 tests with 95 skips in
613.884 seconds; the Python sdist and wheel were rebuilt successfully, Ruff F821
and scoped whitespace checks pass, and the structural validator remains at 9
families, 19 registry identities, and 1,334 tuples. The phase remains open for
the previously documented unverified routes and external execution evidence.

### P09 selected-device E8 basis correction — 2026-10-18

The selected-device Mixtral-derived E8 path was applying its orthogonal
transform twice: `decode_device_expert` reconstructed already-rotated weights,
while `TorchMixtralExperts` also rotated activations before matmul. The decoder
now retains codebook-basis weights so the consumer's single activation
rotation matches the native E8 equation. Device decoding also accepts finite
signed scale records consistently with the native E8 reader (the packer still
emits canonical nonnegative scales). A numerical regression compares the
rotated device-side product with the persisted-weight reference, including
signed scales. All six E8 rotation tests pass under the CPU Torch environment.
The complete Python suite after this correction passes 1,639 tests with 95
skips in 592.595 seconds; Ruff F821 and structural validation pass, and the
final Python sdist/wheel build succeeds. This fixes the consumer equation but
does not certify actual CUDA/MPS placement or cross-platform parity; P09/P10
and P15 remain open.

### P02/P03/P04/P05/P06 regression and full-suite audit — 2026-10-18

The native family CLI now rejects `--runtime-library` on flat GLM5/Inkling
routes that do not consume that library, with a regression at the source-scope
inventory boundary. Publisher contract validation now rejects unknown MTP or
next-token configuration claims for all applicable families, including
Inkling; recognized fields are parsed separately, so this prevents config-only
inventory claims from reaching admission without an owner.

Prepared K3 video pixels and geometry are snapshotted into immutable backing
storage; callers can no longer re-enable NumPy writes and mutate the validated
vision input. Temporary-image cleanup now attempts every owner even when an
exception cannot be formatted, and cleanup diagnostics preserve the primary
conversion error. Speculative rollback now verifies an exposed draft history
actually returned to its saved prefix after restore; a silent restore miss
quarantines the split owner. Distributed reset consensus now binds the
zero-context/empty-history boundary across ranks and rejects stale rank history
instead of accepting a constant reset summary.

The combined focused K3, distributed checkpoint, speculative transaction /
cancellation, and source inventory suites pass 93 tests. Follow-up P11/P13/P14
ABI geometry, evidence-owner, and recovery-stream cancellation checks pass in
their focused suites (67 in the combined source-scope/ABI/stream run, including
the standalone geometry regression). Ruff F821 and scoped
`git diff --check` pass on the changed Python source and regression files. One
full discovery run produced a single Windows loopback abort in
`test_all_mutation_verbs_are_limited_before_dispatch`; that test passed alone,
its complete 17-test module passed. An intermediate full run exposed three
expected-error ordering regressions after adding the P11 geometry guard; moving
that check after route, inventory and checkpoint provenance validation (but
still before ABI loading) restored the established errors. All three prior
assertions and the new geometry regression now pass, and the final full
discovery passes 1,655 tests with 95 skips in 505.603 seconds. The structural source validator
passes for nine families and 1,334 execution rows: 126 verified-admitted, 800
candidate-unverified, 322 rejected-unadmitted, and 86 restricted. It reports
`phase1_closed=false`, `executed=false`,
`numerical_parity_verified=false`, and `platform_certified=false`.

The P13 source tooling owner map now includes
`platform_certification/source_inventory.py`, which emits source-only receipts;
the owner-completeness regression pins that module. P14 recovered checkpoint
streaming now accepts and forwards the provider-level cancellation callback to
the worker, and the regression verifies cancellation unwinds and releases the
stream lock. P11 direct public publisher construction now checks the backing
service vocabulary/context against the checkpoint-bound ABI geometry before
loading the ABI library, after P02 route and checkpoint provenance checks so
those established errors remain authoritative. These close specific source
gaps only: P13 still lacks runtime artifact pins, cache/placement receipts and
comparison evidence; P11's complete family ABI/caller map and P14 recovery
callers remain open.

After the final source edits, Ruff F821 passes for the changed Python source and
tests, scoped `git diff --check` is clean, and `python -m build --no-isolation`
successfully builds both the sdist and wheel. The full test run and package
build validate source/package integrity on this host only; they do not provide
native compiler, accelerator placement, real-checkpoint, numerical-parity, or
runtime evidence.

These are bounded source fixes, not family-wide P02/P03 closure. Remaining
items include end-to-end family/backend/format and inventory-consumer traces;
cross-family media and caller lifecycle coverage; full speculative caller and
RNG reconciliation; distributed operation coverage and process/transport-loss
recovery; remaining format/device/ABI/recovery maps; actual P13 pinned runtime,
cache, placement and comparison evidence; and final P15 reconciliation.
Phase 1/P15 therefore remain **OPEN**.

### P02/P04/P13/P14 follow-up corrections — 2026-10-18

The P02 matrix now separately binds the native OLMoE edge's F32/F16/BF16
dense tensors to float32 execution, distinct from the leased layer executor's
INT8-row expert-bank route. This allows the edge constructor's bounded exact
inventory preflight without conflating dense tensor storage with expert-bank
encoding. The generic Segment/Edge factory now checks the canonical family and
checkpoint model identity before runtime construction while preserving the
established range-validation error ordering. The focused OLMoE/source-scope/
factory regression set passes 106 tests. The source matrix now contains 129
verified-admitted, 797 candidate-unverified, 322 rejected-unadmitted, and 86
restricted rows (1,334 total, nine families); the source validator still
reports Phase 1 open and no execution, numerical-parity, or platform
certification evidence.

Additional bounded fixes in this pass: the public K3 video input snapshots
frames into immutable backing storage; Anthropic auth failures use the
Anthropic error envelope; whole-model artifact rows reject unvalidated claim
fields; platform command adapters reject inline interpreter payloads that
cannot be pinned; K3 media evidence rejects inconsistent passed/error
reports; Qwen4 media admission pins its actual decoder source; OLMoE direct
native loading and Python reference CLI paths enforce their applicable route
restrictions; the device-text/Segment tuple is rejected before materialization;
OpenAI streaming closes upstream iterators on all exits; device runtime queues
are bounded; and serving streams close upstream before publishing terminal
success while preserving committed input-usage counts. Focused regressions
pass. Follow-up fixes also revalidate Qwen3.5/Qwen4 checkpoint snapshots at
public-handle registration, bind all whole-model native decoder identities to
the exact bounded inventory, verify speculative proposer history after
restore, clean uncommitted distributed generation directories after manifest
readback failure, and preserve primary chat/OLMoE generation errors when
upstream iterator cleanup fails. The combined latest focused suites pass 51
tests (plus the separate 18-test commercial-security module). The route
contract audit also found and fixed `/api/agent-runs` being declared but not
implemented in the extracted route module; its 3-test module passes. The latest
full discovery passes 1,682 tests with 95 skipped. The package sdist/wheel
rebuild succeeds on the settled tree, Ruff F821 passes, and scoped `git diff
--check` reports no whitespace errors (only existing CRLF normalization
warnings).

Hardware/toolchain probes on this host do not provide P10/P13 certification:
Windows video-controller inventory was access-denied, `nvidia-smi` and native
C/C++ compilers were not present on PATH, and the project virtual environment
has no Torch package. No accelerator placement, native ABI compilation, or
real-checkpoint parity receipt was produced.

These additions close specific source bugs only. The 797 candidate rows,
cross-family caller closure, distributed process/transport-loss handling,
complete ABI/recovery maps, real pinned P13 runtime/cache/placement/comparison
evidence, and final P15 reconciliation remain open. No build or test result on
this host substitutes for actual accelerator placement or external runtime
evidence.
