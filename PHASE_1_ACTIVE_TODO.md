# Phase 1 — active completion queue

Updated 2026-10-19 (full static audit revision 3). This is the short list of **remaining work**, not a record of past edits. The detailed requirements and audit history remain in [the Phase 1 source checklist](docs/audits/phase1/source-checklist.md). Phase 1 and P15 are **OPEN**.

## Claimed-fix verification — 2026-10-19

The seven reported fixes are **not present in this checkout**. Git reports no
staged changes, no untracked files, and no modifications to
`universal_engine.py` or `runtime.py`. The only current product changes are the
assistant CRUD/UI files already audited. The sole stash contains only two
frontend integration-support edits and none of the claimed engine fixes.

| Claimed fix | Verification result | Current source evidence |
| --- | --- | --- |
| Production universal caller/adapters | NOT FIXED | `register_universal_adapter()` and `open_universal_model()` occur only as method definitions; there are no production registrations, callers, or universal-engine tests. Checking whether an adapter is registered inside the open helper would not create the missing registrations or serving caller. |
| Exact source-route admission | NOT FIXED | `UniversalEngine.inspect()` still checks only `route.model_type == config.model_type`; `require_source_route_admission` does not exist in the working tree. |
| Warm-session option identity | NOT FIXED | The session key remains model directory plus six route fields; `**options` are still omitted and silently ignored when a warm session exists. |
| Cancellation rollback | NOT FIXED | `generate()` still raises after post-dispatch cancellation without rollback; `stream()` still has no pre-dispatch cancellation check or transactional restore. No new `rollback()` integration exists. |
| Foreign/stale restore rejection | NOT FIXED | `snapshot()` still returns the raw adapter receipt and `restore()` still forwards it without route/session/epoch/freshness validation. Checking only a route would also be insufficient to establish receipt ownership. |
| Retryable failed cleanup | NOT FIXED | Session pools are still cleared/popped before close; `_closed` is still assigned only after owner close returns. Merely setting `_closed=True` before re-raising would prevent retry rather than preserve cleanup ownership. |
| Task goal validation | NOT FIXED | `create_task()` still inserts the supplied `goal_id` without querying `assistant_goals`; the database schema still has no foreign key. Only `update_task()` performs a separate pre-update query. |

**Verified total: 0 of 7.** These claims must not be marked complete or staged.
If the fixes exist in another checkout or patch, they must be brought into
`D:\openagent` and then audited again. The existing BUG-01 through BUG-10
register remains active.

## Project structure audit — 2026-10-19

The source layout and local workspace were carrying avoidable duplication.
Tracked organizational files were moved without changing their contents, and
duplicated QA package copies were removed from the Git index while preserved
locally under ignored paths.

### Canonical source boundaries

- Python product source: `openagent/` (664 tracked files).
- Current React application: `frontend-second/` (158 tracked files); root npm
  `dev:web`, `build:web`, `check:web`, `test:web`, and `test:e2e` target it.
- Legacy React application: `frontend/` (87 tracked files), retained only by
  explicit `*:legacy` scripts. It should be formally deprecated or removed in
  a separate compatibility change, not silently mixed with the current UI.
- Tests: `tests/` (280 tracked files).
- Product documentation and operations: `docs/`, `scripts/`, `installers/`,
  and `deploy/`.
- `web/` mixes tracked source/static files with ignored generated bundles.
  Old generated assets already tracked remain tracked despite `.gitignore`, so
  the source-versus-build boundary is ambiguous.

### Repository hygiene defects

1. `.qa/` previously contained 386 tracked files, including two separately
   tracked 174-file package snapshots under `batch1-package-staging` and
   `batch1-package-verify`. Those duplicate snapshots are now staged for index
   removal and ignored while their local files remain available. Future QA
   evidence must use manifests rather than parallel source trees.
2. The older `PHASE1_CHECKLIST.md` remains ignored as obsolete local material.
   The detailed current checklist has now moved to
   `docs/audits/phase1/source-checklist.md`, where it can be versioned beside
   its audit index. `PHASE_1_ACTIVE_TODO.md` remains the stable root entrypoint.
3. The root `modelsdebug.png` asset has moved to
   `docs/assets/models-debug.png`; future audit images must live under a
   documented evidence directory or outside source control.
4. Root npm dependencies duplicate part of `frontend-second`'s dependency
   ownership. The workspace has no declared npm-workspaces structure, so two
   lock/dependency contexts can drift.
5. Several very large modules (`frontend-second/src/screens/AdvancedScreen.tsx`,
   `web/src/app.ts`, `openagent/routes/runtime.py`, and Phase 1 catalog/factory
   modules) concentrate unrelated responsibilities and make ownership audits
   difficult. Split them by route/domain with explicit public boundaries.

### Local disk clutter (ignored, not Git source)

- Local model directory: approximately 41.4 GB.
- Root `.tmp*` directories: 14,094 files and approximately 3.35 GB.
- `.accelerator-venv`, `.venv`, and `.python312`: approximately 2.63 GB total.
- Root ignored archives, Codex transcripts, and temporary captures: 38 files,
  1,384,352,458 bytes, including multiple 90–392 MB frontend/evidence archives.
- Duplicate root and frontend `node_modules`: approximately 533 MB.

These ignored objects do not affect commits, but they make the checkout hard
to navigate and consume substantial disk space. Cleanup must be a separate,
explicitly approved destructive operation after preserving any required local
models/evidence.

### Applied target structure

1. Keep only canonical editable trees at root: `openagent/`,
   `frontend-second/`, `tests/`, `docs/`, `scripts/`, `installers/`, and
   `deploy/`.
2. Durable Phase 1 audit documents now live under `docs/audits/phase1/`;
   release audits live under `docs/audits/releases/`. Bulky binary evidence
   remains a release-artifact concern with tracked hashes/manifests.
3. The two tracked package-copy trees under `.qa/` have been removed from the
   Git index without deleting their local files. `.gitignore` prevents them
   from being recommitted.
4. Decide whether `frontend/` is supported; if not, remove its legacy scripts
   and source together in one reviewed change.
5. Make `web/` either a source directory or a generated distribution target,
   not both. Generated hashed assets should not be tracked.
6. Keep model weights, virtual environments, runtime data, build outputs,
   archives, and temporary evidence outside the repository root where
   practical.

## Deep re-audit revision 2 — newest working tree

The entire current working-tree delta was re-read. It contains 174 lines of
new product implementation across assistant goal/task CRUD, scoped assistant
HTTP routes, frontend edit/delete controls, integration-server authentication,
browser smoke coverage, and focused route/UI checks. Together with checklist
edits, the working-tree diff is 243 insertions and 25 deletions across eight
files. There are no untracked files.

None of the seven product-code/test files is under `openagent/adaptive_engine`,
and `openagent/routes/adaptive_engine_routes.py` is unchanged. There is no new
family route, loader, logits consumer, tensor inventory owner, cache/TP owner,
media path, speculative state transition, distributed bank, persisted-format
consumer, device placement owner, ABI symbol, recovery path, evidence adapter,
or inference-serving caller in this delta. It therefore changes **zero**
P02–P15 closure decisions.

The source-of-truth gates were inspected again and remain explicit:

- `source_scope_matrix()` returns `phase1_closed=False`, `executed=False`,
  `numerical_parity_verified=False`, and `platform_certified=False`.
- Row evidence defaults to `source-obligation-unverified`,
  `evidence_state='not-captured'`, `completion_claim=False`, and
  `receipt_present=False`.
- The whole-model runner rejects absent, candidate-unverified, and restricted
  execution contracts before treating them as evidence.
- `phase1_closure_report.md` and
  `phase1_complete_route_summary.json` remain absent.
- `unified_execution.py` remains absent; the implemented shared session layer
  remains `universal_engine.py`.

**Revision-2 conclusion:** the newly changed application functionality is
implemented outside the Phase 1 engine boundary. The completion table and
A01–A11 queue below remain authoritative and unchanged.

## Confirmed static bug register — 2026-10-19

These are concrete source defects found during the full audit. They are not
runtime-certified because this pass deliberately did not execute code.

| ID | Severity | Affected requirements | Confirmed defect | Required correction |
| --- | --- | --- | --- | --- |
| BUG-01 | CRITICAL | P02, P11, P14, P15 | `AdaptiveExecutionEngine` constructs a `UniversalEngine` and exposes registration/open helpers, but no production code registers a family adapter and no route or service caller invokes `open_universal_model()`. Repository-wide references are limited to the method definitions. The advertised universal path is therefore not a reachable serving path. | Register every admitted family through exact resolvers during runtime construction, expose a bounded service/gateway caller, and fail startup/admission when a required adapter is absent. |
| BUG-02 | CRITICAL | P02, P09, P10, P15 | `UniversalEngine.inspect()` validates only that `route.model_type` matches `config.json`. It does not validate route family, backend, arithmetic, source format, or persisted format against the Phase 1 source catalog. A caller-supplied `ModelRoute` can therefore describe an unverified tuple and reach an adapter opener. | Bind universal admission to the exact source-scope route decision and reject candidate/restricted/unadmitted tuples before invoking the opener. |
| BUG-03 | HIGH | P10, P12, P14 | Warm-session identity uses only model directory plus six route fields. `open()` options are passed to the first opener but omitted from the session key, so later calls with different device, placement, cache, isolation, or adapter options silently receive the existing session. | Define and validate a canonical immutable option/isolation identity and include it in the key, or reject option changes while a session is warm. |
| BUG-04 | HIGH | P05, P12, P14 | `UniversalSession.generate()` checks cancellation after adapter execution but does not snapshot/restore state before raising. `stream()` has no cancellation check before adapter dispatch and likewise has no rollback. A cancelled request can mutate warm KV/RNG/continuation state without publishing a valid result. | Add transactional snapshot/restore or require an adapter transaction contract; check cancellation before dispatch, during iteration, and before commit/publication. |
| BUG-05 | HIGH | P12, P15 | Universal snapshots are returned as raw adapter receipts and `restore()` forwards any receipt without model/session/route identity, epoch, freshness, or replay validation. The shared layer cannot reject stale or foreign restores. | Wrap receipts in an immutable identity/epoch envelope and verify exact session, route, checkpoint identity, and freshness before adapter restore. |
| BUG-06 | HIGH | P12, P14 | `close_model()` pops sessions and `close_all()` clears the pool before closing them. `UniversalSession.close()` sets `_closed` only after owner close succeeds. When owner close raises, the still-open owner is no longer reachable through the engine for retry. | Retain failed-close sessions in an explicit cleanup registry (or reinsert them), aggregate errors safely, and expose deterministic cleanup retry/status. |
| BUG-07 | MEDIUM | P14 | `stream()` holds the session lock for the complete generator lifetime. If a consumer abandons an iterator without closing it, later generate/snapshot/restore/close operations can remain blocked indefinitely; there is no lease timeout or engine-owned iterator registry. | Introduce an owned stream lease with explicit cancellation/close tracking and bounded teardown on request disconnect and engine shutdown. |
| BUG-08 | MEDIUM | Product data integrity (outside P02–P15) | `AssistantManager.create_task()` stores any supplied `goal_id` without verifying that the goal exists in the same project, and the schema has no foreign key. `update_task()` performs that validation in a separate transaction. Dangling/cross-project goal links and validation/update races are possible. | Validate and mutate in one transaction and enforce a project-aware relational invariant or trigger. |
| BUG-09 | MEDIUM | Product mutation semantics (outside P02–P15) | New update/delete operations commit their database mutation before writing the audit event. If audit persistence fails, the API can report failure after the mutation has committed, making retries ambiguous. | Commit the domain mutation and audit record atomically in one transaction, or return an explicit committed-with-audit-failure result with idempotency handling. |
| BUG-10 | MEDIUM | Product state integrity (outside P02–P15) | Goal horizon/status and task priority/status are accepted as arbitrary strings by manager and HTTP mutation paths; the schema has no checks. Invalid values can disappear from `active`/`open` queries and break priority/coaching semantics without an error. | Define canonical enums, validate them in one shared manager boundary, and add database checks or a compatible migration guard. |

BUG-01 and BUG-02 invalidate any claim that the universal layer currently
provides end-to-end family admission. BUG-03 through BUG-07 keep warm-session,
cancellation, recovery, and serving lifecycle closure open even after adapters
are wired. BUG-08 through BUG-10 concern the newest assistant changes and do not
change Phase 1 status, but they must be fixed for reliable product behavior.

## Latest changed-tree delta audit — 2026-10-19

The latest uncommitted product changes add assistant goal/task update and
delete operations, their HTTP routes, frontend controls, smoke coverage, and
route tests. The changed files are `openagent/assistant.py`,
`openagent/routes/assistant_routes.py`, the corresponding frontend/test
files, and this queue. No adaptive-engine implementation, loader, tensor
consumer, cache/TP owner, family ABI, format/device path, distributed owner,
recovery owner, whole-model evidence adapter, or source closure gate changed.

Consequently this delta closes **none** of P02–P15. It is valid product work,
but it is outside the Phase 1 inference-engine closure boundary. The source
still explicitly reports `phase1_closed=False`; candidate-unverified routes
remain representable; evidence remains `not-captured` with
`receipt_present=False`; and the whole-model runner rejects candidate,
restricted, or absent execution contracts. No generated closure report or
route-summary receipt is present in the checkout.

This delta audit was source-only. No tests, imports, builds, benchmarks,
device probes, compiler invocation, model loading, or runtime evidence capture
was performed.

## 2026-10-19 deep source audit

This audit inspected the current route catalog and admission gates, loader and
execution owners, media/speculative/distributed/recovery modules, public ABI
owners, serving routes, whole-model tooling, Colibri capability restrictions,
and the universal session layer. It was source-only: no tests, imports, builds,
benchmarks, device probes, or model execution were performed.

The two referenced generated reports, `phase1_closure_report.md` and
`phase1_complete_route_summary.json`, are not present in this checkout and
therefore cannot provide closure evidence. The claimed `unified_execution.py`
is also absent. The implemented shared layer is `universal_engine.py`. It is
instantiated by `AdaptiveExecutionEngine` and exposes helper methods for
registration/open, warm sessions, serialized generation/streaming,
cancellation, snapshot/restore, and teardown while leaving model arithmetic to
family adapters. The deep bug audit found no production adapter registrations
and no serving caller of the open helper, so this is currently a library
boundary rather than an end-to-end engine path. Its route validation and
lifecycle defects are tracked above.

| Requirement | Confirmed implemented source | Remaining before closure | Status |
| --- | --- | --- | --- |
| P01 | Separate admission/recovery/publication budgets and bounded execution ownership. | No open P01 source item identified. | DONE |
| P02 | Nine-family catalog, exact route records, source-reviewed bindings for selected tuples, fail-closed admission, backend/format/media restrictions, factory/CLI guards. | Reconcile every canonical tuple through loader, logits consumer, cache/TP owner, ABI, factory, CLI, gateway, recovery and tooling. Required candidate rows must become verified-admitted or concretely rejected/restricted. | IN PROGRESS |
| P03 | Bounded inventory planners, dtype/shape/name checks, conditional-family preflight, explicit rejection of unowned MTP/media banks. | Finish family-wide optional/shared/MTP/expert/media tensor-to-consumer reconciliation and prove no alternate factory/CLI bypasses the planner. | BOUNDED DONE; FAMILY CLOSURE OPEN |
| P04 | Selected Qwen3.5/Qwen4/GLM5/Inkling/V4.1/K3 media ordering, placeholder, geometry, ownership and cleanup checks; mixed unsupported media fails closed. | Complete every family/caller media-to-logits chain, temporal/video continuation, borrowed-owner lifetime, cancellation and recovery. | IN PROGRESS |
| P05 | Transactional target/draft/cache/RNG snapshots, rejected-suffix RNG rewind, reset/quarantine, cancellation checkpoints and selected publication restrictions. | Reconcile accept/reject/bonus/EOS/context-limit/cancel/restore for every serving caller and provide atomic publication guarantees or enforce exclusions. | IN PROGRESS |
| P06 | Distributed preflight, agreement descriptors, rank failure synchronization, poisoning, cleanup attempts and selected bank validation. | Map every bank and shard layout to collective consumers; close operation failures, cancellation, process/transport loss, cache ownership and recovery consensus. | IN PROGRESS |
| P07 | K3 input/media position checks, rollback, checkpoint/recovery owners and explicit Metal/FLA/nonuniform-video restrictions. | Finish temporal/video semantics, native/continuation ownership and all serving consumers. | IN PROGRESS |
| P08 | Exact evidence schema selection/persistence contracts. | Execution observations remain obligations of P07/P13. | DONE (SOURCE SCHEMA) |
| P09 | Format catalog, bounded decoders, scale/geometry validation, conversion cleanup, E8/IQ3 transforms and explicit unsupported encodings. | Bind every admitted persisted format to the real family/backend consumer and prove packing/scale/rotation behavior per tuple. | IN PROGRESS |
| P10 | Backend discovery/admission, selected device owners, no-fallback restrictions and cancellable staging for selected paths. | Prove actual placement and operation-level no-host-fallback for every admitted backend; registry capability is not placement evidence. | IN PROGRESS |
| P11 | Segment/Edge ABI owners, selected family ranges/schemas, callback guards, immutable identities, snapshot ownership and rollback/quarantine. | Complete family-wide symbol/range/head/cache schema, callback reentry/ownership, close order and factory/CLI/service wiring; capture native compile/execute receipts. | IN PROGRESS |
| P12 | Transactional restore, stale/foreign checks in selected paths, quarantine, rank cleanup and durable/session recovery owners. | Close all-family physical-state ownership, serving publication order, cancellation/reentry, distributed rollback and process-loss consensus. | IN PROGRESS |
| P13 | Exact evidence obligations, artifact-pin/placement/comparison owners, structural validators, media adapter restrictions and false-certification gates. | Produce actual pinned observations, cache leaves, placement receipts, native ABI receipts and numerical comparison reports for every applicable admitted row. | IN PROGRESS / A10 BLOCKED-ENV |
| P14 | Local OpenAI/Anthropic text subsets, SSE terminal/error handling, bounded queue/429/wait headers/metrics, cancellation and iterator cleanup, explicit unsupported-feature admission. | Close caller-by-caller stream/backpressure/stop/media/recovery/warm-session/error behavior and either implement or explicitly scope remaining compatibility/Colibri features. | IN PROGRESS |
| P15 | Source validators prevent unsupported completion claims. | Reconcile every P02–P14 row as implemented, restricted/rejected, or missing; attach all mandatory evidence before setting the closure gate. | OPEN |

Authoritative closure gates remain unchanged in source:

- `source_scope.py` still emits candidate-unverified rows.
- Evidence obligations still declare `evidence_state='not-captured'`,
  `completion_claim=False`, and `receipt_present=False`.
- The generated catalog still declares `phase1_closed=False`, `executed=false`,
  `numerical_parity_verified=false`, and `platform_certified=false`.
- `wholemodel_validation/runner.py` rejects absent, candidate-unverified, or
  restricted execution contracts instead of treating them as evidence.

Phase 1 is therefore **not complete**. P01 and the P08 source schema are done;
P03 is done only for its bounded inventory contract; P02, P04–P07 and P09–P14
remain in progress; P15 remains open.

Remaining work must be tracked as concrete rows, not percentages:

- P02/P03: review every canonical family/backend/arithmetic/source/persisted-format tuple through loader, logits consumer, cache/TP owner, ABI/factory/CLI/gateway, recovery, and tooling; reconcile every optional/shared/MTP/expert/media tensor or enforce a specific exclusion.
- P04/P07/P14: close cross-family media order/geometry/ownership/temporal/continuation and caller-specific stream, backpressure, stop, cancellation, warm-session, recovery, and cleanup behavior.
- P05: cover speculative target/draft state, both RNGs, accept/reject/bonus/EOS/context-limit/cancel/restore, and publication guarantees for every serving caller.
- P06/P12: close every distributed bank/shard/cache owner, operation failure, cancellation, rollback, stale/foreign restore, consensus, and process/transport-loss path; current poisoning and best-effort cleanup are not transparent recovery.
- P09/P10: bind each admitted persisted format to its real loader and execution consumer, including scales/geometry, E8/IQ3 transforms, conversion cleanup, device admission, actual placement, and no-host-fallback evidence.
- P11: finish family-by-family ABI symbols/ranges, tensor/head schemas, continuation, callbacks, ownership/reentry, close order, and factory/CLI/service wiring; native compile/execute receipts remain absent.
- P13: attach exact artifact pins, cache leaves, placement observations, comparison adapters, and captured receipts for every applicable row. Structural catalogs remain `not-captured` and `completion_claim=false`.
- P15: reconcile every applicable requirement as implemented, concretely restricted, or missing; keep Phase 1 open until no required candidate or evidence gap remains.

## Baseline and counting rule

- Latest structural matrix: 1,334 expanded tuples across nine families. The canonical `source_route_work_items()` view collapses publisher variants to 1,133 exact six-field route items. The pre-DeepSeek-format-closure audit reported 129 source-admitted, 797 candidate-unverified, 322 rejected-unadmitted, and 86 restricted rows; the generator now moves impossible V4.1 format pairings into explicit restrictions. A candidate is an unverified combination, **not** automatically a separate code defect. Do not report a completion percentage from these counts.
- Latest full Python discovery: 1,682 passed, 95 skipped. Package build and Ruff F821 passed. These results do not prove real-checkpoint execution, numerical parity, native ABI compatibility, or accelerator placement.
- Current validator flags: `phase1_closed=false`, `executed=false`, `numerical_parity_verified=false`, `platform_certified=false`.
- Priority models for route closure: Qwen3.5-MoE, Qwen4-Exp, GLM5-Next, GLM-MoE-DSA, and DeepSeek V4.1. Priority does not remove the other admitted families from Phase 1.

## Route decision and completion rules

For **each applicable** family/backend/arithmetic/source/format tuple, record one decision in the source matrix: `admitted`, `restricted`, or `rejected`. An admitted route needs an inspected loader → tensor consumer/logits → cache and tensor-parallel owner → public ABI/factory/CLI/gateway → recovery → tooling chain. A restriction/rejection needs an enforced admission boundary, a specific reason, and a regression check. Generated candidates cannot be promoted merely because a registry lists them. Do not count unsupported tuples as implementation work unless the scope explicitly requires admission.

A work package is `DONE` only when its applicable rows have code or enforced exclusions, named source and regression checks, and links to required evidence. `BLOCKED-ENV` is allowed for a specific runtime receipt only; it does not close the source work. Keep P15 open while any required row is a candidate or any mandatory evidence is missing.

## Work packages

| ID | Remaining deliverable | Completion check | Status |
| --- | --- | --- | --- |
| A01 · P02 | Triage the 797 candidates into required routes versus unsupported combinations; retain the 129 admitted routes and existing explicit exclusions only where their real chains hold. Use `source_route_work_items()` so publisher variants are reviewed once per exact six-field route. | Every canonical work item has a reviewed decision, reason, and owner/source references; no required tuple remains candidate-unverified; source validator and route regressions agree. | IN PROGRESS |
| A02 · P02/P03 | Trace the five priority models through config, original/converted tensor inventories, optional/shared/MTP/expert/media banks, loader, execution consumer, and route entrypoints. | One route/consumer record per applicable model tuple; every admitted tensor has a consumer or enforced exclusion; factories and CLIs use the same preflight. | OPEN |
| A03 · P02/P03 | Repeat route/consumer closure for the other admitted families, including K3, Mixtral, OLMoE, and Inkling. | Same A02 checks for all remaining applicable rows; no alternate factory, public ABI, or CLI bypasses preflight. | OPEN |
| A04 · P04/P07/P14 | Close media and serving lifecycle by caller: order, positions/geometry, temporal/video continuation, borrowed-input lifetime, streaming/backpressure, stop, cancellation, warm reuse, recovery, and cleanup. | For each admitted family × caller × media mode, a source trace and focused lifecycle check cover success, interruption, and error; unsupported modes fail before state mutation. | IN PROGRESS |
| A05 · P05 | Reconcile target/draft cache and both RNGs for speculative accept, reject, bonus, EOS, context limit, cancel, and restore across actual serving callers. | Caller-level state invariants and regressions cover each transition; unsupported proposer/sink combinations fail closed, with publication guarantees stated precisely. | IN PROGRESS |
| A06 · P06/P12 | Map distributed banks to collective consumers, shard layouts, cache ownership, operation failures, cancellation, rollback, and recovery consensus. | Every admitted distributed operation has an ownership/failure trace and fault-path check; process/transport-loss behavior now poisons the coordinator and attempts rank-local service cleanup before restart, with cleanup retained on failure. Remaining shard/consensus evidence is still open. | IN PROGRESS |
| A07 · P09/P10 | Bind admitted persisted formats to loader, scale/geometry transform, execution consumer, conversion cleanup, device admission, and no-host-fallback behavior. | Per-tuple format/device trace and regression; E8/IQ3 rotation is correct where admitted and otherwise rejected; actual placement receipts are tracked under A10. | IN PROGRESS |
| A08 · P11/P12 | Complete family ABI and physical-state review: symbols/ranges, tensor/head schemas, continuation, callbacks, ownership/reentry, stale/foreign restore, and caller wiring. | Family-by-family ABI/restore map with source checks; native ABI compile/execute receipts tracked under A10. | OPEN |
| A09 · P13 source tooling | Bind every required row to artifact pins, cache schema, placement constraints, comparison adapter, and expected observations. | Tooling rejects missing/ambiguous pins and produces a complete executable evidence manifest, not just a structural catalog. The conformance report exposes deduplicated route work items, and the runner now rejects candidate-unverified routes before adapter loading; runtime evidence is still missing. | IN PROGRESS |
| A10 · P10/P11/P13 runtime evidence | Obtain real-checkpoint observations, cache/placement receipts, native ABI compilation and execution, numerical comparisons, and platform-specific receipts for required admitted rows. | Pinned artifacts and commands, recorded outputs, comparison thresholds/results, actual device placement, and reproducible receipts are attached to each applicable row. Source tests/builds alone do not close this. | BLOCKED-ENV: required compiler/device/checkpoints and receipts not present on this host |
| A11 · P15 | Reconcile every P02–P14 requirement and matrix row against implementation, enforced restriction/rejection, or a named missing item. | No required candidate or unowned gap remains; all mandatory evidence is linked; validator flags reflect observed results; only then mark P15/Phase 1 closed. | OPEN |

## Colibri parity scope decision

The historical checklist also names persistent `cache_slot`, complete OpenAI/Anthropic semantics, GBNF and family tool protocols, READY/multiplexed engine protocol, route telemetry/cache-aware routing, and multi-disk mirror planning. These are **not silently closed or silently dropped**. For each feature, record whether it is required by an admitted Phase 1 route. If required, add its exact caller/path to A01/A04/A09 and implement it; otherwise enforce and document the unsupported boundary. Until that decision and boundary exist, the feature remains an open scope item, not a completed exclusion.

### Universal-engine design decision (2026-10-19)

The local Colibri reference does **not** use one universal model kernel. It uses
one universal launcher/gateway, planner, storage hierarchy, expert cache,
route-trace format, and serving contract, while selecting a family-specific
engine from `config.json`; the reference describes these as sibling engines
sharing common headers. OpenAgent should adopt the same boundary:
`UniversalModelSession` owns request normalization, admission, placement,
streaming, cancellation, recovery, telemetry, and cache lifecycle, while a
family adapter owns tensor geometry, attention/MLP equations, quantized format,
media projection, and ABI. Unknown or incomplete adapters must fail closed at
admission rather than being routed to a nearest-family kernel.

Reference basis: `reference/colibri/README.md` documents shared storage,
expert cache, routing, and gateway layers, then states that `coli` reads a
model `config.json` and selects the matching engine binary. It documents one C
source file per model family rather than a single universal arithmetic engine.

Source implementation started: `adaptive_engine.universal_engine` now provides
an exact model-type registry, config/route admission, and one serial session
owner for generation, streaming, cancellation, snapshot/restore, and cleanup.
It intentionally delegates tensor math and ABI to the registered family adapter
and rejects unknown model types before adapter state allocation.

The universal `open()` path now accepts either an explicitly supplied exact
six-field route or a registered per-family route resolver driven by
`config.json`; it never guesses a neighbouring family or silently falls back to
another backend.

`AdaptiveExecutionEngine` now owns one `UniversalEngine` instance and exposes
`register_universal_adapter()` / `open_universal_model()`, placing the shared
boundary inside the existing runtime rather than creating a parallel service.
Universal streaming now holds the session lease until iterator teardown,
prevents concurrent adapter use, closes the upstream iterator deterministically,
and exposes route/status introspection for gateway health and telemetry.
Route construction now rejects blank fields, and adapter registration validates
the resolver before mutating the registry, so universal admission cannot leave
partially registered state.

The universal engine now keeps a warm-session pool keyed by canonical model path
and exact route, reuses loaded sessions across requests, supports per-model or
global eviction, and closes all warm adapters during `AdaptiveExecutionEngine`
shutdown while preserving primary and secondary cleanup failures.

## Next concrete pass

Start with A01/A02 on the five priority models: extract each exact candidate tuple, follow its real loader and consumer to the public caller, and write the admitted/restricted/rejected decision and missing edge beside it. Then implement the missing edge or admission guard before advancing to the next tuple. Update this short file by changing package status and linking the route/evidence record; keep detailed session notes in the historical checklist.

Source update 2026-10-18: DeepSeek V4.1 now rejects every generated source/persisted-format pairing except the three publisher-bank routes `BF16/bf16`, `FP8/fp8`, and `FP4/fp4`. The restriction is attached to persisted-format, format-route, loading, and execution cells so impossible Cartesian-product candidates cannot reach a loader or consumer. Backend process-loss restrictions remain preserved on distributed rows.

The V4.1 inventory preflight also recognizes explicit `FP8` and `FP4` source-bank dtypes only for the exact V4.1 model identity; the generic inventory admission path remains closed to those publisher-bank encodings for other families.

The standard `F8_E4M3` SafeTensors spelling is normalized to the V4.1 `FP8` publisher-bank identity before exact six-field route admission.

Colibri capability declarations now have a single `require_colibri_capability()` admission helper: restricted and not-admitted protocols fail before provider, queue, decoder, or persistent-state allocation.

OpenAI/Anthropic compatibility validation now calls that same boundary for tool protocols and constrained-generation requests, preventing route-specific validation from drifting away from the declared Colibri restrictions.

The Brio `cache_slot` rejection also uses the centralized capability boundary, so syntactically valid slot values cannot enter a provider without persistent isolated KV ownership.

The direct adaptive-engine generation route now uses the same constrained-generation restriction, closing the alternate caller that previously had its own duplicate rejection path.

Serving lifecycle work is now tracked as `A04 IN PROGRESS`; the stream cleanup path preserves the primary failure while retaining secondary owner-cleanup failures for retry instead of silently discarding them.

The V4.1 recovery verifier now compares nested target/proposal RNG receipts with
the same array-safe recursive state comparator used by speculative snapshots.
Plain dictionary inequality could raise an ambiguous NumPy-array error instead of
rejecting a mismatched restore boundary, leaving P05 recovery verification
incomplete for DSpark sessions.

The standalone V4.1 checkpoint publisher now uses that comparator as well when
validating a live DSpark owner against the serialized envelope; both resume and
publication paths therefore reject mismatched target or proposal RNG state
without relying on ambiguous ndarray equality.

V4.1 resumed generations now derive `finish_reason` from the checkpoint's
admitted stop-token set (rather than only the model's configured EOS set),
matching fresh generation and preserving caller-visible stop semantics across
recovery.

Filesystem re-audit on 2026-10-19 initially found only `models/olmoe-1b-7b-instruct`
and `models/openvino/qwen2.5-1.5b-instruct-int4-ov`. The official
`Qwen/Qwen3.5-0.8B` repository has now been downloaded to
`models/qwen3.5-0.8b`, including its 1,746,942,600-byte safetensors weight and
configuration/tokenizer files. This small checkpoint is useful for source
loader/plumbing evidence, but it does not close target-scale Qwen3.5 parity or
the remaining DeepSeek/GLM/Qwen4 evidence rows; A10/P13 remain open.
