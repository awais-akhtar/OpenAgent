# Phase 1 — active completion queue

Updated 2026-10-19 (rechecked). This is the short list of **remaining work**, not a record of past edits. The detailed requirements and audit history remain in [PHASE_1_SOURCE_CHECKLIST.md](PHASE_1_SOURCE_CHECKLIST.md). Phase 1 and P15 are **OPEN**.

## 2026-10-19 source-only audit

The desktop repair is source-complete for its narrow scope: the Windows launcher prefers the checked-out runtime, uses writable data/runtime directories, stale directory-server ownership was removed, and the workbench keeps serving after a browser process is reused. The live endpoint was observed separately as `OpenAgent/0.1` HTML, but this is desktop smoke evidence only and does not close Phase 1.

The repository does **not** contain the claimed `unified_execution.py` implementation or commit `96b5007`. The universal layer currently present is `universal_engine.py` (`UniversalEngine`, `UniversalSession`, `ModelRoute`, and `ModelManifest`), wired through the adaptive-engine package/runtime. No audit claim should treat the absent unified-execution work as complete.

Recheck result: the closure decision is unchanged. The working tree has additional assistant/route/test edits, but no new canonical route matrix, runtime receipt, ABI execution receipt, placement observation, or P15 reconciliation was identified; those edits therefore do not change the Phase 1 status.

### 2026-10-19 code audit result

Source inspection confirms the existing admission guards, universal engine/session layer, queue/stream cleanup, speculative rollback boundaries, distributed poisoning/cleanup, format restrictions, and capability disclosures are present. It also confirms the remaining blockers: `unified_execution.py` is absent, route tuples are not all fully traced, family-wide media/speculative/distributed/recovery/ABI closure is incomplete, and P13 evidence fields remain structural (`not-captured`, `executed=false`, `phase1_closed=false`). The newly modified assistant/route files are outside the unresolved engine-route matrix and do not close any P02–P15 package. Status remains unchanged: Phase 1 **OPEN**, P15 **OPEN**.

Repeat audit (2026-10-19): `source_scope.py` still emits candidate-unverified rows and `phase1_closed=False`; `wholemodel_validation/runner.py` still rejects absent/candidate/restricted execution contracts; `unified_execution.py` remains absent while `universal_engine.py` remains present. No source change qualifies a package for closure, so no status transition was made.

Latest delta audit (2026-10-19): only assistant/route and support-test files are newly modified in the working tree; no adaptive-engine route, source-matrix, evidence, ABI, or recovery closure changed. The same gates remain visible in source (`candidate-unverified`, `not-captured`, `completion_claim=False`, and `phase1_closed=False`). Phase 1 and P15 remain **OPEN**.

Audit refresh (2026-10-19): current source still has no `unified_execution.py`; `universal_engine.py` is the available universal layer. The modified assistant files do not touch the Phase 1 route matrix or evidence gates. No package status changed.

Claim review (2026-10-19): the proposed summary is only partially accurate. P04/P05/P06 have substantial source implementations and safeguards, but the checklist still correctly classifies them as **IN PROGRESS** because cross-family/caller closure, atomic publication, full distributed-bank consensus, and transparent process-loss recovery are not complete. The source matrix still contains candidate-unverified rows; exact totals differ by matrix view (the active baseline records 1,334 expanded tuples / 1,133 canonical route items, not a verified-admitted conversion). P13 receipts remain uncaptured, P11 family ABI closure remains partial, P12 distributed physical-state/recovery remains partial, P14 lifecycle/parity remains partial, and P15 remains open. Evidence requirements are enforced by source gates; hardware absence is one blocker for A10 receipts, but not the only remaining closure gap.

Full audit refresh (2026-10-19): source gates and package states were rechecked across the route matrix, universal engine, validation runner, serving routes, distributed/recovery modules, ABI owners, and Colibri capability boundaries. Existing implementations and explicit restrictions are confirmed, but no package is promoted: candidate rows remain non-admitted, evidence remains `receipt_present=False`, and the catalog remains `phase1_closed=False`. P02, P04–P07, P09–P14 remain partial/in progress; P15 remains open.

Current closure status remains: P01 complete; bounded P03 inventory contracts complete; P02, P04–P07, P09–P14 in progress; P15 open. Structural source guards and explicit restrictions are implemented in several paths, but they do not substitute for family-wide route traces, caller lifecycle closure, runtime placement, numerical comparisons, native ABI execution, or pinned P13 receipts.

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
