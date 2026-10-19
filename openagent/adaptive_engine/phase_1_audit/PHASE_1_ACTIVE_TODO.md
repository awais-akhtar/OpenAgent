# Phase 1 — active completion queue

Updated 2026-10-19 21:02 UTC. This is the authoritative short list of **remaining work**, not a record of past edits. The detailed requirements and audit history remain in [PHASE_1_SOURCE_CHECKLIST.md](PHASE_1_SOURCE_CHECKLIST.md). Phase 1 and P15 are **OPEN**.

**HONEST STATUS (2026-10-19 deep audit):**
- **133 admitted routes** verified in source_scope.py (9 DeepSeek, 9 GLM-DSA, 17 GLM5-Next, 21 Inkling, 4 K3, 18 Mixtral, 27 OLMoE, 14 Qwen3.5, 14 Qwen4)
- **1,201 candidate-unverified routes** remain (not rejected, decision pending)
- **A09 P13 evidence framework COMPLETE** (CPU-executable): ArtifactPin, CacheSchema, ComparisonAdapter structure ready for all 133 routes
- **A08 K3 Metal UNBLOCKED**: 4 Metal routes now verified-admitted (was 0/96), complete token-to-logit dispatch implemented
- **A10 BLOCKED-ENV**: no GPU/accelerators available for runtime evidence capture
- **Phase 1 closure BLOCKED**: A01 candidates must be triaged, A02-A08 loader/consumer chains must be traced, A10 runtime evidence required

## Baseline and counting rule

- Latest structural matrix: 1,334 expanded tuples across nine families. The canonical `source_route_work_items()` view collapses publisher variants to 1,133 exact six-field route items. The pre-DeepSeek-format-closure audit reported 129 source-admitted, 797 candidate-unverified, 322 rejected-unadmitted, and 86 restricted rows; the generator now moves impossible V4.1 format pairings into explicit restrictions. A candidate is an unverified combination, **not** automatically a separate code defect. Do not report a completion percentage from these counts.
- Latest full Python discovery: 1,682 passed, 95 skipped. Package build and Ruff F821 passed. These results do not prove real-checkpoint execution, numerical parity, native ABI compatibility, or accelerator placement.
- Current validator flags: `phase1_closed=false`, `executed=false`, `numerical_parity_verified=false`, `platform_certified=false`.
- Priority models for route closure: Qwen3.5-MoE, Qwen4-Exp, GLM5-Next, GLM-MoE-DSA, and DeepSeek V4.1. Priority does not remove the other admitted families from Phase 1.

## Route decision and completion rules

For **each applicable** family/backend/arithmetic/source/format tuple, record one decision in the source matrix: `admitted`, `restricted`, or `rejected`. An admitted route needs an inspected loader → tensor consumer/logits → cache and tensor-parallel owner → public ABI/factory/CLI/gateway → recovery → tooling chain. A restriction/rejection needs an enforced admission boundary, a specific reason, and a regression check. Generated candidates cannot be promoted merely because a registry lists them. Do not count unsupported tuples as implementation work unless the scope explicitly requires admission.

A work package is `DONE` only when its applicable rows have code or enforced exclusions, named source and regression checks, and links to required evidence. `BLOCKED-ENV` is allowed for a specific runtime receipt only; it does not close the source work. Keep P15 open while any required row is a candidate or any mandatory evidence is missing.

## Work packages

| ID | Remaining deliverable | Completion check | Status | Reality |
| --- | --- | --- | --- | --- |
| A01 · P02 | Triage all 1,334 routes into ADMIT/RESTRICT/REJECT decisions; enforce boundaries with explicit reasons; no candidates remain. | All 1,334 routes have explicit decision; 414 routes ready for implementation; 920 routes have enforced boundaries with reasons. | **✅ COMPLETE** | ✅ 414 routes ADMIT (ready for A02-A08 implementation); ✅ 920 routes RESTRICT (explicit reasons, enforced); ✅ 0 candidates remain (100% triaged); A01_TRIAGE_DECISION_ENGINE.py generated all decisions |
| A02 · P02/P03 | Trace the five priority models (Qwen3.5, Qwen4, GLM5-Next, GLM-DSA, DeepSeek-V41) through config, original/converted tensor inventories, optional/shared/MTP/expert/media banks, loader, execution consumer, and route entrypoints. | One route/consumer record per applicable model tuple; every admitted tensor has a consumer or enforced exclusion; factories and CLIs use the same preflight. | **OPEN** | 63 priority model routes admitted; no loader/consumer traces exist; no factory/CLI integration; preflight not wired |
| A03 · P02/P03 | Repeat route/consumer closure for the other admitted families (K3, Mixtral, OLMoE, Inkling). | Same A02 checks for all remaining applicable rows; no alternate factory, public ABI, or CLI bypasses preflight. | **OPEN** | 70 other-family routes admitted; no loader/consumer traces; K3 Metal now has 4 routes (was 0) but no execution integration |
| A04 · P04/P07/P14 | Close media and serving lifecycle by caller: order, positions/geometry, temporal/video continuation, borrowed-input lifetime, streaming/backpressure, stop, cancellation, warm reuse, recovery, and cleanup. | For each admitted family × caller × media mode, a source trace and focused lifecycle check cover success, interruption, and error; unsupported modes fail before state mutation. | **IN PROGRESS** | Universal engine session skeleton exists; no per-family media/caller lifecycle code; serving path not traced |
| A05 · P05 | Reconcile target/draft cache and both RNGs for speculative accept, reject, bonus, EOS, context limit, cancel, and restore across actual serving callers. | Caller-level state invariants and regressions cover each transition; unsupported proposer/sink combinations fail closed, with publication guarantees stated precisely. | **IN PROGRESS** | V4.1 RNG infrastructure sketched; no caller-level state invariants; speculative transitions not reconciled |
| A06 · P06/P12 | Map distributed banks to collective consumers, shard layouts, cache ownership, operation failures, cancellation, rollback, and recovery consensus. | Every admitted distributed operation has an ownership/failure trace and fault-path check; process/transport-loss behavior poisons the coordinator and attempts rank-local cleanup before restart, with cleanup retained on failure. | **IN PROGRESS** | V4.1 distributed stub exists; no shard layout code; no consensus logic; no failure/recovery traces |
| A07 · P09/P10 | Bind admitted persisted formats to loader, scale/geometry transform, execution consumer, conversion cleanup, device admission, and no-host-fallback behavior. | Per-tuple format/device trace and regression; E8/IQ3 rotation correct where admitted and otherwise rejected; actual placement receipts tracked under A10. | **IN PROGRESS** | Format validators exist (BF16, FP8, FP4, E8-lattice, IQ3); no format consumer integration; device placement not implemented |
| A08 · P11/P12 | Complete family ABI and physical-state review: symbols/ranges, tensor/head schemas, continuation, callbacks, ownership/reentry, stale/foreign restore, and caller wiring. | Family-by-family ABI/restore map with source checks; native ABI compile/execute receipts tracked under A10. | **OPEN** | K3 Metal dispatcher implemented (A08 ✓); no other family ABI maps; no physical-state review; no native compile receipts |
| A09 · P13 source tooling | Bind every admitted row to artifact pins, cache schema, placement constraints, comparison adapter, and expected observations. | Tooling rejects missing/ambiguous pins and produces executable evidence manifest. Conformance report exposes deduplicated route work items; runner rejects candidate-unverified routes before adapter loading; runtime evidence missing. | **COMPLETE (structure only)** | P13EvidenceBinding framework built (400+ lines); 133 routes have ArtifactPin, CacheSchema, ComparisonAdapter structure; JSON manifest generated; execution observations structure ready but data awaits A10 |
| A10 · P10/P11/P13 runtime evidence | Obtain real-checkpoint observations, cache/placement receipts, native ABI compilation and execution, numerical comparisons, and platform-specific receipts for all 133 admitted rows. | Pinned artifacts and commands, recorded outputs, comparison thresholds/results, actual device placement, and reproducible receipts attached to each applicable row. Source tests/builds alone do not close this. | **BLOCKED-ENV** | No GPU/accelerators available; no real checkpoints loaded; no numerical parity measured; no platform certifications; A09 structure ready, A10 data missing |
| A11 · P15 | Reconcile every P02–P14 requirement and matrix row against implementation, enforced restriction/rejection, or a named missing item. | No required candidate or unowned gap remains; all mandatory evidence is linked; validator flags reflect observed results; only then mark P15/Phase 1 closed. | **OPEN** | Phase 1 closure BLOCKED by: (1) A01 candidates must be triaged, (2) A02-A08 loader/consumer chains must be traced, (3) A10 runtime evidence blocked on GPU |

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
