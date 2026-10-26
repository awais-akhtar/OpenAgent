# Phase 1 — completion guide

Status basis: the latest recorded reconciliation in `PHASE_1_ACTIVE_TODO.md` and
`docs/audits/phase1/source-checklist.md`, dated 25 October 2026. This is an
execution guide, not a fresh verifier run. Do not infer closure from the age of
the notes or from test counts.

## Where the phase stands

- Phase 1 is **OPEN**. The recorded canonical verifier result is **10/17**;
  seven verifier checks remain open. This is a gate count, not a percentage of
  P requirements complete.
- Source validation, 162 exact route bindings, 178 source-owner references,
  and static ABI consistency (98 exports / 190 ctypes references) pass in the
  latest recorded run. These are source checks, not proof of execution.
- The admitted matrix has 171 tuples and zero unresolved candidates. Of those,
  **126/171 admitted tuples lack exact runtime receipts**. The 45 OLMoE tuples
  have receipts; do not transfer those receipts to other identities, formats,
  or devices.
- Latest recorded missing-receipt breakdown: Mixtral 18, Qwen3.5-MoE 17,
  Qwen4 21, GLM5-Next 21, Inkling 21, GLM-DSA 9, Kimi K3 10, DeepSeek V4.1 9.
  Backend totals: native 46, device 40, public-range 40.
- The recorded host lacked an accelerator and suitable in-scope family
  checkpoints beyond OLMoE. Recheck these prerequisites before scheduling
  captures. Dense `qwen3_5` is explicitly out of scope and is not evidence for
  Qwen3.5-MoE.

## Ordered work — close each line with evidence

### 1. Establish a current baseline

- [ ] Read `PHASE_1_ACTIVE_TODO.md` and the latest dated section of
  `docs/audits/phase1/source-checklist.md`; inspect `git status --short` and
  preserve all existing changes.
- [ ] Run `phase1_verify.py` once and record the actual date, gate count,
  admitted/candidate/restricted/rejected totals, and per-route receipt gaps.
- [ ] Recheck local checkpoints, manifests, compilers, accelerator availability,
  and device count. Record unavailable prerequisites as environment blockers;
  never fabricate receipts or substitute a different model identity.

### 2. P02 — finish route-by-route path closure

- [ ] For every admitted family/backend/format tuple, trace the exact chain:
  config and tensor inventory → persisted loader → logits/execution consumer →
  cache and tensor-parallel owner → ABI → factory → CLI/gateway → recovery →
  evidence/tooling.
- [ ] Compare each binding with reachable production calls. A binding/catalog
  row is not proof that the path is called.
- [ ] Where any link is absent or unsafe, either implement and connect that
  link, or reject that exact tuple before checkpoint/resource/state allocation
  at every reachable entrypoint. State the concrete reason in the matrix.
- [ ] Re-run the canonical route validator. Closure condition: no candidates,
  no falsely admitted tuples, and each of the nine family route decisions
  passes without verifier-side mutation.

### 3. P03 — connect inventory to actual consumers

- [ ] For every admitted family, map required, optional, shared, MTP, expert,
  and media tensors from header admission through partitioning and loader to
  their concrete execution consumer.
- [ ] Check namespace aliases, duplicate/colliding names, absent optional banks,
  geometry, and unsupported extra tensors. Ensure factories and CLIs cannot
  bypass the family inventory planner.
- [ ] Add a production-boundary regression for each discovered mismatch and
  execute representative complete checkpoints when available.
- [ ] Closure condition: every admitted tensor has a real consumer or is
  explicitly and consistently rejected; no catalog-only mappings count.

### 4. P04/P07 — close media and K3 execution paths

- [ ] For Qwen3.5-MoE, Qwen4, GLM5-Next, Inkling, Kimi K3, and DeepSeek V4.1,
  trace applicable image/audio/video inputs through validation, ordering,
  placeholder expansion, preprocessing, positions/geometry, borrowed-buffer
  lifetime, native dispatch, logits, continuation, cancellation, and cleanup.
- [ ] Test mixed-item order and failures during construction/dispatch. Reject
  unsupported media types before allocating model or stream state.
- [ ] For K3 specifically, close temporal/video state, FLA/native continuation,
  batch rollback, and real serving callers. Keep incomplete device-hybrid
  routes rejected until a real continuation owner is connected.
- [ ] Closure condition: each supported media contract is exercised end to end;
  each unsupported case fails closed with a tested reason.

### 5. P05 — close speculative state and publication semantics

- [ ] Trace target/draft token state and both RNG states through accepted and
  rejected draft tokens, bonus token, EOS, context limit, stop, cancellation,
  restore, and every serving caller.
- [ ] Fix the recorded defect where snapshot cleanup can fail after decoder
  state commits: define and implement one observable outcome so the caller
  cannot receive an error while losing the committed step result. Preserve
  retryable cleanup/quarantine behavior and avoid rolling back a live worker.
- [ ] Verify output publication failure and arbitrary sink behavior. If a sink
  cannot support atomic publication, expose a concrete restriction or an
  explicit partial-publication contract; do not claim atomicity.
- [ ] Closure condition: fault-injection regressions prove token/RNG/receipt
  state and caller-visible output agree on every terminal path.

### 6. P06/P12 — close distributed operations and recovery

- [ ] Map every distributed bank to its actual shard layout and collective
  consumer; validate rank/world-size ownership and cache placement.
- [ ] Inject failures at preflight, each collective, result agreement, cache
  install, cancellation, rollback, and restore. Verify bounded timeouts, peer
  wake-up, poisoning/restart rules, and no stale or foreign receipt acceptance.
- [ ] Cover process death and transport loss. If the transport cannot guarantee
  recovery, retain a specific restriction and document that a fresh group is
  required; do not describe local timeout tests as consensus after arbitrary
  process loss.
- [ ] Reconcile physical-state ownership, reentry, cleanup retries, and
  distributed recovery for each family owner.
- [ ] Closure condition: each supported failure boundary has an operation-level
  regression and a deterministic rollback/restart outcome, or is explicitly
  rejected before work starts.

### 7. P09/P10 — close formats and placement

- [ ] For every admitted tuple, connect persisted format and exact geometry,
  scales, and transforms to the real loader and execution kernel. Cover E8
  activation rotation, IQ3, quantized scale layouts, and conversion cleanup.
- [ ] Confirm restrictions are enforced at factories and CLIs, not merely in a
  registry. Verify errors do not fall back to host execution or reinterpret a
  format.
- [ ] For device tuples, collect runtime placement evidence from actual device
  execution and demonstrate no host fallback. Registry selection is
  insufficient.
- [ ] Closure condition: loader/consumer tests cover each admitted format and
  actual placement receipts cover admitted device routes; incomplete tuples are
  concretely rejected.

### 8. P11 — close the public ABI map

- [ ] Enumerate all exported declarations and ctypes references by family;
  resolve each symbol and compare calling convention, parameter/return types,
  ranges, tensor/head schemas, and ownership rules.
- [ ] Exercise continuation, callbacks, reentry, error returns, and release
  behavior against compiled native artifacts for each applicable ABI.
- [ ] Verify the same ABI path is wired through its factory, CLI, and service;
  source-text consistency alone does not prove native execution.
- [ ] Closure condition: every admitted ABI has compiled execution evidence or
  its route is rejected before loading.

### 9. P14 — finish caller-by-caller serving lifecycle

- [ ] Inventory each production caller (direct engine, publisher, inference
  service, Phase 2/multi-session, chat, OpenAI, Anthropic, and family CLIs) and
  trace request admission through response/stream completion and owner teardown.
- [ ] For each caller, cover media order, streaming/backpressure, client
  disconnect, stop sequences, cancellation, warm-session option matching,
  auth/error mapping, recovery, and cleanup after partial construction.
- [ ] Verify bounded queue behavior, timeouts/429 and wait headers, health and
  Prometheus counters where those capabilities are claimed. Record fixture
  integration separately from live socket/runtime evidence.
- [ ] Explicitly implement or reject remaining compatibility features at the
  public boundary: complete OpenAI/Anthropic semantics, persistent isolated
  `cache_slot` and clients, GBNF/tools, READY/mux, route telemetry/cache-aware
  routing, and multi-disk mirror/planning.
- [ ] Closure condition: every production caller has a completed lifecycle
  matrix with regressions for success, failure, cancellation, abandonment, and
  retryable cleanup; unsupported protocol branches fail closed.

### 10. P13 — obtain real route-bound evidence

- [ ] Implement capture adapters for each applicable production owner (family,
  media, speculation, distributed, and device as required). Capture only
  observations emitted by the exact route and pinned artifacts.
- [ ] For each of the 126 missing admitted rows, collect every gate-required
  kind: tokens/logits, cache, routing, recovery, independent comparison, and
  actual placement, or route-derived not-applicable where the gate permits it.
- [ ] Pin the exact checkpoint/config/weight shards, publisher case, manifest,
  backend, device, and independent reference. Run the canonical gate after each
  capture batch and record accepted/rejected rows and reasons.
- [ ] Current historical gap is 126 rows: 46 native, 40 device, 40 public-range.
  Prioritize rows with available exact checkpoints and CPU paths; acquire the
  exact missing checkpoint/toolchain/device for the rest. Do not reuse OLMoE
  receipts or test fixtures as substitutes.
- [ ] Closure condition: every admitted route has valid exact receipts for all
  applicable evidence kinds; no synthetic or generic fixture satisfies P13.

### 11. P15 — reconcile and close

- [ ] For every P02–P14 requirement and every route/feature, record one of:
  **implemented and evidenced**, **restricted with an enforced reason**, or
  **missing**. Name code paths, regressions, receipt IDs, and environment
  blockers.
- [ ] Resolve every remaining `missing` item. A `BLOCKED-ENV` item remains
  open until its required evidence is obtained; it is not a completion state.
- [ ] Run the canonical verifier and required final source/test/ABI checks on
  the final tree. Reconcile the detailed checklist and this queue to the same
  result; preserve earlier dated notes as history.
- [ ] Close Phase 1 only if the verifier reports closure and the requirement-
  by-requirement reconciliation agrees. Otherwise keep P15 **OPEN** and list
  the exact remaining routes, receipts, or restrictions.

## Work-record format

For every completed slice, add one dated note to both the active TODO and the
detailed checklist containing: P item and exact route/caller; changed source
files; implementation or enforced restriction; focused regression and command
result; verifier/evidence count before and after; remaining dependencies. Do
not repeat completed notes, infer percentages, or turn an unavailable hardware
receipt into a source-code completion claim.
