# Phase 1 — current audit and completion queue

Updated 26 October 2026 against the uncommitted `D:\openagent` working tree
(HEAD baseline `70bae5a`). This is the current task list. Older dated status reports must not
be read as current completion evidence. The detailed P02–P15 criteria and audit
history are in [docs/audits/phase1/source-checklist.md](docs/audits/phase1/source-checklist.md);
this untracked root file is the concise working-tree queue. The detailed
checklist’s current reconciliation is dated 26 October; older dated entries
remain historical and are not current completion evidence.

For the ordered, line-by-line completion procedure and closure criteria, see
[PHASE_1_COMPLETION_GUIDE.md](PHASE_1_COMPLETION_GUIDE.md). This guide is based
on the latest recorded reconciliation and does not claim a fresh verifier run.

## Current result

P14 shared batching/forward execution boundary (26 October): partially
prefilled requests are scheduled again and commit exactly their reserved
prompt extent instead of skipping tokens or stalling. Premature prefill token
results are rejected. Fatal forward, scheduling and failure-callback errors
wake every waiter and seal admission before a dead-worker queue can form.
Duplicate live IDs and completed-request resubmission are rejected.

The public Phase2ForwardBridge no longer fabricates random output, retries a
second generation API after state mutation, or hides model/factory failures.
It validates one-token output/logit geometry, retains failed slot closes,
releases completed owners, rejects foreign request ID reuse and serializes
dispatch/teardown. Removed the obsolete duplicate commit implementation.
The bridge-specific shared KV path is explicitly rejected before allocation:
its old placeholder cache publication had no physical/schema/position binding.
Other native cache paths are unchanged. Prompt chunking here stages context;
it does not certify physically chunked native prefill or cache reuse.

Current regression batch: 211 tests pass plus 12/12 batching smoke checks;
scoped Ruff and Python syntax checks pass. Fresh canonical verification remains
10/17 with 126/171 missing route receipts and passing source/ABI checks.
No runtime receipt was generated. Caller-wide P14 and P13/P15 remain open.

P04/P06/P12 cleanup batch (25 October): expired exact-owned universal receipts
can now be released without permitting stale restore; foreign/forged receipt
checks remain enforced. Release-property lookup failures retain transaction
snapshots for retry and preserve the request error. Cancelled media errors
with hostile exception formatting now transition and release their frames.
Distributed factory cleanup-property failures cannot skip destruction of
either process group, and custom exception note methods cannot mask the
primary failure. Regressions reproduced each defect before its fix.

Combined local regression result: 268 tests run, 266 passed and 2 skipped
(platform-dependent process-loss tests). Scoped Ruff passes; the compressed
multimodal files were checked excluding their existing E701/E702 style rules.
Python syntax and scoped diff checks pass. The existing local Segment runtime
DLL resolves all 101 public-header exports, and its GLM/Inkling descriptor,
Tensor, Config and Layer sizes/field counts/offsets match ctypes. These binary
checks do not certify model execution or all-platform ABI compatibility.
Canonical verification remains 10/17 with 126/171 receipts missing and passing
source/ABI checks. No model-route receipt was created; P13/P15 remain open.

P04/P06/P12 cleanup batch (25 October): expired exact-owned universal receipts
can now be released without permitting stale restore; foreign/forged receipt
checks remain enforced. Release-property lookup failures retain transaction
snapshots for retry and preserve the request error. Cancelled media errors
with hostile exception formatting now transition and release their frames.
Distributed factory cleanup-property failures cannot skip destruction of
either process group, and custom exception note methods cannot mask the
primary failure. Regressions reproduced each defect before its fix.

Combined local regression result: 268 tests run, 266 passed and 2 skipped
(platform-dependent process-loss tests). Scoped Ruff passes; the compressed
multimodal files were checked excluding their existing E701/E702 style rules.
Python syntax and scoped diff checks pass. The existing local Segment runtime
DLL resolves all 101 public-header exports, and its GLM/Inkling descriptor,
Tensor, Config and Layer sizes/field counts/offsets match ctypes. These binary
checks do not certify model execution or all-platform ABI compatibility.
Canonical verification remains 10/17 with 126/171 receipts missing and passing
source/ABI checks. No model-route receipt was created; P13/P15 remain open.

P12/P14 output and recovery boundary batch (25 October): direct serving now
rejects invalid max_tokens/n values and nonboolean stream flags before opening
sessions; streaming explicitly rejects n != 1. Backend token iterators are
consumed to at most max_tokens without pulling an extra token, closed at that
boundary, and rejected for fractional/string/boolean/negative/out-of-range
token IDs. Stop-prefix tails at the limit are flushed without over-reading.
Safe bounded exception diagnostics preserve completion/chat error responses.
Eager backend computation itself remains outside this output-consumption bound.

Universal stream close-property lookup now stays inside cleanup handling, so
its failure cannot skip rollback or snapshot release. Completed terminal results
remain committed; incomplete streams roll back. Owner close-property failures
are recorded for retry, and context-manager teardown cannot replace a primary
body error; the primary retains a cleanup callback. Regressions reproduced the
rollback bypass and masked body error before fixing them. Serving/HTTP tests
pass 134/134 and universal adapter/production/lifecycle tests pass 68/68.
Scoped Ruff, Python syntax and diff checks pass. Canonical verification remains
10/17 with 126/171 missing receipts and passing source/ABI checks. No route
receipt was created; these are selected source-path closures, not full P12/P14
or P13/P15 completion.

P12/P14 output and recovery boundary batch (25 October): direct serving now
rejects invalid max_tokens/n values and nonboolean stream flags before opening
sessions; streaming explicitly rejects n != 1. Backend token iterators are
consumed to at most max_tokens without pulling an extra token, closed at that
boundary, and rejected for fractional/string/boolean/negative/out-of-range
token IDs. Stop-prefix tails at the limit are flushed without over-reading.
Safe bounded exception diagnostics preserve completion/chat error responses.
Eager backend computation itself remains outside this output-consumption bound.

Universal stream close-property lookup now stays inside cleanup handling, so
its failure cannot skip rollback or snapshot release. Completed terminal results
remain committed; incomplete streams roll back. Owner close-property failures
are recorded for retry, and context-manager teardown cannot replace a primary
body error; the primary retains a cleanup callback. Regressions reproduced the
rollback bypass and masked body error before fixing them. Serving/HTTP tests
pass 134/134 and universal adapter/production/lifecycle tests pass 68/68.
Scoped Ruff, Python syntax and diff checks pass. Canonical verification remains
10/17 with 126/171 missing receipts and passing source/ABI checks. No route
receipt was created; these are selected source-path closures, not full P12/P14
or P13/P15 completion.

P14 lazy-generation owner lifetime (25 October): fixed direct serving closing
the session before consuming lazy generation results. Token materialization
and iterator release now precede owner teardown; lazy iteration errors retain
priority over iterator/session cleanup failures. Close-property lookup is
inside its cleanup guard, and streaming no longer resolves an unused generate
property. Generate results are checked for None rather than ambiguous container
truth values. Four regressions reproduced failures before the fix; the expanded
serving/gateway/inference/Phase 2 group passes 116 tests, with scoped Ruff,
Python syntax and diff checks passing. This closes the selected direct-serving
ownership path, not all P14 callers. Canonical verification remains 10/17 with
126/171 admitted routes missing receipts; source/ABI checks pass. No model
receipt was added; P13/P15 remain open.

P14 lazy-generation owner lifetime (25 October): fixed direct serving closing
the session before consuming lazy generation results. Token materialization
and iterator release now precede owner teardown; lazy iteration errors retain
priority over iterator/session cleanup failures. Close-property lookup is
inside its cleanup guard, and streaming no longer resolves an unused generate
property. Generate results are checked for None rather than ambiguous container
truth values. Four regressions reproduced failures before the fix; the expanded
serving/gateway/inference/Phase 2 group passes 116 tests, with scoped Ruff,
Python syntax and diff checks passing. This closes the selected direct-serving
ownership path, not all P14 callers. Canonical verification remains 10/17 with
126/171 admitted routes missing receipts; source/ABI checks pass. No model
receipt was added; P13/P15 remain open.

Four audited capture defects fixed (25 October): publisher media evidence is
retained after a receipt publication attempt, including post-commit unlock
errors (an uncertain pre-commit failure may leave a recoverable directory).
Receipt-lock cleanup always attempts file close and preserves the primary
body failure. Malformed route objects/case IDs/manifest paths fail with
`CaptureError`. Plan creation rejects output aliases of its asset configuration
or receipt manifest and writes plans atomically. The isolated publication,
locking/concurrency and plan suite passes 17 tests; 54 related capture tests
also pass. Python syntax checks pass. Ruff passes for media publication and
new publication/lock tests; pre-existing compressed-style E701/E702 findings
remain in the plan runner/tests. The canonical verifier still reports 10/17,
126/171 missing receipts, with source and ABI checks passing. No new model
runtime evidence was generated. P13/P15 remain open.

Four audited capture defects fixed (25 October): publisher media evidence is
retained after a receipt publication attempt, including post-commit unlock
errors (an uncertain pre-commit failure may leave a recoverable directory).
Receipt-lock cleanup always attempts file close and preserves the primary
body failure. Malformed route objects/case IDs/manifest paths fail with
`CaptureError`. Plan creation rejects output aliases of its asset configuration
or receipt manifest and writes plans atomically. The isolated publication,
locking/concurrency and plan suite passes 17 tests; 54 related capture tests
also pass. Python syntax checks pass. Ruff passes for media publication and
new publication/lock tests; pre-existing compressed-style E701/E702 findings
remain in the plan runner/tests. The canonical verifier still reports 10/17,
126/171 missing receipts, with source and ABI checks passing. No new model
runtime evidence was generated. P13/P15 remain open.

Colab batch reliability follow-up (25 October): `phase1_colab.py` now saves an
atomic progress summary after every completed route, retains the completed
summary on capture interruption, records subprocess launch failures without
aborting the remaining batch, and supports an optional finite positive
`--route-timeout`. Reports cannot overwrite the plan, asset configuration or
canonical receipt manifest. Failed report replacement preserves the previous
summary. Summaries never substitute for gate-validated route receipts.
The selected capture regression group passes 70 tests; the expanded runner
module passes 10 tests after two further file-safety regressions. Python
syntax checks pass. Canonical verification remains 10/17 with 126/171
receipts missing; no runtime receipt was created in this follow-up.

Colab batch reliability follow-up (25 October): `phase1_colab.py` now saves an
atomic progress summary after every completed route, retains the completed
summary on capture interruption, records subprocess launch failures without
aborting the remaining batch, and supports an optional finite positive
`--route-timeout`. Reports cannot overwrite the plan, asset configuration or
canonical receipt manifest. Failed report replacement preserves the previous
summary. Summaries never substitute for gate-validated route receipts.
The selected capture regression group passes 70 tests; the expanded runner
module passes 10 tests after two further file-safety regressions. Python
syntax checks pass. Canonical verification remains 10/17 with 126/171
receipts missing; no runtime receipt was created in this follow-up.

Fresh verification rerun (25 October): the 98-test capture suite passes in
23.958 seconds. `phase1_verify.py` again reports 10/17, 126/171 receipts
missing, with source bindings/owners/ABI checks passing. The 108-target result
therefore remains 45 accepted receipts and 63 missing. Workspace artifact
inventory finds OLMoE checkpoints and dense Qwen3.5, but no Inkling, Qwen4 or
GLM5 checkpoint or configured Colab assets/plan. `nvidia-smi` is not found;
the current `.venv` has no PyTorch installed (distinct from the previously
reported accelerator virtualenv). No additional model capture was possible
from these local prerequisites, and no new receipt was created.

Fresh verification rerun (25 October): the 98-test capture suite passes in
23.958 seconds. `phase1_verify.py` again reports 10/17, 126/171 receipts
missing, with source bindings/owners/ABI checks passing. The 108-target result
therefore remains 45 accepted receipts and 63 missing. Workspace artifact
inventory finds OLMoE checkpoints and dense Qwen3.5, but no Inkling, Qwen4 or
GLM5 checkpoint or configured Colab assets/plan. `nvidia-smi` is not found;
the current `.venv` has no PyTorch installed (distinct from the previously
reported accelerator virtualenv). No additional model capture was possible
from these local prerequisites, and no new receipt was created.

**Phase 1 is OPEN.** `validate_phase1_source()` and the static public-ABI
consistency audit pass. The latest `phase1_verify.py` run reports **10/17**
with `phase1_closed=false`: all nine family route-decision checks and A01 pass
with zero candidate tuples; seven evidence checks remain open because **126 of
171 admitted rows** lack exact runtime receipts. All **45/45 admitted OLMoE
routes** have validated CPU receipts. The 1,334-row matrix records 171
admitted, 0 candidates, 877 restricted, and 286 rejected tuples. The ABI audit
checks 101 exported declarations against definitions and 190 ctypes references,
including explicit calling-convention token consistency. The GLM/Inkling public
registration path also compares C and ctypes size/field-count/offsets for its
descriptor, tensor, config, and layer records; focused regressions and the
existing local runtime DLL confirm those four layouts match. This does not
prove all-family platform ABI compatibility, all-export native execution, or
complete caller wiring. Existing OLMoE
format receipts and their provenance limitations are unchanged.

The **126 remaining receipt gaps** partition as: Mixtral 18, Qwen3.5 17,
Qwen4 21, GLM5-Next 21, Inkling 21, GLM-DSA 9, Kimi K3 10, and DeepSeek V4.1
9. By backend they are native 46, device 40, and public-range 40. OLMoE has no
remaining admitted-route receipt gaps. Continue with exact route captures for
other families only when matching local checkpoints and required backends or
devices are available; do not fabricate runtime evidence or relabel a
different route as a substitute.

The 108-route Colab target now has a canonical command planner and resumable
runner in `openagent/adaptive_engine/phase1_colab.py`. Its 108 unique entries
resolve the actual CLI contracts: OLMoE 45, Inkling 21, Qwen4 21, GLM5 21.
This supersedes the old 24/63 source-command count. It does not certify model
execution: all 63 non-OLMoE target rows still need accepted runtime receipts.
The rebuilt `PHASE_1_COLAB_COMPLETE_108.ipynb` uses plan/preflight/run/check;
preflight rejects mismatched checkpoint identities, source encodings and
missing artifacts. Exit code zero alone never counts as an accepted route.

The latest source batch fixes Qwen4 double encoding, full-prefill logit row
count, public-range reference ownership and continuation RoPE positions;
GLM device loader arguments, publisher text geometry and reference expert
library ownership; public wrapper identity mutation; and Inkling MTP media
input vocabulary, EOS/context handling, RNG cleanup and CLI receipt paths.
A direct Qwen4 publisher media CLI now obtains real local processor fixtures.
Native Qwen4's stale rotary call was also corrected after an actual build
exposed it. Earlier no-compiler notes were based on PATH only: the compiler
helper now discovers installed MSVC using vswhere. Media, Inkling, Qwen4 and
GLM5 native kernels compile successfully with the helper.

Validation: 98 focused tests pass, including 108 exact command bindings,
false-success rejection, raw logit geometry, RoPE continuation, GLM loader
signature, and Inkling sampling boundaries. Touched Python sources compile;
notebook JSON/code-cell syntax and scoped diff checks pass. Canonical source
validation passes (162 bindings, 187 owners, 101 exports/190 ctypes names).
Global closure remains 10/17 with 126/171 receipts missing. Actual checkpoint,
accelerator, independent comparison and media execution on Colab remain
unverified; P13/P15 stay open. No new runtime receipts were generated.

Environment recheck (25 October): the local `models/qwen3.5-0.8b` config is
`Qwen3_5ForConditionalGeneration` / `model_type=qwen3_5` (dense, explicitly
out of the Phase 1 MoE scope), and no DeepSeek checkpoint is present under the
workspace model roots. `nvidia-smi` is unavailable; the accelerator virtualenv
has CPU-only PyTorch (`cuda_available=False`, zero CUDA devices, XPU unavailable).
Thus it cannot produce the missing device/family receipts from current local
artifacts. Do not use the dense Qwen checkpoint as Qwen3.5-MoE evidence.

Latest Colab-capture audit and source update (25 October): the newly supplied
`inkling_capture.py`, `qwen4_capture.py`, `qwen35_capture.py`, and notebook did
not implement their advertised evidence paths. The Qwen runners used an
independent NumPy decoder as the candidate for native/device/public-range
routes, wrote incomplete cache/routing/media/speculation records, and could
not satisfy the evidence gate. Replaced the Qwen4/Qwen3.5 adapters with a
compiled native CPU text runner, added the actual native decoder ABI owners to
those exact P02 rows, and added capture-contract regressions. A real native
Inkling text capture is also wired: standalone `inkling_text` correctly has no
MTP coverage obligation (its inventory rejects MTP), and its C decoder now
owns bounded exact-object snapshot/restore/release receipts for cache recovery.
The multimodal `inkling_mm_model` retains MTP/RNG and media requirements and is
not accepted by the text capture. The native text runners support only
F32/F16/BF16. Standalone Inkling and Qwen4 text now also have exact device
capture paths; public-range and multimodal combinations remain unsupported by
these capture runners except the exact Qwen4 and Inkling text public-range
paths added below. Rewrote the Colab notebook so it no longer claims 47
routes or CPU evidence for unavailable backends. No Qwen/Inkling receipts were
produced here. Python syntax checks,
CLI help, source-scope validation, all 162 route bindings, all 182 callable-owner
references, notebook JSON/cell validation, and the 6 contract tests pass.
Follow-up: GLM5 text-native capture is wired to its separately implemented
Torch CUDA/MPS comparator; GLM5 text-device capture now runs the real Torch
device decoder against a separately compiled native C reference, with a
backend-specific pinned reference contract. The native owner has bounded
snapshot/restore/release for complete KDA/DSA state, and its exact decoder/C
source is bound in P02. The Colab notebook now includes both text-native and
text-device loops; the device path requires an explicit accelerator and
compiled reference library.
Seventeen focused tests now
pass, including exact GLM snapshot ownership, dict/tensor observation handling,
and reference-source pins. The
project virtualenv has NumPy but no Torch; neither GCC nor MSVC `cl` is
available, so no GLM compile, accelerator comparison, or Qwen/Inkling/GLM
receipts were produced. The canonical verifier remains 10/17 with 126/171
missing receipts. The 108 target still needs Qwen4 publisher
multimodal/device/public-range; Inkling multimodal target/draft/MTP/RNG/media
plus device/public-range; GLM5 publisher media/public-range; and actual
per-route Colab captures.

Qwen4 and Inkling exact text-device captures are now wired to their real
`TorchQwen4Decoder` / `TorchInklingDecoder` owners on an explicitly selected
CUDA/MPS device, compared with separate NumPy implementations, and bound to
their exact device P02 routes. Receipts pin the candidate forward/source
closures and shared capture/gate files. Notebook device cells follow checkpoint
definitions. With GLM5 text-device capture added, this brings capture-path
coverage to at most 18 of the 63 non-OLMoE target rows (three native and three
device text routes each for Inkling, Qwen4, and GLM5). The other 45 target rows
still lack capture paths, including publisher media/MTP and public-range routes. No fresh
receipts were produced. The focused capture/route/evidence suite passes 55/55;
Python syntax, notebook JSON/order, source validation, and `git diff --check`
pass. The canonical verifier remains 10/17 with 126/171 receipts missing. Do
not count setup or source checks as receipts.

Verification of the GLM5 device addition: the project virtualenv passes 25/25
focused capture/reference/route tests; the changed Python sources compile,
the notebook parses as JSON, all 18 candidate source pins resolve, the CLI
advertises the required device/reference-library arguments, and canonical
source validation passes. A fresh `phase1_verify.py` run still reports 10/17
and 126/171 missing exact receipts. No native build or device probe was run,
and no receipts were generated.

Latest P14 serving-boundary batch (25 October): direct OpenAI-compatible
`ServingEngine` requests now close opened sessions on success, backend failure,
stream abandonment, and close failure; the primary generation exception is
preserved with cleanup failure attached, and a failed stream ends with
`finish_reason="error"` rather than false success. Phase 2 service shutdown
retains expert/KV stores until both the memory manager and batch worker have
stopped, and records per-stage progress so `close()` can be retried. The
load-balancer and `InferenceService` shutdown path now propagates backend close
failures and retries only incomplete stages instead of swallowing them. Backend
leases remain held for each session lifetime, release on close/open failure,
and are drained before backend teardown/removal; proxy release is idempotent
and failed session cleanup remains owned for retry. Duplicate and post-close
backend registration now fail closed. Full-scheduler rejection refunds the
rate-admission tokens it consumed; canceling a queued request frees its queue
slot immediately, and failed in-flight work retires canceled entries. The
worker also fails and retires requests when a forward callback omits, duplicates,
or malforms a scheduled result instead of leaving a caller blocked indefinitely;
empty prompts enter decode rather than staying queued forever. Scheduler
geometry and direct request budgets/IDs are validated before admission, and a
zero-output request completes without occupying the model worker. Shutdown
waits for submissions already inside admission before stopping the worker;
process-level worker failures now wake affected callers and retire the worker.
Partial Phase 2 construction now closes completed owners in order and retains a
retryable cleanup callback if a background worker cannot stop. Session close
failures no longer mark/remove the owner prematurely; service shutdown retains
the handle and shared stores for retry. The focused serving/admission/batching
group passes 70/70, the legacy Phase 3
integration smoke harness passes 25/25 in UTF-8 mode, and scoped Ruff passes.
Fresh `phase1_verify.py` is unchanged at 10/17, with 126/171 admitted routes
missing exact runtime receipts; P14 caller coverage and P13/P15 remain open.

Latest whole-tree code check (25 October): source validation, exact binding
validation (162 unique tuples), callable-owner references (178), and static ABI
consistency (98 exports/190 ctypes references) pass. Earlier discovery exposed
four defects: integration-route parser calls, Personal OS draft
ownership/update/import semantics, a missing `/api/models` route declaration,
and an incomplete runtime tool status response. Those paths were corrected and
their focused route tests passed 13/13. A later full discovery ran 2,170 tests,
97 skipped, and found two errors caused by partial `PublisherSession` fixtures
missing the newly added speculative-cleanup field. The lookup was corrected to
support those partial objects; publisher cleanup and speculative transaction
suites now pass 42/42, but the full suite has not been rerun after that
correction. Frontend serial tests pass 189/189 and the TypeScript check passes;
parallel Vitest execution had two contention-related failures, both passing in
isolation and serially. `git diff --check` is clean. The Ollama
smoke-probe certifier now fails closed and its CLI is diagnostic-only; its
179-record JSON artifact is excluded from runtime evidence. None of these
results changes the 126 missing receipts or closes P15.

The two frontend parallel-only flakes were caused by short test timeouts around
lazy Chat rendering and the multi-step Voice CRUD flow. Their waits now have
realistic bounds; the default parallel frontend suite passes 189/189 and
`npm run check` passes.

Latest P13 gate hardening (24 October): comparison receipts now bind the exact
registered decoder identity and hash every in-repository equation helper;
Mixtral receipts additionally require the pinned config geometry to match the
selected canonical publisher contract. The capture resolver now also requires
an exact `case_id` when multiple pinned publisher artifacts share one
six-field route, rejecting ambiguous or foreign pins rather than choosing a
publisher implicitly. Focused regressions pass 7/7 for the receipt gate and
1/1 for the ambiguity boundary, including the existing real tiny OLMoE capture.
This hardens validation but adds no route receipts; the 126-route
evidence gap and 10/17 closure result are unchanged.

Latest P02/P09/A07 source correction (25 October): the format-only sparse
manifest guard remains distinct from exact checkpoint routes: an FP8
checkpoint route with an exact source inventory may be admitted even though a
generic manifest without that provenance is rejected. The A07 placement
registry now records placement-specific public loaders for OLMoE INT8 and K3
MXFP4 (the K3 source checkpoint loader plus public publisher adapter) and
returns them in the format contract. The P02/A04-A07 focused set passes 27/27;
the expanded set with source-inventory coverage passes 92/92.
The canonical verifier itself previously imported a deleted duplicate module;
its entrypoint now uses the retained `integrated_closure.py`. Latest
verification passes source, owner-reference, and static ABI checks and remains
10/17 with 126 missing exact receipts. Source admission totals are unchanged.

Latest P04 publisher-media boundary (25 October): `NativePublisherProvider`
now keeps a process-local `FamilyMediaLifecycleRegistry`, admits media frames
before dispatch, records decoded media/video-frame consumption at the native
session call (not at generated-text publication), and completes/cancels/errors
and releases the frame on every terminal path. Provider shutdown closes the
registry even when decoder cleanup fails. The decoded-media gateway now also
rejects floating-point waveform inputs with the precise missing sampling-rate
and placeholder-order contract instead of misreporting them as malformed
images. Publisher, gateway-vision, multimodal-session and lifecycle tests
pass 51/51; Ruff F821 and
`git diff --check` pass for the touched files. This
closes lifecycle accounting for the shared publisher request edge only; media
processing remains bounded to each family's existing provider contract, and
the 126 exact-route evidence gaps and 10/17 verifier result are unchanged.

Latest P04 conditional-executor cleanup correction (25 October): if a decoder
fails and the cancellation observer raises while classifying the failure, the
executor now retains the decoder error, records the observer failure as a
secondary note, marks/releases the media frame as an error, and cannot strand
an active frame. The injected double-failure regression passes with the
multimodal lifecycle suite. This closes that cancellation-classification edge;
cross-family and serving-caller closure remains open.

Latest A04 lifecycle error/ownership correction (25 October): media error
handling now attaches diagnostics to the original backend exception instead
of calling the two-argument note helper incorrectly (which could mask the
decoder failure with `TypeError`). Terminal methods return detached frame
records so callers cannot mutate internal lifecycle state. Registry shutdown
now attempts every handler, preserves the first failure and retains handlers
for retry. Media, publisher-media, multimodal, A05 and A06 focused suites pass
45/45 under the project virtualenv. This closes these lifecycle helper edges,
not cross-family caller coverage or missing runtime receipts.

Latest P14 publisher pre-cancellation guard (25 October):
`NativePublisherProvider._stream_run` checks request cancellation on the
consumer thread before it creates a worker or enters model loading. An already
cancelled stream now exits without occupying the provider or starting backend
work; the publisher-stream, inference-route and chat-route focused set passes
30/30. Cancellation after startup remains cooperative and depends on backend
checkpoints; this does not close the other serving callers or provide live
socket evidence.

Latest P14 cleanup fix (24 October): when universal-session validation or a
publisher-owner constructor fails and closing the just-created native owner
also fails, the primary exception now retains a callable retry cleanup instead
of losing the resource after an error note. Universal-engine and publisher
adapter lifecycle suites pass 39/39; this closes those two construction-failure
boundaries, not the remaining caller-by-caller serving lifecycle audit.

Latest P02/A02 factory repair (24 October): Qwen3.5 text, Qwen4 text, GLM5 text
and GLM-DSA factory entrypoints now require the checkpoint's own config and an
explicit native library, preflight exact native/FP32/float32 inventory before
loading, and build a fresh mutable decoder for each service session. The
multimodal Qwen3.5 wrapper is rejected by the text-only factory. V4.1 delegates
to the existing three-bank publisher service and exact inventory gate. The
factory/source-inventory tests pass 103/103; this does not capture route
receipts or certify compiled decoders.

Latest P02 trace correction (24 October): the A02 source catalog no longer
attributes nonexistent family cache-recovery classes, tensor-parallel owners,
or CLI wiring to the four programmatic text factories. It records each exact
factory `model_type` separately from the canonical family route identity and
describes service-owned snapshots/cache transport accurately. The trace
catalog now says explicitly that populated references are not proof of a
resolved or runtime-certified route. Its focused assertions are included in
the 103/103 result. Full family CLI, tensor-parallel and portable recovery
closure remains open.

Latest P14 facade-construction cleanup fix (24 October): if a production
universal owner opens but Phase 1B facade creation fails, the base owner is now
closed; if that close fails, the primary exception retains retryable cleanup.
The new regression injects both failures and verifies the retry closes the
owner. The combined focused factory, source-inventory, candidate-triage and
production-path group passes 118/118. This closes one more construction
failure edge, not all P14 callers or end-to-end lifecycle paths.

Latest P02 A02/A03 trace corrections (24 October): Mixtral, OLMoE, DeepSeek
V4.1, K3 and Inkling rows were reconciled against existing source, and the
Qwen3.5/Qwen4/GLM5 wrapper routes were separated from their distinct text-only
factory identities. Wrapper traces now follow the conditional service and its
CLI; direct text factories have separate exact-identity traces. False
cache-owner, factory, CLI, tensor-parallel, recovery and comparison-adapter
claims were removed or narrowed; missing integration is explicit. The catalog
has 12 selected source-trace rows, runs under Windows' default console encoding,
and regression assertions bind every row to an exact six-field canonical
matrix tuple. These remain source
references and explicit gaps, not proofs of all-family closure or runtime
behavior. The focused regression group remains green at 118/118.

Latest P02 exact-binding coverage (25 October): the new
`phase1_p02_route_bindings.py` materializes a validated immutable binding for
each unique source-reviewed six-field route tuple. It covers **162 unique
tuples** (171 admitted matrix rows collapse across nine publisher variants),
and preserves each row's ordered configuration→inventory→loading→execution→
cache/TP→ABI→factory→CLI→gateway→recovery→tooling owners. Validation resolves
each binding back to the exact canonical row and rejects altered reviews or
chains. The 12 hand-authored traces remain distinct examples; all 12 correspond
to admitted tuples. `phase1_verify.py` now reports this coverage and fails its
source audit if it is inconsistent. Focused A02 tests pass 9/9 and the source
filesystem validator passes. This closes the source-catalog binding coverage gap only:
it does not independently inspect every owner implementation, certify runtime
execution, or provide the **126 missing receipts**; P02/P13/P15 remain open.

Latest P03 MTP namespace guard (25 October): publisher-bank detection now
recognizes MTP path components throughout tensor names, including
`model.layers.N.mtp_block.*`, not only top-level `mtp.*` prefixes. The shared
classifier is used by direct partitioning, both conditional-loader inventory
passes, and GLM5 inventory planning. Non-Inkling families reject those tensors before
payload reads; Inkling still routes them through its exact MTP inventory
planner. `tests.test_source_scope_inventory` passes 65/65, including GLM5's
header-level alternate-namespace rejection. Broader P03
consumer reconciliation and Inkling target-verified serving remain open.

Latest P02 source-reference resolution (25 October): the verifier now parses
the owner strings in all 162 exact route bindings and confirms that every
referenced Python module/symbol or native source file exists in the workspace.
All 178 unique owner references resolve; stale names are a source-audit failure.
This is AST-level existence evidence only, not proof of call reachability or
execution. The verifier source checks pass, but it remains 10/17 with the same
126 runtime receipt gaps.

Latest P14 rollback-receipt cleanup fix (25 October): if request rollback fails,
the universal session now retains that physical receipt but defers releasing it
until owner teardown, preventing a release attempt from racing a still-running
stream worker. Ordinary transaction receipt-release failures are retained,
quarantine the session, and can be retried through the original error's cleanup
callback or `close()`. Focused universal-engine, adapter, and production-path
tests pass 60/60, including the running-worker quarantine case. This closes a
shared cleanup boundary only; the session status exposes the retained-cleanup
count, while P14 caller coverage and P15 remain open. The expanded combined
universal-engine, adapter, and production-path suites now pass 60/60.

Latest P12 restore-failure boundary (25 October): if an owner throws during
receipt restore after a partial physical install, `UniversalSession` now
quarantines the owner, retains the receipt for close-time release, and exposes
retryable close cleanup on the primary error. The focused partial-install
regression passes 1/1. This does not close family cache codecs, distributed
consensus, or runtime recovery evidence.

Latest P06 coordinator-abort safety fix (25 October): `DistributedCoordinator`
now permits an unpoisoned abort only before unanimous accepting preflight. Once
all ranks may have entered a collective—or any result has arrived—abort clears
the active round, poisons the coordinator, and requires a group restart rather
than letting later requests reuse possibly stranded peers. `SpeculativeStateManager.close()`
now refuses to erase active leases, preserving rollback ownership until the
caller resolves and releases them. The A06 + A02/A04/A05/A07 focused unittest
set passes 39/39. Audit confirmed this standalone
helper is not wired into production `DistributedExecutionService` or family
process-group construction, so the code fix is helper-only and earns no live
distributed-execution coverage. P06 still requires the production caller map,
transport-loss behavior, rollback/consensus review, and receipts. Phase 1
remains 10/17 with 126 exact route receipts outstanding.

Latest P11/P14 shared callback guard (25 October): the universal session now
rejects same-thread reentry across generate, stream, snapshot, restore, release,
and close callbacks while preserving serialized cross-thread access. Tests
cover owner callbacks, publication callbacks, a paused stream consumer, and
recursive snapshot/close owners. The universal-engine suite passes 28/28; this
closes only the shared boundary, not family ABI callback maps or caller-wide
serving lifecycle closure.

Latest P14 stream cancellation-thread teardown fix (25 October): the universal
stream wrapper no longer holds its thread-owned reentrant lock across consumer
yields. The active-request marker blocks lifecycle mutation, and a
thread-neutral request gate serializes later callers until completion or
rollback; cleanup reacquires the state lock. Closing a paused stream from
another thread now rolls back cleanly without deadlocking or
leaving the session quarantined. Backend iterator `close()` failures during
abandonment are now surfaced after rollback rather than swallowed, while any
original backend exception remains primary. `tests.test_universal_engine`
passes 31/31, including cross-thread close, cleanup-error, and concurrent
request serialization regressions. This
closes shared teardown boundaries only; caller-specific lifecycle coverage and
P14/P15 remain open.

Latest P14 text-attachment cancellation (25 October): `/api/chat/stream` no
longer enters a blocking, uncancellable `complete()` for text-only attachments.
The cancellation callback now travels through `ChatService` and `ModelGateway`
to the native publisher; cancellation is not converted into fallback work, and
the turn is not persisted as complete. Focused chat/gateway regressions pass
28/28. This closes one more caller seam only; generic synchronous providers
may not support cooperative interruption, and P14/P15 remain open.

Latest P05 grammar/speculation correction (25 October): grammar mask probes and
draft tokens now use isolated FSM state; unmapped vocabulary IDs remain masked,
and an empty valid set fails closed. Speculative generation now propagates
cancellation, preserves only published stream grammar state, stops the accepted
prefix at EOS, and samples the target bonus token when every draft token is
accepted and the output budget permits. The combined grammar/chat/gateway
regression group passes 55/55; the 19 dedicated speculative regressions include
full draft-distribution residual correction under an explicit memory budget,
separate seeded draft/target RNGs, and target-RNG rollback on cancellation.
The decoder now serializes full generation/stream requests and grammar updates,
so simultaneous callers cannot fork and overwrite its persistent FSM/RNG state;
a regression confirms grammar changes wait until an active stream closes.
Also fixed Phase 1G draft installation: it wrote `_draft_fn` while the decoder
uses `draft_fn`, making the trained draft model inert. Phase 1D/1G now reject
requested speculative mode when the target verifier/decoder is absent or the
draft vocabulary differs, instead of silently falling back. Four focused
adapter-wiring regressions cover the path.
Free decoding now validates target and draft logit rows against the configured
vocabulary even when grammar masking is disabled, rejecting malformed or
non-finite logits instead of sampling IDs outside the canonical vocabulary.
Removed an acceptance-probability floor that rejected all low-probability
draft tokens and biased speculative output; acceptance now follows the exact
target/draft ratio, with a regression at a 1.1% threshold.
Unknown grammar modes and malformed regex patterns now raise instead of
silently switching to unconstrained decoding.
Phase 1G now forwards cooperative cancellation through prefix and small-model
draft adapters and never swallows cancellation exceptions from the draft
backend; two adapter regressions cover forwarding and propagation.
P14 adapter lifecycle now holds an operation lease through complete generation
and stream iteration; Phase 1D/1E/1G close waits for active work to relinquish
the nested owner stack. A close-vs-active-stream regression verifies the owner
is not torn down until the stream is closed.
This fixes the optional Phase 1D decoder path; it does not establish production
caller wiring, arbitrary target-model physical-state rollback, RNG receipt
coverage, or full P05/P15 closure.

Latest P05 request-boundary correction (25 October): the canonical grammar
decoder now rejects invalid context token IDs, negative/non-integer generation
budgets, and out-of-vocabulary stop IDs before changing its persistent FSM/RNG
state. The same checks apply to streaming, including zero-token requests. The
focused grammar and Phase 1D wiring suites pass 32/32 under the project venv.
This closes malformed-request admission for that adapter only; physical target
state transactions and caller-wide speculative reconciliation remain open.

Latest P14 Phase 2 batch-session lifecycle correction (25 October):
`Phase2SessionHandle` now bounds stream callback buffering by the admitted
output limit, surfaces backend stream errors, cancels live work when a consumer
abandons a stream, and cancels generation if the caller's cancellation
predicate raises. Synchronous generation polls that predicate on its caller
thread instead of creating a background watcher, and cancellation errors remain
primary if request cleanup also fails. It validates generation limits before
counting a request.
`ContinuousBatchScheduler.submit()` now signals queue-full failure before the
error callback and runs that callback outside its lock; committed token/done
callbacks also run outside the scheduler lock to prevent reentrant deadlocks.
Focused Phase 2, OLMoE continuous and OLMoE service tests pass 37/37; scoped
Ruff and diff checks pass. This closes only this optional batched-session caller;
other serving routes remain open. The canonical verifier was rerun: 10/17,
with 126/171 admitted routes missing exact runtime receipts; P15 remains open.

Latest complete Python verification (25 October 2026):
`.venv\Scripts\python.exe -m unittest discover -s tests` ran **2,181 tests**,
with **97 skipped and zero failures**. The run produced temporary test evidence
files only; they are not route-bound receipts. A fresh `phase1_verify.py` run
passes canonical source validation, 162 exact route bindings, 178 owner
references, and static ABI consistency (98 exports/190 ctypes references), but
remains **10/17** because 126/171 admitted routes have no exact runtime
receipts. P15 remains open.

Latest P05 state-manager correction (25 October): a successful rollback is
now terminal for that owner's lease. Draft/target sampling, acceptance,
commit, and recapture reject until the owner releases, preventing a cancelled
request from advancing the restored RNG state before the next caller. The
manager also refuses authenticated snapshots older than the currently
registered sequence, preserving a newer idle recovery boundary. Its focused
suite passes 7/7. This strengthens the standalone RNG/recovery manager; wiring
it into production speculative callers and physical KV/cache rollback remain
open.

P05 test wiring was also repaired without duplicating implementation: the
two retained grammar/speculative adapter suites now import the canonical
`grammar_speculative_decoder`, `cpu_kernel_adapter`, `aot_kernel_adapter`,
`draft_model_adapter`, and `draft_model` modules instead of deleted duplicate
`phase1d*` modules. The combined grammar, adapter, transaction, cancellation,
state-manager, and distributed-recovery regression set passes 69/69.

Latest P06 recovery-boundary correction (25 October): distributed committed
state now requires a canonical lowercase SHA-256 history digest and cannot
rebind the same token position to a different digest or cache path. This
prevents a same-position divergent state from replacing the recovery receipt
used for shard consensus. Focused shard, recovery and process-loss/launcher
suites pass 19 tests (2 environment skips). Operation-level transport failure,
all bank ownership, and cross-rank restore consensus remain open.

| Family | Source admitted | Candidate | Restricted | Rejected |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek V4.1 | 9 | 0 | 207 | 0 |
| GLM DSA | 9 | 0 | 27 | 18 |
| GLM5 Next | 21 | 0 | 99 | 48 |
| Inkling | 21 | 0 | 99 | 48 |
| Kimi K3 | 10 | 0 | 54 | 32 |
| Mixtral | 18 | 0 | 108 | 36 |
| OLMoE | 45 | 0 | 45 | 12 |
| Qwen3.5 | 17 | 0 | 115 | 44 |
| Qwen4 | 21 | 0 | 123 | 48 |
| **Total** | **171** | **0** | **877** | **286** |

“Source admitted” means the exact tuple has an explicit source-review binding
in the catalog; it is not an independently executed end-to-end proof. The
catalog explicitly marks `executed=false`, `numerical_parity_verified=false`
and `platform_certified=false`; it does not certify 171 runtime routes.
There is no defensible overall completion percentage because P02–P15 have no
agreed weights. The earlier 37% was an estimate, not a measured gate.

## Evening implementation pass (20 October)

### Latest F16-source route update (24 October)

Added `openagent/adaptive_engine/checkpoint_dtype_conversion.py`, a bounded,
staged BF16→FP16 SafeTensors converter. It validates contiguous tensor ranges,
preserves non-BF16 payloads and inventory, rejects non-finite or FP16-overflowing
values, checks that source shards remain unchanged, rejects non-sibling output
destinations, and publishes only a complete sibling directory. Its focused
tests pass (4/4). Converted all 3,219
BF16 tensors in the local OLMoE checkpoint (13,838,718,736 SafeTensors bytes),
packed the derived checkpoint as `int8-row`, and captured a route-bound native
CPU receipt: one generated token, cache/routing and restore/replay observations,
and an independent NumPy comparison with maximum logit error
`2.9325485224609375e-05` at tolerance `1e-3`. The OLMoE CLI also had an
advertised `--dtype float32` option with no default container/manifest mapping,
causing `KeyError('float32')`; added the exact `.f32` defaults and a regression
test. Packed the converted checkpoint's 25,769,803,776-byte FP32 expert bank
and captured its exact F16-source/native receipt with max logit error
`2.8014183044433594e-05` (tolerance `1e-3`). These exercise two admitted
F16-source routes. Then packed `q8_0` (6,845,104,128 bytes) and captured its
no-fallback native route with max logit error `2.7179718017578125e-05`
(tolerance `1e-3`). Packed `q4_0` (3,623,878,656 bytes) and captured its
no-fallback native route with max logit error `3.24249267578125e-05`
(tolerance `1e-3`). These four F16-source routes use tensors derived from local
BF16 weights, so they do not establish independent upstream-F16 provenance or
cover other F16 routes. At that historical checkpoint the verifier was 10/17
with 18/171 receipts; see the current result above for later captures.

The verifier now also reports the exact missing-receipt count and first
canonical route on A09 (binding) as well as A04-A08/A10; previously A09 failed
without an actionable reason. `tests.test_phase1_closure_gate` and
`tests.test_phase1_capture` pass **28/28** with this diagnostic change. The
latest source matrix is still 171 admitted, zero candidates; this improves
failure reporting and does not close any additional route.

Uncommitted working-tree changes; no commit or push. Proof:
`python test_p0_p1_fixes.py` now runs the closure-gate, capture and
manifest-selection suites as well (see the verification list below).

- **P15 evidence gate is now kind-specific.** `phase1_observations.py` parses
  each observation kind and derives pass/fail from measurements: tokens must be
  the greedy argmax of the observed logits; the gate itself recomputes the
  maximum absolute logit error against an independent reference (which must be
  a different implementation, with identical tokens and a declared tolerance no
  larger than `1e-2`); cache position must equal the decoded positions; routing
  must hold one distinct top-k row per processed position and layer within the
  pinned config's expert range; recovery replay must reproduce the
  continuation at the snapshot position; placement must match the route backend
  with zero host-fallback operations. One file can no longer serve several
  kinds, every observation is bound to the route `case_id` and the pinned
  config hash, and claimed booleans are ignored. `not-applicable` is accepted
  only where `coverage_applicability(row)` derives it from the canonical row
  (media: family contract + non-text identity + implemented media cell;
  speculation: `mtp`/`speculative` head; distributed: backend). The old
  positive fixture (one generic file for all seven kinds) is now a rejection
  test, and 10 forgery tests re-pin hashes so only the measured content can
  reject them.
- **Capture runner exists for native OLMoE routes.** `phase1_capture.py` opens
  the exact route through the production universal path, traces the runtime's
  real expert selections, records per-step FP32 logits, the physical cache,
  a snapshot/restore replay and placement, runs the NumPy `OlmoeModel` as the
  independent reference, pins config/weights/manifest/expert payloads/runtime
  library, verifies the receipt with the same gate and only then writes it to
  `phase1_runtime_receipts.json`. On a compiled tiny checkpoint the gate passes
  A04-A10 for that route; eighteen local OLMoE routes now have real receipts,
  while the canonical verifier remains 10/17 because 153 admitted routes lack
  receipts. A separate explicit no-fallback Numba packed-INT4 path is now
  captured and compared with the NumPy reference; it is not routed through the
  leased C segment executor.
- **Local OLMoE opening contract resolved.** A checkpoint may hold several
  packed manifests; the universal path selects exactly one: the explicit
  `manifest=` opening option (the resolver now receives opening options), or
  the unique manifest matching an explicit route's persisted format. The
  default fails closed and lists every manifest and format. Tested on an
  I4/I8/Q4/Q8 checkpoint: I8/Q4/Q8 each open natively and match the NumPy
  model; I4 (`int4-row`) selects the explicit packed-INT4 consumer and is
  captured above; no other route or backend is implied.
- **Dense `qwen3_5` decided: out of Phase 1 scope.** It has no routed experts.
  `source_scope.PHASE1_OUT_OF_SCOPE_IDENTITIES` records the reason, and every
  loader, CLI, factory, resolver and universal open now reports it instead of
  a generic "absent" error; it is never aliased to `qwen3_5_moe`. Reversible:
  remove the entry and add a real dense loader/consumer/route inventory.
- **Stream cleanup hang fixed.** After a consumer abandons a stream, the
  worker gets a bounded grace (`DEFAULT_STREAM_JOIN_SECONDS`). A worker that
  ignores cancellation quarantines its owner: rollback is refused rather than
  racing it, the universal session is held in `closing`, and `close()` raises
  (retryably) until the worker has exited. Cooperative abandonment still rolls
  back normally.

- **First real-checkpoint capture attempt (local OLMoE Q8).** MSVC built the
  native executor, the route was admitted and the leased engine loaded; the
  edge then failed with `source matrix decoding exceeds admission`: decoding
  the 50,304 x 2,048 embedding is charged 12 bytes/element (~1.24 GB) against a
  fixed 1 GiB default. Fixed without widening the default: `OlmoeModel` takes
  `max_matrix_decode_bytes`, `NativeOlmoeEdge` applies its explicit
  `max_conversion_bytes` to decoding, the universal OLMoE owner accepts
  `conversion_mib`, and `phase1_capture --conversion-mib` (default 2048) passes
  it to both the native edge and the NumPy reference. A regression test checks
  that the capture budget reaches the native engine. Re-run pending.
- **Manifest ownership is proved, not assumed.** An explicit or uniquely
  selected manifest must be a direct file of the checkpoint (symlinks resolved)
  *and* bind that checkpoint's architecture and metadata fingerprint, so a
  manifest packed for another checkpoint and copied in is rejected before any
  owner is constructed.
- **Comparison reference provenance is source-closed for six families.** The
  gate pins independent numerical decoders and their local equation helpers
  for OLMoE, Mixtral, Qwen3.5, Qwen4, Inkling, and GLM-DSA. GLM5-Next, Kimi K3,
  and DeepSeek V4.1 remain ineligible because no separate numerical reference
  is registered. This is reference eligibility only: the capture runner still
  executes OLMoE only, and the gate cannot prove a reference was actually run.
  Device placement is likewise not instrumented. The seven P13 checks still
  fail for 126 missing route receipts.
- **Exact receipt route binding tightened (23 October).** The receipt gate now
  compares the six-field route inside each receipt to its canonical source row
  and refuses a receipt for a candidate or restricted row. A regression test
  covers both mismatches; the capture/closure-gate tests pass 20/20. This is
  an integrity check, not a new runtime receipt or route admission.

- **Direct OLMoE FP32 expert route and capture (23 October).** The admitted
  `float32` persisted route previously failed at `OlmoeModel` validation and
  had no packer or CLI output path. The packer now preserves the FP32 expert
  bytes, the OLMoE loader admits them, and the CLI exposes `--dtype float32`.
  An explicit `NativeExpertBackend` with fallback disabled captured the real
  BF16 checkpoint route. One generated token, physical cache, route trace,
  snapshot/restore replay and independent NumPy comparison passed with maximum
  logit error `3.528594970703125e-05` (tolerance `1e-3`). The checkpoint-local
  expert container is 25,769,803,776 bytes; its manifest and payload are
  retained because the receipt binds them.

- **Direct OLMoE packed-INT4 capture (23 October).** Added an explicit capture
  route for the Numba `NativePackedExpertBackend`, with no kernel fallback,
  physical KV digest, route trace, restore/replay, pinned manifest/weights and
  runtime sources, and independent NumPy comparison. The real local
  BF16-source `int4-row` checkpoint produced one token with maximum absolute
  logit error `2.8371810913085938e-05` (tolerance `1e-3`); the exact receipt
  passed the route evidence gate. `tests.test_phase1_capture` passed 20/20
  after also covering refusal by the legacy synthetic-evidence writer.
  This closes one CPU route only—not GPU placement, native ABI compilation,
  or other-family evidence.

- **Direct OLMoE INT2 capture (23 October).** The vectorized INT2 packer was
  checked byte-for-byte against the prior scalar layout, then produced a
  checkpoint-local 1.63 GB expert container. The explicit
  `NativeLowBitExpertBackend` ran with fallback disabled; generation, physical
  cache, routing, snapshot/restore replay, placement and independent NumPy
  comparison passed. One generated token had maximum absolute logit error
  `2.288818359375e-05` (tolerance `1e-3`). Its exact route receipt passed the
  closure gate. A CPU INT2 forward/recovery/reference pass took about fifteen
  minutes; no other route is implied.

- **Direct OLMoE consumer registry expanded (23 October).** Capture dispatch
  now has exact backend constructors for every OLMoE format with a native
  gateway default, including INT8 and Q4/Q5/Q8 variants; the registry-to-
  gateway mapping regression passes. These constructors are not execution
  evidence: eighteen exact route receipts currently exist, including four
  derived F16-source routes; this registry mapping does not generalize evidence.

- **Direct OLMoE Q8_1 capture (24 October).** Replaced the Q8_0 intermediate
  and scalar record conversion with a bounded direct Q8_1 packer; byte-layout
  parity passed across block sizes and boundary values. The real checkpoint
  produced a 7,247,757,312-byte container in 148 seconds. The no-fallback
  `NativeQ8_1ExpertBackend` route completed exact generation, physical cache,
  routing, snapshot/restore replay, CPU placement, and independent NumPy
  comparison; maximum logit error was `2.5272369384765625e-05` (tolerance
  `1e-3`). Its exact route receipt passed the canonical evidence gate.

- **Direct OLMoE INT3-grouped capture (23 October).** The exact 64-value-group
  format was packed into a separate 2.82 GB checkpoint-local container and
  executed with `NativeLowBitExpertBackend` and no fallback. Generation,
  physical-cache observation, route trace, snapshot/restore replay, placement
  and independent NumPy comparison all passed. One token had maximum absolute
  logit error `3.337860107421875e-05` (tolerance `1e-3`). The exact receipt
  passed the closure gate. This is one CPU route, not evidence for INT2 or
  another backend.

- **Direct OLMoE Q4_1 capture and packer optimization (23 October).** Replaced
  the per-block Python packing loop with bounded vectorized extrema, FP16
  metadata and nibble coding. A scalar-layout parity test and the Q4_1 expert,
  native-kernel and OLMoE differential suites pass. Real checkpoint packing
  completed in 147 seconds and produced a 4.03 GB container. The exact
  no-fallback `NativeQ4_1ExpertBackend` route passed cache/routing, recovery,
  placement and independent NumPy comparison with maximum logit error
  `3.24249267578125e-05` (tolerance `1e-3`). Receipt is pinned to the Q4_1
  manifest/payload and passed the route evidence gate.

- **Direct OLMoE Q5_0 capture and packer optimization (23 October).** Replaced
  per-block Python quantization/bit packing with bounded NumPy scale, 5th-bit
  plane and nibble generation. The new regression checks exact legacy bytes;
  Q5_0 format, native-kernel and OLMoE integration suites pass. Real BF16
  checkpoint packing completed in 176 seconds (4.43 GB payload). The explicit
  `NativeQ5_0ExpertBackend` route passed exact generation, cache/routing,
  recovery replay, CPU placement and NumPy comparison with max logit error
  `2.6702880859375e-05` (tolerance `1e-3`); the route receipt passed the gate.

- **Direct OLMoE Q5_1 capture and packer optimization (23 October).** Replaced
  per-record scalar packing with bounded vectorized FP16 metadata, fifth-bit
  plane and nibble generation; an exact legacy-byte parity test and the Q5_1
  expert/native/OLMoE integration suites pass. Real BF16 checkpoint packing
  completed in 174 seconds (4.83 GB payload). The no-fallback
  `NativeQ5_1ExpertBackend` route passed generation, cache/routing, restore and
  replay, CPU placement, and independent NumPy comparison with max logit error
  `3.0517578125e-05` (tolerance `1e-3`); its exact receipt passed the gate.

- **Q8_1 bounded packer completion (23 October).** Replaced the Q8_0 intermediate
  payload plus per-record Python repacking loop with direct bounded Q8_1 block
  construction. Scale/code rounding remains byte-identical to the existing
  Q8_0 policy, and the binary16 sum field is calculated from those same codes.
  Added a regression comparing output byte-for-byte against the legacy
  Q8_0-then-Q8_1 conversion for multiple block sizes, zeros and subnormal-scale
  values. Focused Q8_1/Q5_0/Q5_1 suites pass **23/23**. Full-checkpoint Q8_1
  packing completed (7,247,757,312 bytes) in 148 seconds. Exact no-fallback
  `NativeQ8_1ExpertBackend` inference, cache/routing, recovery replay, CPU
  placement and independent NumPy comparison passed; maximum logit error was
  `2.5272369384765625e-05` (tolerance `1e-3`). Its receipt is accepted by the
  canonical route gate.

- **Grouped INT4 bounded packer completion (24 October).** Replaced the scalar
  per-weight quantization/nibble loop with bounded row tiles, vectorized group
  scales/codes and vector nibble packing. A regression retains the prior scalar
  policy as a byte-for-byte oracle across odd group boundaries, subnormal and
  zero blocks, several tile sizes and partial groups. Grouped-format/native
  suites pass **19/19**. Full BF16-source checkpoint packing completed in
  114 seconds (3,623,878,656 bytes). The exact native no-fallback capture
  passed generation, physical cache, routing, recovery replay, CPU placement
  and independent NumPy comparison; maximum logit error was
  `2.4557113647460938e-05` (tolerance `1e-3`). Its receipt passed the canonical
  gate.

- **MXFP4 bounded packer implementation (24 October).** Replaced per-row,
  per-32-value Python packing with bounded row tiles and vectorized E2M1 nearest
  level selection, UE8M0 power-of-two scales and nibble output. Added byte-for-
  byte comparison to the former scalar policy across partial groups, ties,
  signed values, zeros and exponent boundaries; MXFP4/OLMoE integration tests
  passed **9/9**. Full-checkpoint packing completed in 446 seconds (3,422,552,064
  bytes). Exact native no-fallback generation, cache/routing, recovery replay,
  CPU placement and independent NumPy comparison passed with maximum logit
  error `2.574920654296875e-05` (tolerance `1e-3`); its receipt passed the
  canonical gate.

- **Block-FP8 OLMoE checkpoint pack (24 October).** The existing bounded
  128x128 E4M3/F32-block converter completed the local BF16 checkpoint in 410
  seconds, producing a 6,446,642,688-byte container. Exact native no-fallback
  route capture initially exposed missing default FP8 reference registration.
  Added bounded NumPy E4M3 block decode/GEMV and registered it in the default
  expert-kernel registry. Format/native/OLMoE suites pass **15 tests (1
  optional skip)**. The exact route passed cache/routing, recovery replay, CPU
  placement and independent comparison at `2.586841583251953e-05` maximum logit
  error (tolerance `1e-3`); the canonical route gate accepted its receipt.

- **P02 public-wrapper and Kimi Linear source routes (23 October).** Eight
  GLM5/Qwen4 public-range wrapper tuples and two standalone native Kimi Linear
  tuples now have exact source bindings. Kimi Linear is wired to the native
  family CLI through a bounded `NativeExecutionService`; its service adapter
  binds decoder cache snapshots to route identity and exact token history.
  A focused end-to-end regression exercises generation, restore and replay.
  Testing caught and fixed a physical-cache shape mismatch and a service
  snapshot incompatibility. Validation: Kimi decoder suite 23 tests (one
  official-vendor oracle skipped), P02 triage 20, universal production path
  12, universal/stream/lifecycle group 55, distributed fault suites 16, and
  ABI/media cleanup tests 4; all passed at that checkpoint. **Superseded
  23-Oct implementation update:** the direct OLMoE gateway now binds its
  actual `OlmoeModel`/expert-runtime formats through inventory admission;
  float16 manifests are restricted. K3 native/public routes are bound to
  exact source/persisted formats in the payload gate. The three device/BF16
  combinations were then restricted before config or payload reads because
  their hybrid continuation/recovery owner is incomplete. The matrix is now
  171 admitted, zero candidates, 877 restricted and 286 rejected; the
  verifier is 10/17. A static ABI consistency gate checks 98 exports and 166
  ctypes names. Fourteen local BF16-source receipts are recorded for I8, Q4_0,
  Q8_0, Q4_1, Q5_0, Q5_1, Q8_1, `int4-grouped-g64`, MXFP4, FP8, `int4-row`,
  `float32`, `int2-row` and
  `int3-grouped-g64`; no other
  route is certified.

Capture real OLMoE evidence on this machine (Windows needs MSVC for the
native library; each admitted manifest format is a separate route):

```
python -m openagent.adaptive_engine.phase1_capture olmoe ^
  --checkpoint models\olmoe-1b-7b-instruct ^
  --manifest models\olmoe-1b-7b-instruct\<Q8 manifest>.json ^
  --build-library build\native --prompt-ids <ids> --max-tokens 4
python phase1_verify.py
```

## Audit defects and evidence limits found on 20 October

- **P0 — Closure gate repaired at the fail-closed boundary, but evidence
  generation is still missing.** The old validator had an unset source-scope
  dependency, accepted candidates in A01, passed A04/A10 unconditionally and
  counted method presence as A05–A09. The current implementation checks the
  canonical route decisions and requires one pinned exact-route runtime
  receipt per admitted row, with separate coverage evidence. The historical
  20 October result was 0/17; as of 24 October fourteen real local OLMoE receipts
  exist. Receipt schema/integrity checks alone do not execute a model; exact
  runtime capture and comparison remain required for the other admitted routes.
- **P0 — Synthetic evidence can be mistaken for observations.** The nine
  `a10_evidence/*.json` manifests carry `backend=test_config`, placeholder
  hashes/paths, no recovery receipts and `full_generation_success=false`;
  some set `parity_verified=true` despite that. `A09_evidence_manifest.json`
  currently reports `total_bindings=0` and `bindings=[]`. The in-memory
  `SourceScopeConverter` previously set `receipt_present=True` for source-only
  families without an external receipt; that promotion path is now disabled.
  The new verifier still checks file hashes and asserted booleans, not the
  semantics of the observations: its positive unit fixture reuses one small
  synthetic file for all seven required kinds and passes A10 in isolation.
  `not-applicable` coverage requires only free-text rationale and is not
  checked against the family/route applicability matrix. Do not treat a future
  17/17 as runtime proof until these checks are bound to independently
  captured, kind-specific observations and reviewed applicability decisions.
- **P1 — Local checkpoint readiness is narrower than file presence.**
  `models/olmoe-1b-7b-instruct` has four valid sparse-expert manifests (I4,
  I8, Q4 and Q8). `resolve_checkpoint_route()` raises `exactly one sparse-
  expert manifest is required ... found 4`; the production universal owner
  cannot open this directory without an explicit, admission-bound manifest
  selection or an isolated single-manifest view. `models/qwen3.5-0.8b`
  declares `model_type=qwen3_5`; the resolver rejects it because that dense
  identity is absent from the P02 catalog. Neither model currently supplies
  a real P13 receipt for an admitted route.
- **P1 — Two conflicting live-looking TODOs.** The tracked
  `openagent/adaptive_engine/phase_1_audit/PHASE_1_ACTIVE_TODO.md` says 133
  admitted and 1,201 candidates, yet also claims zero candidates and 414
  admitted in its A01 row. Both conflict with canonical 129/689 counts. The
  dated root status report says Phase 1 is 100% functional. Treat these as
  history, not an active gate; this root file is the current queue.
- **P1 — Green general suite is narrower than certification.** The latest full
  rerun (24 October) passed all 1,888 tests with 95 skips. One preceding run
  had a transient Windows temp-file cleanup error; the isolated process test
  passed ten consecutive repetitions and the full rerun passed. Skip sites
  include optional Torch and
  Transformers oracles, native/compiler availability, Metal/CUDA/HIP and
  selected hardware certification. Classify the actual skip reasons and run
  required certification jobs on matching hosts before marking those routes
  executed, numerically compared or platform certified.
- **P1 — Stream cleanup can hang.** *Fixed in the evening pass:* bounded
  grace, owner quarantine, rollback refused while the worker runs, retryable
  close (`tests/test_universal_adapters.py::StreamQuarantineTests`).

## Done and verified in this tree

- [x] Restored the reviewed route restrictions, removed the unsupported K3
  Metal admission, recovered the V4.1 distributed process loss restriction,
  and made the canonical source validator pass.
- [x] Registered exact openers and route resolvers for the 12 checkpoint model
  types that have source admitted rows. Universal open checks the exact route
  before owner construction. Unknown and candidate routes fail closed.
- [x] Replaced the dummy Phase 1A route with family owners. A compiled tiny
  native OLMoE checkpoint generates tokens matching an independent NumPy model
  in the production universal path. Publisher family adapter wiring uses real
  checkpoint inventory and operator parsing, with the C sessions mocked in
  those tests; full publisher execution is not proven there.
- [x] Isolated Phase 1B learned state by checkpoint/config/route identity under
  the runtime data directory; it no longer writes into checkpoint folders.
  Metrics read actual expert runtime/store counters and use `None` when an
  operation has not been observed.
- [x] Added transactional universal request snapshots, cancellation rollback,
  receipt identity/freshness checks, canonical warm option keys, and retryable
  owner cleanup. Focused fault tests and the native OLMoE path pass.
- [x] Restored distributed checkpoint preflight, rank receipt and peer result
  checks. These are source and focused test improvements; process loss remains
  restricted and has not been exercised with a real multi-process group.
- [x] Removed the seven explicitly superseded files. Five tracked deletions are
  staged; two untracked files (`phase1b_adapter_registry.py` and the old
  `PHASE_1_P0_P1_FIXES_COMPLETE.md`) were deleted. No live Python import under
  `openagent` or `tests` names them. The old report claimed completion before
  the replacement implementation and is not valid evidence.
- [x] Disabled the two legacy candidate-conversion helpers; they no longer
  promote rows or synthesize `receipt_present`. Replaced the P15 method-
  presence checks with canonical route and pinned-receipt predicates, and
  added adversarial tests for candidates, missing receipts and placeholders.
  The historical integration guide now warns against candidate conversion.

Verification run on this tree:

- `test_p0_p1_fixes.py`: **208 run, 205 passed, 3 skipped** after the
  OLMoE manifest, receipt-gate, foreign-manifest fingerprint, reference-source
  and capture decode-budget regressions; this supersedes the earlier 145- and
  176- and 179-test focused counts.
- `test_phase1b.py`: **11 passed**; native OLMoE path compiled on this host.
- `test_phase1b_unit.py`: **28 passed**.
- New universal adapter/engine/production/lifecycle modules: **49 passed**.
- Reinstated the four historical GitHub workflows removed in commit `8088b25`;
  declared `GET /health` as a private production-read route; aligned the
  Anthropic tools error with the existing compatibility contract; updated
  production proof to recognize the current owned Playwright runner.
  The affected focused run now passes **36/36**. These are local source/test
  results, not a completed GitHub CI or release execution.
- `phase1_verify.py`: validator pass; the historical 20 October result was
  **5/17** after the original P02 triage. Superseded by the 23 October
  **7/17** result above; Phase 1 remains open.
- Latest 23 October focused verification: `tests.test_source_scope_inventory`
  **61/61**, `tests.test_p02_candidate_triage` **26/26**,
  `tests.test_phase1_capture` **17/17**, the OLMoE manifest/universal and
  public ABI suites **50/50**, the distributed tensor/launcher, inference,
  media-ordering, public-text and Phase 1B lifecycle group **39/39**, and
  speculative/media lifecycle group **34/34**. A combined OLMoE/Kimi/
  universal/ABI run had five stale expectation failures; after updating those
  expectations to the newly admitted/restricted routes, the rerun passed.
  These are focused tests, not full-suite or hardware certification.
- Prior canonical verifier snapshot after additional F16-source/int8-row,
  F16-source/float32, F16-source/q8_0 and F16-source/q4_0 captures:
  **10/17**, source validator PASS, zero candidates, 171 source-admitted
  routes. At that point the receipt manifest contained eighteen verified
  exact-route runtime receipts; later native/public-range captures supersede
  that count. This is not full P13 evidence and Phase 1 remains OPEN.
- Latest 24 October focused regression group after bounded Q8_1, grouped-INT4,
  MXFP4 and FP8 reference-path changes: `test_phase1_capture`,
  `test_phase1_closure_gate`, `test_source_scope_inventory`, `test_olmoe_batch`
  and `test_olmoe_manifest_selection` passed **102/102**. The FP8 format,
  native and OLMoE suites separately passed **15 tests (1 optional skip)**.
- Latest 24 October combined converter/CLI/capture/closure-gate regression
  group passed **41/41** (`test_checkpoint_dtype_conversion`,
  `test_expert_cli_kernels`, `test_phase1_capture`,
  `test_phase1_closure_gate`) after enforcing sibling-only converter output,
  adding the missing float32 CLI manifest default and per-advertised-format
  path coverage, and improving A09 gap diagnostics. The latest rerun passed
  **42/42** after adding the all-advertised-dtypes contract.
- Earlier 24 October local OLMoE capture sweep: exact BF16-source/native receipts
  cover all fourteen admitted persisted expert formats. Every route used one
  prompt position, one generated token, restore/replay, physical-cache and
  routing observations, CPU placement and independent NumPy comparison. This
  closed only the local BF16 native format sweep. Later exact receipts cover
  all three public-range OLMoE rows and some F16/F32 native rows; current gaps
  are recorded at the top of this file.
- `openagent.adaptive_engine.scalable_execution.test_contracts`: **11/11**;
  fixed Windows SQLite snapshot `fsync` by flushing a writable staged file,
  clarified escaped-buffer cleanup diagnostics, and corrected the scratch and
  cleanup-retry fixtures to obey the planner contract.
- Superseded full `tests/` discovery after the chat-provider change: **1,844
  run, 95 skipped**, with one intermittent Windows HTTP SSE socket-abort error.
  The latest full run below passed and supersedes that transient result.
- P02 triage, 20 October: `test_p0_p1_fixes.py` now runs the new
  `tests/test_p02_candidate_triage.py` too: **196 run, OK, 3 skipped**. A full
  `unittest discover` in the Linux sandbox produced the identical
  failure/error set with and without the change (all of them sandbox-only:
  missing `argon2`/`tkinter` and platform files). The run grew from 1,616 to
  1,633 tests.
- Full suite rerun after checkpoint-local manifest admission and receipt
  metadata binding (on your machine): **1,797 run in 682.210 seconds; OK, 95
  skipped**. That run predates the P02 triage and does not provide real route
  receipts.

## Remaining work, in dependency order

### 1. P02/P03 — decide every route and inventory consumer

- [x] Triage the 689 candidate tuples (initial triage 20 October). That
  historical pass left 47 candidates; later source reviews have resolved them.
  The current canonical matrix has zero `candidate-unverified` tuples. Each
  restriction comes from `source_scope._tuple_restriction`, flows through the existing
  contract-restriction path and fails closed with its reason.
  `tests/test_p02_candidate_triage.py` drives the rejecting owner itself, with
  the P02 gate patched out where it would otherwise fire first. The validator
  now rejects a candidate that hides a restriction.
  - **OLMoE public-range (42):** `NativeOlmoePublicAdapters` rejects every
    non-int8-row manifest layer.
  - **Publisher packed formats (482):** covers Mixtral, Qwen3.5, Qwen4, GLM5,
    Inkling and GLM DSA. `native_family_cli.PublisherSession` rejects every
    `--expert-manifest`, the `execution_factories` packed-provider guards do the
    same, and `native_conditional_cli` has no manifest input.
  - **Text identities with F8_E4M3 source (60):** the text reader header
    parser only accepts F32/F16/BF16.
  - **Wrapper F8_E4M3 → float32 (11):** every loader binds FP8 headers only to
    `fp8-e4m3-b128-f32`.
  - **Qwen3.5 wrapper public-range (18):** `load_publisher_conditional_decoder`
    has no Qwen3.5 Segment path.
  - **`deepseek_v41_text` (9):** a nested text_config identity, never a
    top-level checkpoint.
-  - **`kimi_linear` (14):** 2 standalone native F32/BF16 tuples now have a
    guarded budgeted service and CLI path. The other 12 are concretely
    restricted: device/public-range and MXFP4 variants lack a matching loader.
  - **New source admissions (10, 23 October):** 8 GLM5/Qwen4 wrapper
    public-range F32/F16/BF16→float32 tuples bind through their wrapper decoder,
    public-text decoder and CLI; 2 standalone Kimi Linear native F32/BF16
    tuples bind through its service and CLI. Six earlier Qwen3.5/Qwen4 text
    route bindings remain unchanged. Source admission is not parity or
    placement evidence. Explicit public runtimes remain rejected for the
    distinct native-only Qwen text identities.
- [x] Resolve the 21 former OLMoE candidates. The direct gateway's
  `OlmoeModel` path now runs exact checkpoint-inventory and manifest-format
  admission before reading weights. It binds all 14 formats actually consumed by
  that path; unsupported `float16` manifests are explicitly restricted. The
  universal adapter now uses the leased C owner where supported and the direct
  provider's explicitly selected native kernel for the other admitted formats;
  it never silently falls back. The universal boundary verifies every explicit
  route's family/source dtype against the checkpoint resolver before owner
  construction. Regressions cover int2 generation/rollback, BF16-vs-F32 route
  rejection and the leased INT8 route. The
  canonical verifier remains 10/17 with 126 evidence gaps and zero candidates.
  All 45 OLMoE source-admitted tuples now have exact route-bound CPU receipts across
  local BF16, derived F16/F32 and public-range paths. The derived checkpoints
  share the BF16 upstream weights; this does not establish independent F16/F32
  provenance or accelerator placement.
- [x] Resolve the 3 K3 device/BF16 candidates by exact-tuple restriction.
  K3 native and public routes verify source/persisted formats in the payload
  and factory; dense-to-packed and incomplete device-hybrid routes reject
  before config or tensor payload loading. The K3 device hybrid reason names
  the missing FLA continuation/recovery owner and has a pre-load regression.
- [x] Prove every public execution factory reaches a source-route or tensor-
  inventory gate; AST reachability and per-family guard regressions pass in
  `tests.test_source_scope_inventory` (61 tests). This is source/fault-fixture
  coverage, not successful construction of every full-size family checkpoint.
- [x] Close the V4.1 inventory bypasses across three public boundaries: CLI
  `PublisherSession`, `open_publisher_v41_service()`, and direct
  `PublisherCheckpoint.load_v41_execution_payload()` (the documented static
  convenience API delegates to that guarded instance method). They enforce
  exact `deepseek-v41` / `deepseek_v41` / native-or-device /
  `publisher-bf16-fp8-fp4` checkpoint-inventory admission before mapped payload
  loading. V4.1 distributed service execution remains explicitly rejected by
  the peer-loss-consensus restriction. Four targeted AST regressions pass, and the combined
  `tests.test_source_scope_inventory` plus `tests.test_p02_candidate_triage`
  run passes 94/94. This closes these V4.1 boundaries only; family-wide
  optional/shared/MTP/expert/media consumer reconciliation and other family
  callers remain open.
- [ ] Execute each admitted family factory against a representative complete
  checkpoint inventory where artifacts are available. The current `models/`
  tree has the OLMoE source and derived-F16 checkpoints, dense Qwen3.5-0.8B
  (explicitly out of scope), and dense OpenVINO Qwen2.5; no standalone Kimi
  Linear or other in-scope family checkpoint was present on this recheck.
  Obtain representative family artifacts before claiming their construction
  or P13 observations. Reconcile optional,
  shared, MTP, expert and media tensors on those actual checkpoints. A source
  audit found and fixed one Inkling exception: the recognized `chain_norm.*`
  shared MTP post-normalization tensor now stays in the auxiliary bank from
  header admission through partitioning to `InklingMTP`; inventory regressions
  verify the header and consumer mapping. A second source audit found that
  accepted `model.language_model.*` media aliases were preserved under those
  alias names by component partitioning, although loader validation and exact-
  set media encoders use canonical `model.*` names. Partitioning now canonicalizes
  vision/audio aliases and rejects collisions; two targeted regressions pass.
  The project `.venv` inventory/candidate command now passes 90/90. The global
  Python lacks NumPy; use `.venv` for engine verification. This is not a full-
  checkpoint construction receipt.
- [x] Resolve the local OLMoE multi-manifest opening contract: select one
  checkpoint-local manifest explicitly and bind its persisted format to the exact admitted
  route, or materialize a safe single-manifest checkpoint view. Test all four
  local formats, foreign-manifest rejection and ambiguous/default rejection
  before attempting P13 capture. Follow-up regressions found and fixed an
  explicit-route validation call that omitted the route when resolving among
  several manifests, plus clarified the leased INT4-row rejection with its
  direct-provider alternative. The two failing cases pass now.
- [x] Decide separately whether dense `qwen3_5` is in Phase 1 scope. If yes,
  add its own loader, execution consumer and exact route inventory; do not
  alias it to `qwen3_5_moe`. If no, retain explicit admission rejection and
  do not use the 0.8B checkpoint as MoE evidence. *Decided: out of scope,
  recorded in `PHASE1_OUT_OF_SCOPE_IDENTITIES` with its reason.*
- [ ] Rerun the canonical validator and `phase1_verify.py` after each route
  group. Done when the matrix has zero candidates and all family route checks
  plus A01 pass without mutating rows in the verifier. *Current: 10/17 overall;
  all nine family route checks and A01 pass, with zero source candidates.*

### 2. P04/P05/P07/P14 — media, speculation and serving callers

- [ ] Exercise image/audio/video item order, placeholder expansion, geometry,
  position IDs, temporal state, borrowed payload lifetime, failure cleanup and
  continuation for each admitted media family. Keep unsupported cases rejected.
- [ ] Complete K3 FLA/native continuation, temporal media and batch rollback
  with end-to-end caller tests. Ten K3 tuples now have exact native/public
  source admission; three incomplete device-hybrid tuples are concretely
  rejected pending an FLA continuation owner.
- [ ] Check speculative accept/reject, bonus token, EOS, context limit,
  cancellation, restore, target/draft physical state and both RNGs through each
  serving caller, including publication failure. Fixed a V4.1 stop-boundary
  defect found during caller review: prompt suffix tokens no longer complete
  generation stop prefixes in plain or DSpark decoding, and DSpark keeps the
  original prompt boundary across speculative steps and resumed checkpoints.
  Regressions cover false prompt-boundary matches and a valid prefix spanning
  transactions and preserved-offset continuation; speculative cancellation,
  boundary, stop and V4.1 cleanup suites pass 36/36. Other serving callers and
  speculative publication/recovery closure remain open. Fresh and resumed
  V4.1 now preserve committed tokens when snapshot release fails, expose
  cleanup/checkpoint-pending status, retry transient release, and retain the
  quarantined owner; speculative transaction plus stop/recovery tests pass
  37/37. V4.1 resumed generation stages its response and resets recovered
  owners before return; the focused resume lifecycle regression passes.
- [ ] Cover queue/backpressure, stop sequences, disconnect, warm reuse,
  streaming error propagation, auth and cleanup for every admitted service
  caller. The universal stream bridge bounds abandonment cleanup and
  quarantines an uncooperative worker; OpenAI/Anthropic SSE publication now
  checks cancellation at the handoff boundary. OpenAI `stop` and Anthropic
  `stop_sequences` now encode through the active checkpoint tokenizer and
  reach the existing token-prefix stop path; the matched stop text is withheld
  from both response modes and matching begins at generated output, not across
  the prompt/cache boundary. Anthropic now also reports `stop_reason=stop_sequence`
  and the exact matched sequence in JSON and SSE terminal events; a provider
  regression verifies this metadata survives complete and streamed execution.
  V4.1 plain and DSpark generation/recovery now likewise match stop prefixes
  against generated output only, with the prompt origin preserved through a
  resumed checkpoint.
  Fixed the stream worker lifecycle guard so its
  owning worker may enter execution while other callers remain blocked. The
  focused stop/route/stream/recovery regressions pass 37/37; the expanded
  native/checkpoint/distributed run passes 82 tests with 8 platform skips.
  `ChatService.send` and lazy `stream` now resolve the production provider
  before persisting a user message; chat/provider doubles were updated to the
  same interface and the chat research suite passes 10/10.
  Queue admission has a provider-owned finite FIFO scheduler, request timeouts,
  overload and deadline 429 responses with Retry-After, wait metadata, health
  counters and Prometheus metrics. `tests.test_inference_routes` and
  `tests.test_admission_scheduler` now exercise timeout mapping and queue
  accounting (25/25 combined). This is route/scheduler regression evidence,
  not live socket saturation. A loopback HTTP integration test now closes the
  peer response after its first SSE frame and verifies that the disconnect
  watcher reaches the inference cancellation signal. That signal's
  thread-safe event is now observed directly by the publisher worker even
  while its consumer is suspended at a yielded chunk; arbitrary user
  callbacks remain consumer-thread-only. Server, API-route and publisher
  cancellation suites pass 36/36. This covers the transport-to-worker signal
  seams with fixtures, not every family-native cancellation path or
  warm-session reuse across callers.
  Latest direct-caller review additionally fixed `ServingEngine` session close
  on generate/stream success and failure, accurate stream error terminal
  chunks, retryable Phase 2/load-balancer/inference-service teardown, and
  store-retention ordering after worker stop timeout. Backend leases span
  session lifetime, active leases are drained before teardown/removal, proxy
  release is idempotent, and a failed session close remains reachable for
  service-level retry. Full-queue rejection refunds global/per-session rate
  tokens, queued cancellation frees scheduler capacity immediately, and
  malformed/incomplete forward results fail the batch rather than stranding a
  caller, an empty prompt advances directly to decode, invalid geometry/input
  is rejected before admission, and zero-token requests finish without model
  work. Shutdown now drains in-flight submissions before worker stop, and
  process-level worker interruption wakes callers instead of stranding them.
  Partial construction failure retires opened stores and preserves retry
  cleanup when worker stop fails. Session close failure retains the handle and
  shared stores for retry. Focused serving/admission/batching tests pass 70/70;
  the 25-case
  Phase 3 smoke harness also passes in UTF-8 mode; scoped Ruff
  passes; canonical verification
  remains 10/17 with the same 126 exact receipt gaps. This closes only the
  identified local direct-serving cleanup seam; full caller reconciliation,
  hardware-bound evidence and P15 remain open.
- [x] Keep the caller's universal-stream cancellation predicate on the
  consumer thread; signal backend workers through an event to avoid concurrent
  or thread-affinity-unsafe callback calls. Regression:
  `test_stream_cancellation_predicate_stays_on_consumer_thread`.
- [x] Fail closed in the disconnected A12/A14/A15 legacy inference scaffolds:
  A12 cannot return zero logits; A14 cannot return `None` logits or fabricate
  empty routing rows; A15 cannot return placeholder text, fabricated
  routing/placement metrics, or `phase1_complete=true`. These classes have no
  production caller in the repository; inference entrypoints now direct
  callers to the production family session and real route trace. Regression
  tests confirm rejection occurs before routing-state mutation. This removes
  false alternate serving paths; it does not close lifecycle requirements for
  the real callers.
- [x] Prove or explicitly reject each Colibri compatibility branch (OpenAI,
  Anthropic, GBNF/tools, READY/mux, cache slots and clients) at its public
  entry point. Generic source presence does not establish protocol parity.
  Corrected the public capability catalog's stale stop-sequence exclusions:
  both OpenAI `stop` and Anthropic `stop_sequences` now accurately advertise
  bounded tokenizer-backed token-prefix handling. The catalog regression
  passes; this is documentation/wire-scope consistency only and does not close
  full OpenAI/Anthropic protocol parity.
  COMPLETED 2026-10-25: `tests/test_colibri_branch_enforcement.py` — 59 tests,
  70 subtests, all passing. Covers all 14 capability branches: brio_scoring,
  jev_systemone, brio_cache_slot, brio_terminal_browser_clients,
  direct_k3_v41_scoring, openai_compatible, anthropic_messages,
  gbnf_constrained_generation, family_tool_protocols, ready_mux_engine_protocol,
  colibri_queue_contract, coli_usage_route_telemetry,
  cache_aware_expert_route_substitution, multi_disk_mirrors. Each
  restricted/not_admitted branch proven rejected at the wire; each
  partial/implemented branch proven admitted through the route dispatcher.

### 3. P06/P12 — distributed state and recovery

- [ ] Map each distributed bank to its actual collective and shard layout.
  Focused code now requires finite process-group timeouts (≤300 seconds),
  bounded async tensor-collective waits, and quarantines timed-out groups;
  rank admission now also requires the same control-group timeout on every
  peer. Distributed fault suites pass 26/26. Object collectives still rely on
  that caller-created group's bounded timeout. Real CPU Gloo tests now kill a
  worker before coordinator admission and during an in-flight generation;
  spawn returns within three times the four-second group timeout (about five to
  six seconds here). This demonstrates bounded launcher failure for these two
  cases (`tests.test_distributed_process_loss` passes 2/2), not distributed
  rollback, consensus after arbitrary transport loss,
  or recovery without a fresh process group. A new injected `load_cache`
  partial-install regression verifies rollback to the prior token history,
  temporary receipt release, poisoned-session teardown, and coordinator
  poisoning; distributed recovery/checkpoint/launcher suites pass 20/20.
  The V4.1 distributed route remains restricted for process loss.
- [ ] Check stale/foreign receipts, request reentry, physical cache ownership
  and cleanup retry across all family owners, beyond the universal boundary
  and focused distributed helper tests. A targeted review of Qwen4 cache
  restore, K3 batch/executor restore, and GLM-DSA/Mixtral/Qwen3.5 owner-bound
  snapshots found staged commits and receipt ownership checks in those paths;
  it found no additional reproducible defect in that slice. The private
  V4.1 mHC equation fixture is not a production recovery owner. This is a
  bounded review only; remaining families/callers and distributed consensus
  are still open.

### 4. P09/P10/P11 — formats, devices and ABI

- [ ] Match each admitted format and geometry to a persisted loader and the
  real execution consumer, including scale rotation, E8/IQ3 and conversion
  cleanup. Verify source restrictions remain enforced at factory and CLI.
- [ ] Capture actual accelerator placement and no host fallback evidence for
  admitted device routes. Registry selection does not prove placement.
- [ ] Static source audit now reconciles 98 exported C declarations and 166
  Python ctypes names with source definitions, rejects declaration/definition
  parameter-count and normalized return/parameter-type mismatches (including
  nested callbacks), and passes `tests.test_public_abi_audit` 5/5. This remains
  source-text validation, not native ABI execution. Compiled tiny-fixture ABI
  suites now pass 48/48 for Kimi KDA, DeepSeek V4.1 CED and Segment/Edge,
  including callback reentry, bounds, continuation, restore and ownership.
  A separate native family/checkpoint slice passes 53 tests with 11 skips;
  optional official model/oracle checks account for unavailable evidence.
  Still map and execute the full admitted symbol/range set and verify family
  tensor/head schemas, callbacks, continuation, ownership/reentry and
  factory/CLI/service wiring. These selected exercises are not full ABI closure.

### 5. P13 — replace test fixtures with route-bound observations

- [x] Implement kind-specific P13 evidence parsing and receipt validation.
  Reject placeholder paths/hashes and `test_config` as certification inputs;
  require independent comparison output, actual placement, cache and recovery
  receipts. Parse kind-specific observation contents, reject reuse of one
  generic file across token/logit/cache/routing/recovery/comparison/placement
  kinds, and derive pass/fail from measurements rather than claimed booleans.
  Validate `not-applicable` against the route's media/speculation/distributed
  requirements. Keep structural schema tests distinct from observation tests.
  A further check verifies that routing observations index every configured
  layer exactly once; a re-pinned fake layer index is rejected. The full
  `tests.test_phase1_capture` module passes 30/30 after also requiring the
  receipt to pin every SafeTensors weight shard under the checkpoint exactly
  once; missing and duplicate shard pins are rejected.
  *Status: the gate and native OLMoE capture runner are implemented and tested;
  the legacy `run_a10_evidence.py` now refuses to create family-level or
  synthetic evidence before probing devices or creating output files, and its
  synthetic-binding helper raises. Existing fixture files remain non-evidence.
  All 45 OLMoE admitted routes now have exact capture receipts. The eight F16
  native routes closed in this update were INT3-grouped, INT4-grouped, Q4_1,
  Q5_0, Q5_1, Q8_1, MXFP4 and FP8. Publisher-family and media/speculative/
  distributed capture runners are not written. The gate also rejects
  checkpoint artifacts outside the pinned checkpoint and mismatches between
  measured outputs and declared cache, comparison or placement metadata.*
- [x] Serialize concurrent route-capture receipt merges across processes so
  independent captures cannot silently overwrite one another. Regression:
  `tests/test_phase1_receipt_manifest_lock.py` (12 concurrent writers).
- [ ] Add publisher-family and media/speculative/distributed capture adapters
  that collect only genuine observations from their production owners. Do not
  synthesize the absent routes or treat structural schema fixtures as evidence.
- [ ] The nine `a10_evidence/*.json` files are schema fixtures: they use
  `backend=test_config`, placeholder hashes and checkpoint paths, zero recovery
  receipts and `full_generation_success=false`. They cannot close P13 even
  where `parity_verified=true` is written. Keep them labelled as fixtures.
- [ ] For each applicable admitted tuple, capture real pinned artifact hashes,
  token/logit/cache/routing observations, recovery receipts, comparison bounds,
  and actual CPU/device placement receipts. Bind evidence to the exact row and
  compare against an independent reference. Do not infer evidence for one
  format/backend from another. The ignored `models/` directory contains a
  complete `olmoe-1b-7b-instruct` checkpoint (three SafeTensors shards) and
  `qwen3.5-0.8b` (one shard). The latter declares `model_type=qwen3_5`, not
  the admitted Qwen3.5-MoE identity, so it cannot certify those matrix rows.
  The local OLMoE files enable a real CPU source/evidence attempt, subject to
  selecting exactly one manifest, runtime memory and an independent reference
  comparison; that first attempt failed before recording a receipt. Later
  local OLMoE native captures succeeded on 23 October: one prompt position,
  one generated token and one recovery replay on CPU; greedy tokens matched,
  cache/routing/recovery observations verified, and maximum absolute logit
  error was `3.0994415283203125e-05` at tolerance `1e-3`. Q8_0 and I8 captures
  also passed, with max errors `1.8596649169921875e-05` and
  `2.5510787963867188e-05`, respectively. A direct Numba packed-INT4 route
  produced max error `2.8371810913085938e-05`; float32 and int2-row produced
  `3.528594970703125e-05` and `2.288818359375e-05`, respectively; INT3 grouped
  produced `3.337860107421875e-05`, Q4_1 `3.24249267578125e-05`, Q5_0
  `2.6702880859375e-05`, and Q5_1 `3.0517578125e-05`. Exact receipts are
  stored in `phase1_runtime_receipts.json`. At that historical point, these
  were fourteen exact tuples only. As of 24 October, the local OLMoE campaign
  is complete (45/45 exact admitted routes); do not generalize those receipts
  across other families or device backends. Other required family checkpoints
  were not found in the repository-local model directory, so their receipts
  remain outstanding.

### 6. Repository checks and P15

- [ ] When preparing a commit, include this currently untracked root TODO,
  the restored untracked `.github/workflows/` files and the new regression
  tests. The working tree has many unrelated pre-existing edits; review the
  exact diff before staging. No commit or push was made in this audit.
- [x] Replace the authoritative `Phase1ValidatorComplete` presence-only checks.
  It now validates canonical tuple decisions and kind-specific, pinned runtime
  observations with measured pass/fail and route-derived `not-applicable`.
  Its current 10/17 result is fail-closed at 126 missing receipts. Family
  capture runners and caller-level runtime assertions remain tracked under P13
  and P14; they are not validator-source defects.
- [x] Retire the synthetic family-level A10 producer and disconnected A12/A14/A15
  fake inference/status path. Focused tests prove they cannot emit fake
  receipts, logits, text, placement, route heat, or Phase 1 success claims.
  The root `phase1_complete_closure.py` compatibility command now delegates to
  the canonical receipt-backed `phase1_verify.py`; the historical
  `Phase1ClosureValidator` also delegates closure decisions to that gate, and
  its presence-only stage checks refuse certification. Its former `__main__`
  sample can no longer emit placeholder evidence.
  Latest relevant combined run after Q4_1 changes: Q4_1 expert, native-kernel
  and OLMoE differential suites plus low-bit packing, capture/gate, OLMoE
  gateway/batch and source-inventory suites passed **117/117**. Q5_0 suites
  then passed **18/18**, and Q5_1 suites **19/19**; full-size INT3, Q4_1,
  Q5_0 and Q5_1 receipts are accepted by the canonical receipt gate. The
  separate Q5_0 (18/18) and Q5_1 (19/19) runs passed; a combined packer rerun
  is optional, not a Phase 1 closure dependency.
- [x] Guard `SourceScopeConverter`, `P02NativeCPUConverter` and the old guide
  against promotion from string references or assumed receipts. Canonical
  `source_scope.py` remains the admission source of truth.
- [x] Locate and review the detailed P02–P15 checklist at
  `docs/audits/phase1/source-checklist.md`; this is the authoritative detailed
  requirement history. Added a 24 October reconciliation there; keep that
  dated summary and this active queue aligned without duplicating the full
  historical checklist.
- [x] Restore four missing workflow files from repository history:
  `.github/workflows/ci.yml`, `publish-pypi.yml`, `security.yml` and
  `expert-engine.yml`. Their source contracts pass locally; the workflows
  have not yet executed on GitHub in this working tree.
- [x] Add an explicit private-route scope for inference `GET /health`.
- [x] Reconcile Anthropic `/v1/messages` tool rejection text with the tested
  compatibility error contract. The route continues to reject tools and now
  includes the stable compatibility wording plus the capability reason.
- [x] Keep inference `/health` private under `production:read` across both
  security test contracts. Remove Unicode-only success markers from the CPU
  and Arc evidence-manifest writers and make `phase1_verify.py` report safely
  on cp1252 Windows consoles. These remove three additional full-suite
  failures and a verifier crash without fabricating evidence.
- [x] Rerun the full suite after resolving the ten discovered failures/errors
  and rewriting the closure gate: **1,770 OK, 95 skipped** (superseded by the
  24 October full run below). The 95 skips still require review by
  platform/dependency category; this environment has
  `argon2` but no `torch`. The earlier HTTP connection abort did not reproduce
  in the current full run and is not counted as a current failure.
- [x] Full `python -m unittest discover -s tests` on 24 October:
  **1,882 run, OK, 95 skipped, 635.595 seconds**. It preceded later A09
  diagnostic, float32 CLI-default, all-dtype CLI and Q4_0 route work.
- [x] One full-suite run reported a transient cleanup error in
  `test_managed_process_has_status_logs_and_stop`; the test then passed 10/10
  isolated repetitions, and the complete rerun below passed. No code change
  was made without evidence of a product lifecycle defect.
- [x] Latest full current-tree rerun: **1,888 run, OK, 95 skipped, 698.586
  seconds**. The latest focused converter/CLI/capture/closure-gate group also
  passes 42/42. A green suite does not replace
  missing family route receipts, native public-ABI execution, accelerator
  placement evidence, or distributed process-loss evidence.
- [x] After the 24 October P03 alias and P02 manifest-selection fixes, the
  project `.venv` run of `test_p0_p1_fixes.py` passes **228 tests, 3 skipped**;
  source-scope plus candidate-triage pass **90/90**. The canonical
  `phase1_verify.py` still reports **10/17**, with zero candidates and 126
  missing route-bound receipts; Phase 1 remains open.
- [x] Local evidence-capacity check: only OLMoE source/derived-F16/F32,
  out-of-scope dense Qwen3.5-0.8B, and OpenVINO Qwen2.5 checkpoints exist.
  No other admitted MoE checkpoint is available. The project `.venv` lacks
  Torch; `.accelerator-venv` has CPU-only Torch 2.14.1 and Transformers 4.57.6
  (not pinned 5.16.1), with `torch.cuda.is_available() == False`, no CUDA
  runtime/device and no `nvidia-smi`; the Windows video-controller query is
  access-denied. Do not synthesize the 126 missing receipts; another family
  checkpoint and accelerator access are required for those captures.
- [ ] Obtain the native/platform comparisons and benchmarks required for the
  admitted routes. A tiny OLMoE integration test is useful but is not a
  family wide correctness or performance certificate.
- [ ] Reconcile every P02–P15 obligation as implemented with evidence or
  concretely rejected. Set `phase1_closed=true` only after zero candidates,
  all 17 verifier checks, required receipts and the applicable full suite pass.
- [x] Clearly mark the dated `PHASE_1_STATUS_2026_10_19.md` and
  `PHASE_1B_STATUS.md` completion claims, plus the duplicate tracked TODO,
  as historical so they cannot override this live audit.

## Scope of this audit

This pass rechecked the canonical matrix and closure verifier source/output,
the root and tracked historical TODOs, universal/Phase 1B owner wiring,
evidence-manifest samples and the skipped-test sites. The previous full suite
result is carried forward; it was not rerun in this audit. This pass also
invoked route resolution (metadata only) against the two ignored local
checkpoints and inspected the P15 evidence predicates. It did not run
full-size publisher inference, actual GPU placement, distributed process loss,
or every media family. Those remain verification tasks, not completion claims.

### Recovery stop-prefix follow-up — 24 October 2026

Fixed a recovery-stream mismatch between K3's cumulative token events and
generic native sessions' continuation-only events. Recovery now withholds
possible stop-prefix tokens in token space across the restore boundary,
inherits and verifies the checkpoint's stop policy, derives returned text from
validated output token IDs, truncates at the exact matched token boundary, and
does not leak stop tokens when contextual decoding differs from standalone
stop decoding. A normal stop no longer becomes cancellation on the decoder's
next poll. Recovery events carry their policy and token history so a stop
spanning the saved boundary is detectable.
`tests.test_publisher_stop_sequences` passes 10/10 and the combined focused
publisher-recovery/lifecycle slice passes 21/21; Python compilation and scoped
`git diff --check` are clean.
This closes one recovery-stream defect;
cross-family P04/P14 caller closure and P15 remain open.

### K3 candidate-tooling route admission — 24 October 2026

The whole-model K3 candidate factory constructs `K3Execution` directly, so it
does not pass through the production service factory's P02 route gate. It now
derives the exact backend/arithmetic pair from its explicit execution policies
and calls `k3_route_identity()` before `load_k3()`; the selected tuple is also
passed into the loader for a second source-bound check. Unsupported FLA, Metal,
and BF16-device candidates therefore fail before payload conversion rather
than being evaluated as if they were admitted. A source regression checks gate
ordering and explicit route forwarding. Follow-up review found that the public
`load_k3()` and checkpoint payload methods still permitted route omission, and
the projection-shard loader had no exact route gate. All three loader surfaces
now require backend/arithmetic, and sharded loading verifies the route before
reading a shard. The candidate factory and production CLI already forward the
same identity. P02 K3 loader/tooling admission is now enforced at these reviewed
boundaries; the broader family/caller route map and P02/P15 remain open. The
combined P02 triage, source-inventory and native K3 regression slice passes
119 tests (one optional skip).

### P12 recovery checkpoint-set handoff barrier — 24 October 2026

The bounded replacement-rank recovery driver now re-admits every pinned
checkpoint after the entire restore set is loaded and immediately before it
hands restored sessions to application code. Per-checkpoint before/after
checks alone left earlier manifests unchecked while later checkpoints were
restored. If any pin changes at the final barrier, application work does not
start and all opened replacement sessions are closed in reverse order. The
recovery-driver, distributed checkpoint-error and process-loss slice passes
21 tests (two optional skips). This closes the checkpoint-set handoff window;
it does not solve arbitrary transport loss or establish distributed consensus
for every backend. P06/P12/P15 remain open.

### P14 serving cleanup diagnostics — 24 October 2026

Public publisher and native conditional construction/teardown now use the
shared safe failure formatter instead of `repr()`/`str()` on secondary cleanup
exceptions. The same safe reporting is wired through native-family CLI,
native public text/model adapters, Mixtral range construction, and Inkling,
GLM5, Qwen3.5 and Qwen4 multimodal owners. The original failure remains
primary, retry cleanup stays attached where available, and teardown continues
through independent owners even when a secondary exception is unprintable.
Gateway/provider diagnostics and K3/V4.1 CLI error output now use bounded safe
exception details: ordinary messages are preserved, while hostile `str()`
falls back to exception type. Focused local-AI, inference-route, chat, media,
public-lifecycle and cleanup coverage passes 79 tests (3 optional skips). Prior
sub-slices passed 87 tests (7 skips), 26 native/media tests (9 skips), and
23/23 cleanup tests. This closes error-formatting behavior in reviewed callers
only; media semantics, streaming, queue/backpressure, cancellation and
lifecycle closure for every serving caller remain open. P14/P15 remain open.

### P06/P12 root collective-loss teardown — 24 October 2026

The root coordinator previously set only `_failed` when a non-cancellation
exception escaped its collective request path. A timeout or broken collective
could therefore leave its service and rank-local sessions, snapshots, and
generation boundaries reachable behind an unusable coordinator. Root failures
now use the same poison-and-retire path as worker transport loss, retaining the
original exception and attaching bounded cleanup diagnostics. A focused
regression verifies all local owners are cleared and service teardown runs;
distributed checkpoint/recovery tests pass 20/20. Cancellation that completed
the collective protocol remains non-poisoning. This cannot recover a lost rank
or prove a global consensus after transport loss; replacement ranks are still
required. P06/P12/P15 remain open.

### P14 universal-session cleanup diagnostics — 24 October 2026

Universal-session, Phase 1B facade, and top-level runtime cleanup previously
formatted secondary failures with `repr(error)`. A hostile exception formatter
could interrupt retirement of remaining receipts, skip later profile saves,
or prevent universal/direct runtime owners from closing. All three lifecycle
layers now use bounded safe failure details and best-effort notes across owner
construction, pooled teardown, rollback, stream-iterator close, profile save,
and receipt release. Fault-injection tests confirm later independent cleanup
continues while the first error remains primary. Phase 1B lifecycle, universal
engine and production-path suites pass 45/45. This closes cleanup diagnostics
in these owners only; caller-wide serving lifecycle closure and P14/P15 remain
open.

### P14 Phase 1B warm-facade identity — 24 October 2026

The production runtime previously reused an existing Phase 1B facade without
checking its facade-only `hot_budget_bytes`, so a later caller could receive a
session configured under a different expert-pinning budget. It also silently
ignored a hot budget when `phase1b=False`. Opening now validates facade option
types and combinations before model admission and rejects warm reuse when the
requested budget differs. A production-path regression covers same-budget
reuse, mismatch rejection, disabled-facade misuse and invalid values. The
universal/Phase 1B lifecycle slice passes 46/46. This closes one warm identity
boundary only; cross-caller serving lifecycle coverage remains open (P14/P15).

### P14 Brio shared-prefix cleanup — 24 October 2026

`ClosedSetScoringMixin.scoring_prefix_tokens()` now preserves the original
scoring error while attempting both prefix rollback and snapshot release, even
when either cleanup exception has hostile `str()`/`repr()` methods. When there
is no operation error, rollback remains primary and release failure is kept as
a safe note; when cleanup alone fails, it is still raised. Added fault
injection for unprintable rollback and release failures. The Brio route plus
scoring-cleanup slice passes 22/22. This closes shared-prefix cleanup only;
end-to-end Brio parity and caller-wide P14/P15 remain open.

### P11 selected native ABI execution evidence — 24 October 2026

The actual native Segment C bridge now has an environment-local compile and
callback/handle execution check: `tests.test_segment_abi` compiles the bridge
and passes alongside the static declaration/reference audit in
`tests.test_public_abi_audit` (27/27 combined). This verifies the Segment bridge
contract on this host, not all 98 declared exports, 166 ctypes references,
family-specific calling conventions, platform variants or every public
factory/CLI/service route. P11 remains partial and P13/P15 remain open.

### P07 K3 FLA replacement cleanup boundary — 24 October 2026

When a K3 state swap retired prior FLA recurrent/convolution buffers, failed
`close()` calls were retained for retry and quarantined the executor, but the
operation could still return success when there was no primary request error.
It also formatted the cleanup exception with arbitrary `repr()`. The executor
now raises a bounded cleanup failure for an otherwise successful operation;
when unwinding an existing request failure it retains that failure and adds a
safe type-only note. Failed mixers remain reachable in the close-retry queue.
Fault injection verifies both paths. The complete K3 test glob passes 111
tests (8 optional skips). This closes only the FLA mixer retirement boundary;
K3 temporal/media and end-to-end serving closure, P07 and P15 remain open.

### Latest P11 ctypes-reference audit correction — 25 October 2026

The static ABI audit previously matched Python ctypes names only when the
native symbol was accessed from a variable literally named `library` or `lib`,
omitting nested owners such as `self.kernel.oa_*` and `self._lib.oa_*`. It now
includes dotted Python attribute owners and reports **190** unique references;
all 190 resolve to native header/C source text. `tests.test_public_abi_audit`
passes 6/6 and `phase1_verify.py` reports 98 exports and 190 ctypes names.
This closes an audit-coverage defect only; it does not prove each symbol's
ctypes signature, calling convention, compilation, or runtime compatibility.
P11/P13/P15 remain open.

P11 binding correction: the Qwen3.5 vision version probe now declares its
zero-argument ctypes signature explicitly. A mock-backed initialization
regression passes even without the optional Torch/Transformers oracle; the
full numerical vision module remains skipped because those optional packages
are unavailable.

P11 K3 dense-MLA native-owner lifecycle correction (25 October): the opt-in
`NativeKimiK3DenseMLA` wrapper now releases its owned library deterministically
through `close()`/context-manager use, serializes `forward()` against close,
keeps close retryable if native release fails, and releases the handle if ABI
symbol binding fails during construction. Three mock-only lifecycle
regressions pass without loading or compiling a native library. This addresses
one fixture-wrapper ownership defect only; it does not admit a K3 production
route or establish native ABI compilation/execution. The admitted-family ABI
symbol/range/schema/callback and caller-wiring map remains incomplete, and
P11/P13/P15 stay open.

Latest P14 transport-to-worker cancellation fix (25 October): a loopback HTTP
integration regression closes the response after its first SSE event and
confirms the request cancellation signal observes peer closure. The signal's
thread-safe Event is now observed by the publisher worker while its consumer is
suspended at `yield`; arbitrary caller callbacks remain on the consumer thread.
A publisher regression verifies worker cancellation while the consumer is
paused. The server, inference-route, and publisher cancellation suites pass
36/36. This closes generic transport and worker-signal propagation, not every
family-native cancellation/rollback path or warm-session lifecycle.

### P14 `/api/chat/stream` disconnect cancellation wiring — 2026-10-25

The first-party chat SSE route now follows the transport-cancellation contract
already present on inference-compatible routes: it starts the disconnect
watcher after request-body consumption, passes its event-backed callable
through `ChatRouteDependencies` and `ChatService.stream`, and forwards it
through text/vision gateway streams to the native publisher. Gateway
cancellation is re-raised rather than treated as provider failure eligible
for fallback; cache/buffered chunks are checked before publication, and the
chat generator retains deterministic provider-iterator cleanup. Added a
pre-persistence cancellation regression and a loopback `/api/chat/stream`
disconnect regression, plus gateway regressions proving cancellation cannot
trigger provider fallback and the transport signal reaches native providers.
`py_compile` for touched source/tests and `git diff --check` pass. The base
Python lacked `numpy`, but rerunning the expanded focused chat-route,
chat-service, HTTP server, gateway, and publisher-cancellation suites in the
project `.venv` passes 46/46.
This closes the chat HTTP transport-to-provider seam only. Full caller/backend
rollback, media lifecycle, warm-session and P14/P15 closure remain open.

### P13 capture case identity is selectable end-to-end — 2026-10-25

`canonical_row()` already rejected ambiguous route tuples when multiple
publisher pins shared the same six-field route, but the production OLMoE
capture API and CLI could not provide the required exact case ID. Added
optional `case_id` to the capture API and `--case-id` to the CLI, forwarding it
through native, leased, and public-range capture branches into canonical-row
resolution. Existing unambiguous invocations remain valid; ambiguous pins
still fail unless the chosen case ID resolves exactly. Source compilation
passes; the full `tests.test_phase1_capture` suite passes 32/32, including CLI
forwarding, real tiny-checkpoint route capture, and direct INT4, leased, and
public-range paths. This fixes artifact identity selection for the existing
OLMoE capture runner only; publisher-family capture adapters and the 126
missing route receipts remain outstanding.

Current evidence recheck: `.venv\Scripts\python.exe phase1_verify.py` still
reports 10/17, 171 admitted rows, zero candidates, and 126 missing receipts.
The workspace also contains `models/qwen3.5-0.8b`, but its pinned config says
`model_type: qwen3_5` with `Qwen3_5ForConditionalGeneration`; this is the dense
identity explicitly out of scope, not an admitted `qwen3_5_moe` checkpoint.
The other local model directories are OLMoE and dense OpenVINO Qwen2.5. No
matching admitted non-OLMoE MoE checkpoint was found for honest
publisher-family capture on this host.

### P03 explicit-route universal opener inventory guard — 2026-10-25

`UniversalEngine.open(..., route=...)` previously required a route resolver
only when selecting a default route. A registered adapter with no resolver
could therefore receive an explicit canonical tuple and be opened without
checkpoint-source resolution; canonical matrix admission alone does not bind
that tuple to the local tensor inventory. Every open now requires the exact
model-type resolver, including explicit-route opens, before route admission
or owner construction. The regression supplies a canonical explicit Mixtral
route to a resolver-less adapter and confirms the adapter is never invoked;
the prior mismatched-identity rejection remains checked. The focused test
passes 1/1 and `git diff --check` passes. This closes the generic universal
engine's resolver-less explicit-route bypass only. Resolver output still
binds source family/format rather than independently proving backend placement,
and other family factories, inventory consumers, route evidence, and P15
remain open.

### P02/P13 import compatibility and full-suite re-audit — 2026-10-25

Reconciled the remaining tests with the canonical modules after duplicate
implementations were removed. `phase1_a02_a03_real_routes` now uses the exact
canonical family/model/source-format identities; trace assertions that review
production factory owners use `device_routes`. The legacy A10 CPU and Arc
module names are thin re-exports of `cpu_evidence` and `arc_gpu_evidence`, not
second implementations. Phase 1B lifecycle tests now import the retained
`session_optimizer`, `async_io_engine`, and `memory_pinning` owners. The
documented `phase1_capture` entrypoint now exports `_write_receipts`, fixing a
fresh-process import failure in concurrent manifest updates.

Focused route/factory, A10, capture/receipt, closure-gate, Phase 1B lifecycle,
and media-lifecycle checks pass **92/92**; after the subprocess correction,
capture, reference-registry and parallel-manifest checks pass **39/39**. The
full discovery run completed **2,129 tests: 2,128 passed, 1 failed, 97
skipped**. Its sole failure was the missing `_write_receipts` compatibility
export; the exact failing parallel-capture test passes after the fix. Ruff
F821 and scoped `git diff --check` pass. The Phase 1 verifier still reports
**10/17**, with **171 admitted, 0 candidate, 877 restricted, 286 rejected**
rows and **126 missing exact runtime receipts**. No new runtime evidence was
captured; P13/A10 and P15 remain open.

### P04/P05 media-vs-state admission correction — 2026-10-25

The raw `FamilyMediaLifecycle` previously admitted Inkling/V4.1 MTP draft IDs
and V4.1 DSpark bytes as media frames, even though these are speculative or
recovery state and the production media consumers do not own those raw inputs.
Those fields now fail closed with P05/P12-specific reasons; processed lifecycle
types for MTP/DSpark are also rejected. Inkling image/audio and V4.1 vision
tracking remain admitted. Added direct and processed-input regressions. The
focused media, multimodal, publisher, observation-reference and capture set
passes 61/61. Raw media byte accounting now includes each image in an image
sequence and uses `memoryview.nbytes` for cast views; a regression covers both
image and video payload counts. This removes misleading lifecycle admission
and closes the accounting defect only; family-wide target-verified MTP, DSpark
owner-bound restore, runtime receipts, and P15 remain open.

### P03/P11 targeted source fixes — 2026-10-25

Qwen4 PLE inventory substitution now compares each substituted shard's
source-header shape and dtype with the provider's pinned snapshot, in addition
to exact names/config. A same-name wrong-row-count regression is included;
`tests.test_p02_candidate_triage` passes 33/33 in the project virtualenv.
`NativeFamilyPublicAdapters` public identity now includes the resolved
checkpoint root and verified config digest, preventing separate checkpoint
roots/config revisions from sharing an identity; its focused tests pass 3/3.
These close one P03 shard-header boundary and one P11 cache-identity boundary,
not the family-wide inventory/ABI maps. The canonical verifier was rerun:
source matrix, 162 route bindings, 178 owner references, and static ABI
consistency pass; overall remains 10/17 with 126/171 admitted routes missing
exact runtime receipts. P02/P03/P11/P13/P15 remain open.

### P14 compatibility SSE source-close retry — 2026-10-25

The OpenAI/Anthropic compatibility SSE wrapper now marks its upstream source
closed only after `close()` succeeds and allows one bounded retry after a
transient close failure. This prevents a failed first close from becoming
unreachable while bounding cleanup attempts. The inference-route suite passes
21/21, including the transient-close regression. This closes only the
compatibility SSE source-close edge; full caller lifecycle and provider worker
termination remain open. The Phase 1 verifier remains 10/17 with 126 exact
route receipts missing; P14/P15 remain open.

### P12 recovery session handle validation — 2026-10-25

`_RecoveryDriver` now requires positive exact-integer session handles (matching
`DistributedExecutionService`) and detects reuse by value rather than Python
object identity. Equal-but-distinct integer handles can no longer alias an
already-owned recovery session; rollback closes only acquired sessions. The
recovery-driver suite passes 8/8. Existing handoff re-pinning and safe worker
failure summaries were preserved. This does not solve peer stranding after
transport/process loss or provide distributed consensus receipts; P06/P12/P15
remain open.

### Consolidated verification after the targeted fixes — 2026-10-25

The combined route-binding, source inventory, A04-A07, public identity, P05
transaction, P12 recovery, publisher-stream, inference-route, and chat-route
regressions pass **153/153**. Ruff `F821` and scoped `git diff --check` pass.
The canonical verifier was rerun after these edits and remains **10/17**,
with **171 admitted, 0 candidate, 877 restricted, 286 rejected** routes; **126 admitted
routes** still lack exact runtime receipts. This confirms the source fixes
without changing Phase 1 closure. P02-P15 are not globally complete and P15
remains open.

### P09/P10 targeted Qwen4 packed-FP8 route audit — 2026-10-25

The Qwen4 device packed-FP8 path is traced through source format admission,
A07 placement, CLI provider construction, and the expert decode consumer. The
CLI rejects the legacy packed manifest before provider creation because it
lacks exact source dtype/P02 provenance; direct provider construction repeats
the family/device format check. This is distinct from the admitted Qwen4
F8_E4M3 checkpoint route. A07 tests pass 8/8. No code change was warranted;
this is one verified fail-closed path, not full P09/P10 coverage or runtime
placement evidence.

### P05 production speculative logical-publication rollback — 2026-10-25

The V4.1 DSpark production path composes `publisher_v41_dspark_session` with
`SpeculativeDecoder.step`. Its exception path now restores the prior logical
history, logits, and stop boundary before rolling target/draft caches and both
RNGs back. This closes an interruption window between decoder-state stores
that could otherwise leave the physical owners restored while logical history
remained advanced. A `KeyboardInterrupt` regression checks logical state,
target/draft state, both RNGs, and quarantine status; the focused transaction
suite passes 22/22. This does not reconcile every serving caller, publication
sink, family, or runtime receipt; P05/P13/P15 remain open.

### P06 production cancellation collective ordering — 2026-10-25

`DistributedExecutionService` now rank-agrees local cancellation-flag tensor
allocation failures before the cancellation `all_reduce`, then rank-agrees
scalar extraction failures after the reduction. This prevents surviving peers
from entering different next collectives after one rank fails locally at
either boundary. The focused distributed checkpoint/consensus, recovery,
launcher, and A06 suite passes 42/42. This does not recover a failed process
or transport, nor make an in-progress collective interruptible; P06/P12/P15
remain open.

### P11 selected family binding/ABI-lifecycle suite — 2026-10-25

Selected GLM-DSA, GLM5, K3 ABI lifecycle/decoder, Inkling, Qwen3.5 vision/text,
Qwen4 public lifecycle, public identity, and family source-binding tests run
42 cases: 31 pass and 11 skip because optional oracle dependencies are absent.
This exercises Python ownership and source-binding guards, not native ABI
compilation or execution; P11/P13/P15 remain open.

### Full Python suite after source fixes — 2026-10-25

`.venv\Scripts\python.exe -m unittest discover -s tests` completed with
**2,149 tests, 97 skipped, zero failures**. Some tests emitted egress-denial
logs and wrote temporary evidence fixtures; those fixture outputs are test
artifacts, not P13 runtime receipts. The canonical verifier remains 10/17
with 126 admitted routes missing exact runtime receipts, so P15 remains open.

### P03/P14 serving-caller follow-up — 2026-10-25

Chat now rejects more than one image attachment before creating or persisting a
turn because the current vision gateway contract forwards only one image.
This prevents silent media loss and phantom persisted attachments; it is an
explicit bounded restriction, not multi-image support. The text-attachment
stream also forwards disconnect cancellation to synchronous completion and
checks cancellation before attachment-correction retry. Chat/workflow and chat
route regressions pass 12/12; source inventory plus Inkling consumer tests pass
65/65 with one optional-dependency skip. The targeted P03 audit found no
confirmed factory/CLI inventory bypass or admitted orphan bank in the reviewed
planner, conditional loader, factory, and public-adapter boundaries, but did not
prove every optional/shared/MTP/expert/media bank reaches a live consumer.
The A02/A04-A07 module regression set passes 41/41. This is source-contract
coverage: A05's RNG manager and A06's shard/consensus model are documented
standalone helpers, not replacements for production decoder or process-group
integration.
P03/P14/P15 remain open at family-wide/caller-wide scope; the canonical
verifier remains 10/17 with 126 exact-route runtime receipts missing.

### P13 local evidence-source recheck — 2026-10-25

Current workspace inspection finds only the three OLMoE variants, OpenVINO
assets, and `qwen3.5-0.8b`; the Qwen checkpoint is the already out-of-scope
dense identity. The existing pinned receipt manifest contains 45 routes, all
OLMoE. `nvidia-smi` is unavailable and the accelerator environment reports
CPU-only PyTorch with no CUDA/XPU devices. Therefore no eligible non-OLMoE
checkpoint or accelerator observation source is currently present for the
126 missing route receipts. This is an evidence-source constraint, not a
reason to synthesize receipts or weaken the gate; P13/P15 remain open.

### P13 audit of new Ollama certifier artifact — 2026-10-25

`phase1_ollama_certifier.py` and `phase1_ollama_receipts.json` are not valid
Phase 1 observations. The code probes a generic Ollama prompt, then copies the
same result to every family binding without running the admitted engine route,
pinned checkpoint, persisted format, ABI, cache, recovery path, or device
placement. It also marks proxy families as certified. The JSON contains 179
`certified: true` records, including 58 explicitly proxied records, with no
canonical `case_id` or checkpoint artifact pins. The canonical closure gate
does not load this file; only `phase1_runtime_receipts.json` is authoritative.
Keep the Ollama artifact excluded from P13 totals and do not represent it as
route certification. This adds no route evidence; P13/P15 remain open.

### P11 ABI-layout regression — 2026-10-25

Extracted the production GLM/Inkling public-layout guard into
`_verify_family_abi_layouts()` and added regressions for exact descriptor,
tensor, config, and layer layouts plus wrong size, field-count, and offset
rejection before native registration. `tests.test_family_source_binding` and
`tests.test_public_abi_audit` pass 26/26. A direct probe against the existing
`build/public-range-runtime/Release/openagent_segment_runtime.dll` matched all
four ctypes layouts. The DLL was pre-existing and was not rebuilt. This closes
only these four GLM/Inkling records, not the full family ABI map,
callback/reentry matrix, all-export native execution, or caller wiring. The
canonical `phase1_verify.py` was rerun after this change: source admission,
162 route bindings, 178 owner references, and the static ABI check pass;
closure is unchanged at 10/17, with 126/171 admitted routes missing exact
receipts. P11/P13/P15 remain open.

### P05 publisher speculative snapshot cleanup ownership — 2026-10-25

Fresh DSpark generation and recovery now register their `SpeculativeDecoder` with
the owning `PublisherSession`. Request reset retries pending target/draft
snapshot releases before resetting either warm physical owner; if release still
fails, the session is quarantined and keeps the decoder and owners reachable
for a later `close()` retry. Session close retries snapshot cleanup before
closing borrowed owners. Full discovery initially found two compatibility
errors in tests that construct a partial `PublisherSession` without `__init__`;
the hook now safely handles those legacy/test objects. The affected publisher
cleanup and speculative transaction suites pass 42/42, including transient
release retry and persistent-failure owner retention. Full discovery ran 2,170
tests with 97 skips and those two errors before the compatibility correction;
the full suite was not rerun after the correction. This closes one production
cleanup-ownership edge only. The canonical verifier remains 10/17 with 126
admitted-route receipts missing; cross-caller P05 and P13/P15 remain open.

### P14 cancellable Ollama/OpenAI vision streams — 2026-10-25

Ollama and OpenAI-compatible vision requests now use their provider streaming
protocols instead of the gateway's blocking one-shot fallback. Transport reads
receive a thread-safe cancellation event; the caller-provided cancellation
predicate is polled only on the stream consumer thread. A regression proves
the event cancels a transport even when it has yielded no chunks and that the
predicate is not called from worker threads. The model-gateway, inference-route
and HTTP-server suites pass 55/55; Ruff and scoped `git diff --check` pass.
This closes the provider streaming/cancellation seam only, not remote-provider
parity, arbitrary blocked non-streaming calls, or caller-wide P14/P15.

### 108-route capture source update — 2026-10-25

Qwen4 `qwen4_exp_text` now has a public-range capture CLI path in addition to
native/device text capture. It requires both the compiled Qwen4 family library
and Segment/Edge runtime (either supplied explicitly or built through the
existing compiler helpers), binds the executed `NativeQwen4PublicTextDecoder`
to the canonical execution and ABI chains, requests executed router weights,
measures dispatcher-owned per-layer cache receipts, and pins the complete
57-file Python/C/header source closure plus both runtime libraries. The Colab runbook now includes
the corresponding capture loop. Source capture-path coverage for the 63
non-OLMoE target rows is 21/63 (18 existing text native/device rows plus 3
Qwen4 public-range text rows); 42 rows still lack capture paths. This is not
runtime evidence: do not count any of these three routes as captured until
they produce independently verified route-specific receipts from the real
checkpoint and runtime. Publisher media, Inkling speculative/MM capture, and
GLM5 publisher media/public-range remain unimplemented. P02/P04/P05/P13/P15
remain open.

### P11 calling-convention audit correction — 25 October 2026

The public ABI source audit previously removed `OA_CALL` before comparing
declarations with C definitions, so an exported function could have a
declaration/definition calling-convention mismatch without failing the check.
The audit now compares explicit convention tokens for each declared export;
regressions cover mismatched and matching contracts. `tests.test_public_abi_audit`
passes 8/8. The focused ABI/family set passes 68 tests with 11 skips; skipped
cases require optional exact model/oracle dependencies. The canonical verifier
passes its source checks and remains 10/17 with 126/171 route receipts missing.
This closes a static-audit defect, not P11: family-wide native ABI execution,
tensor/head schema review, and complete factory/CLI/service wiring remain
unverified; P11 and P15 stay open.

Validation for the 108-route capture-path update: the focused 80-test
P02/capture/inventory set passes, and the 12-test capture-contract module was
rerun after expanding the Qwen4 public source-pin closure. Touched Python files
compile, the Colab notebook parses as JSON, and `git diff --check` passes. The canonical verifier after the Qwen4
route-binding change passes source admission, all 162 exact route bindings,
181 callable-owner references, and static ABI checks (98 exports / 190 ctypes
names). It remains 10/17 because 126/171 admitted routes lack route-specific
runtime receipts; this update created none.

### P14 InferenceService SSE abandonment — 25 October 2026

The top-level `InferenceService.stream_complete()` and `stream_chat()` wrappers
now explicitly close their `ServingEngine` iterators when the SSE consumer
disconnects, and attach close failures to an existing cancellation/error rather
than replacing it. Regressions exercise both wrappers' close propagation.
`tests.test_serving_engine_lifecycle` plus `tests.test_inference_routes` pass
57/57; scoped Ruff and `git diff --check` pass. This closes the nested SSE
wrapper edge only; other caller lifecycle paths and P13/P15 remain open.

Latest P14 direct-serving semantics — 2026-10-25: completion and chat usage
counts now use actual generated token IDs rather than re-tokenizing decoded
text. Configured-tokenizer decode failures return an explicit error instead of
silently falling back to byte decoding. Public completion/chat stream wrappers
close their inner `_run_stream()` source on consumer abandonment, and cleanup
errors do not replace `GeneratorExit`. OpenAI completion `echo` is now parsed
strictly and emitted before generated text in normal and streamed responses;
prompt tokens remain prompt usage. Unsupported `logprobs` fail before opening
a session. Explicit temperature/top-p and nonzero presence/frequency penalties
were also silently ignored by this direct API; absent values now mean backend
default, and explicit controls fail before session allocation. The Phase 2
session handle had the same issue for arbitrary keyword options; it now rejects
unforwarded options before batch admission. Regressions cover these guards,
exact token counts, strict decode failure, echo, both stream wrappers, and
disconnect cleanup. The focused serving/inference/admission/batching set passes
84/84; scoped Ruff and `git diff --check` pass. The canonical verifier remains
10/17 with 126/171 admitted routes missing exact runtime receipts. These are
selected direct-serving semantics; caller-wide P14 and P13/P15 remain open.

### P14 active Phase 2 operation ownership — 2026-10-25

`Phase2SessionHandle.generate()` and `stream()` now hold an operation lease for
their full execution, including prefix lookup, scheduler wait, and stream
abandonment cleanup. `close(timeout=10.0)` rejects new work and waits a bounded
time for active calls before closing the borrowed Phase 1 owner; timeout or
owner-close failure leaves teardown retryable. The timeout is finite and
bounded, and an interrupted wait resets the closing state for retry.
Stream-cancel cleanup failures are attached to the primary
consumer/cancellation exception and cannot leak the operation lease. Focused
concurrency, forwarded-Phase-1-method, interrupted-wait, and error-priority
regressions prove callbacks/methods cannot race owner close, and a later close
succeeds after release. Serving stream method lookup is now covered by the
session cleanup guard, so a
raising backend property cannot strand an opened session or mask its exception
with a close error. The expanded serving, gateway, multimodal, publisher, and
Phase 2 lifecycle group passes 166/166; scoped Ruff and `git diff --check` pass.
The canonical verifier remains
10/17 with 126/171 admitted routes missing exact runtime receipts. This closes
one Phase 2 handle race only; cross-caller P14 and P13/P15 remain open.

### P14 generic OpenAI stop handling — 2026-10-25

The direct `ServingEngine` accepted `CompletionRequest.stop` but dropped it at
the backend call, so response and SSE output could include stop text. Stop is
now strictly encoded without special tokens by the configured tokenizer,
matched against
generated token prefixes, and withheld from completion/chat results and
streams. Streaming stops and closes the backend iterator at the match; a
generate-only owner is trimmed after it returns. A configured tokenizer error
fails before session open instead of silently switching to byte IDs.
Regressions cover completion, chat, SSE, generate-only fallback, and rejected
stop/tokenizer inputs. Direct streaming also keeps backend method and iterator
close-property lookup inside the cleanup guard, ensuring a raising proxy still
closes its session without masking the primary error. The post-fix
serving/inference/HTTP set passes 58/58. Terminal chunks now distinguish a
matched/natural stop from max-token
exhaustion (`length`). Integration smoke passes 25/25. The broader
serving/gateway/inference/HTTP set passed 80/80
before strict-encoder hardening; a 171-test cross-caller run had one Windows
SSE socket abort, and that affected test passed 1/1 in isolation. Ruff and
scoped `git diff --check` pass. The canonical verifier remains 10/17 with 126/171 admitted routes
missing exact runtime receipts. This closes only generic serving stop-prefix
handling; generate-only execution cannot stop compute early and broader caller
and route evidence gaps remain.

### P07/P12 K3 packed-owner retryable teardown — 2026-10-25

`K3Execution.close()` previously invoked `NativeMXFP4`'s private weakref
finalizer directly. If the native library release callback failed, that
finalizer became permanently dead, bypassing `NativeMXFP4.close()`'s retry
contract and losing the packed owner. K3 now calls the public `close()` method
and retains the owner until release succeeds. The injected transient-release
regression passes (`tests.test_k3_fla_cleanup`, 3/3); Ruff passes on that test
file and scoped `git diff --check` passes. Existing lint findings in the
pre-existing compressed-style K3 executor are unchanged. This closes one
K3 teardown edge, not K3 continuation/serving or all-family recovery. P07/P12/
P15 remain open; the canonical verifier still requires 126 exact-route
runtime receipts.

### P05 committed speculative step survives snapshot-release failure — 2026-10-25

SpeculativeDecoder.step() previously committed target/draft state and then
raised from its finally block if release of a rollback snapshot failed. The
caller consequently saw an error instead of the committed token tuple. A step
now retries transient release once, returns the committed SpeculativeStep
with cleanup_pending=True if release remains unavailable, and blocks further
steps until pending cleanup is resolved or the decoder is reset. The V4.1
fresh and resumed publisher paths record/emit that committed token result, skip
a checkpoint that cannot be safely captured, and return explicit
cleanup/checkpoint-pending status while retaining a failed warm owner for later
close retry. Speculative transaction plus publisher stop/recovery tests pass
37/37, including persistent release failure, generation termination at the
committed boundary, response retention, and normal transient retry. Phase-wide
P05 callers/publication contracts and
P13/P15 remain open; latest verifier remains 10/17 with 126 exact-route
runtime receipts missing.

Latest P14 compatibility stream error hardening — 2026-10-25: backend
exceptions with hostile `__str__` and cleanup exceptions with hostile `__repr__`
could break OpenAI/Anthropic SSE error reporting. The translator now uses safe,
bounded failure summaries; compatibility-route error envelopes and provider
fallback diagnostics use the same safe formatting. The injected unprintable-
backend and cleanup regressions pass. This is selected error-formatting
coverage, not full protocol or caller lifecycle closure; P14/P15 remain open.

Latest P04 publisher modality preflight — 2026-10-25: unsupported mixed
image/video publisher input now fails before the shared media lifecycle admits
a frame (K3's separately ordered mixed-media path remains allowed). The
regression asserts neither lifecycle nor model dispatch was reached; publisher
media-lifecycle and ordering tests pass 10/10. This closes one pre-admission
ordering edge, not the full family/caller media matrix; P04/P14/P15 remain open.

Latest P14 concurrent universal-owner cleanup — 2026-10-25: simultaneous opens
for one warm-session key could construct a duplicate owner; if closing the
loser failed, it was not retained for retry. `UniversalEngine.open()` now
retains that owner in `_retired`, and `close_model()`/`close()` retry teardown.
The injected race/close-failure regression passes; the universal-engine file
passes Ruff and scoped diff checks. This closes one cleanup edge only and does
not change the 126 missing runtime receipts or P15 status.

### P13 audit of new Ollama certifier artifact — 2026-10-25

`phase1_ollama_certifier.py` and `phase1_ollama_receipts.json` are not valid
Phase 1 observations. The code probes a generic Ollama prompt, then copies the
same result to every family binding without running the admitted engine route,
pinned checkpoint, persisted format, ABI, cache, recovery path, or device
placement. It also marks proxy families as certified. The JSON contains 179
`certified: true` records, including 58 explicitly proxied records, with no
canonical `case_id` or checkpoint artifact pins. The canonical closure gate
does not load this file; only `phase1_runtime_receipts.json` is authoritative.
Keep the Ollama artifact excluded from P13 totals and do not represent it as
route certification. This adds no route evidence; P13/P15 remain open.

### P14 cancellable Ollama/OpenAI vision streams — 2026-10-25

Ollama and OpenAI-compatible vision requests now use their provider streaming
protocols instead of the gateway's blocking one-shot fallback. Transport reads
receive a thread-safe cancellation event; the caller-provided cancellation
predicate is polled only on the stream consumer thread. A regression proves
the event cancels a transport even when it has yielded no chunks and that the
predicate is not called from worker threads. The model-gateway, inference-route
and HTTP-server suites pass 55/55; Ruff and scoped `git diff --check` pass.
This closes the provider streaming/cancellation seam only, not remote-provider
parity, arbitrary blocked non-streaming calls, or caller-wide P14/P15.
