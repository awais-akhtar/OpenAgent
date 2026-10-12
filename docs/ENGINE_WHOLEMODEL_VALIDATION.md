# Whole-Model Correctness Framework

This opt-in framework produces evidence for one exact pinned text model and
input, not full-family certification or upstream ABI compatibility. It does not
modify registry gates, loaders, family_conformance, or CI. This implementation
pass is code only: no tests, compilation, linting, or verification were run.

## Injected Adapters

Supply two importable factories as `module:callable`. Each receives
`offline_root` (resolved Path) and `options` (JSON object), and returns the
Adapter protocol in `wholemodel_validation/contracts.py`. No fake or default
adapter is provided. Factories wrap unchanged authoritative/candidate code;
they must not download, build extensions, rewrite vendor sources, or reuse the
candidate as oracle. Importing this package does not execute comparisons.

Metadata requires model_type, revision, config_sha256, weights_sha256,
tokenizer_sha256, vocabulary_size, num_layers, dtype, implementation,
implementation_revision, device, dependencies and cache_schema. Oracle metadata
additionally asserts authoritative=true and unchanged=true. These declarations
require human review; hashes alone cannot establish authority.

cache_schema maps each leaf name to a zero-based layer index or `global`.
Every layer must be covered. Include occupied KV tensors, recurrent state,
convolution/FIR histories, lengths, positions, and all continuation state.
Canonicalization may transpose/slice physical buffers to occupied logical state,
but must not round, recompute or hide differences. An explicitly declared forward
comparison view may project observed normalized latent state through actual
execution weights, provided raw state and weights are retained separately and
the numerical policy is marked uncertified. This is not a raw cache export.
Unsupported cache equivalence
must raise. Empty and nonfinite cache leaves are rejected.

new_session(seed=...) creates independent empty state, seeds every RNG and uses
deterministic eval/dropout-disabled execution. Same-object session reuse is
rejected. Shared internal mutable state cannot generally be detected and is
prohibited. forward(tokens) appends tokens and returns Observation with detached
floating logits `[appended_token_count, vocabulary_size]` plus cache leaves.
Sessions are closed on success or failure.

## Manifest And CLI

For the next execution pass:

```text
python -m openagent.adaptive_engine.wholemodel_validation manifest.json --output evidence/run-001
```

Input is a JSON object with:

```json
{
  "schema": "openagent.wholemodel-validation-input.v1",
  "offline_root": "offline",
  "seed": 914,
  "tokens": [1, 4, 7, 2],
  "generation_steps": 8,
  "chunk_schedules": [[2, 2], [1, 3]],
  "tolerances": {
    "logits": {"atol": 0.000001, "rtol": 0.00001},
    "cache": {"atol": 0.000001, "rtol": 0.00001}
  },
  "oracle": {
    "factory": "your_oracle_adapter:create",
    "options": {},
    "artifacts": [
      {"kind": "source", "path": "vendor/model.py", "sha256": "SOURCE_HASH"},
      {"kind": "config", "path": "config.json", "sha256": "CONFIG_HASH"},
      {"kind": "weights", "path": "model.safetensors", "sha256": "WEIGHTS_HASH"},
      {"kind": "tokenizer", "path": "tokenizer.json", "sha256": "TOKENIZER_HASH"}
    ]
  },
  "candidate": {
    "factory": "your_candidate_adapter:create",
    "options": {},
    "artifacts": [
      {"kind": "source", "path": "candidate/model.py", "sha256": "SOURCE_HASH"},
      {"kind": "config", "path": "config.json", "sha256": "CONFIG_HASH"},
      {"kind": "weights", "path": "model.safetensors", "sha256": "WEIGHTS_HASH"},
      {"kind": "tokenizer", "path": "tokenizer.json", "sha256": "TOKENIZER_HASH"}
    ]
  }
}
```

Replace descriptive hash strings with actual SHA-256 pins. All artifact paths
must resolve beneath offline_root. Both roles are pinned before importing either
adapter and pins are checked again after execution. Pin all transitive sources,
adapter sources, native libraries and every weight shard. For sharded weights,
weights_sha256 identifies the pinned weight index (also a weights artifact).
Record dependency versions in metadata. The harness cannot infer whether the
declared artifact list is complete or whether an adapter really loaded it.

The harness performs no network operations. Injected code is trusted executable
code, not sandboxed; enforce an offline process policy externally. Use distinct
output directories: fixed artifact names are overwritten on repeated runs.

## Comparisons And Reports

Full prompt, single-token incremental and optional chunk partitions use separate
sessions for both roles. At each boundary, cross-role logits/cache comparisons
run. Incremental accumulated logits and cache are compared to independent fresh
full prefixes for each role; final states also compare to full-prompt baselines.
Greedy continuation runs independent trajectories, exact token comparisons,
per-step logits/cache checks and full-prefix continuation checks. Argmax ties
choose the lowest token ID. generation_steps is exact; no EOS stop or sampling.
An additional fresh shared-input rollout feeds both roles the oracle's generated
tokens. Its checks retain numerical diagnostics after independent trajectories
diverge; they cannot override any failed greedy-token check.

Floating comparison is abs(actual-reference) <= atol + rtol*abs(reference).
Integer/boolean state is exact. No broadcasting, missing leaves, missing layers,
empty/nonfinite evidence, exceptions, unavailable oracle or unsupported model
counts as success. All checks must pass; failures never become skipped successes.
Boolean/integer/floating contract mismatches fail. Signed and unsigned integer
comparison does not promote through floating point. Declared cache_layout values
must agree between roles; identical leaf names are not proof of equivalent state.

report.json includes manifest, pinned hashes, model/runtime provenance,
tolerances, layer-specific cache discrepancies, first mismatch coordinates,
maximum absolute/relative errors, generated tokens and explicit errors.
observations.npz retains detached raw logits/cache snapshots, including full
prefixes. File and per-tensor dtype/shape/content hashes appear in the report.
Partial observations are retained after failure. Output I/O errors can prevent
report creation and raise rather than claiming success.

Exit codes: 0 all comparisons passed; 1 prerequisite/execution/comparison failure;
2 unreadable or malformed JSON. Fresh-prefix work and evidence volume grow with
prompt length. This harness is not a benchmark.

## Remaining Work

Concrete K3 candidate factories and readers now export compressed MLA rows,
rotary coordinates, KDA recurrent matrices, and all convolution histories.
The unchanged authoritative factory requires an explicit reader for the state
actually exposed by its pinned implementation. Expanded vendor KV is not
silently inverted into latent state, and missing FLA or hidden cache fields
produce an unsupported contract, not a passing comparison.

`wholemodel_validation/k3_vendor.py` now supplies a concrete reader and
`create_k3_expanded_authoritative_adapter` for the unchanged publisher text
source at revision `f831ab66814297da540d832a5235f8e904f29d06`, source SHA-256
`9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a`.
The reader checks the actual cache class source hash and exports expanded MLA
key/value tensors, raw FLA recurrent matrices and the complete three convolution
buffers. Inactive slots must remain None; extra instance members fail closed.
Global position is read from occupied MLA tensor lengths, not fabricated from
the requested append count. A KDA-only schedule has no supported observed length
under this publisher contract and is rejected.

Use explicit `layer_types` (`kda`/`mla`), the result of `vendor_cache_schema`, and
metadata cache_layout `kimi-k3-publisher-expanded-kv-raw-fla-v1`. All normal local
loader authorizations/pins remain required; the exact publisher text source must
be among source artifacts. This concrete factory prohibits forward/cache reader
overrides. FLA ordering and convolution buffer extents are preserved raw rather
than guessed from the native executor. There is deliberately no expanded-KV to
compressed-latent conversion. The native K3 candidate's compressed schema cannot
pass this reader's schema/layout gate. Published wrapper auto-model resolution,
FlashAttention and FLA availability are still execution prerequisites, not
implemented substitutes.

### Explicit Forward MLA View

`wholemodel_validation/k3_projection.py` provides the original-code helper
`project_mla_cache` and `create_k3_projected_candidate_adapter`. The equations
establish this forward ordering: normalized latent `[1,T,R]` multiplied by the
actual immutable executor `kv_b_proj.weight` transpose produces
`[1,T,H*(N+V)]`; reshape to `[1,T,H,N+V]`, transpose to `[1,H,T,N+V]`, split each
head's nonrotary key and value, and append the observed shared rotary coordinates
to the key. The helper never applies normalization again, recovers a latent from
expanded KV, reruns hidden-state computation, or changes the oracle.

The full projected candidate adapter supports MLA-only schedules and requires
`publisher-none` rotary semantics and the pinned publisher projection RMSNorm
epsilon `1e-6`. Its normalized cache uses the existing publisher expanded
`vendor_cache_schema` and cache_layout `kimi-k3-publisher-expanded-kv-raw-fla-v1`,
so it can compare against the unchanged expanded authoritative reader. Supply
the usual candidate checkpoint/config/prebuilt-library options and explicit
`projection_tolerance: {"atol": ..., "rtol": ...}`. These finite nonnegative
tolerances must exactly equal the manifest's cache tolerance; there is no hidden
or automatically widened projection budget.

The helper can project an individual MLA layer from a hybrid snapshot, but the
complete reader/factory rejects hybrid schedules with explicit unsupported KDA
fields. MLA alignment alone does not establish FLA recurrent orientation or
convolution buffer ordering/extent. It cannot turn partial alignment into a
whole-cache pass.

The Observation protocol now carries separate `evidence` tensors, explicitly
declared by role-local `evidence_schema` and `empty_evidence_shapes`. For this
candidate they contain the complete occupied raw native cache, per-layer
positions and actual projection weights. These are stored under `.evidence.`
in observations.npz with normal dtype/shape/content hashes; they are not compared
against unlike expanded oracle tensors. Same-role full-prefix/chunk/generation
checks also compare raw evidence, with exact projection weight comparisons.
Weights are repeated in observations, so this opt-in path increases evidence
volume and export memory, not execution cache residency.

`cache_projection` metadata records the equation ordering, pinned source hash,
NumPy FP32 arithmetic, explicit tolerance and `certified: false`. Projected
checks are labeled `projected: true` and `numerically_certified: false` in the
report. Torch/accelerator reductions and lower precision may differ from NumPy
FP32 matmul. Forward projection can also hide latent differences in a weight
nullspace; retaining raw state is evidence, not proof of vendor latent equality.
Neither passing this view nor choosing a tolerance certifies reassociated native
attention arithmetic, complete checkpoint parity or hybrid-model correctness.
No tests, imports for validation, builds or numerical execution were run for
this addition.

The V4.1 incremental reader exports every persistent native layer field: packed
local/global rows, pending projection and gate rows, FP32 index entries, and
Engram positions and token history. `v41_cache_schema` and
`v41_empty_cache_shapes` provide the complete native declaration. The older
history-only reader is insufficient for whole-layer evidence. Publisher cache
alignment remains separate from exporting the native equation schema.
No concrete publisher V4.1 cache alignment is claimed: the current native
`openagent.v41.composed-equations.v1` state is not an authoritative publisher ABI.
Its reader now rejects stale live-position/token-history snapshots as well as
foreign owners. An unchanged, source-pinned publisher continuation contract is
still required before an authoritative V4.1 adapter can be derived responsibly.

No published checkpoint parity is claimed. Authority,
pin completeness, accelerator determinism and internal session independence need
review. Process isolation, provisioning, multi-case aggregation and CI integration
are not implemented here. Execution requires a separately authorized verification
phase. Unavailable oracle prerequisites must be reported as failed evidence rather
than skipped success.
