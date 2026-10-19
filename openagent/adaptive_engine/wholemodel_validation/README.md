# Concrete Adapters

The adapter source has been implemented. Numerical tests, compilation, linting
and runtime verification have not been performed. All adapters execute real unchanged
implementations; no oracle stand-ins, FLA substitutes or source rewriting exist.

## Torch

Programmatic use with an already loaded unchanged causal model:

```python
import copy
from openagent.adaptive_engine.wholemodel_validation.adapters import TorchCausalAdapter

# loaded_model and model_metadata are supplied by the caller.
# model_metadata includes all runner-required identities and explicit schema.
adapter = TorchCausalAdapter(
    model_factory=lambda seed: copy.deepcopy(loaded_model),
    metadata=model_metadata,
    kv_only=True,
)
```

Every factory invocation must return a fresh model, not a cached singleton.
Deepcopy is an example of independent ownership, not an adapter-supplied source
rewrite. The adapter calls eval and uses inference_mode. Python, NumPy, Torch
CPU and available CUDA RNG streams are session-local and caller state is restored.
Deterministic Torch algorithms are required, cuDNN benchmarking and TF32 are
disabled during calls and backend flags are restored afterward. Unsupported
deterministic kernels fail; CUDA environment prerequisites remain caller-owned.
Only this adapter's calls share its RNG lock; concurrent unrelated code must not
change process-global RNG/backend flags during a run.

Exact model_type is checked on the outer model.config, not replaced by a wrapper's
text_config identity. Optional metadata.text_model_type independently declares
the inner identity and must agree between oracle and candidate. Vocabulary and
layer geometry use text_config when present, otherwise model.config. Text-only
configs therefore retain the ordinary exact model_type check. Runtime evidence
records both observed outer and inner identities.

Scope is text-only token-to-logit/cache execution, even when the authoritative
class is a multimodal wrapper. No pinned image/audio/video fixtures, preprocessing,
media-first context protocol, or vision parity comparisons are supplied. A
forward_call can adapt the unchanged text invocation, but accepting wrapper
identity or running such a callback must never be described as multimodal parity.

Forward uses batch-one input_ids, full occupied attention_mask, past_key_values,
use_cache and return_dict. position_ids defaults to enabled; cache_position is
opt-in. Optional forward_call(model, kwargs) adapts a family call signature, but
must return all appended logits and observed past_key_values. No generation API
is used. The adapter widens bfloat16 to float32 exactly for NumPy export.

For CLI manifests use factory:

```text
openagent.adaptive_engine.wholemodel_validation.adapters:create_torch_adapter
```

Options: model_factory (`module:callable`), model_options, metadata, optional
input_device, position_ids, cache_position, forward_call and cache_reader.
The injected model_factory is called with seed, offline_root and options and
returns an already-loaded model. It must not provision/download/compile.

Standard KV mode requires kv_only=true plus exactly this schema for each layer:
position -> global; layer.N.key -> N; layer.N.value -> N. It supports legacy
two-member tuples and exact standard DynamicCache/DynamicLayer representations.
Canonical tensors are [1, heads, occupied_tokens, dim]. It rejects sliding,
static, offloaded, quantized, hybrid and unknown cache classes rather than using
to_legacy_cache (which can lose state). kv_only is an explicit reviewed assertion,
not inferred from model_type or an architecture alias.

## Family And Native State

Use a custom cache_reader(context) returning exactly every cache_schema key.
CacheContext includes cache, model/executor, output, observed appended-token
position and appended_tokens. Exported leaves are detached immediately, so later
calls cannot mutate prior observations. Missing/extra/nonfinite/empty leaves fail.

MappedCacheReader(schema, readers) requires one callable accessor per leaf. For
Torch recurrent/MLA families, access the actual returned cache or observed model
state, including convolution/FIR histories, recurrent matrices, compressed latent
state, rotary components, lengths and positions. Never fill absent state with
zeros, recompute it, drop it, or silently expand an incompatible representation.

NativeExecutorAdapter(executor_factory, metadata, cache_reader=...) invokes real
executor.forward(np.int64 tokens) and uses a matched `snapshot`/`restore` or
`snapshot_state`/`restore_state` pair for rollback and default cache observation. Optional
forward_call(executor, tokens), cache_getter(executor), close_call(executor) map
existing APIs. A custom cache_getter is mandatory when neither snapshot pair is available.
No reset of a shared executor is used as a substitute for independent sessions.
Python/NumPy RNGs are isolated; seed external native RNGs in the factory using
the supplied seed. close_call must release externally owned handles if executor
close is unavailable; adapters never guess DLL ownership/unload semantics.

### Native OLMoE candidate

For the native leased OLMoE route, use
`wholemodel_validation.olmoe_native:create_olmoe_native_adapter`. Its metadata
must use `cache_schema(num_layers)`, which declares position and both observed
KV tensors for every layer. Candidate options name relative, runner-pinned
`manifest` and prebuilt `library` artifacts and provide explicit context,
resident-weight, expert-cache, transfer, staging, and snapshot byte budgets.
The adapter verifies those two artifact hashes before constructing an owner;
it does not build or substitute a native library. The cache reader exports the
actual occupied key/value tensors from the decoder's owned snapshot and rejects
missing layers, invalid geometry, or nonfinite values. Session close attempts
decoder, edge, and engine cleanup independently. This provides a source adapter
for an OLMoE comparison, not an observation or a certification result; the
runner still needs complete model/source/dependency pins, a separate
authoritative oracle, and a successfully produced report/receipt.

CLI factory:

```text
openagent.adaptive_engine.wholemodel_validation.adapters:create_native_adapter
```

Options: executor_factory, executor_options, metadata and explicit cache_reader
or cache_reader_factory; optional forward_call, cache_getter and close_call.
executor_factory receives seed, offline_root and options and returns a fresh
loaded native executor using existing compiled libraries only.

For native snapshots shaped (position, layer_tuples), use
cache_reader_factory:

```text
openagent.adaptive_engine.wholemodel_validation.cache_readers:create_snapshot_reader
```

cache_reader_options supplies position_name and layers (one list of leaf specs
per layer). Example dense/KDA mapping for an existing Kimi Linear snapshot:

```json
{
  "position_name": "position",
  "layers": [
    [
      {"name": "layer.0.state", "batch_axis": 0},
      {"name": "layer.0.conv_q", "axes": [1, 2, 0], "batch_axis": 0},
      {"name": "layer.0.conv_k", "axes": [1, 2, 0], "batch_axis": 0},
      {"name": "layer.0.conv_v", "axes": [1, 2, 0], "batch_axis": 0}
    ],
    [
      {"name": "layer.1.key", "token_axis": 0, "axes": [1, 0, 2], "batch_axis": 0},
      {"name": "layer.1.value", "token_axis": 0, "axes": [1, 0, 2], "batch_axis": 0}
    ]
  ]
}
```

This explicitly reads all four KDA tuple entries and both dense entries. Use it
only for a config with exactly that layer order and compatible canonical oracle
layout. Actual layer/member counts and observed position must match. Zero-length
histories currently fail the harness's empty-evidence rule, not silently pass.
token_axis is only for capacity-backed token buffers, never recurrent/history
state. SnapshotLayerReader does no implicit omissions. Other families use their
own complete observed-state reader. Custom reader factories receive schema and
options; cache_reader and cache_reader_factory are mutually exclusive.

For Torch recurrent/MLA caches with explicit observable fields, use
cache_reader_factory `openagent.adaptive_engine.wholemodel_validation.cache_readers:create_observed_path_reader`.
cache_reader_options.leaves maps EVERY schema leaf to an observed selector:
source is cache, model, output or context; path is an ordered list of attribute
names/dictionary keys/integer indices. For example a declared recurrent leaf may
use source=cache, path=["layers", 0, "recurrent_state"] if that exact installed
family really exposes that field. MLA compressed latent and rotary state must
each have their own declared selector; no field name or representation is guessed.
Missing attributes fail. token_axis optionally slices occupied capacity; length
can specify a second observed source/path selector for an integer occupied count,
otherwise it uses appended-token position. axes transposes and batch_axis inserts
a batch dimension. Fixed recurrent/history state must not use token_axis. Methods
are never invoked implicitly; families needing API methods use callable readers.
The reader exports actual observed tensors, not fabricated equations or defaults.

## Authorized Local Transformers

CLI factory:

```text
openagent.adaptive_engine.wholemodel_validation.local_transformers:create_local_transformers_adapter
```

Options include metadata and the cache options above, plus:

- model_path: snapshot directory beneath offline_root.
- authorize_authoritative_local_files: must explicitly be true.
- implementation_source_pin: relative path of a source artifact pinned by role.
- trust_local_code: false by default; true also requires authorize_local_code=true.
- torch_dtype: auto, float32, float64, float16 or bfloat16.
- device: cpu by default; attn_implementation: eager by default.

The runner injects reserved _artifact_pins from the role's artifact list. Users
must not supply that option. Every file in the snapshot must be pinned, including
indexes, all shards, local code and tokenizer files. All safetensors shards and
their index must be classified as weights. config.json must be config. Pickle
weights are never selected. auto_map repository references are prohibited and
local targets must be pinned source files. Installed built-in Transformers model
source must also match implementation_source_pin: copy the authoritative source
into the offline root for pinning, rather than assuming an installed version.
The loaded model class's actual source bytes are checked, not just copied files.

Loading uses local_files_only, safetensors, temporary HF offline environment
flags and caller-authorized local code only. Missing real dependencies raise.
Pin/review transitive implementation sources and dependency versions separately;
class-source matching is not a transitive import sandbox. Authorized local code
can still perform arbitrary I/O; enforce external offline/network restrictions.

## Checkpoint Identity And Reports

Before either adapter is imported, runner rejects duplicate resolved artifact
paths (including aliases/different classifications) and invalid hashes. Complete
sorted config/weights/tokenizer digest lists must match between roles; sources
may differ. Selecting one common shard in metadata cannot hide different shards.
Each metadata digest must identify exactly one artifact of its declared kind.
The report records the complete checkpoint digest sets, plus declared metadata
and observed adapter runtime (Torch parameter dtype/device and class, native
executor class, session seed). Pin sets must be complete; generic factories are
trusted to use declared files. Local factory additionally checks actual snapshot
coverage. Fresh full-prefix sessions are closed immediately after export to avoid
retaining a complete model for every boundary.

Remaining gaps: execution/testing, family-specific Torch cache reader functions,
review of transitive pins/authority, accelerator environment provisioning and
subprocess isolation. These are explicit, not claimed complete by this code.

## Opt-In K3 Forward MLA View

`k3_projection:create_k3_projected_candidate_adapter` exports a declared expanded
comparison view from actual occupied normalized latent rows and immutable native
`kv_b_proj.weight`. Publisher head-major rows contain nonrotary key then value;
observed shared rotary coordinates are appended to each head's key. No inverse,
renormalization, source change or oracle change is performed. This is an explicit
exception to raw-export-only mapping, never silent expansion of an unknown ABI.

Use the publisher `vendor_cache_schema`, expanded `cache_layout`, and explicit
`projection_tolerance` equal to manifest cache tolerance. NumPy FP32 projection
is marked numerically uncertified. Complete raw native state, positions and
projection weights are mandatory separate Observation evidence and are retained
in the NPZ; unlike raw/expanded tensors are never cross-compared. Raw evidence is
also checked against same-role fresh prefixes, with exact weight comparisons.

The complete projected adapter accepts MLA-only schedules, publisher-none rotary
semantics and projection epsilon 1e-6. Hybrid KDA alignment remains unsupported
and fails rather than omitting state. Individual MLA projection is available as
`project_mla_cache`; this helper alone is not a whole-model comparison. Forward
projection cannot establish latent nullspace equivalence or certify reassociated
attention arithmetic. See `docs/ENGINE_WHOLEMODEL_VALIDATION.md` for the complete
policy and evidence contract. No validation execution was performed.
