# Native Inkling Text Decoder

## Scope

Implemented independently designed resident FP32 C inference for the exact
`inkling_text` architecture from Transformers 5.16.1. This is a text-only
decoder, not an alias to another family or the multimodal `inkling_mm_model`.
The initial investigation found a working oracle; implementation subsequently
used the official Apache-2.0 modeling implementation to learn the equations and
cache contract, as explicitly authorized. No model implementation was copied or
line-by-line translated. Numerical execution uses a new serial token-stream
algorithm and its own time-major FIR / token-major KV representation.

Owned files:

- `openagent/adaptive_engine/independent_inkling.py`: configuration and tensor
  validation, state ownership, independent NumPy diagnostic baseline.
- `openagent/adaptive_engine/native_inkling.py`: ctypes binding, resident budget,
  serialization, and transactional commit.
- `openagent/adaptive_engine/native_inkling.c`: all native forward math,
  private ABI version 2.
- `tests/test_native_inkling.py`: compilation and pinned black-box oracle tests.
- This document.

The decoder is exported by `openagent.adaptive_engine` and available through the
[native text checkpoint loader](ENGINE_NATIVE_CHECKPOINTS.md). Source packaging
includes its C translation unit. No production checkpoint, quantization,
streamed weights, GPU execution, or throughput certification is claimed.

## Supported Contract

- Explicit full (`hybrid`) and sliding (`hybrid_sliding`) attention schedules,
  with distinct full/sliding head geometry and grouped-query attention.
- Explicit dense, sparse, or mixed MLP schedules; SiLU dense gating and trained
  dense global scale.
- Routed experts with sigmoid/correction-bias top-k selection; selected routed
  and shared expert weights normalized jointly from unbiased log-sigmoid logits.
  Route scale and router global scale are applied. Routed experts are accumulated
  in expert-index order; shared expert gamma scales the intermediate activation.
- Embedding RMS normalization, per-head Q/K normalization, both decoder
  pre-norms, final normalization, and muP head scaling.
- Query-conditioned relative-position profiles, zero outside the backward extent,
  full-attention logarithmic scaling, and an optional unpadded vocabulary head.
- Four independent depthwise causal FIR histories per layer: K, V, attention
  output, and MLP output. Each filtered output adds its unfiltered input.
- Contiguous implicit positions, no masks, FP32 finite unquantized named weights,
  independent embedding and output weights, and a fixed cached batch until reset.
- Prefill, single-token decode, arbitrary chunk boundaries, reset, diagnostic
  cache snapshots, routing-weight diagnostics, and isolated cache-free calls.

Full attention scores for head dimension D are the normalized Q/K dot product
divided by D, plus the projected relative profile for the backward distance.
For full layers with a scaling floor, both normalized Q and relative bias are
multiplied by `1 + alpha * log(max((position+1)/floor, 1))`.
Sliding layers omit this logarithmic scaling and attend only to distances smaller
than the window. Their profile extent equals the window.

The cache retains the complete bounded KV history even for sliding layers; this
differs physically from the oracle's cropped sliding cache, but has the same
supported attention range and numerical results. FIR histories store the last K
unfiltered samples, oldest first, initially zero. FIR coefficients are indexed
oldest-to-newest. Processing tokens serially makes the numerical path independent
of call chunking.

## Rejection Boundaries

Only the exact identity `inkling_text` and an absent or exact
`InklingForCausalLM` architecture declaration are accepted. Unsupported config
keys are rejected, including unknown logits settings, rotary settings, bias
options, quantization settings, and spelling aliases. Tensor names, shapes,
dtype, and finiteness must match exactly; extra or missing tensors reject.

Nondefault `local_layer_ids` rejects: callers must supply an explicit
`layer_types` schedule and leave `local_layer_ids=None`. MTP configurations,
tied weights, non-SiLU activations, nonzero dropout, non-FP32 dtype declarations,
encoder-decoder mode, and feed-forward chunking reject. Only default
cache/output config modes are accepted: `use_cache=True`, `return_dict=True`,
`output_attentions=False`, and `output_hidden_states=False`. The explicit native
return object provides final hidden states and routing diagnostics regardless.
Only default `shared_expert_sink=True` and `rms_norm_eps_moe_gate=1e-6` are accepted; the
5.16.1 text oracle does not implement alternative gate-epsilon/sink behaviors.

Positive shared/routed expert counts, positive route scale, and a valid top-k
are required. If selection has an exact FP32 tie across the top-k cutoff, the
whole call rejects transactionally instead of pretending to match an unspecified
backend-dependent tie order. Ties wholly inside a fully selected set do not
change expert membership and are allowed.

Sliding window 1 rejects. The pinned eager oracle's full-sequence and cached
single-token paths disagree at this boundary. The test suite explicitly retains
an oracle-only reproduction and a config rejection test, rather than weakening
native parity tolerances. Windows 2 and 3 cross-boundary parity are qualified.
This is a concrete unsupported oracle boundary, not a missing text oracle.

Resident limits include 128 layers, 128 batch rows, context capacity 65536,
convolution kernel length 64, and bounded dimensions validated in both Python
and C. Grouped-head products and head-relative products are at most 65536.
Other integer settings are bounded by 1048576. More restrictive per-instance
`max_bytes` checks account conservatively for resident weights, old and pending
cache, outputs, input conversion, and working buffers before state allocation.
Caller-owned inputs and explicitly requested snapshots are outside that budget.

## Transactions and API

`NativeInklingDecoder(config, tensors, library=path_or_cdll, max_bytes=...)`
requires a previously compiled library; importing the module does not compile,
install dependencies, or load Transformers. The NumPy diagnostic implementation
accepts the same config/weight contract without a library.

`forward(token_ids)` accepts integer `[tokens]` or `[batch,tokens]` arrays and
returns FP32 `logits`, `last_hidden_state`, and
`routing_weights[batch,tokens,layers,routed+shared]` (zeros for dense layers).
By default the call advances this decoder's owned cache. `use_cache=False`
instead evaluates from empty KV and all four empty FIR histories at position
zero, leaving the committed cache and position unchanged. It may use a different
batch size from the committed session and is bounded against its own token
length, not the session's remaining capacity. Only actual Python booleans are
accepted. Explicit attention masks or positions still reject.

Each call validates the complete input and budget, copies all previous cache
into staging storage, and invokes C only with staging pointers. C validates
geometry and all tensor counts before writing, then performs all embedding,
attention, FIR, MLP/MoE, normalization, and head math. Python publishes the staged
cache and new position only after success and finite-output checks. Invalid
inputs, capacity/budget failures, exceptions, nonfinite computation, and nonzero
C statuses leave prior committed state unchanged. Allocation failure returns
without committing. A per-decoder reentrant lock serializes calls, snapshots,
and reset; C itself uses no mutable global state.
Cache-free calls stage zeroed state without temporarily changing session fields;
the same native C transition runs with start position zero and its staged state
is discarded on success or failure. The budget accounts for the actual old
cache footprint even when the cache-free batch size differs. Non-finite RMS
sums reject rather than being hidden by a zero inverse-normalization scale.

Weights are owned immutable byte-backed copies; snapshots are detached copies.
The private C ABI assumes valid pointer storage and caller-owned staging buffers.
It is not a public arbitrary-pointer or in-place transaction API.

## Qualification

Oracle: installed `transformers==5.16.1` at
`D:/openagent/.tmp-native-decoder-oracle`, Torch `2.14.1+cpu`, CPU float32,
eager attention and eager experts. The eager expert selection is necessary
because the default grouped-mm CPU backend rejects tiny unaligned strides.
Models use deliberately amplified random weights, non-unit normalization
weights, and trained scales to make architecture mistakes observable.

PowerShell command:

```powershell
$env:PYTHONPATH='D:/openagent/.tmp-native-decoder-oracle'
$env:INKLING_CERTIFY='1'
.accelerator-venv/Scripts/python.exe -m unittest tests.test_native_inkling -v
```

The suite builds the C library in a temporary directory with MSVC
`/W4 /WX /O2 /fp:strict /LD`. Its portable compiler recipe uses C99,
`-Wall -Wextra -Werror -O2 -ffp-contract=off -shared -fPIC` and libm.
Only Windows/MSVC execution is locally qualified; the portable recipe is not a
claim of a tested Linux build. Missing compiler/oracle dependencies cause
explicit skips, not fabricated parity results.
With `INKLING_CERTIFY=1`, missing compiler/oracle dependencies or a version other
than 5.16.1 are errors, not successful certification.

Tests cover dense/full and dense/sliding, all-sparse and mixed layers, multiple
shared experts, top-k 1/2/all, different full/sliding head dimensions, batch 2,
relative-extent and sliding-window boundaries, logarithmic scaling, unpadded
heads, and kernel length 1. Full calls, single-token decode, and mixed chunks
are checked against the black-box oracle, including KV contents, all four FIR
histories, and selected/shared routing weights. Additional tests cover identity
and semantic-config rejection, malformed weights/tokens, budget/capacity checks,
post-computation backend failure, real nonfinite C failure, cutoff ties,
ownership/reset, direct C descriptor validation, concurrency, and a guard
against NumPy or Transformers forward delegation.

Local result: **18 tests passed**, with actual MSVC C compilation and all oracle
comparisons executed (no skips).
The additional fixtures compare cache-free dense/MoE and full/sliding execution
against the official oracle, including output selection, changed batch size,
and calls exceeding the committed session's remaining capacity. They check
unchanged session snapshots, subsequent continuation, invalid booleans,
pre-execution budget rejection, post-computation failure, disabled fallback and
late-token normalization overflow rollback. Existing tolerances are unchanged.

Logits/routing use `atol=3e-6, rtol=2e-5`; hidden/cache use
`atol=5e-6, rtol=2e-5`. Exact architecture does not mean bitwise identical
floating-point reductions. No tolerances were relaxed to hide the window-1
oracle discrepancy.

### Output Selection

`forward(..., logits_to_keep=0)` returns every logit row. Positive integers
select the final rows (values larger than the call length retain all rows).
One-dimensional integer arrays select call-local rows, including repeated or
empty selections. Negative, boolean, floating, multidimensional, and out-of-range
selectors reject before execution. Index arrays longer than the current token
call also reject to preserve bounded output admission; this is a stricter native
boundary than the oracle. All input tokens still update KV/FIR state;
hidden states and routing diagnostics remain full-length. This output contract
is compared directly to 5.16.1 in both prefill and cached calls, including caches.
The current implementation computes the full head before selecting rows, so
this is not a reduced-head-compute or reduced-memory claim.

References for the learned architecture:
[official Inkling documentation](https://huggingface.co/docs/transformers/model_doc/inkling)
and the installed 5.16.1 configuration/modeling files. No runtime dependency on
those model files exists in either decoder.
