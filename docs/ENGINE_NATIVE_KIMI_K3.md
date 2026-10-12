# Native Kimi K3 Investigation

Status: bounded dense NoPE-MLA text fixture and standalone AttnRes primitive
verified; full hybrid decoder and published checkpoints unsupported.
Verification date: 2026-10-03. These opt-in modules do not register a model
family, alter a loader, or grant K3 compatibility to the existing Linear engine.

## Authoritative Evidence

The official [model metadata](https://huggingface.co/api/models/moonshotai/Kimi-K3)
reported revision `f831ab66814297da540d832a5235f8e904f29d06`.
Its complete `siblings` inventory includes `configuration_kimi_k3.py`,
`modeling_kimi_k3.py`, `modeling_kimi_linear.py`, the tensor index and 96
weight shards. Reference sources therefore exist; an absent public oracle
must not be asserted. File presence is not proof of executable parity.

The official [configuration](https://huggingface.co/moonshotai/Kimi-K3/raw/f831ab66814297da540d832a5235f8e904f29d06/config.json)
identifies the outer model as `kimi_k3`, with architecture
`KimiK3ForConditionalGeneration`. Its text subconfig uses `kimi_linear` and
`KimiLinearForCausalLM`, but has K3-specific contracts. Relevant metadata:

| Contract | Published Value |
| --- | --- |
| Text geometry | 93 layers, hidden width 7168, vocabulary 163840 |
| Attention schedule | MLA at layers 4, 8, ..., 92 and 93; KDA elsewhere |
| KDA geometry | 96 heads, width 128, convolution kernel 4 |
| KDA extensions | Full-rank gate; decay lower bound -5 |
| MLA extensions | NoPE enabled; output gate enabled |
| Attention residuals | Block size 12 |
| Activation | SiTU; beta 4; linear beta 25 |
| Routed experts | 896 experts, 16 selected; latent width 3584 |
| Latent expert normalization | Enabled |
| Shared experts | 2 |
| Checkpoint quantization | Compressed-tensors MXFP4, group size 32 |

The [technical report, version 1](https://arxiv.org/html/2607.24653v1)
provides architecture sections on lower-bounded KDA decay, full-rank gates,
gated MLA, block Attention Residuals, normalized LatentMoE and SiTU-GLU.
These establish a distinct architecture, not permission to substitute the
existing Linear implementation. Scalar metadata alone does not settle exact
tensor layout, residual scheduling, gate placement or cache semantics.

## Earlier Import Blockers

The inspected local `.tmp-kimi-oracle` contains only
`configuration_kimi_linear.py` (6922 bytes) and `modeling_kimi_linear.py`
(42359 bytes). The existing Linear test oracle is not K3 evidence.

Read-only dependency inspection with
`.accelerator-venv/Scripts/python.exe` reported:

```text
torch: 2.14.1+cpu
transformers: 4.57.6
fla-core: absent
einops: absent
torch.cuda.is_available(): False
torch.cuda.device_count(): 0
```

Follow-up used the existing Transformers 5.16.1 overlay, not just the baseline.
`tests/probe_kimi_k3_oracle.py` downloads unchanged official sources to a
disposable temporary namespace package, checks these SHA-256 digests, and
attempts a real import without rewriting or stubbing anything:

```text
configuration_kimi_k3.py (11343 bytes)
735eb9ebe593e17d231e08e1df7f7be9b5ee0e079f511aa201f9572077b416ae
modeling_kimi_linear.py (51506 bytes)
9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a
```

The `--provision` option successfully installed `einops==0.8.1` and
`fla-core==0.5.2` with `--no-deps` into that temporary directory only.
No application dependencies or overlay files were changed. Actual outcomes:

| Environment | Exact Unchanged-Module Import Blocker |
| --- | --- |
| 5.16.1 overlay without provisioning | `ModuleNotFoundError: No module named 'einops'` |
| 5.16.1 overlay with provisioning | `ImportError: cannot import name 'OutputRecorder' from 'transformers.utils.generic'` at vendor text source line 44 |
| 4.57.6 baseline with provisioning | `ModuleNotFoundError: No module named 'triton'` in `fla/ops/abc/chunk.py:9`, reached through `fla.modules` at vendor text source line 47; vendor catches it and raises its install-FLA message |

`python -m pip index versions triton` on this Windows Python environment
returned `ERROR: No matching distribution found for triton`. This is an
observed platform/distribution blocker, not a universal CPU impossibility
claim. Provisioning did not silently install a different Torch or Transformers.

Inspection of the official text entry points found an eager PyTorch MLA
attention path, but KDA dispatches to `chunk_kda` or `fused_recurrent_kda` from
FLA and imports fused normalization/convolution modules. Selecting eager MLA
therefore does not bypass the KDA dependency. No supported unchanged CPU KDA
entry was established. An alternative compatible host/backend remains a
possible next investigation. Monkey-patching missing APIs, replacing kernels
with hand-written recurrence or treating MLA-only as full K3 is not acceptable
oracle evidence. These earlier failures are superseded for the bounded dense
MLA fixture below, not for the full hybrid model.

Reproduce the import probe (explicit network operation):

```powershell
$env:PYTHONPATH='D:/openagent/.tmp-native-decoder-oracle;D:/openagent'
.accelerator-venv/Scripts/python.exe tests/probe_kimi_k3_oracle.py --provision
```

## Original AttnRes Primitive

`openagent/adaptive_engine/native_kimi_k3_attnres.c` implements source
aggregation from the report's equations (8)-(10), independently of vendor
implementation bodies. For each row `v_i`, its key score is
`dot(query, gain * v_i / sqrt(mean(v_i^2) + epsilon))`; stabilized softmax
weights combine the **unnormalized** values. Omitted gain means unit gain;
an explicit gain is a caller-supplied key RMSNorm parameter, not inferred
checkpoint metadata. There is no extra query-width scale in equation (9).

The caller supplies the embedding, completed block sums and, when applicable,
the current block partial sum as distinct source rows, per equation (10).
This primitive does not implement block scheduling, projections, token
decoding, recurrent caches or checkpoint admission. Epsilon is explicit rather
than a guessed published-checkpoint constant. FP32 inputs/output use original
double-precision intermediate arithmetic, not vendor low-precision rounding.

The exported function `oa_kimi_k3_attnres` admits 1..128 source rows and
1..4096 channels, finite inputs/gain/query and positive finite epsilon.
It allocates no heap memory, stages results in bounded stack arrays (33792
bytes for the two main double arrays), and leaves output untouched on errors.
Direct C callers must supply correctly sized disjoint buffers; there is no
Python runtime wrapper or family export. Return codes: 0 success, 1 invalid
admission, 2 nonfinite input/intermediate.

`tests/test_native_kimi_k3_attnres.py` actually built a DLL using MSVC
`/W4 /WX /O2 /fp:precise /LD`: **6 tests passed, no skips**. Tests cover
uniform weights, one source, a closed-form tanh result, random independent
scalar analytical comparisons, permutation invariance, 128-row/4096-channel
bounds, zero-valued sources, extreme scores, block/partial-sum inputs and unchanged
output on invalid/nonfinite admission. The random comparison uses absolute
tolerance `5e-7`, not an official model-parity tolerance.

```powershell
.accelerator-venv/Scripts/python.exe -m unittest tests.test_native_kimi_k3_attnres -v
```

This is executable primitive evidence, not decoder logits/state/rollback
certification or published-checkpoint support.

## Bounded Dense MLA Fixture

`native_kimi_k3_decoder.c` and `native_kimi_k3_decoder.py` implement an
explicit dense text fixture: embeddings, low-rank query and KV projections,
expanded causal MLA, sigmoid output gating before output projection, SiTU-GLU,
block Attention Residuals, final residual aggregation, RMSNorm and LM logits.
`independent_kimi_k3_decoder.py` evaluates independently written NumPy math.
Neither implementation imports, copies or aliases the Linear engine or vendor
implementation. Vendor names identify the admitted tensor inventory only.

The fixture requires positive query/KV ranks, equal query/KV head counts,
NoPE, output gating, dense feed-forward layers and Attention Residuals. NoPE
leaves the additional shared key coordinates unrotated; their width may be zero.
Query/KV latent normalization uses epsilon `1e-6`; residual and decoder norms
use the explicit fixture epsilon. SiTU parameters are positive and explicit.

Bounds: hidden/query/KV ranks 1..64, layers 1..8, heads 1..4, NoPE/value widths
1..16, additional key width 0..16, intermediate width 1..128, residual block
size 1..8, vocabulary 1..256 and cache capacity 1..128. Batch size is one.
Weights must exactly match the named FP32 tensor inventory and be finite.
Unknown config fields and forward options, missing/extra tensors, invalid IDs,
nonfinite buffers and capacity overruns are rejected. This is not a general
Hugging Face config loader or published-checkpoint admission API.

Cache storage is expanded `[layer, capacity, head, channel]` K/V, not compressed
MLA state. AttnRes block sums are recomputed across depth for each token; they
are not persistent temporal state. C stages cache and logits in bounded heap
buffers and commits only on success. Return codes are 0 success, 1 admission,
2 nonfinite arithmetic, 3 explicit test fault, 4 capacity and 5 allocation.
Direct C callers must supply correctly sized, disjoint buffers. Detached
snapshots support reset and restoration with model-identity checks. This does
not certify thread-safe direct C buffer sharing or published-weight precision.

## Numerical Verification

The unchanged, hash-checked vendor text module now imports on this Windows
CPU host using Transformers `4.57.6`, `einops==0.8.1`, `fla-core==0.5.2` and
`triton-windows==3.8.0.post29`. Dependencies and vendor files are provisioned
only in a disposable test directory. A child process releases native DLLs
before the probe cleans up. No vendor source or kernel is rewritten or stubbed.
The vendor model's public `set_attn_implementation('eager')` selects the CPU
MLA backend after construction; KDA is not exercised.

The strict vendor fixture uses 5 layers, hidden width 12, 2 heads, query rank 5,
KV rank 6, NoPE width 3, additional key width 2, value width 4, intermediate
width 17, block size 2, vocabulary 23 and seven tokens. Deterministic finite
FP32 weights include nonuniform norm gains and residual queries.

Actual results:

- Local compiled suite: **13 tests passed, no skips** (7 decoder and 6 AttnRes).
- Explicit oracle suite: **8 tests passed, no skips** (7 decoder and 1 vendor).
  The repeated decoder tests are not additional unique coverage.
- MSVC compilation used `/W4 /WX /O2 /fp:precise /LD` for both native sources.
- C and NumPy full logits and all-layer K/V matched unchanged vendor execution
  at absolute/relative tolerance `2e-5`. Maximum full-logit absolute error was
  `2.2649765014648438e-06`.
- Vendor tokenwise and irregular `[2,1,4]` chunk logits and K/V matched the
  native path. Native full/tokenwise/chunk outputs and state were identical.
- Local tests covered block boundaries, zero additional key width, admission
  and capacity failures, detached snapshots, reset/restore and injected native
  mid-chunk failure preserving both caller logits and cache state.

```powershell
.accelerator-venv/Scripts/python.exe -m unittest tests.test_native_kimi_k3_decoder tests.test_native_kimi_k3_attnres -v
$env:PYTHONPATH='D:/openagent'
.accelerator-venv/Scripts/python.exe tests/probe_kimi_k3_oracle.py --provision-windows --verify-decoder
```

## Unsupported Contracts

The dense-MLA fixture does not cover hybrid K3. A separate
[bounded native hybrid path](ENGINE_NATIVE_KIMI_K3_HYBRID.md) now composes
lower-bounded KDA, full-rank gating, convolution/recurrent state and normalized
latent experts. Its tests establish explicit equation-fixture behavior, not
unchanged vendor full-hybrid or published-checkpoint parity. Multimodal inputs,
MXFP4 checkpoint admission, padding/batches, RoPE, training and speculative
decoding remain unsupported. The host has
no CUDA device/toolkit, and FLA reports an unsupported Triton platform; importing
it is not evidence of executable KDA kernels. No unchanged full hybrid numerical
oracle, KDA cache comparison or published-checkpoint logits were obtained.

The earlier overlay `OutputRecorder` incompatibility still excludes that tested
Transformers 5.16.1 environment. Neither this fixture nor its passing tests imply
full decoder conformance, shared runtime integration or complete K3 support.
