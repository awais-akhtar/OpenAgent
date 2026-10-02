# Independent Native Kimi Linear

## Delivered Surface

`openagent.adaptive_engine.native_kimi.NativeKimiLinearDecoder` performs actual
batch-one token-ID-to-logit inference, not just projected KDA. It accepts exact
checkpoint tensor names and explicit configuration. `from_checkpoint` loads
local single-file or indexed/sharded SafeTensors, including explicit F16/BF16
to resident FP32 conversion. No remote code is executed or downloaded.

The resident subset includes embeddings, input/post-attention/final RMSNorm,
KDA projections and caches, expanded causal MLA and caches, dense MLP, routed
experts, shared experts, residual addition and the LM head. Its arithmetic
uses original C kernels; Python schedules operations and owns buffers.

`NativeKimiKDA` is also independently usable as a projected token-mixing
primitive. `IndependentKimiLinearDecoder` and the dense `kda_equations`
function are original NumPy equation checks, not external oracles.

All work is confined to the five requested files. Shared exports, decoder
registries, conformance and packaging have not been extended. Import the new
module explicitly. There is no upstream Segment ABI claim, fallback to
Transformers, or automatic compilation.

## Architecture Boundary

The admitted identity is `model_type: kimi_linear` with
`architectures: [KimiLinearForCausalLM]`, using the official Linear contract:
SiLU, direct MLA query projection (`q_lora_rank: null`), NoPE MLA, low-rank
KDA forget/output gates, sigmoid routing and one expert group. Layer lists
are explicit, one-based and must partition every decoder layer exactly.

This is **not Kimi K3 support**. The local Colibri K3 fixture also labels
itself `kimi_linear` but has SiTU, AttnRes, latent-MoE normalization, full-rank
gating, bounded decay and MLA output gating. These extensions are rejected,
not ignored or interpreted as Linear. Wrapper configurations and
`model_type: kimi_k3` are rejected too.

Other deliberate exclusions: quantized tensors, multiple router groups,
low-rank MLA Q, RoPE MLA, speculative/MTP layers, multimodal wrappers,
batching, attention masks/padding, beam search, expert offload, GPU execution
and production-size weight streaming. API inputs are a nonempty 1-D integer
token sequence, and outputs are `[tokens, vocabulary]` FP32 logits.

## Derivation And Tensor Contracts

For key-by-value state `S`, channel decay is applied first:

```text
D = diag(exp(log_decay)) S
r = beta (v - D^T k)
S' = D + k r^T
o = S'^T q
```

This follows directly by expanding `(I - beta k k^T) diag(alpha) S + beta k v^T`.
The C implementation evaluates the rank-one correction; the NumPy equation
check forms the dense transition matrix. Diagonal decay and the correction
must not be exchanged. See equation (1) of the
[Kimi Linear paper](https://arxiv.org/html/2510.26692v1).

The public primitive's input streams are pre-convolution Q/K/V projections.
Taps are oldest-to-newest `[heads, channels, kernel]`; histories store only
the preceding `kernel-1` raw projections. The primitive applies causal
depthwise convolution, SiLU, Q/K L2 normalization, query scaling, recurrence,
head RMS normalization and a probability output gate. It receives beta/gate
after sigmoid and decay as natural log(alpha), not raw gate logits.

The decoder generates those inputs internally. Checkpoint convolutions have
`[heads*dim, 1, kernel]` shape; `A_log` is `[1,1,heads,1]`, `dt_bias` is
`[heads*dim]`. The forget gate is `-exp(A_log)*softplus(f_b(f_a(x))+dt_bias)`.
Q/K L2 epsilon is `1e-6`; query scale is `dim^-0.5`. Output RMS epsilon comes
from config. MLA latent RMS uses `1e-6`; its shared channels are concatenated
without rotary encoding, and attention scale uses the entire query width.
These contracts were checked against the
[official Linear model](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base/blob/cddba2d/modeling_kimi.py)
and [official config](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base/blob/dd9c549f213e3a1dd4559bf2949fff4e803d62d0/config.json).

Expert matrices retain `block_sparse_moe.experts.N.w1/w2/w3.weight` names;
shared experts retain `gate_proj/up_proj/down_proj.weight`. Routing tracks
the published model's observable score-view update: correction bias is
added before selection **and gathering the selected weights**. Selected
weights are renormalized only for top-k > 1 when configured, then scaled.
This is intentionally not an assumed uncorrected-weight routing rule.
Exact score ties use lowest index locally; vendor tie ordering is not
certified. Single-group admission avoids approximating grouped selection.

There has been no source copying, line-by-line translation or production
import from Colibri/vendor code. Mathematical contracts informed original
buffer-oriented kernels. Local K3 config/provenance were inspected only to
establish the architectural mismatch.

## Transactions And Bounds

Each decoder `forward` call copies every KDA and MLA cache into staging.
Token processing and the head complete before position/caches commit.
Exceptions, nonfinite results, native failures, capacity errors and memory
admission errors leave all live caches unchanged. Calls are serialized by
a reentrant lock. Snapshots are detached copies; reset clears all caches.
Checkpoint weights are copied and marked read-only.

`max_bytes` admits FP32 weights, cache storage, transaction copies, output
rows and conservative native numerical workspace. Loader conversion is
admitted separately before payload reads. This is a numerical-buffer bound,
not a total process RSS promise: Python dictionaries, input caller storage,
DLL code and JSON parser objects are outside it. Caller-built libraries and
direct C callers must supply valid, correctly sized, disjoint buffers.

Decoder limits include 4096 tokens per call, 65536 cache positions, 128 layers,
4096 hidden channels, 128 heads and 256 KDA channels. Full MLA query width
is bounded by 512 and value width by 256. Individual tensors may not exceed
67,108,864 elements. Default resident admission is 256 MiB. The 48B checkpoint
cannot fit this default and is not claimed to run on this machine; scaled
checkpoints using the same admitted architecture are executable.

## Build And Run

Use a caller-built shared library. On POSIX:

```sh
cc -std=c99 -O2 -Wall -Wextra -Werror -shared -fPIC \
  openagent/adaptive_engine/native_kimi.c -lm -o kimi.so
python -m unittest tests.test_native_kimi -v
```

On Windows the test suite locates MSVC through `vswhere` and compiles with
`/W4 /WX /O2 /fp:precise /LD` in a temporary directory. It fails on compile
errors; lack of a compiler produces an explicit test skip.

```python
from openagent.adaptive_engine.native_kimi import NativeKimiLinearDecoder

decoder = NativeKimiLinearDecoder.from_checkpoint(
    "D:/models/tiny-kimi-linear", library="D:/build/kimi.dll",
    capacity=128, max_bytes=256 * 1024 * 1024,
)
logits = decoder.forward([3, 7, 2])
next_token = int(logits[-1].argmax())
next_logits = decoder.forward([next_token])
```

## Verification And Remaining Prerequisite

Parent-facing classification: **executable bounded Kimi Linear resident
subset, in-house equation-tested, externally unverified**. There are zero
whole-model external-oracle comparisons for either Kimi Linear or Kimi K3.
Do not mark this as vendor-conformant, full production-checkpoint support,
or any K3 support. Explicit loader integration must select this exact
constructor only after validating its admitted Linear config; no family alias.

| Evidence | Result | Independence |
| --- | --- | --- |
| C full hybrid decoder logits and layer caches | Pass on scaled fixtures | Original NumPy baseline; shared orchestration/contracts |
| C KDA recurrence | Pass versus dense transition equations | In-house algebraic check, not vendor execution |
| Cached native replay, loading and rollback | Pass | Self-consistency and transactional tests |
| Official Linear runtime, whole model | Not executed | Missing compatible FLA runtime/accelerator |
| Official K3 runtime, whole model | Not executed | K3 not implemented; stored fixture outputs not used |

Executed on this host:

```text
.venv/Scripts/python.exe -m unittest tests.test_native_kimi -v
20 tests passed; actual MSVC C compilation, no skipped tests.
```

Base and optional-environment regression runs also completed:

```text
.venv/Scripts/python.exe -m unittest tests.test_native_kimi tests.test_native_checkpoint tests.test_engine_public_surface tests.test_engine_source_packaging -v
.accelerator-venv/Scripts/python.exe -m unittest tests.test_native_kimi tests.test_native_checkpoint tests.test_engine_public_surface tests.test_engine_source_packaging -v
Each: 37 run, 36 passed, 1 skipped, 0 failures.
```

The skipped test is the existing exact GLM-MoE-DSA external-oracle checkpoint
test, not a Kimi test: the base environment lacks the optional oracle; the
accelerator environment lacks that exact model class. These runs do not
increase the amount of Kimi external-oracle evidence.

Coverage includes dense-equation KDA checks, unequal primitive key/value
widths, kernel sizes 1/4/8, full hybrid decoder logits and every cache,
multiple random seeds, greedy argmax agreement, bitwise chunked/single-token
native replay, local sharded F32/F16/BF16 loading, tied embeddings,
shared-channel MLA attention, routing bias/scaling, late head failure,
real C overflow rollback, invalid IDs/geometry/dtypes, capacity and budgets.

**No independent vendor-runtime parity result is claimed.** The default
venv has no Transformers. The accelerator venv has Transformers 4.57.6,
but neither `kimi_linear` nor `kimi_k3` is in its AutoConfig mapping.
The isolated Transformers 5.16.1 source tree has Kimi K2.5, not these
identities. Accelerator Torch is present, but FLA and einops are absent
and `torch.cuda.is_available()` is false. The official model imports FLA
and selects GPU kernels. No prebuilt Colibri executable or cached vendor
model source was found. The K3 fixture's stored vendor results are not
a live oracle for Linear.

Next verification requires a separately provisioned, pinned official Linear
runtime (including its FLA dependencies and compatible accelerator) that
accepts the same scaled config and weights, then exports logits and all
layer caches. It must run in another process, never become a production
dependency. Compare finite FP32 results with stated tolerances, not pretend
bitwise equality across different reduction kernels. The original NumPy
decoder and C decoder share orchestration and tensor validation, so their
agreement alone does not certify a missed tensor/config contract. K3 still
requires a separate architectural implementation and independent oracle.
