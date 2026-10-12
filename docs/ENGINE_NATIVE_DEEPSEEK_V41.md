# DeepSeek V4.1 Native Decoder Investigation

Checked 2026-10-03. **Full decoder not implemented or certified.** No decoder C ABI,
decoder class, checkpoint acceptance, or model runtime support is provided by this
implementation. V4 must not be substituted. Original C ownership and raw CED
projection primitives are included, alongside a separate
[local E2M1/E4M3 codec](ENGINE_NATIVE_DEEPSEEK_V41_QUANT.md). The codec's explicit
packing and scale policy do not establish publisher checkpoint compatibility.

## Sources And Provenance

- [Publisher model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
  and [config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json)
  were accessible. Config declares outer `deepseek_v41`, nested
  `deepseek_v41_text`, and `DeepseekV41ForCausalLM`.
- [Publisher metadata API](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash)
  returned revision `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`, last modified
  `2026-10-01T05:56:24.000Z`. This records the observed repository head, not a
  downloaded, hash-verified oracle. Main-page reads were not immutable snapshots.
- [Technical report, version 1](https://arxiv.org/html/2609.19969v1), sections
  2.2 through 2.4, was accessible. Architecture information is available;
  claiming the specification does not exist would be incorrect.
- [Publisher inference README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/README.md)
  describes converted tensor-parallel checkpoints and dense-FP8/MoE-FP4
  kernels. Its small self-test uses uninitialized weights and explicitly checks
  shapes/kernel plumbing rather than numerics. It cannot establish parity.
- [Transformers PR 48721](https://github.com/huggingface/transformers/pull/48721)
  was closed, not merged. The contributor branch and its validation claims are
  not an accepted official Transformers oracle. No branch code was imported.
- Fetching the upstream main V4.1 modeling URL and the pinned publisher
  `inference/requirements.txt` URL through the web tool returned internal access
  errors. These are inconclusive transport failures, not proof of absent files.

No publisher inference implementation source was downloaded, copied, translated, patched,
or executed. Search surfaced a community PR Engram source snippet; that snippet
was not used as an architectural contract. Sources are evidence only, not
runtime dependencies.

## Architectural Differences To Preserve

The report describes a 20-layer causal encoder followed by a 20-layer decoder.
Decoder global KV comes from final encoder states; local SWA remains per-layer.
CSA2 eliminates CSA overlap and absolute compressor position embeddings, and
projects index keys from main KV. Full publishes KV, index keys and selection;
Reindex rescores shared keys with its own query; Reuse consumes the latest
selection for those same keys. Decoder candidates use block-maximum scores,
then deeper indexers search only the selected blocks.

Single-Pass mHC consumes the previous block's input-mixing coefficients, not
the current block's coefficients. Engram uses compressed tokens, multi-head
hashing, context-aware gating and multi-branch integration, without causal
convolution. Main KV quantization follows RoPE, using E2M1 values and an E4M3
scale per 16 channels; SWA stays FP8. These are observable numerical contracts,
not optional storage labels. [Report sections 2.2-2.4](https://arxiv.org/html/2609.19969v1)

The observed config separately specifies KV sources `[2,8,14,20]`, index
sources `[2,8,14,20,24,28,32,36]`, candidate source `20`, candidate block size
`8`, candidate block count `2048`, and Engram layers `[1,14]`. It also specifies
compressed vocabulary size `99092`, orders through `4`, eight hash heads and
padding ID `2`. These metadata values alone do not establish hash multipliers,
token normalization, prime-table allocation, tensor layouts or rounding rules.
[Publisher config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json)

The current V4 Python wrapper, V4 certification harness and V4 design document
were inspected. V4 uses its own layer-local overlap compressor and same-block
mHC contract; its native wrapper delegates validation to the V4 independent
decoder. Neither is a V4.1 oracle or a suitable identity alias. They remain
untouched.

## Exact Local Evidence

With `PYTHONPATH=D:/openagent/.tmp-native-decoder-oracle` and
`.accelerator-venv/Scripts/python.exe`:

```text
Transformers: 5.16.1
Source: D:/openagent/.tmp-native-decoder-oracle/transformers/__init__.py
CONFIG_MAPPING['deepseek_v41']: absent
CONFIG_MAPPING['deepseek_v41_text']: absent
transformers.models.deepseek_v41: absent
Torch: 2.14.1+cpu
CUDA available: False; device count: 0
Triton: absent
TileLang: absent
```

There is no verified executable exact V4.1 oracle in this environment. The
publisher's quantized kernel path has not been executed here. No claim is made
that an oracle cannot be run elsewhere or that a CPU port is impossible.
Writing a replacement CPU oracle would itself need independent verification;
removing activation/cache quantization to obtain an FP32 oracle could change
the very semantics being certified.

## Verification And Remaining Gate

### Compiled CSA2 Ownership Primitive

`openagent/adaptive_engine/native_deepseek_v41_csa2.c` implements only publication
ownership from report section 2.3.1. It is not a decoder, compressor, index scorer
or quantization implementation. No V4 imports, aliases or reference code are used.
Caller-provided FP32 buffers are opaque payloads, not a claim of FP32 V4.1 parity.

Each zero-initialized/reset state owns one query's publication chain. Mode `0`
(Full) copies main KV and corresponding index keys and selection; mode `1`
(Reindex) replaces only selection; mode `2` (Reuse) retains all payloads. Both
consumers must name the current Full owner. Layer IDs strictly increase until
reset. New Full invalidates old selection and ownership. Selector provenance is
separate from the last consuming layer. Ownership means source-consumer identity
and lifetime, not zero-copy memory aliasing; copies deliberately isolate callers.

The private ABI bounds rows to 32, KV/index widths to 8 and selection to 8 unique
in-range entries. Counts must match geometry exactly. Finite Full payloads are
required. Validation completes before committing state or writing read outputs;
all error paths leave buffers unchanged. Errors: `1` invalid buffer/geometry,
`2` invalid ownership/lifecycle, `3` nonfinite payload. Use the size query and
reset API, never construct serialized state. The C caller remains responsible
for truthful buffer lengths, nonoverlapping read outputs, valid live pointers
and exclusive access. Concurrent/multi-query use requires separate states.

`tests/test_native_deepseek_v41_csa2.py` compiles this file independently with
MSVC `/W4 /WX /O2` or C99 `-Wall -Wextra -Werror`. Five tests cover analytical
NumPy selection fixtures, Full/Reindex/Reuse payload/provenance, immutable input
copies, owner replacement, byte-for-byte failure rollback, invalid dimensions,
duplicate/out-of-range selections, output bounds, state canaries and reset.
NumPy ranks a hand-defined score vector only to supply a selection: it does not
purport to implement the official indexer's scoring equation. Windows compiled
run: five tests passed. No official oracle or sanitizer validation is implied.

```powershell
.accelerator-venv/Scripts/python.exe tests/test_native_deepseek_v41_csa2.py -v
```

This primitive has no token append/chunking API and no logits. Full/chunked
decoder parity remains untested. Compression visibility, causal masking,
hierarchical candidates and cross-call cache append/eviction are deliberately
outside its contract. The implementation establishes a bounded, tested native
ownership building block, not the entire shared CSA2 cache.

`tests/test_deepseek_v41_oracle.py` prints installed prerequisite evidence and
checks that the probe never certifies numerics or imports the application/V4.
Those are evidence-tool tests, **not decoder tests**. No compiled full-prefix,
chunked, cache boundary, rollback or numerical parity test has passed for V4.1.
No placeholder decoder or fake C implementation was added.

Before full decoder implementation, isolate an immutable publisher oracle plus all kernel
dependencies in tests, initialize deterministic bounded weights, and demonstrate
finite full/chunked outputs. Resolve exact Engram addressing/gating, compressor
visibility, shared-state ownership, candidate ordering/ties, initial/final mHC
coefficients and all output-visible quantization. Then implement original C
arithmetic and compare logits plus component/state traces, including failed-call
rollback of Engram history, shared KV/index selections and delayed mHC state.
An unavailable oracle must fail certification, never count as a parity pass.

## Bounded CED Arithmetic Follow-Up

Rechecked 2026-10-03: the local pinned Transformers 5.16.1 probe still reports
both V4.1 identities and modeling package absent, CPU-only Torch 2.14.1, and
no Triton or TileLang. PR 48721 remains closed and unmerged. Re-fetches of the
immutable publisher inference README/requirements and upstream modeling URL
returned internal transport errors, not verified absence. No unchanged exact
runnable publisher oracle has been obtained or executed.

The same private C source now also implements `oa_v41_ced_project`, an original
bounded implementation of [report equation (1)](https://arxiv.org/html/2609.19969v1#S2.SS2):
`C = H_encoder W_KV` and `Z = H_encoder W_Z`. All arrays are contiguous row-major
FP32; rows are 1..32, hidden/output widths are independently 1..8, and lengths
must match exactly. Products accumulate in FP64 before FP32 output conversion.
This is a deliberate equation-test arithmetic contract, not publisher kernel
rounding, compression, indexer K generation, or quantized inference parity.
Z is a compression-weight projection, not the indexer-key tensor.

Both outputs are staged before either is written. Invalid geometry/pointers
return 1; nonfinite inputs or an output outside finite FP32 range return 3.
Failures preserve both outputs. Callers provide valid aligned float pointers,
truthful lengths, and nonoverlapping output regions. Inputs may alias outputs
because all computation precedes publication. There is no hidden allocation,
checkpoint loading, shared export, V4 dependency, or reference implementation
translation. The original ownership ABI is unchanged.

Eleven independently compiled tests cover the original five lifecycle
fixtures plus NumPy FP64 matrix-product comparisons at minimum, asymmetric,
and maximum geometry; exact full-versus-single-row chunk consistency; late
second-projection overflow/nonfinite rollback of both outputs; and projected
opaque-payload publication followed by Reindex/Reuse and failed-call rollback.
Additional checks reject malformed cached geometry, nonfinite state, invalid
or duplicate selections, overlapping projection outputs, and read outputs
overlapping one another or the state buffer, without partial publication.
Windows MSVC `/W4 /WX /O2` run: eleven compiled tests passed; the isolated
prerequisite probe's two evidence tests also passed. No sanitizer run is claimed.
The latter deliberately stores Z as an opaque test payload and does not assert
that Z is a valid indexer K. These are mathematical equation and state tests,
not an official model oracle or full/chunked decoder certification.

Exact quantization and Engram remain unimplemented. Report section 2.4.4
specifies main KV E2M1 with one E4M3 scale per 16 channels, no second global
scale, applied after RoPE; indexer MXFP4 and SWA FP8 are distinct contracts.
That format description does not independently settle byte/nibble ordering,
scale construction, ties, saturation, or fused kernel rounding. Section 2.4.2
describes Engram orders 2/3/4, eight hash heads, FP8 embeddings/projections and
omission of the causal convolution, but does not establish an executable exact
tokenizer-compression/hash/gating/checkpoint-layout contract here. Guessing
these would invalidate verification. Full decoder logits, cache append/eviction,
publisher-exact hierarchical index arithmetic and delayed mHC remain blocked
on exact evidence. A separate [bounded index equation module](ENGINE_NATIVE_DEEPSEEK_V41_INDEX.md)
implements local scoring, block-maximum candidates and restricted deeper
selection with explicit tie and admission policies. These policies are not
publisher-compatibility claims.
