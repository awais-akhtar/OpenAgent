# Bounded Native DeepSeek V4.1 Engram

## Source Verification

Verified before implementation on 2026-10-03:

- [DeepSeek-V4.1-Flash report, section 2.4.2](https://arxiv.org/html/2609.19969v1#S2.SS4.SSS2): Engram retains compression, multi-head hashing, contextual gating and multi-branch integration, but removes causal convolution. Orders are 2/3/4, with eight heads and 2048 embedding dimensions per order. Tables use distinct prime sizes around 16M rows; embeddings and projections use FP8. Two modules are placed at zero-indexed layers 1 and 14. Sinkhorn-balanced momentum updates concern training, not this inference component.
- [Original Engram paper, sections 2.2-2.4](https://arxiv.org/html/2601.07372v1#S2): compressed-token suffixes feed multiplicative-XOR hashing; retrieved head vectors are concatenated. Query/key RMS normalization precedes a sigmoid dot-product gate. Multi-branch integration shares the retrieved memory and value projection, while key projections and gates are branch-specific.

This implementation was written from the paper equations. No vendor source was
copied or imported. The report does not establish the exact hash constants,
token compression map, padding convention, table primes, tensor packing or
FP8 scales needed for checkpoint parity. Those are not invented here.

## Supported Contract

The native C source and explicit Python binding live under
`openagent/adaptive_engine`. The class is available through the package's lazy
exports, and its compiled tests are included in native-engine CI. There is no
publisher decoder registration, V4 alias or automatic library compilation.
CED composition remains separate.

The private ABI version is 1 and requires a 64-bit host. The C source is the
canonical definition of `oa_engram_head`, `oa_engram_config`, and
`oa_engram_state`; the Python structures mirror them. A caller supplies valid,
properly aligned allocations covering declared counts. C cannot validate an
arbitrary foreign pointer's actual allocation length. No concurrent mutation
of configuration, weights, state or input buffers is permitted during a call.

- All orders 2/3/4 must be represented; 1-24 total heads, configured individually.
  A head has order, width 1-256, rows 1-33554432, uint64 seed, four uint64
  multipliers, and a const row-major FP32 table with exact element count.
  Only the first `order` multipliers participate. Primality and unique primes
  are publisher/model configuration responsibilities, not imposed assumptions.
- Inputs are explicit uint32 **compressed-token IDs**, not raw tokenizer IDs.
  Vocabulary size and a valid padding ID are mandatory. No identity compression
  or textual normalization is silently assumed. Missing left context uses the
  supplied pad ID and still performs lookups for every order.
- For oldest-to-newest suffix IDs `g[j]`, the selected configured hash is
  `seed XOR ((g[0]*multiplier[0]) mod 2^64) XOR ...`, reduced modulo head rows.
  This is a documented selectable configuration contract, not a claim that
  these constants or this exact hash variant match publisher checkpoints.
- Head descriptor order defines concatenation; memory width is its sum, at most
  6144. The caller can configure 8 heads of width 256 for each of 3 orders.
- Hidden width is 1-256 and branch count 1-8. Key weights are row-major
  `[branches, dim, memory]`; shared value weights are `[dim, memory]`.
  Query/key RMS scales are explicitly supplied `[branches, dim]`. Epsilon must
  be positive and finite. There are no projection biases or query projections.
- Outputs are gated **deltas**, `[tokens, branches, dim]`, plus row indices
  `[tokens, heads]`. The decoder adds deltas to residuals and handles mHC placement.
  There is no causal convolution, SiLU, attention, MoE or CED inside this module.

For each token and branch:

```text
e = concatenate(table[configured_hash(suffix)] for head in descriptor_order)
k[b] = W_key[b] e
v = W_value e
q_norm[b] = hidden[b] / sqrt(mean(hidden[b]^2) + epsilon) * query_scale[b]
k_norm[b] = k[b] / sqrt(mean(k[b]^2) + epsilon) * key_scale[b]
gate[b] = sigmoid(dot(q_norm[b], k_norm[b]) / sqrt(dim))
delta[b] = gate[b] * v
```

All numerical computation, addressing and state transitions occur in C. Python
only validates integer geometry, packs buffers, calls the ABI and returns tuples.
Weights/inputs/outputs are FP32; dot products and RMS sums accumulate in double.
Projected keys/values round to FP32 before gating; deltas round to FP32.
The sigmoid uses a sign-stable expression. This is not FP8 numerical parity.

## Admission And State

`oa_v41_engram_state_init(state, capacity)` initializes a caller-owned fixed
4096-ID state, with chosen capacity 1-4096. `oa_v41_engram_run` appends 0-256 IDs
per call. Full execution starts from empty state; chunk execution retains the
prefix token history needed for subsequent suffixes. No hidden-state or
convolution cache exists. History is bounded; there is no silent eviction.

`oa_v41_engram_rollback(state, length)` discards everything after a previously
accepted prefix length and clears discarded IDs. It does not restore returned
outputs or another component's state. The decoder must coordinate speculative
rollback across components. Rollback is destructive; increasing the length
again requires replay. Independent sessions hold independent native states.

Weights must fit both an explicit `weight_budget` and the native cap of 2^28
FP32 elements (1 GiB total across tables, projections and scales). The Python
binding admits geometry before allocating its owned copies. C does not copy
weight data and never writes it. Python owns private copies retained for the
model lifetime, so changes to caller lists cannot change execution weights.
Private ctypes fields are not a public mutation API.

For `T` tokens, `B` branches, hidden width `D`, memory width `M`, and `H` heads,
the explicit temporary budget must cover exactly:

```text
4 * (T*B*D + M + 2*D + T*H) bytes
```

This covers staged deltas, one memory/key/value workspace and staged indices.
The fixed caller-owned state, weights and input/output allocations are separate.
Allocator metadata is not part of this logical budget. Empty calls allocate no
workspace and validate configuration/state without appending.

Errors: 1 invalid geometry/ID/state/count/capacity/alias, 2 admission or allocation
failure, 3 nonfinite input/selected weight/intermediate. Outputs and state remain
unchanged on every rejected run. Dense weights, scales and hidden inputs are
checked for finiteness; table entries are checked when retrieved, not by scanning
every unused row. Projection overflow beyond FP32 is rejected.

Required output spans must be disjoint from each other, all input/weight/config
buffers and state. State must be disjoint from configuration/weights/inputs.
These overlaps are rejected before commit. Caller-provided capacities refer to
elements, not bytes; only required spans are written. Reserved fields must be 0.
The Python wrapper serializes sessions and close with a model lock; closed models
reject further calls. Raw C callers own synchronization and allocation lifetimes.

## Verification

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p test_native_deepseek_v41_engram.py -v
```

The test fixture builds a fresh private DLL with MSVC x64 `/W4 /WX /O2 /fp:strict
/LD`; non-Windows uses C99 `-Wall -Wextra -Werror`. Compilation errors fail setup.
Independent scalar and NumPy equations verify uint64 wraparound, left padding,
multiple orders/heads, concatenation, learned RMS scales, shared values and
branch-specific gating. Tests also cover full/chunk equality, every short-sequence
split, rollback/replay and changed continuation, empty execution, closed models,
state capacity, maximum head/memory/hidden/branch bounds, readonly weight copies,
exact temporary-budget boundary, malformed native geometry, nonfinite/overflow
failure, aliases and transactional rejection.

## Explicit Gaps

This is a genuine bounded Engram equation component, **not** a complete V4.1
decoder or checkpoint loader. Hidden width 256 and the 1 GiB weight cap exclude
publisher-scale deployment. The 6144 memory geometry is expressible, but the
reported giant tables cannot fit this admission contract. FP8 decoding/scales,
publisher hash/layout/compression metadata, RDMA prefetch/offload, GPU execution,
training/Sinkhorn updates, CED/mHC integration, tokenizer processing and production
generation parity remain outside scope. No checkpoint parity, benchmark quality,
publisher-layer placement or whole-decoder certification is claimed.
