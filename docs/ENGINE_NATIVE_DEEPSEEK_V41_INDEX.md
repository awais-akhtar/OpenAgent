# Bounded V4.1 Index Equation Fixture

Original standalone C arithmetic and an explicit-library Python wrapper. This
is a local equation fixture, not an exact publisher indexer, full decoder,
checkpoint reader, V4 alias, or compatibility claim. No publisher/reference
implementation was copied or translated. `NativeDeepSeekV41Index` is available
through the shared lazy export, and its compiled tests are included in CI.

## Mathematical Contract

All score inputs are contiguous row-major FP32. Let R=rows (1..32), H=heads
(1..8), D=width (1..8). Query Q has shape [H,D], keys K [R,H,D], and signed
finite mixing weights W [H]. Independently specified local equations are:

```text
d[r,h] = sum(j=0..D-1) Q[h,j] K[r,h,j]
s[r]   = sum(h=0..H-1) W[h] max(0, d[r,h])
```

Products, dot sums, and head mixing accumulate in FP64 in ascending dimension
and head order, then convert once to FP32. No normalization, softmax, RoPE,
position bias, quantization, or fused rounding is implied. Signed weights are
permitted; final scores may be negative. Finite input and finite FP32 output
are required. Selection compares stored FP32 scores, not unrounded FP64 sums.

For block size B (1..32), block b contains rows with floor(r/B)=b, including a
possibly short final block. Caller supplies a binary admission mask A[R].
An admitted row is one with A[r]=1. Caller must compute causal/compression
visibility; this module does not infer it from row order or layer numbers.

```text
eligible(b) = { r : floor(r/B)=b and A[r]=1 }
m[b] = max { s[r] : r in eligible(b) }
```

Mode 0 returns up to limit block IDs from nonempty eligible blocks, ordered by
descending m[b], then ascending block ID. Empty blocks are excluded, not assigned
zero (important for negative scores). Equal scores at the cutoff favor the lower
ID; exactly min(limit, eligible_count) entries are returned, not all ties.

Mode 1 accepts unique in-range candidate block IDs C, and returns up to limit
row IDs from {r: A[r]=1 and floor(r/B) in C}, ordered by descending s[r], then
ascending row ID. Candidate ordering has no effect. An empty candidate set is
valid and returns zero rows. Deeper layers supply newly computed scores over the
same owned keys; restriction is by block membership, not previous row ranking.
Excluded rows cannot win. All scores, even excluded ones, must be finite.

These equations deliberately settle local ties, admission and arithmetic. They
do not establish the publisher's scoring, block ties, tensor layouts or kernel
rounding. They implement bounded hierarchical selection, not decoder parity.

## Private ABI

The authoritative signatures are in `native_deepseek_v41_index.c`:

```text
oa_v41_index_abi_version() -> uint32 (1)
oa_v41_index_score(q,nq,keys,nk,weights,nw,rows,heads,width,out,no,budget)
oa_v41_index_select(scores,ns,admitted,na,rows,block_size,mode,
                    candidates,nc,limit,out,capacity,count,budget)
```

Lengths/budget are size_t, dimensions/limit/count uint32, IDs int32, mask uint8;
mode and return status are int. Score lengths must exactly match shapes. Limit
is 1..32, output capacity 0..32. Mode 0 requires null candidates and nc=0.
Mode 1 permits nc=0 only with a null candidate pointer. Capacity must cover the
actual selection, not necessarily limit. Success writes only selected IDs and
count; unused output slots remain unchanged. Zero-result success writes count=0.

Status 1: invalid dimensions, extents, alignment, mask, candidates or overlapping
ID/count outputs. Status 2: insufficient logical staging budget. Status 3:
nonfinite input/score or score overflow. Status 4: insufficient output capacity.
All failures preserve every output byte, including count. Complete computation
and validation precede publication. Input/output aliasing is allowed; the two
selection outputs must not overlap. Callers provide truthful live allocation
extents, aligned pointers, and exclusive access. Address checks cannot validate
allocation ownership or protect against invalid live pointers.

Logical scratch requirements are 4R bytes for scoring and
R*(sizeof(float)+2*sizeof(int32))+ceil(R/B) bytes for selection. Fixed stack
arrays reserve 128 and 416 bytes respectively, plus scalar/compiler overhead,
regardless of logical budget. There is no heap allocation or mutable global state.

## Ownership And Quantization

CSA2 Full owns corresponding KV and keys; Reindex changes selection, and Reuse
retains it. This stateless module neither mutates nor interprets CSA2 state.
Caller must maintain source/selector ownership and candidate lifetimes. Block
IDs from mode 0 are not row IDs suitable for CSA2 publication; mode 1 returns
row IDs. CSA2's existing eight-entry selection cap must be respected separately.
CED Z is not assumed to be an index key. No runtime integration is claimed.

The existing quant codec describes a local E2M1/E4M3 main-KV format. This module
accepts caller-supplied FP32 keys and does not relabel that codec as indexer MXFP4.
No packed-key decoding or quantized publisher index arithmetic is asserted.
See [ownership investigation](ENGINE_NATIVE_DEEPSEEK_V41.md) and
[local quant codec](ENGINE_NATIVE_DEEPSEEK_V41_QUANT.md).

Python loads only an explicit resolved local regular library file, checks private
ABI version 1, binds signatures, and releases its handle on close/collection.
Importing the shared lazy export does not build or load a native library.
No automatic build, fallback, decoder registration or decoder import occurs.
Caller trusts native library and dependencies; handshake is not authentication.
Score, select and close are serialized by a per-instance RLock. The operation
count covers input conversion, callbacks, native invocation and output conversion,
and unwinds in a finally block. A foreign-thread close waits for the operation's
lock; same-thread reentrant close raises RuntimeError while an operation is active
without releasing the handle. Close remains idempotent outside an operation.
Do not retain raw functions after close: direct raw ABI calls bypass this guard.
Iterables are
consumed only to bounded geometry plus one; integer wrapping and bool dimensions
are rejected. Methods return immutable tuples after successful native calls.

## Verification

Run `.accelerator-venv/Scripts/python.exe tests/test_native_deepseek_v41_index.py -v`.
Tests build this C source independently using MSVC `/W4 /WX /O2 /fp:strict`
or C99 `-Wall -Wextra -Werror -O2`. Missing compiler is a failure, not a skip.
Analytical fixtures cover multihead arithmetic, ReLU before signed mixing,
maximum geometry, per-row chunk equality, masked block maxima, tail blocks,
cutoff ties, deeper candidate restriction, empty sets, capacity, exact budget
boundaries, canaries, aliasing, pointer checks and transactional failure.
No exact decoder, external numerical oracle, sanitizer or GPU result is claimed.

Observed 2026-10-03: Windows accelerator environment compiled with MSVC
`/W4 /WX /O2 /fp:strict`; eleven tests passed with zero skips. Independent sorting
fixtures exercise every binary admission mask over seven rows at five block
sizes, negative scores, ties, reversed candidates, and maximum-size selection.
Concurrency fixtures hold each wrapper at native dispatch, start a foreign close,
verify the handle stays live, then release and invoke the actual compiled function
before close completes. Iterator callbacks test refused reentrant close in query,
keys, weights, scores, admission and candidates, plus exception unwinding and
successful reuse. No raw-call concurrency protection is claimed.
