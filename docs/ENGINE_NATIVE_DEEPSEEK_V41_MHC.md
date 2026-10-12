# Native V4.1 mHC

Original bounded C equation fixture, with an explicit-library Python wrapper.
No V4 or publisher reference code is used. The class is exported by the engine
package and its compiled tests are included in CI. It has no publisher decoder
registration, checkpoint loading or CED integration.

## Source And Timing

[Section 2.4.1, equation (6)](https://arxiv.org/html/2609.19969v1#S2.SS4.SSS1)
defines the delayed-input rule:

```text
(A_l, B_l, C_l) = H(X_l)
u_l = A_(l-1) X_l
X_(l+1) = B_l X_l + C_l F_l(u_l)
```

The component retains `A_(l-1)` in state. Current `A_l` does not feed the
current block. Current `B_l` and `C_l` do update the current residual. Successful
commit replaces the retained coefficients with `A_l` and advances depth once.
Rows are independent token positions; state is carried across **block depth**,
not reused as a recurrent state across successive tokens. Initial `A_-1` is
required explicitly, not inferred from a claimed publisher boundary rule.

## Equation Fixture Contract

All tensors are contiguous float32, row-major. C accumulates reductions in
double and checks representability before float32 writes. Bounds: rows 1..8,
streams 1..4, width 1..16, total depth capacity 1..64, chunk 1..16, head vocabulary
1..32, Sinkhorn sweeps 1..32. These are local fixture limits, not model metadata.

`X` has shape `[rows, streams, width]`. Coefficients are `[rows, q]`, where
`q = streams * (streams + 2)`, packed as `A[streams]`,
`B[streams, streams]`, `C[streams]`. `B[i,j]` maps old stream j into new stream i.
External commit requires finite A in [0,1], B entries in [0,1], C in [0,2].
It does not certify that external B is doubly stochastic; callers own that
property. This permits explicit analytical coefficient fixtures.

The implemented H is a **local predictor fixture**, not an exact checkpoint
mapping: flatten each row of X, divide by `sqrt(mean(X^2) + epsilon)`, then
apply `H_weight[q, streams*width]` and `H_bias[q]`. Transform A logits by sigmoid,
C logits by twice sigmoid. Transform B by `exp(logit - max(B_logits)) + 1e-12`,
then perform the requested number of row-normalize/column-normalize sweeps.
Columns are normalized last; finite sweeps need not give exact unit row sums.
There are no implicit learned predictor scales or norm weights.

The native `run` fixture uses `F(u) = F_weight[width,width] u + F_bias[width]`.
All four parameter arrays are depth-major, without a token-row parameter axis.
No attention, MoE, pre-norm of F input, quantization, FP8, training, or GPU
Mega-mHC fusion is implied. The paper's single-pass coefficient timing is
preserved; this scalar CPU implementation does not claim its memory-traffic
or deployment-performance results.

Final head behavior is deliberately separate and explicit:

```text
y[row,d] = sum_j final_mix[row,j] * X[row,j,d]
z = norm[d] * y / sqrt(mean_d(y*y) + epsilon)
logits[row,v] = sum_d head_weight[v,d] * z[row,d] + head_bias[v]
```

`final_mix` is caller supplied and finite, with no sigmoid or sum constraint.
It is not silently replaced with the retained delayed A. Head reads state
without advancing depth, and may be called at intermediate depth. The
decoder decides when the final boundary has been reached.

## Reusable API

`NativeDeepSeekV41MHC(library_path, rows, streams, width, x, initial_mix,
capacity=64)` loads a trusted, independently compiled library. No auto-build,
download or automatic compilation is performed.

- `mix()` returns delayed mixed input, shape `[rows,width]` flattened.
- `predict(weight, bias, epsilon=1e-6, sweeps=8)` returns current packed H.
- `commit(coefficients, block_output, expected_depth=...)` updates atomically.
- `run(blocks, h_weight, h_bias, f_weight, f_bias, ...)` executes an atomic chunk.
- `head(final_mix, norm, weight, bias, vocab, ...)` returns flattened logits.
- `depth`, `residual`, `previous_mix` expose copies of current values.
- `snapshot()`, `restore(bytes)`, `reset()`, `close()` provide lifecycle control.

For the compositional decoder: read depth, call mix and predict on unchanged
state, execute the model block on mixed input, then commit its output with
that depth. Keep this sequence serialized at the decoder level. The depth guard
rejects stale depth, but does not detect a restore to a different branch with
the same depth. Snapshot/restore is explicit rollback; it is not model-identity
authentication or a portable checkpoint format.

Private C exports use prefix `oa_v41_mhc_`: `abi_version`, `state_bytes`,
`scratch_bytes`, `validate`, `init`, `mix`, `predict`, `commit`, `run`, `head`.
The signatures are bound in the wrapper. Tensor extents count floats; state
extents count bytes. State layout is six uint32 fields (magic, rows, streams,
width, depth, capacity), then 512 residual floats and 32 delayed-A floats.
Unused tails are zero. All used scalar buffers need 4-byte alignment.

All entry points check exact tensor extents, geometry, null/alignment and
uintptr address-range wraparound before tensor access. Writable outputs cannot
overlap state or input arrays. Read-only input arrays may overlap each other.
The declared extent must correspond to actual accessible memory; C cannot
prove pointer provenance or mapped allocation size. Python builds owned,
bounded arrays from iterables. Raw native callers must supply valid memory and
serialize accesses themselves.

Errors: 1 invalid contract/buffer/depth, 2 insufficient run staging budget,
3 nonfinite/overflow, 4 depth/output capacity. Failed operations leave state
and output unchanged. Run commits only after every block succeeds, including
late-block failures. Head capacity may exceed the required result; unused
output tail stays unchanged. Other output extents must be exact.

`scratch_bytes()` gives the conservative run payload staging budget including
nested predictor and update arrays, independent of chunk length. This is not
compiler stack-frame accounting; fixed-size stack buffers bound allocation,
with no heap allocation in C. Snapshots require this ABI, byte order, validated
geometry, and matching capacity. Reset restores the constructor's X and A.
Wrapper operations are serialized; close waits on other threads and refuses
reentrant close. Mutating wrapper calls also reject reentrant invocation during
tensor ingestion. Do not retain or call raw library function pointers after close.

## Verification And Gaps

Run `python -m unittest discover -s tests -p test_native_deepseek_v41_mhc.py -v`.
Tests build in a temporary directory using MSVC `/W4 /WX /O2 /fp:strict /LD`
on Windows, or C99 `-Wall -Wextra -Werror` on other platforms.

Coverage includes independent dynamic H and affine equation comparisons,
hand-computed delayed/current timing, full/chunk/manual composition,
snapshot replay/reset/corruption, maximum geometry/depth, head equation,
budget boundary, invalid extents, alignment, overlaps, address wraparound,
nonfinite inputs, numerical overflow, late-block rollback, output sentinels,
closed lifecycle, and bounded iterable ingestion.

Local Windows strict MSVC compilation and fixture tests are verified. Unix
compiler branch, sanitizers, publisher checkpoint parity, model-scale accuracy,
accelerator performance, and CED integration remain unverified. Exact
publisher tensor names, predictor parameterization, initial mixing, final head
boundary parameters, and checkpoint conversion require independent evidence.
