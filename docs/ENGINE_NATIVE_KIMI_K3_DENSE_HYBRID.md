# Bounded K3 Dense Hybrid

The explicit FP32 hybrid fixture optionally replaces MoE branches at
`dense_layers` indices with bias-free dense SwiGLU. The tuple defaults to `()`;
indices must be unique integers within the configured layer range. Mixer types
remain independent. For dense-first fixtures, specify `(0,)` or `(0, 1)` explicitly.

`dense_intermediate` defaults to `None`, resolving to `moe.intermediate`.
Explicit widths are bounded to 1 through 128. Dense tensors use
`model.layers.I.mlp.{gate_proj,up_proj,down_proj}.weight`, with shapes
`(intermediate, hidden)`, `(intermediate, hidden)`, and `(hidden, intermediate)`.
Those layers have no MoE tensors. Other layers retain their exact inventory.

Compile `native_kimi_k3_dense.c` into the supplied hybrid library for dense
fixtures. Its `oa_k3_dense` kernel performs all SwiGLU arithmetic in C with
FP32 projection/activation boundaries, finite checks, and staged output.
There is no Python numerical fallback. Whole-call cache commit, snapshot,
restore, and bounded memory admission remain shared with the hybrid scheduler.

Fixture version 1 accepts optional JSON `dense_layers` (list) and
`dense_intermediate` fields. Default serialization omits both, preserving the
old document schema and tensor inventory. Old libraries need no dense symbol
for all-MoE fixtures.

This closes a dense-first structural gap in the bounded equation fixture,
not publisher checkpoint support. Actual publisher dense-first architecture
and complete model semantics still require separate validation. The runtime
continues to report `published_checkpoint_supported = False`.
