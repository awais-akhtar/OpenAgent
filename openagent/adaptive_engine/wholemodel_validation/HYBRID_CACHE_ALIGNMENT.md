# K3 Hybrid Continuation View

This opt-in mapping is source-derived, not executed parity evidence. It does
not complete whole-model certification or establish full physical cache equality.

Set `hybrid_cache_alignment: true` on both concrete factories:
`k3_projection:create_k3_projected_candidate_adapter` and
`k3_vendor:create_k3_expanded_authoritative_adapter`. Both use the complete
`vendor_cache_schema`, layout
`kimi-k3-expanded-kv-kfirst-continuation-history-v1`, and an explicit
`execution_config`. Kernel-one convolution comparison leaves require the empty
shapes returned by `empty_comparison_shapes`. Candidate projection tolerance
must still equal the manifest cache tolerance.

The publisher text source was read at revision
`f831ab66814297da540d832a5235f8e904f29d06`; its fetched UTF-8 bytes matched
SHA-256 `9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a`.
Both KDA paths request transposed state. The separately inspected FLA revision
`3e52d5abb6ad85312846f39c3e946c2e9881a823` defines this as V-first state.
`k3_alignment.FLA_SOURCES` records the six inspected state-layout and convolution
source identities. These must be source artifacts in the oracle manifest and
must match the actual loaded modules. This revision is an explicitly supported
dependency contract, not a claim about the publisher's original installed FLA.
Only observed Triton ShortConvolution instances are accepted; other dependency
bytes, module paths, convolution backends or geometries fail closed.

Canonical recurrent state is `[1, heads, key, value]`. Publisher convolution
buffers are `[1, heads*dim, kernel]`, with oldest-first pre-convolution inputs.
The next append shifts out element zero. The comparison view reshapes head-major
channels and retains `[1:]`, matching native oldest-first `kernel-1` history.
No discarded tap is fabricated on the native side. Every publisher full buffer
and raw recurrent matrix remains mandatory separate Observation evidence.
The candidate likewise retains all raw snapshot leaves, local positions,
compressed latent/rotary rows and MLA projection weights. Same-role comparisons
include raw evidence; unlike raw physical layouts are not cross-compared.

MLA uses the existing explicit normalized-latent forward projection. Only those
MLA leaves are marked numerically uncertified projections. KDA coordinate
mapping does not certify native recurrence arithmetic or continuation outputs.

Remaining gaps include an actual compatible pinned FLA installation, transitive
dependency review beyond these six files, publisher wrapper/accelerator loading,
full-buffer physical equality (native has no oldest-tap counterpart), numerical
MLA and KDA parity, vision scope and runtime verification. No tests, imports,
validation runs, builds, compilation or benchmarks were performed for this work.
