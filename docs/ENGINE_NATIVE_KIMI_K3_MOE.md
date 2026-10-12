# Bounded Native K3 MoE

Standalone original scalar C FP32 inference of normalized LatentMoE, shared
experts, and SiTU-GLU. This is not an alias to KimiLinear, a checkpoint loader,
or full K3 inference. No vendor implementation is copied, translated, shipped,
or imported by the runtime. For bounded decoder composition, see
[Native K3 Hybrid](ENGINE_NATIVE_KIMI_K3_HYBRID.md). The standalone expert
verification below does not establish full published K3 support.

## Mathematical Contract

Primary publisher basis: [Kimi K3 report, sections 2.3.1-2.3.3](https://arxiv.org/html/2607.24653v1#S2.SS3).
Equations 11-13 define latent down-projection, sigmoid router, biased top-k
dispatch, unbiased selected-score renormalization, routed expert aggregation,
RMSNorm before latent up-projection, and addition of two full-width shared
experts. Both expert branches use SiTU: gate softcap times sigmoid of the
uncapped gate, multiplied by independently softcapped up branch. Defaults are
gate beta 4 and up beta 25. Router bias is frozen inference data; no training
quantiles, balancing update, input RMSNorm, or residual addition is performed.

Native router scale is explicitly fixed at 1 and selected-score normalization
is always enabled. There is no implicit `normalize_bias` option: stored bias is
added directly for selection, never used in mixture weights, and never centered
by this module. A caller requiring another router contract must reject it before
using this fixture API. Unknown config keywords raise TypeError.

Selection sorts descending biased FP32 score, ascending expert index within
equal scores. Equality of kth and (k+1)th scores rejects the entire request,
because the report does not prescribe cutoff tie ordering. Internal selected
ties are admitted with index ordering; tests do not claim vendor order equality.
Zero/underflowed selected-score sum and all nonfinite intermediates fail closed.

## Stable API

`KimiK3MoEConfig(hidden, latent, intermediate, experts, selected, shared,
epsilon, beta, linear_beta)` is a frozen explicit fixture config. Defaults:
12, 6, 9, 5, 2, 2, 1e-6, 4, 25. Limits: hidden/latent <=64,
intermediate <=128, experts <=64, 1 <= selected < experts, shared exactly 2;
positive finite scalar range [1e-12, 1e6]. No positional config additions.

`NativeKimiK3MoE(library, config, weights, memory_budget=16777216,
mac_budget=100000000).forward(inputs)` returns `(output, routes)`:
finite FP32 input/output `[T,H]`, int32 routes `[T,K]`, 1<=T<=128. Stateless,
token/chunk invariant, input copied before execution. `close()` is idempotent;
context-manager exit or weakref finalization releases this instance's library
reference. Peer instances loading the same DLL retain their own OS references.
Close and forward serialize through an instance lock. Private raw ctypes calls
are not protected from manual handle misuse.

`tensor_shapes(config)` specifies this exact ordered, C-packed FP32 inventory;
extra/missing tensors, wrong shape/dtype and nonfinite values are rejected:

| Name | Shape |
| --- | --- |
| router.weight | E,H |
| router.bias | E |
| latent.down.weight | L,H |
| latent.norm.weight | L |
| latent.up.weight | H,L |
| routed.gate.weight | E,M,L |
| routed.up.weight | E,M,L |
| routed.down.weight | E,L,M |
| shared.gate.weight | 2,M,H |
| shared.up.weight | 2,M,H |
| shared.down.weight | 2,H,M |

H/L/M/E/K denote hidden/latent/intermediate/experts/selected. Packed storage is
an immutable bytes-backed NumPy view, not merely a reversible writeability flag.
`budgets(config,T)` returns conservative bytes and multiply-accumulate count:
12*weight_elements + 4096 + T*(12H+8K) bytes, and
T*(EH+2HL+3KLM+6HM) MACs. Byte admission includes worst-case tensor flattening,
concatenation and bytes construction copies, fixed bounded C stack workspace,
wrapper input/output/routes and transactional C output/routes. It excludes
caller-owned original arrays, Python allocator/object overhead, and library
code pages. MAC budget counts matvecs, not activation/sort or validation work.
Both Python and C enforce these budgets before execution/heap output allocation.

The C entry point `oa_kimi_k3_moe` validates explicit weight/input/output/route
element counts; raw callers must supply valid appropriately sized, distinct
output/route buffers and must not concurrently mutate input/config/weights.
It stages all output and route data until every token succeeds. Error codes:
1 contract, 2 budget, 3 numeric/injected fault, 4 cutoff tie, 5 allocation.
The `_fail_after_token` keyword is solely a transaction-test fault hook.

## Actual Verification

Analytical suite compiles C with MSVC `/TC /W4 /WX /O2 /fp:strict /LD`;
missing compiler is a failure, not a skip. Run:

```powershell
.accelerator-venv/Scripts/python.exe -m unittest tests.test_native_kimi_k3_moe -v
```

Nine tests cover independent float64 equations, scalar closed form and SiTU
saturation, full-width shared path, bias-shift invariance, cutoff/internal ties,
strict inventory/config/input admission, geometry bounds, zero aggregate,
exact chunk invariance, budgets, immutable weights, multi-instance DLL ownership,
and raw-C late faults/numeric failures preserving both output buffers.

Separate optional exact vendor verification uses only unchanged publisher
public construction/state_dict/forward contracts on original synthetic weights:

```powershell
.accelerator-venv/Scripts/python.exe -m tests.test_native_kimi_k3_moe --vendor-directory <temporary-oracle-root> --vendor-dependencies <temporary-dependencies>
```

This requires the existing probe's namespace package `kimi_k3_vendor_probe`,
Torch and compatible dependencies. No download occurs in the ordinary test suite
or native runtime. Before import the optional verifier checks both hashes from
`tests/probe_kimi_k3_oracle.py`, revision
`f831ab66814297da540d832a5235f8e904f29d06`:

- configuration_kimi_k3.py: 735eb9ebe593e17d231e08e1df7f7be9b5ee0e079f511aa201f9572077b416ae
- modeling_kimi_linear.py: 9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a

Exact comparison executed successfully on Windows CPU with Torch 2.14.1+cpu,
Transformers 4.57.6, einops 0.8.1, fla-core 0.5.2 and
triton-windows 3.8.0.post29. The vendor warns CUDA is unavailable; no kernels
were replaced or source edited. Public `KimiSparseMoeBlock`, not its dense MLP
helper, ran normalized latent experts with two shared experts and SiTU. Strict
state_dict loading mapped fixture tensors, including concatenated shared gate/up
and column-concatenated shared down matrices. Seven-token output maximum
absolute error: 2.384185791015625e-07 (tolerance 2e-5). Selected expert sets and
unbiased normalized router probabilities matched. Separately observed vendor
common-bias-shift invariance, router scale 2.5 multiplication, and raw selected
sigmoid weights with renormalization disabled. Those last two modes are not
native admission claims. The public config/gate exposes `moe_renormalize` and
`routed_scaling_factor`, not a `normalize_bias` switch.

Limitations: this verifies one small synthetic publisher fixture, not published
weights, all routing modes, exact bitwise vendor equality, vendor cutoff ties,
GPU execution, quantization, grouped routing, distributed experts, training,
checkpoint ingestion, complete decoder/hybrid inference, or checkpoint licensing
admission. No shared exports, loaders, CI or other components are changed.
