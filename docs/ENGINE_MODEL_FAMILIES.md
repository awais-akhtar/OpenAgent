# Library-backed MoE model families

`openagent.adaptive_engine.hf_moe.HFMoeAdapter` supports local, unquantized
SafeTensors causal-language-model checkpoints for these Transformers model types:

| Model type | Exact expert owner | Expert width | Sparse layers |
| --- | --- | --- | --- |
| `mixtral` | `model.layers.L.block_sparse_moe.experts.E` | `intermediate_size` | All |
| `qwen2_moe` | `model.layers.L.mlp.experts.E` | `moe_intermediate_size` | Configured sparse step, excluding MLP-only layers |
| `qwen3_moe` | `model.layers.L.mlp.experts.E` | `moe_intermediate_size` | Configured sparse step, excluding MLP-only layers |
| `olmoe` | `model.layers.L.mlp.experts.E` | `intermediate_size` | All |
| `qwen3_next` | `model.layers.L.mlp.experts.E` | `moe_intermediate_size` | Configured sparse step, excluding MLP-only layers |
| `glm4_moe` | `model.layers.L.mlp.experts.E` | `moe_intermediate_size` | After `first_k_dense_replace` |
| `deepseek_v3` | `model.layers.L.mlp.experts.E` | `moe_intermediate_size` | After `first_k_dense_replace` |

This is Transformers/Accelerate integration, not an independent native model
implementation. Attention, routing, experts, cache updates and generation remain
library code. The reference/colibri registry was consulted for architectural
family identities only; no reference implementation was copied or imported.

The loader requires individually callable ModuleList experts and validates the
exact configured layer/expert path set. Checkpoint headers must match the library
model's complete tensor names and shapes, allowing only valid tied-weight aliases.
Shard indexes must agree with actual shard contents. Placement accepts exact
expert owners only, on disk or the resident CPU/explicit CUDA device. Dense
tensors stay resident; disk expert tensors must be meta after dispatch.

Tiny differential tests compare local library checkpoints against resident,
fully disk-offloaded and mixed-placement adapters: logits, router logits,
attention, cached decoding and greedy generation. Tied embeddings, sharded
checkpoints, corrupt tensor headers and invalid placement are also tested.
Verified environment: `.accelerator-venv`, Transformers 4.57.6, CPU execution.
CUDA correctness and production-scale memory/performance are not established by
these tests. Installed Transformers versions with fused expert banks are rejected.

## Remaining Reference Families

This adapter does not support the registry's GLM (`glm5_next`, `glm5_next_text`,
`glm_moe_dsa`, `glm5_moe`, `glm`), Inkling (`inkling_mm_model`, `inkling`),
Kimi (`kimi_k3`, `kimi_linear`), Qwen hybrid (`qwen3_5_moe`,
`qwen3_5_moe_text`, `qwen4_exp`, `qwen4_exp_text`) or DeepSeek
(`deepseek_v4`, `deepseek_v41`, `deepseek_v41_text`) families. Registry names
are architectural data, not evidence of compatible Transformers implementations.
Other Transformers MoE types outside the seven-family allowlist remain unsupported.

## Architecture Boundaries

Local Transformers 4.57.6 source inspection establishes:

- `qwen3_next` is Qwen3NextForCausalLM: gated full GQA attention alternates with
  gated-delta linear attention. Linear layers maintain convolution and recurrent
  states, not KV tensors. Tests exercise both layer types and compare all four
  hybrid-cache state lists after decoding, as well as router logits and generation.
  This loader requires an explicit mixed `layer_types` list. CPU uses the library's
  torch fallback; optional fused recurrent kernels are not verified here.
- `glm4_moe` is Glm4MoeForCausalLM: ordinary GQA with optional Q/K normalization,
  partial RoPE, dense prefix, shared experts and bias-corrected grouped sigmoid
  routing. It has neither recurrent attention nor MLA. It is not GLM-5 MLA/DSA.
- `deepseek_v3` is DeepseekV3ForCausalLM: low-rank Q and KV projections, separate
  rotary/nonrotary key dimensions, shared experts and grouped sigmoid routing.
  The installed eager path expands keys/values into its DynamicCache; this is not
  a native compressed-latent cache engine. Equal attention/KV head counts are
  required. Tests use low-rank Q, asymmetric QK/value dimensions and a dense prefix.
- GLM4-MoE/DeepSeek-V3 do not return router logits in their model output. Tests
  observe real router modules with forward hooks and compare selected expert
  indices and routing weights. No router execution is replaced or mocked.

Each new family has real fully-offloaded, mixed and resident comparisons,
tied/sharded coverage and malformed-geometry tests. Shared experts, routing
correction buffers and recurrent/MLA projections stay resident. The same exact
tensor-header validation covers these tensors and routed expert weights.

## Exact Registry Gap And Native Work

The installed Transformers 4.57.6 configuration mapping lacks every remaining registry type listed
above except `glm`. Its `glm` maps to the older dense GlmForCausalLM with GlmMLP,
not the registry's MoE/MLA family: a name collision, not support. New family names
are explicitly rejected; none is aliased to these three older architectures.
OLMoE library integration does not add to the separate independent OLMoE engine.
These additions broaden library coverage but do not close native family parity.

Separate [independent decoder modules](ENGINE_NATIVE_DECODERS.md) now implement
Mixtral equations and the exact `qwen3_5_moe_text` hybrid text contract. The latter
was checked against an isolated Transformers 5.16.1 oracle. It supports an
explicit compiled gated-delta primitive. Separate resident C decoders now cover
the full Qwen3.5 text arithmetic and exact `glm_moe_dsa` arithmetic, with tiny
independent numerical oracle tests. These do not change the library adapter's
allowlist or establish streamed native decoder coverage. The newer isolated mapping probes and remaining
family-specific work are recorded in that document.

Native work must start from independently obtained model specifications and
checkpoint tensor contracts, not translated reference implementation code:

| Registry family | Concrete implementation requirements |
| --- | --- |
| GLM-5 MLA/DSA (`glm_moe_dsa`, `glm5_moe`, registry `glm`) | Exact MLA projections and latent cache; DSA indexer and sparse selection semantics; grouped routed/shared expert execution and correction buffers. Older GLM4 GQA is not interchangeable. |
| GLM-5 Next (`glm5_next*`) | Exact hybrid layer schedule, recurrent state transition and full-attention cache; text-config/vision-wrapper boundaries and family tensor layout. |
| Qwen3.5 (`qwen3_5_moe*`) | Independent gated-delta prefill/decode, convolution state and rotary full-attention paths using this family's tensor names/projection ordering, rope_parameters and exact layer schedule. Similarity to Qwen3Next does not prove equivalence. |
| Qwen4 experimental (`qwen4_exp*`) | Exact mixed linear/QSA/PLE text execution now includes independent NumPy and resident C arithmetic, learned hyper-connections and routed/shared experts. Wrappers, scaled RoPE, quantized scales and streaming remain unsupported. |
| Kimi (`kimi_k3`, `kimi_linear`) | KDA recurrent layers and short-convolution states; MLA/output gates; attention-residual blocks; latent expert normalization and SiTU activation; grouped routing and wrapper boundaries. These are declared in local tiny config data. |
| Inkling (`inkling_mm_model`, `inkling`) | Obtain exact hybrid mixer equations and recurrent/cache contracts first; implement packed expert-bank layout and text/vision boundaries. Registry identity alone is insufficient to specify execution. |
| DeepSeek V4 (`deepseek_v4`) | Compressed/windowed attention and sparse indexer, grouped low-rank output projections, hash-layer routing, sqrtsoftplus scores, hyperconnection/Sinkhorn mixing and separate compression RoPE. Tiny config also declares FP4 experts and FP8 scales, requiring independent decoding contracts. |
| DeepSeek V4.1 (`deepseek_v41*`) | V4.1-specific compression and cross-layer KV-source ownership, hyperconnections, disk-backed engram tables and MTP layer accounting, plus exact wrapper/tensor contracts. V3 MLA cannot stand in for these. |

For each native family, acceptance needs independent tiny fixtures covering
router decisions, per-layer hidden states, full versus cached decode, state reset,
padding, greedy generation, tensor/scale corruption, expert eviction/reload and
measured peak-memory bounds. Registry metadata and config fixtures identify work
items but are not numerical correctness or performance evidence.

## Bounded Local Recheck (2026-10-02)

No additional exact registry family was enabled. The local
`.accelerator-venv` reports Transformers 4.57.6, torch 2.14.1+cpu and
Accelerate 1.15.0; `.venv` has no Transformers installation. All 16 remaining
registry model-type identities were checked against `CONFIG_MAPPING`: 15 are
absent, and `glm` resolves to `GlmConfig`. A real tiny library checkpoint test
establishes that its causal model is `GlmForCausalLM`, its feed-forward block
is `GlmMLP`, and it has no expert modules. The adapter rejects this checkpoint.
Manifest tests also reject all remaining identities with populated MoE geometry
and nested text-config wrappers, not just empty configurations.

Evidence is local metadata, not reference execution code:

- `reference/colibri/c/family_registry.py` family descriptors declare the exact
  remaining identities; OLMoE is the only registry family already covered.
- `reference/colibri/c/tools/requirements-glm53-tiny.txt` and
  `requirements-qwen38-tiny.txt` pin Transformers 5.16.1; these are declared
  oracle dependencies, not installed or independently verified support.
- `reference/colibri/c/qwen36_tiny_bad/config.hf.json` declares
  `qwen3_5_moe_text`, a hybrid schedule and `rope_parameters`. Its metadata
  cannot establish projection ordering, recurrence or cache equivalence.
- `reference/colibri/c/kimi_k3_tiny/config.json` declares `kimi_linear`, KDA,
  SiTU and latent normalization. Its numpy-only fixture requirements do not
  supply an independent Transformers model or a numerical oracle.
- `reference/colibri/c/deepseek_v4_tiny/config.json` declares FP4 experts,
  FP8 quantization, compression and hyperconnection geometry. Besides the
  absent model class, this exceeds the adapter's unquantized tensor contract.
  Its oracle requirements pin Transformers 5.14.1, also not installed here.

Unblocking requires an independently available exact config/model class or
complete execution specifications, checkpoint tensor/scale ownership, and an
independent tiny numerical oracle. Library availability alone is insufficient:
packed expert banks need a real per-expert dispatch design, wrappers need exact
text/vision ownership, and quantized families need validated decoding contracts.
No dependencies were upgraded and no reference model code or outputs were used
as implementation or correctness evidence during this recheck.
