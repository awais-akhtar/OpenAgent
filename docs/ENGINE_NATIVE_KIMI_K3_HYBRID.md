# Bounded K3 Hybrid Execution

The native hybrid fixture combines KDA and NoPE MLA layers, normalized latent
experts, two full-width shared experts, block attention residuals and a language
model head. It takes token IDs and returns FP32 logits. Python schedules native
operations; projection, recurrent update, attention, normalization, routing,
expert evaluation, residual addition and output-head arithmetic run in C.

Optional [dense SwiGLU layers](ENGINE_NATIVE_KIMI_K3_DENSE_HYBRID.md) can replace
specified MoE branches. Defaults preserve the original all-MoE fixture.

This is an explicitly configured small text fixture, not published K3 checkpoint
support. It does not load vendor configurations, preprocess images, read packed
weights, tokenize text or select a model server. The native resident checkpoint
loader therefore does not admit `kimi_k3` as a general model identity.

## Dataflow

```text
Token IDs -> embedding
    -> block attention residuals -> input normalization
    -> KDA or NoPE MLA
    -> residual accumulation -> attention residuals -> normalization
    -> normalized latent experts + two shared experts
    -> residual accumulation
    -> repeat for configured layers
    -> final attention residuals -> normalization -> head -> logits

Committed layer caches
    -> request-local candidates
    -> all tokens and all layers succeed
    -> one state publication
```

KDA uses the lower-bounded log-decay and full-width output-gate equations from
[the publisher report](https://arxiv.org/html/2607.24653v1). The decay floor is
fixed at -5. Query scaling and normalization epsilons are explicit fixture
parameters, not inferred checkpoint constants. Short-convolution history and
recurrent state are retained between calls. MLA retains expanded K/V rather
than compressed latent caches. Neither path uses rotary positions here.

## Configuration

```python
from openagent.adaptive_engine import (
    KimiK3DenseMLAConfig, KimiK3KDAConfig, KimiK3MoEConfig,
    KimiK3HybridConfig, NativeKimiK3Hybrid,
)
from openagent.adaptive_engine.native_kimi_k3_hybrid import hybrid_tensor_shapes

config = KimiK3HybridConfig(
    mla=KimiK3DenseMLAConfig(layers=4, capacity=16),
    kda=KimiK3KDAConfig(capacity=16),
    moe=KimiK3MoEConfig(),
    layer_types=("kda", "kda", "kda", "mla"),
)
required_tensors = hybrid_tensor_shapes(config)
# Supply finite, contiguous FP32 arrays matching every named shape.
model = NativeKimiK3Hybrid(config, weights, library="native-k3-hybrid.dll")
logits = model.forward([1, 4, 2])
saved = model.snapshot()
next_logits = model.forward([9])
model.restore(saved)
```

The three configurations must share hidden width; KDA and MLA capacity must
match. Every layer type is explicit and the final layer must be MLA. Component
classes enforce their own geometry bounds. No padding masks, batches, training,
speculation, quantized tensors or unknown forward options are accepted.
The expert router rejects an ambiguous top-k cutoff instead of inventing a
tie-breaking rule for publisher parity.

Build one library from `native_kimi.c`, `native_kimi_k3_attnres.c`,
`native_kimi_k3_kda.c`, `native_kimi_k3_moe.c` and `native_kimi_k3_blocks.c`.
For example, with MSVC initialized:

```powershell
cl /W4 /WX /O2 /fp:precise /LD openagent/adaptive_engine/native_kimi.c openagent/adaptive_engine/native_kimi_k3_attnres.c openagent/adaptive_engine/native_kimi_k3_kda.c openagent/adaptive_engine/native_kimi_k3_moe.c openagent/adaptive_engine/native_kimi_k3_blocks.c /link /OUT:native-k3-hybrid.dll
```

## State And Validation

Snapshots contain detached MLA and recurrent/convolution buffers with a model
identity derived from the explicit configuration and weights. Restore validates
all layers before publishing state. Token capacity, memory admission and tensor
checks precede generation. Projection, expert or output failures leave the live
state unchanged, including failures after earlier layers have completed.
Memory admission is a conservative buffer estimate, not a process-RSS promise.
Raw C calls require valid allocated buffers and disjoint output regions.
The MLA cache/output regions must also be disjoint from its weights and input.
Hybrid restore accepts base NumPy arrays only, not ndarray subclasses whose
copy methods could defeat snapshot detachment. Identical configuration/weight
instances may exchange snapshots; changed weights or configuration may not.
The admission estimate includes a 1 MiB workspace allowance and does not track
caller-retained snapshots or enforce a process-wide allocator limit.

```powershell
.accelerator-venv/Scripts/python.exe -m unittest tests.test_native_kimi_k3_kda tests.test_native_kimi_k3_moe tests.test_native_kimi_k3_hybrid -v
```

Tests compile the actual C sources and compare fixture logits, layer caches,
tokenwise and irregular-chunk execution against separately evaluated equations.
They cover rollback, reset/restore, detached snapshots and unsupported inputs.
These analytical tests are distinct from the unchanged vendor dense-MLA oracle.
No unchanged vendor full-hybrid KDA run or published-checkpoint logits have been
verified on this host. GPU/NPU execution, packed streaming, full geometry,
multimodal K3 and production-scale performance remain outside this path.
