# Decoder Execution Workstreams

The new execution packages separate model arithmetic, checkpoint admission,
storage, and comparison. They are explicit APIs: adding a package does not
automatically route application chat traffic through it.

```mermaid
flowchart TD
    Files[Local publisher configuration and tensor shards] --> Loader[Publisher checkpoint admission]
    Loader --> Weights[Validated tensor inventory and packed expert provider]
    Weights --> K3[K3 execution]
    Weights -. incomplete V4.1 handoff .-> V41[V4.1 execution]
    K3 --> Sessions[Transactional generation sessions]
    V41 --> Sessions
    Storage[Leases and durable cache pages] --> K3
    Storage --> V41
    Sessions --> Logits[Logits and token IDs]
    Logits --> Compare[Whole-model comparison adapters]
    Reference[Unchanged local reference implementation] --> Compare
    Compare --> Evidence[Comparison reports]
    Runners[Compatible self-hosted platform runners] --> Evidence
```

## Implementation Guides

| Area | Code | Guide |
| --- | --- | --- |
| V4.1 composition | `openagent/adaptive_engine/v41_execution/` | [V4.1 execution](ENGINE_V41_EXECUTION.md) |
| Publisher loading | `openagent/adaptive_engine/publisher_checkpoint/` | [Checkpoint loading](ENGINE_PUBLISHER_LOADING.md) |
| K3 composition | `openagent/adaptive_engine/k3_execution/` | [K3 execution](ENGINE_K3_EXECUTION.md) |
| Storage and recovery | `openagent/adaptive_engine/scalable_execution/` | [Scalable execution](ENGINE_SCALABLE_EXECUTION.md) |
| Authoritative comparison | `openagent/adaptive_engine/wholemodel_validation/` | [Whole-model validation](ENGINE_WHOLEMODEL_VALIDATION.md) |
| Platform runners | `openagent/adaptive_engine/platform_certification/` | [Platform certification](ENGINE_PLATFORM_CERTIFICATION.md) |
| Session coordination | `openagent/adaptive_engine/execution_service.py` | [Execution service](ENGINE_EXECUTION_SERVICE.md) |
| Shared native arithmetic | `openagent/adaptive_engine/native_tensor_ops.c` | [Native tensor operations](ENGINE_NATIVE_TENSOR_OPS.md) |

## Admission and Ownership

Checkpoint admission starts with the local configuration and shard headers.
Tensor names, shapes, scales, and encodings must match an explicit contract.
Unsupported fields remain findings; they are not silently replaced with a
different architecture. A prepared payload is not numerical certification.

Each generation session has a separate mutable executor. Immutable weights may
be shared where the executor permits it. Request failure or cancellation restores
the starting snapshot. A failed restore or reset quarantines the session rather
than allowing another request to use uncertain cache state.

Packed expert leases belong to the provider. A decoder consumes the selected
expert during the lease and must not keep borrowed arrays after release. Native
library paths are explicit; the session coordinator does not compile or download
code automatically.

## Verification Boundary

This code-first pass has not run tests, native compilation, benchmarks, or
platform certification. Earlier engine results do not establish correctness for
these additions. The comparison package provides the machinery for full and
incremental logit, cache, and generation comparisons; results require actually
running unchanged reference implementations with matching checkpoint pins.

Linux, CUDA, HIP, and Metal runners require compatible machines. A probe result
is separate from engine certification. Missing hardware, skipped checks, and CPU
fallbacks do not count as successful target-platform verification.

## Current Boundaries

| Workstream | Implemented source | Remaining requirements |
| --- | --- | --- |
| V4.1 decoder | Private equation composition plus a separate publisher-shaped text scheduler and explicit CPU arithmetic provider | Executed publisher comparisons, full cache alignment, and remaining operational limits documented in the V4.1 guide |
| Publisher loading | K3 text/vision preparation; V4.1 backbone, MTP and vision mappings, source-name handoff, selected Engram and packed-expert providers | Complete shard-header coverage and numerical admission of the handoff |
| K3 execution | Composed attention, residuals, dense/shared/routed experts, image tower/projector, packed leases; publisher-order MLA projection and AttnRes arithmetic | Unchanged publisher numerical parity, complete processor/placeholder certification, unsupported video behavior |
| Scale and recovery | Chunked generation, memory admission, variable recovery frames, opt-in periodic KDA persistence, atomic V4.1 checkpoint files | Measured large-model resource behavior; single-token dispatch still persists a terminal KDA state |
| Whole-model comparison | Pinned local Torch/native adapters, logits, token trajectories, exact integer checks, raw cache readers and explicit forward MLA projection | Executed unchanged-oracle comparisons, hybrid KDA alignment, and exact authoritative V4.1 cache alignment |
| Platforms | Strict Linux/CUDA/HIP/Metal runners, unavailable outcomes and workflow | Execution on real compatible hosts; current suites do not certify complete model families |

The six code workstreams have implementations, but that is not a statement that
every requested contract is complete. In particular, the remaining publisher
cache alignment and operational paths are code work, not merely missing hardware tests. No completeness
percentage or performance result is assigned to this unexecuted pass.
