# Local K3 Hybrid Fixtures

The native hybrid path can load local FP32 SafeTensors shards using the explicit
`openagent_kimi_k3_hybrid_fixture` format. This format describes the bounded
equation implementation; it is not a publisher checkpoint format. A `kimi_k3`
config is rejected rather than silently converted.

## Loading Pipeline

```text
config.json + local SafeTensors shards
    -> native library symbol and component-ABI preflight
    -> exact version and component configurations
    -> tensor inventory, dtype and shape checks
    -> weight and transaction-memory admission
    -> finite FP32 payload loading
    -> compiled KDA / MLA / latent experts / AttnRes
    -> logits and transactional hybrid state
```

`config.json` contains `model_type`, `fixture_version` (1), complete `mla`,
`kda` and `moe` configurations, and an explicit `layer_types` list. Unknown
fields, missing component fields and unsupported geometry are rejected.
`hybrid_fixture_document(config)` creates this metadata from a validated
`KimiK3HybridConfig`. Store finite contiguous FP32 tensors whose names and
shapes match `hybrid_tensor_shapes(config)` in local SafeTensors files.

## Usage

```python
from openagent.adaptive_engine import load_native_kimi_k3_fixture

decoder = load_native_kimi_k3_fixture(
    "local-fixture", library="native-k3-hybrid.dll",
    max_weight_bytes=64 * 1024**2, max_bytes=128 * 1024**2,
)
logits = decoder.forward([1, 4, 2])
saved = decoder.snapshot()
next_logits = decoder.forward([6])
decoder.restore(saved)
```

Build the library as described in [Hybrid Execution](ENGINE_NATIVE_KIMI_K3_HYBRID.md).
The loader does not download files, import vendor code or fall back to another
model. Both weight-loading and runtime budgets are explicit. These admission
estimates do not bound total Python process RSS.

## Verification

```powershell
.accelerator-venv/Scripts/python.exe -m unittest tests.test_native_kimi_k3_checkpoint -v
```

Five tests compile the native library and check local sharded loading,
analytical logits, identical full/chunked native outputs, exact schema rejection
and metadata-first rejection without reading tensor payloads. Incompatible
component ABIs and missing symbols are rejected before checkpoint reads;
the temporary preflight handle is released on success and failure. This establishes
fixture loading, not published K3 weight or full-hybrid vendor parity.
