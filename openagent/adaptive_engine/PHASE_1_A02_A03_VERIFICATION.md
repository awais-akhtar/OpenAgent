# Phase 1 A02/A03 Verification — Priority Model Route Closure

**Date:** 2026-10-19
**Status:** Route traces defined, verification in progress
**Goal:** Verify each priority model has complete loader→consumer→TP→ABI→recovery chain

## Priority Models (5 families, 229 native routes)

### Qwen 3.5 (57 native routes)
**Trace file:** `phase1_a02_a03_real_routes.QWEN35_NATIVE_TRACE`

| Component | Status | Module | Symbol |
|-----------|--------|--------|--------|
| Config validator | ✓ | `publisher_family_contracts` | `component_contract()` |
| Loader | ✓ | `native_checkpoint` | `read_native_text_checkpoint()` |
| Consumer class | ✓ | `native_qwen35` | `NativeQwen35TextDecoder` |
| Forward path | ? | `native_qwen35` | `forward(tokens) → (logits, cache_update)` |
| Cache owner | ? | `qwen35_cache_recovery` | `Qwen35CacheRecovery` |
| TP owner | ✓ | `tensor_parallel` | `TensorParallel` |
| Factory | ✗ | `execution_factories` | `create_qwen35_native_session()` **MISSING** |
| CLI entry | ✓ | `native_family_cli` | `PublisherSession` |
| Recovery writer | ? | `service_generation_checkpoint` | `AtomicGenerationCheckpoint.write()` |
| Recovery reader | ? | `qwen35_cache_recovery` | `restore_from_checkpoint()` |
| Validation adapter | ? | `wholemodel_validation` | `qwen35_publisher_caches.compare_cache_state()` |

**Next steps for A02:**
1. Verify `NativeQwen35TextDecoder.forward()` method exists and returns (logits, cache_update)
2. Create or wire `execution_factories.create_qwen35_native_session(model_path, config_path)`
3. Verify cache recovery lifecycle (write/read) works with real session state
4. Run focused Qwen3.5 native route test: config→load→forward→cache→recovery

### Qwen 4 (57 native routes)
**Trace file:** `phase1_a02_a03_real_routes.QWEN4_NATIVE_TRACE`

| Component | Status | Module | Symbol |
|-----------|--------|--------|--------|
| Loader | ✓ | `native_checkpoint` | `read_native_text_checkpoint()` |
| Consumer class | ✓ | `native_qwen4` | `NativeQwen4TextDecoder` |
| Cache owner | ? | `qwen4_cache_recovery` | `Qwen4CacheRecovery` |
| Factory | ✗ | `execution_factories` | `create_qwen4_native_session()` **MISSING** |
| CLI entry | ✓ | `native_family_cli` | `PublisherSession` |

**Gap:** Factory not wired. Qwen4 also has specialized attention (`QSA` + gated) requiring state validation.

### GLM5-Next (49 native routes)
**Trace file:** `phase1_a02_a03_real_routes.GLM5_NATIVE_TRACE`

| Component | Status | Module | Symbol |
|-----------|--------|--------|--------|
| Loader | ✓ | `native_checkpoint` | `read_native_text_checkpoint()` |
| Consumer class | ✓ | `native_glm5_next` | `NativeGLM5NextTextDecoder` |
| Factory | ✗ | `execution_factories` | `create_glm5_native_session()` **MISSING** |

**Gap:** Factory not wired. GLM5-Next uses shared expert bank requiring preflight inventory validation.

### GLM-DSA (15 native routes)
**Trace file:** `phase1_a02_a03_real_routes.GLM_DSA_NATIVE_TRACE`

| Component | Status | Module | Symbol |
|-----------|--------|--------|--------|
| Loader | ✓ | `native_checkpoint` | `read_native_text_checkpoint()` |
| Consumer class | ✓ | `native_glm_dsa` | `NativeGLMDSATextDecoder` |
| Cache owner | ? | `native_glm_dsa_cache` | `GLMDSACacheRecovery` |
| Cache ranges | ✓ | `native_glm_dsa_ranges` | `GLMDSAActivationRanges` |
| Factory | ✗ | `execution_factories` | `create_glm_dsa_native_session()` **MISSING** |

**Gap:** Factory not wired. GLM-DSA has dynamic activation routing requiring cache state mapping.

### DeepSeek V4.1 (51 native routes, BF16/FP8/FP4 only)
**Trace file:** `phase1_a02_a03_real_routes.DEEPSEEK_V41_NATIVE_BF16_TRACE`

**Note:** DeepSeek V4.1 restricted to 3 publisher-bank format routes:
- BF16 → BF16 (48 routes: native only)
- FP8 → FP8 (not native, device-only)
- FP4 → FP4 (not native, device-only)

| Component | Status | Module | Symbol |
|-----------|--------|--------|--------|
| Loader | ✓ | `native_checkpoint` | `read_native_text_checkpoint()` |
| Consumer class | ✓ | `native_deepseek_v41_*` | Multiple modules (CED, Engram, MHC, Quant, Index) |
| TP owner | ✓ | `tensor_parallel` | `TensorParallel` |
| Factory | ✗ | `execution_factories` | `create_deepseek_v41_native_session()` **MISSING** |
| Recovery (complex) | ? | `distributed_execution` | Distributed TP consensus + failure-summary |

**Gaps:**
1. Factory not wired
2. DeepSeek V4.1 distributed paths require TP consensus and failure recovery (P06) — skipped in this pass (blocked on GPU)

## A03 — Other Families (70 native routes)

### K3/Kimi (4 Metal, rest native)
- Metal: 4 verified routes with token-to-logit dispatch (A08 ✓)
- Native: routes exist but no consumer traces yet

### Mixtral (18 native routes)
### OLMoE (27 native routes)
### Inkling (21 native routes)

Status: All have native_*.py modules but no factory wiring or cache recovery yet.

## Missing Implementations (Blocker for all families)

### 1. Factory Functions (CRITICAL)
**Files to create/update:** `execution_factories.py`

For each priority model, add:
```python
def create_qwen35_native_session(model_path: str, config_path: str) -> NativeQwen35Session:
    """Load Qwen3.5 from native checkpoint and prepare for generation."""
    config = load_config(config_path)
    weights = native_checkpoint.read_native_text_checkpoint(Path(model_path), family='qwen3.5')
    decoder = NativeQwen35TextDecoder(weights, config)
    cache = Qwen35CacheRecovery()
    return NativeQwen35Session(decoder, cache, config)
```

**Impact:** Blocks all A02/A03 route closure. Without factories, admitted routes cannot be instantiated.

### 2. Cache Recovery Modules (CRITICAL)
**Status:** `qwen35_cache_recovery.py`, `qwen4_cache_recovery.py` mentioned but need verification:
- `Qwen35CacheRecovery` class exists and has `write()` / `restore()` methods
- Same for Qwen4, GLM-DSA

### 3. Family-Specific ABI Ranges
**Status:** Most have `native_*_ranges.py`:
- ✓ `native_qwen35_ranges.py`
- ✓ `native_qwen4_ranges.py`
- ✓ `native_glm_dsa_ranges.py`
- ✓ `native_mixtral_ranges.py`

Ranges define valid activation bounds for model state validation.

### 4. Public Adapters (ABI entry points)
**Status:** Some have `native_*_public_adapters.py`, others use `native_family_public_adapters.py`
- ✓ `native_qwen4_public_adapters.py`
- Qwen3.5: check if separate or in `native_family_public_adapters`

## Test Plan for Each Route

Once factory is wired, test the complete path:

```python
def test_qwen35_native_route_complete():
    """Verify config→load→forward→cache→recovery for Qwen3.5 native BF16."""
    config = load_test_config('models/qwen3.5-test')
    
    # 1. Load
    session = create_qwen35_native_session(
        model_path='models/qwen3.5-test',
        config_path='models/qwen3.5-test/config.json'
    )
    
    # 2. Generate (forward)
    tokens = [1, 2, 3]
    logits, cache = session.forward(tokens)
    assert logits.shape == (vocab_size,)
    assert cache is not None
    
    # 3. Snapshot
    receipt = session.snapshot()
    
    # 4. Restore
    session2 = create_qwen35_native_session(...)
    session2.restore(receipt)
    logits2, cache2 = session2.forward([4])  # next token
    
    # 5. Compare (P13 evidence)
    assert compare_logits(logits, logits2, threshold=0.0001)
    
    # 6. Cleanup
    session.close()
    session2.close()
```

## Completion Criteria

For each priority model route to be `COMPLETE`:
1. ✓ Loader function exists and reads checkpoint
2. ✓ Consumer class exists with forward() method
3. ✓ Cache recovery class exists with write/restore
4. ✓ Factory function exists and wires all pieces
5. ✓ Complete path test passes (config→load→forward→cache→restore→compare)
6. ✓ CLI entry available for manual testing
7. ✓ Route marked as `verified-admitted` in source_scope_matrix

**Current blocker:** Factory functions missing for all priority models.

Once factories are created and wired, remaining tests are straightforward (P13 evidence requires GPU, which is blocked-env).

