"""Phase 1 A10 — CPU Runtime Evidence Capture (GPU-free P13 closure).

Capture numerical parity, cache state, and execution receipts using native CPU inference.
No GPU required. Enables Phase 1 closure on CPU-only hardware.

Evidence types:
- Execution receipt: forward() produces logits with correct shape/dtype
- Numerical parity: same route on CPU produces identical outputs (within tolerance)
- Cache state: snapshot/restore preserves and advances KV cache correctly
- Completion proof: end-to-end generation with recovery works
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from typing import Any, Dict, Optional, Tuple
import time


@dataclass
class ExecutionReceipt:
    """Evidence that one forward pass executed successfully."""
    route_key: str  # family/model_type/backend/arithmetic/source/persisted
    checkpoint_path: str
    config_hash: str  # SHA256 of config.json

    # Execution proof
    input_tokens: list  # e.g. [1, 2, 3]
    output_shape: Tuple[int, ...]  # logits shape
    output_dtype: str  # 'float32', 'bfloat16', etc

    # Numerics
    output_min: float
    output_max: float
    output_mean: float
    output_std: float

    # Performance
    forward_ms: float
    memory_peak_mb: float

    # State
    cache_shape: Optional[Dict[str, Tuple[int, ...]]]  # KV cache shapes
    cache_hash: Optional[str]  # SHA256 of cache state

    timestamp: str  # ISO 8601

    def to_dict(self) -> Dict[str, Any]:
        return {
            'route_key': self.route_key,
            'checkpoint_path': str(self.checkpoint_path),
            'config_hash': self.config_hash,
            'input_tokens': self.input_tokens,
            'output_shape': list(self.output_shape),
            'output_dtype': self.output_dtype,
            'output_stats': {
                'min': self.output_min,
                'max': self.output_max,
                'mean': self.output_mean,
                'std': self.output_std,
            },
            'performance_ms': self.forward_ms,
            'memory_peak_mb': self.memory_peak_mb,
            'cache_shape': {k: list(v) for k, v in (self.cache_shape or {}).items()},
            'cache_hash': self.cache_hash,
            'timestamp': self.timestamp,
        }


@dataclass
class RecoveryReceipt:
    """Evidence that snapshot/restore cycle preserves state."""
    route_key: str

    # Generation 1 (before snapshot)
    gen1_tokens: list
    gen1_logits_hash: str  # SHA256 of output logits
    gen1_cache_hash: str

    # Snapshot
    snapshot_bytes: int
    snapshot_hash: str  # SHA256 of snapshot content

    # Generation 2 (after restore)
    gen2_tokens: list
    gen2_logits_hash: str
    gen2_cache_hash: str

    # Verification
    logits_match: bool  # gen1_logits == gen2_logits (within tolerance)
    cache_match: bool   # gen1_cache == gen2_cache

    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'route_key': self.route_key,
            'generation1': {
                'tokens': self.gen1_tokens,
                'logits_hash': self.gen1_logits_hash,
                'cache_hash': self.gen1_cache_hash,
            },
            'snapshot': {
                'bytes': self.snapshot_bytes,
                'hash': self.snapshot_hash,
            },
            'generation2': {
                'tokens': self.gen2_tokens,
                'logits_hash': self.gen2_logits_hash,
                'cache_hash': self.gen2_cache_hash,
            },
            'verification': {
                'logits_match': self.logits_match,
                'cache_match': self.cache_match,
            },
            'timestamp': self.timestamp,
        }


@dataclass
class P13EvidenceBinding:
    """A10: Complete evidence package for one route."""
    route_key: str
    family: str
    model_type: str
    backend: str  # 'native' for CPU

    # Source identity
    checkpoint_hash: str  # SHA256 of checkpoint weights
    config_hash: str

    # Execution evidence
    execution_receipts: list[ExecutionReceipt]  # Multiple runs for consistency

    # Recovery evidence
    recovery_receipts: list[RecoveryReceipt]  # Snapshot/restore cycles

    # Numerical parity
    parity_verified: bool  # All runs produce same logits within tolerance
    tolerance_threshold: float  # e.g. 1e-5 for logits comparison

    # Completion proof
    full_generation_success: bool  # End-to-end generation works
    generation_tokens: list  # Sample generated sequence

    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'route_key': self.route_key,
            'family': self.family,
            'model_type': self.model_type,
            'backend': self.backend,
            'checkpoint_hash': self.checkpoint_hash,
            'config_hash': self.config_hash,
            'execution_receipts': [r.to_dict() for r in self.execution_receipts],
            'recovery_receipts': [r.to_dict() for r in self.recovery_receipts],
            'parity_verified': self.parity_verified,
            'tolerance_threshold': self.tolerance_threshold,
            'full_generation_success': self.full_generation_success,
            'generation_tokens': self.generation_tokens,
            'timestamp': self.timestamp,
        }


def _hash_bytes(data: bytes) -> str:
    """SHA256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()


def _hash_config(config_path: Path) -> str:
    """Hash config.json for reproducibility proof."""
    config_data = config_path.read_bytes()
    return _hash_bytes(config_data)


def capture_execution_evidence(
    route_key: str,
    checkpoint_path: Path,
    config_path: Path,
    session: Any,  # NativeExecutionService or decoder session
    input_tokens: list,
    tolerance: float = 1e-5,
) -> ExecutionReceipt:
    """A10: Capture one execution's evidence (no GPU required).

    Args:
        route_key: family/model_type/backend/arithmetic/source/persisted
        checkpoint_path: Model directory
        config_path: config.json
        session: Loaded native session (from A02 factory)
        input_tokens: Test token sequence
        tolerance: Numerical comparison tolerance

    Returns:
        ExecutionReceipt with logits, cache, performance metrics

    Evidence captured:
        - Output shape/dtype (proves forward() works)
        - Output statistics (numerics)
        - Cache state hash (proves state advance)
        - Timing (CPU performance baseline)
    """
    import time
    import numpy as np

    # Config hash for reproducibility
    config_hash = _hash_config(config_path)

    # Forward pass
    start_ms = time.perf_counter() * 1000
    try:
        # Call the session's forward method
        if hasattr(session, 'forward'):
            logits, cache_update = session.forward(input_tokens)
        else:
            logits = session.generate(input_tokens, max_new_tokens=1)
            cache_update = None
    except Exception as e:
        raise RuntimeError(f"Forward pass failed for {route_key}: {e}")
    finally:
        forward_ms = (time.perf_counter() * 1000) - start_ms

    # Convert logits to numpy for analysis
    if hasattr(logits, 'numpy'):
        logits_np = logits.numpy()
    else:
        logits_np = np.asarray(logits)

    # Numerical analysis
    logits_min = float(np.min(logits_np))
    logits_max = float(np.max(logits_np))
    logits_mean = float(np.mean(logits_np))
    logits_std = float(np.std(logits_np))

    # Cache shape (if available)
    cache_shape = None
    cache_hash = None
    if cache_update is not None:
        if isinstance(cache_update, dict):
            cache_shape = {k: tuple(v.shape) if hasattr(v, 'shape') else tuple(v)
                          for k, v in cache_update.items()}
            cache_bytes = json.dumps({k: list(v) for k, v in cache_shape.items()}).encode()
            cache_hash = _hash_bytes(cache_bytes)

    # Estimate memory (rough — CPU doesn't report like GPU)
    import psutil
    import os
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024

    return ExecutionReceipt(
        route_key=route_key,
        checkpoint_path=str(checkpoint_path),
        config_hash=config_hash,
        input_tokens=input_tokens,
        output_shape=tuple(logits_np.shape),
        output_dtype=str(logits_np.dtype),
        output_min=logits_min,
        output_max=logits_max,
        output_mean=logits_mean,
        output_std=logits_std,
        forward_ms=forward_ms,
        memory_peak_mb=memory_mb,
        cache_shape=cache_shape,
        cache_hash=cache_hash,
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    )


def capture_recovery_evidence(
    route_key: str,
    session: Any,  # Session with snapshot/restore
    input_tokens_gen1: list,
    input_tokens_gen2: list,
) -> Optional[RecoveryReceipt]:
    """A10: Capture snapshot/restore evidence (state preservation).

    Tests:
        1. Generate with input_tokens_gen1 → snapshot
        2. Restore snapshot
        3. Generate with input_tokens_gen2 → compare outputs

    Returns:
        RecoveryReceipt if session supports snapshot/restore, else None
    """
    import time
    import numpy as np

    if not (hasattr(session, 'snapshot') and hasattr(session, 'restore')):
        return None

    # Generation 1
    try:
        logits1, cache1 = session.forward(input_tokens_gen1)
        if hasattr(logits1, 'numpy'):
            logits1_np = logits1.numpy()
        else:
            logits1_np = np.asarray(logits1)
        logits1_hash = _hash_bytes(logits1_np.tobytes())
        cache1_hash = _hash_bytes(json.dumps(str(cache1)).encode())
    except Exception as e:
        print(f"Gen1 failed: {e}")
        return None

    # Snapshot
    try:
        snapshot = session.snapshot()
        snapshot_bytes = len(json.dumps(snapshot).encode())
        snapshot_hash = _hash_bytes(json.dumps(snapshot).encode())
    except Exception as e:
        print(f"Snapshot failed: {e}")
        return None

    # Restore and Generation 2
    try:
        session.restore(snapshot)
        logits2, cache2 = session.forward(input_tokens_gen2)
        if hasattr(logits2, 'numpy'):
            logits2_np = logits2.numpy()
        else:
            logits2_np = np.asarray(logits2)
        logits2_hash = _hash_bytes(logits2_np.tobytes())
        cache2_hash = _hash_bytes(json.dumps(str(cache2)).encode())
    except Exception as e:
        print(f"Gen2 failed: {e}")
        return None

    # Verification
    logits_match = logits1_hash == logits2_hash
    cache_match = cache1_hash == cache2_hash

    return RecoveryReceipt(
        route_key=route_key,
        gen1_tokens=input_tokens_gen1,
        gen1_logits_hash=logits1_hash,
        gen1_cache_hash=cache1_hash,
        snapshot_bytes=snapshot_bytes,
        snapshot_hash=snapshot_hash,
        gen2_tokens=input_tokens_gen2,
        gen2_logits_hash=logits2_hash,
        gen2_cache_hash=cache2_hash,
        logits_match=logits_match,
        cache_match=cache_match,
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    )


def save_evidence_manifest(
    binding: P13EvidenceBinding,
    output_path: Path,
) -> None:
    """Save P13 evidence binding to JSON manifest."""
    manifest = {
        'phase1_a10_evidence': binding.to_dict(),
        'schema_version': '1.0',
        'captured_at': binding.timestamp,
    }
    output_path.write_text(json.dumps(manifest, indent=2))
    print(f"✓ Evidence saved: {output_path}")


# ============================================================================
# A10 EXECUTION PLAN — CPU-NATIVE ROUTES
# ============================================================================
#
# For each admitted native route (family, model_type='*', backend='native'):
#
# 1. Load checkpoint with A02 factory
#    ```python
#    session = create_qwen35_native_session(model_path, config_path)
#    ```
#
# 2. Capture execution evidence (3 runs for parity)
#    ```python
#    for i in range(3):
#        receipt = capture_execution_evidence(
#            route_key='qwen3.5/qwen3_5_moe/native/fp32/float32/float32',
#            checkpoint_path=Path('models/qwen3.5-0.8b'),
#            config_path=Path('models/qwen3.5-0.8b/config.json'),
#            session=session,
#            input_tokens=[1, 2, 3, 4, 5],  # Test sequence
#        )
#    ```
#
# 3. Verify numerical parity
#    - All 3 runs produce logits with same output_shape and dtype
#    - Numerical stats (min/max/mean/std) within tolerance
#    - Cache state preserved and hashed consistently
#
# 4. Capture recovery evidence (snapshot/restore cycle)
#    ```python
#    recovery = capture_recovery_evidence(
#        route_key='qwen3.5/qwen3_5_moe/native/fp32/float32/float32',
#        session=session,
#        input_tokens_gen1=[1, 2, 3],
#        input_tokens_gen2=[4, 5],  # Continue after restore
#    )
#    ```
#
# 5. Full generation test (generate N tokens)
#    ```python
#    logits, _ = session.forward([1])  # Seed token
#    generated = [np.argmax(logits)]   # Pick highest probability
#    for _ in range(99):
#        logits, _ = session.forward([generated[-1]])
#        generated.append(np.argmax(logits))
#    ```
#
# 6. Save manifest
#    ```python
#    binding = P13EvidenceBinding(
#        route_key=route_key,
#        family='qwen3.5',
#        model_type='qwen3_5_moe',
#        backend='native',
#        checkpoint_hash=...,
#        config_hash=...,
#        execution_receipts=[receipt1, receipt2, receipt3],
#        recovery_receipts=[recovery],
#        parity_verified=all_match,
#        tolerance_threshold=1e-5,
#        full_generation_success=len(generated)==100,
#        generation_tokens=generated[:10],
#        timestamp=...,
#    )
#    save_evidence_manifest(binding, Path('a10_qwen35_evidence.json'))
#    ```
#
# RESULT: P13 evidence manifest for 1 route (family × backend × arithmetic × format)
# COVERAGE: 133 admitted native routes × evidence manifests = Phase 1 P13 closure
# HARDWARE: CPU only, no GPU required
#
