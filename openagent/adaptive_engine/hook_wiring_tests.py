"""Phase 1F smoke tests — run without Torch or GPU hardware.

Each test is a standalone function that returns True on pass, False on fail.
Run all tests::

    python -m openagent.adaptive_engine.phase1f_smoke_tests

"""
from __future__ import annotations

import sys
import traceback
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tiny stub sessions for testing without real model weights
# ---------------------------------------------------------------------------

class _StubBaseSession:
    def __init__(self):
        self.on_expert_dispatch = None
        self.vocab_size = 128
        self.id_to_text: Dict[int, str] = {i: str(i) for i in range(128)}
        self._stores = ()
        self.state_directory = None
        self.manifest = None

    def generate(self, *a, **kw):
        return [1, 2, 3]

    def stream(self, *a, **kw):
        yield from [1, 2, 3]

    def close(self): pass

    def metrics(self): return {"base": True}


class _StubP1BSession:
    """Minimal Phase 1B duck-type for testing without real model weights."""
    def __init__(self, state_dir: Path) -> None:
        self.base_session = _StubBaseSession()
        self._stores = ()
        self._closed = False
        self.state_directory = state_dir
        self.manifest = None
        import threading; self._lock = threading.RLock()

    def generate(self, *a, **kw): return [1, 2, 3]
    def stream(self, *a, **kw): yield from [1, 2, 3]
    def close(self): pass
    def metrics(self): return {"phase1b": True}


def _build_stack(tmp_dir: Path) -> Any:
    """Build the full 1C→1D→1E session stack using a stub 1B session."""
    import threading
    from openagent.adaptive_engine.phase1c_adapter import Phase1COptimizedSession
    from openagent.adaptive_engine.phase1d_adapter import Phase1DOptimizedSession
    from openagent.adaptive_engine.phase1e_adapter import Phase1EOptimizedSession
    from openagent.adaptive_engine.phase1c_extended_prefetch import AffinityClusterMap, ExtendedPrefetchController
    from openagent.adaptive_engine.phase1c_model_kernels import Phase1CKernelRegistry

    p1b = _StubP1BSession(tmp_dir)

    p1c = Phase1COptimizedSession.__new__(Phase1COptimizedSession)
    p1c.base_session = p1b
    p1c._prefetch_depth = 3
    p1c._lock = threading.RLock()
    p1c._closed = False
    p1c._phase1c_saved = False
    p1c._affinity_map = AffinityClusterMap()
    p1c._affinity_path = None
    p1c._prefetch_ctrl = ExtendedPrefetchController(affinity_map=p1c._affinity_map, depth=3)
    p1c.kernels = Phase1CKernelRegistry()
    p1c._direct_io = None
    p1c._current_trace = []

    p1d = Phase1DOptimizedSession(p1c)
    p1e = Phase1EOptimizedSession(p1d, enable_arc_gpu=False)
    return p1e


# ---------------------------------------------------------------------------
# Individual smoke tests
# ---------------------------------------------------------------------------

def _test_phase1e_session_create(tmp_dir: Path) -> bool:
    """Phase 1E session opens and closes cleanly."""
    sess = _build_stack(tmp_dir)
    m = sess.metrics()
    assert "phase1e" in m, "metrics must include phase1e key"
    sess.close()
    return True


def _test_phase1e_generate(tmp_dir: Path) -> bool:
    """Phase 1E generate() returns tokens."""
    sess = _build_stack(tmp_dir)
    result = sess.generate(context_ids=[1, 2, 3])
    assert result is not None, "generate must return something"
    sess.close()
    return True


def _test_phase1e_stream(tmp_dir: Path) -> bool:
    """Phase 1E stream() yields tokens."""
    sess = _build_stack(tmp_dir)
    tokens = list(sess.stream(context_ids=[1, 2, 3]))
    assert len(tokens) > 0, "stream must yield at least one token"
    sess.close()
    return True


def _test_aot_cache(tmp_dir: Path) -> bool:
    """AotKernelCache precompile runs without error."""
    from openagent.adaptive_engine.phase1e_aot_kernels import AotKernelCache
    cache = AotKernelCache(tmp_dir / "aot")
    result = cache.precompile(force=True)
    # result is True with Numba, False without — both are acceptable
    assert isinstance(result, bool)
    return True


def _test_avx_dispatch(tmp_dir: Path) -> bool:
    """AvxSwiGLUDispatch produces correct-shaped output."""
    from openagent.adaptive_engine.phase1e_aot_kernels import AvxSwiGLUDispatch
    avx = AvxSwiGLUDispatch()
    B, D, F = 2, 64, 32
    h = np.random.randn(B, D).astype(np.float32)
    g = np.random.randn(F, D).astype(np.float32)
    u = np.random.randn(F, D).astype(np.float32)
    d = np.random.randn(D, F).astype(np.float32)
    out = avx.forward(h, g, u, d)
    assert out.shape == (B, D), f"expected ({B},{D}) got {out.shape}"
    return True


def _test_hook_wiring_native(tmp_dir: Path) -> bool:
    """NativeRoutingBridge wires and fires on dispatch."""
    from openagent.adaptive_engine.phase1f_hook_wiring import NativeRoutingBridge

    received: List[Tuple[int, List[str]]] = []

    class _NativeSess:
        on_expert_dispatch = None

    native = _NativeSess()
    bridge = NativeRoutingBridge(native, lambda lid, eids: received.append((lid, eids)))
    ok = bridge.wire()
    assert ok, "wire() should succeed"
    assert native.on_expert_dispatch is not None
    native.on_expert_dispatch(3, ["e0", "e1"])
    assert received == [(3, ["e0", "e1"])], f"unexpected {received}"
    bridge.remove()
    return True


def _test_wire_session_native(tmp_dir: Path) -> bool:
    """wire_session with a native session calls on_token_routed."""
    from openagent.adaptive_engine.phase1f_hook_wiring import wire_session

    routed: List[Tuple[int, List[str]]] = []

    class _FakeOptimized:
        def on_token_routed(self, lid, eids):
            routed.append((lid, eids))

    class _FakeNative:
        on_expert_dispatch = None

    opt = _FakeOptimized()
    nat = _FakeNative()
    registry = wire_session(opt, native_session=nat)
    nat.on_expert_dispatch(1, ["e5"])
    assert routed == [(1, ["e5"])], f"unexpected {routed}"
    registry.close()
    return True


def _test_detect_family(tmp_dir: Path) -> bool:
    """detect_family infers family from class name."""
    from openagent.adaptive_engine.phase1f_hook_wiring import detect_family

    class OlmoeForCausalLM: pass
    class MixtralForCausalLM: pass
    class DeepseekV4ForCausalLM: pass

    assert detect_family(OlmoeForCausalLM()) == "olmoe"
    assert detect_family(MixtralForCausalLM()) == "mixtral"
    assert detect_family(DeepseekV4ForCausalLM()) == "deepseek"
    return True


def _test_phase1f_exports(tmp_dir: Path) -> bool:
    """Phase 1F exports are importable from adaptive_engine."""
    import importlib
    mod = importlib.import_module("openagent.adaptive_engine.phase1f_hook_wiring")
    for name in ("detect_family", "FamilyHookWirer", "NativeRoutingBridge",
                 "HookWiringRegistry", "wire_session", "wire_batch_logits"):
        assert hasattr(mod, name), f"missing export: {name}"
    return True


def _test_full_stack_metrics(tmp_dir: Path) -> bool:
    """Full stack metrics contain keys from all phases."""
    sess = _build_stack(tmp_dir)
    m = sess.metrics()
    for key in ("phase1d", "phase1e"):
        assert key in m, f"missing metrics key: {key}"
    sess.close()
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_TESTS: List[Tuple[str, Callable[[Path], bool]]] = [
    ("phase1e_session_create",  _test_phase1e_session_create),
    ("phase1e_generate",        _test_phase1e_generate),
    ("phase1e_stream",          _test_phase1e_stream),
    ("aot_cache",               _test_aot_cache),
    ("avx_dispatch",            _test_avx_dispatch),
    ("hook_wiring_native",      _test_hook_wiring_native),
    ("wire_session_native",     _test_wire_session_native),
    ("detect_family",           _test_detect_family),
    ("phase1f_exports",         _test_phase1f_exports),
    ("full_stack_metrics",      _test_full_stack_metrics),
]


def run_all(verbose: bool = True) -> Dict[str, bool]:
    import tempfile
    results: Dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="phase1f_smoke_") as td:
        tmp = Path(td)
        for name, fn in _TESTS:
            try:
                ok = fn(tmp)
            except Exception:
                ok = False
                if verbose:
                    print(f"  FAIL  {name}")
                    traceback.print_exc()
            else:
                if verbose:
                    status = "  PASS" if ok else "  FAIL"
                    print(f"{status}  {name}")
            results[name] = ok
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    res = run_all(verbose=True)
    passed = sum(v for v in res.values())
    total  = len(res)
    print(f"\n{passed}/{total} tests passed")
    sys.exit(0 if passed == total else 1)
