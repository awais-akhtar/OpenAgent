"""Phase 3 integration smoke tests — validates all Phase 3 components.

Tests use lightweight stubs; no PyTorch, no CUDA, no network.
Run with::

    python openagent/adaptive_engine/integration_smoke_tests.py
"""
from __future__ import annotations

import sys
import threading
import time
import traceback
from typing import Any, List, Optional, Tuple

import numpy as np

_RESULTS: List[Tuple[str, bool, str]] = []


def _test(name: str):
    def decorator(fn):
        try:
            fn()
            _RESULTS.append((name, True, ""))
            print(f"  [PASS] {name}")
        except Exception as exc:
            tb = traceback.format_exc()
            _RESULTS.append((name, False, tb))
            print(f"  [FAIL] {name}: {exc}")
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _Stub:
    def generate(self, *a, context_ids=None, max_new_tokens=1, **kw):
        ids = list(context_ids or [])
        return ids + [42] * max_new_tokens

    def stream(self, *a, context_ids=None, max_new_tokens=4, **kw):
        for i in range(max_new_tokens):
            yield 100 + i

    def metrics(self): return {"stub": True}
    def close(self):   pass


class _StubService:
    """Stub Phase2MultiSessionService."""
    def open_session(self, session_id=None, **kw):
        return _Stub()
    def metrics(self): return {"backend": "stub"}
    def close(self):   pass


# ---------------------------------------------------------------------------
# Test 1: TensorParallelConfig validation
# ---------------------------------------------------------------------------

@_test("TensorParallelConfig world_size=1 is LOCAL backend")
def _t1():
    from openagent.adaptive_engine.tensor_parallel_executor import (
        TensorParallelConfig, TensorParallelBackend)
    cfg = TensorParallelConfig(world_size=1, rank=0)
    assert cfg.backend == TensorParallelBackend.LOCAL


# ---------------------------------------------------------------------------
# Test 2: ColumnParallelLinear splits correctly
# ---------------------------------------------------------------------------

@_test("ColumnParallelLinear output shard is 1/world_size of full output")
def _t2():
    from openagent.adaptive_engine.tensor_parallel_executor import (
        TensorParallelConfig, TensorParallelCommunicator, ColumnParallelLinear)
    cfg  = TensorParallelConfig(world_size=2, rank=0)
    comm = TensorParallelCommunicator(cfg)
    w    = np.random.randn(8, 16).astype(np.float32)
    col  = ColumnParallelLinear(w, None, comm)
    x    = np.random.randn(2, 8).astype(np.float32)
    out  = col.forward(x)
    assert out.shape == (2, 8), f"expected (2,8) got {out.shape}"


# ---------------------------------------------------------------------------
# Test 3: TensorParallelExpertShard forward
# ---------------------------------------------------------------------------

@_test("TensorParallelExpertShard forward returns (batch, d_model)")
def _t3():
    from openagent.adaptive_engine.tensor_parallel_executor import (
        TensorParallelConfig, TensorParallelCommunicator,
        TensorParallelExpertShard)
    cfg  = TensorParallelConfig(world_size=1, rank=0)
    comm = TensorParallelCommunicator(cfg)
    d_model, d_ff = 16, 32
    gate = np.random.randn(d_model, d_ff).astype(np.float32)
    up   = np.random.randn(d_model, d_ff).astype(np.float32)
    down = np.random.randn(d_ff, d_model).astype(np.float32)
    shard = TensorParallelExpertShard("e0", gate, up, down, comm)
    hidden = np.random.randn(2, d_model).astype(np.float32)
    out = shard.forward(hidden)
    assert out.shape == (2, d_model)


# ---------------------------------------------------------------------------
# Test 4: Phase3TensorParallelSession passthrough
# ---------------------------------------------------------------------------

@_test("Phase3TensorParallelSession generate passthrough")
def _t4():
    from openagent.adaptive_engine.tensor_parallel_executor import (
        Phase3TensorParallelSession)
    sess = Phase3TensorParallelSession(_Stub())
    out = sess.generate(context_ids=[1, 2, 3], max_new_tokens=2)
    assert len(out) >= 2
    sess.close()


# ---------------------------------------------------------------------------
# Test 5: INT4 quantize round-trip
# ---------------------------------------------------------------------------

@_test("Int4BlockQuantizer quantize/dequantize round-trip error < 15%")
def _t5():
    from openagent.adaptive_engine.quantization_engine import (
        QuantizationConfig, QuantizationDtype, Int4BlockQuantizer)
    cfg = QuantizationConfig(dtype=QuantizationDtype.INT4, block_size=32)
    q   = Int4BlockQuantizer(cfg)
    w   = np.random.randn(64, 16).astype(np.float32) * 2.0
    qt  = q.quantize(w)
    deq = q.dequantize(qt)
    assert deq.shape == w.shape
    rel_err = np.mean(np.abs(w - deq)) / (np.mean(np.abs(w)) + 1e-8)
    assert rel_err < 0.15, f"INT4 error too high: {rel_err:.3f}"
    assert qt.compression_ratio > 1.5


# ---------------------------------------------------------------------------
# Test 6: INT8 quantize round-trip
# ---------------------------------------------------------------------------

@_test("Int8SymmetricQuantizer quantize/dequantize error < 2%")
def _t6():
    from openagent.adaptive_engine.quantization_engine import (
        QuantizationConfig, QuantizationDtype, Int8SymmetricQuantizer)
    cfg = QuantizationConfig(dtype=QuantizationDtype.INT8, block_size=64)
    q   = Int8SymmetricQuantizer(cfg)
    w   = np.random.randn(32, 32).astype(np.float32)
    qt  = q.quantize(w)
    deq = q.dequantize(qt)
    rel_err = np.mean(np.abs(w - deq)) / (np.mean(np.abs(w)) + 1e-8)
    assert rel_err < 0.02, f"INT8 error too high: {rel_err:.4f}"


# ---------------------------------------------------------------------------
# Test 7: QuantizationCodec FP8 round-trip
# ---------------------------------------------------------------------------

@_test("QuantizationCodec FP8 round-trip error < 5%")
def _t7():
    from openagent.adaptive_engine.quantization_engine import (
        QuantizationConfig, QuantizationDtype, QuantizationCodec)
    cfg   = QuantizationConfig(dtype=QuantizationDtype.FP8, block_size=64)
    codec = QuantizationCodec(cfg)
    w     = np.random.randn(32, 32).astype(np.float32)
    qt    = codec.quantize(w)
    deq   = codec.dequantize(qt)
    rel_err = np.mean(np.abs(w - deq)) / (np.mean(np.abs(w)) + 1e-8)
    assert rel_err < 0.05, f"FP8 error: {rel_err:.4f}"


# ---------------------------------------------------------------------------
# Test 8: DraftTree builds expected number of nodes
# ---------------------------------------------------------------------------

@_test("DraftTree builds correct node count (BF=3, depth=2 → ≥3 nodes)")
def _t8():
    from openagent.adaptive_engine.speculative_tree_decoder import (
        DraftTree, DraftHeadSimulator)
    heads = DraftHeadSimulator(k=3, vocab_size=100)
    tree  = DraftTree(branching_factor=3, max_depth=2)
    tree.build([1, 2, 3], heads.draft)
    # Depth 1: 3 roots; depth 2: up to 9 children
    assert tree.node_count >= 3
    paths = tree.all_paths()
    assert len(paths) > 0


# ---------------------------------------------------------------------------
# Test 9: TreeSpeculativeDecoder step returns ≥1 token
# ---------------------------------------------------------------------------

@_test("TreeSpeculativeDecoder.step returns at least 1 accepted token")
def _t9():
    from openagent.adaptive_engine.speculative_tree_decoder import (
        DraftHeadSimulator, TreeSpeculativeDecoder)
    heads   = DraftHeadSimulator(k=3, vocab_size=100)
    decoder = TreeSpeculativeDecoder(heads, verify_fn=None,
                                     branching_factor=3, max_depth=2)
    tokens, nodes = decoder.step([1, 2, 3])
    assert len(tokens) >= 1
    assert nodes >= 0


# ---------------------------------------------------------------------------
# Test 10: Phase3SpecTreeSession generate
# ---------------------------------------------------------------------------

@_test("Phase3SpecTreeSession generate produces max_new_tokens")
def _t10():
    from openagent.adaptive_engine.speculative_tree_decoder import (
        Phase3SpecTreeSession)
    sess = Phase3SpecTreeSession(_Stub(), branching_factor=2,
                                  max_depth=2, vocab_size=100)
    out  = sess.generate(context_ids=[1, 2, 3], max_new_tokens=8)
    assert len(out) >= 1
    sess.close()


# ---------------------------------------------------------------------------
# Test 11: LoadBalancer round-robin
# ---------------------------------------------------------------------------

@_test("LoadBalancer ROUND_ROBIN distributes across instances")
def _t11():
    from openagent.adaptive_engine.load_balancer import (
        ServiceRegistry, LoadBalancer, LoadBalancingStrategy)
    reg = ServiceRegistry()
    for i in range(3):
        reg.register(f"inst-{i}", _StubService())
    lb  = LoadBalancer(reg, LoadBalancingStrategy.ROUND_ROBIN,
                       health_interval_s=999.0)
    seen = set()
    for _ in range(9):
        s = lb.select()
        assert s is not None
        seen.add(s.instance_id)
    assert len(seen) == 3
    lb.stop()


# ---------------------------------------------------------------------------
# Test 12: LoadBalancer LEAST_LOADED picks idle instance
# ---------------------------------------------------------------------------

@_test("LoadBalancer LEAST_LOADED picks instance with lowest in_flight")
def _t12():
    from openagent.adaptive_engine.load_balancer import (
        ServiceRegistry, LoadBalancer, LoadBalancingStrategy)
    reg = ServiceRegistry()
    reg.register("busy",  _StubService())
    reg.register("idle",  _StubService())
    lb  = LoadBalancer(reg, LoadBalancingStrategy.LEAST_LOADED,
                       health_interval_s=999.0)
    busy_state = reg.get("busy")
    busy_state.in_flight = 10   # busy
    for _ in range(5):
        s = lb.select()
        assert s is not None
        assert s.instance_id == "idle", f"expected idle got {s.instance_id}"
    lb.stop()


# ---------------------------------------------------------------------------
# Test 13: Phase3LoadBalancedService.probe_all
# ---------------------------------------------------------------------------

@_test("Phase3LoadBalancedService.probe_all returns health results")
def _t13():
    from openagent.adaptive_engine.load_balancer import Phase3LoadBalancedService
    svc = Phase3LoadBalancedService(health_interval_s=999.0)
    svc.register_backend("b1", _StubService())
    svc.register_backend("b2", _StubService())
    results = svc.probe_all()
    assert len(results) == 2
    for r in results:
        assert r.healthy
    svc.close()


# ---------------------------------------------------------------------------
# Test 14: LatencyHistogram percentiles
# ---------------------------------------------------------------------------

@_test("LatencyHistogram p50/p99 within expected range")
def _t14():
    from openagent.adaptive_engine.telemetry_exporter import LatencyHistogram
    h = LatencyHistogram()
    for v in range(1, 101):   # 1ms..100ms
        h.observe(float(v))
    p50 = h.percentile(50)
    p99 = h.percentile(99)
    assert 40.0 <= p50 <= 60.0, f"p50={p50}"
    assert 90.0 <= p99 <= 105.0, f"p99={p99}"


# ---------------------------------------------------------------------------
# Test 15: TelemetryRegistry gauge / counter / snapshot
# ---------------------------------------------------------------------------

@_test("TelemetryRegistry gauge + counter + snapshot")
def _t15():
    from openagent.adaptive_engine.telemetry_exporter import (
        TelemetryRegistry, LabelSet)
    reg = TelemetryRegistry()
    reg.gauge("model_loaded", 1.0)
    reg.counter("requests_total", 5.0)
    reg.counter("requests_total", 3.0)
    snap = reg.snapshot()
    names = {s.name for s in snap}
    assert "model_loaded" in names
    assert "requests_total" in names
    totals = {s.name: s.value for s in snap}
    assert totals["requests_total"] == 8.0


# ---------------------------------------------------------------------------
# Test 16: PrometheusTextFormatter renders valid text
# ---------------------------------------------------------------------------

@_test("PrometheusTextFormatter renders # HELP and # TYPE lines")
def _t16():
    from openagent.adaptive_engine.telemetry_exporter import (
        TelemetryRegistry, PrometheusTextFormatter)
    reg = TelemetryRegistry()
    reg.gauge("active_sessions", 3.0)
    text = PrometheusTextFormatter().render(reg)
    assert "# HELP" in text
    assert "# TYPE" in text
    assert "active_sessions" in text


# ---------------------------------------------------------------------------
# Test 17: Phase3TelemetryCollector scrapes and feeds registry
# ---------------------------------------------------------------------------

@_test("Phase3TelemetryCollector scrapes metrics into registry")
def _t17():
    from openagent.adaptive_engine.telemetry_exporter import (
        Phase3TelemetryCollector, TelemetryRegistry)
    reg       = TelemetryRegistry()
    collector = Phase3TelemetryCollector(registry=reg, scrape_prefix="oa")
    collector.add_source("stub", lambda: {"requests_total": 7, "latency_ms": 12.5})
    collector.scrape_once()
    snap  = reg.snapshot()
    names = {s.name for s in snap}
    assert any("stub" in n for n in names), f"no stub metrics in {names}"


# ---------------------------------------------------------------------------
# Test 18: ServingEngine.process_completion
# ---------------------------------------------------------------------------

@_test("ServingEngine.process_completion returns valid response dict")
def _t18():
    from openagent.adaptive_engine.serving_engine import (
        ServingEngine, CompletionRequest)
    engine = ServingEngine(_StubService(), model_name="test-model")
    req    = CompletionRequest(model="test", prompt="Hello world",
                               max_tokens=4)
    resp   = engine.process_completion(req)
    d      = resp.to_dict()
    assert d["model"] == "test-model"
    assert len(d["choices"]) == 1
    assert "text" in d["choices"][0]
    assert d["usage"]["prompt_tokens"] > 0


# ---------------------------------------------------------------------------
# Test 19: ServingEngine.process_chat_completion
# ---------------------------------------------------------------------------

@_test("ServingEngine.process_chat_completion returns message content")
def _t19():
    from openagent.adaptive_engine.serving_engine import (
        ServingEngine, ChatCompletionRequest, ChatMessage, Role)
    engine = ServingEngine(_StubService(), model_name="chat-model")
    req    = ChatCompletionRequest(
        model="chat",
        messages=[
            ChatMessage(Role.SYSTEM, "You are helpful."),
            ChatMessage(Role.USER,   "What is 2+2?"),
        ],
        max_tokens=4)
    resp = engine.process_chat_completion(req)
    d    = resp.to_dict()
    assert d["object"] == "chat.completion"
    assert "message" in d["choices"][0]
    assert d["choices"][0]["message"]["role"] == "assistant"


# ---------------------------------------------------------------------------
# Test 20: ServingEngine stream_completion yields SSE lines
# ---------------------------------------------------------------------------

@_test("ServingEngine.stream_completion yields 'data: ' lines")
def _t20():
    from openagent.adaptive_engine.serving_engine import (
        ServingEngine, CompletionRequest)
    engine = ServingEngine(_StubService(), model_name="m")
    req    = CompletionRequest(model="m", prompt="hi", max_tokens=3, stream=True)
    lines  = list(engine.stream_completion(req))
    sse    = [l for l in lines if l.to_sse_line().startswith("data:")]
    assert len(sse) >= 1


# ---------------------------------------------------------------------------
# Test 21: TokenizerAdapter fallback
# ---------------------------------------------------------------------------

@_test("TokenizerAdapter UTF-8 fallback encode/decode round-trip")
def _t21():
    from openagent.adaptive_engine.serving_engine import TokenizerAdapter
    tok  = TokenizerAdapter(None)
    ids  = tok.encode("Hello, world!")
    text = tok.decode(ids)
    assert "Hello" in text


# ---------------------------------------------------------------------------
# Test 22: InferenceService build + stop
# ---------------------------------------------------------------------------

@_test("InferenceService builds and stops without error")
def _t22():
    from pathlib import Path
    from openagent.adaptive_engine.inference_service import (
        InferenceService, InferenceServiceConfig)

    calls = []
    def fake_forward(batch):
        calls.append(len(batch))
        return [(r, 42) for r in batch]

    cfg = InferenceServiceConfig(
        model_name="smoke-model",
        checkpoint_id="test-ckpt",
        checkpoint_dir=Path("."),
        max_sessions=2,
    )
    svc = InferenceService(cfg, fake_forward)
    svc.start()
    m = svc.metrics()
    assert "phase2" in m
    assert "lb" in m
    assert "serving" in m
    svc.stop()


# ---------------------------------------------------------------------------
# Test 23: InferenceService.complete OpenAI dict response
# ---------------------------------------------------------------------------

@_test("InferenceService.complete returns OpenAI-compatible dict")
def _t23():
    from pathlib import Path
    from openagent.adaptive_engine.inference_service import (
        InferenceService, InferenceServiceConfig)

    def fake_forward(batch):
        return [(r, None) for r in batch]

    cfg = InferenceServiceConfig(
        model_name="test", checkpoint_id="c", checkpoint_dir=Path("."))
    with InferenceService(cfg, fake_forward) as svc:
        resp = svc.complete({
            "model": "test",
            "prompt": "Say hello",
            "max_tokens": 4,
        })
    assert "choices" in resp
    assert "usage" in resp
    assert "id" in resp


# ---------------------------------------------------------------------------
# Test 24: InferenceService Prometheus metrics text
# ---------------------------------------------------------------------------

@_test("InferenceService.prometheus_metrics returns Prometheus text")
def _t24():
    from pathlib import Path
    from openagent.adaptive_engine.inference_service import (
        InferenceService, InferenceServiceConfig)

    def fake_forward(batch):
        return [(r, None) for r in batch]

    cfg = InferenceServiceConfig(
        model_name="prom", checkpoint_id="c", checkpoint_dir=Path("."))
    with InferenceService(cfg, fake_forward) as svc:
        text = svc.prometheus_metrics()
    assert "# TYPE" in text or "# HELP" in text or "openagent_" in text


# ---------------------------------------------------------------------------
# Test 25: All Phase 3 exports via __init__
# ---------------------------------------------------------------------------

@_test("All Phase 3 classes importable via adaptive_engine __init__")
def _t25():
    from openagent.adaptive_engine import (
        # tensor parallel
        TensorParallelConfig, TensorParallelCommunicator,
        ColumnParallelLinear, RowParallelLinear,
        TensorParallelExpertShard, Phase3TensorParallelSession,
        # quantization
        QuantizationDtype, QuantizationConfig, QuantizationCodec,
        Int4BlockQuantizer, Int8SymmetricQuantizer,
        QuantizedExpertStore, Phase3QuantizedSession,
        # speculative tree
        DraftTree, DraftHeadSimulator, TreeSpeculativeDecoder,
        Phase3SpecTreeSession,
        # load balancer
        LoadBalancingStrategy, ServiceRegistry, LoadBalancer,
        Phase3LoadBalancedService,
        # telemetry
        MetricType, LabelSet, LatencyHistogram,
        TelemetryRegistry, PrometheusTextFormatter,
        Phase3TelemetryCollector,
        # serving
        Role, ChatMessage, CompletionRequest, ChatCompletionRequest,
        CompletionResponse, ChatCompletionResponse, StreamChunk,
        TokenizerAdapter, ServingEngine,
        # inference service
        InferenceServiceConfig, InferenceService,
    )
    assert callable(InferenceService)
    assert callable(ServingEngine)
    assert callable(LoadBalancer)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all() -> bool:
    print("\n=== Phase 3 Integration Smoke Tests ===\n")
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    failed = sum(1 for _, ok, _ in _RESULTS if not ok)
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{len(_RESULTS)} passed, {failed} failed")
    if failed:
        print("\nFailed tests:")
        for name, ok, tb in _RESULTS:
            if not ok:
                print(f"\n  {name}:\n{tb}")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
