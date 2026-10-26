"""Retired A15 API scaffold.

This module has no production session wiring. Its status methods report that
state explicitly, and its generation entrypoint refuses placeholder inference.
Production serving is exposed by the application routes and UniversalEngine.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# A15.1: PerformanceMetrics — Track inference metrics
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics for one inference pass."""
    session_id: str
    model_family: str
    backend: str  # 'arc_gpu', 'cpu'

    # Timing (milliseconds)
    forward_ms: float = 0.0
    ttft_ms: float = 0.0  # Time to first token

    # Throughput
    tokens_generated: int = 0
    tokens_per_second: float = 0.0

    # Memory
    peak_memory_mb: float = 0.0
    cache_memory_mb: float = 0.0

    # I/O
    bytes_read: int = 0
    disk_io_ms: float = 0.0

    # Cache efficiency
    cache_hit_rate: float = 0.0
    prefetch_hit_rate: float = 0.0

    # Placement
    experts_in_vram: int = 0
    experts_in_ram: int = 0
    experts_on_disk: int = 0

    timestamp: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


class MetricsCollector:
    """Collect performance metrics across sessions."""

    def __init__(self):
        self.metrics_history = []
        self.session_metrics = {}  # {session_id: [metrics]}

    def record_metric(self, metrics: PerformanceMetrics):
        """Record one inference metric."""
        self.metrics_history.append(metrics)

        if metrics.session_id not in self.session_metrics:
            self.session_metrics[metrics.session_id] = []
        self.session_metrics[metrics.session_id].append(metrics)

    def get_average_metrics(self, session_id: Optional[str] = None) -> Dict:
        """Get average metrics for a session or all."""
        if session_id:
            metrics_list = self.session_metrics.get(session_id, [])
        else:
            metrics_list = self.metrics_history

        if not metrics_list:
            return {}

        avg = {
            'forward_ms': sum(m.forward_ms for m in metrics_list) / len(metrics_list),
            'tokens_per_second': sum(m.tokens_per_second for m in metrics_list) / len(metrics_list),
            'peak_memory_mb': sum(m.peak_memory_mb for m in metrics_list) / len(metrics_list),
            'cache_hit_rate': sum(m.cache_hit_rate for m in metrics_list) / len(metrics_list),
            'prefetch_hit_rate': sum(m.prefetch_hit_rate for m in metrics_list) / len(metrics_list),
            'count': len(metrics_list),
        }
        return avg

    def get_family_metrics(self, model_family: str) -> Dict:
        """Get metrics for a model family."""
        family_metrics = [m for m in self.metrics_history if m.model_family == model_family]

        if not family_metrics:
            return {}

        return {
            'family': model_family,
            'total_inferences': len(family_metrics),
            'avg_tok_per_sec': sum(m.tokens_per_second for m in family_metrics) / len(family_metrics),
            'max_tok_per_sec': max(m.tokens_per_second for m in family_metrics),
            'avg_memory_mb': sum(m.peak_memory_mb for m in family_metrics) / len(family_metrics),
        }


# ============================================================================
# A15.2: Model Adapter Wrappers — Unified interface for all families
# ============================================================================

class ModelAdapterBase:
    """Base adapter for all model families."""

    def __init__(self, family: str, model_config: Dict):
        self.family = family
        self.model_config = model_config
        self.unified_executor = None  # Set by subclass

    def forward(self, input_tokens: List[int]) -> tuple:
        """Unified forward (all models implement this)."""
        raise NotImplementedError

    def snapshot(self) -> Dict:
        """Snapshot state for recovery."""
        raise NotImplementedError

    def restore(self, snapshot: Dict):
        """Restore from snapshot."""
        raise NotImplementedError


class QwenAdapter(ModelAdapterBase):
    """Adapter for Qwen3.5 and Qwen4."""

    def forward(self, input_tokens: List[int]) -> tuple:
        if self.unified_executor is None:
            raise RuntimeError("Unified executor not initialized")

        logits, metadata = self.unified_executor.forward(input_tokens)
        return logits, metadata


class GLMAdapter(ModelAdapterBase):
    """Adapter for GLM5, GLM-DSA, Inkling."""

    def forward(self, input_tokens: List[int]) -> tuple:
        if self.unified_executor is None:
            raise RuntimeError("Unified executor not initialized")

        logits, metadata = self.unified_executor.forward(input_tokens)
        return logits, metadata


class DeepSeekAdapter(ModelAdapterBase):
    """Adapter for DeepSeek V4.1."""

    def forward(self, input_tokens: List[int]) -> tuple:
        if self.unified_executor is None:
            raise RuntimeError("Unified executor not initialized")

        logits, metadata = self.unified_executor.forward(input_tokens)
        return logits, metadata


class MixtralAdapter(ModelAdapterBase):
    """Adapter for Mixtral."""

    def forward(self, input_tokens: List[int]) -> tuple:
        if self.unified_executor is None:
            raise RuntimeError("Unified executor not initialized")

        logits, metadata = self.unified_executor.forward(input_tokens)
        return logits, metadata


class K3Adapter(ModelAdapterBase):
    """Adapter for Kimi K3."""

    def forward(self, input_tokens: List[int]) -> tuple:
        if self.unified_executor is None:
            raise RuntimeError("Unified executor not initialized")

        logits, metadata = self.unified_executor.forward(input_tokens)
        return logits, metadata


class OLMoEAdapter(ModelAdapterBase):
    """Adapter for OLMoE."""

    def forward(self, input_tokens: List[int]) -> tuple:
        if self.unified_executor is None:
            raise RuntimeError("Unified executor not initialized")

        logits, metadata = self.unified_executor.forward(input_tokens)
        return logits, metadata


# ============================================================================
# A15.3: API Endpoints (FastAPI compatible)
# ============================================================================

class AdaptiveEngineAPI:
    """API endpoints for adaptive engine."""

    def __init__(self, metrics_collector: MetricsCollector = None):
        self.metrics = metrics_collector or MetricsCollector()
        self.model_adapters = {}

    def get_placement(self) -> Dict:
        """Return an explicit empty observation until a production owner is attached."""
        return {'layers': {}, 'observed': False,
                'reason': 'legacy A15 facade has no production expert-store owner'}

    def get_routing_heat(self) -> Dict:
        """Which experts are firing most."""
        return {'layers': {}, 'observed': False,
                'reason': 'legacy A15 facade has no production route-trace source'}

    def get_learning_stats(self) -> Dict:
        """Learning cache statistics."""
        return {'observed': False, 'batches_processed': None,
                'hot_experts_pinned': None, 'disk_reads_saved': None,
                'prefetch_hit_rate': None, 'avg_batch_size': None,
                'reason': 'legacy A15 facade has no production learning-cache owner'}

    def get_session_metrics(self, session_id: str) -> Dict:
        """Metrics for a specific session."""
        avg = self.metrics.get_average_metrics(session_id)
        return avg or {'error': 'Session not found'}

    def get_family_metrics(self, family: str) -> Dict:
        """Metrics for a model family."""
        metrics = self.metrics.get_family_metrics(family)
        return metrics or {'error': 'Family not found'}

    def get_hardware_plan(self) -> Dict:
        """Return measured basic host capacity; never invent a tuning plan."""
        arc_gpu_available = None
        try:
            import torch
            arc_gpu_available = bool(hasattr(torch, 'xpu') and torch.xpu.is_available())
        except ImportError:
            pass

        try:
            import psutil
            cpu_cores = psutil.cpu_count()
            ram_available_gb = psutil.virtual_memory().available / (1024 ** 3)
        except ImportError:
            cpu_cores = None
            ram_available_gb = None

        return {
            'cpu_cores': cpu_cores,
            'ram_available_gb': ram_available_gb,
            'disk_paths': [],
            'disk_bandwidth_mbps': None,
            'arc_gpu_detected': arc_gpu_available,
            'arc_gpu_probe_available': arc_gpu_available is not None,
            'npu_available': None,
            'recommendation': None,
            'observed': cpu_cores is not None and ram_available_gb is not None,
        }

    def list_models(self) -> Dict:
        """List all loaded models."""
        return {'models': [], 'total': 0, 'unified_path': False,
                'observed': False,
                'reason': 'legacy A15 facade does not own production universal sessions'}

    def get_status(self) -> Dict:
        """Overall adaptive engine status."""
        return {
            'status': 'unavailable',
            'unified_inference': False,
            'phase1_complete': False,
            'speedup_enabled': False,
            'prefetch_enabled': False,
            'learning_cache_enabled': False,
            'reason': 'legacy A15 facade is not wired to the production universal engine',
            'timestamp': time.time(),
        }


# ============================================================================
# A15.4: Full Integration Example
# ============================================================================

class AdaptiveEngine:
    """Compatibility shell retained only to produce explicit migration errors."""

    def __init__(self, model_configs: Dict[str, Dict]):
        self.model_configs = model_configs
        self.adapters = {}
        self.metrics = MetricsCollector()
        self.api = AdaptiveEngineAPI(self.metrics)

        self._init_adapters()

    def _init_adapters(self):
        """Initialize adapters for all families."""
        adapter_classes = {
            'qwen3.5': QwenAdapter,
            'qwen4': QwenAdapter,
            'glm5': GLMAdapter,
            'glm_dsa': GLMAdapter,
            'inkling': GLMAdapter,
            'deepseek_v41': DeepSeekAdapter,
            'mixtral': MixtralAdapter,
            'k3': K3Adapter,
            'olmoe': OLMoEAdapter,
        }

        for family, config in self.model_configs.items():
            adapter_class = adapter_classes.get(family, ModelAdapterBase)
            self.adapters[family] = adapter_class(family, config)
            logger.debug("Created retired A15 configuration shell for %s", family)

    def generate(self, family: str, prompt: str, max_tokens: int = 100) -> str:
        """
        Generate text using unified path (all families identical).
        """
        raise RuntimeError(
            "legacy A15 generation is disabled: it used placeholder tokenization and "
            "did not execute model weights; use AdaptiveExecutionEngine.open_universal_model()")

    def start(self):
        """Start adaptive engine (initialize prefetch, etc)."""
        raise RuntimeError(
            "legacy A15 adapters are configuration shells, not loaded models; "
            "use AdaptiveExecutionEngine.open_universal_model()")

    def stop(self):
        """Stop adaptive engine (save learning cache)."""
        logger.info("Stopping adaptive engine")

    def get_metrics(self, family: Optional[str] = None) -> Dict:
        """Get performance metrics."""
        if family:
            return self.metrics.get_family_metrics(family)
        return self.metrics.get_average_metrics()


if __name__ == '__main__':
    raise SystemExit(
        "A15 is a retired compatibility scaffold; start the application and "
        "use its production inference routes backed by UniversalEngine")
