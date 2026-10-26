"""Retired Phase 1B routing/placement scaffold.

``TieredExpertStore`` remains a small explicit-path component utility. The
``UnifiedInferenceExecution`` facade is not a model executor and fails closed;
production generation uses exact family adapters through ``UniversalEngine``.
"""

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# A12.1: TieredExpertStore — 3-tier expert placement (VRAM/RAM/disk)
# ============================================================================

@dataclass
class ExpertPlacement:
    """Expert location in memory hierarchy."""
    expert_id: str
    tier: str  # 'vram', 'ram', 'disk'
    device: Optional[str] = None  # 'xpu:0' for GPU, 'cpu' for RAM
    location: Optional[str] = None  # Disk path or memory address


class TieredExpertStore:
    """
    3-tier expert placement for Colibri speedup:
    - VRAM: GPU device memory (Intel Arc GPU if available)
    - RAM: Learned hot-store (hottest K experts, pinned based on routing history)
    - Disk: NVMe primary storage (model weights on disk)

    Uses learning cache to pin hot experts in RAM, reducing disk I/O.
    """

    def __init__(
        self,
        model_path: Path,
        learning_cache_path: Optional[Path] = None,
        max_ram_experts: int = 16,
        heat_threshold: int = 100,
    ):
        self.model_path = Path(model_path)
        # No shared default: routing history is persisted only to an explicit,
        # caller-owned path so separate checkpoints never share a file.
        self.learning_cache_path = Path(learning_cache_path) if learning_cache_path is not None else None
        self.max_ram_experts = max_ram_experts
        self.heat_threshold = heat_threshold

        # Memory tiers
        self.vram_store = {}  # GPU memory (limited)
        self.ram_hot_store = OrderedDict()  # LRU cache of hottest experts
        self.disk_path = model_path

        # Learning cache: {layer: {expert_id: fire_count}}
        self.learning_cache = {}
        self._load_learning_cache()

        # Thread safety
        self._lock = threading.RLock()

    def _load_learning_cache(self):
        """Load .openagent_usage file (routing history from previous runs)."""
        if self.learning_cache_path is None:
            return
        try:
            if self.learning_cache_path.exists():
                with open(self.learning_cache_path, 'r') as f:
                    self.learning_cache = json.load(f)
                logger.info(f"Loaded learning cache: {len(self.learning_cache)} layers")
        except Exception as e:
            logger.warning(f"Could not load learning cache: {e}")
            self.learning_cache = {}

    def get_placement(self, expert_ids: List[str], layer_id: int) -> Dict[str, ExpertPlacement]:
        """
        Determine placement for each expert in this batch.

        Priority:
        1. VRAM: If Arc GPU available and expert already there
        2. RAM: If expert is hot (high fire count in learning cache)
        3. DISK: Default fallback

        Args:
            expert_ids: Expert IDs fired in this batch
            layer_id: Layer number (for learning cache lookup)

        Returns:
            {expert_id: ExpertPlacement}
        """
        with self._lock:
            placements = {}

            for expert_id in expert_ids:
                # Check VRAM first
                if expert_id in self.vram_store:
                    placements[expert_id] = ExpertPlacement(
                        expert_id=expert_id,
                        tier='vram',
                        device='xpu:0',
                    )
                    continue

                # Check if should be in RAM (hot expert)
                if self._should_load_to_ram(expert_id, layer_id):
                    placements[expert_id] = ExpertPlacement(
                        expert_id=expert_id,
                        tier='ram',
                        device='cpu',
                    )
                    continue

                # Default: disk
                placements[expert_id] = ExpertPlacement(
                    expert_id=expert_id,
                    tier='disk',
                    location=str(self.disk_path / f'expert_{expert_id}.bin'),
                )

            return placements

    def _should_load_to_ram(self, expert_id: str, layer_id: int) -> bool:
        """Decide if expert should be pinned in RAM based on fire frequency."""
        if layer_id not in self.learning_cache:
            return False

        layer_counts = self.learning_cache[layer_id]
        fire_count = layer_counts.get(expert_id, 0)

        return fire_count > self.heat_threshold

    def record_placement_hit(self, expert_id: str, tier: str):
        """Record successful load from tier (for future optimization)."""
        if tier not in {"vram", "ram", "disk"}:
            raise ValueError("placement tier must be vram, ram, or disk")
        with self._lock:
            counts = getattr(self, "placement_hits", None)
            if counts is None:
                counts = self.placement_hits = {"vram": 0, "ram": 0, "disk": 0}
            counts[tier] += 1

    def pin_to_ram(self, expert_id: str, data: bytes):
        """Manually pin expert to RAM (called during prefetch)."""
        with self._lock:
            if len(self.ram_hot_store) >= self.max_ram_experts:
                # Evict LRU
                self.ram_hot_store.popitem(last=False)
            self.ram_hot_store[expert_id] = data

    def get_from_ram(self, expert_id: str) -> Optional[bytes]:
        """Retrieve expert from RAM if present."""
        with self._lock:
            if expert_id in self.ram_hot_store:
                # Move to end (LRU)
                self.ram_hot_store.move_to_end(expert_id)
                return self.ram_hot_store[expert_id]
            return None

    def save_learning_cache(self):
        """Persist routing history for next session."""
        if self.learning_cache_path is None:
            raise RuntimeError("tiered store has no explicit learning-cache path")
        with self._lock:
            try:
                self.learning_cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.learning_cache_path, 'w') as f:
                    json.dump(self.learning_cache, f, indent=2)
                logger.info(f"Saved learning cache to {self.learning_cache_path}")
            except Exception as e:
                logger.warning(f"Could not save learning cache: {e}")


# ============================================================================
# A12.2: UnifiedInferenceExecution — Single inference path for all models
# ============================================================================

@dataclass
class RoutingDecision:
    """Routing decision for one forward pass."""
    layer_id: int
    expert_ids: List[str]  # Which experts fire
    positions: List[int]  # Which batch positions use which expert


class UnifiedInferenceExecution:
    """
    Single unified inference path for all 9 models.

    Per-token inference flow:
    1. Route: Router decides which experts fire (per layer, per position)
    2. Batch-Union: Deduplicate expert loads across batch
    3. Place: Decide where each expert lives (VRAM/RAM/disk)
    4. Overlap: Async I/O while computing previous layer
    5. Learn: Track routing history for next session

    Replaces model-specific native_qwen35.forward(), native_glm5.forward(), etc.
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        expert_store: TieredExpertStore,
        num_layers: int = 24,
        num_experts_per_layer: int = 128,
    ):
        self.model_config = model_config
        self.expert_store = expert_store
        self.num_layers = num_layers
        self.num_experts_per_layer = num_experts_per_layer

        # Router (family-specific, but unified call interface)
        self.router = None  # Set by subclass or passed in

        # Learning cache for this session
        self.routing_history = {}  # {layer: {expert: count}}
        self.batch_count = 0

        # Prefetch thread
        self.prefetch_thread = None
        self.prefetch_queue = asyncio.Queue()

    def forward(
        self,
        input_tokens: List[int],
        batch_size: int = 1,
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        Unified forward pass (replaces model-specific forward()).

        Args:
            input_tokens: Input token IDs
            batch_size: Batch size for routing

        Returns:
            (logits, metadata)
        """
        raise RuntimeError(
            "UnifiedInferenceExecution is a retired routing/placement scaffold and "
            "does not execute model tensors; use AdaptiveExecutionEngine.open_universal_model()")

    def _deduplicate_routing(self, routing_decisions: List[RoutingDecision]) -> Dict[int, List[str]]:
        """
        Batch-union optimization: deduplicate expert loads.

        Input: routing decisions for each layer
        Output: unique experts per layer (load each ONCE, not N times for batch)
        """
        unique_per_layer = {}
        for decision in routing_decisions:
            layer_id = decision.layer_id
            if layer_id not in unique_per_layer:
                unique_per_layer[layer_id] = []

            for expert_id in decision.expert_ids:
                if expert_id not in unique_per_layer[layer_id]:
                    unique_per_layer[layer_id].append(expert_id)

        return unique_per_layer

    def _compute_forward(
        self,
        placements: Dict[int, Dict[str, ExpertPlacement]],
        routing_decisions: List[RoutingDecision],
        input_tokens: List[int],
    ) -> List[float]:
        """Reject the retired zero-logit implementation at every entry point."""
        raise RuntimeError(
            "no model tensor executor is bound to UnifiedInferenceExecution; "
            "use the exact family adapter registered with UniversalEngine")

    def record_routing(self, routing_decisions: List[RoutingDecision]):
        """Record routing decisions for learning cache."""
        for decision in routing_decisions:
            layer_id = decision.layer_id
            if layer_id not in self.routing_history:
                self.routing_history[layer_id] = {}
            for expert_id in decision.expert_ids:
                self.routing_history[layer_id][expert_id] = \
                    self.routing_history[layer_id].get(expert_id, 0) + 1

    def save_routing_history(self):
        """Save routing history to learning cache file."""
        # Merge with expert_store's cache
        self.expert_store.learning_cache.update(self.routing_history)
        self.expert_store.save_learning_cache()
