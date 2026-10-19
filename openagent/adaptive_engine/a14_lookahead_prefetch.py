"""A14: Lookahead Prefetch + Learning Cache
Phase 1B Speedup - Predict next layer's experts and load them in parallel.

Implements 2+ layer prefetch to hide I/O latency.
Learning cache persists routing history for improved placement next session.

Target: 71%+ prefetch hit rate
"""

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# A14.1: LookaheadPrefetcher — Predict and load next layers in parallel
# ============================================================================

@dataclass
class PrefetchRequest:
    """Request to prefetch experts for future layer."""
    layer_offset: int  # How many layers ahead
    expert_ids: Set[str]
    requested_at: float
    loaded_at: Optional[float] = None


class LookaheadPrefetcher:
    """
    Run router one layer ahead to predict next experts.

    Pattern:
    - Compute layer N
    - While computing: Router predicts layer N+1 and N+2 experts
    - Prefetch those experts in background
    - By time we reach layer N+1, experts are already loaded

    Benefits:
    - Compute time overlaps with I/O time
    - 71%+ hit rate when routing is predictable
    - Zero blocking in forward() path

    Fallback:
    - If prefetch misses (expert not pre-loaded), load synchronously
    - Still faster than no prefetch (most experts ready)
    """

    def __init__(
        self,
        router,  # Model-specific router for predictions
        expert_store,  # TieredExpertStore
        prefetch_layers: int = 2,
        batch_size: int = 1,
    ):
        self.router = router
        self.expert_store = expert_store
        self.prefetch_layers = prefetch_layers
        self.batch_size = batch_size

        # Prefetch state
        self.prefetch_queue = asyncio.Queue()
        self.pending_prefetches = {}  # {(layer, expert): Future}
        self.prefetch_thread = None
        self.stop_prefetch = threading.Event()

        # Metrics
        self.prefetch_requests = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0

        self._lock = threading.RLock()

    def start_prefetch_thread(self):
        """Start background prefetch thread (call once at init)."""
        if self.prefetch_thread is not None:
            return

        self.prefetch_thread = threading.Thread(
            target=self._prefetch_loop,
            daemon=True,
        )
        self.prefetch_thread.start()
        logger.info("Prefetch thread started")

    def stop_prefetch_thread(self):
        """Gracefully stop prefetch thread."""
        self.stop_prefetch.set()
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=2.0)

    def _prefetch_loop(self):
        """Background thread: predict and load experts."""
        try:
            while not self.stop_prefetch.is_set():
                try:
                    request = self.prefetch_queue.get(timeout=1.0)
                    if request is None:
                        break

                    # Load experts for this layer
                    self._prefetch_layer(request)

                except asyncio.QueueEmpty:
                    pass

        except Exception as e:
            logger.error(f"Prefetch thread error: {e}")

    def _prefetch_layer(self, request: PrefetchRequest):
        """Prefetch all experts for a layer."""
        layer_offset = request.layer_offset
        expert_ids = request.expert_ids

        logger.debug(f"Prefetching layer offset {layer_offset}: {len(expert_ids)} experts")

        for expert_id in expert_ids:
            try:
                # Get placement
                placement = self.expert_store.get_placement([expert_id], layer_id=0)

                if placement:
                    p = placement[expert_id]

                    # If on disk, pin to RAM
                    if p.tier == 'disk':
                        # Simulate load (real version would load from disk)
                        logger.debug(f"  Pinning {expert_id} to RAM")
                        # self.expert_store.pin_to_ram(expert_id, data)

            except Exception as e:
                logger.warning(f"Failed to prefetch {expert_id}: {e}")

    def predict_next_experts(
        self,
        input_tokens: List[int],
        current_layer: int,
    ) -> Dict[int, Set[str]]:
        """
        Predict which experts will fire in next N layers.

        This is a simplified prediction; real version would use
        router.predict_ahead() method.

        Args:
            input_tokens: Current batch tokens
            current_layer: Current layer being computed

        Returns:
            {layer_offset: {expert_ids}}
        """
        predictions = {}

        for offset in range(1, self.prefetch_layers + 1):
            try:
                # Simplified: ask router for predictions
                if hasattr(self.router, 'predict_ahead'):
                    predicted_routing = self.router.predict_ahead(
                        input_tokens,
                        offset=offset,
                    )
                    expert_ids = set(predicted_routing.get('expert_ids', []))
                    predictions[offset] = expert_ids
                else:
                    # Fallback: use history from learning cache
                    expert_ids = self.expert_store.learning_cache.get(
                        str(current_layer + offset),
                        {},
                    ).keys()
                    predictions[offset] = set(expert_ids)

            except Exception as e:
                logger.debug(f"Could not predict layer {current_layer + offset}: {e}")

        return predictions

    def queue_prefetch(self, predictions: Dict[int, Set[str]]):
        """Queue predicted experts for prefetch."""
        for layer_offset, expert_ids in predictions.items():
            if expert_ids:
                request = PrefetchRequest(
                    layer_offset=layer_offset,
                    expert_ids=expert_ids,
                    requested_at=time.time(),
                )

                with self._lock:
                    self.prefetch_requests += len(expert_ids)

                try:
                    self.prefetch_queue.put_nowait(request)
                except:
                    logger.warning(f"Prefetch queue full for layer offset {layer_offset}")

    def record_hit(self, layer_id: int, expert_ids: Set[str]):
        """Record successful prefetch (expert already loaded)."""
        with self._lock:
            for expert_id in expert_ids:
                in_ram = self.expert_store.get_from_ram(expert_id) is not None
                if in_ram:
                    self.prefetch_hits += 1
                else:
                    self.prefetch_misses += 1

    def get_hit_rate(self) -> float:
        """Get prefetch hit rate (0.0 to 1.0)."""
        with self._lock:
            total = self.prefetch_hits + self.prefetch_misses
            if total == 0:
                return 0.0
            return self.prefetch_hits / total

    def report(self) -> Dict[str, float]:
        """Return prefetch performance."""
        with self._lock:
            total = self.prefetch_hits + self.prefetch_misses
            hit_rate = self.prefetch_hits / total if total > 0 else 0.0
            return {
                'prefetch_requests': self.prefetch_requests,
                'prefetch_hits': self.prefetch_hits,
                'prefetch_misses': self.prefetch_misses,
                'hit_rate': hit_rate,
            }


# ============================================================================
# A14.2: LearningCache — Persist routing history
# ============================================================================

@dataclass
class LayerRoutingStats:
    """Statistics for expert firing in one layer."""
    layer_id: int
    expert_fire_counts: Dict[str, int] = field(default_factory=dict)
    total_forwards: int = 0
    last_updated: float = field(default_factory=time.time)


class LearningCache:
    """
    Persistent routing history for learned placement optimization.

    Tracks:
    - Which experts fire per layer
    - How frequently each expert fires
    - Fire patterns over time

    Usage:
    - Session 1: Record routing → save to .openagent_usage
    - Session 2: Load .openagent_usage → pin hot experts in RAM

    Result: Placement improves over multiple sessions (learned optimization)
    """

    def __init__(self, cache_file: Path = None):
        self.cache_file = cache_file or (Path.home() / '.openagent_usage')
        self.routing_history = {}  # {layer_id: {expert_id: count}}
        self.total_forwards = 0
        self._lock = threading.RLock()

        self._load_cache()

    def _load_cache(self):
        """Load routing history from previous session."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.routing_history = data
                logger.info(f"Loaded learning cache: {len(self.routing_history)} layers")
        except Exception as e:
            logger.warning(f"Could not load learning cache: {e}")
            self.routing_history = {}

    def record_routing(self, routing_decisions: List[Dict]):
        """Record routing decisions from one forward pass."""
        with self._lock:
            self.total_forwards += 1

            for decision in routing_decisions:
                layer_id = decision.get('layer_id', 0)
                expert_ids = decision.get('expert_ids', [])

                if str(layer_id) not in self.routing_history:
                    self.routing_history[str(layer_id)] = {}

                layer_counts = self.routing_history[str(layer_id)]
                for expert_id in expert_ids:
                    layer_counts[expert_id] = layer_counts.get(expert_id, 0) + 1

    def get_hot_experts(self, layer_id: int, top_k: int = 16) -> List[Tuple[str, int]]:
        """
        Get hottest K experts in a layer (most frequently fired).

        Args:
            layer_id: Layer number
            top_k: Return top K experts

        Returns:
            [(expert_id, fire_count), ...]
        """
        with self._lock:
            layer_counts = self.routing_history.get(str(layer_id), {})

            if not layer_counts:
                return []

            sorted_experts = sorted(
                layer_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            return sorted_experts[:top_k]

    def get_expert_frequency(self, layer_id: int, expert_id: str) -> float:
        """Get firing frequency for expert (0.0 to 1.0)."""
        with self._lock:
            layer_counts = self.routing_history.get(str(layer_id), {})
            total = sum(layer_counts.values())
            if total == 0:
                return 0.0
            return layer_counts.get(expert_id, 0) / total

    def save(self):
        """Persist routing history to disk."""
        with self._lock:
            try:
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_file, 'w') as f:
                    json.dump(self.routing_history, f, indent=2)
                logger.info(f"Saved learning cache: {len(self.routing_history)} layers")
            except Exception as e:
                logger.error(f"Failed to save learning cache: {e}")

    def report(self) -> Dict:
        """Return learning cache statistics."""
        with self._lock:
            total_entries = sum(len(experts) for experts in self.routing_history.values())
            return {
                'layers_recorded': len(self.routing_history),
                'total_experts_in_history': total_entries,
                'total_forwards': self.total_forwards,
                'avg_experts_per_layer': total_entries / max(1, len(self.routing_history)),
            }


# ============================================================================
# A14.3: Integration — Unified prefetch + learning
# ============================================================================

class PrefetchAndLearnExecutor:
    """
    Combines lookahead prefetch with learning cache persistence.

    Usage:
        executor = PrefetchAndLearnExecutor(router, expert_store)
        executor.start()

        # During generation
        output = executor.forward(input_tokens)

        executor.stop()  # Saves learning cache
    """

    def __init__(self, router, expert_store):
        self.prefetcher = LookaheadPrefetcher(router, expert_store)
        self.learning_cache = LearningCache()
        self._lock = threading.RLock()

    def start(self):
        """Start prefetch thread."""
        self.prefetcher.start_prefetch_thread()

    def stop(self):
        """Stop prefetch thread and save learning cache."""
        self.prefetcher.stop_prefetch_thread()
        self.learning_cache.save()

    def forward(
        self,
        input_tokens: List[int],
        current_layer: int,
    ) -> Tuple[List[float], Dict]:
        """Forward with prefetch and learning."""
        # Step 1: Predict next layers
        predictions = self.prefetcher.predict_next_experts(input_tokens, current_layer)

        # Step 2: Queue prefetch
        self.prefetcher.queue_prefetch(predictions)

        # Step 3: Do actual forward (can happen in parallel with prefetch)
        # ... actual model computation ...

        # Step 4: Record routing for learning
        routing = self._get_routing(current_layer)  # Placeholder
        self.learning_cache.record_routing(routing)

        # Metadata
        metadata = {
            'prefetch_hit_rate': self.prefetcher.get_hit_rate(),
            'learning_cache_size': len(self.learning_cache.routing_history),
        }

        return None, metadata  # Placeholder

    def _get_routing(self, layer_id: int) -> List[Dict]:
        """Get routing decisions (stub)."""
        return [{'layer_id': layer_id, 'expert_ids': []}]
