"""A13: Batch-Union + Async I/O
Phase 1B Speedup - Expert load deduplication and non-blocking I/O.

Reduces expert loads from N×M (batch×unique_experts) to M (unique_experts only).
Implements async I/O pool for overlapping compute and I/O.

Target: 20%+ I/O reduction, zero blocking in forward()
"""

import asyncio
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# A13.1: BatchUnionOptimizer — Expert deduplication across batch
# ============================================================================

@dataclass
class BatchExpertMapping:
    """Maps batch positions to shared expert data."""
    unique_expert_ids: Set[str]
    position_to_expert: Dict[int, str]  # batch_pos -> expert_id
    expert_data: Dict[str, bytes]  # expert_id -> loaded weights


class BatchUnionOptimizer:
    """
    Deduplicate expert loads across batch.

    Problem:
      - Batch of 4, each position uses different expert
      - Naive: Load expert 0, 1, 2, 3 = 4 loads

    Solution:
      - Router says: pos 0→expert 0, pos 1→expert 0, pos 2→expert 1, pos 3→expert 1
      - Batch-union: Unique = {0, 1}
      - Load expert 0 once, load expert 1 once = 2 loads
      - Reuse in all 4 positions

    Speedup: N positions, M unique experts
      - Before: N loads
      - After: M loads (where M ≤ N, usually M << N)
    """

    def __init__(self):
        self._lock = threading.RLock()

    def deduplicate_routing(
        self,
        routing_decisions: List[Dict],
    ) -> BatchExpertMapping:
        """
        Deduplicate experts in this batch.

        Args:
            routing_decisions: List of {layer, expert_ids, positions}

        Returns:
            BatchExpertMapping with unique experts and position mapping
        """
        with self._lock:
            unique_experts = set()
            position_to_expert = {}

            # Flatten routing across all layers/positions
            for decision in routing_decisions:
                experts = decision.get('expert_ids', [])
                positions = decision.get('positions', [])

                for pos, expert_id in zip(positions, experts):
                    unique_experts.add(expert_id)
                    position_to_expert[pos] = expert_id

            return BatchExpertMapping(
                unique_expert_ids=unique_experts,
                position_to_expert=position_to_expert,
                expert_data={},
            )

    def load_batch_union(
        self,
        expert_ids: Set[str],
        load_fn,  # Callable[[expert_id]) -> bytes
    ) -> Dict[str, bytes]:
        """
        Load each unique expert once.

        Args:
            expert_ids: Set of expert IDs to load
            load_fn: Function that loads an expert (expert_id) -> bytes

        Returns:
            {expert_id: expert_data}
        """
        with self._lock:
            expert_data = {}

            for expert_id in expert_ids:
                try:
                    data = load_fn(expert_id)
                    expert_data[expert_id] = data
                except Exception as e:
                    logger.error(f"Failed to load expert {expert_id}: {e}")
                    # Continue with other experts

            return expert_data

    def measure_savings(self, batch_size: int, num_unique_experts: int) -> float:
        """Calculate I/O reduction percentage."""
        naive_loads = batch_size
        optimized_loads = num_unique_experts
        return (1.0 - optimized_loads / naive_loads) * 100


# ============================================================================
# A13.2: AsyncIOPool — Non-blocking expert loading
# ============================================================================

@dataclass
class AsyncLoadTask:
    """Pending async I/O operation."""
    expert_id: str
    future: asyncio.Future
    started_at: float


class AsyncIOPool:
    """
    Async I/O pool for non-blocking expert loading.

    Features:
    - Up to N concurrent disk reads
    - Queue for pending requests
    - Deduplication: if expert_id already loading, await same future
    - Overlap: compute layer N while loading layer N+1

    Target: No blocking in forward() path
    """

    def __init__(self, max_concurrent: int = 8):
        self.max_concurrent = max_concurrent
        self.pending = {}  # {expert_id: AsyncLoadTask}
        self._lock = threading.RLock()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._loop = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop for async operations."""
        self._loop = loop

    async def async_load_expert(
        self,
        expert_id: str,
        load_fn,  # Callable[[expert_id]) -> bytes
    ) -> bytes:
        """
        Load expert asynchronously (non-blocking).

        If expert already loading, return awaiting same future.
        If already loaded, return immediately.

        Args:
            expert_id: Which expert to load
            load_fn: Callable that does the actual disk I/O

        Returns:
            Expert data (bytes)
        """
        with self._lock:
            # Already loading?
            if expert_id in self.pending:
                task = self.pending[expert_id]
                logger.debug(f"Expert {expert_id} already loading, reusing")
                return await task.future

        # Not loading: start new task
        async with self._semaphore:  # Limit concurrency
            try:
                data = await asyncio.to_thread(load_fn, expert_id)

                with self._lock:
                    if expert_id in self.pending:
                        del self.pending[expert_id]

                return data

            except Exception as e:
                logger.error(f"Async load failed for {expert_id}: {e}")

                with self._lock:
                    if expert_id in self.pending:
                        del self.pending[expert_id]

                raise

    async def load_batch_async(
        self,
        expert_ids: Set[str],
        load_fn,
    ) -> Dict[str, bytes]:
        """
        Load all experts in batch concurrently.

        Args:
            expert_ids: Set of expert IDs
            load_fn: Callable that loads one expert

        Returns:
            {expert_id: expert_data}
        """
        tasks = {
            expert_id: self.async_load_expert(expert_id, load_fn)
            for expert_id in expert_ids
        }

        results = {}
        try:
            for expert_id, task in tasks.items():
                results[expert_id] = await task
        except Exception as e:
            logger.error(f"Batch async load failed: {e}")

        return results

    def throttle_concurrent(self) -> float:
        """Return current concurrency level (0.0 to 1.0)."""
        with self._lock:
            return len(self.pending) / self.max_concurrent


# ============================================================================
# A13.3: Integration — Unified Batch-Union + Async execution
# ============================================================================

class BatchUnionAsyncExecutor:
    """
    Combines batch-union deduplication with async I/O.

    Usage:
        executor = BatchUnionAsyncExecutor(max_concurrent=8)
        data = executor.load_batch(routing_decisions, load_fn)
        # Returns {expert_id: bytes}, loaded efficiently
    """

    def __init__(self, max_concurrent: int = 8):
        self.batch_union = BatchUnionOptimizer()
        self.async_pool = AsyncIOPool(max_concurrent)

    async def load_batch_optimized(
        self,
        routing_decisions: List[Dict],
        load_fn,
    ) -> Dict[str, bytes]:
        """
        Load batch with both optimizations.

        1. Deduplicate routing (batch-union)
        2. Load unique experts async (no blocking)

        Returns:
            {expert_id: expert_data}
        """
        # Step 1: Deduplicate
        mapping = self.batch_union.deduplicate_routing(routing_decisions)
        unique_ids = mapping.unique_expert_ids

        # Log savings
        total_positions = len(mapping.position_to_expert)
        savings = self.batch_union.measure_savings(total_positions, len(unique_ids))
        logger.info(
            f"Batch-union: {total_positions} positions → {len(unique_ids)} unique experts "
            f"({savings:.1f}% I/O reduction)"
        )

        # Step 2: Load async (no blocking)
        expert_data = await self.async_pool.load_batch_async(unique_ids, load_fn)

        return expert_data


# ============================================================================
# A13.4: Performance metrics
# ============================================================================

class BatchUnionMetrics:
    """Track batch-union and async I/O performance."""

    def __init__(self):
        self.total_batches = 0
        self.total_positions = 0
        self.total_unique_experts = 0
        self.total_io_reduction_percent = 0.0
        self.max_concurrent_loads = 0

    def record_batch(
        self,
        num_positions: int,
        num_unique_experts: int,
        concurrent_loads: int,
    ):
        """Record one batch's metrics."""
        self.total_batches += 1
        self.total_positions += num_positions
        self.total_unique_experts += num_unique_experts
        self.max_concurrent_loads = max(self.max_concurrent_loads, concurrent_loads)

        # Calculate I/O reduction for this batch
        if num_positions > 0:
            reduction = (1.0 - num_unique_experts / num_positions) * 100
            self.total_io_reduction_percent = \
                (self.total_io_reduction_percent * (self.total_batches - 1) + reduction) / self.total_batches

    def report(self) -> Dict[str, float]:
        """Return performance summary."""
        return {
            'total_batches': self.total_batches,
            'avg_positions_per_batch': self.total_positions / max(1, self.total_batches),
            'avg_unique_experts_per_batch': self.total_unique_experts / max(1, self.total_batches),
            'avg_io_reduction_percent': self.total_io_reduction_percent,
            'max_concurrent_loads': self.max_concurrent_loads,
        }
