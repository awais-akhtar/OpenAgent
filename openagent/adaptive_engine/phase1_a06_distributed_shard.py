"""A06 — Distributed shard layout, consensus, and failure recovery.

This module provides a standalone three-phase distributed protocol model for
tensor-parallel expert-bank coordination:

  Phase 1 — Pre-collective agreement
      Each rank independently validates the proposed operation (open, generate,
      save_cache, load_cache, snapshot, restore).  Rank 0 collects all
      pre-flight summaries via an all-gather and rejects the operation if any
      rank reports an error or geometry mismatch.

  Phase 2 — Collective operation
      All ranks execute the agreed operation simultaneously, exchanging only
      the bounded result summary (token ids, history digest, position count).

  Phase 3 — Post-collective agreement
      Results are gathered and compared; any rank whose summary differs from
      the coordinator's poisons the group.

Failure recovery (rank loss):
  When a rank exits unexpectedly, the coordinator detects the loss during the
  next collective timeout.  It records the last committed cache manifest for
  each rank.  A caller may launch replacement ranks that load the committed
  cache and rejoin the group for subsequent requests; the failure is final for
  the in-flight request but recovery is possible for later work.

Shard layout:
  Expert banks are partitioned into contiguous ranges by rank index.  The
  layout is determined by (family, world_size, rank) and remains fixed for
  the lifetime of the process group.  Changing the layout requires restarting
  all ranks.

This helper is not the production process-group owner. Production distributed
execution is implemented by ``distributed_execution.DistributedExecutionService``
and the family-specific distributed session adapters. Callers must not treat
``DistributedCoordinator`` receipts or shard ranges as proof that those runtime
paths use this helper. All public methods are thread-safe; no background
threads are started here.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple



# ---------------------------------------------------------------------------
# Shard layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExpertShardRange:
    """Contiguous range of expert indices owned by one rank."""
    rank: int
    world_size: int
    expert_start: int
    expert_end: int   # exclusive

    @property
    def count(self) -> int:
        return self.expert_end - self.expert_start

    def contains(self, expert_id: int) -> bool:
        return self.expert_start <= expert_id < self.expert_end

    def validate(self, total_experts: int) -> None:
        if (type(total_experts) is not int or total_experts <= 0 or
                type(self.rank) is not int or type(self.world_size) is not int or
                not (0 <= self.rank < self.world_size)):
            raise ValueError(f'rank {self.rank} out of range for world_size {self.world_size}')
        if (type(self.expert_start) is not int or type(self.expert_end) is not int or
                not (0 <= self.expert_start < self.expert_end <= total_experts)):
            raise ValueError(
                f'shard range [{self.expert_start}, {self.expert_end}) out of '
                f'bounds for {total_experts} experts')


# Per-family expert counts for layout calculation.
# These are the default routed-expert pool sizes from the source catalog;
# they do not include shared/global experts.
_FAMILY_ROUTED_EXPERTS: Dict[str, int] = {
    'mixtral': 8,
    'olmoe': 64,
    'qwen35': 64,
    'qwen4': 64,
    'glm5-next': 32,
    'inkling': 64,
    'glm-dsa': 32,
    'kimi-k3': 64,
    'deepseek-v41': 256,
}


def compute_shard_layout(
    family: str,
    world_size: int,
    *,
    total_experts: Optional[int] = None,
) -> List[ExpertShardRange]:
    """Return the canonical shard ranges for all ranks.

    The layout divides routed experts into contiguous blocks.  When the
    expert count is not evenly divisible, the remainder is distributed one
    per rank from rank 0 upward.
    """
    if type(family) is not str or family not in _FAMILY_ROUTED_EXPERTS:
        raise ValueError('exact admitted family identity required for shard layout')
    if type(world_size) is not int or not (1 <= world_size <= 64):
        raise ValueError('1–64 ranks required for shard layout')
    if total_experts is None:
        total_experts = _FAMILY_ROUTED_EXPERTS.get(family)
        if total_experts is None:
            raise ValueError(
                f'family {family!r} has no default expert count; '
                f'pass total_experts explicitly')
    if type(total_experts) is not int or total_experts <= 0:
        raise ValueError('positive expert count required')
    if world_size > total_experts:
        raise ValueError(
            f'world_size ({world_size}) exceeds total_experts ({total_experts}); '
            f'every rank must own at least one expert')

    base, remainder = divmod(total_experts, world_size)
    ranges: List[ExpertShardRange] = []
    start = 0
    for rank in range(world_size):
        extra = 1 if rank < remainder else 0
        end = start + base + extra
        shard = ExpertShardRange(
            rank=rank, world_size=world_size,
            expert_start=start, expert_end=end,
        )
        shard.validate(total_experts)
        ranges.append(shard)
        start = end
    assert start == total_experts
    return ranges


def rank_for_expert(expert_id: int, layout: List[ExpertShardRange]) -> int:
    """Return the rank that owns expert_id in the given layout."""
    if type(expert_id) is not int or expert_id < 0:
        raise ValueError('nonnegative integer expert ID required')
    for shard in layout:
        if shard.contains(expert_id):
            return shard.rank
    raise ValueError(f'expert_id {expert_id} not covered by any shard in layout')


# ---------------------------------------------------------------------------
# Consensus protocol state
# ---------------------------------------------------------------------------

class ConsensusError(RuntimeError):
    """Raised when all-rank agreement cannot be reached."""


@dataclass
class RankSummary:
    """Bounded result summary exchanged during the collective phase."""
    rank: int
    action: str
    tokens: Tuple[int, ...] = field(default_factory=tuple)
    position: int = 0
    history_digest: str = ''
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank < 0:
            raise ValueError('nonnegative integer rank required')
        if type(self.action) is not str or not self.action or len(self.action) > 64:
            raise ValueError('bounded non-empty distributed action required')
        if type(self.tokens) is not tuple or any(type(token) is not int or token < 0 for token in self.tokens):
            raise ValueError('distributed result tokens must be a tuple of nonnegative integers')
        if type(self.position) is not int or self.position < 0:
            raise ValueError('nonnegative distributed cache position required')
        if type(self.history_digest) is not str:
            raise ValueError('distributed history digest must be text')
        if self.history_digest and (len(self.history_digest) != 64 or any(
                char not in '0123456789abcdef' for char in self.history_digest)):
            raise ValueError('distributed history digest must be a SHA-256 hex digest')
        if self.error is not None and (type(self.error) is not str or not self.error or len(self.error) > 512):
            raise ValueError('distributed error summary must be bounded non-empty text')
        if self.action in {'generate', 'save_cache', 'load_cache', 'snapshot', 'restore'} and self.error is None and not self.history_digest:
            raise ValueError('stateful distributed result requires a committed-history digest')

    def is_error(self) -> bool:
        return self.error is not None

    def matches(self, other: 'RankSummary') -> bool:
        """Check that this summary agrees with another rank's summary."""
        if self.action != other.action:
            return False
        if self.tokens != other.tokens:
            return False
        if self.position != other.position:
            return False
        return self.history_digest == other.history_digest


class DistributedConsensus:
    """Three-phase consensus tracker for a single distributed operation.

    Usage::

        consensus = DistributedConsensus(action='generate', world_size=4)
        # Each rank calls:
        consensus.submit_preflight(
            rank=my_rank, ok=True, contract_digest=exact_operation_digest)
        # Rank 0 waits for all preflight submissions, then:
        consensus.require_preflight_agreement()   # raises ConsensusError on mismatch
        # Each rank runs its collective and calls:
        consensus.submit_result(RankSummary(rank=my_rank, ...))
        # Rank 0 waits for all results, then:
        consensus.require_result_agreement()      # raises ConsensusError on mismatch
    """

    def __init__(self, action: str, world_size: int):
        if type(world_size) is not int or not (1 <= world_size <= 64):
            raise ValueError('1–64 ranks required for consensus')
        if type(action) is not str or not action or len(action) > 64:
            raise ValueError('bounded non-empty action name required')
        self._action = action
        self._world_size = world_size
        self._lock = threading.Lock()
        self._preflight: Dict[int, bool] = {}
        self._preflight_errors: Dict[int, str] = {}
        self._preflight_contracts: Dict[int, str] = {}
        self._results: Dict[int, RankSummary] = {}

    def submit_preflight(
        self, rank: int, ok: bool, error: str = '', *, contract_digest: str,
    ) -> None:
        """Record one rank's pre-collective validation result."""
        if type(rank) is not int or not (0 <= rank < self._world_size):
            raise ValueError(f'rank must be in [0, {self._world_size})')
        if type(ok) is not bool:
            raise ValueError('preflight result must be a boolean')
        if type(error) is not str or len(error) > 512:
            raise ValueError('preflight error summary must be bounded text')
        if (type(contract_digest) is not str or len(contract_digest) != 64 or
                any(char not in '0123456789abcdef' for char in contract_digest.lower())):
            raise ValueError('preflight requires the exact operation SHA-256 contract digest')
        with self._lock:
            if rank in self._preflight:
                raise RuntimeError(f'rank {rank} already submitted preflight for {self._action!r}')
            self._preflight[rank] = ok
            self._preflight_contracts[rank] = contract_digest.lower()
            if not ok and error:
                self._preflight_errors[rank] = error[:512]

    def require_preflight_agreement(self) -> None:
        """Raise ConsensusError if any rank rejected the preflight."""
        with self._lock:
            missing = set(range(self._world_size)) - set(self._preflight.keys())
            if missing:
                raise ConsensusError(
                    f'{self._action}: preflight incomplete; '
                    f'ranks {sorted(missing)} have not submitted')
            failing = [r for r, ok in self._preflight.items() if not ok]
            if failing:
                details = '; '.join(
                    f'rank {r}: {self._preflight_errors.get(r, "rejected")}'
                    for r in sorted(failing))
                raise ConsensusError(
                    f'{self._action}: preflight rejected by ranks {sorted(failing)}: {details}')
            digests = set(self._preflight_contracts.values())
            if len(digests) != 1:
                raise ConsensusError(
                    f'{self._action}: preflight operation contract differs across ranks')

    def can_abort_before_collective(self) -> bool:
        """Return whether protocol state proves that no rank was admitted.

        The protocol forbids entering the collective until every rank has
        submitted an accepting preflight for the same operation contract.  An
        incomplete, rejected, or mismatched preflight can therefore be
        discarded safely.  Once unanimous preflight is present, the
        coordinator cannot know whether peers have already entered the
        collective, even if no result has arrived yet, so abort must poison
        the group instead of making it reusable.
        """
        with self._lock:
            if self._results:
                return False
            if set(self._preflight) != set(range(self._world_size)):
                return True
            if not all(self._preflight.values()):
                return True
            return len(set(self._preflight_contracts.values())) != 1

    def submit_result(self, summary: RankSummary) -> None:
        """Record one rank's post-collective result summary."""
        if type(summary) is not RankSummary:
            raise ValueError('exact bounded RankSummary required')
        if summary.action != self._action:
            raise ValueError(
                f'result action {summary.action!r} != expected {self._action!r}')
        if type(summary.rank) is not int or not (0 <= summary.rank < self._world_size):
            raise ValueError(f'result rank {summary.rank} out of range')
        with self._lock:
            missing_preflight = set(range(self._world_size)) - set(self._preflight)
            if missing_preflight or any(not accepted for accepted in self._preflight.values()):
                raise ConsensusError(
                    f'{self._action}: result submitted before unanimous preflight agreement')
            if len(set(self._preflight_contracts.values())) != 1:
                raise ConsensusError(
                    f'{self._action}: result submitted with mismatched operation contracts')
            if summary.rank in self._results:
                raise RuntimeError(
                    f'rank {summary.rank} already submitted result for {self._action!r}')
            self._results[summary.rank] = summary

    def require_result_agreement(self) -> RankSummary:
        """Raise ConsensusError if results diverge; return rank-0 summary."""
        with self._lock:
            missing = set(range(self._world_size)) - set(self._results.keys())
            if missing:
                raise ConsensusError(
                    f'{self._action}: results incomplete; '
                    f'ranks {sorted(missing)} have not submitted')
            # Check for backend errors first
            error_ranks = [r for r, s in self._results.items() if s.is_error()]
            if error_ranks:
                details = '; '.join(
                    f'rank {r}: {self._results[r].error}'
                    for r in sorted(error_ranks))
                raise ConsensusError(
                    f'{self._action}: backend error in ranks {sorted(error_ranks)}: {details}')
            # Check result agreement against rank 0
            ref = self._results[0]
            diverged = [r for r, s in self._results.items() if r != 0 and not s.matches(ref)]
            if diverged:
                raise ConsensusError(
                    f'{self._action}: result divergence between rank 0 and '
                    f'ranks {sorted(diverged)}; group must be restarted')
            return ref


# ---------------------------------------------------------------------------
# Rank loss detection and recovery planning
# ---------------------------------------------------------------------------

@dataclass
class RankHealthRecord:
    rank: int
    world_size: int
    last_committed_position: int = 0
    last_committed_digest: str = ''
    last_committed_cache_path: Optional[str] = None
    failed: bool = False
    failure_reason: str = ''


class DistributedFailureRecovery:
    """Track per-rank health and plan recovery after rank loss.

    When rank loss is detected (e.g. a collective times out), the coordinator
    calls mark_failed().  A recovery caller inspects recovery_plan() to
    determine which cache paths to load before rejoining ranks.
    """

    def __init__(self, world_size: int):
        if type(world_size) is not int or not (1 <= world_size <= 64):
            raise ValueError('1–64 ranks required')
        self._world_size = world_size
        self._lock = threading.Lock()
        self._records: Dict[int, RankHealthRecord] = {
            r: RankHealthRecord(rank=r, world_size=world_size)
            for r in range(world_size)
        }

    def update_committed(
        self, rank: int, position: int,
        history_digest: str, cache_path: Optional[str] = None,
    ) -> None:
        """Record a successfully committed cache position for one rank."""
        if type(rank) is not int or not (0 <= rank < self._world_size):
            raise ValueError(f'rank must be in [0, {self._world_size})')
        if type(position) is not int or position < 0:
            raise ValueError('nonnegative committed position required')
        if (type(history_digest) is not str or len(history_digest) != 64 or
                any(character not in '0123456789abcdef'
                    for character in history_digest)):
            raise ValueError('committed history must use a lowercase SHA-256 digest')
        if cache_path is not None and (
                type(cache_path) is not str or not cache_path or '\x00' in cache_path or
                len(cache_path) > 4096):
            raise ValueError('bounded rank-local cache path required')
        with self._lock:
            rec = self._records[rank]
            if rec.failed:
                raise RuntimeError(
                    f'rank {rank} is already marked failed; '
                    f'cannot update committed state')
            if position < rec.last_committed_position:
                raise ValueError(
                    f'rank {rank}: new committed position ({position}) must be '
                    f'>= current ({rec.last_committed_position})')
            if (position == rec.last_committed_position and
                    rec.last_committed_digest and
                    history_digest != rec.last_committed_digest):
                raise ValueError(
                    f'rank {rank}: history digest changed without advancing '
                    f'the committed position ({position})')
            rec.last_committed_position = position
            rec.last_committed_digest = history_digest
            if cache_path is not None:
                rec.last_committed_cache_path = cache_path

    def mark_failed(self, rank: int, reason: str = 'rank loss detected') -> None:
        """Record that a rank has failed and must be recovered."""
        if type(rank) is not int or not (0 <= rank < self._world_size):
            raise ValueError(f'rank must be in [0, {self._world_size})')
        if type(reason) is not str or not reason:
            raise ValueError('bounded rank failure reason required')
        with self._lock:
            rec = self._records[rank]
            rec.failed = True
            rec.failure_reason = reason[:256]

    def failed_ranks(self) -> List[int]:
        with self._lock:
            return [r for r, rec in self._records.items() if rec.failed]

    def healthy_ranks(self) -> List[int]:
        with self._lock:
            return [r for r, rec in self._records.items() if not rec.failed]

    def recovery_plan(self) -> List[dict]:
        """Return per-rank recovery descriptors for failed ranks.

        Each entry describes what a replacement rank must load to rejoin the
        group at the last committed checkpoint.  The coordinator rank (0)
        selects the minimum committed position across all healthy ranks as
        the safe recovery point; no rank may resume past a failed rank.
        """
        with self._lock:
            if not any(rec.failed for rec in self._records.values()):
                return []
            healthy = [rec for rec in self._records.values() if not rec.failed]
            if not healthy:
                return [{'rank': r, 'recovery_position': 0,
                          'recovery_digest': '',
                          'cache_path': None,
                          'reason': 'all ranks failed; full restart required'}
                        for r in range(self._world_size)]
            safe_position = min(rec.last_committed_position for rec in healthy)
            safe_records = [rec for rec in healthy if rec.last_committed_position == safe_position]
            safe_digests = {rec.last_committed_digest for rec in safe_records}
            if len(safe_digests) != 1:
                digest = None
            else:
                digest = next(iter(safe_digests))
            plan: List[dict] = []
            for rec in self._records.values():
                if rec.failed:
                    same_boundary = (
                        digest is not None and
                        rec.last_committed_position == safe_position and
                        rec.last_committed_digest == digest and
                        rec.last_committed_cache_path is not None
                    )
                    plan.append({
                        'rank': rec.rank,
                        'recovery_position': safe_position,
                        'recovery_digest': digest or '',
                        # Cache state is rank-local. Never point a replacement
                        # rank at another rank's shard snapshot.
                        'cache_path': rec.last_committed_cache_path if same_boundary else None,
                        'recoverable': same_boundary,
                        'reason': (rec.failure_reason if same_boundary else
                                   'failed rank lacks its own cache receipt at the unanimous safe boundary; full group restart required'),
                    })
            return plan

    def reset_failed(self, rank: int) -> None:
        """Clear the failed flag after a successful replacement rank joins."""
        if type(rank) is not int or not (0 <= rank < self._world_size):
            raise ValueError(f'rank must be in [0, {self._world_size})')
        with self._lock:
            rec = self._records[rank]
            rec.failed = False
            rec.failure_reason = ''

    def all_healthy(self) -> bool:
        with self._lock:
            return all(not rec.failed for rec in self._records.values())


# ---------------------------------------------------------------------------
# Integrated distributed session coordinator (rank 0 side)
# ---------------------------------------------------------------------------

class DistributedCoordinator:
    """Standalone coordinator-side model for one distributed process group.

    This class is not wired into production process-group construction; the
    live service uses ``DistributedExecutionService``. This object is useful
    for isolated protocol reasoning and must not be used as runtime evidence.
    """

    def __init__(
        self,
        family: str,
        world_size: int,
        *,
        total_experts: Optional[int] = None,
    ):
        if type(family) is not str or family not in _FAMILY_ROUTED_EXPERTS:
            raise ValueError('exact admitted family identity required for distributed coordination')
        if type(world_size) is not int or not (1 <= world_size <= 64):
            raise ValueError('1–64 ranks required')
        self._family = family
        self._world_size = world_size
        self._layout = compute_shard_layout(family, world_size, total_experts=total_experts)
        self._lock = threading.Lock()
        self._health = DistributedFailureRecovery(world_size)
        self._active_consensus: Optional[DistributedConsensus] = None
        self._operation_sequence: int = 0
        self._poisoned: bool = False

    @property
    def layout(self) -> List[ExpertShardRange]:
        return list(self._layout)

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def poisoned(self) -> bool:
        with self._lock:
            return self._poisoned

    def begin_operation(self, action: str) -> DistributedConsensus:
        """Start a new consensus round; returns the consensus tracker."""
        with self._lock:
            if self._poisoned:
                raise RuntimeError(
                    f'distributed coordinator for {self._family!r} is poisoned; '
                    f'all ranks must be restarted')
            if self._active_consensus is not None:
                raise RuntimeError(
                    f'another operation ({self._active_consensus._action!r}) '
                    f'is still in progress; complete or abort it first')
            consensus = DistributedConsensus(action=action, world_size=self._world_size)
            self._active_consensus = consensus
            self._operation_sequence += 1
            return consensus

    def complete_operation(self) -> RankSummary:
        """Finalize the current operation; poison on consensus failure."""
        with self._lock:
            if self._active_consensus is None:
                raise RuntimeError('no active operation to complete')
            try:
                self._active_consensus.require_preflight_agreement()
                result = self._active_consensus.require_result_agreement()
            except ConsensusError:
                self._poisoned = True
                self._active_consensus = None
                raise
            self._active_consensus = None
            return result

    def abort_operation(self, reason: str = 'aborted') -> None:
        """Discard only a provably pre-collective operation.

        A complete successful preflight is the point after which peers may
        enter a blocking collective.  If cancellation or an exception arrives
        then, there is no safe local rollback of the process group: poison it
        and require a full group restart.  This avoids allowing a subsequent
        request to reuse peers that may still be blocked in the old round.
        """
        if type(reason) is not str or not reason.strip() or len(reason) > 512:
            raise ValueError('bounded non-empty distributed abort reason required')
        with self._lock:
            consensus = self._active_consensus
            if consensus is None:
                return
            if consensus.can_abort_before_collective():
                self._active_consensus = None
                return
            self._poisoned = True
            self._active_consensus = None
        raise ConsensusError(
            f'{consensus._action}: abort occurred after peers may have entered '
            f'the collective ({reason.strip()[:256]}); group poisoned and must be restarted')

    def record_rank_loss(self, rank: int, reason: str = 'rank loss') -> None:
        """Poison this group after rank loss; it must be recreated to recover."""
        with self._lock:
            self._health.mark_failed(rank, reason=reason)
            self._active_consensus = None
            self._poisoned = True

    def recovery_plan(self) -> List[dict]:
        return self._health.recovery_plan()

    def record_committed(
        self, rank: int, position: int,
        history_digest: str, cache_path: Optional[str] = None,
    ) -> None:
        self._health.update_committed(rank, position, history_digest, cache_path)

    def shard_for_rank(self, rank: int) -> ExpertShardRange:
        if type(rank) is not int or not (0 <= rank < self._world_size):
            raise ValueError(f'rank {rank} out of range for world_size {self._world_size}')
        return self._layout[rank]

    def rank_for_expert(self, expert_id: int) -> int:
        return rank_for_expert(expert_id, self._layout)
