"""Phase 1D: Speculative Expert Pre-Placement.

The Phase 1B/1C prefetch controllers submit I/O hints *after* a layer completes
its routing decision.  Speculative placement goes one step further: it predicts
which experts will be needed *before* the routing token arrives, using a Markov
chain trained over the session's own routing history, and pre-moves those experts
to a warmer tier (disk → RAM or RAM → VRAM) so they are already staged when the
router makes its decision.

Architecture
------------
``RoutingMarkovChain``
    Learns transition probabilities between routing patterns across tokens.
    A "pattern" is the sorted tuple of expert IDs fired in one layer.  The
    chain tracks P(next_pattern | current_pattern) with smoothed counts.

``TierPlacementScheduler``
    Converts a placement prediction into concrete placement requests against
    the underlying tiered store.  Requests are fire-and-forget: a rejected
    hint never blocks the caller.

``SpeculativePlacementEngine``
    Combines the Markov chain with the scheduler.  Exposes an
    ``on_token_routed(layer_id, expert_ids)`` hook the production session
    calls after each token's routing decision; the engine updates the chain
    and fires placement hints for the predicted next pattern.

State persistence
-----------------
The Markov chain is saved per checkpoint alongside the Phase 1C affinity map.
On a cold start the chain is empty; it warms up within the first few hundred
tokens and asymptotes quickly for repetitive prompts.

All public methods are thread-safe.  No threads are started.
"""
from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PHASE1D_MARKOV_STATE_FORMAT  = "openagent.phase1d-routing-markov.v1"
PHASE1D_MARKOV_MAX_PATTERNS  = 4096   # cap memory use
PHASE1D_MARKOV_TOP_K         = 8      # top-K successors to prefetch
PHASE1D_MARKOV_SMOOTH_ALPHA  = 0.5    # Laplace smoothing pseudo-count
PHASE1D_PLACEMENT_MIN_CONF   = 0.10   # minimum P(successor) to fire hint


# ---------------------------------------------------------------------------
# RoutingMarkovChain
# ---------------------------------------------------------------------------

@dataclass
class PatternTransition:
    """Counts of observed transitions from one routing pattern to another."""
    total: int = 0
    successors: Dict[str, int] = field(default_factory=dict)   # pattern_key → count


def _pattern_key(expert_ids: Sequence[str]) -> str:
    """Canonical string key for a routing pattern (sorted, comma-joined)."""
    return ",".join(sorted(expert_ids))


class RoutingMarkovChain:
    """First-order Markov chain over per-layer routing patterns.

    State: (layer_id, pattern_key) — the set of experts that fired in the
    most recent token for a given layer.

    Transition: from state S to state S' when the same layer fires a
    different expert set in the next token.

    The chain is trained online and never requires a training corpus.
    """

    def __init__(self) -> None:
        # {(layer_id, pattern_key): PatternTransition}
        self._transitions: Dict[Tuple[int, str], PatternTransition] = {}
        # last observed pattern per layer
        self._last_pattern: Dict[int, str] = {}
        self._lock = threading.RLock()
        self._total_updates = 0

    def record(self, layer_id: int, expert_ids: Sequence[str]) -> None:
        """Observe one routing event and update the chain."""
        key = _pattern_key(expert_ids)
        with self._lock:
            if layer_id in self._last_pattern:
                prev = self._last_pattern[layer_id]
                state = (layer_id, prev)
                if state not in self._transitions:
                    # Evict oldest state if at capacity
                    if len(self._transitions) >= PHASE1D_MARKOV_MAX_PATTERNS:
                        oldest = next(iter(self._transitions))
                        del self._transitions[oldest]
                    self._transitions[state] = PatternTransition()
                t = self._transitions[state]
                t.total += 1
                t.successors[key] = t.successors.get(key, 0) + 1
            self._last_pattern[layer_id] = key
            self._total_updates += 1

    def predict(self, layer_id: int, expert_ids: Sequence[str],
                top_k: int = PHASE1D_MARKOV_TOP_K) -> List[Tuple[str, float]]:
        """Return (pattern_key, probability) pairs for the most likely next patterns.

        Args:
            layer_id: Which layer to predict for.
            expert_ids: Current firing pattern (used as the Markov state).
            top_k: Maximum number of successors to return.

        Returns:
            Sorted list of (pattern_key, prob) descending by probability.
            Empty list if no data available yet.
        """
        key = _pattern_key(expert_ids)
        state = (layer_id, key)
        with self._lock:
            t = self._transitions.get(state)
        if t is None or t.total == 0:
            return []
        alpha = PHASE1D_MARKOV_SMOOTH_ALPHA
        vocab = len(t.successors)
        denom = t.total + alpha * vocab
        predictions = []
        for succ_key, count in t.successors.items():
            prob = (count + alpha) / denom
            if prob >= PHASE1D_PLACEMENT_MIN_CONF:
                predictions.append((succ_key, prob))
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]

    def expert_ids_from_key(self, pattern_key: str) -> List[str]:
        """Decode a pattern key back to a list of expert IDs."""
        if not pattern_key:
            return []
        return pattern_key.split(",")

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "format": PHASE1D_MARKOV_STATE_FORMAT,
                "transitions": {
                    f"{lid}:{pk}": {
                        "total": t.total,
                        "successors": dict(t.successors),
                    }
                    for (lid, pk), t in self._transitions.items()
                },
                "total_updates": self._total_updates,
            }

    @classmethod
    def from_dict(cls, data: dict) -> "RoutingMarkovChain":
        obj = cls()
        if data.get("format") != PHASE1D_MARKOV_STATE_FORMAT:
            logger.warning("phase1d markov: unknown format %s, starting fresh",
                           data.get("format"))
            return obj
        for compound_key, td in data.get("transitions", {}).items():
            lid_str, pk = compound_key.split(":", 1)
            state = (int(lid_str), pk)
            obj._transitions[state] = PatternTransition(
                total=td["total"],
                successors=dict(td["successors"]),
            )
        obj._total_updates = data.get("total_updates", 0)
        return obj

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, separators=(",", ":"))
        tmp.replace(path)
        logger.debug("phase1d markov chain saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "RoutingMarkovChain":
        try:
            with open(path, "r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        except FileNotFoundError:
            return cls()
        except Exception as exc:
            logger.warning("phase1d markov load failed (%s), starting fresh", exc)
            return cls()

    def stats(self) -> dict:
        with self._lock:
            return {
                "markov_states": len(self._transitions),
                "markov_total_updates": self._total_updates,
                "markov_layers_tracked": len(self._last_pattern),
            }


# ---------------------------------------------------------------------------
# TierPlacementScheduler
# ---------------------------------------------------------------------------

class TierPlacementScheduler:
    """Translate placement predictions into store hints.

    The scheduler calls ``store.promote_expert(expert_id, target_tier)`` when
    available.  The target tier is determined by the store's current tier for
    that expert (available via ``store.tier_of(expert_id)``).  If neither
    method is available on the store, hints are silently dropped.

    Tier order (coldest → hottest): disk (0) → ram (1) → vram (2).
    Pre-placement advances by one tier at a time.
    """

    TIER_DISK  = 0
    TIER_RAM   = 1
    TIER_VRAM  = 2

    def __init__(self) -> None:
        self._hints_submitted = 0
        self._hints_dropped   = 0
        self._lock = threading.Lock()

    def schedule(self, store: Any, expert_ids: Sequence[str],
                 target_tier: int = TIER_RAM) -> None:
        """Submit promotion hints for a list of expert IDs.

        Args:
            store: The expert store (duck-typed — ``promote_expert`` optional).
            expert_ids: Experts to pre-place.
            target_tier: Desired tier (default RAM).
        """
        promote_fn = getattr(store, "promote_expert", None)
        if promote_fn is None:
            # Try the submit_prefetch hook as a lighter alternative
            prefetch_fn = getattr(store, "submit_prefetch", None)
            if prefetch_fn is not None:
                try:
                    prefetch_fn(-1, list(expert_ids))  # layer -1 = "any layer"
                    with self._lock:
                        self._hints_submitted += len(expert_ids)
                except Exception as exc:
                    logger.debug("phase1d placement: submit_prefetch error: %s", exc)
                    with self._lock:
                        self._hints_dropped += len(expert_ids)
            else:
                with self._lock:
                    self._hints_dropped += len(expert_ids)
            return

        for eid in expert_ids:
            try:
                promote_fn(eid, target_tier)
                with self._lock:
                    self._hints_submitted += 1
            except Exception as exc:
                logger.debug("phase1d: promote_expert(%s) error: %s", eid, exc)
                with self._lock:
                    self._hints_dropped += 1

    def stats(self) -> dict:
        with self._lock:
            return {
                "placement_hints_submitted": self._hints_submitted,
                "placement_hints_dropped":   self._hints_dropped,
            }


# ---------------------------------------------------------------------------
# SpeculativePlacementEngine
# ---------------------------------------------------------------------------

class SpeculativePlacementEngine:
    """Combine Markov routing prediction with tier placement scheduling.

    Call ``on_token_routed(layer_id, expert_ids)`` after each layer's routing
    decision.  The engine updates the Markov chain and immediately fires
    placement hints for the predicted next routing pattern.

    If no store is bound, hints are accumulated but not submitted.
    """

    def __init__(
        self,
        chain: RoutingMarkovChain,
        store: Any = None,
        top_k: int = PHASE1D_MARKOV_TOP_K,
        target_tier: int = TierPlacementScheduler.TIER_RAM,
    ) -> None:
        self.chain     = chain
        self.store     = store
        self.top_k     = top_k
        self.target_tier = target_tier
        self._scheduler  = TierPlacementScheduler()
        self._lock = threading.Lock()

    def bind_store(self, store: Any) -> None:
        """Attach or replace the expert store."""
        with self._lock:
            self.store = store

    def on_token_routed(self, layer_id: int, expert_ids: Sequence[str]) -> None:
        """Handle one routing event.

        1. Record the current pattern in the Markov chain.
        2. Predict the likely next pattern.
        3. Expand each prediction to its expert IDs and fire placement hints.
        """
        self.chain.record(layer_id, expert_ids)
        predictions = self.chain.predict(layer_id, expert_ids, top_k=self.top_k)
        if not predictions:
            return
        with self._lock:
            store = self.store
        if store is None:
            return
        # Collect all predicted expert IDs (deduplicated)
        predicted_expert_ids: list = []
        seen: set = set()
        for pattern_key, prob in predictions:
            for eid in self.chain.expert_ids_from_key(pattern_key):
                if eid not in seen:
                    predicted_expert_ids.append(eid)
                    seen.add(eid)
        self._scheduler.schedule(store, predicted_expert_ids, self.target_tier)

    def metrics(self) -> dict:
        return {
            **self.chain.stats(),
            **self._scheduler.stats(),
        }


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------

def markov_chain_path(state_directory: Path, model_id: str) -> Path:
    """Return the path for the Markov chain file in a Phase 1B state directory."""
    import hashlib
    key = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:24]
    return state_directory / f"phase1d-markov-{key}.json"


__all__ = [
    "RoutingMarkovChain",
    "TierPlacementScheduler",
    "SpeculativePlacementEngine",
    "markov_chain_path",
    "PHASE1D_MARKOV_STATE_FORMAT",
]
