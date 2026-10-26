"""Phase 1C: Extended 3+ Layer Lookahead Prefetch with Routing Affinity Clustering.

Phase 1B (``expert_runtime.SparseExpertRuntime``) prefetches the next predicted
layer's experts in a background thread.  Phase 1C extends that to 3+ layers and
adds routing affinity clustering: experts that frequently fire together in
adjacent layers are grouped into clusters, so a single prefetch decision pulls
the whole cluster — not just the router's point prediction — dramatically
improving hit rate for correlated routing patterns.

Design:
  - ``AffinityClusterMap`` — builds expert co-occurrence clusters from observed
    route traces.  Persisted per-checkpoint alongside the Phase 1B route profile.
  - ``ExtendedPrefetchController`` — consumes a route trace, looks up the
    affinity cluster for each fired expert, and submits prefetch requests for
    the next N+1 to N+K layers (configurable depth, default 3).
  - ``ExtendedPrefetchSession`` — thin wrapper around a
    ``Phase1BOptimizedSession`` (or any session with an expert telemetry source).

None of these classes execute model tensors.  They observe route traces that the
production session emits and submit prefetch hints to the underlying expert store;
the actual prefetch is handled by ``DiskExpertStore``'s existing read-ahead
workers.  If the store does not expose a ``submit_prefetch`` hook the hints are
silently skipped — the wrapper never blocks generation.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

PHASE1C_PREFETCH_DEPTH_DEFAULT = 3      # layers ahead to prefetch (Phase 1C: 3+)
PHASE1C_CLUSTER_COOCCUR_THRESHOLD = 3   # min co-occurrences to link two experts
PHASE1C_CLUSTER_MIN_FIRE_COUNT = 5      # expert must have fired at least this many times
PHASE1C_STATE_FORMAT = "openagent.phase1c-affinity-cluster-state.v1"
PHASE1C_MAX_CLUSTER_SIZE = 32           # cap cluster size to avoid over-prefetch


# ---------------------------------------------------------------------------
# AffinityClusterMap
# ---------------------------------------------------------------------------

@dataclass
class AffinityCluster:
    """A set of experts that tend to fire together in the same or adjacent layers."""
    cluster_id: str
    layer_id: int
    expert_ids: FrozenSet[str]
    centroid_expert: str  # most-frequently-fired member
    fire_count: int       # total fires of centroid (proxy for cluster confidence)


class AffinityClusterMap:
    """Build and query an expert routing affinity cluster map.

    Terminology
    -----------
    - **Co-occurrence**: two experts fired in the same forward pass at the same layer.
    - **Successor link**: expert E fired in layer L → expert F frequently fired in layer
      L+1 in the same session (cross-layer affinity).
    - **Cluster**: transitive closure of co-occurrence links above the threshold, capped
      at ``PHASE1C_MAX_CLUSTER_SIZE``.

    The map is built incrementally from route traces and persisted as a small JSON blob
    alongside the Phase 1B route profile.  It does not store raw token data.
    """

    def __init__(self) -> None:
        # {layer_id: {expert_id: {peer_expert_id: cooccur_count}}}
        self._cooccur: Dict[int, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        # {layer_id: {expert_id: {next_layer_expert: cross_count}}}
        self._successor: Dict[int, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        # {layer_id: {expert_id: fire_count}}
        self._fire_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._lock = threading.RLock()
        self._clusters: Optional[Dict[int, List[AffinityCluster]]] = None  # cached, invalidated on update

    # -- building the map -------------------------------------------------------

    def record_trace(self, layer_traces: List[Tuple[int, List[str]]]) -> None:
        """Ingest one forward-pass route trace.

        Args:
            layer_traces: [(layer_id, [expert_id, ...]), ...] ordered by layer.
        """
        with self._lock:
            self._clusters = None  # invalidate cache
            for i, (layer_id, expert_ids) in enumerate(layer_traces):
                for eid in expert_ids:
                    self._fire_counts[layer_id][eid] += 1
                # co-occurrence within the same layer
                for j, e1 in enumerate(expert_ids):
                    for e2 in expert_ids[j + 1:]:
                        if e1 != e2:
                            self._cooccur[layer_id][e1][e2] += 1
                            self._cooccur[layer_id][e2][e1] += 1
                # cross-layer successor links
                if i + 1 < len(layer_traces):
                    next_layer_id, next_experts = layer_traces[i + 1]
                    for e_cur in expert_ids:
                        for e_next in next_experts:
                            self._successor[layer_id][e_cur][e_next] += 1

    def _build_clusters(self) -> Dict[int, List[AffinityCluster]]:
        """Build clusters from accumulated co-occurrence data (call under lock)."""
        result: Dict[int, List[AffinityCluster]] = defaultdict(list)
        for layer_id, expert_peers in self._cooccur.items():
            fire_counts = self._fire_counts[layer_id]
            visited: Set[str] = set()
            for seed_expert in fire_counts:
                if seed_expert in visited:
                    continue
                if fire_counts[seed_expert] < PHASE1C_CLUSTER_MIN_FIRE_COUNT:
                    continue
                # BFS over co-occurrence links above threshold
                cluster_members: Set[str] = {seed_expert}
                frontier = [seed_expert]
                while frontier and len(cluster_members) < PHASE1C_MAX_CLUSTER_SIZE:
                    node = frontier.pop()
                    for peer, count in expert_peers.get(node, {}).items():
                        if peer not in cluster_members and count >= PHASE1C_CLUSTER_COOCCUR_THRESHOLD:
                            cluster_members.add(peer)
                            frontier.append(peer)
                visited.update(cluster_members)
                if len(cluster_members) < 2:
                    continue
                centroid = max(cluster_members, key=lambda e: fire_counts.get(e, 0))
                cid = hashlib.sha1(
                    (str(layer_id) + ":" + ",".join(sorted(cluster_members))).encode()
                ).hexdigest()[:16]
                result[layer_id].append(AffinityCluster(
                    cluster_id=cid,
                    layer_id=layer_id,
                    expert_ids=frozenset(cluster_members),
                    centroid_expert=centroid,
                    fire_count=fire_counts[centroid],
                ))
        return dict(result)

    def get_cluster_experts(self, layer_id: int, fired_experts: Sequence[str]) -> FrozenSet[str]:
        """Return the union of cluster members for all fired experts in a layer.

        Used by the prefetcher: when expert E fires in layer L, also prefetch all
        other experts in E's affinity cluster (they are likely to fire in nearby layers).
        """
        with self._lock:
            if self._clusters is None:
                self._clusters = self._build_clusters()
            clusters = self._clusters.get(layer_id, [])
            fired_set = set(fired_experts)
            union: Set[str] = set(fired_experts)
            for cluster in clusters:
                if cluster.expert_ids & fired_set:
                    union.update(cluster.expert_ids)
            return frozenset(union)

    def get_successor_experts(self, layer_id: int, fired_experts: Sequence[str],
                               top_k: int = 16) -> FrozenSet[str]:
        """Return the top-K most likely successor experts in the next layer."""
        with self._lock:
            successor_counts: Dict[str, int] = defaultdict(int)
            for eid in fired_experts:
                for succ_eid, count in self._successor[layer_id].get(eid, {}).items():
                    successor_counts[succ_eid] += count
            sorted_succ = sorted(successor_counts.items(), key=lambda x: x[1], reverse=True)
            return frozenset(s for s, _ in sorted_succ[:top_k])

    # -- persistence ------------------------------------------------------------

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "format": PHASE1C_STATE_FORMAT,
                "cooccur": {str(l): {e: dict(peers) for e, peers in ex.items()}
                            for l, ex in self._cooccur.items()},
                "successor": {str(l): {e: dict(succs) for e, succs in ex.items()}
                              for l, ex in self._successor.items()},
                "fire_counts": {str(l): dict(ex) for l, ex in self._fire_counts.items()},
            }

    @classmethod
    def from_dict(cls, data: dict) -> "AffinityClusterMap":
        obj = cls()
        if data.get("format") != PHASE1C_STATE_FORMAT:
            logger.warning("phase1c affinity map: unknown format %s, discarding", data.get("format"))
            return obj
        for l_str, ex in data.get("cooccur", {}).items():
            l = int(l_str)
            for e, peers in ex.items():
                obj._cooccur[l][e] = defaultdict(int, peers)
        for l_str, ex in data.get("successor", {}).items():
            l = int(l_str)
            for e, succs in ex.items():
                obj._successor[l][e] = defaultdict(int, succs)
        for l_str, ex in data.get("fire_counts", {}).items():
            l = int(l_str)
            obj._fire_counts[l] = defaultdict(int, ex)
        return obj

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, separators=(",", ":"))
        tmp.replace(path)
        logger.debug("phase1c affinity map saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "AffinityClusterMap":
        try:
            with open(path, "r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        except FileNotFoundError:
            return cls()
        except Exception as exc:
            logger.warning("phase1c affinity map load failed (%s), starting fresh", exc)
            return cls()

    def stats(self) -> dict:
        with self._lock:
            if self._clusters is None:
                self._clusters = self._build_clusters()
            total_clusters = sum(len(v) for v in self._clusters.values())
            total_experts_in_clusters = sum(
                sum(len(c.expert_ids) for c in clusters)
                for clusters in self._clusters.values()
            )
            return {
                "layers_with_cooccur": len(self._cooccur),
                "total_clusters": total_clusters,
                "total_experts_in_clusters": total_experts_in_clusters,
                "total_successor_links": sum(
                    sum(len(v) for v in ex.values())
                    for ex in self._successor.values()
                ),
            }


# ---------------------------------------------------------------------------
# ExtendedPrefetchController
# ---------------------------------------------------------------------------

class ExtendedPrefetchController:
    """Submit extended (3+ layer) prefetch hints to the underlying expert store.

    The controller receives route traces from the production session, expands
    each fired expert to its affinity cluster, and calls ``store.submit_prefetch``
    for layers ``current + 1`` through ``current + depth``.

    If the store does not expose ``submit_prefetch`` the hints are dropped;
    generation is never blocked.

    Thread-safety: all public methods are thread-safe.
    """

    def __init__(
        self,
        affinity_map: AffinityClusterMap,
        depth: int = PHASE1C_PREFETCH_DEPTH_DEFAULT,
    ) -> None:
        if depth < 1:
            raise ValueError("prefetch depth must be >= 1")
        self.affinity_map = affinity_map
        self.depth = depth
        self._lock = threading.Lock()
        # metrics
        self._hints_submitted = 0
        self._hints_dropped = 0
        self._trace_calls = 0

    def on_layer_complete(
        self,
        layer_id: int,
        fired_experts: Sequence[str],
        all_layer_traces: List[Tuple[int, List[str]]],
        store: Any,
    ) -> None:
        """Call after layer ``layer_id`` completes its forward pass.

        Args:
            layer_id: The layer that just finished.
            fired_experts: Experts that fired in this layer.
            all_layer_traces: Accumulated route trace for this forward pass so far
                (list of (layer_id, [expert_ids]) tuples from layer 0 to now).
            store: The expert store — may expose ``submit_prefetch(layer_id, expert_ids)``.
        """
        with self._lock:
            self._trace_calls += 1

        # Update affinity map with the full trace so far
        if all_layer_traces:
            self.affinity_map.record_trace(all_layer_traces)

        # Build prefetch set for layers current+1 .. current+depth
        # Layer current+1: cluster expansion of current fired experts
        # Layer current+2+: successor prediction chained forward
        submit_fn = getattr(store, "submit_prefetch", None)

        prefetch_expert_ids = list(fired_experts)
        for offset in range(1, self.depth + 1):
            target_layer = layer_id + offset
            # Expand to affinity cluster for same-layer correlation
            cluster_experts = self.affinity_map.get_cluster_experts(target_layer, prefetch_expert_ids)
            # Add cross-layer successors
            succ_experts = self.affinity_map.get_successor_experts(layer_id + offset - 1, prefetch_expert_ids)
            combined = frozenset(cluster_experts) | frozenset(succ_experts)
            if combined and submit_fn is not None:
                try:
                    submit_fn(target_layer, list(combined))
                    with self._lock:
                        self._hints_submitted += len(combined)
                except Exception as exc:
                    logger.debug("phase1c: submit_prefetch error at layer %d: %s", target_layer, exc)
                    with self._lock:
                        self._hints_dropped += len(combined)
            elif combined:
                with self._lock:
                    self._hints_dropped += len(combined)
            # Roll forward: next iteration predicts from these experts
            prefetch_expert_ids = list(combined) if combined else list(fired_experts)

    def metrics(self) -> dict:
        with self._lock:
            return {
                "phase1c_trace_calls": self._trace_calls,
                "phase1c_prefetch_hints_submitted": self._hints_submitted,
                "phase1c_prefetch_hints_dropped": self._hints_dropped,
                "phase1c_depth": self.depth,
                **self.affinity_map.stats(),
            }


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def affinity_map_path(state_directory: Path, model_id: str) -> Path:
    """Return the path for the affinity map file in a Phase 1B state directory."""
    key = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:24]
    return state_directory / f"phase1c-affinity-{key}.json"


__all__ = [
    "PHASE1C_STATE_FORMAT",
    "AffinityCluster",
    "AffinityClusterMap",
    "ExtendedPrefetchController",
    "affinity_map_path",
]
