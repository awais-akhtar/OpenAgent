"""Speculative Tree Decoding — Medusa-style multi-head draft with tree verification.

Extends Phase 1D's single-sequence speculative decode to a *tree* of candidate
continuations so that one target-model forward pass can verify multiple
draft branches simultaneously.

How it works
------------
1. K *draft heads* each predict the next token in parallel from the same hidden
   state.  This produces K candidate tokens at depth 1.
2. Each candidate is then expanded by K more heads to depth 2, yielding up to
   K² leaves, but a configurable branching factor limits growth.
3. The target model evaluates all tree paths in one batched forward call.
4. The longest accepted prefix path is chosen; rejected suffixes are discarded.
5. Net effect: more tokens accepted per target-model step → higher throughput.

Components
----------
``DraftTreeNode``          -- one node in the candidate tree
``DraftTree``              -- builds and traverses the speculation tree
``DraftHeadSimulator``     -- K parallel draft heads (no real weights; samples
                             from an optional n-gram LM or uniform distribution)
``TreeSpeculativeDecoder`` -- orchestrates build → verify → accept
``Phase3SpecTreeSession``  -- wraps a base session; replaces its single-sequence
                             speculative decoder with the tree decoder
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Type alias: draft_fn(context_ids) → List[int]  (K candidate next tokens)
DraftFn = Callable[[List[int]], List[int]]

# Type alias: verify_fn(context_ids, draft_tokens) → List[float]  (acceptance probs)
VerifyFn = Callable[[List[int], List[int]], List[float]]


# ---------------------------------------------------------------------------
# DraftTreeNode
# ---------------------------------------------------------------------------

@dataclass
class DraftTreeNode:
    """One node in the draft speculation tree."""
    token:    int
    depth:    int
    prob:     float              # draft probability or acceptance score
    parent:   Optional["DraftTreeNode"] = None
    children: List["DraftTreeNode"]    = field(default_factory=list)
    accepted: bool = False

    def path_to_root(self) -> List[int]:
        """Return token sequence from root to this node (inclusive)."""
        tokens: List[int] = []
        node: Optional[DraftTreeNode] = self
        while node is not None:
            tokens.append(node.token)
            node = node.parent
        return list(reversed(tokens))


# ---------------------------------------------------------------------------
# DraftTree
# ---------------------------------------------------------------------------

class DraftTree:
    """Builds a tree of candidate draft continuations.

    Args:
        branching_factor: how many children each node expands to
        max_depth:        maximum tree depth (draft length)
    """

    def __init__(self, branching_factor: int = 4, max_depth: int = 3) -> None:
        self._bf    = branching_factor
        self._depth = max_depth
        self.roots: List[DraftTreeNode] = []
        self._all_nodes: List[DraftTreeNode] = []

    def build(
        self,
        context_ids: List[int],
        draft_fn: DraftFn,
    ) -> None:
        """Expand the tree from context_ids using draft_fn."""
        self.roots.clear()
        self._all_nodes.clear()
        # Depth 1 — K roots
        root_tokens = draft_fn(context_ids)[:self._bf]
        for tok in root_tokens:
            node = DraftTreeNode(token=tok, depth=1, prob=1.0 / len(root_tokens))
            self.roots.append(node)
            self._all_nodes.append(node)

        if self._depth < 2:
            return

        # Expand deeper levels BFS
        frontier: List[DraftTreeNode] = list(self.roots)
        for d in range(2, self._depth + 1):
            next_frontier: List[DraftTreeNode] = []
            for parent in frontier:
                ctx = context_ids + parent.path_to_root()
                child_toks = draft_fn(ctx)[:self._bf]
                for ctok in child_toks:
                    child = DraftTreeNode(
                        token=ctok, depth=d,
                        prob=parent.prob / len(child_toks),
                        parent=parent)
                    parent.children.append(child)
                    self._all_nodes.append(child)
                    next_frontier.append(child)
            frontier = next_frontier

    def leaves(self) -> List[DraftTreeNode]:
        return [n for n in self._all_nodes if not n.children]

    def all_paths(self) -> List[List[DraftTreeNode]]:
        """All root-to-leaf paths."""
        paths: List[List[DraftTreeNode]] = []

        def _dfs(node: DraftTreeNode, path: List[DraftTreeNode]) -> None:
            path.append(node)
            if not node.children:
                paths.append(list(path))
            else:
                for child in node.children:
                    _dfs(child, path)
            path.pop()

        for root in self.roots:
            _dfs(root, [])
        return paths

    @property
    def node_count(self) -> int:
        return len(self._all_nodes)


# ---------------------------------------------------------------------------
# DraftHeadSimulator
# ---------------------------------------------------------------------------

class DraftHeadSimulator:
    """K parallel draft heads for tree speculation.

    In production, these would be small MLP heads trained jointly with the
    target model (Medusa style).  Here they delegate to an optional n-gram LM
    or sample uniformly from the vocabulary.
    """

    def __init__(
        self,
        k: int = 4,
        vocab_size: int = 32000,
        ngram_fn: Optional[DraftFn] = None,
    ) -> None:
        self._k          = k
        self._vocab_size = vocab_size
        self._ngram_fn   = ngram_fn
        self._calls      = 0

    def draft(self, context_ids: List[int]) -> List[int]:
        """Return K candidate next-tokens."""
        self._calls += 1
        if self._ngram_fn is not None:
            try:
                base = self._ngram_fn(context_ids)
                if isinstance(base, (list, np.ndarray)) and len(base) > 0:
                    # top-1 from n-gram + K-1 random neighbours
                    top_tok = int(np.argmax(base)) if len(base) == self._vocab_size else int(base[0])
                    random_toks = np.random.randint(0, self._vocab_size, size=self._k - 1).tolist()
                    return [top_tok] + random_toks
            except Exception:
                pass
        # Fallback: uniform random
        return np.random.randint(0, self._vocab_size, size=self._k).tolist()

    @property
    def calls(self) -> int:
        return self._calls


# ---------------------------------------------------------------------------
# TreeSpeculativeDecoder
# ---------------------------------------------------------------------------

class TreeSpeculativeDecoder:
    """Orchestrates tree draft → batch verify → greedy accept.

    Args:
        draft_heads:     ``DraftHeadSimulator`` or any object with ``draft(ctx)``
        verify_fn:       target-model function returning per-token acceptance
                         logits; signature: ``(context, draft_tokens) → logits``
        branching_factor: tree width at each depth
        max_depth:       maximum draft depth (tokens per tree step)
        acceptance_threshold: minimum acceptance prob to admit a token
    """

    def __init__(
        self,
        draft_heads: DraftHeadSimulator,
        verify_fn: Optional[VerifyFn] = None,
        *,
        branching_factor: int = 3,
        max_depth: int = 3,
        acceptance_threshold: float = 0.5,
    ) -> None:
        self._heads    = draft_heads
        self._verify   = verify_fn
        self._bf       = branching_factor
        self._depth    = max_depth
        self._threshold = acceptance_threshold

        self._tree = DraftTree(branching_factor, max_depth)

        # Telemetry
        self._steps         = 0
        self._tokens_accepted = 0
        self._tokens_rejected = 0

    def step(self, context_ids: List[int]) -> Tuple[List[int], int]:
        """Run one tree-speculative step.

        Returns:
            accepted_tokens: list of accepted token ids (at least 1)
            tree_nodes_evaluated: total draft nodes verified
        """
        self._steps += 1
        self._tree.build(context_ids, self._heads.draft)
        paths = self._tree.all_paths()

        if not paths:
            # Degenerate case: no draft tokens; return a random fallback
            tok = int(np.random.randint(0, self._heads._vocab_size))
            self._tokens_accepted += 1
            return [tok], 0

        best_path: List[int] = []
        tree_nodes = self._tree.node_count

        if self._verify is not None:
            # Batch-verify all unique draft tokens across the tree
            all_tokens = [n.token for n in self._tree._all_nodes]
            try:
                logits = self._verify(context_ids, all_tokens)
            except Exception as exc:
                logger.debug("tree_spec: verify_fn failed: %s", exc)
                logits = [0.5] * len(all_tokens)

            # Build acceptance map: node_index → accepted bool
            accepted_map: Dict[int, bool] = {}
            for i, node in enumerate(self._tree._all_nodes):
                prob = float(logits[i]) if i < len(logits) else 0.5
                accepted_map[id(node)] = prob >= self._threshold
                node.accepted = accepted_map[id(node)]

            # Find longest accepted prefix path
            for path in sorted(paths, key=lambda p: -len(p)):
                if all(accepted_map.get(id(n), False) for n in path):
                    best_path = [n.token for n in path]
                    break
        else:
            # No verifier: accept everything (useful for draft-only testing)
            longest = max(paths, key=len)
            best_path = [n.token for n in longest]

        # Always accept at least one token
        if not best_path and self._tree.roots:
            best_path = [self._tree.roots[0].token]

        self._tokens_accepted += len(best_path)
        self._tokens_rejected += tree_nodes - len(best_path)
        return best_path, tree_nodes

    def metrics(self) -> dict:
        return {
            "spec_tree_steps":            self._steps,
            "spec_tree_tokens_accepted":  self._tokens_accepted,
            "spec_tree_tokens_rejected":  self._tokens_rejected,
            "spec_tree_accept_rate": (
                self._tokens_accepted /
                max(1, self._tokens_accepted + self._tokens_rejected)
            ),
            "spec_tree_draft_calls": self._heads.calls,
        }


# ---------------------------------------------------------------------------
# Phase3SpecTreeSession
# ---------------------------------------------------------------------------

class Phase3SpecTreeSession:
    """Wraps a base session; replaces or supplements its speculative decoder
    with the tree-speculative decoder.

    ``generate()`` uses the tree decoder to produce tokens in multi-token
    steps.  Each step:
    1. Calls ``TreeSpeculativeDecoder.step(context)`` → accepted tokens.
    2. Appends accepted tokens to context.
    3. Repeats until max_new_tokens reached or stop_token hit.
    """

    def __init__(
        self,
        base_session: Any,
        draft_heads: Optional[DraftHeadSimulator] = None,
        verify_fn: Optional[VerifyFn] = None,
        *,
        branching_factor: int = 3,
        max_depth: int = 3,
        vocab_size: int = 32000,
    ) -> None:
        if base_session is None:
            raise ValueError("base_session required")
        self.base_session = base_session
        heads = draft_heads or DraftHeadSimulator(
            k=branching_factor, vocab_size=vocab_size)
        self._decoder = TreeSpeculativeDecoder(
            heads, verify_fn,
            branching_factor=branching_factor,
            max_depth=max_depth)
        self._lock   = threading.RLock()
        self._closed = False

    # ---- generation with tree spec ------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Phase3SpecTreeSession is closed")

    def generate(
        self,
        *args: Any,
        context_ids: Optional[List[int]] = None,
        max_new_tokens: int = 128,
        stop_token_id: Optional[int] = None,
        cancelled: Optional[Callable[[], bool]] = None,
        **kwargs: Any,
    ) -> List[int]:
        with self._lock:
            self._check_open()
        ctx = list(context_ids or (args[0] if args else []))
        generated: List[int] = []

        while len(generated) < max_new_tokens:
            if cancelled is not None and cancelled():
                break
            accepted, _ = self._decoder.step(ctx + generated)
            for tok in accepted:
                if len(generated) >= max_new_tokens:
                    break
                generated.append(tok)
                if stop_token_id is not None and tok == stop_token_id:
                    return generated
        return generated

    def stream(
        self,
        *args: Any,
        context_ids: Optional[List[int]] = None,
        max_new_tokens: int = 128,
        stop_token_id: Optional[int] = None,
        cancelled: Optional[Callable[[], bool]] = None,
        **kwargs: Any,
    ):
        with self._lock:
            self._check_open()
        ctx = list(context_ids or (args[0] if args else []))
        generated_count = 0

        while generated_count < max_new_tokens:
            if cancelled is not None and cancelled():
                break
            accepted, _ = self._decoder.step(ctx + [])
            for tok in accepted:
                if generated_count >= max_new_tokens:
                    break
                generated_count += 1
                yield tok
                if stop_token_id is not None and tok == stop_token_id:
                    return
            ctx.extend(accepted[:max_new_tokens - generated_count + len(accepted)])

    def on_token_routed(self, layer_id: int, expert_ids: List[str]) -> None:
        hook = getattr(self.base_session, "on_token_routed", None)
        if callable(hook):
            hook(layer_id, expert_ids)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.base_session, name)

    def metrics(self) -> dict:
        base: dict = {}
        get_m = getattr(self.base_session, "metrics", None)
        if callable(get_m):
            base = get_m()
        return {**base, "spec_tree": self._decoder.metrics()}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        close_fn = getattr(self.base_session, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as exc:
                logger.warning("spec_tree: base session close error: %s", exc)

    def __enter__(self) -> "Phase3SpecTreeSession":
        self._check_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def wrap_session_with_spec_tree(
    base_session: Any,
    *,
    branching_factor: int = 3,
    max_depth: int = 3,
    vocab_size: int = 32000,
    verify_fn: Optional[VerifyFn] = None,
) -> Phase3SpecTreeSession:
    """Wrap a base session with tree-speculative decoding."""
    return Phase3SpecTreeSession(
        base_session, verify_fn=verify_fn,
        branching_factor=branching_factor,
        max_depth=max_depth,
        vocab_size=vocab_size)


__all__ = [
    "DraftTreeNode",
    "DraftTree",
    "DraftHeadSimulator",
    "TreeSpeculativeDecoder",
    "Phase3SpecTreeSession",
    "wrap_session_with_spec_tree",
]
