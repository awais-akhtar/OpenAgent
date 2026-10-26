"""Phase 1D integration bridge: CPU kernels + speculative placement + speculative decoding.

This module is the single entry point for Phase 1D.  It wraps a
``Phase1COptimizedSession`` (or any compatible session) with three Phase 1D
capabilities:

1. **Numba-JIT CPU kernels** — replaces plain NumPy FFN paths with
   compiled native kernels for all model families
   (``phase1d_cpu_kernels.Phase1DKernelDispatch``).

2. **Speculative expert pre-placement** — a Markov chain trained on the
   session's own routing history pre-moves experts to warmer tiers before
   the routing token arrives
   (``phase1d_speculative_placement.SpeculativePlacementEngine``).

3. **Grammar-constrained speculative decoding** — draft-then-verify loop
   with FSM-masked logits for structured output
   (``phase1d_speculative_decode.GrammarSpeculativeDecoder``).

The same ``generate`` / ``stream`` / ``close`` / ``metrics`` interface as
Phases 1B and 1C is preserved.  Phase 1D wraps Phase 1C which wraps Phase 1B:

    base_session
      └── Phase1BOptimizedSession
            └── Phase1COptimizedSession
                  └── Phase1DOptimizedSession   ← this module

Lifecycle
---------
1. On ``__init__``: load the Markov chain from the Phase 1B state directory
   (if present), wire the Numba dispatch to the kernel registry, and bind the
   placement engine to the base store.
2. On ``generate`` / ``stream``: reset the route trace, call Phase 1C's
   generate/stream (which fires Phase 1C prefetch hooks), and optionally route
   through ``GrammarSpeculativeDecoder`` when structured output is requested.
3. On ``close``: save the Markov chain, then close the Phase 1C session.

This module starts no threads.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import logging

from .failure_reporting import add_failure_note, failure_detail
from .cpu_kernel_dispatch import Phase1DKernelDispatch
from .speculative_placement import (
    RoutingMarkovChain,
    SpeculativePlacementEngine,
    markov_chain_path,
)
from .grammar_speculative_decoder import (
    GrammarSpeculativeDecoder,
    GrammarMode,
    PHASE1D_SPEC_DRAFT_LEN,
)

logger = logging.getLogger(__name__)

PHASE1D_STATE_FORMAT = "openagent.phase1d-session-state.v1"


# ---------------------------------------------------------------------------
# Phase1DOptimizedSession
# ---------------------------------------------------------------------------

class Phase1DOptimizedSession:
    """Wrap a Phase 1C session with Phase 1D capabilities.

    Args:
        phase1c_session: A ``Phase1COptimizedSession`` (or duck-typed equiv).
        enable_speculative_placement: Enable Markov-chain expert pre-placement.
        enable_speculative_decode: Enable draft-then-verify speculative decoding.
        draft_fn: Optional lightweight model callable for speculative decoding.
        draft_len: Number of draft tokens per speculation step.
        grammar_mode: Default grammar FSM mode for structured output.
        grammar_pattern: Regex pattern (used when grammar_mode == "regex").
        model_dir: Checkpoint directory (for direct I/O, passed through to 1C).
    """

    def __init__(
        self,
        phase1c_session: Any,
        *,
        enable_speculative_placement: bool = True,
        enable_speculative_decode: bool = False,
        draft_fn: Optional[Callable] = None,
        draft_len: int = PHASE1D_SPEC_DRAFT_LEN,
        grammar_mode: str = GrammarMode.FREE,
        grammar_pattern: str = "",
        model_dir: Optional[Path] = None,
    ) -> None:
        if phase1c_session is None:
            raise ValueError("phase1c_session is required")

        self.base_session = phase1c_session
        self._lock  = threading.RLock()
        self._operation_lock = threading.RLock()
        self._closed = False
        self._phase1d_saved = False

        # ---- Phase 1D sub-systems ----

        # 1. Numba-JIT kernel dispatch
        self.kernels = Phase1DKernelDispatch()

        # 2. Speculative expert pre-placement
        state_dir: Optional[Path] = getattr(phase1c_session, "state_directory", None)
        if state_dir is None:
            # Phase 1C wraps Phase 1B; try the base session's state_directory
            base1b = getattr(phase1c_session, "base_session", None)
            state_dir = getattr(base1b, "state_directory", None)

        manifest  = getattr(phase1c_session, "manifest", None)
        model_id  = (getattr(manifest, "model_identity", None) or
                     getattr(manifest, "model_id", None) or "unknown")

        self._markov_path: Optional[Path] = None
        markov = RoutingMarkovChain()
        if state_dir is not None and enable_speculative_placement:
            self._markov_path = markov_chain_path(state_dir, model_id)
            if self._markov_path.exists():
                markov = RoutingMarkovChain.load(self._markov_path)
                logger.info("phase1d: loaded Markov chain from %s", self._markov_path)

        # Bind the placement engine to the first available store
        stores = getattr(phase1c_session, "_stores", ())
        if not stores:
            base1b_session = getattr(phase1c_session, "base_session", None)
            stores = getattr(base1b_session, "_stores", ())
        store_ref = stores[0] if stores else None

        self._placement: Optional[SpeculativePlacementEngine] = None
        if enable_speculative_placement:
            self._placement = SpeculativePlacementEngine(
                chain=markov,
                store=store_ref,
            )

        # 3. Grammar-constrained speculative decoder (optional)
        self._spec_decoder: Optional[GrammarSpeculativeDecoder] = None
        if enable_speculative_decode:
            # We need a main_verify_fn from the base session; use a shim
            # that calls base_session.generate with a single-token batch.
            # Real wiring happens when the production session exposes
            # batch_logits() or similar; this shim is the hook point.
            main_verify_fn = getattr(phase1c_session, "batch_logits", None)
            vocab_size = getattr(phase1c_session, "vocab_size", 32000)
            id_to_text: Dict[int, str] = getattr(phase1c_session, "id_to_text", {})
            if callable(main_verify_fn):
                self._spec_decoder = GrammarSpeculativeDecoder(
                    main_verify_fn=main_verify_fn,
                    vocab_size=vocab_size,
                    id_to_text=id_to_text,
                    draft_fn=draft_fn,
                    draft_len=draft_len,
                )
                if grammar_mode != GrammarMode.FREE:
                    self._spec_decoder.set_grammar(grammar_mode, grammar_pattern)
                logger.info("phase1d: speculative decoder enabled (draft_len=%d, grammar=%s)",
                            draft_len, grammar_mode)
            else:
                raise ValueError(
                    "speculative decoding was requested but the base session does "
                    "not expose a callable batch_logits verifier"
                )

    # -- routing hook ----------------------------------------------------------

    def on_token_routed(self, layer_id: int, expert_ids: List[str]) -> None:
        """Call after each token's routing decision in layer layer_id.

        Updates the Markov chain and fires speculative placement hints.
        Also calls the Phase 1C on_layer_complete hook if available.
        """
        # Forward to Phase 1C hook
        p1c_hook = getattr(self.base_session, "on_layer_complete", None)
        if callable(p1c_hook):
            p1c_hook(layer_id, expert_ids)
        # Phase 1D placement
        if self._placement is not None:
            self._placement.on_token_routed(layer_id, expert_ids)

    # -- generation ------------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Phase 1D session is closed")

    def generate(self, *args: Any, cancelled: Callable[[], bool] | None = None,
                  **kwargs: Any) -> Any:
        with self._operation_lock:
            with self._lock:
                self._check_open()
            # Grammar-constrained speculative decode path
            if self._spec_decoder is not None:
                context_ids   = kwargs.pop("context_ids", args[0] if args else None)
                max_new_tokens = kwargs.pop("max_new_tokens", 128)
                stop_token_id  = kwargs.pop("stop_token_id", None)
                if context_ids is not None:
                    return self._spec_decoder.generate(
                        context_ids, max_new_tokens, stop_token_id, cancelled=cancelled)
            return self.base_session.generate(*args, cancelled=cancelled, **kwargs)

    def stream(self, *args: Any, cancelled: Callable[[], bool] | None = None,
                **kwargs: Any) -> Iterator[Any]:
        with self._operation_lock:
            with self._lock:
                self._check_open()
            if self._spec_decoder is not None:
                context_ids    = kwargs.pop("context_ids", args[0] if args else None)
                max_new_tokens  = kwargs.pop("max_new_tokens", 128)
                stop_token_id   = kwargs.pop("stop_token_id", None)
                if context_ids is not None:
                    yield from self._spec_decoder.stream(
                        context_ids, max_new_tokens, stop_token_id, cancelled=cancelled)
                    return
            yield from self.base_session.stream(*args, cancelled=cancelled, **kwargs)

    def set_grammar(self, mode: str, pattern: str = "") -> None:
        """Change grammar FSM mode for the speculative decoder."""
        with self._operation_lock:
            with self._lock:
                self._check_open()
            if self._spec_decoder is not None:
                self._spec_decoder.set_grammar(mode, pattern)
            else:
                logger.debug("phase1d: set_grammar called but no speculative decoder active")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.base_session, name)

    # -- telemetry -------------------------------------------------------------

    def metrics(self) -> dict:
        base: dict = {}
        get_m = getattr(self.base_session, "metrics", None)
        if callable(get_m):
            base = get_m()

        phase1d: dict = {**self.kernels.stats()}
        if self._placement is not None:
            phase1d.update(self._placement.metrics())
        if self._spec_decoder is not None:
            phase1d.update(self._spec_decoder.metrics())

        return {**base, "phase1d": phase1d}

    def status(self) -> dict:
        base: dict = {}
        get_s = getattr(self.base_session, "status", None)
        if callable(get_s):
            base = get_s()
        with self._lock:
            return {**base, "phase1d": {
                "closed": self._closed,
                "phase1d_saved": self._phase1d_saved,
                "markov_path": str(self._markov_path) if self._markov_path else None,
                "placement_enabled": self._placement is not None,
                "spec_decode_enabled": self._spec_decoder is not None,
            }}

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        """Save Phase 1D state, then close the Phase 1C session."""
        with self._operation_lock:
            self._close_once()

    def _close_once(self) -> None:
        with self._lock:
            if self._closed:
                return
            primary: BaseException | None = None

            def record(err: BaseException, label: str) -> None:
                nonlocal primary
                if primary is None:
                    primary = err
                else:
                    add_failure_note(primary, label + ": " + failure_detail(err))

            # Save Markov chain
            if not self._phase1d_saved and self._placement is not None:
                saved = True
                if self._markov_path is not None:
                    try:
                        self._placement.chain.save(self._markov_path)
                        logger.info("phase1d: Markov chain saved → %s", self._markov_path)
                    except BaseException as err:
                        saved = False
                        record(err, "phase1d markov chain save failure")
                self._phase1d_saved = saved

            # Close Phase 1C session
            if primary is None:
                try:
                    self.base_session.close()
                except BaseException as err:
                    record(err, "phase1c session close failure")

            if primary is not None:
                raise primary
            self._closed = True

    def __enter__(self) -> "Phase1DOptimizedSession":
        self._check_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def wrap_phase1c_with_phase1d(
    phase1c_session: Any,
    *,
    enable_speculative_placement: bool = True,
    enable_speculative_decode: bool = False,
    draft_fn: Optional[Callable] = None,
    draft_len: int = PHASE1D_SPEC_DRAFT_LEN,
    grammar_mode: str = GrammarMode.FREE,
    grammar_pattern: str = "",
    model_dir: Optional[Path] = None,
) -> Phase1DOptimizedSession:
    """Bind Phase 1D capabilities to an existing Phase 1C session.

    Args:
        phase1c_session: An admitted ``Phase1COptimizedSession``.
        enable_speculative_placement: Enable Markov pre-placement (default True).
        enable_speculative_decode: Enable draft-then-verify loop (default False;
            requires ``draft_fn`` or a ``batch_logits`` method on the session).
        draft_fn: Lightweight model for speculative decoding.
        draft_len: Draft depth K (default 4).
        grammar_mode: Grammar FSM mode for structured output.
        grammar_pattern: Regex pattern (``grammar_mode == "regex"``).
        model_dir: Checkpoint directory.

    Returns:
        A ``Phase1DOptimizedSession`` that enhances the Phase 1C session.
    """
    return Phase1DOptimizedSession(
        phase1c_session,
        enable_speculative_placement=enable_speculative_placement,
        enable_speculative_decode=enable_speculative_decode,
        draft_fn=draft_fn,
        draft_len=draft_len,
        grammar_mode=grammar_mode,
        grammar_pattern=grammar_pattern,
        model_dir=model_dir,
    )


__all__ = [
    "PHASE1D_STATE_FORMAT",
    "Phase1DOptimizedSession",
    "wrap_phase1c_with_phase1d",
]
