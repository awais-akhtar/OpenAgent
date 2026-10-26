"""Phase 1G integration bridge: draft model for speculative decoding.

Wraps a ``Phase1EOptimizedSession`` (or Phase 1D) with a trained draft model
that activates Phase 1D's ``GrammarSpeculativeDecoder``.

Session stack after Phase 1G::

    base_session
      └── Phase1BOptimizedSession
            └── Phase1COptimizedSession
                  └── Phase1DOptimizedSession
                        └── Phase1EOptimizedSession
                              └── Phase1GOptimizedSession   ← this module

Phase 1G responsibilities:

1. Build a ``Phase1GDraftSelector`` (n-gram + optional small model + prefix cache).
2. Wire ``draft_fn`` into Phase 1D's ``GrammarSpeculativeDecoder`` at open time.
3. Call ``feedback(tokens)`` after each generation to train the n-gram LM online.
4. Save the n-gram model state on ``close()``.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional

import logging

from .failure_reporting import add_failure_note, failure_detail
from .draft_model import Phase1GDraftSelector

logger = logging.getLogger(__name__)

PHASE1G_STATE_FORMAT = "openagent.phase1g-session-state.v1"


class Phase1GOptimizedSession:
    """Wrap a Phase 1E session with a trained draft model for spec-decode.

    Args:
        phase1e_session: Phase 1E (or Phase 1D) session.
        vocab_size: Vocabulary size for the draft model.
        small_session: Optional smaller model session for SmallModelDraftAdapter.
        enable_prefix_cache: Enable prefix-match cache on top of n-gram LM.
        enable_speculative_decode: Wire draft_fn and enable spec-decode in Phase 1D.
        state_dir: Override state directory for n-gram LM persistence.
    """

    def __init__(
        self,
        phase1e_session: Any,
        *,
        vocab_size: int = 32000,
        small_session: Optional[Any] = None,
        enable_prefix_cache: bool = True,
        enable_speculative_decode: bool = True,
        state_dir: Optional[Path] = None,
    ) -> None:
        if phase1e_session is None:
            raise ValueError("phase1e_session is required")

        self.base_session = phase1e_session
        self._lock   = threading.RLock()
        self._operation_lock = threading.RLock()
        self._closed = False
        self._g_saved = False

        # Resolve state directory from session stack if not given
        if state_dir is None:
            state_dir = getattr(phase1e_session, "state_directory", None)
            if state_dir is None:
                base1d = getattr(phase1e_session, "base_session", None)
                state_dir = getattr(base1d, "state_directory", None)
            if state_dir is None:
                base1b = getattr(
                    getattr(phase1e_session, "base_session", None),
                    "base_session", None)
                if base1b is not None:
                    base1b = getattr(base1b, "base_session", None)
                state_dir = getattr(base1b, "state_directory", None) if base1b else None

        self._draft_selector = Phase1GDraftSelector(
            vocab_size=vocab_size,
            small_session=small_session,
            state_dir=Path(state_dir) if state_dir else None,
            enable_prefix_cache=enable_prefix_cache,
        )

        # Wire draft_fn into Phase 1D's spec-decoder if it exists
        if enable_speculative_decode:
            self._wire_draft_fn()

    def _wire_draft_fn(self) -> None:
        """Find Phase 1D's spec-decoder through the session stack and wire draft_fn."""
        # Walk up the stack looking for _spec_decoder
        session = self.base_session
        for _ in range(8):  # max stack depth
            spec_dec = getattr(session, "_spec_decoder", None)
            if spec_dec is not None:
                if spec_dec.vocab_size != self._draft_selector.vocab_size:
                    raise ValueError(
                        "Phase 1G draft vocabulary does not match the target decoder"
                    )
                draft_fn = self._draft_selector.draft_fn
                if not callable(draft_fn):
                    raise ValueError("Phase 1G did not produce a callable draft model")
                if spec_dec.draft_fn is not None:
                    raise ValueError("Phase 1D decoder already has a draft model")
                spec_dec.draft_fn = draft_fn
                logger.info("phase1g: draft_fn wired into %s._spec_decoder",
                            type(session).__name__)
                return
            session = getattr(session, "base_session", None)
            if session is None:
                break
        raise ValueError(
            "Phase 1G speculative decoding requires an enabled Phase 1D decoder "
            "with a callable batch_logits verifier"
        )

    # -- routing hook (forward) ------------------------------------------------

    def on_token_routed(self, layer_id: int, expert_ids: List[str]) -> None:
        hook = getattr(self.base_session, "on_token_routed", None)
        if callable(hook):
            hook(layer_id, expert_ids)

    # -- generation ------------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Phase 1G session is closed")

    def generate(self, *args: Any, cancelled: Callable[[], bool] | None = None,
                  **kwargs: Any) -> Any:
        with self._operation_lock:
            with self._lock:
                self._check_open()
            result = self.base_session.generate(*args, cancelled=cancelled, **kwargs)
            # Train n-gram LM on the generated tokens
            if isinstance(result, (list, tuple)) and result:
                self._draft_selector.feedback(list(result))
            return result

    def stream(self, *args: Any, cancelled: Callable[[], bool] | None = None,
                **kwargs: Any) -> Iterator[Any]:
        with self._operation_lock:
            with self._lock:
                self._check_open()
            collected: List[Any] = []
            for tok in self.base_session.stream(*args, cancelled=cancelled, **kwargs):
                collected.append(tok)
                yield tok
            if collected:
                self._draft_selector.feedback([t for t in collected if isinstance(t, int)])

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
        return {**base, "phase1g": self._draft_selector.stats()}

    def status(self) -> dict:
        base: dict = {}
        get_s = getattr(self.base_session, "status", None)
        if callable(get_s):
            base = get_s()
        with self._lock:
            return {**base, "phase1g": {"closed": self._closed, "g_saved": self._g_saved}}

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
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

            if not self._g_saved:
                try:
                    self._draft_selector.save()
                    self._g_saved = True
                except BaseException as err:
                    record(err, "phase1g draft model save failure")

            try:
                self.base_session.close()
            except BaseException as err:
                record(err, "phase1e session close failure")

            if primary is not None:
                raise primary
            self._closed = True

    def __enter__(self) -> "Phase1GOptimizedSession":
        self._check_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def wrap_phase1e_with_phase1g(
    phase1e_session: Any,
    *,
    vocab_size: int = 32000,
    small_session: Optional[Any] = None,
    enable_prefix_cache: bool = True,
    enable_speculative_decode: bool = True,
    state_dir: Optional[Path] = None,
) -> Phase1GOptimizedSession:
    """Bind Phase 1G draft model to an existing Phase 1E session."""
    return Phase1GOptimizedSession(
        phase1e_session,
        vocab_size=vocab_size,
        small_session=small_session,
        enable_prefix_cache=enable_prefix_cache,
        enable_speculative_decode=enable_speculative_decode,
        state_dir=state_dir,
    )


__all__ = [
    "PHASE1G_STATE_FORMAT",
    "Phase1GOptimizedSession",
    "wrap_phase1e_with_phase1g",
]
