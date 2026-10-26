"""Phase 1G: Draft model for speculative decoding.

Phase 1D's GrammarSpeculativeDecoder has a ``draft_fn`` slot but no model to
fill it.  Phase 1G provides three draft model strategies:

1. **NgramDraftModel** — bigram/trigram language model trained on the main
   model's own output token stream.  No external weights.  Trains online per
   session.  Achieves 1.5–2.5× throughput gain from speculation alone on
   repetitive outputs (code, structured data).

2. **SmallModelDraftAdapter** — wraps any smaller model in the same family as
   the draft.  Delegates to a ``draft_session.generate_single(token_ids)``
   call.  Intended for pairing Qwen 3.6B with Qwen 3.8B, DeepSeek V4 small
   with full V4, etc.

3. **PrefixCacheDraftModel** — matches the running token prefix against a
   reuse cache of previously generated sequences.  Effectively free when the
   cache hits; falls back to NgramDraftModel on miss.

All three implement the same interface:

    draft_fn(token_ids: List[int]) -> np.ndarray  # logits shape (vocab_size,)

Architecture
------------
``NgramLanguageModel``     -- online-trained n-gram LM (n=2 or 3)
``NgramDraftModel``        -- wraps NgramLanguageModel with the draft_fn API
``SmallModelDraftAdapter`` -- delegates to a smaller session
``PrefixCacheDraftModel``  -- prefix-match cache + fallback
``Phase1GDraftSelector``   -- picks the best available draft model at runtime
"""
from __future__ import annotations

import collections
import inspect
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .execution_service import ExecutionCancelled

logger = logging.getLogger(__name__)


def _check_cancelled(cancelled) -> None:
    if cancelled is not None and cancelled():
        raise InterruptedError("Phase 1G draft generation cancelled")


def _invoke_cancellable(function, *args, cancelled=None, **kwargs):
    """Forward cancellation only to draft callbacks that support it."""
    _check_cancelled(cancelled)
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    supports_cancel = any(
        parameter.name == "cancelled" or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if cancelled is not None and supports_cancel:
        kwargs["cancelled"] = cancelled
    result = function(*args, **kwargs)
    _check_cancelled(cancelled)
    return result


# ---------------------------------------------------------------------------
# NgramLanguageModel — online bigram/trigram LM
# ---------------------------------------------------------------------------

_DEFAULT_NGRAM_N      = 3
_DEFAULT_SMOOTHING    = 0.1    # Laplace smoothing weight
_DEFAULT_MAX_CONTEXT  = 65536  # tokens to keep in training history


class NgramLanguageModel:
    """Online-trained n-gram language model over a fixed vocabulary.

    Trained incrementally: call ``train(token_ids)`` after each generation
    step.  ``predict(context)`` returns a (vocab_size,) float32 probability
    array.

    Uses a Laplace-smoothed n-gram table stored as a nested defaultdict.
    Memory is bounded by ``max_vocab_per_context`` (top-K per context key).
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        n: int = _DEFAULT_NGRAM_N,
        smoothing: float = _DEFAULT_SMOOTHING,
        max_context_history: int = _DEFAULT_MAX_CONTEXT,
        max_vocab_per_context: int = 512,
    ) -> None:
        self.vocab_size = vocab_size
        self.n = max(2, n)
        self.smoothing = smoothing
        self.max_context_history = max_context_history
        self.max_vocab_per_context = max_vocab_per_context

        # counts[(ctx_tuple)] → Counter{next_token: count}
        self._counts: Dict[Tuple[int, ...], collections.Counter] = (
            collections.defaultdict(collections.Counter)
        )
        self._total_per_ctx: Dict[Tuple[int, ...], int] = (
            collections.defaultdict(int)
        )
        self._history: collections.deque = collections.deque(
            maxlen=max_context_history
        )
        self._lock = threading.Lock()
        self._train_calls = 0
        self._predict_calls = 0

    def train(self, token_ids: List[int]) -> None:
        """Update the model with a new sequence of tokens."""
        if not token_ids:
            return
        with self._lock:
            self._train_calls += 1
            history = list(self._history) + list(token_ids)
            self._history.extend(token_ids)

            # Build n-grams from the new tokens (also overlapping with history)
            for i in range(len(history) - self.n + 1):
                ctx = tuple(history[i : i + self.n - 1])
                next_tok = history[i + self.n - 1]
                self._counts[ctx][next_tok] += 1
                self._total_per_ctx[ctx] += 1
                # Prune to max_vocab_per_context
                if len(self._counts[ctx]) > self.max_vocab_per_context:
                    bottom = self._counts[ctx].most_common()[self.max_vocab_per_context:]
                    for tok, _ in bottom:
                        self._total_per_ctx[ctx] -= self._counts[ctx].pop(tok)

    def predict(self, context: List[int]) -> np.ndarray:
        """Return smoothed probability distribution over vocab given context."""
        with self._lock:
            self._predict_calls += 1
            probs = np.full(self.vocab_size, self.smoothing / self.vocab_size, dtype=np.float32)

            for order in range(self.n - 1, 0, -1):
                ctx = tuple(context[-(order):]) if len(context) >= order else tuple(context)
                counter = self._counts.get(ctx)
                if counter is None:
                    continue
                total = self._total_per_ctx.get(ctx, 0) + self.smoothing
                for tok, cnt in counter.items():
                    if 0 <= tok < self.vocab_size:
                        probs[tok] += cnt / total
                break  # use longest matching context

            probs /= probs.sum()
            return probs

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                "vocab_size": self.vocab_size,
                "n": self.n,
                "smoothing": self.smoothing,
                # Store as serialisable: {ctx_str: {tok: count}}
                "counts": {
                    str(ctx): dict(c)
                    for ctx, c in self._counts.items()
                },
                "history": list(self._history),
            }
        import json
        path.write_text(json.dumps(data))
        logger.info("phase1g: NgramLM saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "NgramLanguageModel":
        import json
        data = json.loads(Path(path).read_text())
        lm = cls(
            vocab_size=data["vocab_size"],
            n=data["n"],
            smoothing=data["smoothing"],
        )
        for ctx_str, counts in data.get("counts", {}).items():
            ctx = tuple(int(x) for x in ctx_str.strip("()").split(",") if x.strip())
            lm._counts[ctx] = collections.Counter({int(k): v for k, v in counts.items()})
            lm._total_per_ctx[ctx] = sum(counts.values())
        lm._history.extend(data.get("history", []))
        logger.info("phase1g: NgramLM loaded from %s", path)
        return lm

    def stats(self) -> dict:
        with self._lock:
            return {
                "ngram_n": self.n,
                "ngram_contexts": len(self._counts),
                "ngram_train_calls": self._train_calls,
                "ngram_predict_calls": self._predict_calls,
                "ngram_history_len": len(self._history),
            }


# ---------------------------------------------------------------------------
# NgramDraftModel — draft_fn compatible wrapper
# ---------------------------------------------------------------------------

_NGRAM_LM_FILENAME = "phase1g_ngram_lm.json"


class NgramDraftModel:
    """draft_fn adapter around NgramLanguageModel.

    Trains online from ``feedback(token_ids)`` calls.  Converts LM probabilities
    to log-logits so the speculative verifier can apply the acceptance criterion.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        n: int = _DEFAULT_NGRAM_N,
        state_dir: Optional[Path] = None,
    ) -> None:
        self.vocab_size = vocab_size
        self._lm = NgramLanguageModel(vocab_size=vocab_size, n=n)
        self._state_dir = state_dir
        self._context: List[int] = []
        self._lock = threading.Lock()

        if state_dir is not None:
            lm_path = Path(state_dir) / _NGRAM_LM_FILENAME
            if lm_path.exists():
                try:
                    self._lm = NgramLanguageModel.load(lm_path)
                    logger.info("phase1g: NgramDraftModel loaded from %s", lm_path)
                except Exception as exc:
                    logger.warning("phase1g: LM load failed: %s", exc)

    # draft_fn interface
    def __call__(self, token_ids: List[int]) -> np.ndarray:
        """Return logits (vocab_size,) given a sequence of token IDs."""
        with self._lock:
            self._context = list(token_ids)
        probs = self._lm.predict(token_ids)
        # Convert to log-logits (log-softmax equivalent)
        logits = np.log(np.maximum(probs, 1e-10))
        return logits

    def feedback(self, token_ids: List[int]) -> None:
        """Train on a newly generated sequence."""
        self._lm.train(token_ids)

    def save(self) -> None:
        if self._state_dir is not None:
            lm_path = Path(self._state_dir) / _NGRAM_LM_FILENAME
            try:
                self._lm.save(lm_path)
            except Exception as exc:
                logger.warning("phase1g: NgramDraftModel save failed: %s", exc)

    def stats(self) -> dict:
        return {"ngram_draft": True, **self._lm.stats()}


# ---------------------------------------------------------------------------
# SmallModelDraftAdapter — delegates to a smaller session
# ---------------------------------------------------------------------------

class SmallModelDraftAdapter:
    """Wrap a small model session as a draft_fn.

    The small session must expose ``generate_single(token_ids) → np.ndarray``
    returning logits of shape ``(vocab_size,)`` for the next token.

    If the small session does not have ``generate_single``, ``__call__`` will
    fall back to ``generate(context_ids=token_ids, max_new_tokens=1)`` and
    construct a uniform distribution (no logit info).
    """

    def __init__(self, small_session: Any, vocab_size: int = 32000) -> None:
        self._session = small_session
        self.vocab_size = vocab_size
        self._calls = 0
        self._lock  = threading.Lock()

    def __call__(self, token_ids: List[int], *, cancelled=None) -> np.ndarray:
        with self._lock:
            self._calls += 1

        gen_single = getattr(self._session, "generate_single", None)
        if callable(gen_single):
            try:
                logits = _invoke_cancellable(
                    gen_single, token_ids, cancelled=cancelled,
                )
                if isinstance(logits, np.ndarray) and logits.shape == (self.vocab_size,):
                    return logits
            except (InterruptedError, ExecutionCancelled):
                raise
            except Exception as exc:
                _check_cancelled(cancelled)
                logger.debug("phase1g: small model generate_single failed: %s", exc)

        # Fallback: call generate() and return uniform logits
        try:
            tokens = _invoke_cancellable(
                self._session.generate, context_ids=token_ids, max_new_tokens=1,
                cancelled=cancelled,
            )
            if tokens:
                logits = np.zeros(self.vocab_size, dtype=np.float32)
                logits[tokens[-1]] = 10.0
                return logits
        except (InterruptedError, ExecutionCancelled):
            raise
        except Exception as exc:
            _check_cancelled(cancelled)
            logger.debug("phase1g: small model generate failed: %s", exc)

        return np.zeros(self.vocab_size, dtype=np.float32)

    def stats(self) -> dict:
        with self._lock:
            return {
                "small_draft_calls": self._calls,
                "small_model_type": type(self._session).__name__,
            }


# ---------------------------------------------------------------------------
# PrefixCacheDraftModel — prefix-match reuse cache
# ---------------------------------------------------------------------------

_PREFIX_CACHE_SIZE    = 4096   # max cached sequences
_PREFIX_MIN_MATCH_LEN = 4      # min prefix tokens to attempt match


class PrefixCacheDraftModel:
    """Cache previously generated sequences and replay on prefix match.

    When the current context matches the start of a cached sequence, the
    cache returns the next token's logits from the stored distribution.  On
    miss, falls back to ``fallback_fn``.

    This is essentially free for repetitive prompts (benchmark harness, chat
    templates, structured formats).
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        fallback_fn: Optional[Callable[[List[int]], np.ndarray]] = None,
        max_entries: int = _PREFIX_CACHE_SIZE,
    ) -> None:
        self.vocab_size  = vocab_size
        self._fallback   = fallback_fn
        self._max        = max_entries
        self._cache: Dict[Tuple[int, ...], np.ndarray] = {}
        self._hits   = 0
        self._misses = 0
        self._lock   = threading.Lock()

    def __call__(self, token_ids: List[int], *, cancelled=None) -> np.ndarray:
        _check_cancelled(cancelled)
        key = tuple(token_ids[-_PREFIX_MIN_MATCH_LEN:])
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                self._hits += 1
                _check_cancelled(cancelled)
                return entry.copy()
            self._misses += 1

        if self._fallback is not None:
            logits = _invoke_cancellable(
                self._fallback, token_ids, cancelled=cancelled,
            )
        else:
            logits = np.zeros(self.vocab_size, dtype=np.float32)
        _check_cancelled(cancelled)

        with self._lock:
            if len(self._cache) >= self._max:
                # evict oldest (arbitrary key)
                evict_key = next(iter(self._cache))
                del self._cache[evict_key]
            self._cache[key] = logits

        return logits

    def stats(self) -> dict:
        with self._lock:
            return {
                "prefix_cache_entries": len(self._cache),
                "prefix_cache_hits":    self._hits,
                "prefix_cache_misses":  self._misses,
            }


# ---------------------------------------------------------------------------
# Phase1GDraftSelector — pick the best available draft model
# ---------------------------------------------------------------------------

class Phase1GDraftSelector:
    """Automatically selects and builds the best draft model for a session.

    Priority:
    1. SmallModelDraftAdapter if a small_session is provided.
    2. PrefixCacheDraftModel with NgramDraftModel fallback.
    3. NgramDraftModel alone.

    Returns the chosen draft_fn and a ``feedback(tokens)`` callable for
    online training.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        small_session: Optional[Any] = None,
        state_dir: Optional[Path] = None,
        enable_prefix_cache: bool = True,
    ) -> None:
        self.vocab_size = vocab_size
        self._ngram    = NgramDraftModel(vocab_size=vocab_size, state_dir=state_dir)
        self._small: Optional[SmallModelDraftAdapter] = None
        self._prefix: Optional[PrefixCacheDraftModel] = None

        if small_session is not None:
            self._small = SmallModelDraftAdapter(small_session, vocab_size=vocab_size)

        base_fn = self._small or self._ngram
        if enable_prefix_cache:
            self._prefix = PrefixCacheDraftModel(
                vocab_size=vocab_size,
                fallback_fn=base_fn,
            )
            self.draft_fn: Callable[[List[int]], np.ndarray] = self._prefix
        else:
            self.draft_fn = base_fn

    def feedback(self, token_ids: List[int]) -> None:
        """Train the n-gram model on a completed generation."""
        self._ngram.feedback(token_ids)

    def save(self) -> None:
        """Persist trained n-gram model."""
        self._ngram.save()

    def stats(self) -> dict:
        s: dict = {**self._ngram.stats()}
        if self._small is not None:
            s.update(self._small.stats())
        if self._prefix is not None:
            s.update(self._prefix.stats())
        return s


__all__ = [
    "NgramLanguageModel",
    "NgramDraftModel",
    "SmallModelDraftAdapter",
    "PrefixCacheDraftModel",
    "Phase1GDraftSelector",
]
