"""Phase 1D: Grammar-Constrained Speculative Decoding.

Speculative decoding runs a lightweight "draft" model to generate K candidate
tokens in one pass, then verifies all K tokens against the main model in a
single forward pass.  Accepted tokens are committed; the first rejected token
triggers a correction and the draft restarts.

Grammar-constrained speculative decoding adds a finite-state constraint on top:
a ``GrammarFSM`` tracks the set of legal next tokens at each generation step,
masking the draft and verifier logits to prevent structurally invalid outputs.
This is particularly valuable for:

  - JSON-mode responses (strict RFC 8259 syntax)
  - Function-call argument marshalling (OpenAgent tool use)
  - Regex-bounded completions (e.g. phone numbers, identifiers)

Architecture
------------
``GrammarFSM``
    Pushdown FSM that tracks parser state for a restricted grammar.
    Supports: JSON (nested objects, arrays, strings, numbers), regex
    (via a NFA compiled from the pattern), and a pass-through "free" mode.

``TokenMask``
    Applies a vocabulary mask derived from the FSM state to a logit tensor.

``SpeculativeDraftBuffer``
    Holds K draft tokens + log-probs from the lightweight draft model.

``SpeculativeVerifier``
    Compares draft log-probs against main-model log-probs using the
    standard speculative sampling acceptance criterion
    (Leviathan et al., 2023: accept token t with prob min(1, p_main/p_draft)).

``GrammarSpeculativeDecoder``
    Orchestrates the draft → mask → verify → accept/reject loop.
    Exposed as a generate() wrapper the Phase 1D adapter calls.

Draft model
-----------
The draft model is any callable ``(token_ids) → logits`` that is faster than
the main model.  In OpenAgent this is typically:
  - A small distilled version of the same model family.
  - A speculative prefix match against a prompt cache.
  - A bigram/trigram LM trained on the model's own output (no external weights).

When no draft model is available, ``GrammarSpeculativeDecoder`` degrades to
grammar-constrained greedy decoding (mask only, no speculative speedup).

This module starts no threads and performs no I/O.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import inspect
import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise InterruptedError("grammar-constrained generation cancelled")


def _call_cancellable(function, *args, cancelled=None, rng=None):
    """Pass cooperative controls only when a callback advertises them."""
    _check_cancelled(cancelled)
    options = {}
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        parameters = {}
    keyword_parameters = {
        parameter.name
        for parameter in parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    accepts_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if cancelled is not None and (
        "cancelled" in keyword_parameters or accepts_keywords
    ):
        options["cancelled"] = cancelled
    if rng is not None and ("rng" in keyword_parameters or accepts_keywords):
        options["rng"] = rng
    result = function(*args, **options)
    _check_cancelled(cancelled)
    return result


def _is_cancellation_error(error: BaseException) -> bool:
    if isinstance(error, (InterruptedError, KeyboardInterrupt)):
        return True
    try:
        from .execution_service import ExecutionCancelled
    except ImportError:
        return False
    return isinstance(error, ExecutionCancelled)


# Default draft length (tokens to speculate ahead)
PHASE1D_SPEC_DRAFT_LEN = 4
# Maximum grammar-masked candidates (after masking) before falling back
PHASE1D_SPEC_MIN_VALID_TOKENS = 1


# ---------------------------------------------------------------------------
# Grammar FSM
# ---------------------------------------------------------------------------


class GrammarMode:
    FREE = "free"
    JSON = "json"
    REGEX = "regex"


@dataclass
class FSMState:
    mode: str = GrammarMode.FREE
    # JSON parser stack; entries are object/array expectation states.
    json_stack: List[str] = field(default_factory=list)
    json_started: bool = False
    json_root_done: bool = False
    json_lexeme: Optional[str] = None
    json_string_role: Optional[str] = None
    json_str_escape: bool = False
    json_unicode_remaining: int = 0
    json_number_state: Optional[str] = None
    json_literal_text: str = ""
    json_literal_index: int = 0
    # Regex NFA current positions (set of NFA node IDs)
    regex_nfa_pos: FrozenSet[int] = frozenset()
    # Generated so far (for regex match prefix tracking)
    regex_generated: str = ""


class GrammarFSM:
    """Finite-state constraint over generation tokens.

    Each ``step(token_text)`` advances the FSM and returns True if the token
    is structurally valid.  ``valid_starts()`` returns a set of character
    classes (single chars, or regex patterns) that are valid at the current
    position; the caller uses this to build a vocabulary mask.

    JSON mode
    ---------
    Tracks a simple stack-based JSON state machine:
      STATES: "value", "object_key", "object_sep", "object_val",
              "array_val", "array_sep", "string", "number", "done"

    Regex mode
    ----------
    Compiles the pattern to a minimal NFA and tracks the active NFA state
    set as tokens are generated.

    Free mode
    ---------
    All tokens accepted; ``valid_starts()`` returns None (no mask).
    """

    # JSON parser transitions and lexical classes.
    _JSON_WS = frozenset(" \t\r\n")
    _JSON_HEX = frozenset("0123456789abcdefABCDEF")

    def __init__(self, mode: str = GrammarMode.FREE, pattern: str = "") -> None:
        if mode not in (GrammarMode.FREE, GrammarMode.JSON, GrammarMode.REGEX):
            raise ValueError(f"unsupported grammar mode: {mode!r}")
        if mode == GrammarMode.REGEX and type(pattern) is not str:
            raise ValueError("regex grammar pattern must be text")
        self.mode = mode
        self._pattern = pattern
        self._lock = threading.Lock()
        self._state = FSMState(mode=mode)
        self._steps = 0
        self._rejects = 0

        if mode == GrammarMode.REGEX:
            try:
                self._regex = re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid regex grammar pattern: {exc}") from exc
        else:
            self._regex = None

    def reset(self) -> None:
        with self._lock:
            self._state = FSMState(mode=self.mode)
            self._steps = 0
            self._rejects = 0

    def clone(self) -> "GrammarFSM":
        """Return an independent copy of the current grammar boundary.

        Draft generation and vocabulary probing are speculative: neither may
        advance the target FSM until the corresponding token is accepted.
        """
        with self._lock:
            clone = object.__new__(GrammarFSM)
            clone.mode = self.mode
            clone._pattern = self._pattern
            clone._regex = self._regex
            clone._lock = threading.Lock()
            clone._state = FSMState(
                mode=self._state.mode,
                json_stack=list(self._state.json_stack),
                json_started=self._state.json_started,
                json_root_done=self._state.json_root_done,
                json_lexeme=self._state.json_lexeme,
                json_string_role=self._state.json_string_role,
                json_str_escape=self._state.json_str_escape,
                json_unicode_remaining=self._state.json_unicode_remaining,
                json_number_state=self._state.json_number_state,
                json_literal_text=self._state.json_literal_text,
                json_literal_index=self._state.json_literal_index,
                regex_nfa_pos=frozenset(self._state.regex_nfa_pos),
                regex_generated=self._state.regex_generated,
            )
            clone._steps = self._steps
            clone._rejects = self._rejects
            return clone

    def preview_step(self, token_text: str) -> bool:
        """Check a transition without changing this FSM or its counters."""
        return self.clone().step(token_text)

    def _commit_from(self, other: "GrammarFSM") -> None:
        committed = other.clone()
        with self._lock:
            self.mode = committed.mode
            self._pattern = committed._pattern
            self._regex = committed._regex
            self._state = committed._state
            self._steps = committed._steps
            self._rejects = committed._rejects

    def step(self, token_text: str) -> bool:
        """Advance the FSM by one token.  Returns True if the token is valid."""
        if type(token_text) is not str:
            raise TypeError("grammar token text must be a string")
        with self._lock:
            self._steps += 1
            if self.mode == GrammarMode.FREE:
                return True
            original = self._state
            self._state = self._clone_state(original)
            if self.mode == GrammarMode.JSON:
                ok = self._json_step(token_text)
            elif self.mode == GrammarMode.REGEX:
                ok = self._regex_step(token_text)
            else:
                ok = True
            if not ok:
                self._state = original
                self._rejects += 1
            return ok

    @staticmethod
    def _clone_state(state: FSMState) -> FSMState:
        return FSMState(
            mode=state.mode,
            json_stack=list(state.json_stack),
            json_started=state.json_started,
            json_root_done=state.json_root_done,
            json_lexeme=state.json_lexeme,
            json_string_role=state.json_string_role,
            json_str_escape=state.json_str_escape,
            json_unicode_remaining=state.json_unicode_remaining,
            json_number_state=state.json_number_state,
            json_literal_text=state.json_literal_text,
            json_literal_index=state.json_literal_index,
            regex_nfa_pos=frozenset(state.regex_nfa_pos),
            regex_generated=state.regex_generated,
        )

    def _json_start_value(self, ch: str) -> bool:
        state = self._state
        state.json_started = True
        if ch == '"':
            state.json_lexeme = "string"
            state.json_string_role = "value"
        elif ch == "{":
            state.json_stack.append("object_key_or_end")
        elif ch == "[":
            state.json_stack.append("array_value_or_end")
        elif ch in ("t", "f", "n"):
            state.json_lexeme = "literal"
            state.json_literal_text = {"t": "true", "f": "false", "n": "null"}[ch]
            state.json_literal_index = 1
        elif ch == "-":
            state.json_lexeme = "number"
            state.json_number_state = "minus"
        elif ch == "0":
            state.json_lexeme = "number"
            state.json_number_state = "zero"
        elif ch in "123456789":
            state.json_lexeme = "number"
            state.json_number_state = "integer"
        else:
            state.json_started = False
            return False
        return True

    def _json_complete_value(self) -> bool:
        state = self._state
        if not state.json_stack:
            state.json_root_done = True
            return True
        top = state.json_stack[-1]
        if top == "object_value":
            state.json_stack[-1] = "object_comma_or_end"
        elif top in ("array_value", "array_value_or_end"):
            state.json_stack[-1] = "array_comma_or_end"
        else:
            return False
        return True

    def _json_close_container(self) -> bool:
        self._state.json_stack.pop()
        return self._json_complete_value()

    def _json_number_char(self, ch: str) -> Optional[bool]:
        """Consume a number char; None signals a complete number delimiter."""
        state = self._state
        number_state = state.json_number_state
        if number_state == "minus":
            if ch == "0":
                state.json_number_state = "zero"
            elif ch in "123456789":
                state.json_number_state = "integer"
            else:
                return False
        elif number_state == "zero":
            if ch == ".":
                state.json_number_state = "fraction_start"
            elif ch in "eE":
                state.json_number_state = "exponent_start"
            elif ch in self._JSON_WS or ch in ",]}":
                return None
            else:
                return False
        elif number_state == "integer":
            if ch.isascii() and ch.isdigit():
                pass
            elif ch == ".":
                state.json_number_state = "fraction_start"
            elif ch in "eE":
                state.json_number_state = "exponent_start"
            elif ch in self._JSON_WS or ch in ",]}":
                return None
            else:
                return False
        elif number_state == "fraction_start":
            if ch.isascii() and ch.isdigit():
                state.json_number_state = "fraction"
            else:
                return False
        elif number_state == "fraction":
            if ch.isascii() and ch.isdigit():
                pass
            elif ch in "eE":
                state.json_number_state = "exponent_start"
            elif ch in self._JSON_WS or ch in ",]}":
                return None
            else:
                return False
        elif number_state == "exponent_start":
            if ch in "+-":
                state.json_number_state = "exponent_sign"
            elif ch.isascii() and ch.isdigit():
                state.json_number_state = "exponent"
            else:
                return False
        elif number_state == "exponent_sign":
            if ch.isascii() and ch.isdigit():
                state.json_number_state = "exponent"
            else:
                return False
        elif number_state == "exponent":
            if ch.isascii() and ch.isdigit():
                pass
            elif ch in self._JSON_WS or ch in ",]}":
                return None
            else:
                return False
        else:
            return False
        return True

    def _json_step(self, text: str) -> bool:
        """Incrementally validate JSON syntax across arbitrary token pieces."""
        state = self._state
        offset = 0
        while offset < len(text):
            ch = text[offset]
            if state.json_lexeme == "string":
                if state.json_unicode_remaining:
                    if ch not in self._JSON_HEX:
                        return False
                    state.json_unicode_remaining -= 1
                elif state.json_str_escape:
                    state.json_str_escape = False
                    if ch == "u":
                        state.json_unicode_remaining = 4
                    elif ch not in '"\\/bfnrt':
                        return False
                elif ch == "\\":
                    state.json_str_escape = True
                elif ch == '"':
                    state.json_lexeme = None
                    if state.json_string_role == "key":
                        if not state.json_stack or state.json_stack[-1] not in (
                            "object_key",
                            "object_key_or_end",
                        ):
                            return False
                        state.json_stack[-1] = "object_colon"
                    else:
                        if not self._json_complete_value():
                            return False
                    state.json_string_role = None
                elif ord(ch) < 0x20:
                    return False
                offset += 1
                continue

            if state.json_lexeme == "literal":
                if ch != state.json_literal_text[state.json_literal_index]:
                    return False
                state.json_literal_index += 1
                if state.json_literal_index == len(state.json_literal_text):
                    state.json_lexeme = None
                    state.json_literal_text = ""
                    state.json_literal_index = 0
                    if not self._json_complete_value():
                        return False
                offset += 1
                continue

            if state.json_lexeme == "number":
                consumed = self._json_number_char(ch)
                if consumed is False:
                    return False
                if consumed is True:
                    offset += 1
                    continue
                state.json_lexeme = None
                state.json_number_state = None
                if not self._json_complete_value():
                    return False
                continue  # reprocess the structural delimiter

            if ch in self._JSON_WS:
                offset += 1
                continue
            if state.json_root_done:
                return False
            if not state.json_stack:
                if state.json_started or not self._json_start_value(ch):
                    return False
                offset += 1
                continue

            top = state.json_stack[-1]
            if top in ("object_key_or_end", "object_key"):
                if ch == "}" and top == "object_key_or_end":
                    if not self._json_close_container():
                        return False
                elif ch == '"':
                    state.json_lexeme = "string"
                    state.json_string_role = "key"
                else:
                    return False
            elif top == "object_colon":
                if ch != ":":
                    return False
                state.json_stack[-1] = "object_value"
            elif top == "object_value":
                if not self._json_start_value(ch):
                    return False
            elif top == "object_comma_or_end":
                if ch == ",":
                    state.json_stack[-1] = "object_key"
                elif ch == "}":
                    if not self._json_close_container():
                        return False
                else:
                    return False
            elif top in ("array_value_or_end", "array_value"):
                if ch == "]" and top == "array_value_or_end":
                    if not self._json_close_container():
                        return False
                elif not self._json_start_value(ch):
                    return False
            elif top == "array_comma_or_end":
                if ch == ",":
                    state.json_stack[-1] = "array_value"
                elif ch == "]":
                    if not self._json_close_container():
                        return False
                else:
                    return False
            else:
                return False
            offset += 1
        return True

    def _regex_step(self, text: str) -> bool:
        """Check that appending text keeps the generated string a valid prefix."""
        s = self._state
        candidate = s.regex_generated + text
        # Accept if candidate is either a full match or a valid prefix
        pattern = self._regex
        if pattern is None:
            return True
        # Full match
        if pattern.fullmatch(candidate):
            s.regex_generated = candidate
            return True
        # Prefix match: does the pattern match a string that starts with candidate?
        prefix_ok = bool(pattern.match(candidate))
        if prefix_ok:
            s.regex_generated = candidate
        return prefix_ok

    def is_complete(self) -> bool:
        """Return True if the current state is a valid terminal state."""
        with self._lock:
            if self.mode == GrammarMode.JSON:
                state = self._state
                if (
                    state.json_root_done
                    and not state.json_stack
                    and state.json_lexeme is None
                ):
                    return True
                return (
                    state.json_started
                    and not state.json_stack
                    and state.json_lexeme == "number"
                    and state.json_number_state
                    in {"zero", "integer", "fraction", "exponent"}
                )
            if self.mode == GrammarMode.REGEX and self._regex is not None:
                return bool(self._regex.fullmatch(self._state.regex_generated))
            return True

    def stats(self) -> dict:
        with self._lock:
            return {
                "fsm_mode": self.mode,
                "fsm_steps": self._steps,
                "fsm_rejects": self._rejects,
            }


# ---------------------------------------------------------------------------
# TokenMask
# ---------------------------------------------------------------------------


class TokenMask:
    """Build and apply vocabulary masks from FSM state.

    A mask is a boolean array of shape (vocab_size,) where True means the
    token is allowed.  Applying it zeros the logits of disallowed tokens.
    """

    def __init__(self, vocab_size: int) -> None:
        if type(vocab_size) is not int or vocab_size <= 0:
            raise ValueError("positive speculative vocabulary size required")
        self.vocab_size = vocab_size

    def apply_fsm(
        self,
        logits: np.ndarray,
        fsm: GrammarFSM,
        id_to_text: Dict[int, str],
        *,
        stop_token_id: Optional[int] = None,
    ) -> np.ndarray:
        """Zero logits for tokens that the FSM would reject.

        Args:
            logits: (vocab_size,) float32 array.
            fsm: Current FSM.
            id_to_text: Mapping token_id → token_text (e.g. from tokeniser).

        Returns:
            Modified logits (same shape); disallowed tokens set to -inf.
        """
        if (
            not isinstance(logits, np.ndarray)
            or logits.ndim != 1
            or logits.shape != (self.vocab_size,)
            or logits.dtype.kind not in "fiu"
            or np.isnan(logits).any()
            or np.isposinf(logits).any()
        ):
            raise ValueError(
                "grammar mask requires one logit per configured vocabulary ID"
            )
        if stop_token_id is not None and (
            type(stop_token_id) is not int or not 0 <= stop_token_id < self.vocab_size
        ):
            raise ValueError("grammar stop token is outside the configured vocabulary")
        if fsm.mode == GrammarMode.FREE:
            return logits
        # Initialize every ID as forbidden. IDs missing from the tokenizer
        # vocabulary map must not retain their target logits unmasked.
        masked = np.full_like(logits, -np.inf)
        allowed_count = 0
        for tok_id, tok_text in id_to_text.items():
            if tok_id < 0 or tok_id >= self.vocab_size:
                continue
            if tok_id == stop_token_id and not fsm.is_complete():
                continue
            if tok_id == stop_token_id:
                masked[tok_id] = logits[tok_id]
                allowed_count += 1
                continue
            if fsm.preview_step(tok_text):
                allowed_count += 1
                masked[tok_id] = logits[tok_id]
        if allowed_count < PHASE1D_SPEC_MIN_VALID_TOKENS:
            raise ValueError("active grammar has no token in the configured vocabulary")
        return masked

    @staticmethod
    def top_p_filter(logits: np.ndarray, top_p: float = 0.95) -> np.ndarray:
        """Nucleus (top-p) filtering: zero tokens outside the top-p probability mass."""
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        sorted_idx = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[sorted_idx])
        cutoff = np.searchsorted(cumulative, top_p)
        remove = sorted_idx[cutoff + 1 :]
        logits = logits.copy()
        logits[remove] = -1e9
        return logits

    @staticmethod
    def distribution(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        if (
            not isinstance(logits, np.ndarray)
            or logits.ndim != 1
            or logits.size == 0
            or type(temperature) not in (int, float)
            or not math.isfinite(temperature)
            or temperature <= 0
            or np.isnan(logits).any()
            or np.isposinf(logits).any()
        ):
            raise ValueError(
                "finite one-dimensional logits and positive temperature required"
            )
        scaled = logits.astype(np.float64, copy=False) / temperature
        maximum = np.max(scaled)
        if not math.isfinite(float(maximum)):
            raise ValueError("logit row has no finite candidate")
        probs = np.exp(scaled - maximum)
        total = float(probs.sum())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("logit row has no normalizable probability mass")
        probs /= total
        return probs

    @staticmethod
    def sample(
        logits: np.ndarray, temperature: float = 1.0, *, rng=None
    ) -> Tuple[int, float]:
        """Sample one token; return (token_id, log_prob)."""
        probs = TokenMask.distribution(logits, temperature)
        generator = np.random.default_rng() if rng is None else rng
        tok = int(generator.choice(len(probs), p=probs))
        return tok, float(math.log(probs[tok] + 1e-45))


# ---------------------------------------------------------------------------
# SpeculativeDraftBuffer
# ---------------------------------------------------------------------------


@dataclass
class DraftToken:
    token_id: int
    log_prob: float  # log P_draft(token | context)
    probabilities: np.ndarray  # complete grammar-constrained P_draft distribution
    token_text: str = ""


class SpeculativeDraftBuffer:
    """Holds K draft tokens and their full grammar-constrained distributions."""

    def __init__(self, *, max_probability_bytes: int = 64 * 1024**2) -> None:
        if type(max_probability_bytes) is not int or max_probability_bytes <= 0:
            raise ValueError("positive speculative distribution memory budget required")
        self.max_probability_bytes = max_probability_bytes
        self._tokens: List[DraftToken] = []
        self._lock = threading.Lock()

    def fill(
        self,
        draft_fn: Callable[[List[int]], np.ndarray],
        context_ids: List[int],
        k: int,
        fsm: GrammarFSM,
        token_mask: TokenMask,
        id_to_text: Dict[int, str],
        *,
        cancelled=None,
        rng=None,
        temperature: float = 1.0,
        stop_token_id: Optional[int] = None,
    ) -> None:
        """Generate K draft tokens.

        Args:
            draft_fn: Callable[token_ids → logits (vocab_size,)].
            context_ids: Current generation context.
            k: Number of tokens to draft.
            fsm: Grammar FSM for masking.
            token_mask: TokenMask helper.
            id_to_text: Token vocabulary.
        """
        if (
            type(k) is not int
            or not 1 <= k <= 256
            or k * token_mask.vocab_size * np.dtype(np.float64).itemsize
            > self.max_probability_bytes
        ):
            raise MemoryError(
                "full speculative draft distributions exceed their admitted memory budget"
            )
        with self._lock:
            self._tokens.clear()
            ids = list(context_ids)
            for _ in range(k):
                try:
                    logits = _call_cancellable(
                        draft_fn, ids, cancelled=cancelled, rng=rng
                    )
                    if logits is None or len(logits) == 0:
                        break
                    logits = token_mask.apply_fsm(
                        logits,
                        fsm,
                        id_to_text,
                        stop_token_id=stop_token_id,
                    )
                    probabilities = token_mask.distribution(logits, temperature)
                    generator = np.random.default_rng() if rng is None else rng
                    tok = int(generator.choice(len(probabilities), p=probabilities))
                    lp = float(math.log(probabilities[tok] + 1e-45))
                    text = id_to_text.get(tok, "")
                    self._tokens.append(DraftToken(tok, lp, probabilities.copy(), text))
                    ids.append(tok)
                    if stop_token_id is not None and tok == stop_token_id:
                        break
                    if not fsm.step(text):
                        break  # grammar violation in draft
                except Exception as exc:
                    if _is_cancellation_error(exc):
                        raise
                    _check_cancelled(cancelled)
                    logger.debug(
                        "phase1d draft: draft_fn error at position %d: %s",
                        len(self._tokens),
                        exc,
                    )
                    break

    def tokens(self) -> List[DraftToken]:
        with self._lock:
            return list(self._tokens)

    def clear(self) -> None:
        with self._lock:
            self._tokens.clear()


# ---------------------------------------------------------------------------
# SpeculativeVerifier
# ---------------------------------------------------------------------------


class SpeculativeVerifier:
    """Accept/reject draft tokens using speculative sampling.

    Per Leviathan et al. (2023): accept token t with probability
        min(1, p_main(t|ctx) / p_draft(t|ctx))
    On rejection, sample a correction token from the residual distribution
        max(0, p_main - p_draft) / Z.
    """

    def __init__(self) -> None:
        self._accepted = 0
        self._rejected = 0
        self._lock = threading.Lock()

    def verify(
        self,
        main_logits_batch: np.ndarray,  # (K, vocab_size) — one row per draft step
        draft_tokens: List[DraftToken],
        context_ids: List[int],
        fsm: GrammarFSM,
        token_mask: TokenMask,
        id_to_text: Dict[int, str],
        temperature: float = 1.0,
        cancelled=None,
        stop_token_id: Optional[int] = None,
        return_status: bool = False,
        rng=None,
    ) -> Tuple[List[int], List[str]] | Tuple[List[int], List[str], bool]:
        """Verify K draft tokens against K main-model logit rows.

        Returns:
            (accepted_ids, accepted_texts) — the accepted prefix plus a
            correction sampled from the full positive part of p_target-q_draft.
        """
        accepted_ids: List[int] = []
        accepted_texts: List[str] = []
        all_accepted = True
        accepted_draft_count = 0
        rejected_draft_count = 0
        working_fsm = fsm.clone()
        caller_rng = rng
        working_rng = np.random.default_rng()
        if caller_rng is not None:
            working_rng.bit_generator.state = copy.deepcopy(
                caller_rng.bit_generator.state
            )

        for i, dt in enumerate(draft_tokens):
            _check_cancelled(cancelled)
            if i >= len(main_logits_batch):
                all_accepted = False
                break
            logits_main = main_logits_batch[i]
            logits_main = token_mask.apply_fsm(
                logits_main,
                working_fsm,
                id_to_text,
                stop_token_id=stop_token_id,
            )
            p_main_all = token_mask.distribution(logits_main, temperature)
            q_all = np.asarray(dt.probabilities)
            if (
                q_all.dtype.kind != "f"
                or q_all.ndim != 1
                or q_all.shape != (token_mask.vocab_size,)
                or not np.isfinite(q_all).all()
                or np.any(q_all < 0)
                or not math.isclose(float(q_all.sum()), 1.0, rel_tol=1e-6, abs_tol=1e-8)
                or not 0 <= dt.token_id < token_mask.vocab_size
            ):
                raise ValueError(
                    "draft token is missing its normalized full-vocabulary distribution"
                )
            p_main_t = float(p_main_all[dt.token_id])
            p_draft_t = float(q_all[dt.token_id])
            if (
                p_draft_t <= 0
                or not math.isfinite(dt.log_prob)
                or not math.isclose(
                    math.log(p_draft_t), dt.log_prob, rel_tol=1e-5, abs_tol=1e-6
                )
            ):
                raise ValueError(
                    "draft token probability differs from its recorded distribution"
                )
            if (
                stop_token_id is not None
                and dt.token_id == stop_token_id
                and not working_fsm.is_complete()
            ):
                raise ValueError(
                    "draft emitted a stop token before completing the grammar"
                )

            accept_prob = min(1.0, p_main_t / p_draft_t)

            # The exact acceptance threshold is essential for preserving the
            # target distribution. A fixed minimum-probability cutoff biases
            # rare-but-valid target tokens and is not a safe optimization.
            if working_rng.random() < accept_prob:
                accepted_ids.append(dt.token_id)
                accepted_texts.append(dt.token_text)
                if stop_token_id is not None and dt.token_id == stop_token_id:
                    accepted_draft_count += 1
                    all_accepted = False
                    break
                if not working_fsm.step(dt.token_text):
                    raise ValueError("accepted draft token violates the active grammar")
                accepted_draft_count += 1
            else:
                all_accepted = False
                rejected_draft_count += 1
                # Exact speculative correction samples from max(0, p - q).
                residual = np.maximum(0, p_main_all - q_all)
                residual_sum = residual.sum()
                if residual_sum > 1e-9:
                    residual /= residual_sum
                    correction_id = int(working_rng.choice(len(residual), p=residual))
                    correction_text = id_to_text.get(correction_id, "")
                    if (stop_token_id is not None and correction_id == stop_token_id and
                            working_fsm.is_complete()):
                        accepted_ids.append(correction_id)
                        accepted_texts.append(correction_text)
                    elif working_fsm.step(correction_text):
                        accepted_ids.append(correction_id)
                        accepted_texts.append(correction_text)
                break  # stop at first rejection; restart draft from here

        _check_cancelled(cancelled)
        fsm._commit_from(working_fsm)
        if caller_rng is not None:
            caller_rng.bit_generator.state = copy.deepcopy(
                working_rng.bit_generator.state
            )
        with self._lock:
            self._accepted += accepted_draft_count
            self._rejected += rejected_draft_count
        if return_status:
            return (
                accepted_ids,
                accepted_texts,
                all_accepted and len(accepted_ids) == len(draft_tokens),
            )
        return accepted_ids, accepted_texts

    def stats(self) -> dict:
        with self._lock:
            total = self._accepted + self._rejected
            return {
                "spec_accepted": self._accepted,
                "spec_rejected": self._rejected,
                "spec_acceptance_rate": self._accepted / total if total else None,
            }


# ---------------------------------------------------------------------------
# GrammarSpeculativeDecoder
# ---------------------------------------------------------------------------


class GrammarSpeculativeDecoder:
    """Orchestrate grammar-constrained speculative decoding.

    Usage::

        decoder = GrammarSpeculativeDecoder(
            draft_fn=my_small_model,
            main_verify_fn=my_main_model_batch,
            vocab_size=32000,
            id_to_text=tokenizer.vocab,
        )
        decoder.set_grammar(GrammarMode.JSON)
        tokens = decoder.generate(context_ids, max_new_tokens=256)

    When no ``draft_fn`` is provided the decoder falls back to grammar-masked
    greedy decoding using the main model only.
    """

    def __init__(
        self,
        main_verify_fn: Callable[[List[int], int], np.ndarray],
        vocab_size: int,
        id_to_text: Dict[int, str],
        draft_fn: Optional[Callable[[List[int]], np.ndarray]] = None,
        draft_len: int = PHASE1D_SPEC_DRAFT_LEN,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        max_draft_bytes: int = 64 * 1024**2,
    ) -> None:
        """
        Args:
            main_verify_fn: Callable(context_ids, k) → (k, vocab_size) logits.
                            Runs k forward passes on the main model.
            vocab_size: Vocabulary size.
            id_to_text: {token_id: token_text} mapping.
            draft_fn: Optional lightweight model, callable(context_ids) → (vocab_size,) logits.
            draft_len: Speculation depth K.
            temperature: Sampling temperature.
        """
        if not callable(main_verify_fn):
            raise ValueError("callable target batch-logit verifier required")
        if (
            type(vocab_size) is not int
            or vocab_size <= 0
            or type(draft_len) is not int
            or not 1 <= draft_len <= 256
            or type(temperature) not in (int, float)
            or not math.isfinite(temperature)
            or temperature <= 0
            or type(max_draft_bytes) is not int
            or max_draft_bytes <= 0
        ):
            raise ValueError(
                "bounded vocabulary, draft length, temperature, and draft memory required"
            )
        if (
            draft_fn is not None
            and draft_len * vocab_size * np.dtype(np.float64).itemsize > max_draft_bytes
        ):
            raise MemoryError(
                "full speculative draft distributions exceed the configured budget"
            )
        self.main_verify_fn = main_verify_fn
        self.vocab_size = vocab_size
        self.id_to_text = id_to_text
        self.draft_fn = draft_fn
        self.draft_len = draft_len
        self.temperature = temperature
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("speculative RNG seed must be a nonnegative integer")
        draft_seed, target_seed = np.random.SeedSequence(seed).spawn(2)
        self._draft_rng = np.random.default_rng(draft_seed)
        self._target_rng = np.random.default_rng(target_seed)

        self._fsm = GrammarFSM(mode=GrammarMode.FREE)
        self._token_mask = TokenMask(vocab_size)
        self._draft_buf = SpeculativeDraftBuffer(max_probability_bytes=max_draft_bytes)
        self._verifier = SpeculativeVerifier()
        self._lock = threading.Lock()
        # A decoder owns persistent grammar and RNG state. Serialize complete
        # requests (including the lifetime of a stream), not merely snapshots
        # and commits, so concurrent calls cannot fork and overwrite state.
        self._request_lock = threading.Lock()
        self._total_generated = 0

    def set_grammar(self, mode: str, pattern: str = "") -> None:
        """Switch grammar mode.  Resets the FSM."""
        with self._request_lock:
            with self._lock:
                self._fsm = GrammarFSM(mode=mode, pattern=pattern)

    def generate(
        self,
        context_ids: List[int],
        max_new_tokens: int = 128,
        stop_token_id: Optional[int] = None,
        *,
        cancelled=None,
    ) -> List[int]:
        self._validate_request(context_ids, max_new_tokens, stop_token_id, cancelled)
        with self._request_lock:
            return self._generate_request(
                context_ids,
                max_new_tokens,
                stop_token_id,
                cancelled=cancelled,
            )

    def _generate_request(
        self,
        context_ids: List[int],
        max_new_tokens: int = 128,
        stop_token_id: Optional[int] = None,
        *,
        cancelled=None,
    ) -> List[int]:
        """Generate up to ``max_new_tokens`` tokens.

        Returns:
            List of generated token IDs.
        """
        if cancelled is not None and not callable(cancelled):
            raise ValueError("speculative cancellation must be callable")
        _check_cancelled(cancelled)
        generated: List[int] = []
        ids = list(context_ids)
        with self._lock:
            fsm = self._fsm.clone()
            draft_rng = np.random.default_rng()
            draft_rng.bit_generator.state = copy.deepcopy(
                self._draft_rng.bit_generator.state
            )
            target_rng = np.random.default_rng()
            target_rng.bit_generator.state = copy.deepcopy(
                self._target_rng.bit_generator.state
            )

        while len(generated) < max_new_tokens:
            _check_cancelled(cancelled)
            remaining = max_new_tokens - len(generated)
            k = min(self.draft_len, remaining) if self.draft_fn else 1

            if self.draft_fn is not None and k > 1:
                speculative_draft_rng_state = copy.deepcopy(
                    draft_rng.bit_generator.state
                )
                speculative_target_rng_state = copy.deepcopy(
                    target_rng.bit_generator.state
                )
                self._draft_buf.fill(
                    self.draft_fn,
                    ids,
                    k,
                    fsm.clone(),
                    self._token_mask,
                    self.id_to_text,
                    cancelled=cancelled,
                    rng=draft_rng,
                    temperature=self.temperature,
                    stop_token_id=stop_token_id,
                )
                draft_tokens = self._draft_buf.tokens()
                if not draft_tokens:
                    draft_rng.bit_generator.state = speculative_draft_rng_state
                    target_rng.bit_generator.state = speculative_target_rng_state
                    k = 1
                else:
                    verification_committed = False
                    try:
                        bonus_available = (
                            len(generated) + len(draft_tokens) < max_new_tokens
                        )
                        verify_count = len(draft_tokens) + int(bonus_available)
                        logits_batch = _call_cancellable(
                            self.main_verify_fn,
                            ids,
                            verify_count,
                            cancelled=cancelled,
                            rng=target_rng,
                        )
                        if (
                            not isinstance(logits_batch, np.ndarray)
                            or logits_batch.ndim != 2
                            or logits_batch.shape[1] != self.vocab_size
                            or logits_batch.shape[0] < len(draft_tokens)
                        ):
                            draft_rng.bit_generator.state = speculative_draft_rng_state
                            target_rng.bit_generator.state = (
                                speculative_target_rng_state
                            )
                            k = 1
                        else:
                            verified_fsm = fsm.clone()
                            accepted_ids, _, all_drafts_accepted = (
                                self._verifier.verify(
                                    logits_batch,
                                    draft_tokens,
                                    ids,
                                    verified_fsm,
                                    self._token_mask,
                                    self.id_to_text,
                                    self.temperature,
                                    cancelled=cancelled,
                                    stop_token_id=stop_token_id,
                                    return_status=True,
                                    rng=target_rng,
                                )
                            )
                            fsm = verified_fsm
                            verification_committed = True
                            generated.extend(accepted_ids)
                            ids.extend(accepted_ids)
                            if (
                                stop_token_id is not None
                                and stop_token_id in accepted_ids
                            ):
                                break
                            if (
                                all_drafts_accepted
                                and bonus_available
                                and logits_batch.shape[0] > len(draft_tokens)
                            ):
                                bonus_logits = logits_batch[len(draft_tokens)]
                                bonus_logits = self._token_mask.apply_fsm(
                                    bonus_logits,
                                    fsm,
                                    self.id_to_text,
                                    stop_token_id=stop_token_id,
                                )
                                bonus, _ = self._token_mask.sample(
                                    bonus_logits,
                                    self.temperature,
                                    rng=target_rng,
                                )
                                if not (
                                    stop_token_id is not None and bonus == stop_token_id
                                ) and not fsm.step(self.id_to_text.get(bonus, "")):
                                    raise ValueError(
                                        "bonus token violates the active grammar"
                                    )
                                generated.append(bonus)
                                ids.append(bonus)
                                if stop_token_id is not None and bonus == stop_token_id:
                                    break
                            continue
                    except Exception as exc:
                        if _is_cancellation_error(exc):
                            raise
                        _check_cancelled(cancelled)
                        if not verification_committed:
                            draft_rng.bit_generator.state = speculative_draft_rng_state
                            target_rng.bit_generator.state = (
                                speculative_target_rng_state
                            )
                        logger.warning(
                            "phase1d spec decode: verify error: %s — "
                            "falling back to greedy",
                            exc,
                        )
                        k = 1

            greedy_rng_state = copy.deepcopy(target_rng.bit_generator.state)
            try:
                logits_batch = _call_cancellable(
                    self.main_verify_fn,
                    ids,
                    1,
                    cancelled=cancelled,
                    rng=target_rng,
                )
                if logits_batch is None or len(logits_batch) == 0:
                    target_rng.bit_generator.state = greedy_rng_state
                    break
                logits = logits_batch[0] if logits_batch.ndim == 2 else logits_batch
                logits = self._token_mask.apply_fsm(
                    logits,
                    fsm,
                    self.id_to_text,
                    stop_token_id=stop_token_id,
                )
                tok, _ = self._token_mask.sample(
                    logits, self.temperature, rng=target_rng
                )
                if not (stop_token_id is not None and tok == stop_token_id) and not fsm.step(
                    self.id_to_text.get(tok, "")
                ):
                    raise ValueError("sampled token violates the active grammar")
                generated.append(tok)
                ids.append(tok)
                if stop_token_id is not None and tok == stop_token_id:
                    break
            except Exception as exc:
                if _is_cancellation_error(exc):
                    raise
                _check_cancelled(cancelled)
                target_rng.bit_generator.state = greedy_rng_state
                logger.warning("phase1d spec decode: main model error: %s", exc)
                break

        _check_cancelled(cancelled)
        with self._lock:
            self._fsm = fsm
            self._draft_rng.bit_generator.state = copy.deepcopy(
                draft_rng.bit_generator.state
            )
            self._target_rng.bit_generator.state = copy.deepcopy(
                target_rng.bit_generator.state
            )
            self._total_generated += len(generated)
        return generated

    def _validate_request(
        self,
        context_ids: List[int],
        max_new_tokens: int,
        stop_token_id: Optional[int],
        cancelled,
    ) -> None:
        """Reject malformed requests before touching persistent decoder state."""
        if (
            not isinstance(context_ids, (list, tuple))
            or any(type(token_id) is not int or not 0 <= token_id < self.vocab_size
                   for token_id in context_ids)
        ):
            raise ValueError("context IDs must be integer IDs in the configured vocabulary")
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if stop_token_id is not None and (
            type(stop_token_id) is not int or not 0 <= stop_token_id < self.vocab_size
        ):
            raise ValueError("stop token is outside the configured vocabulary")
        if cancelled is not None and not callable(cancelled):
            raise ValueError("speculative cancellation must be callable")

    def stream(
        self,
        context_ids: List[int],
        max_new_tokens: int = 128,
        stop_token_id: Optional[int] = None,
        *,
        cancelled=None,
    ) -> Iterator[int]:
        self._validate_request(context_ids, max_new_tokens, stop_token_id, cancelled)
        with self._request_lock:
            yield from self._stream_request(
                context_ids,
                max_new_tokens,
                stop_token_id,
                cancelled=cancelled,
            )

    def _stream_request(
        self,
        context_ids: List[int],
        max_new_tokens: int = 128,
        stop_token_id: Optional[int] = None,
        *,
        cancelled=None,
    ) -> Iterator[int]:
        """Streaming variant: yield one token ID at a time."""
        if cancelled is not None and not callable(cancelled):
            raise ValueError("speculative cancellation must be callable")
        _check_cancelled(cancelled)
        generated_so_far: List[int] = []
        ids = list(context_ids)
        with self._lock:
            fsm = self._fsm.clone()
            target_rng = np.random.default_rng()
            target_rng.bit_generator.state = copy.deepcopy(
                self._target_rng.bit_generator.state
            )
        while len(generated_so_far) < max_new_tokens:
            _check_cancelled(cancelled)
            try:
                logits_batch = _call_cancellable(
                    self.main_verify_fn,
                    ids,
                    1,
                    cancelled=cancelled,
                    rng=target_rng,
                )
                if logits_batch is None or len(logits_batch) == 0:
                    return
                logits = logits_batch[0] if logits_batch.ndim == 2 else logits_batch
                logits = self._token_mask.apply_fsm(
                    logits,
                    fsm,
                    self.id_to_text,
                    stop_token_id=stop_token_id,
                )
                tok, _ = self._token_mask.sample(
                    logits, self.temperature, rng=target_rng
                )
                if not (stop_token_id is not None and tok == stop_token_id) and not fsm.step(
                    self.id_to_text.get(tok, "")
                ):
                    raise ValueError("sampled token violates the active grammar")
                generated_so_far.append(tok)
                ids.append(tok)
                with self._lock:
                    self._fsm = fsm.clone()
                    self._target_rng.bit_generator.state = copy.deepcopy(
                        target_rng.bit_generator.state
                    )
                    self._total_generated += 1
                yield tok
                if stop_token_id is not None and tok == stop_token_id:
                    return
            except InterruptedError:
                raise
            except Exception as exc:
                if _is_cancellation_error(exc):
                    raise
                _check_cancelled(cancelled)
                logger.warning("phase1d spec stream: error: %s", exc)
                return

    def metrics(self) -> dict:
        with self._lock:
            return {
                "spec_total_generated": self._total_generated,
                "spec_draft_len": self.draft_len,
                "spec_has_draft_model": self.draft_fn is not None,
                **self._fsm.stats(),
                **self._verifier.stats(),
            }


__all__ = [
    "GrammarMode",
    "GrammarFSM",
    "TokenMask",
    "SpeculativeDraftBuffer",
    "SpeculativeVerifier",
    "GrammarSpeculativeDecoder",
    "PHASE1D_SPEC_DRAFT_LEN",
]
