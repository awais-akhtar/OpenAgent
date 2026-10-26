"""A05 — Standalone speculative RNG state contract helper.

This module provides an opt-in ownership and lifecycle contract for target/draft
RNG state in speculative generation. It is not automatically wired into the
family-specific production speculative decoders, does not snapshot physical
KV/model state, and cannot by itself certify production rollback. The
invariants it protects when explicitly used:

  1. **Exclusive ownership**: only one caller holds the combined target+draft
     RNG state at any moment; a second caller is rejected rather than forking
     the random stream.

  2. **Separate seeded RNGs**: target and draft samplers use independent seeds
     derived from the session key so they never advance in lock-step.

  3. **Capture-before-mutate**: a caller must capture the current target RNG
     state before issuing any draft step, and the captured state is the only
     valid rollback point on cancellation or error.

  4. **Rollback on cancellation**: cancelling a speculative sequence restores
     the target RNG to the pre-draft captured state and discards all draft
     state so the next caller starts from a consistent position.

  5. **Cross-session isolation**: state slots are keyed by (session_id,
     sequence_number); a caller that presents a stale sequence number is
     rejected rather than silently advancing a previous sequence.

All public methods are thread-safe.  No threads are started.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import struct
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple



# ---------------------------------------------------------------------------
# RNG state representation
# ---------------------------------------------------------------------------

_XORSHIFT128_WORDS = 4  # 4 × uint32 = 128-bit state
_STATE_BYTES = _XORSHIFT128_WORDS * 4


@dataclass(frozen=True)
class RNGState:
    """Immutable snapshot of a 128-bit xorshift RNG."""
    words: Tuple[int, ...]

    def __post_init__(self):
        if (not isinstance(self.words, tuple) or
                len(self.words) != _XORSHIFT128_WORDS or
                not all(type(w) is int and 0 <= w <= 0xFFFFFFFF
                        for w in self.words)):
            raise ValueError(
                'RNG state must be a tuple of four uint32 words')

    def to_bytes(self) -> bytes:
        return struct.pack('<' + 'I' * _XORSHIFT128_WORDS, *self.words)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'RNGState':
        if len(data) != _STATE_BYTES:
            raise ValueError(f'RNG state requires exactly {_STATE_BYTES} bytes')
        return cls(words=tuple(struct.unpack('<' + 'I' * _XORSHIFT128_WORDS, data)))

    @classmethod
    def from_seed(cls, seed: bytes) -> 'RNGState':
        """Derive a non-zero RNG state from an arbitrary seed."""
        digest = hashlib.sha256(seed).digest()
        words = list(struct.unpack('<IIIIIIII', digest)[:_XORSHIFT128_WORDS])
        if all(w == 0 for w in words):
            words[0] = 1
        return cls(words=tuple(words))

    def advance(self) -> Tuple['RNGState', int]:
        """Return (next_state, sample_uint32) via xorshift128."""
        a, b, c, d = self.words
        t = a ^ (a << 11) & 0xFFFFFFFF
        a = b
        b = c
        c = d
        d = (d ^ (d >> 19)) ^ (t ^ (t >> 8))
        d &= 0xFFFFFFFF
        return RNGState(words=(a & 0xFFFFFFFF, b & 0xFFFFFFFF,
                                c & 0xFFFFFFFF, d)), d


# ---------------------------------------------------------------------------
# Per-session speculative state slot
# ---------------------------------------------------------------------------

@dataclass
class _SpecSlot:
    session_id: str
    sequence: int
    target_state: RNGState
    draft_state: RNGState
    pre_draft_target: Optional[RNGState]   # captured before first draft step
    pre_draft_draft: Optional[RNGState]
    owner_token: str                       # opaque token held by the active caller
    draft_steps: int = 0
    accepted_steps: int = 0
    cancelled: bool = False

    def capture_pre_draft(self) -> None:
        """Snapshot the current target state before any draft tokens."""
        if self.pre_draft_target is not None:
            raise RuntimeError(
                f'session {self.session_id}: pre-draft target state already captured; '
                f'rollback before re-capturing')
        self.pre_draft_target = self.target_state
        self.pre_draft_draft = self.draft_state

    def rollback_target(self) -> None:
        """Restore target RNG to the pre-draft snapshot and discard draft state."""
        if self.pre_draft_target is None or self.pre_draft_draft is None:
            raise RuntimeError(
                f'session {self.session_id}: no pre-draft target state to roll back to')
        self.target_state = self.pre_draft_target
        self.draft_state = self.pre_draft_draft
        self.pre_draft_target = None
        self.pre_draft_draft = None
        self.draft_steps = 0
        self.accepted_steps = 0

    def advance_draft(self) -> int:
        """Advance the draft RNG and return a uint32 sample."""
        next_state, sample = self.draft_state.advance()
        self.draft_state = next_state
        self.draft_steps += 1
        return sample

    def advance_target(self) -> int:
        """Advance the target RNG and return a uint32 sample."""
        next_state, sample = self.target_state.advance()
        self.target_state = next_state
        return sample

    def accept_step(self) -> None:
        self.accepted_steps += 1


# ---------------------------------------------------------------------------
# Global speculative state manager
# ---------------------------------------------------------------------------

class SpeculativeStateManager:
    """Thread-safe manager for speculative RNG state across sessions.

    This helper manages RNG snapshots only. Production callers must separately
    snapshot/rollback target and draft model/cache state and explicitly invoke
    this API; constructing the manager alone does not enable speculation.

    Typical call pattern for a speculative sequence::

        token = manager.begin(session_id, owner_token)
        manager.capture_pre_draft(session_id, owner_token)
        for _ in range(draft_steps):
            draft_sample = manager.sample_draft(session_id, owner_token)
            # ... run draft model ...
        target_sample = manager.sample_target(session_id, owner_token)
        if accepted:
            manager.accept_step(session_id, owner_token)
        # ... on cancellation ...
        manager.rollback(session_id, owner_token)
        manager.release(session_id, owner_token)
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._slots: Dict[str, _SpecSlot] = {}
        self._session_keys: Dict[str, bytes] = {}
        self._closed = False

    def open_session(self, session_id: str, *, session_key: bytes) -> None:
        """Register a new session with deterministically seeded RNG states.

        session_key is a stable per-session secret (e.g. derived from the
        checkpoint identity + session UUID).  It must be at least 16 bytes.
        """
        if (type(session_id) is not str or not session_id or len(session_id) > 256 or
                '\x00' in session_id):
            raise ValueError('bounded non-empty session identifier required')
        if not isinstance(session_key, (bytes, bytearray)) or not 16 <= len(session_key) <= 4096:
            raise ValueError('16–4096-byte session key required')
        with self._lock:
            if self._closed:
                raise RuntimeError('speculative state manager is closed')
            if session_id in self._slots:
                raise ValueError(
                    f'session {session_id!r} is already registered; '
                    f'close it before re-opening')
            target_seed = hmac.new(session_key, b'target-rng-v1', hashlib.sha256).digest()
            draft_seed = hmac.new(session_key, b'draft-rng-v1', hashlib.sha256).digest()
            self._slots[session_id] = _SpecSlot(
                session_id=session_id,
                sequence=0,
                target_state=RNGState.from_seed(target_seed),
                draft_state=RNGState.from_seed(draft_seed),
                pre_draft_target=None,
                pre_draft_draft=None,
                owner_token='',
            )
            self._session_keys[session_id] = bytes(session_key)

    def begin(self, session_id: str, owner_token: str) -> int:
        """Claim exclusive ownership; return the current sequence number.

        A caller that wins this call must eventually call release() or rollback().
        A caller that presents a stale owner_token is rejected immediately.
        """
        if (type(owner_token) is not str or not owner_token or len(owner_token) > 256 or
                '\x00' in owner_token):
            raise ValueError('bounded non-empty owner token required')
        with self._lock:
            slot = self._require_slot(session_id)
            if slot.owner_token:
                raise RuntimeError(
                    f'session {session_id!r} is already owned by another caller; '
                    f'only one speculative generation may run at a time')
            slot.owner_token = owner_token
            slot.cancelled = False
            return slot.sequence

    def capture_pre_draft(self, session_id: str, owner_token: str) -> RNGState:
        """Snapshot target state before the first draft step."""
        with self._lock:
            slot = self._require_mutable_owned(session_id, owner_token)
            slot.capture_pre_draft()
            return slot.pre_draft_target  # type: ignore[return-value]

    def sample_draft(self, session_id: str, owner_token: str) -> int:
        """Advance the draft RNG and return a uint32 sample."""
        with self._lock:
            slot = self._require_mutable_owned(session_id, owner_token)
            if slot.pre_draft_target is None:
                raise RuntimeError(
                    f'session {session_id!r}: capture_pre_draft must be called '
                    f'before any draft samples')
            return slot.advance_draft()

    def sample_target(self, session_id: str, owner_token: str) -> int:
        """Advance the target RNG and return a uint32 sample."""
        with self._lock:
            slot = self._require_mutable_owned(session_id, owner_token)
            return slot.advance_target()

    def accept_step(self, session_id: str, owner_token: str) -> None:
        """Record acceptance of one draft step."""
        with self._lock:
            slot = self._require_mutable_owned(session_id, owner_token)
            slot.accept_step()

    def rollback(self, session_id: str, owner_token: str) -> None:
        """Restore target RNG to pre-draft state and clear draft progress.

        Must be called on cancellation or error before release().
        """
        with self._lock:
            slot = self._require_owned(session_id, owner_token)
            slot.rollback_target()
            slot.cancelled = True

    def commit(self, session_id: str, owner_token: str) -> None:
        """Commit a resolved proposal while retaining both advanced RNGs."""
        with self._lock:
            slot = self._require_mutable_owned(session_id, owner_token)
            if slot.pre_draft_target is None or slot.pre_draft_draft is None:
                raise RuntimeError(
                    f'session {session_id!r}: no active speculative proposal to commit')
            slot.pre_draft_target = None
            slot.pre_draft_draft = None

    def release(self, session_id: str, owner_token: str) -> Tuple[int, int]:
        """Release ownership and advance the sequence counter.

        Returns (draft_steps, accepted_steps) for telemetry.
        """
        with self._lock:
            slot = self._require_owned(session_id, owner_token)
            if (slot.pre_draft_target is not None or slot.pre_draft_draft is not None) and not slot.cancelled:
                raise RuntimeError(
                    f'session {session_id!r}: pre-draft state was captured but '
                    f'the sequence was not completed or rolled back; '
                    f'call rollback() on cancellation/error before release()')
            draft_steps = slot.draft_steps
            accepted = slot.accepted_steps
            slot.owner_token = ''
            slot.pre_draft_target = None
            slot.pre_draft_draft = None
            slot.draft_steps = 0
            slot.accepted_steps = 0
            slot.cancelled = False
            slot.sequence += 1
            return draft_steps, accepted

    def close_session(self, session_id: str) -> None:
        """Deregister a session; must not be actively owned."""
        with self._lock:
            slot = self._slots.get(session_id)
            if slot is None:
                return
            if slot.owner_token:
                raise RuntimeError(
                    f'session {session_id!r} is still owned; release it before closing')
            del self._slots[session_id]
            self._session_keys.pop(session_id, None)

    def close(self) -> None:
        """Close the manager after every active owner has released its lease.

        Clearing a slot while a decode still owns it would make later rollback
        impossible and could leave the decoder's physical target/draft state
        out of sync with its RNG state. Refuse close in that case; the caller
        can retry after the active sequence commits or rolls back and releases.
        """
        with self._lock:
            active = sorted(
                slot.session_id for slot in self._slots.values() if slot.owner_token
            )
            if active:
                raise RuntimeError(
                    'speculative state manager has active owners; release them '
                    f'before closing: {active!r}')
            self._closed = True
            self._slots.clear()
            self._session_keys.clear()

    def snapshot(self, session_id: str) -> dict:
        """Return a serializable snapshot for distributed recovery."""
        with self._lock:
            slot = self._require_slot(session_id)
            if slot.owner_token:
                raise RuntimeError(
                    f'session {session_id!r}: cannot snapshot while owned')
            snapshot = {
                'version': 1,
                'session_id': session_id,
                'sequence': slot.sequence,
                'target_state': list(slot.target_state.words),
                'draft_state': list(slot.draft_state.words),
                'key_id': hashlib.sha256(self._session_keys[session_id]).hexdigest(),
            }
            key = self._session_keys[session_id]
            snapshot['auth'] = hmac.new(
                key, self._snapshot_payload(snapshot), hashlib.sha256
            ).hexdigest()
            return snapshot

    @staticmethod
    def _snapshot_payload(snapshot: dict) -> bytes:
        payload = {key: snapshot[key] for key in (
            'version', 'session_id', 'sequence', 'target_state', 'draft_state', 'key_id')}
        return json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')

    def restore(self, snapshot: dict, *, session_key: bytes | None = None) -> None:
        """Restore an authenticated snapshot; pass the key for cross-process recovery."""
        required = {'version', 'session_id', 'sequence', 'target_state', 'draft_state', 'key_id', 'auth'}
        if type(snapshot) is not dict or set(snapshot) != required or snapshot.get('version') != 1:
            raise ValueError('invalid speculative state snapshot')
        session_id = snapshot['session_id']
        sequence = snapshot['sequence']
        if (type(session_id) is not str or not session_id or len(session_id) > 256 or
                '\x00' in session_id or type(sequence) is not int or sequence < 0):
            raise ValueError('invalid speculative session identity or sequence')
        if session_key is None:
            with self._lock:
                session_key = self._session_keys.get(session_id)
        if (not isinstance(session_key, (bytes, bytearray)) or
                not 16 <= len(session_key) <= 4096):
            raise ValueError('snapshot restore requires its 16–4096-byte session key')
        key = bytes(session_key)
        key_id = hashlib.sha256(key).hexdigest()
        if type(snapshot['key_id']) is not str or snapshot['key_id'] != key_id:
            raise ValueError('speculative snapshot belongs to a different session key')
        expected = hmac.new(key, self._snapshot_payload(snapshot), hashlib.sha256).hexdigest()
        if type(snapshot['auth']) is not str or not hmac.compare_digest(snapshot['auth'], expected):
            raise ValueError('speculative snapshot authentication failed')
        target = RNGState(words=tuple(snapshot['target_state']))
        draft = RNGState(words=tuple(snapshot['draft_state']))
        with self._lock:
            if self._closed:
                raise RuntimeError('speculative state manager is closed')
            existing = self._slots.get(session_id)
            if existing is not None and existing.owner_token:
                raise RuntimeError(
                    f'session {session_id!r}: cannot restore while owned')
            existing_key = self._session_keys.get(session_id)
            if existing_key is not None and not hmac.compare_digest(existing_key, key):
                raise ValueError('speculative session restore key differs from registered session')
            if existing is not None and sequence < existing.sequence:
                raise ValueError(
                    f'session {session_id!r}: stale speculative snapshot sequence '
                    f'{sequence} is older than current sequence {existing.sequence}')
            self._slots[session_id] = _SpecSlot(
                session_id=session_id,
                sequence=sequence,
                target_state=target,
                draft_state=draft,
                pre_draft_target=None,
                pre_draft_draft=None,
                owner_token='',
            )
            self._session_keys[session_id] = key

    def _require_slot(self, session_id: str) -> _SpecSlot:
        slot = self._slots.get(session_id)
        if slot is None:
            raise KeyError(f'speculative session {session_id!r} is not registered')
        return slot

    def _require_owned(self, session_id: str, owner_token: str) -> _SpecSlot:
        slot = self._require_slot(session_id)
        if not owner_token or slot.owner_token != owner_token:
            raise RuntimeError(
                f'session {session_id!r}: caller does not hold ownership '
                f'(expected={slot.owner_token!r}, got={owner_token!r})')
        return slot

    def _require_mutable_owned(self, session_id: str, owner_token: str) -> _SpecSlot:
        """Require a live, non-rolled-back ownership lease for state mutation."""
        slot = self._require_owned(session_id, owner_token)
        if slot.cancelled:
            raise RuntimeError(
                f'session {session_id!r}: speculative lease was rolled back; '
                f'release it before beginning another sequence')
        return slot
