"""A04 — Media/serving lifecycle per family.

Each family that admits media inputs has a distinct lifecycle contract:
  - Admit: validate the family, media inputs, and resource limits; return a
    frame_id bound to this stream.
  - Streaming: per-frame hook called by the generation loop as the model
    processes each media chunk.
  - Complete (success): release media resources and update counters.
  - Cancel: ordered teardown on caller-driven cancellation before completion.
  - Error: emergency teardown when a backend exception prevents normal cleanup.

All public methods are thread-safe.  No threads are started by this module.
A FamilyMediaLifecycleRegistry is the top-level object; callers look up a
per-family LifecycleHandler by family identity string before admitting media.

Families with no media contract ('mixtral', 'olmoe', 'glm-dsa') are
explicitly rejected rather than silently falling through to a no-op.
"""
from __future__ import annotations

import math
import threading
import uuid
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

from .failure_reporting import add_failure_note, failure_summary


# ---------------------------------------------------------------------------
# Media frame tracking
# ---------------------------------------------------------------------------

_ADMITTED_FAMILIES = frozenset({
    'qwen35', 'qwen4', 'glm5-next', 'inkling', 'kimi-k3', 'deepseek-v41',
})
_TEXT_ONLY_FAMILIES = frozenset({'mixtral', 'olmoe', 'glm-dsa'})
_FAMILY_MEDIA_TYPES = {
    'qwen35': frozenset({'image', 'video'}),
    'qwen4': frozenset({'image', 'video'}),
    'glm5-next': frozenset({'image', 'video'}),
    'inkling': frozenset({'image', 'audio'}),
    'kimi-k3': frozenset({'image', 'video'}),
    'deepseek-v41': frozenset({'vision'}),
}
_FAMILY_INPUT_FIELDS = {
    'qwen35': frozenset({'images', 'video', 'fps', 'text', 'audio'}),
    'qwen4': frozenset({'images', 'video', 'fps', 'text'}),
    'glm5-next': frozenset({'images', 'video', 'fps', 'text'}),
    'inkling': frozenset({'images', 'audio', 'sampling_rate', 'text'}),
    'kimi-k3': frozenset({'images', 'video', 'fps', 'text'}),
    'deepseek-v41': frozenset({'images', 'text'}),
}


def _validate_fields(inputs: dict, family: str) -> None:
    if type(inputs) is not dict:
        raise ValueError(f'{family} media: explicit media input mapping required')
    unknown = set(inputs) - _FAMILY_INPUT_FIELDS[family]
    if unknown:
        raise ValueError(f'{family} media: unsupported input fields: {sorted(unknown)}')
    if 'text' in inputs and type(inputs['text']) is not str:
        raise ValueError(f'{family} media: optional text input must be a string')


def _payload_nbytes(value) -> int:
    """Count the actual borrowed-buffer extent, not a view's element count."""
    if type(value) is bytes:
        return len(value)
    if type(value) is memoryview:
        return value.nbytes
    return 0


def _input_payload_nbytes(inputs: dict) -> int:
    """Count media buffers, including each member of an image sequence."""
    total = 0
    images = inputs.get('images')
    if isinstance(images, (list, tuple)):
        total += sum(_payload_nbytes(image) for image in images)
    for name in ('audio', 'video'):
        total += _payload_nbytes(inputs.get(name))
    return total


_FRAME_STATES = ('admitted', 'streaming', 'complete', 'cancelled', 'error')


def _raw_images(inputs: dict, family: str) -> None:
    images = inputs.get('images')
    if not isinstance(images, (list, tuple)) or not images:
        raise ValueError(f'{family} media: non-empty image list required')
    for image in images:
        if _payload_nbytes(image) < 8:
            raise ValueError(f'{family} media: raw image bytes required')


def _valid_fps(inputs: dict, family: str) -> None:
    fps = inputs.get('fps')
    if (type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0):
        raise ValueError(f'{family} media: finite positive fps required with video')


@dataclass
class MediaFrame:
    frame_id: str
    family: str
    media_types: Tuple[str, ...]
    input_byte_count: int
    state: str = 'admitted'
    stream_chunks: int = 0
    error_note: Optional[str] = None

    def _transition(self, target: str):
        _ALLOWED: Dict[str, Tuple[str, ...]] = {
            'admitted': ('streaming', 'complete', 'cancelled', 'error'),
            'streaming': ('streaming', 'complete', 'cancelled', 'error'),
            'complete': (),
            'cancelled': (),
            'error': (),
        }
        if target not in _ALLOWED.get(self.state, ()):
            raise ValueError(
                f'media frame {self.frame_id}: invalid state transition '
                f'{self.state!r} → {target!r}')
        self.state = target


# ---------------------------------------------------------------------------
# Per-family lifecycle contracts
# ---------------------------------------------------------------------------

def _validate_qwen35_media(inputs: dict) -> Tuple[str, ...]:
    """Qwen3.5: image/video processor fields; audio has no admitted tower."""
    _validate_fields(inputs, 'qwen35')
    types: List[str] = []
    if 'images' in inputs:
        _raw_images(inputs, 'qwen35')
        types.append('image')
    if 'audio' in inputs:
        raise ValueError('qwen35 media: audio tensors have no admitted consumer')
    if 'video' in inputs:
        video = inputs['video']
        if _payload_nbytes(video) < 8:
            raise ValueError('qwen35 media: raw video bytes required')
        _valid_fps(inputs, 'qwen35')
        types.append('video')
    if not types:
        raise ValueError('qwen35 media: at least one of images or video required')
    return tuple(types)


def _validate_qwen4_media(inputs: dict) -> Tuple[str, ...]:
    """Qwen4: image+text or video+text, PLE position state required."""
    _validate_fields(inputs, 'qwen4')
    types: List[str] = []
    if 'images' in inputs:
        _raw_images(inputs, 'qwen4')
        types.append('image')
    if 'video' in inputs:
        vid = inputs['video']
        if _payload_nbytes(vid) < 8:
            raise ValueError('qwen4 media: raw video bytes required')
        _valid_fps(inputs, 'qwen4')
        types.append('video')
    if not types:
        raise ValueError('qwen4 media: at least one of images or video required')
    return tuple(types)


def _validate_glm5_media(inputs: dict) -> Tuple[str, ...]:
    """GLM5-Next: image+text, KDA/MLA-aware position ordering."""
    _validate_fields(inputs, 'glm5-next')
    if 'images' not in inputs:
        raise ValueError('glm5-next media: images required')
    _raw_images(inputs, 'glm5-next')
    types = ['image']
    if 'video' in inputs:
        video = inputs['video']
        if _payload_nbytes(video) < 8:
            raise ValueError('glm5-next media: raw video bytes required')
        _valid_fps(inputs, 'glm5-next')
        types.append('video')
    return tuple(types)


def _validate_inkling_media(inputs: dict) -> Tuple[str, ...]:
    """Inkling media inputs only; MTP proposals belong to the P05 verifier."""
    if type(inputs) is dict and 'mtp_draft_tokens' in inputs:
        raise ValueError(
            'inkling MTP draft tokens are speculative state, not media; '
            'use a target-verified P05 generation path')
    _validate_fields(inputs, 'inkling')
    types: List[str] = []
    if 'images' in inputs:
        _raw_images(inputs, 'inkling')
        types.append('image')
    if 'audio' in inputs:
        audio = inputs['audio']
        if _payload_nbytes(audio) < 4:
            raise ValueError('inkling media: raw audio bytes required')
        rate = inputs.get('sampling_rate')
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
            raise ValueError('inkling media: finite positive sampling rate required with audio')
        types.append('audio')
    if not types:
        raise ValueError('inkling media: at least one admitted image or audio input required')
    return tuple(types)


def _validate_k3_media(inputs: dict) -> Tuple[str, ...]:
    """Kimi K3: image+video+text; exact K3ExactEvidence admission gating."""
    _validate_fields(inputs, 'kimi-k3')
    types: List[str] = []
    if 'images' in inputs:
        _raw_images(inputs, 'kimi-k3')
        types.append('image')
    if 'video' in inputs:
        vid = inputs['video']
        if _payload_nbytes(vid) < 8:
            raise ValueError('kimi-k3 media: raw video bytes required')
        _valid_fps(inputs, 'kimi-k3')
        types.append('video')
    if not types:
        raise ValueError('kimi-k3 media: at least one of images or video required')
    return tuple(types)


def _validate_v41_media(inputs: dict) -> Tuple[str, ...]:
    """V4.1 media input only; MTP and DSpark are stateful execution paths."""
    if type(inputs) is dict and 'mtp_draft_tokens' in inputs:
        raise ValueError(
            'DeepSeek MTP draft tokens are speculative state, not media; '
            'use a target-verified P05 generation path')
    if type(inputs) is dict and 'dspark_continuation' in inputs:
        raise ValueError(
            'DSpark continuation is owner-bound P12 recovery state, not media payload')
    _validate_fields(inputs, 'deepseek-v41')
    types: List[str] = []
    if 'images' in inputs:
        _raw_images(inputs, 'deepseek-v41')
        types.append('vision')
    if not types:
        raise ValueError('deepseek-v41 media: an admitted vision image is required')
    return tuple(types)


_FAMILY_VALIDATORS = {
    'qwen35': _validate_qwen35_media,
    'qwen4': _validate_qwen4_media,
    'glm5-next': _validate_glm5_media,
    'inkling': _validate_inkling_media,
    'kimi-k3': _validate_k3_media,
    'deepseek-v41': _validate_v41_media,
}


# ---------------------------------------------------------------------------
# Lifecycle manager
# ---------------------------------------------------------------------------

class FamilyMediaLifecycle:
    """Thread-safe media stream lifecycle tracker for one family."""

    def __init__(self, family: str, *, max_concurrent_frames: int = 8):
        if family not in _ADMITTED_FAMILIES:
            if family in _TEXT_ONLY_FAMILIES:
                raise ValueError(
                    f'{family!r} is a text-only family; it has no media lifecycle contract')
            raise ValueError(f'unrecognised family {family!r}; cannot admit media streams')
        if not isinstance(max_concurrent_frames, int) or not (1 <= max_concurrent_frames <= 64):
            raise ValueError('1–64 concurrent frames required')
        self._family = family
        self._max_frames = max_concurrent_frames
        self._lock = threading.Lock()
        self._frames: Dict[str, MediaFrame] = {}
        self._closed = False

    @property
    def family(self) -> str:
        return self._family

    def admit(self, inputs: dict) -> str:
        """Validate and admit one media stream; return its frame_id."""
        validator = _FAMILY_VALIDATORS[self._family]
        media_types = validator(inputs)
        byte_count = _input_payload_nbytes(inputs)
        return self._admit_types(media_types, byte_count)

    def admit_processed(self, media_types: Tuple[str, ...], *, input_byte_count: int) -> str:
        """Track processor-owned inputs after the family wrapper validated them.

        The production multimodal executor accepts numeric processor arrays,
        not raw HTTP media bytes. This entrypoint keeps lifecycle tracking on
        that real boundary without pretending the raw-input validators were
        run on an already transformed representation.
        """
        if type(media_types) is not tuple or not media_types or any(
            type(media_type) is not str for media_type in media_types
        ):
            raise ValueError('non-empty immutable processed media type tuple required')
        if len(set(media_types)) != len(media_types):
            raise ValueError('processed media types must be unique')
        if not set(media_types) <= _FAMILY_MEDIA_TYPES[self._family]:
            raise ValueError(f'{self._family} processor media types have no admitted consumer')
        if type(input_byte_count) is not int or input_byte_count <= 0:
            raise ValueError('positive processor media byte count required')
        return self._admit_types(media_types, input_byte_count)

    def _admit_types(self, media_types: Tuple[str, ...], byte_count: int) -> str:
        if (not media_types or len(set(media_types)) != len(media_types) or
                not set(media_types) <= _FAMILY_MEDIA_TYPES[self._family]):
            raise ValueError(f'{self._family} media types are not admitted by its family contract')
        with self._lock:
            if self._closed:
                raise RuntimeError('media lifecycle is closed')
            active = sum(1 for f in self._frames.values()
                         if f.state in ('admitted', 'streaming'))
            if active >= self._max_frames:
                raise MemoryError(
                    f'{self._family} media lifecycle: max concurrent frames '
                    f'({self._max_frames}) reached; close an active frame first')
            frame_id = 'media-' + uuid.uuid4().hex
            self._frames[frame_id] = MediaFrame(
                frame_id=frame_id,
                family=self._family,
                media_types=media_types,
                input_byte_count=byte_count,
            )
            return frame_id

    def on_stream_chunk(self, frame_id: str) -> None:
        """Record one processed media chunk during streaming."""
        with self._lock:
            frame = self._require_frame(frame_id)
            frame._transition('streaming')
            frame.stream_chunks += 1

    def complete(self, frame_id: str) -> MediaFrame:
        """Success path: mark frame complete and return its record."""
        with self._lock:
            frame = self._require_frame(frame_id)
            frame._transition('complete')
            return replace(frame)

    def cancel(self, frame_id: str, reason: str = 'cancelled') -> MediaFrame:
        """Caller-driven cancellation before completion."""
        if not isinstance(reason, str) or not reason:
            reason = 'cancelled'
        with self._lock:
            frame = self._require_frame(frame_id)
            frame._transition('cancelled')
            if reason != 'cancelled':
                frame.error_note = reason[:256]
            return replace(frame)

    def error(self, frame_id: str, exc: BaseException) -> MediaFrame:
        """Emergency cleanup on backend exception."""
        with self._lock:
            frame = self._require_frame(frame_id)
            frame._transition('error')
            try:
                frame.error_note = type(exc).__name__ + ': ' + str(exc)[:200]
            except BaseException:
                frame.error_note = type(exc).__name__
            add_failure_note(
                exc, f'{self._family} media frame {frame_id} error: {frame.error_note}')
            return replace(frame)

    def release(self, frame_id: str) -> None:
        """Remove a completed/cancelled/error frame from the registry."""
        with self._lock:
            frame = self._frames.get(frame_id)
            if frame is None:
                return
            if frame.state in ('admitted', 'streaming'):
                raise RuntimeError(
                    f'cannot release active media frame {frame_id!r} '
                    f'(state={frame.state!r}); cancel or complete it first')
            del self._frames[frame_id]

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for f in self._frames.values()
                       if f.state in ('admitted', 'streaming'))

    def close(self) -> None:
        """Cancel all active frames and close the lifecycle."""
        with self._lock:
            self._closed = True
            for frame in list(self._frames.values()):
                if frame.state in ('admitted', 'streaming'):
                    try:
                        frame._transition('cancelled')
                        frame.error_note = 'lifecycle closed'
                    except Exception as exc:
                        frame.error_note = (
                            'lifecycle close cancellation failed: '
                            + failure_summary(exc))[:256]

    def _require_frame(self, frame_id: str) -> MediaFrame:
        frame = self._frames.get(frame_id)
        if frame is None:
            raise KeyError(f'media frame {frame_id!r} not found in {self._family} lifecycle')
        return frame


# ---------------------------------------------------------------------------
# Registry — one per serving process
# ---------------------------------------------------------------------------

class FamilyMediaLifecycleRegistry:
    """Registry of per-family lifecycle managers.

    Callers look up the handler for a family before admitting media.  A missing
    family key means media is not admitted for that family on this host.
    """

    def __init__(self, *, max_frames_per_family: int = 8):
        if not isinstance(max_frames_per_family, int) or not (1 <= max_frames_per_family <= 64):
            raise ValueError('1–64 concurrent frames per family required')
        self._max = max_frames_per_family
        self._lock = threading.Lock()
        self._handlers: Dict[str, FamilyMediaLifecycle] = {}
        self._closed = False

    def enable(self, family: str) -> FamilyMediaLifecycle:
        """Enable and return the lifecycle handler for a family."""
        with self._lock:
            if self._closed:
                raise RuntimeError('lifecycle registry is closed')
            if family not in self._handlers:
                self._handlers[family] = FamilyMediaLifecycle(
                    family, max_concurrent_frames=self._max)
            return self._handlers[family]

    def get(self, family: str) -> 'FamilyMediaLifecycle | None':
        """Return the handler for a family, or None if not enabled."""
        with self._lock:
            return self._handlers.get(family)

    def require(self, family: str) -> FamilyMediaLifecycle:
        """Return the handler, raising RuntimeError if not enabled."""
        handler = self.get(family)
        if handler is None:
            raise RuntimeError(
                f'no media lifecycle handler registered for family {family!r}; '
                f'call enable({family!r}) at service startup')
        return handler

    def close(self) -> None:
        """Close all registered handlers."""
        with self._lock:
            self._closed = True
            failures = []
            for handler in self._handlers.values():
                try:
                    handler.close()
                except BaseException as exc:
                    failures.append((handler.family, exc))
            if failures:
                primary = failures[0][1]
                for family, failure in failures[1:]:
                    add_failure_note(
                        primary, f'media lifecycle {family} close also failed: '
                        + failure_summary(failure))
                raise primary
            self._handlers.clear()

    def active_counts(self) -> dict:
        with self._lock:
            return {fam: h.active_count() for fam, h in self._handlers.items()}
