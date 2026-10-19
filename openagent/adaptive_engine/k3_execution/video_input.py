"""Explicit decoded-frame K3 video inputs with bounded MoonViT3d packing.

The pinned image processor defines no video sampling policy. Callers provide
selected decoded frames and their timestamps; this module preserves selection
and applies the existing patch14/merge2 image transformation to each frame.
"""
from dataclasses import dataclass
import math
import sys

import numpy as np

from .preprocessing import preprocess_images
from ..failure_reporting import add_failure_note, failure_summary


def validate_frame_timestamps(timestamps):
    """Require uniform sampling because the K3 tower encodes frame indices.

    K3's available temporal position path is index-based (0..T-1); elapsed
    seconds are provenance and are not consumed by the tower.  Reject irregular
    sampling instead of silently pretending that unequal gaps are uniform.
    """
    if not timestamps:
        return
    previous = -1.0
    normalized = []
    for timestamp in timestamps:
        if type(timestamp) not in (float, int) or isinstance(timestamp, bool):
            raise ValueError('finite nonnegative strictly increasing frame timestamps required')
        # `math.isfinite(10**1000)` raises OverflowError instead of returning
        # false. Convert explicitly so hostile/unbounded Python integers fail
        # as ordinary invalid timestamp metadata before interval arithmetic or
        # media allocation begins.
        try:
            value = float(timestamp)
        except (OverflowError, ValueError):
            raise ValueError('finite nonnegative strictly increasing frame timestamps required') from None
        if not math.isfinite(value) or value < 0 or value <= previous:
            raise ValueError('finite nonnegative strictly increasing frame timestamps required')
        normalized.append(value)
        previous = value
    if len(timestamps) > 2:
        intervals = [b - a for a, b in zip(normalized, normalized[1:])]
        nominal = intervals[0]
        tolerance = max(1e-9, abs(nominal) * 1e-6)
        if any(abs(interval - nominal) > tolerance for interval in intervals[1:]):
            raise ValueError('K3 video currently requires uniformly sampled timestamps; temporal RoPE uses frame indices')


@dataclass(frozen=True)
class K3VideoInput:
    frames: tuple
    timestamps: tuple
    selection_identity: str

    @classmethod
    def _from_owned_frames(cls, frames, timestamps, selection_identity):
        """Adopt detached private RGB frames without a second full-frame copy.

        This is reserved for the serving gateway after its bounded,
        cancellation-aware media freeze. The public constructor remains the
        ownership boundary for borrowed caller frames.
        """
        if (type(frames) is not tuple or not 1 <= len(frames) <= 4 or
                type(timestamps) is not tuple or len(timestamps) != len(frames) or
                type(selection_identity) is not str or not selection_identity or
                len(selection_identity.encode('utf-8')) > 4096):
            raise ValueError('one to four owned frames and bounded K3 selection metadata required')
        validate_frame_timestamps(timestamps)
        dimensions = []
        admitted_bytes = 0
        for frame in frames:
            if (type(frame) is not np.ndarray or frame.dtype != np.uint8 or
                    frame.ndim != 3 or frame.shape[2] != 3 or
                    any(type(value) is not int or value <= 0 for value in frame.shape[:2]) or
                    not frame.flags.owndata or frame.base is not None or frame.flags.writeable):
                raise ValueError('adopted K3 frames must be immutable detached uint8 RGB arrays')
            dimensions.append((frame.shape[1], frame.shape[0]))
            admitted_bytes += frame.shape[0] * frame.shape[1] * 16
            if admitted_bytes > 256 * 1024**2:
                raise MemoryError('K3 selected-frame copies exceed detached-media admission')
        if any(value != dimensions[0] for value in dimensions[1:]):
            raise ValueError('selected K3 video frames must share one spatial geometry')
        instance = object.__new__(cls)
        object.__setattr__(instance, 'frames', frames)
        object.__setattr__(instance, 'timestamps', timestamps)
        object.__setattr__(instance, 'selection_identity', selection_identity)
        return instance

    def __post_init__(self):
        if (type(self.frames) is not tuple or not 1 <= len(self.frames) <= 4 or
                type(self.timestamps) is not tuple or len(self.timestamps) != len(self.frames)):
            raise ValueError('one to four explicitly selected frames and timestamps required')
        if (type(self.selection_identity) is not str or not self.selection_identity or
                len(self.selection_identity.encode('utf-8')) > 4096):
            raise ValueError('bounded explicit video selection identity required')
        validate_frame_timestamps(self.timestamps)
        owned = []
        dimensions = []
        admitted_bytes = 0
        for frame in self.frames:
            width, height = _dimensions(frame)
            if width <= 0 or height <= 0:
                raise ValueError('selected video frame dimensions must be positive')
            dimensions.append((width, height))
            admitted_bytes += width * height * 16
            if admitted_bytes > 256 * 1024**2:
                raise MemoryError('K3 selected-frame copies exceed detached-media admission')
        if any(value != dimensions[0] for value in dimensions[1:]):
            raise ValueError('selected K3 video frames must share one spatial geometry')
        # Validate every frame and the aggregate allocation before detaching
        # any of them. A late geometry mismatch must not force copies of the
        # preceding full-resolution frames merely to reject the request.
        for frame in self.frames:
            if isinstance(frame, np.ndarray):
                source = frame[..., :3] if frame.shape[2] == 4 else frame
                copied = np.array(source, dtype=np.uint8, order='C', copy=True)
            else:
                # PIL images remain mutable even when copied.  Publish an
                # owned immutable RGB buffer so later caller mutations cannot
                # alter the admitted temporal sample.
                converted = frame.convert('RGB')
                try:
                    copied = np.array(converted, dtype=np.uint8,
                                      order='C', copy=True)
                except BaseException as primary:
                    close = getattr(converted, 'close', None)
                    if callable(close):
                        try:
                            close()
                        except BaseException as cleanup:
                            add_failure_note(
                                primary, 'temporary K3 RGB conversion cleanup failed: ' +
                                failure_summary(cleanup))
                    raise
                close = getattr(converted, 'close', None)
                if callable(close):
                    close()
            # Clearing WRITEABLE on an owning ndarray is reversible by any
            # holder of the published frame. Back the public snapshot with
            # immutable bytes instead, so validated temporal media cannot be
            # changed between admission and preprocessing.
            frozen = np.frombuffer(copied.tobytes(order='C'), dtype=np.uint8).reshape(copied.shape)
            owned.append(frozen)
        object.__setattr__(self, 'frames', tuple(owned))


@dataclass(frozen=True)
class K3PreparedMedia:
    pixels: object
    grids: object
    timestamps: tuple
    selection_identities: tuple

    def __post_init__(self):
        # This value crosses the media-preparation boundary and is treated as
        # execution-owned immutable storage.  ``asarray`` alone can preserve a
        # caller's view (or memmap) and flipping its writeable flag below would
        # neither detach the bytes nor prevent the original owner from
        # mutating them.  Require owning buffers instead; prepare_media builds
        # both arrays with concatenate/asarray and therefore satisfies this
        # contract without another full-size copy.
        if (not isinstance(self.pixels, np.ndarray) or
                not isinstance(self.grids, np.ndarray) or
                not self.pixels.flags.owndata or self.pixels.base is not None or
                not self.grids.flags.owndata or self.grids.base is not None):
            raise ValueError('prepared K3 media requires detached owning pixel and grid buffers')
        pixels = np.asarray(self.pixels)
        grids = np.asarray(self.grids)
        if (pixels.dtype != np.float32 or pixels.ndim != 4 or
                pixels.shape[1:] != (3, 14, 14) or not pixels.size or
                not np.isfinite(pixels).all() or grids.dtype != np.int64 or
                grids.ndim != 2 or grids.shape[1] != 3 or
                grids.shape[0] == 0 or not np.all(grids > 0)):
            raise ValueError('complete finite packed media and temporal grids required')
        if (type(self.timestamps) is not tuple or
                type(self.selection_identities) is not tuple or
                len(self.timestamps) != grids.shape[0] or
                len(self.selection_identities) != grids.shape[0]):
            raise ValueError('packed media metadata must match grid inventory')
        expected = 0
        for index, (temporal, identity) in enumerate(
                zip(self.timestamps, self.selection_identities)):
            frames = int(grids[index, 0])
            if (type(identity) is not str or not identity or
                    len(identity.encode('utf-8')) > 4096 or
                    type(temporal) is not tuple):
                raise ValueError('bounded media selection identity and timestamps required')
            if temporal:
                if len(temporal) != frames:
                    raise ValueError('video timestamp extent differs from temporal grid')
                validate_frame_timestamps(temporal)
            elif frames != 1:
                raise ValueError('multi-frame temporal grid requires timestamps')
            expected += frames * int(grids[index, 1]) * int(grids[index, 2])
        if expected != pixels.shape[0]:
            raise ValueError('packed media extent differs from temporal grid inventory')
        # A NumPy owner can re-enable WRITEABLE after it has merely been
        # cleared.  The frozen dataclass must not expose buffers that a caller
        # can mutate during the subsequent tower pass, so detach them onto
        # immutable bytes-backed storage before publishing this receipt.
        frozen_pixels = np.frombuffer(pixels.tobytes(order='C'), dtype=pixels.dtype).reshape(pixels.shape)
        frozen_grids = np.frombuffer(grids.tobytes(order='C'), dtype=grids.dtype).reshape(grids.shape)
        object.__setattr__(self, 'pixels', frozen_pixels)
        object.__setattr__(self, 'grids', frozen_grids)


def _dimensions(frame):
    if isinstance(frame, np.ndarray):
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise ValueError('video frames must be uint8 RGB/RGBA')
        return frame.shape[1], frame.shape[0]
    from PIL import Image
    if not isinstance(frame, Image.Image):
        raise TypeError('decoded PIL or uint8 video frames required')
    if getattr(frame,'is_animated',False) or getattr(frame,'n_frames',1)!=1:
        raise ValueError('selected video frame must be detached from its animated container')
    return frame.size


def _pack_owned_media(items, *, decoded_bytes, max_bytes, cancelled, boundary):
    banks, grids, times, identities = [], [], [], []
    used = 0
    pixels = grid = None
    try:
        for item in items:
            boundary()
            frames = item.frames if type(item) is K3VideoInput else (item,)
            dimensions = tuple(_dimensions(frame) for frame in frames)
            if any(d != dimensions[0] for d in dimensions) or min(dimensions[0]) <= 0:
                raise ValueError('all selected video frames require identical positive dimensions')
            spatial = None
            for frame in frames:
                boundary()
                remaining = max_bytes-decoded_bytes-used-len(items)*128
                if remaining <= 0:
                    raise MemoryError('combined K3 media exceeds admission')
                pixels, grid = preprocess_images((frame,), max_bytes=remaining, cancelled=cancelled)
                shape = tuple(int(v) for v in grid[0, 1:])
                if spatial is not None and shape != spatial:
                    raise ValueError('video preprocessing produced inconsistent spatial grids')
                spatial = shape
                used += pixels.nbytes
                banks.append(pixels)
            grids.append((len(frames), *spatial))
            times.append(item.timestamps if type(item) is K3VideoInput else ())
            identities.append(item.selection_identity if type(item) is K3VideoInput else 'k3-still-patch14-merge2-v1')
        if decoded_bytes+2*used+len(grids)*128 > max_bytes:
            raise MemoryError('combined media publication exceeds admission')
        boundary()
        pixels = np.concatenate(banks, axis=0)
        grid = np.asarray(grids, dtype=np.int64)
        # The prepared receipt snapshots onto immutable bytes-backed buffers.
        # Release per-frame patch banks first so the admitted peak remains
        # bounded by decoded input plus the concatenated and published banks.
        banks.clear()
        boundary()
        return K3PreparedMedia(pixels, grid, tuple(times), tuple(identities))
    except BaseException:
        # Exception tracebacks keep this frame alive. Drop all partial patch
        # banks and a completed-but-unpublished concatenation now, especially
        # on cancellation between video items, instead of waiting for GC.
        banks.clear();grids.clear();times.clear();identities.clear()
        pixels=None;grid=None;items=();frames=();item=None;frame=None
        raise


def _close_owned_images(images, primary=None):
    cleanup_errors = []
    for image in images:
        try:
            image.close()
        except BaseException as cleanup:
            cleanup_errors.append(cleanup)
    if cleanup_errors:
        if primary is not None:
            add_failure_note(primary, 'owned K3 media image cleanup failed: ' +
                             '; '.join(failure_summary(error) for error in cleanup_errors))
        else:
            first = cleanup_errors[0]
            if len(cleanup_errors) > 1:
                add_failure_note(first, 'additional owned K3 media image cleanup failed: ' +
                                 '; '.join(failure_summary(error)
                                           for error in cleanup_errors[1:]))
            raise first


def prepare_media(items, *, max_bytes, cancelled=None):
    """Pack mixed still images and selected videos in placeholder order.

    Frame-major patch order matches the tower's [time,height,width] layout.
    Each video's spatial geometry must be constant before and after resizing.
    Timestamp metadata stays attached for application prompt construction.
    """
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('positive media preprocessing budget required')
    if not isinstance(items, (tuple, list)) or not 0 < len(items) <= 4096:
        raise ValueError('bounded nonempty media sequence required')
    if cancelled is not None and not callable(cancelled):
        raise TypeError('callable media cancellation required')
    def boundary():
        if cancelled is not None and cancelled():
            raise InterruptedError('K3 media preparation cancelled')

    # Bound decoded inputs before detaching them.  Previously the full
    # caller-owned inventory was copied here and only charged against
    # max_bytes in the later preprocessing loop, allowing oversized media to
    # allocate its entire duplicate before admission failed.
    decoded_bytes = 0
    for item in items:
        boundary()
        frames = item.frames if type(item) is K3VideoInput else (item,)
        for frame in frames:
            boundary()
            width, height = _dimensions(frame)
            if width <= 0 or height <= 0:
                raise ValueError('positive decoded frame dimensions required')
            decoded_bytes += width * height * 16
            if decoded_bytes + len(items) * 128 >= max_bytes:
                raise MemoryError('complete decoded media inventory exceeds admission')

    # Detach caller-owned decoded frames before preprocessing. The selected
    # order and timestamp metadata remain unchanged, while the execution owner
    # receives stable source buffers for the whole admission.
    owned = []
    owned_images = []
    try:
        for item in items:
            boundary()
            if type(item) is K3VideoInput:
                frames = []
                for frame in item.frames:
                    boundary()
                    if isinstance(frame, np.ndarray):
                        frames.append(np.array(frame, dtype=np.uint8, order='C', copy=True))
                    else:
                        copied = frame.copy()
                        frames.append(copied)
                        # PIL copies are temporary execution-owned objects.
                        # Track each immediately so cancellation or a later
                        # frame-copy failure closes every owner already made.
                        close = getattr(copied, 'close', None)
                        if copied is not frame and callable(close):
                            owned_images.append(copied)
                owned.append(K3VideoInput(tuple(frames), item.timestamps, item.selection_identity))
            else:
                boundary()
                if isinstance(item, np.ndarray):
                    source = item[..., :3] if item.ndim == 3 and item.shape[2] == 4 else item
                    owned.append(np.array(source, dtype=np.uint8, order='C', copy=True))
                else:
                    copier = getattr(item, 'copy', None)
                    copied = copier() if callable(copier) else item
                    owned.append(copied)
                    if copied is not item and callable(getattr(copied, 'close', None)):
                        owned_images.append(copied)
    except BaseException as primary:
        _close_owned_images(owned_images,primary)
        raise
    items = tuple(owned)
    try:
        return _pack_owned_media(items,decoded_bytes=decoded_bytes,max_bytes=max_bytes,
                                 cancelled=cancelled,boundary=boundary)
    finally:
        # The packed pixel bank is the only media object that crosses this
        # boundary. Clear detached decoded frames even when the caller keeps
        # an exception (and its traceback) for diagnostics.
        items=()
        owned.clear()
        _close_owned_images(owned_images,sys.exc_info()[1])


__all__ = ['K3VideoInput', 'K3PreparedMedia', 'prepare_media', 'validate_frame_timestamps']
