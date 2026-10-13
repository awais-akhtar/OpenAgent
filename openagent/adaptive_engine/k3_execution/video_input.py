"""Explicit decoded-frame K3 video inputs with bounded MoonViT3d packing.

The pinned image processor defines no video sampling policy. Callers provide
selected decoded frames and their timestamps; this module preserves selection
and applies the existing patch14/merge2 image transformation to each frame.
"""
from dataclasses import dataclass
import math

import numpy as np

from .preprocessing import preprocess_images


def _validate_frame_times(timestamps):
    """Require uniform sampling because the K3 tower encodes frame indices.

    K3's available temporal position path is index-based (0..T-1); elapsed
    seconds are provenance and are not consumed by the tower.  Reject irregular
    sampling instead of silently pretending that unequal gaps are uniform.
    """
    if not timestamps:
        return
    previous = -1.0
    for timestamp in timestamps:
        if (type(timestamp) not in (float, int) or isinstance(timestamp, bool) or
                not math.isfinite(timestamp) or timestamp < 0 or timestamp <= previous):
            raise ValueError('finite nonnegative strictly increasing frame timestamps required')
        previous = timestamp
    if len(timestamps) > 2:
        intervals = [float(b) - float(a) for a, b in zip(timestamps, timestamps[1:])]
        nominal = intervals[0]
        tolerance = max(1e-9, abs(nominal) * 1e-6)
        if any(abs(interval - nominal) > tolerance for interval in intervals[1:]):
            raise ValueError('K3 video currently requires uniformly sampled timestamps; temporal RoPE uses frame indices')


@dataclass(frozen=True)
class K3VideoInput:
    frames: tuple
    timestamps: tuple
    selection_identity: str

    def __post_init__(self):
        if (type(self.frames) is not tuple or not 1 <= len(self.frames) <= 4 or
                type(self.timestamps) is not tuple or len(self.timestamps) != len(self.frames)):
            raise ValueError('one to four explicitly selected frames and timestamps required')
        if (type(self.selection_identity) is not str or not self.selection_identity or
                len(self.selection_identity.encode('utf-8')) > 4096):
            raise ValueError('bounded explicit video selection identity required')
        _validate_frame_times(self.timestamps)
        owned = []
        dimensions = []
        for frame in self.frames:
            width, height = _dimensions(frame)
            if width <= 0 or height <= 0:
                raise ValueError('selected video frame dimensions must be positive')
            dimensions.append((width, height))
            if isinstance(frame, np.ndarray):
                source = frame[..., :3] if frame.shape[2] == 4 else frame
                copied = np.array(source, dtype=np.uint8, order='C', copy=True)
                copied.flags.writeable = False
                owned.append(copied)
            else:
                # PIL images remain mutable even when copied.  Publish an
                # owned immutable RGB buffer so later caller mutations cannot
                # alter the admitted temporal sample.
                copied = np.array(frame.convert('RGB'), dtype=np.uint8,
                                  order='C', copy=True)
                copied.flags.writeable = False
                owned.append(copied)
        if any(value != dimensions[0] for value in dimensions[1:]):
            raise ValueError('selected K3 video frames must share one spatial geometry')
        object.__setattr__(self, 'frames', tuple(owned))


@dataclass(frozen=True)
class K3PreparedMedia:
    pixels: object
    grids: object
    timestamps: tuple
    selection_identities: tuple

    def __post_init__(self):
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
                _validate_frame_times(temporal)
            elif frames != 1:
                raise ValueError('multi-frame temporal grid requires timestamps')
            expected += frames * int(grids[index, 1]) * int(grids[index, 2])
        if expected != pixels.shape[0]:
            raise ValueError('packed media extent differs from temporal grid inventory')
        if pixels.flags.writeable:
            pixels.flags.writeable = False
        if grids.flags.writeable:
            grids.flags.writeable = False


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
    # Detach caller-owned decoded frames before any budgeted preprocessing.
    # The selected order and timestamp metadata remain unchanged, while the
    # execution owner receives stable source buffers for the whole admission.
    owned = []
    for item in items:
        if type(item) is K3VideoInput:
            frames = []
            for frame in item.frames:
                if isinstance(frame, np.ndarray):
                    frames.append(np.array(frame, dtype=np.uint8, order='C', copy=True))
                else:
                    frames.append(frame.copy())
            owned.append(K3VideoInput(tuple(frames), item.timestamps, item.selection_identity))
        else:
            if isinstance(item, np.ndarray):
                source = item[..., :3] if item.ndim == 3 and item.shape[2] == 4 else item
                owned.append(np.array(source, dtype=np.uint8, order='C', copy=True))
            else:
                copier = getattr(item, 'copy', None)
                owned.append(copier() if callable(copier) else item)
    items = tuple(owned)
    def boundary():
        if cancelled is not None and cancelled():
            raise InterruptedError('K3 media preparation cancelled')
    banks, grids, times, identities = [], [], [], []
    decoded_bytes=0
    for item in items:
        boundary()
        frames=item.frames if type(item) is K3VideoInput else (item,)
        for frame in frames:
            width,height=_dimensions(frame)
            if width<=0 or height<=0:raise ValueError('positive decoded frame dimensions required')
            decoded_bytes+=width*height*16
            if decoded_bytes+len(items)*128>=max_bytes:
                raise MemoryError('complete decoded media inventory exceeds admission')
    used = 0
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
    pixels.flags.writeable = False
    grid.flags.writeable = False
    boundary()
    return K3PreparedMedia(pixels, grid, tuple(times), tuple(identities))


__all__ = ['K3VideoInput', 'K3PreparedMedia', 'prepare_media']
