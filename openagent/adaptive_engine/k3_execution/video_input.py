"""Explicit decoded-frame K3 video inputs with bounded MoonViT3d packing.

The pinned image processor defines no video sampling policy. Callers provide
selected decoded frames and their timestamps; this module preserves selection
and applies the existing patch14/merge2 image transformation to each frame.
"""
from dataclasses import dataclass
import math

import numpy as np

from .preprocessing import preprocess_images


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
        previous = -1.0
        for timestamp in self.timestamps:
            if (type(timestamp) not in (float, int) or not math.isfinite(timestamp) or
                    timestamp < 0 or timestamp <= previous):
                raise ValueError('finite nonnegative strictly increasing frame timestamps required')
            previous = timestamp


@dataclass(frozen=True)
class K3PreparedMedia:
    pixels: object
    grids: object
    timestamps: tuple
    selection_identities: tuple


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
    return K3PreparedMedia(pixels, grid, tuple(times), tuple(identities))


__all__ = ['K3VideoInput', 'K3PreparedMedia', 'prepare_media']
