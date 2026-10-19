"""Explicit selected-frame inputs; no implicit sampling or timestamp policy."""
from dataclasses import dataclass
import math

from .failure_reporting import add_failure_note, failure_summary


def normalize_video_fps(value):
    """Return a canonical finite positive FPS value for selected-frame media."""
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError('finite positive FPS required')
    try:
        normalized = float(value)
    except (OverflowError, ValueError):
        raise ValueError('finite positive FPS required') from None
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError('finite positive FPS required')
    return normalized


@dataclass(frozen=True)
class PublisherVideoInput:
    frames: tuple
    frames_indices: tuple
    total_num_frames: int
    fps: float

    @classmethod
    def _from_owned_frames(cls, frames, frames_indices, total_num_frames, fps):
        """Adopt private detached RGB frames without another opaque copy.

        This is reserved for the gateway's cancellation-aware media freezer.
        The public constructor remains the boundary for borrowed caller data.
        """
        fps = normalize_video_fps(fps)
        import numpy as np
        if (type(frames) is not tuple or not 1 <= len(frames) <= 4096 or
                type(frames_indices) is not tuple or len(frames_indices) != len(frames) or
                type(total_num_frames) is not int or not 1 <= total_num_frames <= 2**63-1 or
                any(type(index) is not int or not 0 <= index < total_num_frames
                    for index in frames_indices) or
                any(left >= right for left, right in zip(frames_indices, frames_indices[1:]))):
            raise ValueError('explicit ordered owned frames and temporal metadata required')
        dimensions = []
        total_bytes = 0
        for frame in frames:
            if (type(frame) is not np.ndarray or frame.dtype != np.uint8 or
                    frame.ndim != 3 or frame.shape[2] != 3 or
                    any(type(value) is not int or value <= 0 for value in frame.shape[:2]) or
                    frame.flags.writeable):
                raise ValueError('adopted frames must be immutable detached uint8 RGB arrays')
            dimensions.append((frame.shape[1], frame.shape[0]))
            total_bytes += frame.shape[0] * frame.shape[1] * 16
            if total_bytes > 256 * 1024**2:
                raise MemoryError('publisher selected-frame copies exceed detached-media admission')
        if any(value != dimensions[0] for value in dimensions[1:]):
            raise ValueError('selected video frames must share one spatial geometry')
        instance = object.__new__(cls)
        object.__setattr__(instance, 'frames', frames)
        object.__setattr__(instance, 'frames_indices', frames_indices)
        object.__setattr__(instance, 'total_num_frames', total_num_frames)
        object.__setattr__(instance, 'fps', fps)
        return instance

    def __post_init__(self):
        fps = normalize_video_fps(self.fps)
        if (type(self.frames) is not tuple or not 1<=len(self.frames)<=4096
                or type(self.frames_indices) is not tuple or len(self.frames_indices)!=len(self.frames)
                or type(self.total_num_frames) is not int or isinstance(self.total_num_frames, bool) or
                not 1<=self.total_num_frames<=2**63-1
                or any(type(index) is not int or isinstance(index, bool) or
                       not 0<=index<self.total_num_frames for index in self.frames_indices)
                or any(left>=right for left,right in zip(self.frames_indices,self.frames_indices[1:]))):
            raise ValueError('explicit ordered selected frames, frame indices, total count and FPS required')
        object.__setattr__(self, 'fps', fps)
        if any(frame is None or not (
                (hasattr(frame, 'shape') and hasattr(frame, 'dtype')) or
                (hasattr(frame, 'size') and isinstance(frame.size, tuple)))
               for frame in self.frames):
            raise ValueError('selected video frames must be decoded image objects')
        if any(getattr(frame, 'n_frames', 1) != 1 for frame in self.frames):
            raise ValueError('selected video frames must be detached from animated containers')
        dimensions = []
        admitted_bytes = 0
        for frame in self.frames:
            if hasattr(frame, 'shape') and hasattr(frame, 'dtype'):
                shape = tuple(frame.shape)
                if (frame.dtype != 'uint8' or len(shape) != 3 or shape[2] not in (3, 4) or
                        any(type(value) is not int or value <= 0 for value in shape[:2])):
                    raise ValueError('selected video arrays must be uint8 RGB/RGBA frames')
                dimensions.append((shape[1], shape[0]))
            else:
                size = tuple(frame.size)
                if (len(size) != 2 or
                        any(type(value) is not int or value <= 0 for value in size)):
                    raise ValueError('selected video images must have positive dimensions')
                dimensions.append(size)
            width, height = dimensions[-1]
            admitted_bytes += width * height * 16
            if admitted_bytes > 256 * 1024**2:
                raise MemoryError('publisher selected-frame copies exceed detached-media admission')
        if any(value != dimensions[0] for value in dimensions[1:]):
            raise ValueError('selected video frames must share one spatial geometry')
        owned = []
        for frame in self.frames:
            if hasattr(frame, 'shape') and hasattr(frame, 'dtype'):
                import numpy as np
                source = frame[..., :3] if frame.shape[2] == 4 else frame
                copied = np.array(source, dtype=np.uint8, order='C', copy=True)
                copied.flags.writeable = False
            else:
                import numpy as np
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
                            add_failure_note(primary, 'temporary publisher RGB conversion cleanup failed: '+failure_summary(cleanup))
                    raise
                close = getattr(converted, 'close', None)
                if callable(close):
                    close()
                copied.flags.writeable = False
            owned.append(copied)
        object.__setattr__(self, 'frames', tuple(owned))

    def metadata(self):
        # Keep the transport contract limited to the source-owned fields.  A
        # derived timestamp view is available through ``timestamps`` but must
        # not be serialized into processor metadata: the processor validates
        # this mapping exactly so frame ordering/count/FPS remain the only
        # admitted temporal inputs.
        return dict(frames_indices=list(self.frames_indices),
                    total_num_frames=self.total_num_frames, fps=self.fps)

    @property
    def timestamps(self):
        """Return stable source-time positions without applying sampling policy."""
        return tuple(index / self.fps for index in self.frames_indices)


__all__=['PublisherVideoInput','normalize_video_fps']
