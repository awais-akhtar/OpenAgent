"""Explicit selected-frame inputs; no implicit sampling or timestamp policy."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PublisherVideoInput:
    frames: tuple
    frames_indices: tuple
    total_num_frames: int
    fps: float

    def __post_init__(self):
        if (type(self.frames) is not tuple or not 1<=len(self.frames)<=4096
                or type(self.frames_indices) is not tuple or len(self.frames_indices)!=len(self.frames)
                or type(self.total_num_frames) is not int or isinstance(self.total_num_frames, bool) or
                not 1<=self.total_num_frames<=2**63-1
                or type(self.fps) not in (int,float) or isinstance(self.fps, bool) or
                not math.isfinite(self.fps) or self.fps<=0
                or any(type(index) is not int or isinstance(index, bool) or
                       not 0<=index<self.total_num_frames for index in self.frames_indices)
                or any(left>=right for left,right in zip(self.frames_indices,self.frames_indices[1:]))):
            raise ValueError('explicit ordered selected frames, frame indices, total count and FPS required')
        if any(frame is None or not (
                (hasattr(frame, 'shape') and hasattr(frame, 'dtype')) or
                (hasattr(frame, 'size') and isinstance(frame.size, tuple)))
               for frame in self.frames):
            raise ValueError('selected video frames must be decoded image objects')
        if any(getattr(frame, 'n_frames', 1) != 1 for frame in self.frames):
            raise ValueError('selected video frames must be detached from animated containers')
        dimensions = []
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
                copied = np.array(frame.convert('RGB'), dtype=np.uint8,
                                  order='C', copy=True)
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


__all__=['PublisherVideoInput']
