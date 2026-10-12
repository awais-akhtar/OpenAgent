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
                or type(self.total_num_frames) is not int or not 1<=self.total_num_frames<=2**63-1
                or type(self.fps) not in (int,float) or not math.isfinite(self.fps) or self.fps<=0
                or any(type(index) is not int or not 0<=index<self.total_num_frames for index in self.frames_indices)
                or any(left>=right for left,right in zip(self.frames_indices,self.frames_indices[1:]))):
            raise ValueError('explicit ordered selected frames, frame indices, total count and FPS required')

    def metadata(self):
        return dict(frames_indices=list(self.frames_indices),total_num_frames=self.total_num_frames,fps=self.fps)


__all__=['PublisherVideoInput']
