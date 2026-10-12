"""Pinned K3 still-image preprocessing, without importing publisher code."""
import math

import numpy as np


def preprocess_images(images, *, max_bytes=256 * 1024 * 1024, cancelled=None):
    """Return FP32 MoonViT patches and int64 grids from PIL images/uint8 RGB(A).

    Pinned preprocessor_config: patch14, merge2, 65536 input patches, side512,
    mean/std .5, bicubic resize, transparency chessboard after resize.
    The official Python processor only accepts still images; video sampling is
    not specified by its executable path, despite video fields in the config.
    """
    from PIL import Image
    if type(max_bytes) is not int or max_bytes <= 0 or not isinstance(images, (list, tuple)) or not 0 < len(images) <= 4096:
        raise ValueError("nonempty image sequence and positive byte budget required")
    if cancelled is not None and not callable(cancelled):
        raise TypeError('callable image preprocessing cancellation required')
    def boundary():
        if cancelled is not None and cancelled():
            raise InterruptedError('K3 image preprocessing cancelled')
    batches, grids, allocated = [], [], 0
    def geometry(image):
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in (3, 4):
                raise ValueError("raw images must be uint8 RGB/RGBA")
            width,height=image.shape[1],image.shape[0]
        elif isinstance(image, Image.Image):
            if getattr(image,'is_animated',False) or getattr(image,'n_frames',1)!=1:
                raise ValueError('explicit decoded still image required; detach selected video frames')
            width,height=image.size
        else:
            raise ValueError("PIL image or uint8 RGB/RGBA array required")
        if width <= 0 or height <= 0:
            raise ValueError('positive decoded image dimensions required')
        return width,height
    decoded_bytes=0
    for image in images:
        boundary()
        width,height=geometry(image)
        decoded_bytes+=16*width*height
        if decoded_bytes+len(images)*128>=max_bytes:
            raise MemoryError('complete decoded image inventory exceeds admission')
    def prepare(image, available):
        boundary()
        width,height=geometry(image)
        scale = min(1.0, math.sqrt(65536 / (max(1.0, width // 14) * max(1.0, height // 14))),
                    512 * 14 / width, 512 * 14 / height)
        nw, nh = min(max(1, int(width * scale)), 512 * 14), min(max(1, int(height * scale)), 512 * 14)
        pw, ph = (-nw) % 28, (-nh) % 28
        required = 64 * (nw + pw) * (nh + ph) + 16 * width * height
        if required > available:
            raise ValueError("image preprocessing budget exceeded")
        if isinstance(image,np.ndarray):
            image=Image.fromarray(image)
        boundary()
        resized = image.resize((nw, nh), resample=Image.Resampling.BICUBIC)
        boundary()
        alpha_present = "A" in resized.getbands() or "transparency" in resized.info
        if alpha_present:
            rgba = np.asarray(resized.convert("RGBA"))
            yy, xx = np.indices((nh, nw))
            background = np.where(((yy // 8 + xx // 8) % 2)[..., None] == 0, 255, 180).astype(np.uint8)
            alpha = rgba[..., 3:4].astype(np.float32) / 255.0
            rgb = (alpha * rgba[..., :3] + (1 - alpha) * background).astype(np.uint8)
        else:
            rgb = np.asarray(resized.convert("RGB"))
        boundary()
        rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="constant", constant_values=0)
        normalized = (rgb / 255.0).astype(np.float32)
        normalized -= np.array([0.5, 0.5, 0.5])
        normalized *= np.array([2.0, 2.0, 2.0])
        gh, gw = (nh + ph) // 14, (nw + pw) // 14
        patches = normalized.reshape(1, gh, 14, gw, 14, 3).transpose(0, 1, 3, 5, 2, 4)
        patches = np.ascontiguousarray(patches.reshape(-1, 3, 14, 14))
        boundary()
        return patches,(1,gh,gw)
    for image in images:
        width,height=geometry(image)
        patches,grid=prepare(image,max_bytes-allocated-decoded_bytes+16*width*height-len(images)*128)
        batches.append(patches)
        grids.append(grid)
        allocated += patches.nbytes
    if decoded_bytes+2 * allocated+len(images)*128 > max_bytes:
        raise ValueError("patch concatenation budget exceeded")
    boundary()
    result=np.concatenate(batches),np.asarray(grids,dtype=np.int64)
    boundary()
    return result
