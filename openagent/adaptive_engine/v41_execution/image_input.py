"""Bounded local image preparation for the V4.1 patch and token-span contract."""
from dataclasses import dataclass
import io
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PublisherImageInput:
    start: int
    patches: object
    n_vit_h: int
    n_vit_w: int
    types: object


class PublisherImageProcessor:
    def __init__(self, *, patch_size, downsample_ratio, max_tokens,
                 min_pixels, max_width_height_ratio, max_bytes, cast):
        for value in (patch_size, downsample_ratio, max_tokens, max_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError('positive image geometry and byte budget required')
        if max_tokens < 4 or type(min_pixels) is not int or min_pixels < 0:
            raise ValueError('at least four image tokens and nonnegative minimum pixels required')
        if (max_width_height_ratio is not None and
                (type(max_width_height_ratio) not in (int, float) or
                 not math.isfinite(max_width_height_ratio) or max_width_height_ratio <= 0)):
            raise ValueError('positive finite aspect limit or None required')
        if not callable(cast):
            raise ValueError('publisher activation cast required')
        self.patch = patch_size
        self.downsample = downsample_ratio
        self.max_tokens, self.min_pixels = max_tokens, min_pixels
        self.aspect_limit, self.max_bytes, self.cast = max_width_height_ratio, max_bytes, cast

    def plan(self, width, height):
        if any(type(v) is not int or v <= 0 for v in (width, height)):
            raise ValueError('positive integer source dimensions required')
        w, h = width, height
        if self.aspect_limit is not None:
            w = min(w, h*self.aspect_limit)
        if w*h < self.min_pixels:
            scale = math.sqrt(self.min_pixels/(w*h))
            w, h = int(w*scale), int(h*scale)
        p, r = self.patch, self.downsample
        target_w, target_h = math.ceil(w/p)*p, math.ceil(h/p)*p
        rows, columns = math.ceil(target_h/(p*r)), math.ceil(target_w/(p*r))
        if rows*(columns+1)+2 > self.max_tokens:
            aspect = h/w
            cells_w = (math.sqrt(1+4*(self.max_tokens-2)/aspect)-1)/2
            cells_h = cells_w*aspect
            if cells_w < 1:
                target_h, target_w = ((self.max_tokens-2)//2)*p*r, p*r
            elif cells_h < 1:
                target_h, target_w = p*r, (self.max_tokens-3)*p*r
            else:
                scale = min(math.floor(cells_w)*p*r/w, math.floor(cells_h)*p*r/h)
                target_h, target_w = math.floor(h*scale/p)*p, math.floor(w*scale/p)*p
            rows, columns = math.ceil(target_h/(p*r)), math.ceil(target_w/(p*r))
        if min(target_h, target_w) <= 0 or rows*(columns+1)+2 > self.max_tokens:
            raise ValueError('no valid admitted image grid')
        return target_h, target_w, rows, columns

    def prepare(self, source, *, start):
        from PIL import Image, ImageOps
        if type(start) is not int or start < 0:
            raise ValueError('nonnegative image-span position required')
        decoded=isinstance(source,(np.ndarray,Image.Image))
        if decoded:
            if isinstance(source,np.ndarray):
                if source.dtype!=np.uint8 or source.ndim!=3 or source.shape[2]!=3:
                    raise ValueError('decoded publisher image must be uint8 RGB HWC')
                height,width=source.shape[:2]
            else:
                if getattr(source, 'n_frames', 1) != 1:
                    raise ValueError('animated publisher images require explicit frame admission')
                width,height=source.size
            if width<=0 or height<=0 or width*height*16>self.max_bytes:
                raise MemoryError('decoded publisher image exceeds input admission')
            opened_source=Image.fromarray(source) if isinstance(source,np.ndarray) else source.copy()
            encoded=b''
        elif isinstance(source, bytes):
            encoded = source
        else:
            path = Path(source).resolve(strict=True)
            if not path.is_file() or path.stat().st_size > self.max_bytes:
                raise MemoryError('local image file exceeds admission')
            with path.open('rb') as stream:
                encoded = stream.read(self.max_bytes+1)
        if not decoded and (not encoded or len(encoded) > self.max_bytes):
            raise MemoryError('image bytes exceed admission')
        with (opened_source if decoded else Image.open(io.BytesIO(encoded))) as opened:
            if getattr(opened, 'n_frames', 1) != 1:
                raise ValueError('still image input required; animated media needs explicit frame handling')
            original_w, original_h = opened.size
            h, w, rows, columns = self.plan(original_w, original_h)
            if len(encoded)+16*original_w*original_h+48*w*h > self.max_bytes:
                raise MemoryError('decoded image and patch workspace exceed admission')
            image = opened.convert('RGB')
            if self.aspect_limit is not None and original_w >= self.aspect_limit*original_h:
                resized = image.resize((w, h), resample=Image.Resampling.BICUBIC)
            else:
                resized = ImageOps.pad(image, (w, h), method=Image.Resampling.BICUBIC,
                    color=(127, 127, 127))
            try:
                pixels = np.asarray(resized, dtype=np.float32)
                normalized = (pixels/255.0-0.5)/0.5
                rounded = np.asarray(self.cast(normalized.reshape(-1)), dtype=np.float32)
                if rounded.size != normalized.size or not np.isfinite(rounded).all():
                    raise ValueError('invalid publisher image activation cast')
                p = self.patch
                vh, vw = h//p, w//p
                chw = rounded.reshape(h, w, 3).transpose(2, 0, 1)
                patches = np.ascontiguousarray(chw.reshape(3, vh, p, vw, p)
                    .transpose(1, 3, 0, 2, 4).reshape(vh*vw, 3*p*p))
            finally:
                resized.close()
                image.close()
        kinds = np.empty(rows*(columns+1)+2, dtype=np.int64)
        kinds[0], kinds[-1] = 0, 3
        grid = kinds[1:-1].reshape(rows, columns+1)
        grid[:, :-1], grid[:, -1] = 1, 2
        patches.flags.writeable = False
        kinds.flags.writeable = False
        return PublisherImageInput(start, patches, vh, vw, kinds)

    def expand(self, tokens, images, *, image_token_id):
        if type(image_token_id) is not int or image_token_id < 0:
            raise ValueError('configured image placeholder token required')
        ids, sources = tuple(tokens), tuple(images)
        if any(type(t) is not int or t < 0 for t in ids):
            raise ValueError('nonnegative integer prompt tokens required')
        if ids.count(image_token_id) != len(sources):
            raise ValueError('image placeholder count differs from local image inputs')
        result, types, prepared = [], [], []
        source_index, used = 0, 0
        for token in ids:
            if token != image_token_id:
                result.append(token)
                types.append(-1)
                continue
            image = self.prepare(sources[source_index], start=len(result))
            source_index += 1
            used += image.patches.nbytes+image.types.nbytes
            if used > self.max_bytes:
                raise MemoryError('combined prepared images exceed admission')
            prepared.append(image)
            result.extend([image_token_id]*len(image.types))
            types.extend(int(t) for t in image.types)
        return result, types, tuple(prepared)
