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
    downsample_ratio: int = 1

    @property
    def positions(self):
        """Expanded token positions owned by this immutable image span."""
        return tuple(range(self.start, self.start + int(np.asarray(self.types).size)))

    def __post_init__(self):
        if type(self.start) is not int or self.start < 0:
            raise ValueError('nonnegative image span start required')
        if any(type(value) is not int or value <= 0
               for value in (self.n_vit_h, self.n_vit_w)):
            raise ValueError('positive image patch grid required')
        if type(self.downsample_ratio) is not int or self.downsample_ratio <= 0:
            raise ValueError('positive image aligner ratio required')
        patches = np.asarray(self.patches)
        types = np.asarray(self.types)
        aligned_h = math.ceil(self.n_vit_h/self.downsample_ratio)
        aligned_w = math.ceil(self.n_vit_w/self.downsample_ratio)
        expected = np.empty(aligned_h*(aligned_w+1)+2, dtype=np.int64)
        expected[0], expected[-1] = 0, 3
        expected[1:-1].reshape(aligned_h, aligned_w+1)[:, :-1] = 1
        expected[1:-1].reshape(aligned_h, aligned_w+1)[:, -1] = 2
        if (patches.ndim != 2 or patches.dtype.kind != 'f' or
                patches.shape[0] != self.n_vit_h * self.n_vit_w or
                patches.shape[1] <= 0 or not np.isfinite(patches).all() or
                types.ndim != 1 or types.dtype.kind not in 'iu' or
                not np.array_equal(types, expected) or
                np.any((types < 0) | (types > 3)) or
                int(types[0]) != 0 or int(types[-1]) != 3 or
                np.any(types[1:-1] == 0) or np.any(types[1:-1] == 3)):
            raise ValueError('complete finite image span tensors required')
        # The token span describes the aligner's output grid, while patches
        # retain the pre-aligner vision grid. Keep these geometries distinct.
        # For ratio r, one language token represents an r×r patch group.
        # ``types`` was produced by the processor and its dimensions are
        # validated again by the vision consumer before dispatch.
        # Own the published buffers.  A caller can otherwise retain an
        # immutable view, re-enable writability on its original array, and
        # mutate the span after admission.
        patches = np.array(patches, dtype=patches.dtype, order='C', copy=True)
        types = np.array(types, dtype=types.dtype, order='C', copy=True)
        patches.flags.writeable = False
        types.flags.writeable = False
        object.__setattr__(self, 'patches', patches)
        object.__setattr__(self, 'types', types)


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
        # Patches retain the processor grid; language tokens follow the
        # downsampled aligner grid.
        p, r = self.patch, self.downsample
        target_w, target_h = math.ceil(w/p)*p, math.ceil(h/p)*p
        input_rows, input_columns = math.ceil(target_h/p), math.ceil(target_w/p)
        rows, columns = math.ceil(input_rows/r), math.ceil(input_columns/r)
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
            input_rows, input_columns = math.ceil(target_h/p), math.ceil(target_w/p)
            rows, columns = math.ceil(input_rows/r), math.ceil(input_columns/r)
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
            from ..publisher_media_files import read_regular
            encoded = read_regular(path, max_bytes=self.max_bytes)
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
                aligned_h, aligned_w = math.ceil(vh/self.downsample), math.ceil(vw/self.downsample)
                if (vh <= 0 or vw <= 0 or
                        aligned_h*(aligned_w+1)+2 > self.max_tokens):
                    raise ValueError('prepared image patch grid exceeds token admission')
                chw = rounded.reshape(h, w, 3).transpose(2, 0, 1)
                patches = np.ascontiguousarray(chw.reshape(3, vh, p, vw, p)
                    .transpose(1, 3, 0, 2, 4).reshape(vh*vw, 3*p*p))
            finally:
                resized.close()
                image.close()
        # The language stream represents aligned output cells, not raw vision
        # patches. This allows non-unit aligner ratios without mismatching
        # image slots to projected embeddings.
        aligned_h, aligned_w = math.ceil(vh/self.downsample), math.ceil(vw/self.downsample)
        kinds = np.empty(aligned_h*(aligned_w+1)+2, dtype=np.int64)
        kinds[0], kinds[-1] = 0, 3
        grid = kinds[1:-1].reshape(aligned_h, aligned_w+1)
        grid[:, :-1], grid[:, -1] = 1, 2
        patches.flags.writeable = False
        kinds.flags.writeable = False
        return PublisherImageInput(start, patches, vh, vw, kinds, self.downsample)

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
        # Reconcile the expanded stream with the immutable span inventory
        # before handing either representation to a backend.  This keeps
        # caller ordering authoritative and prevents a later consumer from
        # silently pairing a span with a different placeholder range.
        if len(result) != len(types):
            raise ValueError('expanded image token/type stream is inconsistent')
        cursor = -1
        for span in prepared:
            width = int(np.asarray(span.types).size)
            if span.start <= cursor or span.start + width > len(result):
                raise ValueError('expanded image spans are not ordered')
            if any(result[pos] != image_token_id
                   for pos in range(span.start, span.start + width)):
                raise ValueError('expanded image span lost its placeholder range')
            if tuple(types[span.start:span.start+width]) != tuple(int(v) for v in span.types):
                raise ValueError('expanded image types differ from prepared span')
            cursor = span.start + width - 1
        if sum(1 for kind in types if kind >= 0) != sum(
                int(np.asarray(span.types).size) for span in prepared):
            raise ValueError('expanded image span coverage is incomplete')
        if any(token == image_token_id and types[index] < 0
               for index, token in enumerate(result)):
            raise ValueError('unexpanded image placeholder remains in text range')
        return result, types, tuple(prepared)
