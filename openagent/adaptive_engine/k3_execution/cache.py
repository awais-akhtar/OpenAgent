"""Append-only compressed MLA latent storage with cheap transaction forks."""
import numpy as np


def _immutable_page(value):
    """Publish bytes-backed rows whose write permission cannot be restored."""
    value=np.ascontiguousarray(value,dtype=np.float32)
    return np.frombuffer(value.tobytes(),dtype=np.float32).reshape(value.shape)


class MLALatentPages:
    """Normalized latent plus shared rotary-key coordinates, no expanded heads."""
    def __init__(self, rank, rope, *, page_tokens=256):
        self.rank, self.rope, self.page_tokens = rank, rope, page_tokens
        self.pages, self.length = (), 0

    @property
    def nbytes(self):
        return sum(page.nbytes for page in self.pages)

    def fork(self):
        candidate = MLALatentPages(self.rank, self.rope, page_tokens=self.page_tokens)
        candidate.pages, candidate.length = self.pages, self.length
        return candidate

    def append(self, latent, rotary_key):
        row = np.ascontiguousarray(np.concatenate((latent, rotary_key)), dtype=np.float32).reshape(1, self.rank + self.rope)
        if not np.isfinite(row).all():
            raise ValueError("nonfinite MLA latent")
        if self.pages and len(self.pages[-1]) < self.page_tokens:
            row, prefix = np.concatenate((self.pages[-1], row)), self.pages[:-1]
        else:
            prefix = self.pages
        row=_immutable_page(row)
        self.pages, self.length = (*prefix, row), self.length + 1

    def append_many(self, latent, rotary_key):
        """Publish an admitted chunk without repeatedly copying its tail page."""
        latent, rotary_key = np.asarray(latent), np.asarray(rotary_key)
        if (latent.ndim != 2 or not len(latent) or latent.shape[1] != self.rank
                or rotary_key.shape != (len(latent), self.rope)
                or latent.dtype != np.float32 or rotary_key.dtype != np.float32
                or not np.isfinite(latent).all() or not np.isfinite(rotary_key).all()):
            raise ValueError('finite FP32 latent/rotary chunk required')
        joined = np.ascontiguousarray(np.concatenate((latent, rotary_key), axis=1))
        pages, offset = list(self.pages), 0
        if pages and len(pages[-1]) < self.page_tokens:
            extent = min(self.page_tokens-len(pages[-1]), len(joined))
            tail = np.concatenate((pages[-1], joined[:extent]))
            tail=_immutable_page(tail)
            pages[-1], offset = tail, extent
        while offset < len(joined):
            page = _immutable_page(joined[offset:offset+self.page_tokens])
            pages.append(page)
            offset += len(page)
        self.pages, self.length = tuple(pages), self.length+len(joined)

    def snapshot(self):
        # Separate array metadata, shared immutable payload. Later appends
        # replace only the tail and cannot change a retained continuation.
        return tuple(page.view() for page in self.pages)

    def from_snapshot(self, pages, length):
        if type(pages) is not tuple:
            raise ValueError("tuple of latent pages required")
        candidate = MLALatentPages(self.rank, self.rope, page_tokens=self.page_tokens)
        copies, count = [], 0
        for index, page in enumerate(pages):
            if type(page) is not np.ndarray or page.ndim != 2 or page.shape[1] != self.rank + self.rope:
                raise ValueError("invalid latent page shape")
            n = len(page)
            if not 1 <= n <= self.page_tokens or (index < len(pages) - 1 and n != self.page_tokens):
                raise ValueError("invalid latent page extent")
            if page.dtype != np.float32 or not page.flags.c_contiguous or not np.isfinite(page).all():
                raise ValueError("invalid latent page data")
            page = _immutable_page(page)
            copies.append(page)
            count += n
        if count != length:
            raise ValueError("latent pages disagree with snapshot position")
        candidate.pages, candidate.length = tuple(copies), count
        return candidate


class MLAPages:
    def __init__(self, heads, key, value, *, page_tokens=256):
        self.heads, self.key, self.value = heads, key, value
        self.page_tokens = page_tokens
        self.pages = ()
        self.length = 0

    @property
    def nbytes(self):
        return sum(k.nbytes + v.nbytes for k, v in self.pages)

    def fork(self):
        candidate = MLAPages(self.heads, self.key, self.value, page_tokens=self.page_tokens)
        candidate.pages, candidate.length = self.pages, self.length
        return candidate

    def append(self, key, value):
        key = np.ascontiguousarray(key, dtype=np.float32).reshape(1, self.heads, self.key)
        value = np.ascontiguousarray(value, dtype=np.float32).reshape(1, self.heads, self.value)
        if self.pages and self.pages[-1][0].shape[0] < self.page_tokens:
            oldk, oldv = self.pages[-1]
            key, value = np.concatenate((oldk, key)), np.concatenate((oldv, value))
            prefix = self.pages[:-1]
        else:
            key, value = key.copy(), value.copy()
            prefix = self.pages
        key,value=_immutable_page(key),_immutable_page(value)
        self.pages = (*prefix, (key, value))
        self.length += 1

    def snapshot(self):
        return tuple((k.view(), v.view()) for k, v in self.pages)

    def from_snapshot(self, pages, length):
        if type(pages) is not tuple:
            raise ValueError("tuple of MLA pages required")
        candidate = MLAPages(self.heads, self.key, self.value, page_tokens=self.page_tokens)
        copied, count = [], 0
        for index, pair in enumerate(pages):
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError("invalid MLA page")
            k, v = pair
            if type(k) is not np.ndarray or type(v) is not np.ndarray or k.ndim != 3:
                raise ValueError("invalid MLA page arrays")
            n = k.shape[0]
            if not 1 <= n <= self.page_tokens or (index < len(pages) - 1 and n != self.page_tokens):
                raise ValueError("invalid MLA page lengths")
            for a, shape in ((k, (n, self.heads, self.key)), (v, (n, self.heads, self.value))):
                if a.shape != shape or a.dtype != np.float32 or not a.flags.c_contiguous or not np.isfinite(a).all():
                    raise ValueError("invalid MLA page data")
            k,v=_immutable_page(k),_immutable_page(v)
            copied.append((k, v))
            count += n
        if count != length:
            raise ValueError("MLA page count disagrees with snapshot position")
        candidate.pages, candidate.length = tuple(copied), count
        return candidate
