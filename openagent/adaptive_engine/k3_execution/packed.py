"""On-demand read-only MXFP4 tables; never materializes an expert bank."""
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import hashlib
import mmap
import os
from pathlib import Path
import stat
import threading

import numpy as np
from ..session_recovery import _path
from ..native_library_identity import _stamp as _file_stamp
from ..native_checkpoint import _open_local


@dataclass(frozen=True)
class PackedTensorSpan:
    path: str
    offset: int
    shape: tuple


def _stamp(file):
    import os
    stat = os.fstat(file.fileno())
    return _file_stamp(stat)


class MappedMXFP4Experts:
    """Explicit SafeTensors payload spans supplied by the checkpoint owner.

    Keys are full publisher tensor names. Offset is an absolute file offset,
    including the SafeTensors header. No automatic config/codec admission.
    File stamps detect ordinary mutation, not malicious same-stamp rewriting.
    """

    def __init__(self, spans, *, max_expert_bytes=64 * 1024 * 1024,
                 max_mapped_bytes=None):
        if type(max_expert_bytes) is not int or max_expert_bytes <= 0 or not spans:
            raise ValueError("nonempty packed span inventory and positive expert budget required")
        self.max_expert_bytes = max_expert_bytes
        self.max_mapped_bytes = (max_expert_bytes+6*(mmap.ALLOCATIONGRANULARITY+mmap.PAGESIZE)
                                 if max_mapped_bytes is None else max_mapped_bytes)
        if type(self.max_mapped_bytes) is not int or self.max_mapped_bytes <= 0:
            raise ValueError('positive mapped-expert budget required')
        self._lock = threading.Lock()
        self._closed = False
        self._pending = {}
        self._mapped_bytes = 0
        self._spans, self._stamps = {}, {}
        digest = hashlib.sha256()
        for name in sorted(spans):
            span = spans[name]
            if type(span) is not PackedTensorSpan or not isinstance(name, str) or ".experts." not in name or not name.endswith(("weight_packed", "weight_scale")):
                raise ValueError("explicit publisher packed expert span required")
            if type(span.offset) is not int or span.offset < 0 or type(span.shape) is not tuple or len(span.shape) != 2 or any(type(n) is not int or n <= 0 for n in span.shape):
                raise ValueError("invalid packed span extent")
            path = str(_path(span.path).resolve(strict=True))
            size = span.shape[0] * span.shape[1]
            if size > max_expert_bytes:
                raise ValueError("packed table exceeds lease budget")
            with ExitStack() as opened:
                file,stamp=_open_local(Path(path).parent,Path(path),opened)
                if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
                    raise ValueError('regular packed tensor source required')
                stamp = _stamp(file)
                if span.offset + size > stamp[2]:
                    raise ValueError("packed span outside file")
                if _file_stamp(_path(path).stat())!=stamp:
                    raise ValueError('packed tensor source changed during admission')
            self._spans[name] = PackedTensorSpan(path, span.offset, span.shape)
            self._stamps[name] = stamp
            digest.update(repr((name, path, span.offset, span.shape, stamp)).encode("utf-8"))
        self.identity = digest.hexdigest()

    @property
    def tensor_shapes(self):
        return {name: span.shape for name, span in self._spans.items()}

    def _reap(self):
        for key, (mapped, charge) in tuple(self._pending.items()):
            try:
                mapped.close()
            except BufferError:
                continue
            else:
                self._mapped_bytes -= charge
                del self._pending[key]

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('cannot close mapped experts during an active lease')
        try:
            self._closed = True
            self._reap()
            if self._pending:
                raise RuntimeError('packed tensor views remain live; cleanup is pending')
        finally:
            self._lock.release()

    def status(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('mapped expert lease is active')
        try:
            self._reap()
            return dict(closed=self._closed, mapped_bytes=self._mapped_bytes,
                        mapping_capacity=self.max_mapped_bytes,
                        pending_mappings=len(self._pending))
        finally:
            self._lock.release()

    @contextmanager
    def lease(self, layer, expert):
        if type(layer) is not int or layer < 0 or type(expert) is not int or expert < 0:
            raise ValueError('nonnegative integer layer and expert required')
        prefix = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}."
        names = [prefix + matrix + "." + suffix for matrix in ("w1", "w3", "w2")
                 for suffix in ("weight_packed", "weight_scale")]
        if any(name not in self._spans for name in names):
            raise ValueError("missing selected packed expert")
        if sum(self._spans[n].shape[0] * self._spans[n].shape[1] for n in names) > self.max_expert_bytes:
            raise ValueError("selected packed expert exceeds lease budget")
        charges = []
        for name in names:
            span = self._spans[name]
            size = span.shape[0]*span.shape[1]
            delta = span.offset % mmap.ALLOCATIONGRANULARITY
            charges.append(((delta+size+mmap.PAGESIZE-1)//mmap.PAGESIZE)*mmap.PAGESIZE)
        if sum(charges) > self.max_mapped_bytes:
            raise MemoryError('selected expert aligned mappings exceed admission')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('mapped experts already have an active lease')
        tables, mappings = {}, []
        primary = None
        try:
            if self._closed:
                raise RuntimeError('mapped experts are closed')
            self._reap()
            if self._pending:
                raise RuntimeError('previous packed tensor views remain live; cleanup is pending')
            with ExitStack() as stack:
                files = []
                for name, charge in zip(names, charges):
                    span = self._spans[name]
                    path=_path(span.path)
                    file,_=_open_local(path.parent,path,stack)
                    if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
                        raise ValueError('regular packed tensor source required')
                    if _stamp(file) != self._stamps[name]:
                        raise ValueError('packed expert file changed')
                    size = span.shape[0]*span.shape[1]
                    delta = span.offset % mmap.ALLOCATIONGRANULARITY
                    mapped = mmap.mmap(file.fileno(), delta+size, access=mmap.ACCESS_READ,
                                       offset=span.offset-delta)
                    mappings.append((mapped, charge))
                    self._mapped_bytes += charge
                    tables[name.removeprefix(prefix)] = np.frombuffer(
                        mapped, dtype=np.uint8, count=size, offset=delta).reshape(span.shape)
                    files.append((name, file))
                yield tables
                if any(_stamp(file) != self._stamps[name] or
                        _file_stamp(_path(self._spans[name].path).stat())!=self._stamps[name]
                        for name, file in files):
                    raise ValueError('packed expert file changed during execution')
        except BaseException as error:
            primary = error
            raise
        finally:
            tables.clear()
            failures = []
            for mapped, charge in reversed(mappings):
                try:
                    mapped.close()
                except BaseException as error:
                    self._pending[id(mapped)] = (mapped, charge)
                    failures.append(error)
                else:
                    self._mapped_bytes -= charge
            self._lock.release()
            if failures:
                if primary is not None:
                    primary.add_note('packed mapping cleanup is pending: '+str(failures[0]))
                else:
                    raise RuntimeError('packed tensor views remain live; cleanup is pending') from failures[0]
