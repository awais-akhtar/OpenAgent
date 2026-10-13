"""Bounded source-name providers; no publisher arithmetic or resident bank."""
from contextlib import contextmanager
import hashlib
import math
import threading
from types import MappingProxyType

import numpy as np

from .storage import positive


class _LeaseProvider:
    def __init__(self, checkpoint, rules, kind):
        if checkpoint.report.family != "deepseek_v41":
            raise ValueError("exact V4.1 publisher checkpoint required")
        checkpoint._select(None)
        self._checkpoint = checkpoint
        self._rules = MappingProxyType({r.sources[0]: r for r in rules})
        self.tensor_shapes = MappingProxyType({r.sources[0]: r.shape for r in rules})
        stamps = tuple(sorted(checkpoint._metadata.files.items()))
        self.identity = kind + ":" + hashlib.sha256(repr((
            str(checkpoint._metadata.root), checkpoint.report.config_sha256,
            stamps, tuple(rules))).encode("utf-8")).hexdigest()
        self._lock = threading.Lock()
        self._closed = False
        self._local = threading.local()

    @contextmanager
    def cancellation_scope(self, cancel):
        if cancel is not None and not callable(getattr(cancel, 'is_set', None)):
            raise ValueError('publisher cancellation must provide is_set')
        previous = getattr(self._local, 'cancel', None)
        self._local.cancel = cancel
        try:
            yield
        finally:
            self._local.cancel = previous

    def _check_cancelled(self):
        cancel = getattr(self._local, 'cancel', None)
        if cancel is not None and cancel.is_set():
            from ..execution_service import ExecutionCancelled
            raise ExecutionCancelled('publisher tensor preparation cancelled')

    @contextmanager
    def _lease(self):
        self._check_cancelled()
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("provider already has an active lease")
        try:
            if self._closed:
                raise RuntimeError("provider is closed")
            self._checkpoint._metadata.assert_unchanged()
            try:
                self._check_cancelled()
                yield
                self._check_cancelled()
            finally:
                self._checkpoint._metadata.assert_unchanged()
        finally:
            self._lock.release()

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot close provider during an active lease")
        try:
            self._closed = True
        finally:
            self._lock.release()


class V41TokenEmbeddingProvider(_LeaseProvider):
    """Read original unquantized token rows without retaining the vocabulary."""
    def __init__(self,checkpoint,*,max_output_bytes):
        self.max_output_bytes=positive(max_output_bytes,'max_output_bytes')
        rules=tuple(rule for rule in checkpoint.report.contracts
                    if rule.target=='v41.text.embedding' or
                    (rule.target.startswith('v41.mtp.') and rule.target.endswith('.markov_head.embed')))
        if sum(rule.target=='v41.text.embedding' for rule in rules)!=1:
            raise ValueError('one exact original publisher token embedding required')
        for rule in rules:
            if len(rule.shape)!=2 or len(rule.sources)!=1:
                raise ValueError('exact original publisher token table required')
            record=checkpoint._metadata.tensors[rule.sources[0]]
            if record.dtype not in ('BF16','F16','F32') or record.shape!=rule.shape:
                raise ValueError('original floating token table geometry required')
            if rule.shape[1]*128>self.max_output_bytes:
                raise MemoryError('one publisher token row exceeds conversion admission')
        super().__init__(checkpoint,rules,'v41-streamed-token-embedding-v1')
        self.identity+=':'+str(self.max_output_bytes)

    def read_row(self,source,index):
        from ..publisher_tensor_reader import PublisherTensorReader,TensorSlice
        rule=self._rules[source];rows,width=rule.shape
        if type(index) is not int or not 0<=index<rows:
            raise ValueError('valid original publisher token row required')
        with self._lease():
            reader=PublisherTensorReader(self._checkpoint._metadata,max_work_bytes=self.max_output_bytes)
            try:value=reader.read(TensorSlice(source,(width,),index*width))
            finally:reader.close()
            self._check_cancelled();value.flags.writeable=False
            return value


class V41EngramRowProvider(_LeaseProvider):
    """Borrow selected readonly FP32 rows under the exact embedding source name."""
    format = "fp32"

    def __init__(self, checkpoint, *, max_output_bytes, max_rows=4096,
                 round_to_bfloat16=True):
        self.max_output_bytes = positive(max_output_bytes, "max_output_bytes")
        self.max_rows = positive(max_rows, "max_rows")
        if type(round_to_bfloat16) is not bool:
            raise ValueError("round_to_bfloat16 must be boolean")
        self.round_to_bfloat16 = round_to_bfloat16
        rules = tuple(r for r in checkpoint.report.contracts
                      if r.encoding == "fp8-e4m3-row-g32")
        super().__init__(checkpoint, rules, "v41-engram-metadata")
        self.identity += ":bf16-rounded" if round_to_bfloat16 else ":fp32"

    @contextmanager
    def lease_rows(self, source, row_ids):
        if type(source) is not str or source not in self._rules:
            raise ValueError("unknown exact Engram embedding source")
        with self._lease():
            rows = self._checkpoint.read_embedding_rows(self._rules[source].target, row_ids,
                max_output_bytes=self.max_output_bytes, max_rows=self.max_rows,
                round_to_bfloat16=self.round_to_bfloat16)
            self._check_cancelled()
            tables = {source: rows}
            try:
                yield MappingProxyType(tables)
            finally:
                tables.clear()


class V41PackedExpertProvider(_LeaseProvider):
    """Selected six-table U8 preparation using the loader's bounded FP4 stream.

    Stored I8 weights are treated as raw nibble bytes, not numerically cast.
    Namespace is layers or mtp; a draft stage is never a backbone layer alias.
    """
    format = "e2m1-low-first-e8m0-g32"

    def __init__(self, checkpoint, *, max_expert_bytes, max_chunk_bytes=1024**2):
        self.max_expert_bytes = positive(max_expert_bytes, "max_expert_bytes")
        self.max_chunk_bytes = positive(max_chunk_bytes, "max_chunk_bytes")
        if self.max_chunk_bytes < 4096:
            raise MemoryError("conversion chunk budget must be at least 4096 bytes")
        self.max_preparation_bytes = self.max_expert_bytes + 2*self.max_chunk_bytes
        rules = tuple(r for r in checkpoint.report.contracts if r.encoding == "v41-fp4-g32")
        groups = {}
        shapes = {}
        for rule in rules:
            parts = rule.sources[0].split(".")
            if (len(parts) != 7 or parts[0] not in ("layers", "mtp")
                    or parts[2:4] != ["ffn", "experts"] or parts[5] not in ("w1", "w3", "w2")):
                raise ValueError("unsupported exact expert source declaration")
            groups.setdefault((parts[0], int(parts[1]), int(parts[4])), []).append(rule)
            for source in rule.sources:
                shapes[source] = checkpoint.tensors[source].shape
        for group in groups.values():
            if len(group) != 3 or {r.sources[0].split(".")[5] for r in group} != {"w1", "w3", "w2"}:
                raise ValueError("complete three-matrix expert required")
            size = sum(math.prod(r.shape)//32*17 for r in group)
            if size > self.max_expert_bytes:
                raise MemoryError("selected packed expert exceeds provider byte budget")
        super().__init__(checkpoint, rules, "v41-packed-metadata")
        self.tensor_shapes = MappingProxyType(shapes)
        self._groups = MappingProxyType({k: tuple(v) for k, v in groups.items()})

    @contextmanager
    def lease(self, layer, expert, *, namespace="layers"):
        if (type(layer) is not int or isinstance(layer, bool) or
                type(expert) is not int or isinstance(expert, bool) or type(namespace) is not str
                or (namespace, layer, expert) not in self._groups):
            raise ValueError("unknown exact V4.1 namespace/layer/expert")
        with self._lease():
            rules = self._groups[(namespace, layer, expert)]
            tables = {}
            try:
                for rule in rules:
                    self._check_cancelled()
                    rows, columns = rule.shape
                    tables[rule.sources[0]] = np.empty((rows, columns//2), np.uint8)
                    tables[rule.sources[1]] = np.empty((rows, columns//32), np.uint8)
                by_target = {r.target: r for r in rules}
                # Reuse admitted raw-byte repacking; publish only after exhaustion.
                for block in self._checkpoint.stream(targets=tuple(by_target),
                        max_chunk_bytes=self.max_chunk_bytes,
                        output_encoding="mxfp4-e2m1-g32-ue8m0"):
                    self._check_cancelled()
                    rule = by_target[block.target]
                    records = block.payload.reshape(-1, 17)
                    start, end = block.offset, block.offset + len(records)
                    tables[rule.sources[0]].reshape(-1, 16)[start:end] = records[:, 1:]
                    tables[rule.sources[1]].reshape(-1)[start:end] = records[:, 0]
                for array in tables.values():
                    array.setflags(write=False)
                self._check_cancelled()
                yield MappingProxyType(tables)
            finally:
                tables.clear()


class V41FP4ProjectionProvider(_LeaseProvider):
    """Original FP4 output-row tiles without complete expert preparation."""
    format = 'e2m1-low-first-e8m0-g32'

    def __init__(self, checkpoint, *, max_projection_bytes=64*1024**2):
        self.max_projection_bytes = positive(max_projection_bytes, 'max_projection_bytes')
        rules = tuple(rule for rule in checkpoint.report.contracts if rule.encoding == 'v41-fp4-g32')
        if not rules:
            raise ValueError('original publisher packed FP4 projections required')
        super().__init__(checkpoint, rules, 'v41-fp4-row-projection-metadata')

    def tile_rows(self, source):
        rule = self._rules[source]
        rows, columns = rule.shape
        if columns % 32:
            raise ValueError('exact rowwise FP4 column groups required')
        count = self.max_projection_bytes // (columns//2 + columns//32)
        if not count:
            raise MemoryError('FP4 row exceeds projection tile admission')
        return min(rows, count)

    @contextmanager
    def lease(self, source, *, row_start=0, row_count=None):
        if type(source) is not str or source not in self._rules:
            raise ValueError('unknown original FP4 projection')
        rule = self._rules[source]
        rows, columns = rule.shape
        if row_count is None:
            row_count = rows - row_start
        if (type(row_start) is not int or type(row_count) is not int or row_start < 0
                or row_count <= 0 or row_start + row_count > rows):
            raise ValueError('exact FP4 row interval required')
        size = row_count * (columns//2 + columns//32)
        if size > self.max_projection_bytes:
            raise MemoryError('FP4 output-row tile exceeds admission')
        with self._lease():
            weights = self._checkpoint._metadata.read_range(rule.sources[0],
                row_start*(columns//2), row_count*(columns//2))
            self._check_cancelled()
            scales = self._checkpoint._metadata.read_range(rule.sources[1],
                row_start*(columns//32), row_count*(columns//32))
            scale_array = np.frombuffer(scales, np.uint8).reshape(row_count, columns//32)
            if np.any(scale_array == 255):
                raise ValueError('nonfinite original FP4 group scale')
            tables = {rule.sources[0]: np.frombuffer(weights, np.uint8).reshape(row_count, columns//2),
                      rule.sources[1]: scale_array}
            try:
                self._check_cancelled()
                yield MappingProxyType(tables)
            finally:
                tables.clear()


class V41FP8ProjectionProvider(_LeaseProvider):
    """Read row-aligned FP8 projection tiles without allocating FP32 weights.

    Each scale covers 32 rows and 32 columns. Tiles begin at a multiple of 32
    rows; only the final tile may have a partial row block. Borrowed raw-byte
    arrays describe this tile, not the complete matrix. Immutable checkpoint
    files are required; file stamps are not content authentication.
    """
    format = "fp8-e4m3-g32x32-ue8m0"

    def __init__(self, checkpoint, *, max_projection_bytes=64*1024**2):
        self.max_projection_bytes = positive(max_projection_bytes, "max_projection_bytes")
        rules = tuple(r for r in checkpoint.report.contracts
                      if r.encoding == "fp8-e4m3-g32x32"
                      and not r.sources[0].endswith("wo_a.weight"))
        super().__init__(checkpoint, rules, "v41-fp8-projection-metadata")

    def tile_rows(self, source):
        if source not in self._rules:
            raise ValueError("unknown exact FP8 projection source")
        rows, columns = self._rules[source].shape
        block_bytes = 32*columns + columns//32
        blocks = self.max_projection_bytes//block_bytes
        if blocks < 1:
            raise MemoryError("projection budget cannot hold one 32-row scale block")
        return min(rows, blocks*32)

    @contextmanager
    def lease(self, source, *, row_start=0, row_count=None):
        if type(source) is not str or source not in self._rules:
            raise ValueError("unknown exact FP8 projection source")
        rule = self._rules[source]
        rows, columns = rule.shape
        if type(row_start) is not int:
            raise ValueError("FP8 tile row_start must be an integer")
        if row_count is None:
            row_count = rows-row_start
        if (type(row_start) is not int or type(row_count) is not int
                or row_start < 0 or row_start % 32 or row_count <= 0
                or row_start+row_count > rows
                or (row_count % 32 and row_start+row_count != rows)):
            raise ValueError("FP8 tile must cover exact aligned row blocks")
        scale_rows = (row_count+31)//32
        needed = row_count*columns + scale_rows*(columns//32)
        if needed > self.max_projection_bytes:
            raise MemoryError("FP8 projection tile exceeds provider budget")
        with self._lease():
            weights = self._checkpoint._metadata.read_range(
                rule.sources[0], row_start*columns, row_count*columns)
            self._check_cancelled()
            scales = self._checkpoint._metadata.read_range(
                rule.sources[1], (row_start//32)*(columns//32), scale_rows*(columns//32))
            self._check_cancelled()
            scale_array = np.frombuffer(scales, np.uint8).reshape(scale_rows, columns//32)
            if np.any(scale_array == 255):
                raise ValueError("nonfinite UE8M0 FP8 projection scale")
            tables = {rule.sources[0]: np.frombuffer(weights, np.uint8).reshape(row_count, columns),
                      rule.sources[1]: scale_array}
            try:
                yield MappingProxyType(tables)
            finally:
                tables.clear()
