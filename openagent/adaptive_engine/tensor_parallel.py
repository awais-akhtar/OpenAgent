"""Explicit distributed projection shards with caller-owned process groups.

Every rank must invoke registration and projection in the same order. This
module neither starts workers nor initializes/destroys a process group. Row
shards gather output channels; column shards sum partial products in FP32.
Neither arithmetic policy is a claim of publisher numerical equivalence.
"""
from dataclasses import dataclass
from contextlib import closing, contextmanager
import hashlib
import math
import threading

import numpy as np


def _safe_agreement_descriptor(value, *, max_nodes=4096, max_depth=16):
    """Accept only bounded immutable scalar/tuple descriptors from peers."""
    remaining=max_nodes
    stack=[(value,0)]
    while stack:
        current,depth=stack.pop()
        remaining-=1
        if remaining<0 or depth>max_depth:
            return False
        if current is None or type(current) is bool:
            continue
        if type(current) is int:
            # Agreement descriptors are control-plane input from other ranks.
            # Bound integer comparisons too; a peer must not be able to send an
            # arbitrarily large scalar that defeats the otherwise bounded
            # descriptor traversal.
            if current.bit_length()>256:return False
            continue
        if type(current) is float:
            if not math.isfinite(current):return False
            continue
        if type(current) is str:
            if len(current)>8192:return False
            continue
        if type(current) is tuple and len(current)<=256:
            stack.extend((item,depth+1) for item in current)
            continue
        return False
    return True


def _failure_type_name(error):
    """Describe a rank-local exception without invoking its user code."""
    if isinstance(error, KeyboardInterrupt):
        return 'KeyboardInterrupt'
    if isinstance(error, InterruptedError):
        return 'InterruptedError'
    try:
        name=type(error).__name__
    except BaseException:
        return 'BaseException'
    if type(name) is not str or not name or len(name)>128:
        return 'BaseException'
    return name


@dataclass(frozen=True)
class ProjectionShard:
    name: str
    shape: tuple
    axis: int
    start: int
    stop: int
    weights: object
    bank_format: str = 'dense'
    source_dtype: str = 'F32'
    execution_dtype: str = 'F32'

    def __post_init__(self):
        if (type(self.name) is not str or not self.name or
                type(self.shape) is not tuple or len(self.shape) != 2 or
                any(type(value) is not int or value <= 0 for value in self.shape) or
                type(self.axis) is not int or self.axis not in (0, 1) or
                type(self.start) is not int or type(self.stop) is not int or
                not 0 <= self.start <= self.stop <= self.shape[self.axis]):
            raise ValueError('explicit projection shard geometry required')
        if (type(self.bank_format) is not str or
                self.bank_format not in ('dense', 'bf16', 'fp8', 'fp4')):
            raise ValueError('explicit projection shard bank format required')
        if (type(self.source_dtype) is not str or not self.source_dtype or
                type(self.execution_dtype) is not str or not self.execution_dtype):
            raise ValueError('explicit source and execution dtypes required')
        if self.execution_dtype.upper() not in ('F32', 'FP32', 'BF16', 'BFLOAT16'):
            raise ValueError('unsupported projection execution dtype')
        source_upper = self.source_dtype.upper()
        if (self.bank_format == 'bf16' and source_upper not in ('BF16', 'BFloat16'.upper()) or
                self.bank_format == 'fp8' and not source_upper.startswith('FP8') or
                self.bank_format == 'fp4' and source_upper not in ('FP4', 'MXFP4')):
            raise ValueError('projection bank format and source dtype differ')
        expected = list(self.shape)
        expected[self.axis] = self.stop - self.start
        weights = np.asarray(self.weights)
        if (weights.dtype != np.float32 or weights.shape != tuple(expected) or
                not np.isfinite(weights).all()):
            raise ValueError('finite exact-shape FP32 projection shard required')


def shard_bounds(extent, rank, world_size, *, allow_empty=False):
    if (type(extent) is not int or type(rank) is not int or
            type(world_size) is not int or world_size <= 0 or
            type(allow_empty) is not bool or extent<=0 or
            (not allow_empty and extent < world_size) or not 0 <= rank < world_size):
        raise ValueError('positive extent and valid admitted rank partition required')
    return extent*rank//world_size, extent*(rank+1)//world_size


def load_projection_shard(checkpoint, target, *, rank, world_size,
                          max_weight_bytes, max_chunk_bytes=1024**2, axis=0,
                          cancelled=None, bank_format='dense'):
    """Stream only into a rank-local output-row bank, without a full bank copy.

    Conversion currently reads the tensor sequentially on every rank. Storage
    traffic is therefore replicated even though retained weights are sharded.
    The checkpoint service retains its file/layout/conversion admission rules.
    """
    if (type(max_weight_bytes) is not int or max_weight_bytes <= 0 or
            type(max_chunk_bytes) is not int or max_chunk_bytes <= 0):
        raise ValueError('positive shard and conversion budgets required')
    if type(bank_format) is not str or bank_format not in ('dense', 'bf16', 'fp8', 'fp4'):
        raise ValueError('explicit projection shard bank format required')
    rules = [r for r in checkpoint.report.contracts if r.target == target]
    if len(rules) != 1 or len(rules[0].shape) != 2:
        raise ValueError('one exact checkpoint matrix target required')
    if type(axis) is not int or axis not in (0, 1):
        raise ValueError('projection partition axis must be zero or one')
    if cancelled is not None and not callable(cancelled):
        raise TypeError('callable shard cancellation required')
    rows, cols = rules[0].shape
    # Empty local partitions are admitted for uneven expert/projection banks;
    # collective assembly still uses the same deterministic half-open bounds.
    start, stop = shard_bounds((rows, cols)[axis], rank, world_size,
                               allow_empty=True)
    count = (stop-start)*(cols if axis == 0 else rows)
    if count*4+max_chunk_bytes*8 > max_weight_bytes:
        raise MemoryError('rank-local weight and conversion buffers exceed admission')
    local = np.empty(count, np.float32)
    first, last, cursor = start*cols, stop*cols, 0
    with closing(checkpoint.stream(targets=(target,), output_encoding='float32',
                                   max_chunk_bytes=max_chunk_bytes)) as blocks:
        for block in blocks:
            if cancelled is not None and cancelled():
                raise InterruptedError('checkpoint shard loading cancelled')
            values = np.asarray(block.payload)
            if (block.target != target or block.shape != (rows, cols) or
                    block.encoding != 'float32' or block.offset != cursor or
                    values.dtype != np.float32 or values.ndim != 1 or
                    not values.size or not np.isfinite(values).all()):
                raise ValueError('noncontiguous or invalid checkpoint projection stream')
            end = cursor+values.size
            if end > rows*cols:
                raise ValueError('projection stream exceeds declared extent')
            if axis == 0:
                left, right = max(first, cursor), min(last, end)
                if right > left:
                    local[left-first:right-first] = values[left-cursor:right-cursor]
            else:
                # A conversion block may split rows at arbitrary columns.
                # Copy only this rank's intersections, never a full matrix.
                width = stop-start
                row = cursor//cols
                while row*cols < end:
                    left = max(cursor, row*cols+start)
                    right = min(end, row*cols+stop)
                    if right > left:
                        offset = row*width+left-row*cols-start
                        local[offset:offset+right-left] = values[left-cursor:right-cursor]
                    row += 1
            cursor = end
    if cursor != rows*cols:
        raise ValueError('incomplete checkpoint projection stream')
    if cancelled is not None and cancelled():
        raise InterruptedError('checkpoint shard loading cancelled')
    shape = (stop-start, cols) if axis == 0 else (rows, stop-start)
    local = np.frombuffer(local.tobytes(), dtype=np.float32).reshape(shape)
    source_dtype = getattr(rules[0], 'dtype', 'F32')
    return ProjectionShard(rules[0].sources[0], (rows, cols), axis, start, stop, local,
                            bank_format, source_dtype, 'F32')


class TensorParallelProjections:
    """A registered bank of local shards, returning replicated host outputs."""
    boundary_observation_supported=True

    def __init__(self, *, group, device, torch_version, max_bytes,
                 dtype='float32', bank_format='dense'):
        import torch
        import torch.distributed as distributed
        if str(torch.__version__) != torch_version:
            raise ValueError('explicit Torch version pin differs from installed runtime')
        if not distributed.is_available() or not distributed.is_initialized():
            raise ValueError('caller must initialize the distributed process group')
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('positive rank-local buffer budget required')
        if dtype not in ('float32', 'bfloat16'):
            raise ValueError('explicit float32 or bfloat16 projection policy required')
        if (type(bank_format) is not str or
                bank_format not in ('dense', 'bf16', 'fp8', 'fp4')):
            raise ValueError('explicit distributed projection bank format required')
        self.torch, self.distributed, self.group = torch, distributed, group
        self.device = torch.device(device)
        backend = str(distributed.get_backend(group)).lower()
        self.backend, self.torch_version = backend, torch_version
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('distributed projections require CPU or CUDA/HIP devices')
        if 'nccl' in backend and self.device.type != 'cuda':
            raise ValueError('NCCL requires a CUDA/HIP device')
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                raise ValueError('selected accelerator is unavailable')
            self.device = torch.device('cuda', torch.cuda.current_device()
                                       if self.device.index is None else self.device.index)
            torch.cuda.set_device(self.device)
        self.rank = distributed.get_rank(group)
        self.world_size = distributed.get_world_size(group)
        self.dtype_name, self.bank_format, self.max_bytes = dtype, bank_format, max_bytes
        self.dtype = getattr(torch, dtype)
        self._shards, self._pins = {}, {}
        self._lock = threading.RLock()
        self._closed = False
        self._poisoned = False
        self._active_stages = 0
        self._resident_bytes = 0
        self._cancellation = threading.local()
        self._observations=threading.local()

    @contextmanager
    def boundary_observation_scope(self,observer):
        """Capture actual converted operands before disjoint shard selection."""
        with self._lock:
            if self._closed or getattr(self._observations,'observer',None) is not None:
                raise RuntimeError('idle open projection observation owner required')
            if (not callable(getattr(observer,'projection',None)) or
                    not hasattr(observer,'projection_names') or not hasattr(observer,'maximum')):
                raise ValueError('bounded executed projection observer required')
            self._observations.observer=observer
            try:yield
            finally:self._observations.observer=None

    def _observation_policy(self):
        observer=getattr(self._observations,'observer',None)
        if observer is None:return None
        return hashlib.sha256(repr((tuple(sorted(observer.projection_names)),observer.maximum)).encode()).hexdigest()

    def _observe_input(self,name,inputs):
        observer=getattr(self._observations,'observer',None)
        if observer is not None and name in observer.projection_names:
            observer.projection(name,'input',inputs)

    def _admit_product_policy(self):
        from .device_execution_policy import admit_fp32
        admit_fp32(self.torch,self.device)
        if (self.dtype_name=='bfloat16' and self.device.type=='cuda' and
                self.torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction):
            raise ValueError('distributed BF16 products require reduced-precision reductions disabled')

    @contextmanager
    def cancellation_scope(self, event):
        """Bind local cancellation; stage agreement propagates it to peers.

        A rank must never leave a scheduled collective independently. Checks
        therefore happen inside local_stage's gathered error boundary.
        """
        if event is not None and not callable(getattr(event, 'is_set', None)):
            raise TypeError('event-like tensor parallel cancellation required')
        old = getattr(self._cancellation, 'event', None)
        self._cancellation.event = event
        try:
            yield self
        finally:
            self._cancellation.event = old

    def _check_stage_cancelled(self):
        event = getattr(self._cancellation, 'event', None)
        if event is not None and event.is_set():
            raise InterruptedError('tensor parallel projection cancelled')

    @property
    def identity(self):
        payload = (self.world_size, self.dtype_name, self.bank_format, self.backend, self.torch_version,
                   self.device.type, 'fp32-partial-sum-row-gather-precision-v2',
                   tuple((name, self._pins[name]) for name in sorted(self._pins)))
        return hashlib.sha256(repr(payload).encode('utf-8')).hexdigest()

    def _agree(self, descriptor, error=None):
        records = [None]*self.world_size
        try:
            self.distributed.all_gather_object(records, (descriptor, error), group=self.group)
        except BaseException:
            # A failed control collective leaves rank consensus unknown. Never
            # allow this bank to be reused as though its sequence were intact.
            self._poisoned=True;self._closed=True
            raise
        malformed = [rank for rank, value in enumerate(records)
                     if (type(value) is not tuple or len(value) != 2 or
                         (value[0] is not None and type(value[0]) is not tuple) or
                         (value[0] is not None and
                          not _safe_agreement_descriptor(value[0])) or
                         (value[1] is not None and
                          (type(value[1]) is not str or len(value[1])>4096)))]
        if malformed:
            # Agreement payloads cross a rank boundary. Treat unexpected
            # shapes/types as an uncertain schedule and quarantine before any
            # caller can attempt another tensor collective on this bank.
            self._poisoned=True;self._closed=True
            raise RuntimeError('malformed distributed projection agreement record from rank(s) '+
                               ','.join(str(rank) for rank in malformed))
        errors = [f'rank {rank}: {value[1]}' for rank, value in enumerate(records)
                  if value[1] is not None]
        if errors:
            # Preserve cooperative cancellation as cancellation after every
            # rank has reached the same stage boundary. Reporting it as an
            # admission ValueError makes serving owners misclassify a stopped
            # request and can leave cancellation-specific cleanup unrun.
            failures = [value[1] for value in records if value[1] is not None]
            # Preserve process-level user interruption after every rank has
            # reached the same control boundary.  In particular, translating
            # KeyboardInterrupt into ValueError makes a coordinated Ctrl+C
            # look like bad admission and can skip caller cancellation cleanup.
            # Do not propagate arbitrary exception classes across the group:
            # rank-local data/provider errors still use the stable ValueError
            # agreement envelope below.
            if failures and all(value.startswith('InterruptedError:') for value in failures):
                raise InterruptedError('distributed projection stage cancelled: '+'; '.join(errors))
            if failures and all(value.startswith('KeyboardInterrupt:') for value in failures):
                raise KeyboardInterrupt('distributed projection stage interrupted: '+'; '.join(errors))
            raise ValueError('distributed admission failed: '+'; '.join(errors))
        try:
            disagreement=any(value[0] != descriptor for value in records)
        except BaseException as error:
            # Even a structurally tuple-shaped descriptor may contain an
            # object whose equality is ambiguous or raises. Agreement itself
            # is no longer trustworthy after attempting that comparison.
            self._poisoned=True;self._closed=True
            raise RuntimeError('malformed distributed projection descriptor comparison; bank quarantined') from error
        if disagreement:
            # All surviving ranks have observed the same disagreement, but
            # their next scheduled tensor collective may differ.  A caller
            # retrying on this bank could therefore pair (for example) an
            # all-reduce with an all-gather and strand the group.  Quarantine
            # the bank before exposing the mismatch to the caller.
            self._poisoned=True;self._closed=True
            raise RuntimeError('rank projection schedules or policies differ; bank quarantined')

    def _collective(self, operation, *args, **kwargs):
        """Run a transport collective and quarantine this bank on failure."""
        if self._closed or self._poisoned:
            raise RuntimeError('distributed projection bank is closed or rank state is uncertain')
        try:
            return getattr(self.distributed,operation)(*args,**kwargs)
        except BaseException:
            self._poisoned=True;self._closed=True
            raise

    @contextmanager
    def local_stage(self, descriptor):
        """Synchronize local staging failures before a tensor collective.

        Allocation, provider reads and host exports can fail independently on
        any rank. Every peer reaches this control boundary before proceeding.
        Process loss or transport failure still requires group restart.
        """
        with self._lock:
            if self._closed or self._poisoned:
                raise RuntimeError('distributed projection bank is closed')
            self._active_stages += 1
        error = None
        try:
            try:
                self._check_stage_cancelled()
                yield
                self._check_stage_cancelled()
            except BaseException as failure:
                # A rank-local interruption (including KeyboardInterrupt) must
                # still join the stage agreement. Letting it escape here can
                # strand peers in the control collective that closes this
                # local staging boundary.
                name = _failure_type_name(failure)
                error = (name+': cooperative interruption'
                         if isinstance(failure, (InterruptedError, KeyboardInterrupt)) else name)
            try:self._agree(descriptor,error)
            except BaseException:
                self._poisoned=True;self._closed=True
                raise
        finally:
            with self._lock:
                self._active_stages -= 1

    def register(self, shard):
        with self._lock:
            error, weights = None, None
            descriptor = None
            try:
                if self._closed or type(shard) is not ProjectionShard:
                    raise ValueError('open projection bank and explicit shard required')
                if (type(shard.name) is not str or not shard.name or
                        shard.name in self._shards or type(shard.shape) is not tuple or
                        len(shard.shape) != 2 or
                        any(type(v) is not int or v <= 0 for v in shard.shape) or
                        type(shard.axis) is not int or shard.axis not in (0, 1) or
                        shard.bank_format != self.bank_format):
                    raise ValueError('unique named matrix and partition axis required')
                expected = shard_bounds(shard.shape[shard.axis], self.rank, self.world_size,
                                        allow_empty=True)
                if (shard.start, shard.stop) != expected:
                    raise ValueError('shard does not match the rank partition')
                shape = list(shard.shape)
                shape[shard.axis] = shard.stop-shard.start
                observed = np.asarray(shard.weights)
                if (observed.dtype != np.float32 or observed.shape != tuple(shape) or
                        not np.isfinite(observed).all()):
                    raise ValueError('finite exact-shape FP32 local weights required')
                if self._resident_bytes+observed.nbytes*3 > self.max_bytes:
                    raise MemoryError('rank-local projection weights exceed admission')
                weights = np.array(observed, dtype=np.float32, order='C', copy=True)
                weights.flags.writeable = False
                descriptor = (shard.name, shard.shape, shard.axis, self.dtype_name,
                              self.bank_format, shard.source_dtype, shard.execution_dtype,
                              self.backend, self.torch_version,
                              self.device.type)
            except BaseException as failure:
                error = _failure_type_name(failure)
                if isinstance(failure, (InterruptedError, KeyboardInterrupt)):
                    error += ': cooperative interruption'
            self._agree(descriptor, error)
            # Fingerprint serialization allocates a temporary byte view and
            # the receive list. Agree on those local preparations before any
            # peer enters the fingerprint collective, so rank-local memory
            # failure cannot strand ranks at the next stage.
            with self.local_stage((descriptor,'registration-fingerprint')):
                fingerprint = hashlib.sha256(weights.tobytes()).hexdigest()
                pins = [None]*self.world_size
            self._collective('all_gather_object',pins,
                fingerprint, group=self.group)
            try:
                with self.local_stage((descriptor,'registration-publish')):
                    self._shards[shard.name] = ProjectionShard(shard.name, shard.shape,
                        shard.axis, shard.start, shard.stop, weights, self.bank_format,
                        shard.source_dtype, shard.execution_dtype)
                    self._pins[shard.name] = (descriptor, tuple(pins))
                    self._resident_bytes += weights.nbytes
            except BaseException:
                # Publication is local and may fail (for example during a
                # dictionary growth) after peers have installed their entry.
                # The local-stage agreement has informed every surviving rank;
                # remove any partial entry and retire the bank consistently.
                registered = self._shards.pop(shard.name, None)
                self._pins.pop(shard.name, None)
                if registered is not None:
                    self._resident_bytes -= registered.weights.nbytes
                self._poisoned = True
                self._closed = True
                raise

    def handles(self, name):
        return not self._closed and name in self._shards

    def shape(self, name):
        return self._shards[name].shape

    def linear(self, name, value):
        with self._lock:
            error, observed, shard, vector, descriptor = None, None, None, None, None
            try:
                if self._closed:
                    raise RuntimeError('distributed projection bank is closed')
                if type(name) is not str or not name:
                    raise ValueError('non-empty exact projection name required')
                shard = self._shards[name]
                observed = np.asarray(value)
                rows, cols = shard.shape
                if (observed.dtype.kind != 'f' or observed.ndim not in (1, 2) or
                        observed.shape[-1] != cols or not observed.size or
                        not np.isfinite(observed).all()):
                    raise ValueError('finite replicated vector or batch required')
                batch = 1 if observed.ndim == 1 else observed.shape[0]
                maximum = (rows+self.world_size-1)//self.world_size
                # Account for retained host weights, conversion/device copies,
                # padded gather banks, host output and input staging.
                scratch = (shard.weights.nbytes*3+observed.size*16+
                           batch*rows*16+batch*maximum*self.world_size*8)
                if scratch > self.max_bytes-self._resident_bytes:
                    raise MemoryError('distributed projection buffers exceed rank admission')
                vector = np.array(observed, dtype=np.float32, order='C', copy=True)
                descriptor = (name, observed.shape, self.identity,
                              self._observation_policy())
            except BaseException as failure:
                error = _failure_type_name(failure)
                if isinstance(failure, (InterruptedError, KeyboardInterrupt)):
                    error += ': cooperative interruption'
            # Never serialize an unadmitted caller object in the cross-rank
            # agreement record. In particular, array-like identifiers can
            # make tuple equality raise during rank schedule comparison.
            if descriptor is None:
                descriptor = (name if type(name) is str and name else None,
                              None if observed is None else observed.shape,
                              None,None)
            self._agree(descriptor, error)
            # Replicated input identity matters: unlike a shape check, this
            # catches different session histories before combining outputs.
            input_digest=None
            with self.local_stage((descriptor,'linear-input-fingerprint')):
                input_digest=hashlib.sha256(vector.tobytes()).hexdigest()
            self._agree((descriptor,input_digest))
            torch = self.torch
            with torch.inference_mode():
                with self.local_stage((descriptor,'local-product')):
                    self._admit_product_policy()
                    matrix = torch.tensor(shard.weights, device=self.device, dtype=self.dtype)
                    inputs = torch.tensor(vector, device=self.device, dtype=self.dtype)
                    self._observe_input(name,inputs)
                    if shard.axis == 1:
                        inputs = inputs[..., shard.start:shard.stop]
                    output = torch.nn.functional.linear(inputs, matrix).float()
                    if shard.axis == 0:
                        maximum = (shard.shape[0]+self.world_size-1)//self.world_size
                        padded = torch.zeros((*output.shape[:-1], maximum),
                                             device=self.device, dtype=torch.float32)
                        padded[..., :shard.stop-shard.start] = output
                        gathered = [torch.empty_like(padded) for _ in range(self.world_size)]
                if shard.axis == 1:
                    self._collective('all_reduce',output, group=self.group)
                else:
                    self._collective('all_gather',gathered, padded, group=self.group)
                    with self.local_stage((descriptor,'row-assembly')):
                        output = torch.cat([part[..., :shard_bounds(shard.shape[0], rank,
                            self.world_size, allow_empty=True)[1]-shard_bounds(shard.shape[0], rank,
                            self.world_size, allow_empty=True)[0]] for rank, part in enumerate(gathered)], dim=-1)
                with self.local_stage((descriptor,'finite-boundary')):
                    finite = torch.isfinite(output).all().to(torch.int32)
                self._collective('all_reduce',finite, op=self.distributed.ReduceOp.MIN,
                                 group=self.group)
                with self.local_stage((descriptor,'finite-result-check')):
                    if not bool(finite.item()):
                        raise ValueError('nonfinite distributed projection result')
                with self.local_stage((descriptor,'host-export')):
                    result = output.cpu().numpy().copy()
                return result

    def grouped_linear(self, name, value, *, groups):
        """Contract each row group against its own input, preserving row order.

        Each local weight row selects the input of its logical output group.
        Row shards gather channels; column shards sum partial group products.
        """
        with self._lock:
            error, observed, shard, vector, descriptor = None, None, None, None, None
            try:
                if self._closed:
                    raise RuntimeError('distributed projection bank is closed')
                if type(name) is not str or not name:
                    raise ValueError('non-empty exact projection name required')
                shard = self._shards[name]
                rows, cols = shard.shape
                if type(groups) is not int or groups <= 0 or rows % groups:
                    raise ValueError('projection rows must divide into positive equal groups')
                observed = np.asarray(value)
                if (observed.dtype.kind != 'f' or observed.ndim not in (2, 3) or
                        observed.shape[-2:] != (groups, cols) or not observed.size or
                        not np.isfinite(observed).all()):
                    raise ValueError('finite group-local projection inputs required')
                batch = 1 if observed.ndim == 2 else observed.shape[0]
                maximum = (rows+self.world_size-1)//self.world_size
                local_rows = shard.weights.shape[0]
                local_cols = shard.weights.shape[1]
                scratch = (shard.weights.nbytes*3+observed.size*16+
                           batch*local_rows*local_cols*8+local_rows*16+
                           batch*rows*16+batch*maximum*self.world_size*8)
                if scratch > self.max_bytes-self._resident_bytes:
                    raise MemoryError('grouped projection buffers exceed rank admission')
                vector = np.array(observed, dtype=np.float32, order='C', copy=True)
                descriptor = (name, groups, observed.shape, self.identity,
                              'grouped-linear-v1',self._observation_policy())
            except BaseException as failure:
                error = _failure_type_name(failure)
                if isinstance(failure, (InterruptedError, KeyboardInterrupt)):
                    error += ': cooperative interruption'
            if descriptor is None:
                descriptor = (name if type(name) is str and name else None,
                              groups if type(groups) is int and groups > 0 else None,
                              None if observed is None else observed.shape,
                              None, 'grouped-linear-v1',None)
            self._agree(descriptor, error)
            input_digest=None
            with self.local_stage((descriptor,'grouped-input-fingerprint')):
                input_digest=hashlib.sha256(vector.tobytes()).hexdigest()
            self._agree((descriptor,input_digest))
            torch = self.torch
            with torch.inference_mode():
                with self.local_stage((descriptor,'grouped-local-product')):
                    self._admit_product_policy()
                    matrix = torch.tensor(shard.weights, device=self.device, dtype=self.dtype)
                    inputs = torch.tensor(vector.reshape(batch, groups, cols),
                                          device=self.device, dtype=self.dtype)
                    self._observe_input(name,inputs)
                    first = shard.start if shard.axis == 0 else 0
                    last = shard.stop if shard.axis == 0 else rows
                    owners = torch.arange(first, last, device=self.device)//(rows//groups)
                    if shard.axis == 1:
                        inputs = inputs[..., shard.start:shard.stop]
                    selected = inputs.index_select(1, owners)
                    output = torch.einsum('brc,rc->br', selected, matrix).float()
                    if shard.axis == 0:
                        padded = torch.zeros((batch, maximum), device=self.device,
                                             dtype=torch.float32)
                        padded[:, :last-first] = output
                        gathered = [torch.empty_like(padded) for _ in range(self.world_size)]
                if shard.axis == 1:
                    self._collective('all_reduce',output, group=self.group)
                else:
                    self._collective('all_gather',gathered, padded, group=self.group)
                    with self.local_stage((descriptor,'grouped-row-assembly')):
                        output = torch.cat([part[:, :shard_bounds(rows, rank,
                            self.world_size, allow_empty=True)[1]-shard_bounds(rows, rank,
                            self.world_size, allow_empty=True)[0]] for rank, part in enumerate(gathered)], dim=-1)
                with self.local_stage((descriptor,'grouped-finite-boundary')):
                    finite = torch.isfinite(output).all().to(torch.int32)
                self._collective('all_reduce',finite, op=self.distributed.ReduceOp.MIN,
                                 group=self.group)
                shape = None
                with self.local_stage((descriptor,'grouped-host-export')):
                    if not bool(finite.item()):
                        raise ValueError('nonfinite distributed grouped projection result')
                    shape = ((groups, rows//groups) if vector.ndim == 2 else
                             (*vector.shape[:-2], groups, rows//groups))
                    result = output.reshape(shape).cpu().numpy().copy()
                return result

    def close(self):
        with self._lock:
            if self._active_stages:
                raise RuntimeError('cannot close an active distributed projection stage')
            self._closed = True
            self._shards.clear()
            self._pins.clear()
            self._resident_bytes = 0
