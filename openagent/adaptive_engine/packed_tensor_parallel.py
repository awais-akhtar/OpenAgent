"""Leased V4.1 FP8 row shards without resident decoded projection matrices.

Packed dot products run in the explicitly supplied native CPU kernel; the
caller-owned process group gathers their results. This is a CPU arithmetic
policy even when the collective transport uses an accelerator device.
"""
import hashlib
import threading
from contextlib import contextmanager

import numpy as np

from .tensor_parallel import TensorParallelProjections, shard_bounds, _failure_type_name


def block_row_bounds(rows, rank, world_size):
    if type(rows) is not int or rows <= 0:
        raise ValueError('positive packed projection rows required')
    start, stop = shard_bounds((rows+31)//32, rank, world_size,allow_empty=True)
    return min(rows,start*32), min(rows, stop*32)


class PackedFP8TensorParallelProjections(TensorParallelProjections):
    boundary_observation_supported=False
    """Borrow provider/kernel lifetime and retain only projection metadata.

    All ranks register and invoke the same schedule. Any rank-local read or
    arithmetic failure is gathered before entering tensor result collectives.
    The provider may have at most one live tile lease at a time on each rank.
    """
    provider_format = 'fp8-e4m3-g32x32-ue8m0'
    row_alignment = 32
    numerical_policy = 'native-cpu-tiled32-fp8-row-gather-v1'

    def _row_bounds(self, rows, rank):
        return block_row_bounds(rows, rank, self.world_size)

    def _tile_rows(self, source):
        limit=getattr(self.kernel,'tile_rows',None)
        return min(self.provider.tile_rows(source),limit(self.provider.tensor_shapes[source][1])) if callable(limit) else self.provider.tile_rows(source)

    def __init__(self, *, provider, kernel, checkpoint_identity, group, device,
                 torch_version, max_bytes, collective_timeout_seconds=120):
        super().__init__(group=group,device=device,torch_version=torch_version,
                         max_bytes=max_bytes,dtype='float32',
                         collective_timeout_seconds=collective_timeout_seconds)
        if (getattr(provider,'format',None) != self.provider_format or
                not callable(getattr(provider,'lease',None)) or
                not callable(getattr(kernel,'matvec',None))):
            raise ValueError('exact leased tiled32 FP8 provider and native kernel required')
        if type(checkpoint_identity) is not str or not checkpoint_identity:
            raise ValueError('explicit complete checkpoint identity required')
        self.provider,self.kernel = provider,kernel
        self.numerical_policy=getattr(kernel,'numerical_policy',self.numerical_policy)
        self.checkpoint_identity = checkpoint_identity
        self._provider_identity = provider.identity
        self._kernel_identity=getattr(kernel,'identity',None)
        if type(self._kernel_identity) is not str or not self._kernel_identity:
            raise ValueError('content-identified native packed projection kernel required')
        self._packed = {}
        self._active = False

    @property
    def identity(self):
        return hashlib.sha256(repr((self.checkpoint_identity,self.world_size,
            self.backend,self.torch_version,self.device.type,self.numerical_policy,
            self._kernel_identity,tuple(sorted(self._packed.items())),tuple(sorted(self._pins.items())))).encode()).hexdigest()

    def register_source(self, source):
        with self._lock:
            error, descriptor = None, None
            try:
                if type(source) is not str or not source:
                    raise ValueError('non-empty exact packed source name required')
                if self._closed or self._active or source in self._packed:
                    raise RuntimeError('open idle packed bank and unique source required')
                shape = self.provider.tensor_shapes[source]
                if (len(shape)!=2 or any(type(v) is not int or v<=0 for v in shape) or shape[1]%32):
                    raise ValueError('exact tiled32 matrix geometry required')
                bounds = self._row_bounds(shape[0],self.rank)
                self._tile_rows(source)
                descriptor = (source,tuple(shape),self.checkpoint_identity,self._kernel_identity,self.backend,
                              self.torch_version,self.device.type)
            except BaseException as failure:
                message=_failure_type_name(failure)
                error=(message+': cooperative interruption'
                       if isinstance(failure,(InterruptedError,KeyboardInterrupt)) else message)
            self._agree(descriptor,error)
            with self.local_stage((descriptor,'source-pin-allocation')):
                providers = [None]*self.world_size
            self._collective('all_gather_object',providers,self._provider_identity,group=self.group)
            try:
                with self.local_stage((descriptor,'source-registration-publish')):
                    self._packed[source] = tuple(shape)
                    self._pins[source] = tuple(providers)
            except BaseException:
                self._packed.pop(source,None);self._pins.pop(source,None)
                self._poisoned=True;self._closed=True
                raise

    def handles(self, source):
        return not self._closed and source in self._packed

    @contextmanager
    def cancellation_scope(self,event):
        """Keep packed tile reads inside the request's cancellation boundary."""
        with super().cancellation_scope(event), self.provider.cancellation_scope(event):
            yield self

    def contains_projection_owner(self,owner):
        return owner is self.provider or owner is self.kernel

    def shape(self, source):
        return self._packed[source]

    def linear(self, source, value):
        return self._project(source,value,groups=None)

    def grouped_linear(self, source, value, *, groups):
        return self._project(source,value,groups=groups)

    def _project(self, source, value, *, groups):
        with self._lock:
            error, vector, descriptor = None, None, None
            try:
                if type(source) is not str or not source:
                    raise ValueError('non-empty exact packed source name required')
                if self._closed or self._active:
                    raise RuntimeError('packed projection bank closed or active')
                if self.provider.identity != self._provider_identity:
                    raise ValueError('packed projection provider identity changed')
                if self.kernel.identity!=self._kernel_identity:
                    raise ValueError('packed projection kernel identity changed')
                rows,columns = self._packed[source]
                observed = np.asarray(value)
                if observed.dtype.kind!='f' or not observed.size or not np.isfinite(observed).all():
                    raise ValueError('finite packed projection activations required')
                if groups is None:
                    if observed.ndim not in (1,2) or observed.shape[-1]!=columns:
                        raise ValueError('replicated vector or matrix required')
                    batch = 1 if observed.ndim==1 else observed.shape[0]
                    output_shape = (*observed.shape[:-1],rows)
                else:
                    if (type(groups) is not int or groups<=0 or rows%groups or
                            (rows//groups)%self.row_alignment or observed.ndim not in (2,3) or
                            observed.shape[-2:]!=(groups,columns)):
                        raise ValueError('grouped FP8 rows require exact 32-row-aligned groups')
                    batch = 1 if observed.ndim==2 else observed.shape[0]
                    output_shape = (*observed.shape[:-2],groups,rows//groups)
                start,stop = self._row_bounds(rows,self.rank)
                extents = [self._row_bounds(rows,r) for r in range(self.world_size)]
                maximum = max(right-left for left,right in extents)
                tile_rows = self._tile_rows(source)
                scratch = (observed.size*16+batch*(rows*12+maximum*(self.world_size+2)*8)+
                           self.provider.max_projection_bytes+4*(columns+2*tile_rows))
                if scratch > self.max_bytes:
                    raise MemoryError('packed rank projection and gather exceed admission')
                vector = np.array(observed,dtype=np.float32,order='C',copy=True)
                descriptor = (source,observed.shape,groups,self.identity)
            except BaseException as failure:
                message=_failure_type_name(failure)
                error=(message+': cooperative interruption'
                       if isinstance(failure,(InterruptedError,KeyboardInterrupt)) else message)
            self._agree(descriptor,error)
            input_digest=None
            with self.local_stage((descriptor,'packed-input-fingerprint')):
                input_digest=hashlib.sha256(vector.tobytes()).hexdigest()
            self._agree((descriptor,input_digest))
            self._active = True
            try:
                local,error = None,None
                try:
                    self._check_stage_cancelled()
                    local = np.empty((batch,stop-start),np.float32)
                    vectors = vector.reshape(batch,1 if groups is None else groups,columns)
                    for begin in range(start,stop,tile_rows):
                        self._check_stage_cancelled()
                        end = min(stop,begin+tile_rows)
                        # Grouped products use a different input for each group.
                        # Split at the group boundary before leasing packed rows.
                        while begin<end:
                            owner = 0 if groups is None else begin//(rows//groups)
                            right = end if groups is None else min(end,(owner+1)*(rows//groups))
                            with self.provider.lease(source,row_start=begin,row_count=right-begin) as tables:
                                scales = tables[source.removesuffix('weight')+'scale']
                                for index in range(batch):
                                    self._check_stage_cancelled()
                                    result = self.kernel.matvec(tables[source],scales,vectors[index,owner],
                                                               rows=right-begin,cols=columns)
                                    if not np.isfinite(result).all():
                                        raise ArithmeticError('nonfinite packed shard output')
                                    local[index,begin-start:right-start] = result
                            begin = right
                    self._check_stage_cancelled()
                except BaseException as failure:
                    message=_failure_type_name(failure)
                    error=(message+': cooperative interruption'
                           if isinstance(failure,(InterruptedError,KeyboardInterrupt)) else message)
                self._agree(descriptor,error)
                torch = self.torch
                with torch.inference_mode():
                    with self.local_stage((descriptor,'gather-allocation')):
                        padded = torch.zeros((batch,maximum),dtype=torch.float32,device=self.device)
                        padded[:,:stop-start] = torch.tensor(local,dtype=torch.float32,device=self.device)
                        gathered = [torch.empty_like(padded) for _ in range(self.world_size)]
                    self._collective('all_gather',gathered,padded,group=self.group)
                    with self.local_stage((descriptor,'gather-export')):
                        output = torch.cat([part[:,:right-left] for part,(left,right) in zip(gathered,extents)],dim=-1)
                        if not bool(torch.isfinite(output).all()):
                            raise ArithmeticError('nonfinite assembled packed collective')
                        result = output.reshape(output_shape).cpu().numpy().copy()
                    return result
            finally:
                self._active = False

    def close(self):
        with self._lock:
            if self._active:
                raise RuntimeError('cannot close active packed projection bank')
            self._closed = True
            self._packed.clear()
            self._pins.clear()


class PackedMXFP4TensorParallelProjections(PackedFP8TensorParallelProjections):
    """Original rowwise packed experts; only each rank's output rows are read.

    Nonlinear gate/up activation remains replicated between complete gathered
    projections. All namespaces, including MTP stages, retain original names.
    """
    provider_format = 'e2m1-low-first-e8m0-g32'
    row_alignment = 1
    numerical_policy = 'native-cpu-rowwise-mxfp4-row-gather-v1'

    def _row_bounds(self, rows, rank):
        return shard_bounds(rows, rank, self.world_size, allow_empty=True)

    def _tile_rows(self, source):
        _, columns = self.provider.tensor_shapes[source]
        limit=getattr(self.kernel,'tile_rows',None)
        if callable(limit):return min(limit(columns),self.provider.tile_rows(source))
        available = self.kernel.max_bytes - 4*columns
        count = available // (columns//2 + columns//32 + 8)
        if count <= 0:
            raise MemoryError('native FP4 input and one output row exceed kernel admission')
        return min(count, self.provider.tile_rows(source))


__all__ = ['PackedFP8TensorParallelProjections','PackedMXFP4TensorParallelProjections','block_row_bounds']
