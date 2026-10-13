"""Original BF16 row streaming with rank agreement and collective assembly."""
from contextlib import contextmanager
import hashlib

import numpy as np

from .tensor_parallel import TensorParallelProjections,shard_bounds
from .v41_execution.bf16_projections import PublisherBF16Projections


class BF16TensorParallelProjections(TensorParallelProjections):
    boundary_observation_supported=False
    def __init__(self,*,provider,checkpoint_identity,group,device,torch_version,max_bytes):
        super().__init__(group=group,device=device,torch_version=torch_version,max_bytes=max_bytes,dtype='float32')
        if type(provider) is not PublisherBF16Projections:
            raise TypeError('original publisher BF16 row owner required')
        if type(checkpoint_identity) is not str or not checkpoint_identity:
            raise ValueError('identified complete distributed BF16 checkpoint required')
        self.provider=provider;self.checkpoint_identity=checkpoint_identity
        self._provider_identity=provider.identity;self._sources={};self._active=False

    @property
    def identity(self):
        return hashlib.sha256(repr((self.checkpoint_identity,self.world_size,self.backend,
            self.torch_version,self.device.type,'native-bf16-logical-streamed-row-gather-v2',
            tuple(sorted(self._sources.items())),tuple(sorted(self._pins.items())))).encode()).hexdigest()

    def contains_projection_owner(self,owner):return owner is self.provider
    @contextmanager
    def cancellation_scope(self,event):
        with super().cancellation_scope(event), self.provider.cancellation_scope(event):
            yield self

    def register_source(self,source):
        with self._lock:
            error=descriptor=None
            try:
                if self._closed or self._active or source in self._sources:
                    raise RuntimeError('idle open BF16 rank and unique source required')
                shape=self.provider.shape(source)
                if len(shape)!=2 or any(type(value) is not int or value<=0 for value in shape):
                    raise ValueError('positive original BF16 matrix geometry required')
                descriptor=(source,shape,self.checkpoint_identity,self.backend,self.torch_version,self.device.type)
            except Exception as failure:error=str(failure) or type(failure).__name__
            self._agree(descriptor,error)
            identities=[None]*self.world_size
            self._collective('all_gather_object',identities,self._provider_identity,group=self.group)
            self._sources[source]=shape;self._pins[source]=tuple(identities)

    def handles(self,source):return not self._closed and source in self._sources
    def shape(self,source):return self._sources[source]
    def linear(self,source,value):return self._project(source,value,groups=None)
    def grouped_linear(self,source,value,*,groups):return self._project(source,value,groups=groups)

    def _project(self,source,value,*,groups):
        with self._lock:
            error=descriptor=vector=None
            try:
                if self._closed or self._active or self.provider.identity!=self._provider_identity:
                    raise RuntimeError('open idle immutable BF16 rank provider required')
                rows,columns=self._sources[source];observed=np.asarray(value)
                if observed.dtype.kind!='f' or not observed.size or not np.isfinite(observed).all():
                    raise ValueError('finite BF16 collective activations required')
                if groups is None:
                    if observed.ndim not in (1,2) or observed.shape[-1]!=columns:
                        raise ValueError('replicated BF16 vector or matrix required')
                    batch=1 if observed.ndim==1 else observed.shape[0]
                    output_shape=(*observed.shape[:-1],rows);count=1
                else:
                    if type(groups) is not int or groups<=0 or rows%groups or observed.ndim not in (2,3) or observed.shape[-2:]!=(groups,columns):
                        raise ValueError('group-local collective BF16 geometry required')
                    batch=1 if observed.ndim==2 else observed.shape[0];count=groups
                    output_shape=(*observed.shape[:-2],groups,rows//groups)
                extents=[shard_bounds(rows,rank,self.world_size,allow_empty=True) for rank in range(self.world_size)]
                start,stop=extents[self.rank];maximum=max(right-left for left,right in extents)
                needed=observed.size*16+batch*(rows*12+maximum*(self.world_size+2)*8)+self.provider.max_bytes
                if needed>self.max_bytes:raise MemoryError('BF16 rank conversion and collective export exceed admission')
                vector=np.array(observed,dtype=np.float32,order='C',copy=True)
                arithmetic=self.provider._arithmetic
                if arithmetic is None:raise RuntimeError('native BF16 arithmetic owner must be bound before collective execution')
                libraries=tuple(record.sha256 for record in arithmetic._library_identities)
                policy=getattr(arithmetic,'collective_policy_identity',None)
                if type(policy) is not str or not policy:raise ValueError('rank-independent arithmetic policy identity required')
                descriptor=(source,observed.shape,groups,self.identity,libraries,policy)
            except Exception as failure:error=str(failure) or type(failure).__name__
            self._agree(descriptor,error)
            self._agree((descriptor,hashlib.sha256(vector.tobytes()).hexdigest()))
            self._active=True
            try:
                local=None;error=None
                try:
                    self._check_stage_cancelled()
                    local=self.provider.project_rows(source,vector.reshape(batch,count,columns),
                        row_start=start,row_count=stop-start,groups=count)
                    self._check_stage_cancelled()
                except Exception as failure:error=str(failure) or type(failure).__name__
                self._agree(descriptor,error)
                torch=self.torch
                with torch.inference_mode():
                    with self.local_stage((descriptor,'allocate-row-gather')):
                        padded=torch.zeros((batch,maximum),dtype=torch.float32,device=self.device)
                        padded[:,:stop-start]=torch.tensor(local,device=self.device,dtype=torch.float32)
                        gathered=[torch.empty_like(padded) for _ in range(self.world_size)]
                    self._collective('all_gather',gathered,padded,group=self.group)
                    with self.local_stage((descriptor,'export-row-gather')):
                        output=torch.cat([part[:,:right-left] for part,(left,right) in zip(gathered,extents)],dim=-1)
                        if not bool(torch.isfinite(output).all()):raise ArithmeticError('nonfinite BF16 assembled collective')
                        return output.reshape(output_shape).cpu().numpy().copy()
            finally:self._active=False

    def close(self):
        with self._lock:
            if self._active:raise RuntimeError('cannot close an active BF16 collective')
            self._sources.clear();self._pins.clear();self._closed=True


__all__=['BF16TensorParallelProjections']
