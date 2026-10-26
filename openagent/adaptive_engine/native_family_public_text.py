"""Owned publisher text decoder executed through native Segment and Edge ABI."""
from types import SimpleNamespace
import threading
from functools import wraps

import numpy as np

from .native_family_public_adapters import NativeFamilyPublicAdapters
from .segment_edge_runtime import SegmentEdgeExecution,SegmentRange,_cancel_requested


class _Cancellation:
    def __init__(self,callback):self.callback=callback
    def is_set(self):
        if self.callback is None:return False
        # This wrapper consumes a Python cancellation predicate, not an ABI
        # callback. Normalize its ordinary truthy/falsey result to the exact
        # integer wire values before sharing the ABI validator.
        return _cancel_requested(lambda _user:int(bool(self.callback())),None)


def _exclusive(operation):
    @wraps(operation)
    def owned(self,*args,**kwargs):
        with self._lock:
            if getattr(self,'_executing',False):
                raise RuntimeError('native public transaction owns continuation state')
            self._executing=True
            try:return operation(self,*args,**kwargs)
            finally:self._executing=False
    return owned


class NativeFamilyPublicTextDecoder:
    implementation='native-segment-edge-publisher-text-fp32'
    native_range_execution=True

    def __init__(self,runtime_library,decoder,*,directory,max_bytes,max_work_bytes,
                 max_rows=32,max_snapshots=4,max_snapshot_bytes=None,layer_ranges=None):
        self._lock=threading.RLock();self._closed=False
        self.owner=self.execution=None
        self.config=decoder.config
        self.contract=getattr(decoder,'contract',decoder.config)
        self.max_bytes=max_bytes
        if type(max_bytes) is not int or max_bytes<4096:
            raise ValueError('positive bounded public text aggregate admission required')
        if type(max_work_bytes) is not int or max_work_bytes<=0:
            raise ValueError('positive bounded native work admission required')
        self._host_limit=max_bytes//4
        self._transfer_limit=min(max_work_bytes,self._host_limit//4)
        self.family='glm5_next_text' if hasattr(self.contract,'hc_mult') else 'inkling_text'
        self.max_tokens=(decoder.max_tokens if self.family=='glm5_next_text'
                         else decoder.config['max_position_embeddings'])
        if type(max_snapshots) is not int or not 1<=max_snapshots<=62:
            raise ValueError('bounded native public snapshot count required')
        layers=(decoder.config.num_hidden_layers if self.family=='glm5_next_text'
                else decoder.config['num_hidden_layers'])
        if layer_ranges is None:partitions=((0,layers),)
        else:
            if type(layer_ranges) not in (tuple,list) or not layer_ranges:
                raise ValueError('explicit nonempty native publisher layer partitions required')
            partitions=[];cursor=0
            for pair in layer_ranges:
                if type(pair) not in (tuple,list) or len(pair)!=2 or any(type(v) is not int for v in pair) or pair[0]!=cursor or not cursor<pair[1]<=layers:
                    raise ValueError('ordered contiguous layer partitions must cover the complete decoder')
                partitions.append(tuple(pair));cursor=pair[1]
            if cursor!=layers:raise ValueError('native layer partitions must end at the final layer')
            partitions=tuple(partitions)
        self.layer_ranges=partitions
        snapshot_bytes=max_bytes if max_snapshot_bytes is None else max_snapshot_bytes
        try:
            self.owner=NativeFamilyPublicAdapters(runtime_library,decoder,model_directory=directory,
                max_resource_bytes=max_bytes-self._host_limit,max_work_bytes=max_work_bytes,max_batch_rows=max_rows,
                own_decoder=True,enable_layer_ranges=layer_ranges is not None)
            self.execution=SegmentEdgeExecution(runtime_library,edge_adapter=self.owner.edge_id,
                model_directory=str(self.owner.root),checkpoint_identity=self.owner.identity,
                ranges=tuple(SegmentRange(self.owner.segment_id,str(self.owner.root),begin,end)
                    for begin,end in partitions),
                context_tokens=self.owner.context,native_memory_bytes=max_bytes-self._host_limit,
                max_transfer_bytes=self._transfer_limit,max_snapshot_bytes=snapshot_bytes,
                max_snapshots=max_snapshots+2)
            self._stream_wire=self.family=='glm5_next_text' and self.owner.layer_ranges_enabled
            self.routing_observation_supported=self.owner.routing_observation_supported
        except BaseException as error:
            try:self.close()
            except BaseException as failure:
                from .failure_reporting import retain_cleanup_failure
                retain_cleanup_failure(error,self.close,failure,
                    'native public text cleanup retained: ')
            raise

    @property
    def identity(self):return self.execution.identity
    @identity.setter
    def identity(self,value):
        self._check()
        # Segment/Edge snapshots bind to this digest.  Allowing a caller to
        # replace it after construction could make a cache from another model
        # appear compatible with this adapter's physical state.
        if value!=self.execution.identity:
            raise ValueError('native public ABI identity is derived from the admitted model and cannot be overridden')
    @property
    def position(self):return self.execution.position

    def _check(self):
        if self._closed or self.execution is None:raise RuntimeError('closed native public text decoder')

    @_exclusive
    def embed_tokens(self,ids,*,cancelled=None):
        with self._lock:
            self._check();values=np.asarray(ids)
            if values.ndim!=2 or values.shape[0]!=1 or values.dtype.kind not in 'iu' or not values.size:
                raise ValueError('batch-one publisher embedding token IDs required')
            width=self.owner.hidden_size
            if values.size*(width*8+8)+self._transfer_limit>self._host_limit:
                raise MemoryError('complete native publisher embedding staging exceeds admission')
            result=np.empty((*values.shape,width),np.float32)
            rows=self.execution.max_prefill_tokens
            for begin in range(0,values.size,rows):
                end=min(values.size,begin+rows)
                wire=self.execution.embed_tokens(
                    tuple(int(token) for token in values[0,begin:end]),cancel=_Cancellation(cancelled))
                result[0,begin:end]=wire[:,:width]
            return result

    @_exclusive
    def forward(self,ids,*,use_cache=True,inputs_embeds=None,cancelled=None,
                attention_mask=None,position_ids=None,capture_routing=False):
        with self._lock:
            self._check();values=np.asarray(ids)
            if (type(use_cache) is not bool or type(capture_routing) is not bool or
                    attention_mask is not None or position_ids is not None):
                raise ValueError('contiguous native public publisher text transaction required')
            if capture_routing and not self.routing_observation_supported:
                raise ValueError('this native public family has no executed route observation implementation')
            if values.ndim!=2 or values.shape[0]!=1 or values.dtype.kind not in 'iu' or not values.size:
                raise ValueError('batch-one native publisher text IDs required')
            width=self.owner.hidden_size;vocab=self.owner.vocabulary
            start=self.position if use_cache else 0
            if start+values.size>self.max_tokens:
                raise ValueError('native publisher text context capacity exceeded')
            if np.any(values<0) or np.any(values>=vocab):raise ValueError('native publisher text token outside vocabulary')
            media=None if inputs_embeds is None else np.asarray(inputs_embeds)
            if media is not None and (media.shape!=(*values.shape,width) or media.dtype!=np.float32):
                raise ValueError('matching FP32 native publisher media rows required')
            needed=values.size*(width+vocab)*4+values.nbytes+(0 if media is None else media.nbytes)
            if needed+2*self._transfer_limit>self._host_limit:
                raise MemoryError('complete native publisher output exceeds admission')
            logits=np.empty((1,values.size,vocab),np.float32)
            hidden=np.empty((1,values.size,width),np.float32)
            receipt=self.execution.snapshot();primary=None;route_chunks=[]
            try:
                if not use_cache:self.execution.reset()
                rows=self.execution.max_prefill_tokens
                for begin in range(0,values.size,rows):
                    end=min(values.size,begin+rows)
                    incoming=None if media is None else media[0,begin:end]
                    if incoming is not None and self._stream_wire:
                        wire=np.zeros((end-begin,self.owner.transport_width),np.float32)
                        for stream in range(self.owner.streams):
                            wire[:,stream*width:(stream+1)*width]=incoming
                        incoming=wire
                    executed=self.execution.forward(
                        tuple(int(token) for token in values[0,begin:end]),cancel=_Cancellation(cancelled),
                        inputs_embeds=incoming,return_hidden=True,capture_routes=capture_routing,
                        route_width=((self.contract.n_routed_experts if self.family=='glm5_next_text' else
                                      self.contract['n_routed_experts']+self.contract['n_shared_experts'])
                                     if capture_routing else None))
                    if capture_routing:
                        chunk_logits,chunk_hidden,chunk_routes=executed
                        route_chunks.append(chunk_routes)
                    else:
                        chunk_logits,chunk_hidden=executed
                    if self._stream_wire:chunk_hidden=self.owner.final_hidden(chunk_hidden)
                    logits[0,begin:end]=chunk_logits;hidden[0,begin:end]=chunk_hidden
                if not use_cache:self.execution.restore(receipt)
                result={'logits':logits,'last_hidden_state':hidden,'position':start+values.size}
                if capture_routing:
                    result['routing_weights']=np.concatenate(route_chunks,axis=0)[None]
                return SimpleNamespace(**result)
            except BaseException as error:
                primary=error
                try:self.execution.restore(receipt)
                except BaseException as failure:
                    from .failure_reporting import add_failure_note, failure_summary
                    add_failure_note(error,'native public text rollback failed: '+failure_summary(failure))
                raise
            finally:
                try:self.execution.release_snapshot(receipt)
                except BaseException as failure:
                    if primary is not None:
                        from .failure_reporting import add_failure_note, failure_summary
                        add_failure_note(primary,'native text snapshot cleanup retained: '+failure_summary(failure))
                    else:
                        # Snapshot disposal is part of request commit: callers
                        # must not see a failed request while the native KV
                        # state silently remains advanced. Keep the still-owned
                        # receipt for executor cleanup retry and restore it
                        # before propagating the disposal failure.
                        try:
                            self.execution.restore(receipt)
                        except BaseException as rollback:
                            from .failure_reporting import add_failure_note, failure_summary
                            add_failure_note(failure,
                                'native text state rollback after snapshot cleanup failure failed: '+
                                failure_summary(rollback))
                        raise

    @_exclusive
    def snapshot(self):self._check();return self.execution.snapshot()
    @_exclusive
    def restore(self,receipt):self._check();return self.execution.restore(receipt)
    @_exclusive
    def release_snapshot(self,receipt):self._check();return self.execution.release_snapshot(receipt)
    @_exclusive
    def reset(self):self._check();return self.execution.reset()
    @_exclusive
    def write_cache(self,output,*,max_bytes):self._check();return self.execution.write_cache(output,max_bytes=max_bytes)
    @_exclusive
    def restore_cache(self,source,*,max_bytes):self._check();return self.execution.restore_cache(source,max_bytes=max_bytes)
    def status(self):
        with self._lock:
            self._check()
            return dict(self.execution.status(),implementation=self.implementation,family=self.family)
    @_exclusive
    def close(self):
        with self._lock:
            if self._closed:return
            if self.execution is not None:self.execution.close();self.execution=None
            if self.owner is not None:self.owner.close();self.owner=None
            self._closed=True


__all__=['NativeFamilyPublicTextDecoder']
