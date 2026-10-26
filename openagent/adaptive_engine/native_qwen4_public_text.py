"""Qwen4 text transactions on the native public Segment/Edge dispatcher.

Python marshals lexical and media inputs; every decoder range, final hidden
projection and vocabulary projection executes in the registered native model.
"""
import threading
import struct
import hashlib
import hmac
from contextlib import nullcontext
from functools import wraps
from types import SimpleNamespace
import numpy as np
from .native_qwen4_public_adapters import NativeQwen4PublicAdapters
from .native_family_public_text import _Cancellation
from .qwen4_segment_wire import Qwen4SegmentWire
from .segment_edge_runtime import SegmentEdgeExecution,SegmentRange


def _exclusive(operation):
    """Serialize public transactions and reject callback-driven reentry."""
    @wraps(operation)
    def owned(self,*args,**kwargs):
        with self._lock:
            if getattr(self,'_executing',False):raise RuntimeError('native Qwen4 transaction owns continuation state')
            self._executing=True
            try:return operation(self,*args,**kwargs)
            finally:self._executing=False
    return owned


class NativeQwen4PublicTextDecoder:
    implementation='native-segment-edge-qwen4-text-fp32'
    native_range_execution=True

    def __init__(self,runtime_library,decoder,*,directory,max_bytes,max_work_bytes,
                 max_rows=32,max_snapshots=4,max_snapshot_bytes=None,layer_ranges=None):
        self._lock=threading.RLock();self._closed=False;self.owner=self.execution=None
        self.config=decoder.config;self.contract=self.config;self.family='qwen4_exp_text'
        self.mrope_section=decoder.mrope_section
        self.max_tokens=self.config.max_position_embeddings;self.max_bytes=max_bytes
        if type(max_bytes) is not int or max_bytes<4096:raise ValueError('bounded aggregate admission required')
        if type(max_snapshots) is not int or not 1<=max_snapshots<=62:raise ValueError('bounded snapshot inventory required')
        self._host_limit=max_bytes//4;self._transfer_limit=min(max_work_bytes,self._host_limit//4)
        self.wire=Qwen4SegmentWire(self.config.hidden_size,self.config.hc_count,self.config.vocab_size,
            self.max_tokens,decoder.mrope_section is not None)
        partitions=((0,self.config.num_hidden_layers),) if layer_ranges is None else tuple(layer_ranges)
        cursor=0
        for pair in partitions:
            if (type(pair) not in (tuple,list) or len(pair)!=2 or any(type(v) is not int for v in pair)
                    or pair[0]!=cursor or not cursor<pair[1]<=self.config.num_hidden_layers):
                raise ValueError('ordered contiguous complete native layer partitions required')
            cursor=pair[1]
        if cursor!=self.config.num_hidden_layers:raise ValueError('layer partitions must cover the complete decoder')
        self.layer_ranges=tuple(tuple(pair) for pair in partitions)
        try:
            self.owner=NativeQwen4PublicAdapters(runtime_library,decoder,model_directory=directory,
                max_resource_bytes=max_bytes-self._host_limit,max_work_bytes=max_work_bytes,
                max_batch_rows=max_rows,own_decoder=True)
            self.execution=SegmentEdgeExecution(runtime_library,edge_adapter=self.owner.edge_id,
                model_directory=str(self.owner.root),checkpoint_identity=self.owner.identity,
                ranges=tuple(SegmentRange(self.owner.segment_id,str(self.owner.root),a,b) for a,b in self.layer_ranges),
                context_tokens=self.max_tokens,native_memory_bytes=max_bytes-self._host_limit,
                max_transfer_bytes=self._transfer_limit,max_snapshots=max_snapshots+2,
                max_snapshot_bytes=max_bytes if max_snapshot_bytes is None else max_snapshot_bytes)
        except BaseException as error:
            try:self.close()
            except BaseException as cleanup:
                from .failure_reporting import retain_cleanup_failure
                retain_cleanup_failure(error,self.close,cleanup,
                    'native Qwen4 public cleanup retained: ')
            raise

    @property
    def identity(self):return self.execution.identity
    @identity.setter
    def identity(self,value):
        self._check()
        # The dispatcher digest binds the registered model and physical cache
        # schema. A caller override could make foreign continuation state
        # appear to belong to this Qwen4 adapter.
        if value!=self.execution.identity:
            raise ValueError('native Qwen4 public identity is derived from the admitted model and cannot be overridden')
    @property
    def position(self):return self.execution.position

    def _check(self):
        if self._closed or self.execution is None:raise RuntimeError('closed public Qwen4 decoder')

    def _admit_prefix(self,mask,start):
        """Compare against native-owned masks, including restored continuations.

        Capture uses the dispatcher's bounded snapshot storage. Read only the
        occupied mask tail, in chunks; do not reconstruct cache tensors in Python.
        """
        if not start:return
        with self.execution._lock:
            self.execution._check();files=self.execution._capture()
            try:
                for state,(begin,end) in zip(files,self.layer_ranges,strict=True):
                    state.file.seek(0);digest=hashlib.sha256();remaining=state.size
                    while remaining:
                        data=state.file.read(min(65536,remaining))
                        if not data:raise ValueError('native Qwen4 snapshot truncated')
                        digest.update(data);remaining-=len(data)
                    if not hmac.compare_digest(digest.digest(),state.digest):
                        raise ValueError('native Qwen4 snapshot contents changed')
                    state.file.seek(0);header=state.file.read(128)
                    if len(header)!=128 or header[:8]!=b'OAQ4PUB1':
                        raise ValueError('native Qwen4 snapshot contract differs')
                    position,payload,first,last=struct.unpack_from('<QQQQ',header,8)
                    context,width,mrope=struct.unpack_from('<QQQ',header,104)
                    if (position!=start or (first,last)!=(begin,end) or context!=self.max_tokens
                            or width!=self.wire.width or mrope!=int(self.wire.mrope)
                            or header[40:104]!=self.owner.identity.encode('ascii')
                            or state.size!=160+payload):
                        raise ValueError('native Qwen4 continuation inventory differs')
                    tail=start*(1+8*(3 if self.wire.mrope else 1))
                    if payload<tail:raise ValueError('native Qwen4 mask extent is truncated')
                    state.file.seek(128+payload-tail)
                    for offset in range(0,start,65536):
                        count=min(65536,start-offset);data=state.file.read(count)
                        if len(data)!=count or not np.array_equal(np.frombuffer(data,np.uint8),mask[0,offset:offset+count]):
                            raise ValueError('cached attention mask changed; reset and replay required')
            finally:self.execution._dispose(files)

    @_exclusive
    def embed_tokens(self,ids,*,cancelled=None):
        with self._lock:
            self._check();ids=np.asarray(ids)
            if ids.ndim!=2 or ids.shape[0]!=1 or ids.dtype.kind not in 'iu' or not ids.size:
                raise ValueError('batch-one original embedding IDs required')
            if np.any(ids<0) or np.any(ids>=self.wire.vocabulary):raise ValueError('token outside vocabulary')
            if ids.size*self.wire.hidden*4+2*self._transfer_limit>self._host_limit:
                raise MemoryError('embedding output exceeds host admission')
            result=np.empty((*ids.shape,self.wire.hidden),np.float32)
            rows=self.execution.max_prefill_tokens
            for begin in range(0,ids.size,rows):
                end=min(ids.size,begin+rows)
                wire=self.execution.embed_tokens(tuple(int(v) for v in ids[0,begin:end]),cancel=_Cancellation(cancelled))
                result[0,begin:end]=wire[:,:self.wire.hidden]
            return result

    routing_observation_supported = True

    @_exclusive
    def forward(self,ids,*,use_cache=True,inputs_embeds=None,attention_mask=None,position_ids=None,cancelled=None,capture_routing=False):
        with self._lock:
            self._check();ids=np.asarray(ids)
            if type(capture_routing) is not bool:
                raise ValueError('explicit native routing observation policy required')
            if type(use_cache) is not bool or ids.ndim!=2 or ids.shape[0]!=1 or ids.dtype.kind not in 'iu' or not ids.size:
                raise ValueError('batch-one original token transaction required')
            if np.any(ids<0) or np.any(ids>=self.wire.vocabulary):raise ValueError('token outside vocabulary')
            start=self.position if use_cache else 0;end=start+ids.size
            if end>self.max_tokens:raise ValueError('native context capacity exceeded')
            mask=np.ones((1,end),np.uint8) if attention_mask is None else np.asarray(attention_mask)
            if mask.shape!=(1,end) or mask.dtype.kind not in 'biu' or not np.isin(mask,[0,1]).all():
                raise ValueError('complete occupied binary mask required')
            self._admit_prefix(mask,start)
            positions=np.arange(start,end,dtype=np.int64)[None,:] if position_ids is None else np.asarray(position_ids)
            if positions.shape==(1,ids.size):positions=np.broadcast_to(positions,(3,1,ids.size))
            if positions.shape!=(3,1,ids.size) or positions.dtype.kind not in 'iu':
                raise ValueError('occupied text or three-axis position IDs required')
            media=None if inputs_embeds is None else np.asarray(inputs_embeds)
            if media is not None and (media.dtype!=np.float32 or media.shape!=(1,ids.size,self.wire.hidden) or not np.isfinite(media).all()):
                raise ValueError('occupied finite FP32 media embedding rows required')
            required=ids.size*(self.wire.vocabulary+self.wire.hidden)*4+ids.nbytes+mask.nbytes+positions.nbytes
            if capture_routing:
                required += ids.size*self.config.num_hidden_layers*self.config.num_experts*32
            if required+3*self._transfer_limit+(0 if media is None else media.nbytes)>self._host_limit:
                raise MemoryError('whole transaction output exceeds host admission')
            logits=np.empty((1,ids.size,self.wire.vocabulary),np.float32)
            hidden=np.empty((1,ids.size,self.wire.hidden),np.float32)
            observations = {}
            if capture_routing:
                shape=(1,ids.size,self.config.num_hidden_layers,self.config.num_experts)
                observations={'router_logits':np.empty(shape,np.float32),
                              'routing_weights':np.zeros(shape,np.float32)}
            receipt=self.execution.snapshot();primary=None
            try:
                if not use_cache:self.execution.reset()
                rows=max(1,min(self.execution.max_prefill_tokens,
                    self._transfer_limit//(4*(4*self.wire.width+self.wire.vocabulary+1))))
                if capture_routing:
                    route_rows=self._transfer_limit//(self.config.num_experts*12+self.config.num_experts_per_tok*40)
                    if route_rows < 1:
                        raise MemoryError('one native routing row exceeds transfer admission')
                    rows=min(rows,route_rows)
                for begin in range(0,ids.size,rows):
                    stop=min(ids.size,begin+rows);tokens=tuple(int(v) for v in ids[0,begin:stop])
                    if media is None:
                        lexical=self.execution.embed_tokens(tokens,cancel=_Cancellation(cancelled))
                        activation=lexical[:,:self.wire.activation_width]
                    else:
                        activation=np.tile(media[0,begin:stop],(1,self.wire.streams))
                    incoming=self.wire.pack(activation,ids[0,begin:stop],mask[0,start+begin:start+stop],
                        positions[:,0,begin:stop],max_bytes=self._transfer_limit)
                    observed = [0]*self.config.num_hidden_layers
                    def capture(layer,raw,selected,scales):
                        if cancelled is not None and cancelled():
                            raise InterruptedError('native routing observation cancelled')
                        c=self.config;count=len(raw);offset=observed[layer]
                        if (raw.shape!=(count,c.num_experts) or selected.shape!=(count,c.num_experts_per_tok)
                                or scales.shape!=selected.shape or offset+count>stop-begin or
                                np.any(selected<0) or np.any(selected>=c.num_experts) or
                                not np.isfinite(raw).all() or not np.isfinite(scales).all() or
                                any(np.unique(row).size!=c.num_experts_per_tok for row in selected)):
                            raise ValueError('native executed routing trace differs from complete decoder contract')
                        first,last=begin+offset,begin+offset+count
                        observations['router_logits'][0,first:last,layer,:]=raw
                        dense=np.zeros((count,c.num_experts),np.float32)
                        np.put_along_axis(dense,selected,scales,axis=-1)
                        observations['routing_weights'][0,first:last,layer,:]=dense
                        observed[layer]+=count
                    scope=(self.owner.routing_scope(capture,max_host_bytes=self._transfer_limit)
                           if capture_routing else nullcontext())
                    with scope:
                        chunk,residual=self.execution.forward(tokens,cancel=_Cancellation(cancelled),inputs_embeds=incoming,
                            return_hidden=True)
                    if capture_routing and any(count!=stop-begin for count in observed):
                        raise ValueError('complete native public routing inventory required')
                    logits[0,begin:stop]=chunk
                    hidden[0,begin:stop]=self.owner.final_hidden(residual,cancelled=cancelled,
                        max_host_bytes=self._transfer_limit)
                if not use_cache:self.execution.restore(receipt)
                return SimpleNamespace(logits=logits,last_hidden_state=hidden,position=end,**observations)
            except BaseException as error:
                primary=error
                try:self.execution.restore(receipt)
                except BaseException as failure:
                    from .failure_reporting import add_failure_note, failure_summary
                    add_failure_note(error,'native Qwen4 rollback failed: '+failure_summary(failure))
                raise
            finally:
                try:self.execution.release_snapshot(receipt)
                except BaseException as failure:
                    if primary is None:raise
                    from .failure_reporting import add_failure_note, failure_summary
                    add_failure_note(primary,'native Qwen4 receipt cleanup retained: '+
                        failure_summary(failure))

    @_exclusive
    def snapshot(self):
        self._check();return self.execution.snapshot()
    @_exclusive
    def restore(self,receipt):
        self._check();return self.execution.restore(receipt)
    @_exclusive
    def release_snapshot(self,receipt):
        self._check();return self.execution.release_snapshot(receipt)
    @_exclusive
    def reset(self):
        self._check();return self.execution.reset()
    @_exclusive
    def write_cache(self,output,*,max_bytes):
        self._check();return self.execution.write_cache(output,max_bytes=max_bytes)
    @_exclusive
    def restore_cache(self,source,*,max_bytes,cancelled=None):
        self._check();return self.execution.restore_cache(source,max_bytes=max_bytes,cancelled=cancelled)
    @_exclusive
    def status(self):
        self._check();return dict(self.execution.status(),implementation=self.implementation,family=self.family)
    @_exclusive
    def close(self):
        with self._lock:
            if self._closed:return
            if self.execution is not None:self.execution.close();self.execution=None
            if self.owner is not None:self.owner.close();self.owner=None
            self._closed=True


__all__=['NativeQwen4PublicTextDecoder']
