"""Native Inkling dense MTP stages with shared embedding and vocabulary head.

The caller supplies shifted token ids and the target's aligned hidden states.
Each depth owns its real attention/convolution continuation independently.
"""
from dataclasses import dataclass
from math import prod
from uuid import uuid4
import hashlib
import threading
import numpy as np
from .native_inkling import NativeInklingDecoder
from .independent_inkling import GLOBAL_NAMES,LAYER_NAMES
from .flat_native_session import FlatNativeSession
from .native_media_ops import NativeMediaOps
from .failure_reporting import add_failure_note, failure_summary


def _canonical_mtp(original):
    name=original
    for prefix in ('model.language_model.mtp.','model.mtp.','mtp.'):
        if name.startswith(prefix):name=name[len(prefix):];break
    name=name.replace('.transformer_block.','.mtp_block.')
    name=name.replace('.hidden_norm.','.hnorm.').replace('.embed_norm.','.enorm.').replace('.input_proj.','.eh_proj.')
    if name.startswith('chain_norm.'):name='shared_post_norm.'+name[len('chain_norm.'):]
    return name


def admit_mtp_inventory(text_config,auxiliary_config,headers,*,max_total_bytes=None):
    """Admit complete original MTP headers before any component payload read."""
    from .independent_inkling import IndependentInklingDecoder
    if (type(auxiliary_config) is not dict or
            set(auxiliary_config)-{'num_mtp_layers','mtp_local_layer_ids',
                'chain_hidden_post_norm','mtp_hidden_states_first'}):
        raise ValueError('Inkling MTP configuration contains fields without an admitted consumer')
    depth=auxiliary_config.get('num_mtp_layers',0)
    if type(depth) is not int or not 0<=depth<=32:
        raise ValueError('bounded declared Inkling MTP depth required')
    # Do not normalize malformed false-y values (notably ``0`` and ``''``)
    # into the valid empty schedule.  This config controls which MTP blocks
    # use sliding-window attention, so it is part of the physical tensor
    # inventory and must be admitted exactly before any stage weights load.
    local=auxiliary_config.get('mtp_local_layer_ids', [])
    chain=auxiliary_config.get('chain_hidden_post_norm',False)
    hidden_first=auxiliary_config.get('mtp_hidden_states_first',True)
    if (type(chain) is not bool or type(hidden_first) is not bool or
            type(local) not in (list,tuple) or len(set(local))!=len(local) or
            any(type(value) is not int or not 0<=value<depth for value in local)):
        raise ValueError('invalid declared Inkling MTP policy')
    if not depth:
        if headers:raise ValueError('MTP weights require declared Inkling stages')
        return
    normalized={}
    for original,value in headers.items():
        name=_canonical_mtp(original)
        if name in normalized:raise ValueError('duplicate canonical Inkling MTP header')
        normalized[name]=value
    hidden=text_config['hidden_size'];required={}
    if chain:required['shared_post_norm.weight']=(hidden,)
    for index in range(depth):
        prefix=f'layers.{index}.'
        for name,shape in (('enorm.weight',(hidden,)),('hnorm.weight',(hidden,)),('eh_proj.weight',(hidden,2*hidden))):
            required[prefix+name]=shape
        config=dict(text_config);config['num_hidden_layers']=1
        config['layer_types']=['hybrid_sliding' if index in local else 'hybrid']
        config['mlp_layer_types']=['dense']
        planner=IndependentInklingDecoder(config,{},_inventory_only=True)
        for name in LAYER_NAMES:
            required[prefix+'mtp_block.'+name]=planner.tensor_shapes['model.layers.0.'+name]
    optional={'embed_tokens.weight':(text_config['vocab_size'],hidden),
        'shared_head.weight':(text_config['vocab_size'],hidden),
        'shared_head.head.weight':(text_config['vocab_size'],hidden)}
    required.update((name,shape) for name,shape in optional.items() if name in normalized)
    if set(normalized)!=set(required):raise ValueError('complete original Inkling MTP inventory differs')
    if max_total_bytes is not None:
        if type(max_total_bytes) is not int or max_total_bytes<=0:
            raise ValueError('positive Inkling MTP total ownership admission required')
        # Non-alias stage/projection tensors become FP32-owned MTP data. Reject
        # a bank whose unavoidable resident lower bound exceeds the MTP budget
        # while only checkpoint headers are available, before payload reads.
        # Shared embedding/head tensors are compared with the backbone and do
        # not become MTP-owned copies.
        shared={'embed_tokens.weight','shared_head.weight','shared_head.head.weight'}
        resident_lower_bound=sum(prod(shape)*4
            for name,shape in required.items() if name not in shared)
        if resident_lower_bound>max_total_bytes:
            raise MemoryError('declared Inkling MTP resident tensors exceed total ownership admission')
    for name,expected in required.items():
        shape,dtype=normalized[name]
        scalar=name.endswith('global_scale') and tuple(shape)==() and expected==(1,)
        if (tuple(shape)!=expected and not scalar) or dtype not in ('F32','F16','BF16','F8_E4M3'):
            raise ValueError('Inkling MTP source shape/dtype differs: '+name)

@dataclass(frozen=True)
class InklingMTPSnapshot:
    owner: str
    identity: str
    key: str
    position: int

class InklingMTP:
    def __init__(self,backbone,tensors,config,*,text_library,media_library,
                 stage_max_bytes,max_total_bytes,max_work_bytes,max_snapshots=4):
        from .torch_inkling_decoder import TorchInklingDecoder
        if type(backbone) not in (NativeInklingDecoder,TorchInklingDecoder):
            raise ValueError('exact native or device Inkling backbone required')
        self._device=type(backbone) is TorchInklingDecoder
        depth=config.get('num_mtp_layers')
        if type(depth) is not int or not 1<=depth<=32:
            raise ValueError('explicit bounded Inkling MTP depth required')
        if any(type(v) is not int or v<=0 for v in (stage_max_bytes,max_total_bytes,max_work_bytes,max_snapshots)):
            raise ValueError('positive MTP ownership budgets required')
        if max_snapshots>62: raise ValueError('MTP snapshot count leaves no rollback capacity')
        if stage_max_bytes*depth+max_work_bytes>max_total_bytes:
            raise MemoryError('all MTP stage budgets exceed aggregate admission')
        self.backbone=backbone;self.depth=depth;self.max_bytes=max_total_bytes;self.max_work_bytes=max_work_bytes
        # CPU stages retain immutable global arrays independently of the text
        # decoder's library/cache lifetime. Public Segment registration closes
        # that decoder after copying its model into the native runtime.
        self._config=dict(backbone.config)
        self._globals={name:backbone.weights[name] for name in GLOBAL_NAMES}
        self._lock=threading.RLock();self._busy=False;self._closed=False;self._poisoned=False;self._owner=uuid4().hex
        self._snapshots={};self._orphaned_snapshots=[];self._pending_snapshot_handles=[];self.max_snapshots=max_snapshots;self.stages=[];self.weights={};self.ops=None
        self.hidden_first=config.get('mtp_hidden_states_first',True)
        self.chain_norm=config.get('chain_hidden_post_norm',False)
        if type(self.hidden_first) is not bool or type(self.chain_norm) is not bool:
            raise ValueError('explicit Inkling MTP projection/norm switches required')
        # Preserve the same exact schema enforced by admit_mtp_inventory.
        # False-y malformed declarations such as 0 or "" must not be
        # normalized into the valid empty schedule by direct construction.
        local=config.get('mtp_local_layer_ids', [])
        if type(local) not in (list,tuple) or len(set(local))!=len(local) or any(type(v) is not int or not 0<=v<depth for v in local):
            raise ValueError('valid MTP sliding layer schedule required')
        normalized={}
        for original,value in tensors.items():
            name=_canonical_mtp(original)
            if name in normalized: raise ValueError('duplicate canonical Inkling MTP tensor')
            normalized[name]=value
        hidden=backbone.config['hidden_size'];owned=0;consumed=set()
        digest=hashlib.sha256(repr((backbone.config,config,depth,self.hidden_first,self.chain_norm)).encode())
        def weight(name,shape):
            nonlocal owned
            source=np.asarray(normalized[name])
            if source.shape!=shape or source.dtype!=np.float32 or not np.isfinite(source).all():
                raise ValueError('invalid Inkling MTP weight: '+name)
            owned+=source.nbytes
            if 2*owned>max_work_bytes: raise MemoryError('MTP projection weights exceed admission')
            value=np.frombuffer(source.tobytes(),np.float32).reshape(shape)
            self.weights[name]=value;consumed.add(name)
            digest.update(name.encode());digest.update(memoryview(value).cast('B'))
            return value
        try:
            if not self._device:self.ops=NativeMediaOps(media_library)
            if self.chain_norm: weight('shared_post_norm.weight',(hidden,))
            for index in range(depth):
                prefix=f'layers.{index}.'
                for name,shape in (('enorm.weight',(hidden,)),('hnorm.weight',(hidden,)),('eh_proj.weight',(hidden,2*hidden))):
                    weight(prefix+name,shape)
                c=dict(backbone.config);c['num_hidden_layers']=1
                c['layer_types']=['hybrid_sliding' if index in local else 'hybrid'];c['mlp_layer_types']=['dense']
                stage_weights={} if self._device else {name:backbone.weights[name] for name in GLOBAL_NAMES}
                for name in LAYER_NAMES:
                    key=prefix+'mtp_block.'+name
                    if key not in normalized: raise ValueError('missing Inkling dense MTP block weight: '+key)
                    stage_weights['model.layers.0.'+name]=normalized[key];consumed.add(key)
                if self._device:
                    from .torch_inkling_decoder import _SOURCE_ADMISSION as _TEXT_SOURCE_ADMISSION
                    work=min(64*1024**2,stage_max_bytes//8)
                    stage=TorchInklingDecoder(c,stage_weights,device=backbone.device,max_bytes=stage_max_bytes,
                        max_snapshots=max_snapshots+2,max_work_bytes=work,max_head_bytes=work,
                        _borrow_globals={name:backbone.weights[name] for name in GLOBAL_NAMES},
                        _shared_global_identity=backbone.identity,
                        _source_admission=_TEXT_SOURCE_ADMISSION)
                else:
                    stage=NativeInklingDecoder(c,stage_weights,library=text_library,max_bytes=stage_max_bytes,_borrow_globals=True)
                try:
                    session=stage if self._device else FlatNativeSession(stage,family='inkling_text',max_bytes=stage_max_bytes,
                        max_snapshots=max_snapshots+2)
                except BaseException:
                    stage.close();raise
                self.stages.append(session);digest.update(session.identity.encode())
            for name,global_name in (('embed_tokens.weight','model.embed_tokens.weight'),
                                     ('shared_head.weight','lm_head.weight'),('shared_head.head.weight','lm_head.weight')):
                if name in normalized:
                    equal=self._shared_equal(normalized[name],backbone.weights[global_name])
                    if not equal:
                        raise ValueError('MTP shared weight differs from backbone')
                    consumed.add(name)
            if set(normalized)!=consumed: raise ValueError('unowned Inkling MTP checkpoint tensors: '+str(sorted(set(normalized)-consumed)))
            self.identity='openagent-inkling-mtp-v1:'+digest.hexdigest()
            self._device_weights={}
            if self._device:
                resident=sum(value.nbytes for value in self.weights.values())
                if 3*resident>max_work_bytes:raise MemoryError('device MTP projection ownership exceeds admission')
                self._device_weights={name:backbone.torch.tensor(value,device=backbone.device)
                                     for name,value in self.weights.items()}
            else:
                # No CPU operation dereferences the backbone after admission.
                # Keep only shared weight ownership, not its released library
                # or its potentially very large attention continuation.
                self.backbone=None
        except BaseException as error:
            try: self.close()
            except BaseException as failure:
                error.cleanup=self.close;add_failure_note(error, 'MTP native ownership cleanup remains pending: '+failure_summary(failure))
            raise

    def _shared_equal(self,source,target):
        if not self._device:return np.array_equal(source,target)
        value=np.asarray(source)
        if value.dtype!=np.float32 or value.shape!=tuple(target.shape) or value.ndim!=2:return False
        rows=max(1,min(128,self.max_work_bytes//max(1,value.shape[1]*16)))
        if value.shape[1]*16>self.max_work_bytes:raise MemoryError('shared MTP tensor comparison exceeds admission')
        for begin in range(0,value.shape[0],rows):
            end=min(value.shape[0],begin+rows)
            # Original streamed heads gather only the admitted row interval.
            selected=self.backbone.torch.arange(begin,end,device=self.backbone.device)
            actual=target[selected].detach().cpu().numpy()
            if not np.array_equal(value[begin:end],actual):return False
        return True

    @property
    def position(self):
        positions=tuple(stage.position for stage in self.stages)
        if positions and len(set(positions))!=1: raise RuntimeError('MTP depths have different continuation extents')
        return positions[0] if positions else 0

    def _check(self,*,cleanup=False):
        if self._closed or self._busy or (self._poisoned and not cleanup):
            raise RuntimeError('Inkling MTP owner is closed, poisoned or executing')

    def snapshot(self,*,_internal=False):
        with self._lock:
            self._check()
            if len(self._snapshots)>=self.max_snapshots+(2 if _internal else 0): raise MemoryError('MTP snapshot count exceeds admission')
            # Validate the common logical position before acquiring any stage
            # receipts; a mismatch must not strand snapshots with no handle.
            position=self.position
            receipts=[]
            try:
                for stage in self.stages: receipts.append(stage.snapshot())
                key=uuid4().hex
                receipt=InklingMTPSnapshot(self._owner,self.identity,key,position)
                self._snapshots[key]=list(receipts)
                return receipt
            except BaseException as error:
                for stage,receipt in zip(self.stages,receipts):
                    try: stage.release_snapshot(receipt)
                    except BaseException as cleanup:
                        self._orphaned_snapshots.append((stage,receipt))
                        add_failure_note(error, 'MTP partial snapshot release retained for owner close retry: '+failure_summary(cleanup))
                raise

    def _receipt(self,receipt):
        if type(receipt) is not InklingMTPSnapshot or receipt.owner!=self._owner or receipt.identity!=self.identity or receipt.key not in self._snapshots:
            raise ValueError('owned Inkling MTP snapshot required')
        return self._snapshots[receipt.key]

    def release_snapshot(self,receipt):
        with self._lock:
            self._check(cleanup=True);values=self._receipt(receipt)
            for index,(stage,value) in enumerate(zip(self.stages,values)):
                if value is not None: stage.release_snapshot(value);values[index]=None
            del self._snapshots[receipt.key]
            self._pending_snapshot_handles=[pending for pending in self._pending_snapshot_handles
                if pending is not receipt]

    def _release_internal_snapshot(self,receipt,primary=None):
        try:self.release_snapshot(receipt)
        except BaseException as cleanup:
            if all(pending is not receipt for pending in self._pending_snapshot_handles):
                self._pending_snapshot_handles.append(receipt)
            if primary is None:raise
            add_failure_note(primary, 'MTP private rollback snapshot retained for owner close retry: '+failure_summary(cleanup))

    def _restore_stages(self,receipts,failure):
        errors=[]
        for stage,value in zip(self.stages,receipts):
            try:stage.restore(value)
            except BaseException as rollback_error:errors.append(rollback_error)
        if errors:
            self._poisoned=True
            for rollback_error in errors:
                add_failure_note(failure, 'MTP stage rollback failed; owner quarantined: '+failure_summary(rollback_error))

    def restore(self,receipt):
        with self._lock:
            self._check();values=self._receipt(receipt)
            if any(value is None for value in values): raise ValueError('MTP snapshot release is partially complete')
            rollback=self.snapshot(_internal=True)
            primary=None
            try:
                for stage,value in zip(self.stages,values): stage.restore(value)
            except BaseException as error:
                primary=error
                self._restore_stages(self._receipt(rollback),error)
                raise
            finally:
                self._release_internal_snapshot(rollback,primary)

    def draft(self,shifted_ids,previous_hidden,*,cancelled=None):
        if self._device:return self._draft_device(shifted_ids,previous_hidden,cancelled=cancelled)
        with self._lock:
            self._check();ids=np.asarray(shifted_ids);hidden=np.asarray(previous_hidden)
            width=self._config['hidden_size'];vocabulary=self._config['unpadded_vocab_size'] or self._config['vocab_size']
            if ids.ndim!=2 or ids.shape[0]!=1 or not ids.shape[1] or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=vocabulary):
                raise ValueError('shifted unpadded batch-one MTP ids required')
            if hidden.shape!=(*ids.shape,width) or hidden.dtype!=np.float32 or not np.isfinite(hidden).all():
                raise ValueError('aligned FP32 target hidden states required')
            resident=sum(value.nbytes for value in self.weights.values())
            if resident+hidden.nbytes*12+vocabulary*8*self.depth>self.max_work_bytes:
                raise MemoryError('MTP projection and candidate staging exceeds admission')
            rollback=self.snapshot(_internal=True);tokens=[];logits=[];self._busy=True;primary=None
            try:
                ids=np.array(ids,dtype=np.int64,order='C',copy=True)
                for index,stage in enumerate(self.stages):
                    if cancelled is not None and cancelled(): raise InterruptedError('Inkling MTP cancelled')
                    prefix=f'layers.{index}.';eps=self._config['rms_norm_eps']
                    embeds=np.ascontiguousarray(self._globals['model.embed_tokens.weight'][ids])
                    e=self.ops.norm(embeds,self.weights[prefix+'enorm.weight'],rms=True,epsilon=eps)
                    h=self.ops.norm(hidden,self.weights[prefix+'hnorm.weight'],rms=True,epsilon=eps)
                    pair=np.ascontiguousarray(np.concatenate((h,e) if self.hidden_first else (e,h),axis=-1))
                    projection=self.ops.linear(pair,self.weights[prefix+'eh_proj.weight'])
                    value=stage.decoder.forward(ids,inputs_embeds=projection,cancelled=cancelled,
                        _raw_hidden=True,_position_origin=index+1)
                    hidden=value.last_hidden_state
                    if self.chain_norm: hidden=self.ops.norm(hidden,self.weights['shared_post_norm.weight'],rms=True,epsilon=eps)
                    scaled=np.ascontiguousarray(hidden[:,-1:]/np.float32(self._config['logits_mup_width_multiplier']))
                    row=self.ops.linear(scaled,self._globals['lm_head.weight'][:vocabulary])[0,0]
                    token=int(np.argmax(row));tokens.append(token);logits.append(row)
                    ids=np.concatenate((ids[:,1:],np.array([[token]],np.int64)),axis=1)
                if cancelled is not None and cancelled(): raise InterruptedError('Inkling MTP cancelled before commit')
                return tuple(tokens),np.stack(logits)
            except BaseException as error:
                primary=error
                self._restore_stages(self._receipt(rollback),error)
                raise
            finally:
                self._busy=False;self._release_internal_snapshot(rollback,primary)

    def _draft_device(self,shifted_ids,previous_hidden,*,cancelled=None):
        from contextlib import nullcontext
        from .device_publisher_projections import device_projection
        from .device_logits import device_logits
        class Event:
            def is_set(event):return bool(cancelled is not None and cancelled())
        event=Event()
        with self._lock:
            self._check();ids=np.asarray(shifted_ids);source=np.asarray(previous_hidden)
            c=self.backbone.config;h=c['hidden_size'];vocab=c['unpadded_vocab_size'] or c['vocab_size']
            if ids.ndim!=2 or ids.shape[0]!=1 or not ids.shape[1] or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=vocab):
                raise ValueError('shifted unpadded batch-one device MTP IDs required')
            if source.shape!=(*ids.shape,h) or source.dtype!=np.float32 or not np.isfinite(source).all():
                raise ValueError('aligned finite FP32 device MTP target hidden required')
            resident=2*sum(value.nbytes for value in self.weights.values())
            if resident+source.nbytes*16+c['vocab_size']*8*self.depth>self.max_work_bytes:
                raise MemoryError('complete device MTP staging exceeds work admission')
            if event.is_set():raise InterruptedError('device Inkling MTP cancelled')
            rollback=self.snapshot(_internal=True);self._busy=True;tokens=[];logits=[];primary=None
            torch=self.backbone.torch
            provider=self.backbone.projection_provider
            try:
                with torch.inference_mode(),(provider.cancellation_scope(event) if provider is not None else nullcontext()):
                    ids=np.array(ids,dtype=np.int64,order='C',copy=True)
                    hidden=torch.tensor(source,device=self.backbone.device)
                    def norm(value,weight):
                        return value*torch.rsqrt(value.square().mean(dim=-1,keepdim=True)+c['rms_norm_eps'])*weight
                    for index,stage in enumerate(self.stages):
                        if event.is_set():raise InterruptedError('device Inkling MTP cancelled')
                        prefix=f'layers.{index}.'
                        embeds=self.backbone.weights['model.embed_tokens.weight'][torch.tensor(ids,device=self.backbone.device)]
                        e=norm(embeds,self._device_weights[prefix+'enorm.weight'])
                        previous=norm(hidden,self._device_weights[prefix+'hnorm.weight'])
                        pair=torch.cat((previous,e) if self.hidden_first else (e,previous),dim=-1)
                        projection=device_projection(torch,pair,self._device_weights[prefix+'eh_proj.weight'],cancel=event)
                        with stage.cancellation_scope(event):
                            value=stage.forward(ids,inputs_embeds=projection.detach().cpu().numpy(),
                                _raw_hidden=True,_position_origin=index+1)
                        hidden=torch.tensor(value.last_hidden_state,device=self.backbone.device)
                        if self.chain_norm:hidden=norm(hidden,self._device_weights['shared_post_norm.weight'])
                        rows=device_logits(torch,hidden[:,-1:]/c['logits_mup_width_multiplier'],
                            self.backbone.weights['lm_head.weight'],max_device_bytes=self.max_work_bytes-resident,
                            max_host_bytes=c['vocab_size']*4,cancel=event)
                        row=rows[0,0,:vocab];token=int(np.argmax(row));tokens.append(token);logits.append(row.copy())
                        ids=np.concatenate((ids[:,1:],np.array([[token]],np.int64)),axis=1)
                    if event.is_set():raise InterruptedError('device Inkling MTP cancelled before commit')
                    return tuple(tokens),np.stack(logits)
            except BaseException as error:
                primary=error
                self._restore_stages(self._receipt(rollback),error)
                raise
            finally:
                self._busy=False;self._release_internal_snapshot(rollback,primary)

    def reset(self):
        with self._lock:
            self._check()
            rollback=self.snapshot(_internal=True);self._busy=True;primary=None
            try:
                for stage in self.stages:stage.reset()
            except BaseException as error:
                primary=error
                self._restore_stages(self._receipt(rollback),error)
                raise
            finally:
                self._busy=False;self._release_internal_snapshot(rollback,primary)

    def write_cache(self,output,*,max_bytes,cancelled=None):
        from .compound_cache import write_components
        with self._lock:
            self._check();self.position;self._busy=True
            try:
                return write_components(output,self.stages,identity=self.identity,
                    max_bytes=max_bytes,cancelled=cancelled)
            finally: self._busy=False

    def restore_cache(self,source,*,max_bytes,cancelled=None):
        from .compound_cache import restore_components
        with self._lock:
            self._check();self._busy=True
            try:
                return restore_components(source,self.stages,identity=self.identity,
                    max_bytes=max_bytes,cancelled=cancelled,validate=lambda:self.position)
            finally: self._busy=False

    def close(self):
        with self._lock:
            if self._busy: raise RuntimeError('MTP native inference owns resources')
            if self._closed: return
            failures=[]
            pending=[]
            for stage,receipt in self._orphaned_snapshots:
                try: stage.release_snapshot(receipt)
                except BaseException as error:
                    failures.append(error);pending.append((stage,receipt))
            self._orphaned_snapshots=pending
            if failures:
                raise RuntimeError('MTP snapshot cleanup failed; owner retained for retry') from failures[0]
            pending_handles=[]
            for receipt in self._pending_snapshot_handles:
                try:self.release_snapshot(receipt)
                except BaseException as error:
                    failures.append(error);pending_handles.append(receipt)
            self._pending_snapshot_handles=pending_handles
            if failures:
                raise RuntimeError('MTP private snapshot cleanup failed; owner retained for retry') from failures[0]
            if self._snapshots:
                raise RuntimeError('cannot close MTP owner with live snapshot receipts; release them and retry close')
            while self.stages: self.stages[-1].close();self.stages.pop()
            if self.ops is not None: self.ops.close();self.ops=None
            getattr(self,'_device_weights',{}).clear()
            self._globals.clear();self.backbone=None
            self._snapshots.clear();self.weights.clear();self._closed=True

__all__=['InklingMTP','InklingMTPSnapshot']
