"""Durable compressed-MLA/KDA recovery for the explicit K3 executor."""
from dataclasses import asdict
import hashlib
import json
import math
import struct

import numpy as np

from ..k3_execution import K3Execution, K3Snapshot
from ..k3_execution.cache import MLALatentPages
from .facade import ExecutionPlan, ScalableSession
from .resources import CacheLayout, Geometry, positive
from .scheduler import DispatchStep


_MAGIC = b'OAK3C001'


class K3CompressedStateCodec:
    """MLA rows per token, with optional boundary-safe recurrent checkpoints.

    The template is borrowed and must remain open. This layout favors bounded,
    exact recovery. Native interval one preserves the original fixed frame
    identity; variable checkpoints, raw FLA BF16 convolution transport and
    selected media evidence use distinct identities. install_prefix admits
    complete executed media state before a text continuation is dispatched.
    """
    def __init__(self, template, *, max_payload_bytes, checkpoint_interval=1):
        if type(template) is not K3Execution:
            raise ValueError('current K3Execution template required')
        if template._closed or template._busy:
            raise RuntimeError('available borrowed K3 codec template required')
        self.template, self.config = template, template.config
        self.fla_policy=template._fla_policy
        self.media_case=template._exact_media_case
        self.exact_evidence=None if template._exact_evidence is None else template._exact_evidence.identity
        self.max_payload_bytes = positive(max_payload_bytes, 'max_payload_bytes')
        self.checkpoint_interval = positive(checkpoint_interval, 'checkpoint_interval')
        if checkpoint_interval >= 2**64:
            raise ValueError('checkpoint_interval exceeds unsigned-64-bit range')
        if self.media_case is not None and checkpoint_interval==1:
            raise ValueError('media prefix installation requires explicit variable recurrent checkpoint frames')
        self.magic = _MAGIC if checkpoint_interval == 1 else b'OAK3C002'
        if self.fla_policy is not None:
            self.magic=b'OAK3C003' if checkpoint_interval==1 else b'OAK3C004'
        if self.media_case is not None:self.magic=b'OAK3C005'
        self.model_identity = template._identity
        config_bytes = json.dumps(asdict(self.config), sort_keys=True, separators=(',', ':'),
                                 allow_nan=False).encode('ascii')
        policy = b'' if checkpoint_interval == 1 else struct.pack('<Q', checkpoint_interval)
        if self.exact_evidence is not None:policy+=self.exact_evidence.encode('ascii')
        if self.media_case is not None:policy+=self.media_case.encode('ascii')
        if self.fla_policy is not None:
            policy+=json.dumps(asdict(self.fla_policy),sort_keys=True,separators=(',',':'),allow_nan=False).encode('ascii')
        self.identity = hashlib.sha256(self.magic + policy + config_bytes + self.model_identity.encode('ascii')).hexdigest()
        c = self.config
        self.specs = []
        self.layer_bytes = []
        for layer, kind in enumerate(c.layer_types):
            if kind=='mla':leaves=(((c.kvrank+c.rope,),'<f4'),)
            elif self.fla_policy is not None:
                leaves=(((1,c.kda_heads,c.kda_dim,c.kda_dim),'<f4'),)+(
                    ((1,c.kda_heads*c.kda_dim,c.kernel),'<u2'),)*3
            else:
                leaves=(((c.kda_heads,c.kda_dim,c.kda_dim),'<f4'),)+(
                    ((c.kernel-1,c.kda_heads,c.kda_dim),'<f4'),)*3
            self.layer_bytes.append(sum(math.prod(shape)*np.dtype(dtype).itemsize for shape,dtype in leaves))
            self.specs.extend((layer,shape,dtype) for shape,dtype in leaves)
        self.header_bytes = 80 + (checkpoint_interval != 1)
        self.step_bytes = self.header_bytes + sum(self.layer_bytes)
        self.latent_bytes = sum(size for size, kind in zip(self.layer_bytes, c.layer_types) if kind == 'mla')
        self.delta_bytes = self.header_bytes + self.latent_bytes
        if self.step_bytes > self.max_payload_bytes:
            raise MemoryError('compressed K3 recovery frame exceeds payload budget')

    def geometry(self, *, max_batch_tokens=512):
        c = self.config
        return Geometry(self.identity, c.hidden, c.expert_intermediate, len(c.layer_types),
            c.heads, 1, c.kvrank + c.rope, 4, c.capacity, max_batch_tokens,
            frozenset((c.expert_format,)), tuple(
                CacheLayout('custom', size + (self.header_bytes if layer == 0 else 0),
                            self.identity + ':layer:' + str(layer), self.checkpoint_interval != 1)
                for layer, size in enumerate(self.layer_bytes)))

    def binding(self,model,*,allow_empty=False):
        if type(model) is not K3Execution or model.config != self.config or model._identity != self.model_identity:
            raise ValueError('K3 recovery model/config/weight identity mismatch')
        if self.template._closed or model._closed:
            raise RuntimeError('closed K3 recovery owner')
        evidence=None if model._exact_evidence is None else model._exact_evidence.identity
        media_matches=(model._exact_media_case==self.media_case or
            (allow_empty and model.position==0 and model._exact_media_case is None))
        if evidence!=self.exact_evidence or not media_matches or model._fla_policy!=self.fla_policy:
            raise ValueError('compressed K3 recovery exact policy or media case differs')

    def encode_model_step(self, model, *, checkpoint=True):
        """Encode borrowed rows while holding the exact decoder's operation guard."""
        self.binding(model)
        with model._operation():
            model._admit(0)
            snapshot = K3Snapshot(self.model_identity, model._position,
                tuple((i, pages.pages) for i, pages in model._mla.items()),
                tuple((i, m.snapshot() if self.fla_policy is not None and checkpoint else
                    None if self.fla_policy is not None else (m._position,m._state,m._history))
                    for i,m in model._kda.items()),self.exact_evidence,self.media_case)
            return self.encode_step(snapshot, checkpoint=checkpoint)

    def encode_step(self, snapshot, *, checkpoint=True):
        if self.template._closed:raise RuntimeError('closed borrowed K3 recovery template')
        c = self.config
        if type(checkpoint) is not bool or (not checkpoint and self.checkpoint_interval == 1):
            raise ValueError('invalid K3 recovery checkpoint policy')
        if type(snapshot) is not K3Snapshot or snapshot.identity != self.model_identity or type(snapshot.position) is not int or not 1 <= snapshot.position <= c.capacity:
            raise ValueError('nonempty current K3 snapshot required')
        if snapshot.exact_evidence!=self.exact_evidence or snapshot.media_case!=self.media_case:
            raise ValueError('compressed K3 snapshot exact policy differs')
        if checkpoint and self.template._exact_evidence is not None:
            self.template._exact_evidence.admit_snapshot(self.media_case,snapshot.position)
        mla, kda = dict(snapshot.mla), dict(snapshot.kda)
        if len(mla) != len(snapshot.mla) or len(kda) != len(snapshot.kda):
            raise ValueError('duplicate recovery layers')
        expected_mla = {i for i, kind in enumerate(c.layer_types) if kind == 'mla'}
        expected_kda = set(range(len(c.layer_types))) - expected_mla
        if set(mla) != expected_mla or set(kda) != expected_kda:
            raise ValueError('incomplete recovery layers')
        arrays = []
        for layer, kind in enumerate(c.layer_types):
            if kind == 'mla':
                pages = mla[layer]
                if not pages or sum(len(page) for page in pages) != snapshot.position:
                    raise ValueError('compressed MLA recovery position mismatch')
                arrays.append(pages[-1][-1])
            else:
                if not checkpoint:
                    arrays.extend((None,)*4)
                    continue
                if self.fla_policy is not None:
                    from ..k3_execution.fla_mixer import FLAMixerSnapshot
                    item=kda[layer]
                    if type(item) is not FLAMixerSnapshot or item.identity!=self.template._kda[layer].identity:
                        raise ValueError('exact FLA recurrent owner identity required')
                    position,state,history=item.position,item.recurrent,item.convolution
                else:position,state,history=kda[layer]
                if position != snapshot.position or type(history) is not tuple or len(history) != 3:
                    raise ValueError('recurrent recovery position/history mismatch')
                arrays.extend((state, *history))
        payload = bytearray(self.magic + bytes.fromhex(self.identity) + struct.pack('<Q', snapshot.position))
        if self.checkpoint_interval != 1:
            payload.append(int(checkpoint))
        for (layer,shape,dtype),array in zip(self.specs,arrays):
            if not checkpoint and c.layer_types[layer] != 'mla':
                continue
            if self.fla_policy is not None and c.layer_types[layer]=='kda':
                torch=self.template._kda[layer]._torch
                expected_dtype=torch.bfloat16 if dtype=='<u2' else torch.float32
                if (not isinstance(array,torch.Tensor) or tuple(array.shape)!=shape or array.dtype!=expected_dtype or
                        array.device!=self.template._kda[layer].device or not bool(torch.isfinite(array).all())):
                    raise ValueError('invalid original FLA recovery tensor')
                array=array.detach().contiguous().cpu()
                if dtype=='<u2':array=array.view(torch.uint16)
                array=array.numpy()
            if type(array) is not np.ndarray or array.shape!=shape or array.dtype!=np.dtype(dtype) or not np.isfinite(array).all():
                raise ValueError('invalid recovery array')
            payload.extend(np.asarray(array,dtype=dtype,order='C').tobytes())
        if len(payload) != (self.step_bytes if checkpoint else self.delta_bytes) - 32:
            raise ValueError('recovery frame extent mismatch')
        payload.extend(hashlib.sha256(payload).digest())
        return bytes(payload)

    def decode_step(self, payload, *, position):
        checkpoint, arrays = self.decode_frame(payload, position=position)
        if not checkpoint:
            raise ValueError('complete recurrent checkpoint required')
        return arrays

    def install_prefix(self,cache,session_id,owner,token_ids,*,max_seed_bytes,cancelled=None):
        """Atomically install an executed template prefix without token replay.

        token_ids are the caller's actual expanded occupied history. Earlier
        records carry observed MLA rows only; the terminal record carries all
        physical recurrent/convolution state. Partial prefixes never commit.
        """
        from .resources import check_cancel
        from .cache import DurablePagedCache
        if type(cache) is not DurablePagedCache or cache.geometry!=self.geometry(
                max_batch_tokens=cache.geometry.max_batch_tokens):
            raise ValueError('exact durable K3 prefix cache binding required')
        if self.checkpoint_interval==1:
            raise ValueError('executed prefix installation requires variable checkpoint frames')
        positive(max_seed_bytes,'max_seed_bytes')
        self.binding(self.template)
        position=self.template.position
        if not 0<position<=cache.geometry.max_batch_tokens:
            raise ValueError('executed prefix must fit one admitted atomic seed transaction')
        required=4*position*self.latent_bytes+8*self.step_bytes+128*position+32768
        if required>max_seed_bytes:raise MemoryError('executed prefix snapshot/encoding exceeds seed admission')
        tokens=[]
        for token in token_ids:
            check_cancel(cancelled)
            if len(tokens)>=position or type(token) is not int or not 0<=token<self.config.vocab:
                raise ValueError('complete original expanded prefix token history required')
            tokens.append(token)
        if len(tokens)!=position:raise ValueError('expanded history differs from executed prefix extent')
        check_cancel(cancelled)
        snapshot=self.template.snapshot()
        if snapshot.position!=position:raise ValueError('executed prefix changed during immutable capture')
        terminal=self.encode_step(snapshot)
        banks=dict(snapshot.mla)
        row_streams={layer:iter(row for page in pages for row in page) for layer,pages in banks.items()}
        with cache.transaction(session_id,owner,tuple(tokens),cancelled=cancelled) as transaction:
            if transaction.position:raise ValueError('prefix installation requires an empty durable session')
            for row,token in enumerate(tokens):
                check_cancel(cancelled)
                rows={layer:next(stream) for layer,stream in row_streams.items()}
                if row+1==position:frame=terminal
                else:
                    frame=self._encode_latent_rows(rows,position=row+1)
                transaction.append_page((token,),frame)
            check_cancel(cancelled)
        return position

    def _encode_latent_rows(self,rows,*,position):
        """Encode observed seed rows without reconstructing earlier recurrence."""
        if self.checkpoint_interval==1 or type(position) is not int or not 1<=position<=self.config.capacity:
            raise ValueError('bounded variable MLA delta position required')
        expected={layer for layer,kind in enumerate(self.config.layer_types) if kind=='mla'}
        if set(rows)!=expected:raise ValueError('complete original MLA delta rows required')
        payload=bytearray(self.magic+bytes.fromhex(self.identity)+struct.pack('<Q',position)+b'\x00')
        for layer,shape,dtype in self.specs:
            if self.config.layer_types[layer]!='mla':continue
            array=rows[layer]
            if type(array) is not np.ndarray or array.shape!=shape or array.dtype!=np.float32 or not np.isfinite(array).all():
                raise ValueError('finite observed original MLA row required')
            payload.extend(np.asarray(array,dtype=dtype,order='C').tobytes())
        if len(payload)!=self.delta_bytes-32:raise ValueError('MLA seed delta extent mismatch')
        payload.extend(hashlib.sha256(payload).digest())
        return bytes(payload)

    def decode_frame(self, payload, *, position):
        with memoryview(payload) as view:
            if not view.c_contiguous or view.nbytes not in (self.step_bytes, self.delta_bytes):
                raise ValueError('invalid K3 recovery frame length')
            data = view.cast('B')
            try:
                if bytes(data[:8]) != self.magic or bytes(data[8:40]) != bytes.fromhex(self.identity):
                    raise ValueError('foreign K3 recovery frame')
                if type(position) is not int or not 1 <= position <= self.config.capacity or struct.unpack_from('<Q', data, 40)[0] != position:
                    raise ValueError('K3 recovery frame position mismatch')
                if hashlib.sha256(data[:-32]).digest() != bytes(data[-32:]):
                    raise ValueError('K3 recovery frame checksum mismatch')
                checkpoint = True
                if self.checkpoint_interval != 1:
                    if data[48] not in (0, 1):
                        raise ValueError('invalid K3 checkpoint marker')
                    checkpoint = bool(data[48])
                if view.nbytes != (self.step_bytes if checkpoint else self.delta_bytes):
                    raise ValueError('K3 checkpoint extent mismatch')
                arrays, offset = [], self.header_bytes - 32
                for layer,shape,dtype in self.specs:
                    if not checkpoint and self.config.layer_types[layer] != 'mla':
                        continue
                    size=math.prod(shape)*np.dtype(dtype).itemsize
                    array=np.frombuffer(data[offset:offset+size],dtype=dtype).reshape(shape).copy()
                    if not np.isfinite(array).all() or (dtype=='<u2' and np.any((array&0x7f80)==0x7f80)):
                        raise ValueError('nonfinite recovered state')
                    arrays.append(array)
                    offset += size
                return checkpoint, arrays
            finally:
                data.release()


class K3CompressedPlanner:
    def __init__(self, codec, geometry, factory, *, executor_memory_bytes):
        if type(codec) is not K3CompressedStateCodec or codec.geometry(max_batch_tokens=geometry.max_batch_tokens) != geometry or not callable(factory):
            raise ValueError('exact compressed K3 codec, geometry, and fresh factory required')
        self.codec, self.geometry, self.factory = codec, geometry, factory
        self.executor_memory_bytes = positive(executor_memory_bytes, 'executor_memory_bytes')

    def __call__(self, token_ids):
        holder = {}
        codec, c = self.codec, self.codec.config

        def restore(context, _):
            try:model = self.factory()
            except BaseException as primary:
                cleanup=getattr(primary,'cleanup',None)
                if callable(cleanup):context.defer(cleanup)
                raise
            if model is codec.template:
                raise ValueError('factory returned the borrowed recovery template')
            try:
                context.defer(model.close)
            except BaseException as primary:
                try:model.close()
                except BaseException as cleanup:
                    context._cleanup_callbacks.append(model.close)
                    primary.add_note('compressed K3 construction cleanup retained: '+repr(cleanup))
                raise
            codec.binding(model,allow_empty=True)
            context.bind_cancellation(model)
            model.reset()
            holder['model'] = model
            pages = {i: MLALatentPages(c.kvrank, c.rope) for i, kind in enumerate(c.layer_types) if kind == 'mla'}
            recurrent, position, checkpoint_position = {}, 0, 0
            for metadata in context.cache.page_metadata():
                context.check_cancel()
                if metadata.position != position or len(metadata.token_ids) != 1 or not 0 <= metadata.token_ids[0] < c.vocab:
                    raise ValueError('nonconsecutive K3 recovery records')
                with context.cache.read_page(metadata.ordinal) as payload:
                    checkpoint, arrays = codec.decode_frame(payload, position=position + 1)
                cursor = 0
                for layer, kind in enumerate(c.layer_types):
                    if kind == 'mla':
                        row = arrays[cursor]
                        pages[layer].append(row[:c.kvrank], row[c.kvrank:])
                        cursor += 1
                    elif checkpoint:
                        if codec.fla_policy is not None:
                            from ..k3_execution.fla_mixer import FLAMixerSnapshot
                            mixer=model._kda[layer];torch=mixer._torch
                            state=torch.from_numpy(arrays[cursor]).to(device=mixer.device,dtype=torch.float32)
                            history=tuple(torch.from_numpy(array).view(torch.bfloat16).to(mixer.device)
                                for array in arrays[cursor+1:cursor+4])
                            recurrent[layer]=FLAMixerSnapshot(mixer.identity,position+1,state,history)
                        else:
                            recurrent[layer]=(position+1,arrays[cursor],tuple(arrays[cursor+1:cursor+4]))
                        cursor += 4
                position += 1
                if checkpoint:
                    checkpoint_position = position
            if position != context.position:
                raise ValueError('K3 recovered prefix length mismatch')
            if checkpoint_position != position:
                raise ValueError('K3 committed prefix lacks terminal recurrent checkpoint')
            if position:
                model.restore(K3Snapshot(codec.model_identity, position,
                    tuple((i,cache.snapshot()) for i,cache in pages.items()),tuple(recurrent.items()),
                    codec.exact_evidence,codec.media_case))
            elif codec.media_case is not None:
                raise ValueError('media-bound scalable continuation requires an installed complete media prefix')
            holder['logits'] = np.empty((len(context.token_ids), c.vocab), np.float32)

        def execute(context, _):
            model = holder['model']
            for index, token in enumerate(context.token_ids):
                context.check_cancel()
                holder['logits'][index] = model.forward([token])[0]
                context.check_cancel()
                checkpoint = (model.position % codec.checkpoint_interval == 0 or
                              index == len(context.token_ids) - 1)
                context.cache.append_page((token,), codec.encode_model_step(model, checkpoint=checkpoint))

        # Reconstruction, snapshot and atomic restore can coexist with live MLA.
        latent_prefix = c.capacity * codec.latent_bytes
        frame_copies=8 if codec.fla_policy is not None else 6
        scratch = (self.executor_memory_bytes + 4 * latent_prefix + frame_copies * codec.step_bytes +
                   8 * len(token_ids) * c.vocab + 5 * min(4096, len(token_ids) * c.vocab))
        return ExecutionPlan((DispatchStep(None, restore), DispatchStep(None, execute)),
                             lambda context: holder['logits'], scratch)


def bind_compressed_k3_session(executor, codec, factory, session_id, owner, *,
                               executor_memory_bytes, create=True, prefix_id=None):
    planner = K3CompressedPlanner(codec, executor.geometry, factory,
                                  executor_memory_bytes=executor_memory_bytes)
    return ScalableSession(executor, session_id, owner, planner,
                           vocab_size=codec.config.vocab, create=create, prefix_id=prefix_id)
