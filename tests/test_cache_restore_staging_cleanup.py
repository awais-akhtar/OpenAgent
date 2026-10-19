import gc
import hashlib
import io
import json
import struct
import threading
import unittest
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine import (
    compound_cache, mixtral_cache_recovery, native_glm_dsa_cache,
    torch_inkling_cache,
    qwen4_cache_recovery, qwen35_cache_recovery,
    torch_inkling_decoder, torch_mixtral_cache, torch_qwen35_cache,
    torch_qwen4_cache,
)
from openagent.adaptive_engine.native_mixtral_decoder import NativeMixtralDecoder


class CacheRestoreStagingCleanupTests(unittest.TestCase):
    class _Torch:
        float32=np.float32
        int64=np.int64

        def __init__(self):
            self.references=[]

        def tensor(self,value,**_kwargs):
            result=np.array(value,copy=True)
            self.references.append(weakref.ref(result))
            return result

    class _CompoundComponent:
        def __init__(self,name,log,*,fail=False):
            self.name=name;self.log=log;self.fail=fail;self.state=name+'-old'

        def write_cache(self,output,*,max_bytes):
            output.write((self.name+'-cache').encode())

        def snapshot(self):
            return self.state

        def restore_cache(self,source,*,max_bytes):
            self.state=source.read(max_bytes).decode()
            if self.fail:raise ValueError('component restore failed after mutation')

        def restore(self,receipt):
            self.log.append(self.name);self.state=receipt

        def release_snapshot(self,_receipt):
            pass

    def test_compound_cache_failure_rolls_back_components_in_reverse_order(self):
        source_components=[self._CompoundComponent('a',[]),self._CompoundComponent('b',[])]
        output=io.BytesIO()
        compound_cache.write_components(output,source_components,identity='group',max_bytes=1024)
        log=[]
        target_components=[self._CompoundComponent('a',log),self._CompoundComponent('b',log,fail=True)]
        with self.assertRaisesRegex(ValueError,'after mutation'):
            compound_cache.restore_components(io.BytesIO(output.getvalue()),target_components,
                                               identity='group',max_bytes=1024)
        self.assertEqual(log,['b','a'])
        self.assertEqual([component.state for component in target_components],['a-old','b-old'])

    def test_grouped_continuation_cancellation_reaches_component_and_rolls_back(self):
        from openagent.adaptive_engine.grouped_continuation import GroupedContinuation

        class CancellableComponent(self._CompoundComponent):
            def restore_cache(self,source,*,max_bytes,cancelled=None):
                self.state=source.read(max_bytes).decode()
                if cancelled is not None and cancelled():
                    raise InterruptedError('component restore cancelled after staging')

        source=GroupedContinuation((self._CompoundComponent('a',[]),
                                    self._CompoundComponent('b',[])),
                                   identity='inkling-group',max_snapshots=4)
        output=io.BytesIO()
        source.write_cache(output,max_bytes=1024)
        target_components=(CancellableComponent('a',[]),CancellableComponent('b',[]))
        target=GroupedContinuation(target_components,identity='inkling-group',max_snapshots=4)
        cancelled=lambda: target_components[0].state=='a-cache'

        with self.assertRaisesRegex(InterruptedError,'component restore cancelled'):
            target.restore_cache(io.BytesIO(output.getvalue()),max_bytes=1024,cancelled=cancelled)

        self.assertEqual([component.state for component in target_components],['a-old','b-old'])

    def test_grouped_continuation_export_cancellation_does_not_publish_partial_frame(self):
        from openagent.adaptive_engine.grouped_continuation import GroupedContinuation

        class CancellableWriter(self._CompoundComponent):
            def write_cache(self,output,*,max_bytes,cancelled=None):
                output.write((self.name+'-cache').encode())
                self.state='writing'
                if cancelled is not None and cancelled():
                    raise InterruptedError('component export cancelled after staging')

        components=(CancellableWriter('a',[]),self._CompoundComponent('b',[]))
        grouped=GroupedContinuation(components,identity='inkling-group',max_snapshots=4)
        output=io.BytesIO()
        cancelled=lambda: components[0].state=='writing'

        with self.assertRaisesRegex(InterruptedError,'component export cancelled'):
            grouped.write_cache(output,max_bytes=1024,cancelled=cancelled)

        self.assertEqual(output.getvalue(),b'')

    def test_device_qwen35_rejected_restore_releases_host_and_device_staging(self):
        torch=self._Torch();references=[];empty=np.empty
        expected=dict(identity='device-q35',batch=1,position=1,shapes=[[[2]]])
        model=SimpleNamespace(
            _lock=threading.RLock(),_busy=False,_check=lambda:None,_cancel=lambda:None,
            capacity=4,identity='device-q35',weight_bytes=0,_expert_bytes=0,
            _projection_bytes=0,_snapshot_bytes=0,_state_bytes=lambda:0,
            max_bytes=4*1024**2,torch=torch,device='cpu',
            _cache=['old-cache'],_mask='old-mask',position=5,
        )
        encoded=torch_qwen35_cache._json(expected)
        wire=(b'OATQ3501'+struct.pack('<I',len(encoded))+encoded+
              np.asarray([1.0,2.0],dtype='<f4').tobytes()+b'\x01'+b'\x00'*32)

        def tracked_empty(*args,**kwargs):
            value=empty(*args,**kwargs);references.append(weakref.ref(value));return value

        with patch.object(torch_qwen35_cache,'_schema',return_value=expected), \
                patch.object(torch_qwen35_cache,'_payload',return_value=8), \
                patch.object(torch_qwen35_cache.np,'empty',side_effect=tracked_empty):
            with self.assertRaisesRegex(ValueError,'checksum/EOF mismatch'):
                torch_qwen35_cache.restore_device_cache(model,io.BytesIO(wire),max_bytes=4*1024**2)

        gc.collect()
        self.assertEqual(len(references),1)
        self.assertTrue(all(reference() is None for reference in references+torch.references))
        self.assertEqual((model._cache,model._mask,model.position),(['old-cache'],'old-mask',5))

    def test_device_qwen4_rejected_restore_releases_host_and_device_staging(self):
        torch=self._Torch();references=[];empty=np.empty
        expected=dict(identity='device-q4',batch=1,position=1,banks=[1],
                      lexical=[None],position_axes=[])
        model=SimpleNamespace(
            _lock=threading.RLock(),_busy=False,_check=lambda:None,_cancel=lambda:None,
            identity='device-q4',config=SimpleNamespace(max_position_embeddings=4,vocab_size=16),
            weight_bytes=0,_expert_bytes=0,_snapshot_bytes=0,_state_bytes=lambda:0,
            max_bytes=4*1024**2,torch=torch,device='cpu',_cache=['old-cache'],
            _ple=['old-ple'],_positions='old-positions',_mask='old-mask',position=5,
        )
        encoded=torch_qwen4_cache._json(expected)
        wire=(b'OATQW401'+struct.pack('<I',len(encoded))+encoded+
              np.asarray([1.0,2.0],dtype='<f4').tobytes()+b'\x01'+b'\x00'*32)

        def tracked_empty(*args,**kwargs):
            value=empty(*args,**kwargs);references.append(weakref.ref(value));return value

        with patch.object(torch_qwen4_cache,'_schema',return_value=expected), \
                patch.object(torch_qwen4_cache,'_size',return_value=8), \
                patch.object(torch_qwen4_cache,'_entries',return_value=[('layer',0,0,(2,),'<f4')]), \
                patch.object(torch_qwen4_cache.np,'empty',side_effect=tracked_empty):
            with self.assertRaisesRegex(ValueError,'checksum/EOF mismatch'):
                torch_qwen4_cache.restore_device_cache(model,io.BytesIO(wire),max_bytes=4*1024**2)

        gc.collect()
        self.assertEqual(len(references),1)
        self.assertTrue(all(reference() is None for reference in references+torch.references))
        self.assertEqual((model._cache,model._ple,model._positions,model._mask,model.position),
                         (['old-cache'],['old-ple'],'old-positions','old-mask',5))

    def test_device_mixtral_rejected_restore_releases_host_and_device_staging(self):
        torch=self._Torch();references=[];empty=np.empty
        expected=dict(identity='device-mx',batch=1,position=1,shape=[1,1,1,1],layers=1)
        model=SimpleNamespace(
            _lock=threading.RLock(),_busy=False,_check=lambda:None,
            identity='device-mx',capacity=4,weight_bytes=0,_snapshot_bytes=0,
            _expert_bytes=0,max_bytes=4*1024**2,torch=torch,device='cpu',
            _cache=[()],_mask='old-mask',position=5,
        )
        encoded=torch_mixtral_cache._json(expected)
        wire=(b'OATMXR01'+struct.pack('<I',len(encoded))+encoded+
              np.asarray([1.0,2.0],dtype='<f4').tobytes()+b'\x01'+b'\x00'*32)

        def tracked_empty(*args,**kwargs):
            value=empty(*args,**kwargs);references.append(weakref.ref(value));return value

        with patch.object(torch_mixtral_cache,'_schema',return_value=expected), \
                patch.object(torch_mixtral_cache.np,'empty',side_effect=tracked_empty):
            with self.assertRaisesRegex(ValueError,'checksum/EOF mismatch'):
                torch_mixtral_cache.restore_device_cache(model,io.BytesIO(wire),max_bytes=4*1024**2)

        gc.collect()
        self.assertEqual(len(references),2)
        self.assertTrue(all(reference() is None for reference in references+torch.references))
        self.assertEqual((model._cache,model._mask,model.position),([()], 'old-mask',5))

    def test_device_inkling_cancel_after_tensor_staging_releases_all_candidates(self):
        torch=self._Torch();references=[];empty=np.empty;calls=0
        class Config(dict):
            pass
        model=SimpleNamespace(
            _lock=threading.RLock(),_busy=False,_check=lambda:None,
            identity='device-inkling',geometry=(1,),
            _state_size=lambda _geometry:2,config=Config(max_position_embeddings=4),
            _cache=['old-cache'],position=5,torch=torch,device='cpu',
            _admit=lambda _size:None,
        )
        def cancel():
            nonlocal calls
            calls+=1
            if calls==4:raise InterruptedError('cancel after candidate device state')
        model._cancel=cancel
        encoded=torch_inkling_cache.json.dumps(
            torch_inkling_cache._header(model,1,1),sort_keys=True,separators=(',',':')).encode()
        prefix=b'OATINK01'+struct.pack('<I',len(encoded))+encoded
        payload=np.asarray([1.0,2.0],dtype='<f4').tobytes()
        wire=prefix+payload+hashlib.sha256(prefix+payload).digest()

        def tracked_empty(*args,**kwargs):
            value=empty(*args,**kwargs);references.append(weakref.ref(value));return value

        with patch.object(torch_inkling_cache.np,'empty',side_effect=tracked_empty):
            with self.assertRaisesRegex(InterruptedError,'candidate device state'):
                torch_inkling_cache.restore_cache(model,io.BytesIO(wire),max_bytes=4*1024**2)

        gc.collect()
        self.assertEqual(len(references),1)
        self.assertTrue(all(reference() is None for reference in references+torch.references))
        self.assertEqual((model._cache,model.position),(['old-cache'],5))

    def test_device_inkling_snapshot_restore_cancellation_releases_candidates(self):
        torch=self._Torch();references=[];array=np.asarray([1.0,2.0],dtype='<f4')
        references.append(weakref.ref(array));receipt=SimpleNamespace(position=3)
        snapshot_banks=(array,);snapshot_size=array.nbytes
        model=object.__new__(torch_inkling_decoder.TorchInklingDecoder)
        model._lock=threading.RLock();model._check=lambda:None
        model._snapshot=lambda _receipt:(None,snapshot_banks,snapshot_size)
        model._admit=lambda _size:None;calls=0
        def cancel():
            nonlocal calls
            calls+=1
            if calls==2:raise InterruptedError('cancel after snapshot tensor staging')
        model._cancel=cancel;model.torch=torch;model.device='cpu'
        model._cache=('old-cache',);model.position=9
        with self.assertRaisesRegex(InterruptedError,'snapshot tensor staging'):
            model.restore(receipt)
        del array
        gc.collect()
        self.assertIsNotNone(references[0]())  # The live snapshot remains owned.
        self.assertTrue(all(reference() is None for reference in torch.references))
        self.assertEqual((model._cache,model.position),(('old-cache',),9))

    def test_qwen4_rejected_restore_releases_allocated_state_arrays(self):
        config = SimpleNamespace(
            max_position_embeddings=4, num_hidden_layers=1,
            layer_types=('linear_attention',), ple_layer_ids=(), vocab_size=16,
        )
        model = SimpleNamespace(
            _lock=threading.RLock(), config=config, identity='qwen4-fixture',
            weight_bytes=0, _snapshot_bytes=0, _state_bytes=lambda: 0,
            max_bytes=4 * 1024**2, _expert_bytes=0, _ple_bytes=0,
            layer_begin=0, layer_end=1, mrope_section=None,
            cache=['old-cache'], ple_cache=['old-ple'], mask='old-mask',
            positions='old-positions', position=7,
        )
        schema = [('cache/0/0', '<f4', [2])]
        header = dict(version=1, identity=model.identity, batch=1, position=1,
                      context=config.max_position_embeddings, schema=schema)
        encoded = qwen4_cache_recovery._canonical(header)
        wire = (qwen4_cache_recovery._MAGIC + struct.pack('<I', len(encoded)) +
                encoded + np.asarray([1.0, 2.0], dtype='<f4').tobytes() + b'\x00' * 32)
        references = []
        empty = np.empty

        def tracked_empty(*args, **kwargs):
            value = empty(*args, **kwargs)
            references.append(weakref.ref(value))
            return value

        with patch.object(qwen4_cache_recovery, '_access', return_value=nullcontext()), \
                patch.object(qwen4_cache_recovery, '_schema', return_value=schema), \
                patch.object(qwen4_cache_recovery.np, 'empty', side_effect=tracked_empty):
            with self.assertRaisesRegex(ValueError, 'checksum or EOF'):
                qwen4_cache_recovery.restore_qwen4_cache(
                    model, io.BytesIO(wire), max_bytes=4 * 1024**2)

        gc.collect()
        self.assertEqual(len(references), 1)
        self.assertIsNone(references[0]())
        self.assertEqual((model.cache, model.ple_cache, model.mask,
                          model.positions, model.position),
                         (['old-cache'], ['old-ple'], 'old-mask', 'old-positions', 7))

    def test_qwen35_rejected_restore_releases_allocated_state_arrays(self):
        config = SimpleNamespace(
            num_hidden_layers=1, layer_types=('linear_attention',),
        )
        old_mask=np.ones((1,1),np.uint8)
        model = SimpleNamespace(
            _lock=threading.RLock(), config=config, identity='qwen35-fixture',
            capacity=4, layer_begin=0, layer_end=1, weights={},
            _expert_bytes=0, _snapshot_bytes=0, max_bytes=4 * 1024**2,
            cache=[None], mask=old_mask, position=1,
        )
        schema = [('cache/0/0', '<f4', [2])]
        header = dict(identity=model.identity, position=1, batch=1, schema=schema)
        encoded = qwen35_cache_recovery._canonical(header)
        wire = (qwen35_cache_recovery._MAGIC + struct.pack('<I', len(encoded)) +
                encoded + np.asarray([1.0, 2.0], dtype='<f4').tobytes() + b'\x00' * 32)
        references = []
        zeros = np.zeros

        def tracked_zeros(*args, **kwargs):
            value = zeros(*args, **kwargs)
            references.append(weakref.ref(value))
            return value

        with patch.object(qwen35_cache_recovery, '_access', return_value=nullcontext()), \
                patch.object(qwen35_cache_recovery, '_schema', return_value=schema), \
                patch.object(qwen35_cache_recovery.np, 'zeros', side_effect=tracked_zeros):
            with self.assertRaisesRegex(ValueError, 'checksum/EOF mismatch'):
                qwen35_cache_recovery.restore_qwen35_cache(
                    model, io.BytesIO(wire), max_bytes=4 * 1024**2)

        gc.collect()
        self.assertEqual(len(references), 1)
        self.assertIsNone(references[0]())
        self.assertEqual((model.cache, model.position), ([None], 1))
        self.assertIs(model.mask,old_mask)

    def test_glm_rejected_restore_releases_allocated_cache_arrays(self):
        config = SimpleNamespace(
            max_position_embeddings=2, kv_lora_rank=1, qk_rope_head_dim=1,
            index_head_dim=1, indexer_types=('full',),
        )
        model = SimpleNamespace(
            _lock=threading.RLock(), _closed=False, _busy=False,
            config=config, identity='glm-fixture', cache=[], mask=None,
            position=0, _check_open=lambda: None,
            _admit_snapshot=lambda _size: None, _cancel=lambda: None,
        )
        header = native_glm_dsa_cache._header(model, 1, 1)
        encoded = json.dumps(header, sort_keys=True, separators=(',', ':')).encode()
        payload = (b'\x00' * 12 + b'\x01' + b'\x00' * 32)
        wire = (native_glm_dsa_cache.MAGIC + struct.pack('<I', len(encoded)) +
                encoded + payload)
        references = []
        zeros = np.zeros

        def tracked_zeros(*args, **kwargs):
            value = zeros(*args, **kwargs)
            references.append(weakref.ref(value))
            return value

        with patch.object(native_glm_dsa_cache.np, 'zeros', side_effect=tracked_zeros):
            with self.assertRaisesRegex(ValueError, 'checksum or trailing extent'):
                native_glm_dsa_cache.restore_cache(model, io.BytesIO(wire), max_bytes=4096)

        gc.collect()
        self.assertEqual(len(references), 3)
        self.assertTrue(all(reference() is None for reference in references))
        self.assertEqual((model.cache, model.mask, model.position), ([], None, 0))
        self.assertFalse(model._busy)

    def test_mixtral_rejected_restore_releases_allocated_cache_arrays(self):
        model = object.__new__(NativeMixtralDecoder)
        model._lock = threading.RLock()
        model._closed = model._busy = False
        model._expert_provider = None
        model._expert_identity = None
        model.identity = 'mixtral-fixture'
        model.max_context = 2
        model.config = dict(num_key_value_heads=1, head_dim=1, num_hidden_layers=1)
        model._cache = None
        model._position = 0
        model._mask = None
        model._weight_bytes = model._snapshot_bytes = model._expert_bytes = 0
        model._scratch_bytes = 0
        model.max_bytes = 8 * 1024**2
        model._run_inputs = None
        header = dict(version=1, identity=model.identity, context=2, batch=1,
                      position=1, shape=[1, 1, 1, 1], arrays=2, mask_bytes=1)
        encoded = mixtral_cache_recovery._canonical(header)
        payload = np.asarray([1.0, 2.0], dtype='<f4').tobytes() + b'\x01'
        wire = (mixtral_cache_recovery._MAGIC + struct.pack('<I', len(encoded)) +
                encoded + payload + b'\x00' * 32)
        references = []
        zeros = np.zeros

        def tracked_zeros(*args, **kwargs):
            value = zeros(*args, **kwargs)
            references.append(weakref.ref(value))
            return value

        with patch.object(mixtral_cache_recovery.np, 'zeros', side_effect=tracked_zeros):
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                mixtral_cache_recovery.restore_mixtral_cache(
                    model, io.BytesIO(wire), max_bytes=8 * 1024**2)

        gc.collect()
        self.assertEqual(len(references), 2)
        self.assertTrue(all(reference() is None for reference in references))
        self.assertIsNone(model._cache)
        self.assertEqual(model._position, 0)
        self.assertFalse(model._busy)


if __name__ == '__main__':
    unittest.main()
