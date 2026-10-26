from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from openagent.adaptive_engine.family_source_binding import (
    bind_decoder_to_snapshot, require_decoder_snapshot,
)
from openagent.adaptive_engine.native_glm5_next import NativeGlm5NextTextDecoder
from openagent.adaptive_engine.native_family_public_adapters import (
    FamilyModelDescriptor, NativeFamilyPublicAdapters, _verify_family_abi_layouts,
)
from openagent.adaptive_engine.native_qwen4 import NativeQwen4TextDecoder
from openagent.adaptive_engine.native_qwen4_public_adapters import NativeQwen4PublicAdapters
from openagent.adaptive_engine.segment_edge_adapters import NativeOlmoePublicAdapters
from openagent.adaptive_engine.publisher_public_adapters import NativePublisherPublicAdapters
from openagent.adaptive_engine.expert_runtime import ExpertLayer, SparseExpertManifest
from openagent.adaptive_engine.expert_store import ExpertRegion
from openagent.adaptive_engine.mixtral_public_adapters import NativeMixtralPublicAdapters
from openagent.adaptive_engine.qwen35_public_adapters import NativeQwen35PublicAdapters
from openagent.adaptive_engine.glm_dsa_public_adapters import NativeGlmDsaPublicAdapters
from openagent.adaptive_engine.qwen4_public_adapters import NativeQwen4PublicAdapters as NativeQwen4RangePublicAdapters


class FamilySourceBindingTests(unittest.TestCase):
    def test_family_public_abi_probe_requires_exact_nested_record_layouts(self):
        import ctypes as C
        from openagent.adaptive_engine.native_inkling import Config, Layer, Tensor

        records=((1,FamilyModelDescriptor),(2,Tensor),(3,Config),(4,Layer))
        expected={number:[C.sizeof(record),len(record._fields_)]+[
            getattr(record,name).offset for name,_ in record._fields_]
            for number,record in records}

        class Probe:
            argtypes=None
            restype=None

            def __init__(self, mutate=None):
                self.mutate=mutate

            def __call__(self, number, output, capacity):
                values=list(expected[number])
                if self.mutate is not None:
                    values=self.mutate(number,values)
                if len(values)>capacity:
                    return len(values)
                for index,value in enumerate(values):
                    output[index]=value
                return len(values)

        valid=Probe()
        _verify_family_abi_layouts(valid)
        self.assertEqual(valid.restype,C.c_size_t)
        self.assertEqual(valid.argtypes,[C.c_uint32,C.POINTER(C.c_uint64),C.c_size_t])

        mutations=(
            lambda number,values: values[:-1] if number==1 else values,
            lambda number,values: [values[0],values[1]+1,*values[2:]] if number==2 else values,
            lambda number,values: [values[0]+1,*values[1:]] if number==3 else values,
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.assertRaisesRegex(
                    ValueError,'native family descriptor ABI layout differs'):
                _verify_family_abi_layouts(Probe(mutate))

    def test_qwen4_range_adapter_does_not_inherit_mixtral_source_identity(self):
        self.assertEqual(NativeQwen4RangePublicAdapters._source_route_identity,
                         ('qwen4', 'qwen4_exp_text'))

    def test_public_range_adapters_bind_their_own_source_families(self):
        self.assertEqual(NativeMixtralPublicAdapters._source_route_identity,
                         ('mixtral', 'mixtral'))
        self.assertEqual(NativeQwen35PublicAdapters._source_route_identity,
                         ('qwen35', 'qwen3_5_moe_text'))
        self.assertEqual(NativeGlmDsaPublicAdapters._source_route_identity,
                         ('glm-dsa', 'glm_moe_dsa'))

    def test_public_range_subclass_cannot_inherit_another_family_gate(self):
        class MissingRouteAdapter(NativeMixtralPublicAdapters):
            pass

        with self.assertRaisesRegex(ValueError, 'declare an exact source family/model identity'):
            MissingRouteAdapter('unused-runtime', kernel_library='unused-kernel',
                model_directory='.', max_staging_bytes=1024, max_snapshot_bytes=1024)

    def test_mixtral_public_range_rejects_unbound_expert_provider_before_opening_libraries(self):
        with patch('openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission') as admission, \
                patch('openagent.adaptive_engine.native_library_identity.NativeLibraryIdentity.read') as read:
            with self.assertRaisesRegex(ValueError,'packed experts are restricted'):
                NativeMixtralPublicAdapters('missing-runtime',kernel_library='missing-kernel',
                    model_directory='.',max_staging_bytes=1024,max_snapshot_bytes=1024,
                    expert_provider=object())
        admission.assert_not_called()
        read.assert_not_called()

    def test_glm_dsa_public_range_rejects_unbound_expert_provider_before_opening_libraries(self):
        with patch('openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission') as admission, \
                patch('openagent.adaptive_engine.native_library_identity.NativeLibraryIdentity.read') as read:
            with self.assertRaisesRegex(ValueError,'packed experts are restricted'):
                NativeGlmDsaPublicAdapters('missing-runtime',kernel_library='missing-kernel',
                    model_directory='.',max_staging_bytes=1024,max_snapshot_bytes=1024,
                    expert_provider=object())
        admission.assert_not_called()
        read.assert_not_called()

    def test_direct_olmoe_public_adapter_rejects_foreign_manifest_before_loading_abi(self):
        layer = ExpertLayer('model.layers.0', 8, 6, 1, 'int8-row', 'swiglu', {
            '0': ExpertRegion('experts.i8', 0, 224, 'a' * 64),
        })
        manifest = SparseExpertManifest('model', 'foreign', 'sha256:' + 'b' * 64,
                                        (layer,))
        with tempfile.TemporaryDirectory() as directory, patch(
                'openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission',
                return_value=SimpleNamespace(tensors={})):
            with self.assertRaisesRegex(ValueError, 'requires an OLMoE expert manifest'):
                NativeOlmoePublicAdapters('missing-runtime', kernel_library='missing-kernel',
                    model_directory=directory, manifest=manifest, expert_cache_bytes=1024,
                    max_staging_bytes=1024, max_snapshot_bytes=1024)

    def test_direct_olmoe_public_adapter_binds_manifest_fingerprint_before_abi_load(self):
        config = SimpleNamespace(hidden_size=8, intermediate_size=6,
            num_hidden_layers=1, num_experts=1, num_experts_per_tok=1)
        layer = ExpertLayer('model.layers.0', 8, 6, 1, 'int8-row', 'swiglu', {
            '0': ExpertRegion('experts.i8', 0, 224, 'a' * 64),
        })
        manifest = SparseExpertManifest('model', 'olmoe', 'sha256:' + 'b' * 64,
                                        (layer,))
        with tempfile.TemporaryDirectory() as directory, patch(
                'openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission',
                return_value=SimpleNamespace(tensors={})), patch(
                'openagent.adaptive_engine.olmoe.OlmoeConfig.read', return_value=config), patch(
                'openagent.adaptive_engine.olmoe.olmoe_metadata_fingerprint',
                return_value='sha256:' + 'c' * 64):
            with self.assertRaisesRegex(ValueError, 'belongs to a different checkpoint'):
                NativeOlmoePublicAdapters('missing-runtime', kernel_library='missing-kernel',
                    model_directory=directory, manifest=manifest, expert_cache_bytes=1024,
                    max_staging_bytes=1024, max_snapshot_bytes=1024)

    def test_direct_publisher_adapter_enforces_route_before_loading_abi(self):
        from openagent.adaptive_engine.k3_execution.executor import K3Config

        class FakeService:
            pass

        config=K3Config(hidden=8,vocab=16,layer_types=('mla',),heads=1,nope=2,
            rope=2,value=4,qrank=2,kvrank=2,kda_heads=1,kda_dim=4,kernel=2,
            dense_intermediate=4,latent=2,expert_intermediate=4,experts=2,
            selected=1,shared=1,capacity=8)
        with tempfile.TemporaryDirectory() as directory, \
                patch('openagent.adaptive_engine.execution_service.NativeExecutionService',FakeService), \
                patch('openagent.adaptive_engine.source_scope.require_source_family_backend_admission',
                      side_effect=ValueError('P02 route rejected')) as admission, \
                patch('openagent.adaptive_engine.native_library_identity.NativeLibraryIdentity.read') as read:
            with self.assertRaisesRegex(ValueError,'P02 route rejected'):
                NativePublisherPublicAdapters('missing-runtime',FakeService(),family='kimi-k3',
                    config=config,model_directory=directory,checkpoint_identity='pinned',
                    executor_bytes=1024,max_resource_bytes=4096,max_snapshot_bytes=1024,
                    max_work_bytes=1024)
        admission.assert_called_once_with(family='kimi-k3',model_type='kimi_k3',
            backend='public-range',arithmetic='fp32-approximate')
        read.assert_not_called()

    def test_direct_publisher_adapter_binds_checkpoint_headers_before_loading_abi(self):
        from openagent.adaptive_engine.k3_execution.executor import K3Config

        class FakeService:
            pass

        config=K3Config(hidden=8,vocab=16,layer_types=('mla',),heads=1,nope=2,
            rope=2,value=4,qrank=2,kvrank=2,kda_heads=1,kda_dim=4,kernel=2,
            dense_intermediate=4,latent=2,expert_intermediate=4,experts=2,
            selected=1,shared=1,capacity=8)
        with tempfile.TemporaryDirectory() as directory, \
                patch('openagent.adaptive_engine.execution_service.NativeExecutionService',FakeService), \
                patch('openagent.adaptive_engine.source_scope.require_source_family_backend_admission') as admission, \
                patch('openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission',
                      side_effect=ValueError('P02 checkpoint rejected')) as inventory, \
                patch('openagent.adaptive_engine.native_library_identity.NativeLibraryIdentity.read') as read:
            with self.assertRaisesRegex(ValueError,'P02 checkpoint rejected'):
                NativePublisherPublicAdapters('missing-runtime',FakeService(),family='kimi-k3',
                    config=config,model_directory=directory,checkpoint_identity='pinned',
                    executor_bytes=1024,max_resource_bytes=4096,max_snapshot_bytes=1024,
                    max_work_bytes=1024)
        admission.assert_called_once_with(family='kimi-k3',model_type='kimi_k3',
            backend='public-range',arithmetic='fp32-approximate')
        inventory.assert_called_once_with(Path(directory).resolve(),family='kimi-k3',
            model_type='kimi_k3',backend='public-range',arithmetic='fp32-approximate',
            max_metadata_bytes=1024)
        read.assert_not_called()

    def test_public_publisher_rejects_payload_from_another_checkpoint_before_abi(self):
        from openagent.adaptive_engine.k3_execution.executor import K3Config

        class FakeService:
            def __init__(self,payload):
                self.publisher_payload=payload

        config=K3Config(hidden=8,vocab=16,layer_types=('mla',),heads=1,nope=2,
            rope=2,value=4,qrank=2,kvrank=2,kda_heads=1,kda_dim=4,kernel=2,
            dense_intermediate=4,latent=2,expert_intermediate=4,experts=2,
            selected=1,shared=1,capacity=8)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            inventory=SimpleNamespace(root=root,files={'config.json':(1,2,3)},
                config_sha256='current-config',assert_unchanged=Mock())
            payload=SimpleNamespace(_source_root=str(root),
                _source_files=(('config.json',(1,2,999)),),
                config_sha256='current-config',config=config)
            with patch('openagent.adaptive_engine.execution_service.NativeExecutionService',FakeService), \
                    patch('openagent.adaptive_engine.source_scope.require_source_family_backend_admission'), \
                    patch('openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission',
                          return_value=inventory), \
                    patch('openagent.adaptive_engine.publisher_checkpoint.k3_adapter.require_k3_payload_admission'), \
                    patch('openagent.adaptive_engine.native_library_identity.NativeLibraryIdentity.read') as read:
                with self.assertRaisesRegex(ValueError,'not the same checkpoint snapshot'):
                    NativePublisherPublicAdapters('missing-runtime',FakeService(payload),
                        family='kimi-k3',config=config,model_directory=root,
                        checkpoint_identity='pinned',executor_bytes=1024,
                        max_resource_bytes=4096,max_snapshot_bytes=1024,max_work_bytes=1024)
        inventory.assert_unchanged.assert_not_called()
        read.assert_not_called()

    def test_lazy_public_range_load_releases_owner_when_checkpoint_changes(self):
        adapter=object.__new__(NativeMixtralPublicAdapters)
        adapter._source_snapshot=SimpleNamespace(assert_unchanged=Mock(
            side_effect=[None,ValueError('checkpoint changed')]))
        adapter.lock=threading.RLock();adapter.objects={};adapter.pending={}
        owner=SimpleNamespace(close=Mock())
        with self.assertRaisesRegex(ValueError,'checkpoint changed'):
            adapter._load_source_owner(lambda:owner,kind='engine')
        owner.close.assert_called_once_with()
        self.assertEqual(adapter.objects,{})
        self.assertEqual(adapter.pending,{})

    def test_lazy_public_range_load_retains_owner_if_changed_source_cleanup_fails(self):
        adapter=object.__new__(NativeMixtralPublicAdapters)
        adapter._source_snapshot=SimpleNamespace(assert_unchanged=Mock(
            side_effect=[None,ValueError('checkpoint changed')]))
        adapter.lock=threading.RLock();adapter.objects={};adapter.pending={}
        owner=SimpleNamespace(close=Mock(side_effect=OSError('close failed')))
        with self.assertRaisesRegex(ValueError,'checkpoint changed') as raised:
            adapter._load_source_owner(lambda:owner,kind='engine')
        owner.close.assert_called_once_with()
        self.assertEqual(tuple(adapter.objects.values()),(('engine',owner),))
        self.assertEqual(tuple(adapter.pending.values()),(owner.close.side_effect,))
        self.assertTrue(any('cleanup retained' in note for note in raised.exception.__notes__))

    def snapshot(self,*,files=None):
        return SimpleNamespace(root=Path('.'),config_sha256='config-digest',
            files={'config.json':(1,2,3,4,5)} if files is None else files,
            assert_unchanged=Mock())

    def test_public_binding_requires_exact_checkpoint_file_snapshot(self):
        decoder=SimpleNamespace();inventory=self.snapshot()
        bind_decoder_to_snapshot(decoder,inventory,family='glm5-next',model_type='glm5_next_text')
        self.assertTrue(require_decoder_snapshot(decoder,inventory,
            family='glm5-next',model_type='glm5_next_text'))
        inventory.assert_unchanged.assert_called()

    def test_public_binding_rejects_other_root_or_changed_shard_set(self):
        decoder=SimpleNamespace();loaded=self.snapshot()
        bind_decoder_to_snapshot(decoder,loaded,family='inkling',model_type='inkling_text')
        other=self.snapshot(files={'config.json':(1,2,3,4,5),'model.safetensors':(6,7,8,9,10)})
        with self.assertRaisesRegex(ValueError,'exact unchanged family checkpoint'):
            require_decoder_snapshot(decoder,other,family='inkling',model_type='inkling_text')

    def test_public_binding_rejects_decoder_without_loader_receipt(self):
        with self.assertRaisesRegex(ValueError,'exact unchanged family checkpoint'):
            require_decoder_snapshot(SimpleNamespace(),self.snapshot(),
                                     family='inkling',model_type='inkling_text')

    def test_public_adapter_rejects_unbound_decoder_before_loading_native_abi(self):
        decoder=object.__new__(NativeGlm5NextTextDecoder)
        decoder.position=0;decoder.config=SimpleNamespace();decoder._expert_provider=None
        decoder.weights={}
        inventory=self.snapshot()
        with patch('openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission',
                   return_value=inventory):
            with self.assertRaisesRegex(ValueError,'exact unchanged family checkpoint'):
                NativeFamilyPublicAdapters('nonexistent-native-runtime',decoder,
                    model_directory='.',max_resource_bytes=1024*1024,max_work_bytes=1024)

    def test_qwen4_public_adapter_rejects_header_only_checkpoint_admission(self):
        decoder=object.__new__(NativeQwen4TextDecoder)
        decoder.range_mode=False;decoder.position=0;decoder._snapshots={}
        inventory=self.snapshot()
        with patch('openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission',
                   return_value=inventory):
            with self.assertRaisesRegex(ValueError,'exact unchanged family checkpoint'):
                NativeQwen4PublicAdapters('nonexistent-native-runtime',decoder,
                    model_directory='.',max_resource_bytes=1024*1024,max_work_bytes=1024)


if __name__=='__main__':
    unittest.main()
