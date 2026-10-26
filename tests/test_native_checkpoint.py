"""Local native checkpoint admission, conversion and exact dispatch."""
import json
import ctypes
import importlib.util
import os
from pathlib import Path
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from openagent.adaptive_engine.native_checkpoint import (
    read_native_text_checkpoint, load_native_text_decoder,
)
from openagent.adaptive_engine.source_scope import require_source_matrix_admission as _require_source_matrix_admission


class NativeCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config('qwen4_exp_text')
        # These tests cover checkpoint parsing/conversion and decoder plumbing
        # independently of the production P02 route allowlist. Keep a separate
        # assertion below for fail-closed candidate rejection.
        admission = patch(
            'openagent.adaptive_engine.source_scope.require_source_matrix_admission',
            lambda **identity: {'test_only_parser_contract': identity})
        admission.start()
        self.addCleanup(admission.stop)

    def test_candidate_tuple_is_rejected_by_production_admission(self):
        from openagent.adaptive_engine.source_scope import source_matrix_rows
        # Pick an exact generated tuple that remains provisional. Verified
        # routes are added explicitly as their source chains are established.
        row = next(row for row in source_matrix_rows()
                   if row['admission'] != 'verified-admitted')
        self.assertNotEqual(row['admission'], 'verified-admitted')
        with self.assertRaisesRegex(ValueError, 'not admitted'):
            _require_source_matrix_admission(**{
                key: row[key] for key in ('family', 'model_type', 'backend',
                    'arithmetic', 'source_format', 'persisted_format')})

    def test_public_family_text_loader_seals_loaded_decoder_to_exact_checkpoint(self):
        self.config('glm5_next_text')
        self.shard('model.safetensors','model.embed_tokens.weight','F32')
        library=self.root/'family-kernel.dll'
        library.write_bytes(b'fixture')
        decoder=SimpleNamespace(close=Mock())
        factory=Mock(return_value=decoder)
        from openagent.adaptive_engine.native_checkpoint import _DECODERS
        decoder_map=dict(_DECODERS)
        decoder_map['glm5_next_text']=('fixture','Decoder')
        with patch('openagent.adaptive_engine.native_checkpoint._DECODERS',
                   decoder_map), \
                patch('openagent.adaptive_engine.native_checkpoint.import_module',
                      return_value=SimpleNamespace(Decoder=factory)), \
                patch('openagent.adaptive_engine.native_checkpoint.read_native_text_checkpoint',
                      return_value=({'model_type':'glm5_next_text'},{})) as read:
            loaded=load_native_text_decoder(self.root,model_type='glm5_next_text',
                library=library,max_weight_bytes=4096,context_tokens=8,
                source_backend='public-range')
        self.assertIs(loaded,decoder)
        read.assert_called_once_with(self.root,model_type='glm5_next_text',
            max_weight_bytes=4096,read_limits=None,cancel=None,context_tokens=8,
            source_backend='public-range')
        from openagent.adaptive_engine.publisher_checkpoint.storage import snapshot
        from openagent.adaptive_engine.family_source_binding import require_decoder_snapshot
        inventory=snapshot(self.root,max_metadata_bytes=256*1024**2,
            max_tensors=200000,max_shards=256)
        self.assertTrue(require_decoder_snapshot(loaded,inventory,
            family='glm5-next',model_type='glm5_next_text'))

    def test_native_family_loader_seals_every_family_to_exact_checkpoint(self):
        """The provenance contract applies beyond the first two public adapters."""
        self.config('qwen4_exp_text')
        self.shard('model.safetensors','model.embed_tokens.weight','F32')
        library=self.root/'family-kernel.dll'
        library.write_bytes(b'fixture')
        decoder=SimpleNamespace(close=Mock())
        factory=Mock(return_value=decoder)
        from openagent.adaptive_engine.native_checkpoint import _DECODERS
        decoder_map=dict(_DECODERS)
        decoder_map['qwen4_exp_text']=('fixture','Decoder')
        with patch('openagent.adaptive_engine.native_checkpoint._DECODERS',
                   decoder_map), \
                patch('openagent.adaptive_engine.native_checkpoint.import_module',
                      return_value=SimpleNamespace(Decoder=factory)), \
                patch('openagent.adaptive_engine.native_checkpoint.read_native_text_checkpoint',
                      return_value=({'model_type':'qwen4_exp_text'},{})):
            loaded=load_native_text_decoder(self.root,model_type='qwen4_exp_text',
                library=library,max_weight_bytes=4096,source_backend='native')
        from openagent.adaptive_engine.publisher_checkpoint.storage import snapshot
        from openagent.adaptive_engine.family_source_binding import require_decoder_snapshot
        inventory=snapshot(self.root,max_metadata_bytes=256*1024**2,
            max_tensors=200000,max_shards=256)
        self.assertTrue(require_decoder_snapshot(loaded,inventory,
            family='qwen4',model_type='qwen4_exp_text'))

    def config(self, identity):
        (self.root / 'config.json').write_text(json.dumps({'model_type': identity}), encoding='utf-8')

    def shard(self, filename, name, dtype='F32', raw=None):
        if raw is None:
            raw = np.array([1, 2], dtype='<f4').tobytes()
        header = json.dumps({name: {'dtype': dtype, 'shape': [2], 'data_offsets': [0, len(raw)]}}).encode()
        (self.root / filename).write_bytes(struct.pack('<Q', len(header)) + header + raw)

    def read(self, budget=1024):
        # This class exercises the generic SafeTensors parser contract, not the
        # Qwen4 family geometry planner or production route admission. Its
        # explicit callback keeps that parser-only fixture use distinct from
        # production readers, which provide a family/fixture-specific gate.
        from openagent.adaptive_engine.native_checkpoint import _read_native_checkpoint
        return _read_native_checkpoint(self.root, model_type='qwen4_exp_text',
                                       max_weight_bytes=budget,
                                       metadata_admission=lambda *_: None)

    def test_low_level_reader_requires_metadata_admission_before_payload(self):
        from openagent.adaptive_engine.native_checkpoint import _read_native_checkpoint
        self.shard('a.safetensors', 'a')
        with patch('openagent.adaptive_engine.native_checkpoint._open_local',
                   side_effect=AssertionError('checkpoint opened before route admission')), \
             patch('openagent.adaptive_engine.native_checkpoint._read_tensor',
                   side_effect=AssertionError('payload read before route admission')):
            with self.assertRaisesRegex(ValueError, 'metadata admission callback required'):
                _read_native_checkpoint(self.root, model_type='qwen4_exp_text',
                                         max_weight_bytes=1024)

    def test_shards_convert_fp16_bf16_and_validate_index(self):
        self.shard('a.safetensors', 'a', 'F16', np.array([1, 2], dtype='<f2').tobytes())
        self.shard('b.safetensors', 'b', 'BF16', np.array([0x3f80, 0x4000], dtype='<u2').tobytes())
        (self.root / 'model.safetensors.index.json').write_text(
            json.dumps({'weight_map': {'a': 'a.safetensors', 'b': 'b.safetensors'}}), encoding='utf-8')
        config, weights = self.read()
        self.assertEqual(config['model_type'], 'qwen4_exp_text')
        for value in weights.values():
            self.assertEqual(value.dtype, np.float32)
            np.testing.assert_array_equal(value, [1, 2])

    def test_budget_rejects_before_payload_read(self):
        self.shard('a.safetensors', 'a')
        with patch('openagent.adaptive_engine.native_checkpoint._read_tensor', side_effect=AssertionError):
            with self.assertRaises(MemoryError):
                self.read(31)
        self.read(32)

    def test_dispatch_passes_library_and_limits_without_fallback(self):
        self.shard('a.safetensors', 'a')
        library = self.root / 'explicit.dll'
        library.write_bytes(b'placeholder for mocked constructor')
        with patch('openagent.adaptive_engine.native_checkpoint.read_native_text_checkpoint',
                   return_value=({'model_type': 'qwen4_exp_text'}, {'a': np.ones(2, np.float32)})), \
             patch('openagent.adaptive_engine.native_qwen4.NativeQwen4TextDecoder') as factory:
            self.assertIs(load_native_text_decoder(self.root, model_type='qwen4_exp_text',
                library=library, max_weight_bytes=1024, max_bytes=800), factory.return_value)
            self.assertEqual(factory.call_args.kwargs, {'library': library.resolve(), 'max_bytes': 800})

    def test_invalid_runtime_budget_and_missing_library_reject_before_loading(self):
        with patch('openagent.adaptive_engine.native_checkpoint.read_native_text_checkpoint', side_effect=AssertionError):
            for budget in (True, 0, -1):
                with self.assertRaises(ValueError):
                    load_native_text_decoder(self.root, model_type='qwen4_exp_text',
                        library='missing.dll', max_weight_bytes=1024, max_bytes=budget)
            with self.assertRaises(FileNotFoundError):
                load_native_text_decoder(self.root, model_type='qwen4_exp_text',
                    library=self.root / 'missing.dll', max_weight_bytes=1024)

    def test_family_specific_context_limits_are_explicit(self):
        from openagent.adaptive_engine.native_checkpoint import _DECODERS
        self.shard('a.safetensors', 'a')
        library = self.root / 'explicit.dll'
        library.write_bytes(b'placeholder for mocked constructor')
        for identity, keyword in (('glm5_next_text', 'max_tokens'),
                                  ('mixtral', 'max_context')):
            self.config(identity)
            module, name = _DECODERS[identity]
            with patch('openagent.adaptive_engine.native_checkpoint.read_native_text_checkpoint',
                       return_value=({'model_type': identity}, {'a': np.ones(2, np.float32)})), \
                 patch(f'openagent.adaptive_engine.{module}.{name}') as factory:
                load_native_text_decoder(self.root, model_type=identity, library=library,
                    max_weight_bytes=1024, context_tokens=7)
                self.assertEqual(factory.call_args.kwargs[keyword], 7)
        with self.assertRaisesRegex(ValueError, 'restricted without a complete'):
            load_native_text_decoder(self.root, model_type='deepseek_v4',
                library=library, max_weight_bytes=1024, context_tokens=7)
        # Kimi Linear is supported only through its exact standalone native
        # route. A mismatched checkpoint must fail at its identity validator.
        with self.assertRaisesRegex(ValueError, 'exact official Kimi Linear identity'):
            load_native_text_decoder(self.root, model_type='kimi_linear',
                library=library, max_weight_bytes=1024, context_tokens=7)
        with patch('openagent.adaptive_engine.native_checkpoint.read_native_text_checkpoint', side_effect=AssertionError):
            with self.assertRaisesRegex(ValueError, 'requires explicit context_tokens'):
                load_native_text_decoder(self.root, model_type='kimi_linear', library=library,
                    max_weight_bytes=1024)
            for context in (True, 0, -1):
                with self.assertRaisesRegex(ValueError, 'positive integer'):
                    load_native_text_decoder(self.root, model_type='glm5_next_text', library=library,
                        max_weight_bytes=1024, context_tokens=context)

    def test_exact_identity_duplicate_config_and_quantization_rejected(self):
        self.shard('a.safetensors', 'a')
        self.config('qwen4_exp')
        with self.assertRaisesRegex(ValueError, 'exact identity'):
            self.read()
        (self.root / 'config.json').write_text('{"model_type":"qwen4_exp_text","model_type":"qwen4_exp_text"}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.read()
        (self.root / 'config.json').write_text(json.dumps({'model_type': 'qwen4_exp_text', 'quantization_config': {}}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'quantized'):
            self.read()

    def test_nonfinite_duplicate_and_mismatched_shards_rejected(self):
        self.shard('a.safetensors', 'a', raw=np.array([np.nan, 1], dtype='<f4').tobytes())
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            self.read()
        self.shard('b.safetensors', 'a')
        with self.assertRaisesRegex(ValueError, 'more than one shard'):
            self.read()
        (self.root / 'b.safetensors').unlink()
        self.shard('a.safetensors', 'a')
        (self.root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': '../a.safetensors'}}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'nonlocal'):
            self.read()

    def test_invalid_budget_and_unsupported_identity(self):
        for budget in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                self.read(budget)
        with self.assertRaisesRegex(ValueError, 'unsupported exact'):
            read_native_text_checkpoint(self.root, model_type='kimi_k3', max_weight_bytes=1024)
        for identity in (None, {}, [], True):
            with self.assertRaisesRegex(ValueError, 'unsupported exact'):
                read_native_text_checkpoint(self.root, model_type=identity, max_weight_bytes=1024)

    def test_duplicate_shard_json_and_escaping_resolution_rejected(self):
        record = '{"dtype":"F32","shape":[2],"data_offsets":[0,8]}'
        header = ('{"a":' + record + ',"a":' + record + '}').encode()
        (self.root / 'a.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + bytes(8))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.read()
        (self.root / 'a.safetensors').unlink()
        self.shard('a.safetensors', 'a')
        original = Path.resolve
        def resolve(path, *args, **kwargs):
            if path.name == 'a.safetensors':
                return self.root.parent / 'escaped.safetensors'
            return original(path, *args, **kwargs)
        with patch.object(Path, 'resolve', resolve):
            with self.assertRaisesRegex(ValueError, 'escapes'):
                self.read()

    def test_opened_file_identity_checked_before_header_read(self):
        self.shard('a.safetensors', 'a')
        with tempfile.TemporaryDirectory() as outside:
            replacement = Path(outside) / 'replacement.safetensors'
            replacement.write_bytes(b'outside checkpoint')
            original = os.open
            def substitute(path, flags, *args, **kwargs):
                return original(replacement if Path(path).name == 'a.safetensors' else path,
                                flags, *args, **kwargs)
            with patch('openagent.adaptive_engine.native_checkpoint.os.open', substitute):
                with self.assertRaisesRegex(ValueError, 'changed during opening'):
                    self.read()

    def test_inventory_is_pinned_and_not_rescanned(self):
        from openagent.adaptive_engine import native_checkpoint
        self.shard('a.safetensors', 'a')
        original = native_checkpoint._header
        def add_shard(source, identity, **kwargs):
            result = original(source, identity, **kwargs)
            self.shard('late.safetensors', 'late')
            return result
        with patch.object(native_checkpoint, '_header', add_shard):
            _, weights = self.read()
        self.assertEqual(set(weights), {'a'})

    def test_hash_tables_are_exact_integer_tensors_not_float_weights(self):
        from openagent.adaptive_engine import native_checkpoint
        self.config('deepseek_v4')
        name = 'model.layers.0.mlp.gate.tid2eid'
        self.shard('a.safetensors', name, 'I64', np.array([1, 2], dtype='<i8').tobytes())
        _, weights = native_checkpoint._read_native_checkpoint(
            self.root, model_type='deepseek_v4', max_weight_bytes=1024,
            metadata_admission=lambda *_: None)
        self.assertEqual(weights[name].dtype, np.int64)
        np.testing.assert_array_equal(weights[name], [1, 2])
        self.shard('a.safetensors', 'model.layers.0.mlp.gate.weight', 'I64', bytes(16))
        with self.assertRaisesRegex(ValueError, 'unsupported weight'):
            native_checkpoint._read_native_checkpoint(
                self.root, model_type='deepseek_v4', max_weight_bytes=1024,
                metadata_admission=lambda *_: None)

    def test_in_place_mutation_rejects_whole_read(self):
        from openagent.adaptive_engine import native_checkpoint
        self.shard('a.safetensors', 'a')
        original = native_checkpoint._read_tensor
        def mutate(source, tensor, base, **kwargs):
            result = original(source, tensor, base, **kwargs)
            stamp = os.stat(self.root / 'a.safetensors')
            os.utime(self.root / 'a.safetensors',
                     ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1000000000))
            return result
        with patch.object(native_checkpoint, '_read_tensor', mutate):
            with self.assertRaisesRegex(ValueError, 'changed during loading'):
                self.read()


class NativeCheckpointExecutionTests(unittest.TestCase):
    @staticmethod
    def write_checkpoint(root, config, weights):
        (root / 'config.json').write_text(json.dumps(config), encoding='utf-8')
        mapping = {}
        items = list(weights.items())
        for shard, entries in enumerate((items[::2], items[1::2])):
            filename = f'model-{shard}.safetensors'
            header, payload = {}, bytearray()
            for name, value in entries:
                integer = value.dtype == np.int64
                start = len(payload)
                payload.extend(value.astype('<i8' if integer else '<f4').tobytes())
                header[name] = dict(dtype='I64' if integer else 'F32', shape=list(value.shape),
                                    data_offsets=[start, len(payload)])
                mapping[name] = filename
            encoded = json.dumps(header).encode()
            (root / filename).write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)
        (root / 'model.safetensors.index.json').write_text(
            json.dumps({'weight_map': mapping}), encoding='utf-8')

    def test_sharded_checkpoint_executes_compiled_glm_against_oracle(self):
        if any(importlib.util.find_spec(name) is None for name in ('torch', 'transformers')):
            self.skipTest('optional exact model oracle unavailable')
        from tests.test_independent_glm_dsa import fixture
        from tests.test_native_glm_dsa import build
        import torch

        oracle, baseline = fixture(self)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = build(root)
            self.write_checkpoint(root, oracle.config.to_dict(), baseline.weights)
            decoder = load_native_text_decoder(root, model_type='glm_moe_dsa', library=library,
                max_weight_bytes=16 * 1024**2, max_bytes=16 * 1024**2)
            try:
                ids = [1, 3, 5, 7, 9]
                with torch.no_grad():
                    expected = oracle(torch.tensor([ids])).logits.numpy()
                np.testing.assert_allclose(decoder.forward(ids).logits, expected, atol=4e-6, rtol=5e-5)
                decoder.reset()
                decoder.forward(ids[:2])
                np.testing.assert_allclose(decoder.forward(ids[2:]).logits, expected[:, 2:], atol=4e-6, rtol=5e-5)
            finally:
                if os.name == 'nt':
                    ctypes.windll.kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
                    ctypes.windll.kernel32.FreeLibrary(decoder.library._handle)

    def test_glm5_checkpoint_loads_native_head_and_continues(self):
        from tests.test_native_glm5_next import Glm5NextOracleContractTests

        case_type = Glm5NextOracleContractTests
        case_type.setUpClass()
        try:
            case = case_type()
            oracle, head, in_memory, weights = case.fixture()
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.write_checkpoint(root, oracle.config.to_dict(), weights)
                decoder = load_native_text_decoder(root, model_type='glm5_next_text',
                    library=case.library, context_tokens=5,
                    max_weight_bytes=16 * 1024**2, max_bytes=64 * 1024**2)
                case.decoders.append(decoder)
                from openagent.adaptive_engine.publisher_checkpoint.storage import snapshot
                from openagent.adaptive_engine.family_source_binding import require_decoder_snapshot
                inventory=snapshot(root,max_metadata_bytes=256*1024**2,
                    max_tensors=200000,max_shards=256)
                self.assertTrue(require_decoder_snapshot(decoder,inventory,
                    family='glm5-next',model_type='glm5_next_text'))
                ids = [2, 3, 4, 5, 6]
                with case.torch.no_grad():
                    hidden = oracle(case.torch.tensor([ids]), use_cache=False).last_hidden_state
                    expected = case.torch.nn.functional.linear(hidden, head).numpy()
                np.testing.assert_allclose(decoder.forward([ids])['logits'], expected,
                                           atol=1e-5, rtol=1e-5)
                decoder.reset()
                decoder.forward([ids[:2]])
                np.testing.assert_allclose(decoder.forward([ids[2:]])['logits'], expected[:, 2:],
                                           atol=1e-5, rtol=1e-5)
                in_memory.forward([ids])
                for loaded, reference in zip(decoder._cache, in_memory._cache):
                    for actual, wanted in zip(loaded, reference):
                        # Capacity-sized tails differ; active prefixes and recurrent state do not.
                        prefix = actual.shape
                        np.testing.assert_allclose(actual, wanted[tuple(slice(0, n) for n in prefix)],
                                                   atol=1e-5, rtol=1e-5)
                saved = tuple(tuple(value.copy() for value in layer) for layer in decoder._cache)
                with self.assertRaisesRegex(ValueError, 'context capacity'):
                    decoder.forward([[7]])
                self.assertEqual(decoder.position, len(ids))
                for before, after in zip(saved, decoder._cache):
                    for old, current in zip(before, after):
                        np.testing.assert_array_equal(old, current)
        finally:
            case_type.tearDownClass()

    def test_inkling_checkpoint_loads_native_hybrid_and_continues(self):
        from tests.test_native_inkling import NativeInklingTests

        case_type = NativeInklingTests
        case_type.setUpClass()
        decoder = None
        try:
            case = case_type()
            oracle, baseline, _ = case.fixture(
                layer_types=['hybrid', 'hybrid_sliding'], mlp_layer_types=['dense', 'sparse'],
                max_position_embeddings=5)
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.write_checkpoint(root, oracle.config.to_dict(), baseline.weights)
                decoder = load_native_text_decoder(root, model_type='inkling_text',
                    library=case.path, max_weight_bytes=16 * 1024**2, max_bytes=64 * 1024**2)
                from openagent.adaptive_engine.publisher_checkpoint.storage import snapshot
                from openagent.adaptive_engine.family_source_binding import require_decoder_snapshot
                inventory=snapshot(root,max_metadata_bytes=256*1024**2,
                    max_tensors=200000,max_shards=256)
                self.assertTrue(require_decoder_snapshot(decoder,inventory,
                    family='inkling',model_type='inkling_text'))
                ids = [2, 3, 4, 5, 6]
                with case.torch.no_grad():
                    reference = oracle(case.torch.tensor([ids]), use_cache=True)
                    expected = reference.logits.numpy()
                np.testing.assert_allclose(decoder.forward(ids).logits, expected,
                                           atol=3e-6, rtol=2e-5)
                case.compare_cache(decoder, reference)
                decoder.reset()
                decoder.forward(ids[:2])
                np.testing.assert_allclose(decoder.forward(ids[2:]).logits, expected[:, 2:],
                                           atol=3e-6, rtol=2e-5)
                case.compare_cache(decoder, reference)
                saved = decoder.cache_snapshot()
                with self.assertRaisesRegex(ValueError, 'context capacity'):
                    decoder.forward([7])
                self.assertEqual(decoder.position, len(ids))
                for before, after in zip(saved, decoder.cache_snapshot()):
                    np.testing.assert_array_equal(before, after)
        finally:
            if decoder is not None and os.name == 'nt':
                ctypes.windll.kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
                ctypes.windll.kernel32.FreeLibrary(decoder.library._handle)
            case_type.tearDownClass()

    def test_deepseek_hash_checkpoint_preserves_integer_tables_and_logits(self):
        self.check_deepseek_checkpoint()

    def test_mixtral_local_shards_execute_native_full_and_cached(self):
        from tests.test_native_mixtral_decoder import NativeMixtralDecoderTests
        case_type = NativeMixtralDecoderTests
        case_type.setUpClass()
        try:
            case = case_type()
            oracle, _, weights = case.fixture()
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.write_checkpoint(root, oracle.config.to_dict(), weights)
                decoder = load_native_text_decoder(root, model_type='mixtral',
                    library=case.library, max_weight_bytes=16 * 1024**2,
                    max_bytes=64 * 1024**2, context_tokens=16)
                case.loaded.append(decoder)
                ids = [1, 3, 5, 7, 9, 4]
                with case.torch.no_grad():
                    expected = oracle(case.torch.tensor([ids]), use_cache=True).logits.numpy()
                case.compare(decoder.forward(ids).logits, expected)
                decoder.reset()
                decoder.forward(ids[:2])
                case.compare(decoder.forward(ids[2:]).logits, expected[:, 2:])
        finally:
            case_type.doClassCleanups()

    def test_kimi_fused_local_shards_execute_native_full_and_cached(self):
        from tests.test_native_kimi import NativeKimiPinnedOracleTests
        case_type = NativeKimiPinnedOracleTests
        case_type.setUpClass()
        try:
            case = case_type()
            oracle, _, weights = case.fixture(q_lora_rank=6)
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.write_checkpoint(root, oracle.config.to_dict(), weights)
                decoder = load_native_text_decoder(root, model_type='kimi_linear',
                    library=case.path, max_weight_bytes=16 * 1024**2,
                    max_bytes=64 * 1024**2, context_tokens=96)
                case.instances.append(decoder)
                ids = [1, 3, 5, 7, 9, 4]
                with case.torch.inference_mode():
                    expected = oracle(case.torch.tensor([ids]), use_cache=True)
                    logits = expected.logits[0].numpy()
                np.testing.assert_allclose(decoder.forward(ids), logits, atol=5e-6, rtol=4e-5)
                with case.torch.inference_mode():
                    case.compare_cache(decoder, expected, oracle)
                decoder.reset()
                decoder.forward(ids[:2])
                np.testing.assert_allclose(decoder.forward(ids[2:]), logits[2:], atol=5e-6, rtol=4e-5)
                with case.torch.inference_mode():
                    case.compare_cache(decoder, expected, oracle)
        finally:
            case_type.tearDownClass()

    def test_deepseek_relu_checkpoint_executes_native_full_and_cached(self):
        self.check_deepseek_checkpoint(hidden_act='relu')

    def test_deepseek_tied_checkpoint_executes_native_full_and_cached(self):
        self.check_deepseek_checkpoint(tie_word_embeddings=True)

    def check_deepseek_checkpoint(self, **options):
        from tests.test_native_deepseek_v4 import require_oracle, build_native, NativeDeepseekV4Tests
        require_oracle()
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = build_native(root)
            case = NativeDeepseekV4Tests()
            case.torch, case.library, case.loaded = torch, library, []
            decoder = None
            try:
                oracle, baseline, _ = case.fixture(**options)
                self.write_checkpoint(root, oracle.config.to_dict(), baseline.weights)
                decoder = load_native_text_decoder(root, model_type='deepseek_v4', library=library,
                    max_weight_bytes=16 * 1024**2, max_bytes=64 * 1024**2, context_tokens=32)
                table = decoder.weights['model.layers.0.mlp.gate.tid2eid']
                self.assertEqual(table.dtype, np.int64)
                ids = [2, 3, 4, 5, 6]
                with torch.no_grad():
                    expected = oracle(torch.tensor([ids]), use_cache=False).logits.numpy()
                np.testing.assert_allclose(decoder.forward(ids).logits, expected, atol=8e-6, rtol=8e-5)
                decoder.reset()
                decoder.forward(ids[:2])
                np.testing.assert_allclose(decoder.forward(ids[2:]).logits, expected[:, 2:], atol=8e-6, rtol=8e-5)
            finally:
                if os.name == 'nt':
                    ctypes.windll.kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
                    for native in case.loaded + ([] if decoder is None else [decoder]):
                        ctypes.windll.kernel32.FreeLibrary(native.library._handle)
