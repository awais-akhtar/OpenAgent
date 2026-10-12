"""Compiled loading and metadata-first rejection of local K3 equation fixtures."""
import copy
import gc
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np

from openagent.adaptive_engine.native_kimi_k3_checkpoint import (
    hybrid_fixture_config, hybrid_fixture_document, load_native_kimi_k3_fixture,
)
from tests import test_native_checkpoint as checkpoint
from tests import test_native_kimi_k3_hybrid as hybrid


class KimiK3FixtureCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.library = hybrid.build(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        gc.collect()
        cls.directory.cleanup()

    def test_explicit_schema_rejects_vendor_identity_and_defaults(self):
        config = hybrid.configuration()
        document = hybrid_fixture_document(config)
        self.assertEqual(hybrid_fixture_config(document), config)
        for name, value in (('model_type', 'kimi_k3'), ('fixture_version', True),
                            ('fixture_version', 2), ('layer_types', 'kda')):
            bad = copy.deepcopy(document)
            bad[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                hybrid_fixture_config(bad)
        for component in ('mla', 'kda', 'moe'):
            bad = copy.deepcopy(document)
            bad[component].pop('hidden')
            with self.assertRaises(ValueError):
                hybrid_fixture_config(bad)
        bad = copy.deepcopy(document)
        bad['trust_remote_code'] = True
        with self.assertRaises(ValueError):
            hybrid_fixture_config(bad)

    def test_metadata_rejections_do_not_read_tensors(self):
        config = hybrid.configuration()
        document = hybrid_fixture_document(config)
        weights = hybrid.fixture(config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint.NativeCheckpointExecutionTests.write_checkpoint(root, document, weights)
            with patch('openagent.adaptive_engine.native_checkpoint._read_tensor',
                       side_effect=AssertionError('rejected payload read')):
                with self.assertRaisesRegex(ValueError, 'memory budget'):
                    load_native_kimi_k3_fixture(root, library=self.library,
                        max_weight_bytes=64*1024**2, max_bytes=1)
                with patch('openagent.adaptive_engine.native_kimi_k3_checkpoint.hybrid_tensor_shapes',
                           return_value={'missing': (1,)}):
                    with self.assertRaisesRegex(ValueError, 'inventory'):
                        load_native_kimi_k3_fixture(root, library=self.library,
                            max_weight_bytes=64*1024**2)

    def test_local_shards_full_and_cached_logits(self):
        config = hybrid.configuration()
        weights = hybrid.fixture(config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint.NativeCheckpointExecutionTests.write_checkpoint(
                root, hybrid_fixture_document(config), weights)
            decoder = load_native_kimi_k3_fixture(root, library=self.library,
                max_weight_bytes=64*1024**2)
            try:
                expected, _ = hybrid.analytical(config, weights, [1, 4, 2, 6])
                full = decoder.forward([1, 4, 2, 6])
                np.testing.assert_allclose(full, expected, atol=2e-5, rtol=2e-5)
                decoder.reset()
                chunked = np.concatenate([decoder.forward([1]), decoder.forward([4, 2, 6])])
                np.testing.assert_array_equal(full, chunked)
            finally:
                del decoder
                gc.collect()

    def test_missing_library_rejects_before_checkpoint_loading(self):
        with patch('openagent.adaptive_engine.native_kimi_k3_checkpoint._read_native_checkpoint',
                   side_effect=AssertionError('checkpoint read')):
            with self.assertRaises(FileNotFoundError):
                load_native_kimi_k3_fixture('missing', library='missing-library.dll',
                    max_weight_bytes=1024)

    def test_incompatible_library_rejected_before_payload_and_released(self):
        from openagent.adaptive_engine import native_kimi_k3_checkpoint as loader
        for symbol, version in [('oa_k3_blocks_version', 2),
                                ('oa_k3_kda_projection_version', 2),
                                ('oa_kimi_kda_version', 2), ('oa_kimi_k3_moe', None)]:
            fake = Mock()
            fake._handle = 123
            for name in ('oa_k3_blocks_version', 'oa_k3_kda_projection_version', 'oa_kimi_kda_version'):
                getattr(fake, name).return_value = 1
            if version is None:
                delattr(fake, symbol)
            else:
                getattr(fake, symbol).return_value = version
            with patch.object(loader.C, 'CDLL', return_value=fake), \
                 patch.object(loader, '_release_library') as release, \
                 patch.object(loader, '_read_native_checkpoint', side_effect=AssertionError('payload')):
                with self.subTest(symbol=symbol), self.assertRaisesRegex(ValueError, 'ABI|symbol'):
                    load_native_kimi_k3_fixture('missing', library=self.library, max_weight_bytes=1024)
                release.assert_called_once_with(123)
