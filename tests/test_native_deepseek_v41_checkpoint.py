"""Local CED shard loading and metadata-first admission checks."""
import copy
import gc
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np

from openagent.adaptive_engine.native_deepseek_v41_checkpoint import (
    ced_fixture_config, ced_fixture_document, load_native_ced_fixture,
)
from tests import test_native_deepseek_v41_ced as ced
from tests import test_native_checkpoint as checkpoint


class CEDCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.library = ced.build(cls.temporary.name)

    @classmethod
    def tearDownClass(cls):
        gc.collect()
        cls.temporary.cleanup()

    def test_schema_is_explicit_and_not_vendor_alias(self):
        config = ced.DeepSeekV41CEDConfig()
        document = ced_fixture_document(config)
        self.assertEqual(ced_fixture_config(document), config)
        for name, value in [('model_type', 'deepseek_v41'), ('fixture_version', True),
                            ('fixture_version', 2), ('ced', {})]:
            bad = copy.deepcopy(document)
            bad[name] = value
            with self.assertRaises(ValueError):
                ced_fixture_config(bad)
        document['ced'].pop('epsilon')
        with self.assertRaises(ValueError):
            ced_fixture_config(document)

    def test_local_shards_logits_chunking_and_restore(self):
        config = ced.DeepSeekV41CEDConfig()
        weights = ced.fixture(config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint.NativeCheckpointExecutionTests.write_checkpoint(root, ced_fixture_document(config), weights)
            decoder = load_native_ced_fixture(root, library=self.library, max_weight_bytes=64*1024**2)
            expected, _ = ced.analytical(config, weights, [1, 3, 5, 2])
            full = decoder.forward([1, 3, 5, 2])
            np.testing.assert_allclose(full, expected, atol=2e-5, rtol=2e-5)
            decoder.reset()
            first = decoder.forward([1, 3])
            snapshot = decoder.snapshot()
            second = decoder.forward([5, 2])
            np.testing.assert_array_equal(full, np.concatenate([first, second]))
            decoder.restore(snapshot)
            np.testing.assert_array_equal(second, decoder.forward([5, 2]))

    def test_budget_and_inventory_reject_before_payload(self):
        config = ced.DeepSeekV41CEDConfig()
        weights = ced.fixture(config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint.NativeCheckpointExecutionTests.write_checkpoint(root, ced_fixture_document(config), weights)
            with patch('openagent.adaptive_engine.native_checkpoint._read_tensor',
                       side_effect=AssertionError('payload read')):
                with self.assertRaisesRegex(ValueError, 'budget'):
                    load_native_ced_fixture(root, library=self.library,
                                            max_weight_bytes=64*1024**2, max_bytes=1)
                with patch('openagent.adaptive_engine.native_deepseek_v41_checkpoint.ced_tensor_shapes',
                           return_value={'wrong': (1,)}):
                    with self.assertRaisesRegex(ValueError, 'inventory'):
                        load_native_ced_fixture(root, library=self.library, max_weight_bytes=64*1024**2)

    def test_library_is_checked_before_checkpoint(self):
        with patch('openagent.adaptive_engine.native_deepseek_v41_checkpoint._read_native_checkpoint',
                   side_effect=AssertionError('checkpoint read')):
            with self.assertRaises(FileNotFoundError):
                load_native_ced_fixture('missing', library='missing.dll', max_weight_bytes=1024)

    def test_bad_abi_releases_library_without_payload(self):
        from openagent.adaptive_engine import native_deepseek_v41_checkpoint as loader
        for missing in (False, True):
            library = Mock()
            library._handle = 123
            library.oa_v41_ced_version.return_value = 2 if not missing else 1
            if missing:
                del library.oa_v41_ced_forward
            with patch.object(loader.C, 'CDLL', return_value=library), \
                 patch.object(loader, '_release_library') as release, \
                 patch.object(loader, '_read_native_checkpoint', side_effect=AssertionError('payload')):
                with self.assertRaisesRegex(ValueError, 'ABI|symbol'):
                    load_native_ced_fixture('unused', library=self.library, max_weight_bytes=1024)
                release.assert_called_once_with(123)
