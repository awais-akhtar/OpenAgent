import hashlib
import tempfile
from pathlib import Path
import unittest

import numpy as np

from openagent.adaptive_engine.wholemodel_validation.olmoe_native import (
    cache_schema,
    create_olmoe_native_adapter,
    read_complete_cache,
)
from openagent.adaptive_engine.wholemodel_validation.cache_readers import CacheContext


class WholemodelOlmoeAdapterTest(unittest.TestCase):
    def test_schema_and_cache_reader_export_every_physical_layer(self):
        schema = cache_schema(2)
        self.assertEqual(schema, {
            'position': 'global',
            'layer.0.key': 0, 'layer.0.value': 0,
            'layer.1.key': 1, 'layer.1.value': 1,
        })
        keys = [np.ones((2, 3, 4), dtype=np.float32),
                np.full((2, 3, 4), 2, dtype=np.float32)]
        values = [np.full((2, 3, 4), 3, dtype=np.float32),
                  np.full((2, 3, 4), 4, dtype=np.float32)]
        state = {
            'format': 'openagent.native-olmoe-decoder.v1',
            'edge_execution_identity': 'edge-fixture',
            'range_state': {'position': 3, 'keys': keys, 'values': values},
        }
        observed = read_complete_cache(CacheContext(state, None, None, 3, 1))
        self.assertEqual(set(observed), set(schema))
        self.assertEqual(observed['layer.0.key'].shape, (1, 2, 3, 4))
        np.testing.assert_array_equal(observed['layer.1.value'][0], values[1])

    def test_reader_rejects_incomplete_layer_or_position_state(self):
        state = {
            'format': 'openagent.native-olmoe-decoder.v1',
            'edge_execution_identity': 'edge-fixture',
            'range_state': {
                'position': 2,
                'keys': [np.zeros((1, 2, 3), dtype=np.float32)],
                'values': [],
            },
        }
        with self.assertRaisesRegex(ValueError, 'layer inventory differs'):
            read_complete_cache(CacheContext(state, None, None, 2, 1))

    def test_adapter_requires_pinned_manifest_and_native_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / 'experts.manifest.json'
            library = root / 'native.dll'
            manifest.write_bytes(b'fixture manifest')
            library.write_bytes(b'fixture native library')
            metadata = {
                'model_type': 'olmoe', 'revision': 'fixture-revision',
                'config_sha256': 'a' * 64, 'weights_sha256': 'b' * 64,
                'tokenizer_sha256': 'c' * 64, 'vocabulary_size': 8,
                'num_layers': 2, 'dtype': 'float32',
                'implementation': 'native-fixture',
                'implementation_revision': 'fixture-build', 'device': 'cpu',
                'dependencies': {'native': 'fixture'},
                'cache_schema': cache_schema(2),
            }
            pins = [
                {'kind': 'weights', 'path': manifest.name,
                 'sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()},
                {'kind': 'dependency', 'path': library.name,
                 'sha256': hashlib.sha256(library.read_bytes()).hexdigest()},
            ]
            adapter = create_olmoe_native_adapter(offline_root=root, options={
                '_artifact_pins': pins, 'metadata': metadata,
                'manifest': manifest.name, 'library': library.name,
                'context_tokens': 8, 'resident_bytes': 4096,
                'expert_cache_bytes': 4096, 'transfer_bytes': 4096,
                'staging_bytes': 4096, 'snapshot_bytes': 4096,
            })
            self.assertEqual(adapter.metadata()['cache_schema'], cache_schema(2))
            pins[0]['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'differs from runner pin'):
                create_olmoe_native_adapter(offline_root=root, options={
                    '_artifact_pins': pins, 'metadata': metadata,
                    'manifest': manifest.name, 'library': library.name,
                    'context_tokens': 8, 'resident_bytes': 4096,
                    'expert_cache_bytes': 4096, 'transfer_bytes': 4096,
                    'staging_bytes': 4096, 'snapshot_bytes': 4096,
                })


if __name__ == '__main__':
    unittest.main()
