"""Public family ABI identity must retain admitted checkpoint provenance."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from openagent.adaptive_engine.native_family_public_adapters import (
    _update_checkpoint_provenance,
)


class NativeFamilyPublicIdentityTests(unittest.TestCase):
    def test_identity_differs_for_distinct_checkpoint_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / 'first', root / 'second'
            first.mkdir()
            second.mkdir()
            inventories = [
                type('Inventory', (), {'root': path, 'config_sha256': 'a' * 64,
                                        'assert_unchanged': Mock()})()
                for path in (first, second)
            ]
            digests = []
            for inventory in inventories:
                digest = hashlib.sha256(b'same-runtime-and-tensors')
                _update_checkpoint_provenance(digest, inventory)
                inventory.assert_unchanged.assert_called_once_with()
                digests.append(digest.hexdigest())

            self.assertNotEqual(*digests)

    def test_identity_differs_for_checkpoint_config_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventories = [
                type('Inventory', (), {'root': root, 'config_sha256': config_digest,
                                        'assert_unchanged': Mock()})()
                for config_digest in ('a' * 64, 'b' * 64)
            ]
            digests = []
            for inventory in inventories:
                digest = hashlib.sha256(b'same-runtime-and-tensors')
                _update_checkpoint_provenance(digest, inventory)
                digests.append(digest.hexdigest())

            self.assertNotEqual(*digests)

    def test_identity_rejects_malformed_config_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            inventory = type('Inventory', (), {
                'root': Path(temporary), 'config_sha256': 'not-a-sha256',
                'assert_unchanged': Mock(),
            })()
            with self.assertRaisesRegex(ValueError, 'configuration digest'):
                _update_checkpoint_provenance(hashlib.sha256(), inventory)
            inventory.assert_unchanged.assert_not_called()


if __name__ == '__main__':
    unittest.main()
