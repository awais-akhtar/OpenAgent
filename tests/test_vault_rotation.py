from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.storage import OpenAgentStore
from openagent.vault import EncryptedVault


class VaultRotationTests(unittest.TestCase):
    def test_failed_entry_aborts_rotation_without_changing_other_entries(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            vault = EncryptedVault(store)
            first = vault.store_secret("First", "one", "old-pass", project="core")
            second = vault.store_secret("Second", "two", "old-pass", project="core")
            with store.session() as connection:
                before = connection.execute(
                    "SELECT ciphertext, key_fingerprint FROM vault_entries WHERE id=?", (first["id"],)
                ).fetchone()
                connection.execute(
                    "UPDATE vault_entries SET ciphertext=? WHERE id=?", ("invalid-ciphertext", second["id"])
                )
            with self.assertRaisesRegex(ValueError, "no keys were changed"):
                vault.rotate_key("old-pass", "new-pass", project="core")
            with store.session() as connection:
                after = connection.execute(
                    "SELECT ciphertext, key_fingerprint FROM vault_entries WHERE id=?", (first["id"],)
                ).fetchone()
            self.assertEqual(tuple(before), tuple(after))
            self.assertEqual(vault.retrieve_secret(first["id"], "old-pass", project="core")["secret"], "one")


if __name__ == "__main__":
    unittest.main()
