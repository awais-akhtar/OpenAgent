from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.documents import DocumentManager
from openagent.storage import OpenAgentStore


class DocumentVersionLifecycleTests(unittest.TestCase):
    def test_restore_creates_new_head_version(self) -> None:
        with TemporaryDirectory() as tmp:
            docs = DocumentManager(OpenAgentStore(tmp))
            created = docs.create_document("Plan", "v1", owner="owner")
            updated = docs.update_document(created["doc_id"], content="v2", if_version=1, owner="owner")
            restored = docs.restore_version(created["doc_id"], 1, owner="owner")

            self.assertEqual(updated["version"], 2)
            self.assertEqual(restored["new_version"], 3)
            self.assertEqual(docs.get_document(created["doc_id"], owner="owner")["content"], "v1")


if __name__ == "__main__":
    unittest.main()
