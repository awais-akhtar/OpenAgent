from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.documents import DocumentManager
from openagent.storage import OpenAgentStore


class EditorDraftLifecycleTests(unittest.TestCase):
    def test_draft_update_records_versions(self) -> None:
        with TemporaryDirectory() as tmp:
            docs = DocumentManager(OpenAgentStore(tmp))
            doc = docs.create_document("Drafted", "base", owner="owner")
            draft = docs.create_draft(doc["doc_id"], "draft v1", base_version=1, owner="owner")
            updated = docs.update_draft(draft["draft_id"], "draft v2", base_version=1, owner="owner")
            versions = docs.list_draft_versions(draft["draft_id"], owner="owner")

            self.assertEqual(updated["content"], "draft v2")
            self.assertGreaterEqual(len(versions), 2)


if __name__ == "__main__":
    unittest.main()
