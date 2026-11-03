"""The G25 retrieval, reasoning and memory records belong to one owner.

Documents, graph nodes and the scope gate were already covered. These three
services hold their own records, so each needs its own check that a second
owner, or the same owner under another project, is refused rather than served.
"""

from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

from openagent.advanced_reasoning_service import AdvancedReasoningService
from openagent.document_processor_service import DocumentProcessorService
from openagent.knowledge_graph_service import KnowledgeGraphService
from openagent.rag_service import RAGService
from openagent.semantic_memory_service import SemanticMemoryService
from openagent.vector_embeddings_service import VectorEmbeddingsService


class ReasoningChainIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = AdvancedReasoningService()
        self.chain = self.service.create_chain("durability", project="p", owner="owner-a")
        self.chain_id = str(self.chain.get("chain_id") or self.chain.get("id") or "")
        self.assertTrue(self.chain_id, f"create_chain returned no id: {self.chain}")

    def test_its_own_owner_can_read_it(self) -> None:
        self.assertIsNotNone(self.service.get_chain(self.chain_id, project="p", owner="owner-a"))

    def test_another_owner_cannot_read_it(self) -> None:
        self.assertIsNone(self.service.get_chain(self.chain_id, project="p", owner="owner-b"))

    def test_another_project_cannot_read_it(self) -> None:
        self.assertIsNone(self.service.get_chain(self.chain_id, project="other", owner="owner-a"))

    def test_an_unknown_chain_is_not_found(self) -> None:
        self.assertIsNone(self.service.get_chain("no-such-chain", project="p", owner="owner-a"))


class MemorySessionIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.service = SemanticMemoryService(state_dir=self._tmp.name)
        self.session_id = self.service.create_session(project="p", owner="owner-a")["session_id"]
        self.service.store_message(self.session_id, "user", "mine", project="p", owner="owner-a")

    def test_its_own_owner_reads_the_history(self) -> None:
        history = self.service.get_conversation_history(self.session_id, project="p", owner="owner-a")
        self.assertEqual([m["content"] for m in history["messages"]], ["mine"])

    def test_another_owner_is_refused_the_history(self) -> None:
        history = self.service.get_conversation_history(self.session_id, project="p", owner="owner-b")
        self.assertIn("error", history)

    def test_another_project_is_refused_the_history(self) -> None:
        history = self.service.get_conversation_history(self.session_id, project="other", owner="owner-a")
        self.assertIn("error", history)

    def test_another_owner_cannot_write_to_the_session(self) -> None:
        stored = self.service.store_message(self.session_id, "user", "theirs", project="p", owner="owner-b")
        self.assertIn("error", stored)
        history = self.service.get_conversation_history(self.session_id, project="p", owner="owner-a")
        self.assertEqual(len(history["messages"]), 1)

    def test_listing_shows_only_the_requesting_owner(self) -> None:
        self.service.create_session(project="p", owner="owner-b")
        self.assertEqual([s["id"] for s in self.service.list_sessions(project="p", owner="owner-a")], [self.session_id])
        self.assertNotIn(self.session_id, [s["id"] for s in self.service.list_sessions(project="p", owner="owner-b")])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class RetrievalIndexIsolationTests(unittest.TestCase):
    """The index is keyed by project, owner and document, not a joined string."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        embeddings = VectorEmbeddingsService()
        self.documents = DocumentProcessorService()
        self.service = RAGService(embeddings, self.documents, KnowledgeGraphService())

    def _index(self, project: str, owner: str, text: str) -> str:
        processed = self.documents.process_document(
            "note.txt", text.encode("utf-8"), project=project, owner=owner
        )
        doc_id = str(processed.get("document_id") or processed.get("id") or "")
        self.assertTrue(doc_id, f"process_document returned no id: {processed}")
        indexed = self.service.index_document(doc_id, project=project, owner=owner)
        self.assertTrue(indexed.get("indexed"), indexed)
        return doc_id

    def test_a_search_only_sees_the_requesting_scope(self) -> None:
        self._index("p", "owner-a", "Retrieval of the owner-a sentence.")
        self._index("p", "owner-b", "Retrieval of the owner-b sentence.")
        mine = self.service.search("retrieval", project="p", owner="owner-a")
        self.assertTrue(mine["results"])
        self.assertTrue(all("owner-b" not in row["text"] for row in mine["results"]), mine["results"])

    def test_another_project_sees_nothing(self) -> None:
        self._index("p", "owner-a", "Retrieval of the owner-a sentence.")
        self.assertEqual(self.service.search("retrieval", project="other", owner="owner-a")["results"], [])

    def test_a_colon_in_a_name_does_not_merge_two_scopes(self) -> None:
        # ("a", "b:c") and ("a:b", "c") joined on colons give the same string.
        self._index("a", "b:c", "Retrieval of the first scope sentence.")
        self._index("a:b", "c", "Retrieval of the second scope sentence.")
        first = self.service.search("retrieval", project="a", owner="b:c")["results"]
        self.assertEqual(len(first), 1, first)
        self.assertIn("first scope", first[0]["text"])

    def test_doc_id_narrows_within_the_scope(self) -> None:
        wanted = self._index("p", "owner-a", "Retrieval of the first document sentence.")
        self._index("p", "owner-a", "Retrieval of the second document sentence.")
        narrowed = self.service.search("retrieval", project="p", owner="owner-a", doc_id=wanted)
        self.assertTrue(narrowed["results"])
        self.assertTrue(all(row["doc_id"] == wanted for row in narrowed["results"]))


class MemoryRouteStatusTests(unittest.TestCase):
    """A refused memory read must not be reported as a completed operation."""

    def setUp(self) -> None:
        from http import HTTPStatus

        from openagent.routes import g25_advanced_routes

        self.routes = g25_advanced_routes
        self.ok = HTTPStatus.OK
        self.forbidden = HTTPStatus.FORBIDDEN
        self.not_found = HTTPStatus.NOT_FOUND

    def test_a_usable_result_keeps_its_ok_status(self) -> None:
        result = self.routes._memory_result({"session_id": "s", "messages": []})
        self.assertEqual(result.status, self.ok)

    def test_a_foreign_session_is_forbidden(self) -> None:
        result = self.routes._memory_result({"error": "Session does not belong to this project/owner"})
        self.assertEqual(result.status, self.forbidden)

    def test_an_unknown_session_is_not_found(self) -> None:
        result = self.routes._memory_result({"error": "Session not found"})
        self.assertEqual(result.status, self.not_found)
