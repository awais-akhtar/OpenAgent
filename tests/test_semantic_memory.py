from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.knowledge import KnowledgeEngine
from openagent.semantic_memory import SemanticMemoryManager
from openagent.storage import OpenAgentStore


class SemanticMemoryManagerTests(unittest.TestCase):
    def test_collection_rename_and_delete_preserve_project_boundaries_and_records(self) -> None:
        with TemporaryDirectory() as tmp:
            semantic = SemanticMemoryManager(OpenAgentStore(tmp))
            collection = semantic.create_collection("research", project="core")
            other_project = semantic.create_collection("research", project="other")
            record = semantic.upsert_vector_record(collection["id"], content="Persisted vector", project="core")

            renamed = semantic.rename_collection("research", new_name="research-v2", project="core")
            self.assertEqual(renamed["id"], collection["id"])
            self.assertEqual(renamed["name"], "research-v2")
            self.assertEqual(semantic.get_vector_record(record["id"])["content"], "Persisted vector")
            self.assertEqual(semantic.collections(project="other")[0]["id"], other_project["id"])
            with self.assertRaises(ValueError):
                semantic.rename_collection("research-v2", new_name="research-v2", project="core")
            with self.assertRaises(KeyError):
                semantic.rename_collection("research-v2", new_name="forged", project="other")

            removed = semantic.delete_collection("research-v2", project="core")
            self.assertEqual(removed["records_removed"], 1)
            self.assertIsNone(semantic.get_vector_record(record["id"]))
            self.assertEqual(semantic.collections(project="core"), [])
            self.assertEqual(len(semantic.collections(project="other")), 1)

    def test_repeated_writes_get_distinct_record_ids(self) -> None:
        with TemporaryDirectory() as tmp:
            semantic = SemanticMemoryManager(OpenAgentStore(tmp))
            first = semantic.add_memory(title="Note", content="Same content", project="core")
            second = semantic.add_memory(title="Note", content="Same content", project="core")
            self.assertNotEqual(first["id"], second["id"])
            first_run = semantic.rerank("same", [{"title": "Note", "content": "Same content"}], project="core")
            second_run = semantic.rerank("same", [{"title": "Note", "content": "Same content"}], project="core")
            self.assertNotEqual(first_run["id"], second_run["id"])
            first_graph = semantic.graph_analytics(project="core")
            second_graph = semantic.graph_analytics(project="core")
            self.assertNotEqual(first_graph["id"], second_graph["id"])

    def test_collection_archive_restore_preserves_vector_records_and_project_scope(self) -> None:
        with TemporaryDirectory() as tmp:
            semantic = SemanticMemoryManager(OpenAgentStore(tmp))
            collection = semantic.create_collection("research", project="core")
            other = semantic.create_collection("research", project="other")
            vector = semantic.upsert_vector_record(collection["id"], content="Keep this vector", project="core")
            archived = semantic.set_collection_archived("research", archived=True, project="core")
            self.assertTrue(archived["archived"])
            self.assertEqual(semantic.collections(project="core"), [])
            self.assertEqual(semantic.collections(project="core", include_archived=True)[0]["id"], collection["id"])
            with self.assertRaises(KeyError):
                semantic.set_collection_archived("research", archived=False, project="other-missing")
            restored = semantic.set_collection_archived("research", archived=False, project="core")
            self.assertFalse(restored["archived"])
            self.assertEqual(restored["id"], collection["id"])
            self.assertEqual(semantic.get_vector_record(vector["id"])["content"], "Keep this vector")
            self.assertEqual(semantic.collections(project="other")[0]["id"], other["id"])

    def test_memory_vector_rerank_consolidation_and_graph_analytics(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            knowledge = KnowledgeEngine(store)
            semantic = SemanticMemoryManager(store)

            memory = semantic.add_memory(
                title="Awais prefers local models",
                content="Use local Ollama models for private OpenAgent research unless cloud is approved.",
                project="core",
                memory_type="preference",
                profile_id="local-user",
                tags=["privacy", "models"],
            )
            self.assertEqual(memory["memory_type"], "preference")
            self.assertNotIn("embedding", memory)

            results = semantic.search_memories("private local models", project="core")
            self.assertEqual(results[0]["id"], memory["id"])
            self.assertIn("semantic", results[0]["retrieval"])

            collection = semantic.create_collection("research", project="core")
            record = semantic.upsert_vector_record(
                collection["id"],
                content="OpenAgent stores private context locally.",
                project="core",
                external_id="doc-1",
            )
            self.assertEqual(record["external_id"], "doc-1")
            vector_results = semantic.vector_search(collection["id"], "private context")
            self.assertEqual(vector_results[0]["id"], record["id"])
            self.assertNotIn("vector", vector_results[0])

            reranked = semantic.rerank(
                "private local memory",
                [
                    {"id": "a", "title": "Cloud sync", "content": "Send all data to a hosted provider."},
                    {"id": "b", "title": "Local memory", "content": "Keep private memory on the local machine."},
                ],
                project="core",
            )
            self.assertEqual(reranked["results"][0]["id"], "b")

            consolidated = semantic.consolidate(project="core")
            self.assertEqual(consolidated["memory"]["memory_type"], "long_term")

            knowledge.ingest_text(
                "OpenAgent graph",
                "OpenAgent connects Awais with Local Models on 2026-06-25.",
                project="core",
            )
            graph = semantic.graph_analytics(project="core")
            self.assertIn("degree_centrality", graph["analytics"])
            self.assertGreaterEqual(graph["entities"], 1)

            status = semantic.status(project="core")
            self.assertGreaterEqual(status["semantic_memories"], 2)
            self.assertIn("locally", status["privacy_boundary"].lower())

            audit_actions = {event["action"] for event in store.list_audit_events(project="core", limit=50)}
            self.assertIn("memory.add", audit_actions)
            self.assertIn("retrieval.rerank", audit_actions)
            self.assertIn("graph.analytics", audit_actions)


if __name__ == "__main__":
    unittest.main()
