"""Real G25 document extraction must not persist synthetic placeholder text."""

from __future__ import annotations

import io
import base64
import unittest
from http import HTTPStatus
from types import SimpleNamespace
from tempfile import TemporaryDirectory

from openagent.document_processor_service import DocumentProcessorService
from openagent.knowledge_graph_service import KnowledgeGraphService
from openagent.rag_service import RAGService
from openagent.routes.g25_advanced_routes import handle_post
from openagent.storage import OpenAgentStore
from openagent.vector_embeddings_service import VectorEmbeddingsService
from tests.support.optional_backends import missing

# python-docx and reportlab are the optional "documents" group. Imported at
# module level, an absent optional package became a pytest collection error,
# which stops the entire run instead of skipping this one file.
DOCUMENT_PARSERS_MISSING = missing("docx", "reportlab")
if not DOCUMENT_PARSERS_MISSING:
    from docx import Document
    from reportlab.pdfgen.canvas import Canvas


@unittest.skipIf(DOCUMENT_PARSERS_MISSING, f"document extraction {DOCUMENT_PARSERS_MISSING}")
class G25DocumentExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DocumentProcessorService()

    def test_pdf_uses_declared_parser_and_persists_extracted_text(self) -> None:
        output = io.BytesIO()
        canvas = Canvas(output)
        canvas.drawString(72, 720, "Verified local PDF content")
        canvas.save()

        result = self.service.process_document("source.pdf", output.getvalue(), project="alpha", owner="local")
        self.assertNotIn("error", result)
        saved = self.service.get_document(result["document_id"], project="alpha", owner="local")
        self.assertIn("Verified local PDF content", saved["text"])
        self.assertIsNone(self.service.get_document(result["document_id"], project="beta", owner="local"))

    def test_docx_extracts_text_instead_of_placeholder(self) -> None:
        output = io.BytesIO()
        document = Document()
        document.add_paragraph("Verified local DOCX content")
        document.save(output)

        result = self.service.process_document("source.docx", output.getvalue(), project="alpha", owner="local")
        self.assertNotIn("error", result)
        saved = self.service.get_document(result["document_id"], project="alpha", owner="local")
        self.assertIn("Verified local DOCX content", saved["text"])

    def test_colon_in_project_or_owner_cannot_cross_document_scope(self) -> None:
        created = self.service.process_document("private.txt", b"Private owner content", project="a", owner="b:c")
        doc_id = created["document_id"]
        self.assertIsNotNone(self.service.get_document(doc_id, project="a", owner="b:c"))
        self.assertIsNone(self.service.get_document(doc_id, project="a:b", owner="c"))
        self.assertEqual(self.service.list_documents(project="a:b", owner="c"), [])

    def test_invalid_pdf_and_docx_fail_without_a_saved_document(self) -> None:
        for extension in ("pdf", "docx"):
            with self.subTest(extension=extension):
                result = self.service.process_document(f"invalid.{extension}", b"not a document", project="alpha", owner="local")
                self.assertIn("Could not extract text", result["error"])
        self.assertEqual(self.service.list_documents(project="alpha", owner="local"), [])

    def test_upload_route_rejects_invalid_payload_and_extraction(self) -> None:
        def request(filename: str, content: str):
            deps = SimpleNamespace(
                read_json=lambda: {"filename": filename, "content": content, "project": "alpha"},
                auth_scope_owner=lambda _payload: "local",
                doc_processor=self.service,
            )
            return handle_post("/api/g25/documents/upload", deps)

        malformed = request("source.pdf", "%%%")
        self.assertEqual(malformed.status, HTTPStatus.BAD_REQUEST)
        invalid_pdf = request("source.pdf", base64.b64encode(b"not a pdf").decode())
        self.assertEqual(invalid_pdf.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("Could not extract text", invalid_pdf.payload["error"])
        self.assertEqual(self.service.list_documents(project="alpha", owner="local"), [])

    def test_graph_routes_validate_properties_and_conceal_foreign_nodes(self) -> None:
        graph = KnowledgeGraphService()
        foreign = graph.create_node("Private", {}, project="alpha", owner="other")

        def request(path: str, payload: dict[str, object]):
            deps = SimpleNamespace(
                read_json=lambda: {"project": "alpha", **payload},
                auth_scope_owner=lambda _payload: "local",
                knowledge_graph=graph,
            )
            return handle_post(path, deps)

        invalid = request("/api/g25/graph/nodes", {"label": "New", "properties": ["not", "an", "object"]})
        self.assertEqual(invalid.status, HTTPStatus.BAD_REQUEST)
        missing = request("/api/g25/graph/edges", {"source_id": "not-found", "target_id": "also-missing", "relationship": "LINKS"})
        foreign_attempt = request("/api/g25/graph/edges", {"source_id": foreign["id"], "target_id": foreign["id"], "relationship": "LINKS"})
        self.assertEqual(missing.status, HTTPStatus.NOT_FOUND)
        self.assertEqual(foreign_attempt.status, HTTPStatus.NOT_FOUND)
        self.assertEqual(missing.payload, foreign_attempt.payload)

    def test_saved_document_and_index_survive_service_restart_with_scope(self) -> None:
        with TemporaryDirectory() as directory:
            store = OpenAgentStore(directory)
            documents = DocumentProcessorService(store)
            rag = RAGService(VectorEmbeddingsService(), documents, KnowledgeGraphService(), store=store)
            created = documents.process_document(
                "source.txt", b"Private renewable energy research notes.", project="alpha", owner="owner-a"
            )
            doc_id = created["document_id"]
            unindexed = documents.process_document(
                "other.txt", b"Unindexed private follow-up.", project="alpha", owner="owner-a"
            )["document_id"]
            self.assertTrue(rag.index_document(doc_id, project="alpha", owner="owner-a")["indexed"])

            # New service instances simulate a server restart over the same
            # installed data directory; no process memory is retained.
            reopened_store = OpenAgentStore(directory)
            reopened_documents = DocumentProcessorService(reopened_store)
            reopened_rag = RAGService(
                VectorEmbeddingsService(), reopened_documents, KnowledgeGraphService(), store=reopened_store
            )
            self.assertEqual(reopened_documents.get_document(doc_id, project="alpha", owner="owner-a")["text"], "Private renewable energy research notes.")
            self.assertIsNone(reopened_documents.get_document(doc_id, project="alpha", owner="owner-b"))
            self.assertIsNone(reopened_documents.get_document(doc_id, project="other", owner="owner-a"))
            self.assertEqual(len(reopened_documents.list_documents(project="alpha", owner="owner-a")), 2)
            matches = reopened_rag.search("renewable energy", project="alpha", owner="owner-a")["results"]
            self.assertTrue(any(row["doc_id"] == doc_id for row in matches), matches)
            self.assertFalse(any(row["doc_id"] == unindexed for row in matches), matches)
            self.assertEqual(reopened_rag.search("renewable", project="alpha", owner="owner-b")["results"], [])
            self.assertEqual(reopened_rag.search("renewable", project="other", owner="owner-a")["results"], [])


if __name__ == "__main__":
    unittest.main()
