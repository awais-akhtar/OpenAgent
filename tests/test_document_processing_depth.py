from __future__ import annotations

from io import BytesIO
from tempfile import TemporaryDirectory
import unittest

from reportlab.pdfgen.canvas import Canvas

from openagent.document_processing import DocumentProcessing
from openagent.routes.document_processing_routes import ROUTES
from openagent.storage import OpenAgentStore


def sample_pdf() -> bytes:
    output = BytesIO()
    canvas = Canvas(output, invariant=1)
    canvas.drawString(72, 720, "OpenAgent private document")
    canvas.save()
    return output.getvalue()


class DocumentProcessingDepthTests(unittest.TestCase):
    def test_pdf_render_annotation_redaction_export_and_compare(self) -> None:
        with TemporaryDirectory() as temp:
            processing = DocumentProcessing(OpenAgentStore(temp))
            imported = processing.import_pdf(filename="source.pdf", content=sample_pdf(), project="core")
            doc_id = imported["document"]["doc_id"]
            page = processing.render_pdf_page(doc_id)
            self.assertEqual(page["page_count"], 1)
            processing.add_ocr_overlay(doc_id, page=0, text="OpenAgent")
            self.assertEqual(processing.annotations(doc_id)[0]["kind"], "ocr-overlay")
            redacted = processing.redact_text(doc_id, ["private"])
            self.assertEqual(redacted["replacements"], 1)
            exported = processing.export_pdf(doc_id, signing_key="local-test-key")
            self.assertEqual(len(exported["signature"]), 64)
            second = processing.import_office_or_text(filename="second.txt", content=b"OpenAgent public document", project="core")
            compared = processing.compare(doc_id, second["document"]["doc_id"])
            self.assertLess(compared["similarity"], 1.0)

    def test_routes_cover_deep_document_pipeline(self) -> None:
        paths = {path for _, path in ROUTES}
        self.assertIn("/api/document/{id}/render-page", paths)
        self.assertIn("/api/document/{id}/redact", paths)
        self.assertIn("/api/document/{id}/export-docx", paths)
        self.assertIn("/api/document/{id}/export-pptx", paths)
        self.assertIn("/api/document/{id}/export-xlsx", paths)
        self.assertIn("/api/documents/compare", paths)

    def test_office_exports_explain_the_optional_dependency(self) -> None:
        with TemporaryDirectory() as temp:
            processing = DocumentProcessing(OpenAgentStore(temp))
            imported = processing.import_office_or_text(filename="source.txt", content=b"Office artifact", project="core")
            with self.assertRaisesRegex(ValueError, r"openagent\[office\]"):
                processing.export_docx(imported["document"]["doc_id"])


if __name__ == "__main__":
    unittest.main()
