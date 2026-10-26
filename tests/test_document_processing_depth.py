from __future__ import annotations

from http import HTTPStatus
from io import BytesIO
from tempfile import TemporaryDirectory
import unittest
import zipfile

from reportlab.pdfgen.canvas import Canvas

from openagent.document_processing import DocumentProcessing, extract_office_text
from openagent.routes.document_processing_routes import DocumentProcessingRouteDependencies, ROUTES, handle_post
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

    def test_legacy_binary_office_import_is_rejected_without_creating_garbled_document(self) -> None:
        with TemporaryDirectory() as temp:
            processing = DocumentProcessing(OpenAgentStore(temp))
            for filename in ("old.doc", "old.ppt", "old.xls"):
                with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "not supported for text extraction"):
                    processing.import_office_or_text(filename=filename, content=b"\x00\x01\xffbinary office data", project="core")
            self.assertEqual(processing.documents.list_library(project="core")["items"], [])
            payload = {"project": "core", "filename": "legacy.doc", "content_base64": "AAH/AGJpbmFyeSBvZmZpY2UgZGF0YQ=="}
            deps = DocumentProcessingRouteDependencies(processing, lambda body, default: default, lambda: payload)
            route_response = handle_post("/api/documents/import", deps)
            self.assertEqual(route_response.status, HTTPStatus.BAD_REQUEST)
            self.assertIn("not supported for text extraction", route_response.payload["error"])
            self.assertEqual(processing.documents.list_library(project="core")["items"], [])

    def test_odt_and_rtf_extraction_returns_readable_text_and_rejects_invalid_rtf(self) -> None:
        odt = BytesIO()
        with zipfile.ZipFile(odt, "w") as archive:
            archive.writestr("content.xml", '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><office:body><office:text><text:h>Project brief</text:h><text:p>Local <text:span>document</text:span></text:p></office:text></office:body></office:document-content>')
        self.assertEqual(extract_office_text("brief.odt", odt.getvalue()), "Project brief\n\nLocal document")
        self.assertEqual(extract_office_text("brief.rtf", b"{\\rtf1\\ansi Hello\\par world {\\fonttbl{\\f0 ignored;}} \\u233?!}"), "Hello\nworld \u00e9!")
        with TemporaryDirectory() as temp:
            processing = DocumentProcessing(OpenAgentStore(temp))
            with self.assertRaisesRegex(ValueError, "Could not extract readable text"):
                processing.import_office_or_text(filename="broken.rtf", content=b"not an RTF document", project="core")
            self.assertEqual(processing.documents.list_library(project="core")["items"], [])

    def test_office_exports_explain_the_optional_dependency(self) -> None:
        with TemporaryDirectory() as temp:
            processing = DocumentProcessing(OpenAgentStore(temp))
            imported = processing.import_office_or_text(filename="source.txt", content=b"Office artifact", project="core")
            try:
                exported = processing.export_docx(imported["document"]["doc_id"])
            except ValueError as exc:
                self.assertRegex(str(exc), r"openagent\[office\]")
            else:
                self.assertEqual(exported["format"], "docx")


if __name__ == "__main__":
    unittest.main()
