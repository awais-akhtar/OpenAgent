from __future__ import annotations

import base64
import json
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile
from io import BytesIO

from openagent.auth import required_scope_for_path
from openagent.model_gateway import ModelResult
from openagent.phd_pipeline import run_phd_writing_pipeline
from openagent.phd_workspace import ARTIFACT_SPECS, PhDWorkspace
from openagent.routes.phd_routes import ROUTES
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class _Gateway:
    def complete(self, messages, *, provider_id="auto", model=None, allow_external=False):
        user = messages[-1]["content"]
        return ModelResult(
            provider="test-local",
            model="test-writer",
            content=(
                "# Evidence-grounded draft\n\n"
                "Awais prepared for this doctoral application through an MSc in Computer Science and work on local AI systems. "
                "The proposed position connects that experience with efficient machine learning research. "
                "[NEEDS INPUT: add one verified result from the selected project.]\n\n"
                + user[:120]
            ),
            metadata={"network": False},
        )


class _MultiGateway(_Gateway):
    def discover_models(self):
        return [
            {"available": True, "provider": "ollama", "model": "writer-local"},
            {"available": True, "provider": "llamacpp", "model": "reviewer-local"},
        ]

    def complete(self, messages, *, provider_id="auto", model=None, allow_external=False):
        result = super().complete(messages, provider_id=provider_id, model=model, allow_external=allow_external)
        paragraph = (
            "I am applying to your institution because the position matches my research direction. "
            "My academic preparation in computer science and practical work on efficient local inference gave me a "
            "clear research direction. I measured system behavior, documented reproducible results, and learned to "
            "connect engineering decisions with research questions. This evidence supports a focused interest in "
            "efficient, private artificial intelligence on consumer hardware (evidence_id=internal-test-source). "
        )
        return ModelResult(
            provider=provider_id,
            model=model or f"{provider_id}-model",
            content="# Motivation Letter\n\n" + (paragraph * 8) + "\n\nSincerely,\nAwais Akhtar",
            metadata={"network": False, "latency_ms": 2},
        )


class _Search:
    def search(self, query, *, limit=6):
        return [{"url": "https://example.edu/lab", "title": "Systems Lab", "snippet": "Efficient AI research"}]


class _Scraper:
    def scrape_url(self, url, **kwargs):
        return {
            "status": "ok",
            "url": url,
            "title": "Systems Lab",
            "description": "Official lab page",
            "text_preview": "The lab studies efficient local inference.",
            "text": "The Systems Lab studies efficient local inference, model compression, and reproducible evaluation.",
        }


class _LengthRepairGateway:
    def __init__(self) -> None:
        self.calls = 0

    def discover_models(self):
        return [
            {"available": True, "provider": "ollama", "model": "writer-local"},
            {"available": True, "provider": "llamacpp", "model": "reviewer-local"},
        ]

    def complete(self, messages, *, provider_id="auto", model=None, allow_external=False, max_output_tokens=None):
        self.calls += 1
        if self.calls == 2:
            content = "FACT CHECK position details verified. POSITION FIT strong. STRUCTURE clear. VOICE natural. " * 10
        elif self.calls == 3:
            content = "# Motivation Letter\n\n" + ("Verified preparation supports this research direction. " * 80) + "\n\nSincerely,\nApplicant"
        else:
            content = "# Motivation Letter\n\n" + ("Verified preparation supports this focused research direction. " * 110) + "\n\nSincerely,\nApplicant"
        return ModelResult(provider_id, model or "local-model", content, {"network": False})


class PhDWorkspaceTests(unittest.TestCase):
    def make_workspace(self, temp: str) -> PhDWorkspace:
        return PhDWorkspace(
            OpenAgentStore(temp),
            gateway=_Gateway(),
            scraper=_Scraper(),
            search_client=_Search(),
        )

    def test_private_profile_upload_deduplication_and_owner_scope(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = self.make_workspace(temp)
            profile = workspace.save_profile(
                {
                    "full_name": "Awais Akhtar",
                    "email": "awais@example.test",
                    "education": "MSc Computer Science",
                    "research_interests": "Efficient local AI",
                    "skills": "Python, model optimization",
                },
                project="phd",
                owner="owner-a",
            )
            self.assertEqual(profile["completeness"]["score"], 1.0)
            uploaded = workspace.upload_document(
                filename="cv.txt",
                content=b"MSc Computer Science\nLocal AI project",
                category="cv",
                project="phd",
                owner="owner-a",
            )
            duplicate = workspace.upload_document(
                filename="renamed.txt",
                content=b"MSc Computer Science\nLocal AI project",
                category="cv",
                project="phd",
                owner="owner-a",
            )
            self.assertTrue(duplicate["duplicate"])
            self.assertEqual(uploaded["id"], duplicate["id"])
            self.assertEqual(len(workspace.list_documents(project="phd", owner="owner-a")), 1)
            self.assertTrue(uploaded["metadata"]["rag"]["indexed"])
            self.assertGreater(workspace.status(project="phd", owner="owner-a")["rag"]["indexed_chunks"], 0)
            self.assertEqual(workspace.list_documents(project="phd", owner="owner-b"), [])
            with self.assertRaises(KeyError):
                workspace.get_document(uploaded["id"], owner="owner-b")
            self.assertTrue(workspace.delete_document(uploaded["id"], owner="owner-a")["deleted"])
            self.assertEqual(workspace.status(project="phd", owner="owner-a")["rag"]["indexed_chunks"], 0)

    def test_position_research_generation_revision_and_exports(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = self.make_workspace(temp)
            workspace.save_profile(
                {
                    "full_name": "Awais Akhtar",
                    "email": "awais@example.test",
                    "education": "MSc Computer Science, 2025",
                    "research_interests": "efficient local AI",
                    "skills": "Python, inference optimization",
                    "experience": "Built and evaluated a local AI runtime.",
                },
                project="phd",
            )
            writing = workspace.upload_document(
                filename="sample.md",
                content=b"I prefer short sentences. I explain what I measured and why it matters.",
                category="writing_sample",
                project="phd",
            )
            application = workspace.create_application(
                {
                    "institution": "Example University",
                    "program": "PhD in Computer Science",
                    "position_title": "Efficient AI Systems",
                    "supervisor": "Professor Ada Example",
                    "lab": "Systems Lab",
                    "deadline": "2027-01-15",
                    "description": "Research model efficiency on consumer hardware.",
                    "requirements": ["MSc", "Python"],
                },
                project="phd",
            )
            research = workspace.research_application(
                application["id"],
                query="Ada Example Systems Lab efficient AI",
                allow_external=True,
            )
            self.assertTrue(research["network_used"])
            self.assertEqual(len(research["sources"]), 1)
            generated = workspace.run_command(
                application["id"],
                "Prepare my CV, statement of purpose, and professor email.",
                document_ids=[writing["id"]],
            )
            self.assertEqual(generated["count"], 3)
            professor_email = next(item for item in generated["artifacts"] if item["artifact_type"] == "professor_email")
            self.assertNotIn("Professor Professor", professor_email["content"])
            artifact = generated["artifacts"][0]
            self.assertEqual(artifact["provider"], "test-local")
            self.assertIn("detector_note", artifact["quality"])
            self.assertGreaterEqual(artifact["pipeline"]["stage_count"], 3)
            revised = workspace.revise_artifact(
                artifact["id"],
                content=artifact["content"].replace("[NEEDS INPUT: add one verified result from the selected project.]", "The runtime reduced median response latency in a local benchmark."),
            )
            self.assertEqual(revised["revision"], artifact["revision"] + 1)
            second_revision = workspace.revise_artifact(artifact["id"], content=artifact["content"] + "\n\nHuman-reviewed note.")
            self.assertEqual(second_revision["revision"], revised["revision"] + 1)
            reviewed = workspace.review_artifact(revised["id"])
            self.assertEqual(workspace.get_artifact(revised["id"])["quality"], reviewed["quality"])
            docx = workspace.export_artifact(revised["id"], export_format="docx")
            docx_data = base64.b64decode(docx["content_base64"])
            self.assertTrue(docx_data.startswith(b"PK"))
            with zipfile.ZipFile(BytesIO(docx_data)) as docx_archive:
                self.assertIn("word/styles.xml", docx_archive.namelist())
            pdf = workspace.export_artifact(revised["id"], export_format="pdf")
            self.assertTrue(base64.b64decode(pdf["content_base64"]).startswith(b"%PDF"))
            bundle = workspace.export_application(
                application["id"], include_source_documents=True, document_ids=[writing["id"]]
            )
            with zipfile.ZipFile(BytesIO(base64.b64decode(bundle["content_base64"]))) as archive:
                names = archive.namelist()
            self.assertIn("manifest.json", names)
            self.assertIn("position.json", names)
            self.assertTrue(any(name.endswith(".docx") for name in names))
            self.assertTrue(any(name.endswith(".pdf") for name in names))
            self.assertTrue(any(name.startswith("supporting-documents/") for name in names))

    def test_position_url_rag_and_multi_model_motivation_letter_pipeline(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = PhDWorkspace(
                OpenAgentStore(temp),
                gateway=_MultiGateway(),
                scraper=_Scraper(),
                search_client=_Search(),
            )
            workspace.save_profile(
                {
                    "full_name": "Awais Akhtar",
                    "email": "awais@example.test",
                    "education": "MSc Computer Science",
                    "experience": "Built a CPU inference runtime.",
                    "research_interests": "efficient local AI",
                }
            )
            source = workspace.upload_document(
                filename="thesis.txt",
                content=b"MSc thesis: efficient local inference with reproducible CPU benchmarks.",
                category="masters_thesis",
            )
            application = workspace.create_application(
                {
                    "institution": "Example University",
                    "position_title": "Efficient AI Systems",
                    "position_url": "https://example.edu/lab",
                }
            )
            generated = workspace.run_command(
                application["id"],
                "Prepare a motivation letter",
                document_ids=[source["id"]],
                research_before_generation=True,
                allow_external_research=True,
            )
            self.assertEqual(generated["count"], 1)
            self.assertTrue(generated["research"]["network_used"])
            artifact = generated["artifacts"][0]
            self.assertEqual(artifact["artifact_type"], "motivation_letter")
            self.assertTrue(artifact["pipeline"]["finalized"])
            self.assertEqual(artifact["pipeline"]["distinct_models"], 2)
            self.assertIn("Example University", artifact["content"])
            self.assertNotIn("your institution", artifact["content"].lower())
            self.assertNotIn("evidence_id", artifact["content"].lower())
            self.assertGreaterEqual(workspace.status()["rag"]["indexed_chunks"], 2)

    def test_pipeline_repairs_a_short_final_document_to_the_required_length(self) -> None:
        gateway = _LengthRepairGateway()
        result = run_phd_writing_pipeline(
            gateway,
            [{"role": "user", "content": "Write the motivation letter."}],
            artifact_type="motivation_letter",
            evidence_text="Applicant completed verified research on efficient local inference.",
            target_words=(500, 1000),
        )
        self.assertTrue(result.trace["finalized"])
        self.assertEqual(result.trace["stage_count"], 5)
        self.assertEqual(result.trace["stages"][-1]["stage"], "proofread")
        self.assertEqual(result.trace["quality_gate"], "passed")
        self.assertGreaterEqual(len(result.content.split()), 500)
        self.assertLessEqual(len(result.content.split()), 1000)

    def test_background_generation_persists_live_multi_model_progress(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = PhDWorkspace(OpenAgentStore(temp), gateway=_MultiGateway())
            workspace.save_profile(
                {
                    "full_name": "Awais Akhtar",
                    "education": "MSc Computer Science",
                    "experience": "Built and measured a local inference runtime.",
                    "research_interests": "efficient private AI",
                }
            )
            application = workspace.create_application(
                {
                    "institution": "Example University",
                    "position_title": "Efficient AI Systems",
                    "description": "Research efficient private models on consumer hardware.",
                }
            )
            started = workspace.start_generation(
                application["id"],
                "Prepare a motivation letter",
                artifact_types=["motivation_letter"],
            )
            self.assertIn(started["status"], {"queued", "running", "completed"})
            run = started
            deadline = time.monotonic() + 10
            while run["status"] not in {"completed", "failed"} and time.monotonic() < deadline:
                time.sleep(0.05)
                run = workspace.get_generation_run(started["id"])
            self.assertEqual(run["status"], "completed", run)
            self.assertEqual(run["progress"], 100)
            self.assertEqual(len(run["result"]["artifacts"]), 1)
            stages = {event["stage"] for event in run["events"]}
            self.assertIn("evidence-review", stages)
            self.assertIn("proofread", stages)
            self.assertIn("complete", stages)
            self.assertTrue(run["result"]["artifacts"][0]["pipeline"]["finalized"])

    def test_external_research_and_model_boundaries_are_separate(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = self.make_workspace(temp)
            application = workspace.create_application({"institution": "Example University"})
            with self.assertRaises(ValueError):
                workspace.research_application(application["id"], urls=["https://example.edu/lab"], allow_external=False)
            generated = workspace.run_command(application["id"], "Write an email")
            self.assertIn("local model routing only", generated["privacy_boundary"])

    def test_route_contract_and_auth_scope_cover_phd_workspace(self) -> None:
        paths = {path for _method, path in ROUTES}
        self.assertIn("/api/phd/command", paths)
        self.assertIn("/api/phd/applications/{id}/research", paths)
        self.assertIn("/api/phd/artifacts/{id}/export", paths)
        self.assertEqual(required_scope_for_path("/api/phd/profile", "GET"), "phd:read")
        self.assertEqual(required_scope_for_path("/api/phd/profile", "POST"), "phd:write")
        self.assertGreaterEqual(len(ARTIFACT_SPECS), 9)

    def test_live_http_profile_position_command_and_export(self) -> None:
        with TemporaryDirectory() as temp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(temp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"

            def request(path: str, payload: dict | None = None, method: str = "GET") -> dict:
                body = json.dumps(payload).encode("utf-8") if payload is not None else None
                response = urlopen(
                    Request(base + path, data=body, method=method, headers={"Content-Type": "application/json"}),
                    timeout=15,
                )
                return json.loads(response.read().decode("utf-8"))

            try:
                saved = request(
                    "/api/phd/profile",
                    {"project": "phd", "profile": {"full_name": "Awais", "email": "a@example.test"}},
                    "POST",
                )
                self.assertEqual(saved["profile"]["full_name"], "Awais")
                created = request(
                    "/api/phd/applications",
                    {"project": "phd", "application": {"institution": "Example University", "program": "Computer Science PhD"}},
                    "POST",
                )["application"]
                generated = request(
                    "/api/phd/command",
                    {"project": "phd", "application_id": created["id"], "command": "Write an email", "provider": "local"},
                    "POST",
                )
                artifact_id = generated["artifacts"][0]["id"]
                exported = request(
                    f"/api/phd/artifacts/{artifact_id}/export",
                    {"format": "docx"},
                    "POST",
                )
                self.assertTrue(base64.b64decode(exported["content_base64"]).startswith(b"PK"))
                status = request("/api/phd/status?project=phd")
                self.assertEqual(status["counts"]["applications"], 1)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
