from __future__ import annotations

import base64
import hashlib
import hmac
import http.client
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openagent.server import build_http_server
from openagent.storage import OpenAgentStore
from openagent.voice import synthesize_tone_wav


class ServerTests(unittest.TestCase):
    def test_event_stream_treats_connection_aborted_as_disconnect_and_closes_iterator(self) -> None:
        from openagent.routes import runtime

        class AbortedWriter:
            def write(self, _body):
                raise ConnectionAbortedError("peer aborted the response")

            def flush(self):
                pass

        class Events:
            closed = False

            def __iter__(self):
                return self

            def __next__(self):
                return {"type": "chunk", "content": "partial"}

            def close(self):
                self.closed = True

        with TemporaryDirectory() as tmp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(tmp))
            try:
                handler = server.RequestHandlerClass.__new__(server.RequestHandlerClass)
                handler.wfile = AbortedWriter()
                handler.send_response = lambda _status: None
                handler.send_header = lambda _name, _value: None
                handler.end_headers = lambda: None
                events = Events()

                # Socket disconnect errors are normal stream termination, but
                # they must still release the iterator that owns inference state.
                handler.send_event_stream(events)

                self.assertTrue(events.closed)
            finally:
                server.server_close()

    def test_inference_disconnect_watcher_starts_after_request_body_is_read(self) -> None:
        from openagent.routes import runtime
        from openagent.routes import JsonRouteResult

        events = []

        def fake_inference_post(path, dependencies):
            events.append("handler")
            payload = dependencies.read_json()
            events.append(("body", payload))
            self.assertFalse(dependencies.cancelled())
            return JsonRouteResult({"ok": True})

        with TemporaryDirectory() as tmp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(tmp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                original_read_json = server.RequestHandlerClass.read_json

                def observed_read_json(handler):
                    payload = original_read_json(handler)
                    events.append(("body-read", payload))
                    return payload

                with patch.object(runtime, "handle_inference_post", side_effect=fake_inference_post), \
                     patch.object(server.RequestHandlerClass, "read_json", new=observed_read_json), \
                     patch.object(runtime._RequestDisconnectWatcher, "start",
                                  new=lambda watcher: events.append("watcher-start")), \
                     patch.object(runtime._RequestDisconnectWatcher, "cancelled", return_value=False), \
                     patch.object(runtime._RequestDisconnectWatcher, "close", return_value=None):
                    connection = http.client.HTTPConnection(
                        "127.0.0.1", server.server_address[1], timeout=5)
                    body = json.dumps({"prompt": "hello"})
                    connection.request("POST", "/v1/completions", body=body,
                                       headers={"Content-Type": "application/json"})
                    response = connection.getresponse()
                    self.assertEqual(response.status, 200)
                    self.assertEqual(json.loads(response.read()), {"ok": True})
                    connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

        self.assertEqual(events, ["handler", ("body-read", {"prompt": "hello"}),
                                  "watcher-start", ("body", {"prompt": "hello"})])

    def test_compatibility_stream_route_writes_protocol_sse_without_generic_terminal(self) -> None:
        from openagent.routes import runtime, ServerSentEvent, StreamRouteResult

        with TemporaryDirectory() as tmp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(tmp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                result = StreamRouteResult(iter((
                    ServerSentEvent({"choices": [{"delta": {"role": "assistant"}}]}),
                    ServerSentEvent("[DONE]", raw_data=True, terminal=True),
                )), headers={"X-Queue-Wait-Ms": "12"})
                with patch.object(runtime, "handle_inference_post", return_value=result), \
                     patch.object(runtime._RequestDisconnectWatcher, "start", return_value=None), \
                     patch.object(runtime._RequestDisconnectWatcher, "cancelled", return_value=False), \
                     patch.object(runtime._RequestDisconnectWatcher, "close", return_value=None):
                    connection = http.client.HTTPConnection(
                        "127.0.0.1", server.server_address[1], timeout=5)
                    connection.request("POST", "/v1/chat/completions",
                        body=json.dumps({"model": "fixture", "messages": [{"role": "user", "content": "hi"}],
                                         "stream": True}),
                        headers={"Content-Type": "application/json"})
                    response = connection.getresponse()
                    body = response.read().decode("utf-8")
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.headers.get_content_type(), "text/event-stream")
                    self.assertEqual(response.headers.get("X-Queue-Wait-Ms"), "12")
                    self.assertIn('data: {"choices": [{"delta": {"role": "assistant"}}]}', body)
                    self.assertIn("data: [DONE]", body)
                    self.assertNotIn('"type":"done"', body)
                    connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_static_files_cannot_escape_into_web_root_sibling(self) -> None:
        with TemporaryDirectory() as tmp:
            web_root = Path(tmp) / "site"
            sibling = Path(tmp) / "site-private"
            web_root.mkdir()
            sibling.mkdir()
            (web_root / "index.html").write_text("workbench", encoding="utf-8")
            (sibling / "secret.txt").write_text("private", encoding="utf-8")
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(tmp), web_root=web_root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
                connection.request("GET", "/../site-private/secret.txt")
                response = connection.getresponse()
                self.assertEqual(response.status, 404)
                self.assertNotIn(b"private", response.read())
                connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def setUp(self) -> None:
        self._env_patch = patch.dict(
            os.environ,
            {
                "OPENAGENT_OLLAMA_URL": "http://127.0.0.1:9",
                "OPENAGENT_LLAMACPP_URL": "http://127.0.0.1:9",
                # Server contract tests must not inherit a developer machine's
                # accelerator model and turn deterministic HTTP checks into
                # hardware-dependent generation benchmarks.
                "OPENAGENT_OPENVINO_MODEL_DIR": "",
            },
        )
        self._env_patch.start()

    def tearDown(self) -> None:
        self._env_patch.stop()

    def test_static_icons_and_voice_upload_compatibility(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with urlopen(f"{base_url}/icons/icon-192.png", timeout=5) as icon:
                    self.assertEqual(icon.status, 200)
                    self.assertEqual(icon.headers.get("Content-Type"), "image/png")
                    self.assertTrue(icon.read(8).startswith(b"\x89PNG"))

                boundary = "----OpenAgentVoiceBoundary"
                audio = synthesize_tone_wav("legacy upload transcript")
                body = b"".join(
                    [
                        f"--{boundary}\r\n".encode("utf-8"),
                        b'Content-Disposition: form-data; name="project"\r\n\r\n',
                        b"core\r\n",
                        f"--{boundary}\r\n".encode("utf-8"),
                        b'Content-Disposition: form-data; name="transcript_hint"\r\n\r\n',
                        b"Legacy upload transcript.\r\n",
                        f"--{boundary}\r\n".encode("utf-8"),
                        b'Content-Disposition: form-data; name="audio"; filename="recording.wav"\r\n',
                        b"Content-Type: audio/wav\r\n\r\n",
                        audio,
                        b"\r\n",
                        f"--{boundary}--\r\n".encode("utf-8"),
                    ]
                )
                request = Request(
                    f"{base_url}/api/voice/transcribe-upload",
                    data=body,
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                    method="POST",
                )
                uploaded = json.loads(urlopen(request, timeout=30).read().decode("utf-8"))
                transcription = uploaded["transcription"]
                self.assertEqual(transcription["transcript"], "Legacy upload transcript.")
                self.assertEqual(transcription["audio_profile"]["format"], "wav")
                self.assertTrue(uploaded["session_id"])

                attachment_boundary = "----OpenAgentAttachmentBoundary"
                attachment_body = b"".join([
                    f"--{attachment_boundary}\r\n".encode("utf-8"),
                    b'Content-Disposition: form-data; name="project"\r\n\r\ncore\r\n',
                    f"--{attachment_boundary}\r\n".encode("utf-8"),
                    b'Content-Disposition: form-data; name="file"; filename="evidence.txt"\r\n',
                    b"Content-Type: text/plain\r\n\r\n",
                    b"The measured efficiency is 94 percent.\r\n",
                    f"--{attachment_boundary}--\r\n".encode("utf-8"),
                ])
                attachment_request = Request(
                    f"{base_url}/api/knowledge/upload",
                    data=attachment_body,
                    headers={"Content-Type": f"multipart/form-data; boundary={attachment_boundary}"},
                    method="POST",
                )
                attachment = json.loads(urlopen(attachment_request, timeout=5).read().decode("utf-8"))
                self.assertEqual("evidence.txt", attachment["attachment"]["name"])
                self.assertIn("94 percent", attachment["attachment"]["extracted_text"])
                self.assertTrue(attachment["attachment"]["id"])
            finally:
                server.shutdown()
                server.server_close()

    def test_mission_control_routes(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                mission = read_json(f"{base_url}/api/mission/features?project=core")
                self.assertEqual(mission["status"], "ready")
                self.assertGreaterEqual(mission["feature_count"], 50)

                mission_request = Request(
                    f"{base_url}/api/mission/run-check",
                    data=json.dumps({"project": "server-mission"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                mission_run = json.loads(urlopen(mission_request, timeout=15).read().decode("utf-8"))
                self.assertEqual(mission_run["status"], "passed", mission_run)
                self.assertEqual(mission_run["failed"], 0)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_health_and_event_api(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                health = read_json(f"{base_url}/api/health")
                self.assertEqual(health["status"], "ok")

                static_js = urlopen(f"{base_url}/app.js", timeout=5)
                self.assertIn("javascript", static_js.headers.get("Content-Type", ""))
                self.assertIn("no-store", static_js.headers.get("Cache-Control", ""))
                static_js.close()

                ui_build = read_json(f"{base_url}/api/ui/build")
                self.assertEqual(ui_build["project"], "OpenAgent")
                self.assertIn("app.js", ui_build["assets"])
                self.assertIn("web_root", ui_build)

                chat_diagnose = read_json(f"{base_url}/api/chat/diagnose?project=core")
                self.assertIn(chat_diagnose["status"], {"online", "fallback"})
                self.assertIn("provider", chat_diagnose)
                self.assertIn("model", chat_diagnose)

                module_health = read_json(f"{base_url}/api/modules/health?project=core")
                self.assertIn("modules", module_health)
                self.assertIn("chat", {module["id"] for module in module_health["modules"]})
                self.assertIn(module_health["status"], {"ok", "needs_attention"})
                self.assertEqual(
                    len(module_health["modules"]),
                    sum(module_health["counts"][name] for name in ("responding", "timeout", "error")),
                )
                self.assertTrue(all("capability_status" in module for module in module_health["modules"]))

                system_metrics = read_json(f"{base_url}/api/system/metrics")
                self.assertEqual(system_metrics["status"], "ok")
                self.assertGreaterEqual(system_metrics["cpu"]["count"], 1)
                self.assertIn("memory", system_metrics)
                self.assertIn("disk", system_metrics)

                capabilities = read_json(f"{base_url}/api/capabilities")
                self.assertEqual(capabilities["summary"]["verdict"], "implemented_local_ai_os")
                self.assertGreaterEqual(capabilities["summary"]["counts"]["implemented"], 15)

                mission = read_json(f"{base_url}/api/mission/features?project=core")
                self.assertEqual(mission["status"], "ready")
                self.assertGreaterEqual(mission["feature_count"], 50)

                request = Request(
                    f"{base_url}/api/events",
                    data=json.dumps(
                        {
                            "title": "Local note",
                            "content": "No external network required.",
                            "project": "core",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                created = json.loads(urlopen(request, timeout=5).read().decode("utf-8"))
                self.assertEqual(created["event"]["title"], "Local note")

                events = read_json(f"{base_url}/api/events?project=core&q=network")
                self.assertEqual(len(events["events"]), 1)

                chat_request = Request(
                    f"{base_url}/api/chat",
                    data=json.dumps(
                        {
                            "message": "Summarize local-first behavior.",
                            "project": "core",
                            "provider": "local",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                chat = json.loads(urlopen(chat_request, timeout=5).read().decode("utf-8"))
                self.assertEqual(chat["provider"], "local")
                self.assertEqual(len(chat["messages"]), 2)
                conversation_id = chat["conversation"]["id"]
                self.assertEqual(chat["conversation_id"], conversation_id)
                conversation_messages = read_json(
                    f"{base_url}/api/conversation/messages?conversation_id={conversation_id}"
                )
                self.assertEqual(len(conversation_messages["messages"]), 2)

                branch_request = Request(
                    f"{base_url}/api/chat/branch",
                    data=json.dumps(
                        {
                            "conversation_id": conversation_id,
                            "at_message_id": chat["messages"][0]["id"],
                            "title": "API branch",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                branch = json.loads(urlopen(branch_request, timeout=5).read().decode("utf-8"))
                self.assertEqual(branch["branch"]["conversation"]["title"], "API branch")
                chat_search = read_json(f"{base_url}/api/chat/search?project=core&q=local-first")
                self.assertGreaterEqual(len(chat_search["results"]), 1)
                chat_export = read_json(f"{base_url}/api/chat/export?conversation_id={conversation_id}&write_file=false")
                self.assertEqual(len(chat_export["export"]["messages"]), 2)
                chat_replay = read_json(f"{base_url}/api/chat/replay?conversation_id={conversation_id}")
                self.assertEqual(chat_replay["replay"]["replay"]["message_count"], 2)

                models = read_json(f"{base_url}/api/models")
                self.assertIn("llamacpp", {provider["id"] for provider in models["providers"]})
                setup_status = read_json(f"{base_url}/api/local-ai/setup/status")
                self.assertIn("ollama", setup_status)
                self.assertIn("llamacpp", setup_status)
                setup_jobs = read_json(f"{base_url}/api/local-ai/jobs?project=core")
                self.assertIn("jobs", setup_jobs)

                discovery_request = Request(
                    f"{base_url}/api/models/discover",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                discovered = json.loads(urlopen(discovery_request, timeout=10).read().decode("utf-8"))
                self.assertIn("local", {item["provider"] for item in discovered["discovered"]})

                blocked_pull_request = Request(
                    f"{base_url}/api/local-ai/ollama/pull",
                    data=json.dumps({"model": "phi4-mini", "allow_network": False}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(HTTPError) as blocked_pull:
                    urlopen(blocked_pull_request, timeout=5)
                self.assertEqual(blocked_pull.exception.code, 400)
                blocked_pull.exception.close()

                blocked_setup_request = Request(
                    f"{base_url}/api/local-ai/models/setup-chat",
                    data=json.dumps({"model": "phi4-mini", "allow_network": False}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(HTTPError) as blocked_setup:
                    urlopen(blocked_setup_request, timeout=5)
                self.assertEqual(blocked_setup.exception.code, 400)
                blocked_setup.exception.close()

                offline_vosk_request = Request(
                    f"{base_url}/api/voice/vosk/setup",
                    data=json.dumps({"project": "core", "allow_network": False}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    offline_vosk = json.loads(urlopen(offline_vosk_request, timeout=20).read().decode("utf-8"))
                except HTTPError as error:
                    self.assertEqual(error.code, 400)
                    error.close()
                else:
                    self.assertEqual(offline_vosk["engine"], "vosk")
                    self.assertFalse(offline_vosk["downloaded"])
                    self.assertTrue(offline_vosk["local_archive"])

                blocked_gguf_request = Request(
                    f"{base_url}/api/local-ai/llamacpp/register",
                    data=json.dumps({"local_path": ""}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(HTTPError) as blocked_gguf:
                    urlopen(blocked_gguf_request, timeout=5)
                self.assertEqual(blocked_gguf.exception.code, 400)
                blocked_gguf.exception.close()

                policy = read_json(f"{base_url}/api/model-policy")
                self.assertEqual(policy["routing_policy"], "local-first")
                self.assertIn("smooth_runtime", policy)

                runtime_settings_request = Request(
                    f"{base_url}/api/settings/runtime",
                    data=json.dumps(
                        {
                            "settings": {
                                "quantization_mode": "below-2bit",
                                "model_timeout_seconds": "17",
                            }
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                runtime_settings = json.loads(urlopen(runtime_settings_request, timeout=5).read().decode("utf-8"))
                self.assertEqual(runtime_settings["settings"]["quantization_mode"], "below-2bit")
                self.assertEqual(runtime_settings["settings"]["model_timeout_seconds"], 17.0)

                core_runtime = read_json(f"{base_url}/api/core/runtime?task=research")
                self.assertEqual(core_runtime["task"], "research")
                self.assertIn("profile", core_runtime)

                smooth_runtime = read_json(f"{base_url}/api/local-ai/smooth-runtime?task=chat&target_ram_gb=8")
                self.assertEqual(smooth_runtime["profile"]["tier"], "ultra-light-8gb")
                survival_runtime = read_json(f"{base_url}/api/core/runtime?task=chat&target_ram_gb=3&quantization_mode=below-2bit")
                self.assertLess(survival_runtime["profile"]["effective_bits"], 2)

                embedding_request = Request(
                    f"{base_url}/api/embeddings",
                    data=json.dumps({"texts": ["local embedding"], "provider": "local"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                embedding = json.loads(urlopen(embedding_request, timeout=5).read().decode("utf-8"))
                self.assertEqual(embedding["provider"], "local")
                self.assertEqual(embedding["dimensions"], 64)

                stream_request = Request(
                    f"{base_url}/api/chat/stream",
                    data=json.dumps(
                        {
                            "message": "Stream through HTTP.",
                            "project": "core",
                            "provider": "local",
                            "conversation_id": None,
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                stream_body = urlopen(stream_request, timeout=5).read().decode("utf-8")
                self.assertIn("data:", stream_body)
                self.assertIn('"type": "status"', stream_body)
                self.assertIn('"type": "message"', stream_body)
                self.assertIn("Offline fallback", stream_body)
                stream_events = [
                    json.loads(line.removeprefix("data: "))
                    for line in stream_body.splitlines()
                    if line.startswith("data: ")
                ]
                stream_conversation = next(item["conversation"] for item in stream_events if item.get("type") == "conversation")
                self.assertNotEqual(stream_conversation["id"], "None")
                self.assertEqual(len(stream_conversation["id"]), 32)

                workspace_request = Request(
                    f"{base_url}/api/workspace/items",
                    data=json.dumps(
                        {
                            "kind": "wiki_page",
                            "title": "API workspace",
                            "content": "Workspace wiki for OpenAgent.",
                            "project": "core",
                            "metadata": {"pinned": True},
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                workspace_item = json.loads(urlopen(workspace_request, timeout=5).read().decode("utf-8"))
                self.assertEqual(workspace_item["item"]["kind"], "wiki_page")
                workspace_items = read_json(f"{base_url}/api/workspace/items?project=core&q=wiki")
                self.assertGreaterEqual(len(workspace_items["items"]), 1)
                workspace_dashboard = read_json(f"{base_url}/api/workspace/dashboard?project=core")
                self.assertEqual(workspace_dashboard["by_kind"]["wiki_page"], 1)

                workflow_request = Request(
                    f"{base_url}/api/workflows",
                    data=json.dumps(
                        {
                            "name": "API workflow",
                            "project": "core",
                            "steps": [{"action": "search_context", "query": "local"}],
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                workflow = json.loads(urlopen(workflow_request, timeout=5).read().decode("utf-8"))
                self.assertEqual(workflow["workflow"]["name"], "API workflow")

                workflow_update_request = Request(
                    f"{base_url}/api/workflows/{workflow['workflow']['id']}",
                    data=json.dumps({"name": "Updated API workflow", "enabled": True}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="PUT",
                )
                updated_workflow = json.loads(
                    urlopen(workflow_update_request, timeout=5).read().decode("utf-8")
                )
                self.assertEqual(updated_workflow["workflow"]["name"], "Updated API workflow")
                self.assertTrue(updated_workflow["workflow"]["enabled"])

                builder = read_json(f"{base_url}/api/workflows/builder")
                self.assertIn("file_write", builder["actions"])

                visual = read_json(
                    f"{base_url}/api/workflows/visual?workflow_id={workflow['workflow']['id']}"
                )
                self.assertGreaterEqual(len(visual["nodes"]), 2)
                layout_request = Request(
                    f"{base_url}/api/workflows/layout",
                    data=json.dumps(
                        {
                            "workflow_id": workflow["workflow"]["id"],
                            "layout": {
                                "nodes": [
                                    {"id": "trigger", "x": 40, "y": 80, "order": 0},
                                    {"id": "step-1", "x": 220, "y": 80, "order": 1},
                                ]
                            },
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                layout = json.loads(urlopen(layout_request, timeout=5).read().decode("utf-8"))["layout"]
                self.assertEqual(layout["workflow_id"], workflow["workflow"]["id"])

                workflow_run_request = Request(
                    f"{base_url}/api/workflows/{workflow['workflow']['id']}/run",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                workflow_run = json.loads(urlopen(workflow_run_request, timeout=5).read().decode("utf-8"))["run"]
                self.assertEqual(workflow_run["workflow_id"], workflow["workflow"]["id"])
                self.assertEqual(workflow_run["status"], "completed")

                knowledge_request = Request(
                    f"{base_url}/api/knowledge/ingest",
                    data=json.dumps(
                        {
                            "title": "API knowledge",
                            "project": "core",
                            "content": "OpenAgent stores local documents with source attribution.",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                knowledge = json.loads(urlopen(knowledge_request, timeout=5).read().decode("utf-8"))
                self.assertEqual(knowledge["document"]["title"], "API knowledge")

                research_request = Request(
                    f"{base_url}/api/research/run",
                    data=json.dumps({"query": "local documents", "project": "core"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                research = json.loads(urlopen(research_request, timeout=5).read().decode("utf-8"))
                self.assertEqual(research["project"], "core")
                self.assertIn("report", research)

                search = read_json(f"{base_url}/api/knowledge/search?project=core&q=documents")
                self.assertGreaterEqual(len(search["results"]), 1)
                tags = read_json(f"{base_url}/api/knowledge/tags?project=core")
                self.assertGreaterEqual(len(tags["tags"]), 1)
                timeline = read_json(f"{base_url}/api/knowledge/timeline?project=core")
                self.assertGreaterEqual(len(timeline["timeline"]), 1)
                hybrid_request = Request(
                    f"{base_url}/api/knowledge/hybrid-search",
                    data=json.dumps({"query": "local documents", "project": "core"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                hybrid = json.loads(urlopen(hybrid_request, timeout=5).read().decode("utf-8"))
                self.assertGreaterEqual(len(hybrid["results"]), 1)
                synth_request = Request(
                    f"{base_url}/api/knowledge/synthesize",
                    data=json.dumps({"query": "source attribution", "project": "core"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                synthesis = json.loads(urlopen(synth_request, timeout=5).read().decode("utf-8"))
                self.assertIn("sources", synthesis)

                checked_request = Request(
                    f"{base_url}/api/agents/run-checked",
                    data=json.dumps(
                        {
                            "agent": "planning",
                            "goal": "Plan API test",
                            "project": "core",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                checked = json.loads(urlopen(checked_request, timeout=5).read().decode("utf-8"))
                self.assertEqual(checked["run"]["agent"], "planning")
                self.assertIn("verification", checked)

                goal_request = Request(
                    f"{base_url}/api/assistant/goals",
                    data=json.dumps(
                        {
                            "title": "Ship API assistant",
                            "description": "Finish assistant endpoint coverage.",
                            "project": "core",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                goal = json.loads(urlopen(goal_request, timeout=5).read().decode("utf-8"))
                task_request = Request(
                    f"{base_url}/api/assistant/tasks",
                    data=json.dumps(
                        {
                            "title": "Review assistant brief",
                            "priority": "high",
                            "project": "core",
                            "goal_id": goal["goal"]["id"],
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                task = json.loads(urlopen(task_request, timeout=5).read().decode("utf-8"))
                self.assertEqual(task["task"]["priority"], "high")
                daily = read_json(f"{base_url}/api/assistant/daily-brief?project=core")
                self.assertEqual(len(daily["active_goals"]), 1)
                crm_request = Request(
                    f"{base_url}/api/assistant/crm",
                    data=json.dumps(
                        {
                            "record_type": "lead",
                            "name": "API lead",
                            "project": "core",
                            "contact": "lead@example.com",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                crm = json.loads(urlopen(crm_request, timeout=5).read().decode("utf-8"))
                self.assertEqual(crm["record"]["record_type"], "lead")

                comm_request = Request(
                    f"{base_url}/api/communication/messages/import",
                    data=json.dumps(
                        {
                            "platform": "email",
                            "sender": "awais@example.com",
                            "subject": "Follow up",
                            "body": "Please follow up on OpenAgent.",
                            "project": "core",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                comm = json.loads(urlopen(comm_request, timeout=5).read().decode("utf-8"))
                self.assertTrue(comm["message"]["follow_up"])
                inbox = read_json(f"{base_url}/api/communication/inbox?project=core&follow_up=true")
                self.assertEqual(len(inbox["messages"]), 1)

                calendar_request = Request(
                    f"{base_url}/api/calendar/events",
                    data=json.dumps(
                        {
                            "project": "core",
                            "title": "Sidebar calendar check",
                            "starts_at": "2026-07-01T09:00:00+00:00",
                            "ends_at": "2026-07-01T10:00:00+00:00",
                            "location": "Local",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                calendar_event = json.loads(urlopen(calendar_request, timeout=5).read().decode("utf-8"))["event"]
                self.assertEqual(calendar_event["title"], "Sidebar calendar check")
                calendar_events = read_json(f"{base_url}/api/calendar/events?project=core&limit=20")
                self.assertGreaterEqual(len(calendar_events["events"]), 1)
                free_slots = read_json(
                    f"{base_url}/api/calendar/free-slots?project=core&from=2026-07-01T08:00:00%2B00:00&to=2026-07-01T12:00:00%2B00:00&duration=30"
                )
                self.assertIn("slots", free_slots)
                ical_request = Request(
                    f"{base_url}/api/calendar/ical-export",
                    data=json.dumps({"project": "core"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                exported_calendar = json.loads(urlopen(ical_request, timeout=5).read().decode("utf-8"))
                self.assertIn("BEGIN:VCALENDAR", exported_calendar["ical"])

                twin_profile = read_json(f"{base_url}/api/twin/profile?project=core")
                self.assertEqual(twin_profile["project"], "core")
                twin_goal_request = Request(
                    f"{base_url}/api/twin/goals",
                    data=json.dumps(
                        {
                            "project": "core",
                            "title": "Ship sidebar routes",
                            "description": "Make every web UI feature callable.",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                twin_goal = json.loads(urlopen(twin_goal_request, timeout=5).read().decode("utf-8"))["goal"]
                self.assertEqual(twin_goal["title"], "Ship sidebar routes")
                twin_predict_request = Request(
                    f"{base_url}/api/twin/predict",
                    data=json.dumps({"project": "core", "context": "Research the calendar workflow"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                twin_prediction = json.loads(urlopen(twin_predict_request, timeout=5).read().decode("utf-8"))
                self.assertIn("predicted_intent", twin_prediction)
                twin_suggestions = read_json(f"{base_url}/api/twin/suggestions?project=core")
                self.assertIn("suggestions", twin_suggestions)

                vault_request = Request(
                    f"{base_url}/api/vault/store",
                    data=json.dumps(
                        {
                            "project": "core",
                            "label": "Sidebar secret",
                            "secret": "local-only-value",
                            "passphrase": "correct horse local",
                            "category": "secret",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                vault_entry = json.loads(urlopen(vault_request, timeout=5).read().decode("utf-8"))["entry"]
                self.assertNotIn("secret", vault_entry)
                vault_entries = read_json(f"{base_url}/api/vault/entries?project=core")
                self.assertGreaterEqual(len(vault_entries["entries"]), 1)
                vault_retrieve_request = Request(
                    f"{base_url}/api/vault/retrieve",
                    data=json.dumps(
                        {
                            "project": "core",
                            "entry_id": vault_entry["id"],
                            "passphrase": "correct horse local",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                vault_secret = json.loads(urlopen(vault_retrieve_request, timeout=5).read().decode("utf-8"))["secret"]
                self.assertEqual(vault_secret["secret"], "local-only-value")

                scraper_schedule_request = Request(
                    f"{base_url}/api/scraper/schedule",
                    data=json.dumps(
                        {
                            "project": "core",
                            "url": "https://example.com/sidebar-check",
                            "interval_hours": 12,
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                scraper_job = json.loads(urlopen(scraper_schedule_request, timeout=5).read().decode("utf-8"))["job"]
                self.assertEqual(scraper_job["interval_hours"], 12)
                scraper_jobs = read_json(f"{base_url}/api/scraper/jobs?project=core")
                self.assertGreaterEqual(len(scraper_jobs["jobs"]), 1)

                finetune_dataset_request = Request(
                    f"{base_url}/api/finetune/datasets",
                    data=json.dumps({"project": "core", "name": "Sidebar dataset"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                finetune_dataset = json.loads(urlopen(finetune_dataset_request, timeout=5).read().decode("utf-8"))["dataset"]
                finetune_example_request = Request(
                    f"{base_url}/api/finetune/examples",
                    data=json.dumps(
                        {
                            "project": "core",
                            "dataset_id": finetune_dataset["id"],
                            "user_input": "What is OpenAgent?",
                            "assistant_output": "OpenAgent is a local AI operating system.",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                finetune_example = json.loads(urlopen(finetune_example_request, timeout=5).read().decode("utf-8"))["example"]
                self.assertEqual(finetune_example["dataset_id"], finetune_dataset["id"])
                finetune_job_request = Request(
                    f"{base_url}/api/finetune/jobs",
                    data=json.dumps(
                        {
                            "project": "core",
                            "dataset_id": finetune_dataset["id"],
                            "base_model": "openagent-local",
                            "template_id": "lora-fast",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                finetune_job = json.loads(urlopen(finetune_job_request, timeout=5).read().decode("utf-8"))["job"]
                self.assertEqual(finetune_job["status"], "queued")
                adapter_request = Request(
                    f"{base_url}/api/finetune/adapters",
                    data=json.dumps(
                        {
                            "project": "core",
                            "name": "Sidebar adapter",
                            "base_model": "openagent-local",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                adapter = json.loads(urlopen(adapter_request, timeout=5).read().decode("utf-8"))["adapter"]
                activate_request = Request(
                    f"{base_url}/api/finetune/adapters/activate",
                    data=json.dumps({"project": "core", "adapter_id": adapter["id"]}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                activated_adapter = json.loads(urlopen(activate_request, timeout=5).read().decode("utf-8"))["adapter"]
                self.assertTrue(activated_adapter["active"])

                semantic_request = Request(
                    f"{base_url}/api/semantic/memories",
                    data=json.dumps(
                        {
                            "title": "API semantic memory",
                            "content": "Private semantic memories stay local.",
                            "project": "core",
                            "memory_type": "semantic",
                            "tags": ["privacy"],
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                semantic_memory = json.loads(urlopen(semantic_request, timeout=5).read().decode("utf-8"))["memory"]
                self.assertEqual(semantic_memory["title"], "API semantic memory")
                semantic_search = read_json(f"{base_url}/api/semantic/memories?project=core&q=private")
                self.assertGreaterEqual(len(semantic_search["memories"]), 1)

                collection_request = Request(
                    f"{base_url}/api/semantic/collections",
                    data=json.dumps({"name": "api-vectors", "project": "core"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                collection = json.loads(urlopen(collection_request, timeout=5).read().decode("utf-8"))["collection"]
                vector_request = Request(
                    f"{base_url}/api/semantic/vectors",
                    data=json.dumps(
                        {
                            "collection_id": collection["id"],
                            "content": "Local vector search record",
                            "project": "core",
                            "external_id": "api-record",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                vector_record = json.loads(urlopen(vector_request, timeout=5).read().decode("utf-8"))["record"]
                self.assertEqual(vector_record["external_id"], "api-record")
                vector_search_request = Request(
                    f"{base_url}/api/semantic/vector-search",
                    data=json.dumps({"collection_id": collection["id"], "query": "local search"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                vector_search = json.loads(urlopen(vector_search_request, timeout=5).read().decode("utf-8"))
                self.assertGreaterEqual(len(vector_search["results"]), 1)

                media_request = Request(
                    f"{base_url}/api/media/ingest",
                    data=json.dumps(
                        {
                            "filename": "api.pdf",
                            "title": "API PDF",
                            "project": "core",
                            "data_base64": base64.b64encode(
                                b"%PDF-1.4\nBT (API PDF local text) Tj ET\n%%EOF"
                            ).decode("ascii"),
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                media_asset = json.loads(urlopen(media_request, timeout=5).read().decode("utf-8"))["asset"]
                self.assertIn("API PDF local text", media_asset["analysis"]["extracted_text"])
                media_status = read_json(f"{base_url}/api/media/status?project=core")
                self.assertEqual(media_status["assets_by_type"]["document"], 1)

                sync_request = Request(
                    f"{base_url}/api/communication/sync",
                    data=json.dumps(
                        {
                            "platform": "email",
                            "project": "core",
                            "payload": {
                                "messages": [
                                    {
                                        "sender": "sync@example.com",
                                        "subject": "Synced",
                                        "body": "Could you follow up on synced mail?",
                                    }
                                ]
                            },
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                sync_run = json.loads(urlopen(sync_request, timeout=5).read().decode("utf-8"))["run"]
                self.assertEqual(sync_run["imported_messages"], 1)
                sync_runs = read_json(f"{base_url}/api/communication/sync-runs?project=core")
                self.assertGreaterEqual(len(sync_runs["runs"]), 1)

                voice_session_request = Request(
                    f"{base_url}/api/voice/sessions",
                    data=json.dumps(
                        {
                            "project": "core",
                            "mode": "push_to_talk",
                            "language": "en",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                voice_session = json.loads(urlopen(voice_session_request, timeout=5).read().decode("utf-8"))["session"]
                clone_request = Request(
                    f"{base_url}/api/voice/clone",
                    data=json.dumps(
                        {
                            "project": "core",
                            "name": "Local Clone",
                            "audio_base64": base64.b64encode(synthesize_tone_wav("clone sample")).decode("ascii"),
                            "filename": "voiceprint.wav",
                            "consent": "Owner submitted this sample.",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                clone = json.loads(urlopen(clone_request, timeout=5).read().decode("utf-8"))["clone"]
                self.assertTrue(clone["persona"]["voiceprint_asset_id"])
                transcript_request = Request(
                    f"{base_url}/api/voice/transcript",
                    data=json.dumps(
                        {
                            "session_id": voice_session["id"],
                            "text": "Remember this local transcript.",
                            "speaker": "awais",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                transcript = json.loads(urlopen(transcript_request, timeout=5).read().decode("utf-8"))["turn"]
                self.assertEqual(transcript["speaker"], "awais")
                stt_request = Request(
                    f"{base_url}/api/voice/stt",
                    data=json.dumps(
                        {
                            "session_id": voice_session["id"],
                            "audio_base64": base64.b64encode(synthesize_tone_wav("local speech")).decode("ascii"),
                            "filename": "speech.wav",
                            "speaker": "awais",
                            "transcript_hint": "Local speech was transcribed.",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                stt = json.loads(urlopen(stt_request, timeout=5).read().decode("utf-8"))["transcription"]
                self.assertEqual(stt["transcript"], "Local speech was transcribed.")
                self.assertEqual(stt["audio_profile"]["format"], "wav")
                tts_request = Request(
                    f"{base_url}/api/voice/tts",
                    data=json.dumps(
                        {
                            "session_id": voice_session["id"],
                            "text": "Local audio generated.",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                tts = json.loads(urlopen(tts_request, timeout=5).read().decode("utf-8"))["script"]
                self.assertEqual(tts["audio_status"], "generated_wav")
                self.assertTrue(tts["audio_asset_id"])
                voice_status = read_json(f"{base_url}/api/voice/status?project=core")
                self.assertEqual(voice_status["sessions"], 1)

                production_policy_request = Request(
                    f"{base_url}/api/production/policies",
                    data=json.dumps(
                        {
                            "project": "core",
                            "scope": "external.model",
                            "effect": "deny",
                            "conditions": {"unless": "allow_external"},
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                production_policy = json.loads(urlopen(production_policy_request, timeout=5).read().decode("utf-8"))
                self.assertEqual(production_policy["rule"]["effect"], "deny")
                production_eval_request = Request(
                    f"{base_url}/api/production/policies/evaluate",
                    data=json.dumps({"project": "core", "scope": "external.model"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                production_eval = json.loads(urlopen(production_eval_request, timeout=5).read().decode("utf-8"))
                self.assertFalse(production_eval["allowed"])

                production_index_request = Request(
                    f"{base_url}/api/production/vector/indexes",
                    data=json.dumps({"project": "core", "name": "api-production"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                production_index = json.loads(urlopen(production_index_request, timeout=5).read().decode("utf-8"))["index"]
                production_vector_request = Request(
                    f"{base_url}/api/production/vector/upsert",
                    data=json.dumps(
                        {
                            "index_id": production_index["id"],
                            "external_id": "api-prod-vector",
                            "content": "Production vector database endpoint.",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                production_vector = json.loads(urlopen(production_vector_request, timeout=5).read().decode("utf-8"))
                self.assertEqual(production_vector["record"]["external_id"], "api-prod-vector")

                production_crypto_request = Request(
                    f"{base_url}/api/production/crypto/encrypt",
                    data=json.dumps(
                        {
                            "project": "core",
                            "label": "api-envelope",
                            "plaintext": "private",
                            "passphrase": "passphrase",
                            "recipients": ["device"],
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                production_envelope = json.loads(urlopen(production_crypto_request, timeout=5).read().decode("utf-8"))["envelope"]
                production_status = read_json(f"{base_url}/api/production/status?project=core")
                self.assertGreaterEqual(production_status["crypto_envelopes"], 1)
                self.assertEqual(production_envelope["label"], "api-envelope")

                model_request = Request(
                    f"{base_url}/api/local-ai/models/register",
                    data=json.dumps(
                        {
                            "name": "API Local Model",
                            "provider": "local",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                model = json.loads(urlopen(model_request, timeout=5).read().decode("utf-8"))
                self.assertEqual(model["model"]["name"], "API Local Model")
                hardware = read_json(f"{base_url}/api/local-ai/hardware")
                self.assertGreaterEqual(hardware["cpu_count"], 1)
                auto_quant_request = Request(
                    f"{base_url}/api/local-ai/models/auto-quantize",
                    data=json.dumps(
                        {
                            "model_id": model["model"]["id"],
                            "task": "chat",
                            "target_ram_gb": 8,
                            "quantization_mode": "maximum-compression",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                auto_quant = json.loads(urlopen(auto_quant_request, timeout=5).read().decode("utf-8"))
                self.assertEqual(auto_quant["profile"]["tier"], "ultra-light-8gb")
                self.assertIn("adaptive_plan", auto_quant)

                delete_model_request = Request(
                    f"{base_url}/api/local-ai/models/delete",
                    data=json.dumps({"model_id": model["model"]["id"]}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                deleted_model = json.loads(urlopen(delete_model_request, timeout=5).read().decode("utf-8"))
                self.assertTrue(deleted_model["deleted"])

                user_request = Request(
                    f"{base_url}/api/enterprise/users",
                    data=json.dumps(
                        {
                            "email": "owner@example.com",
                            "name": "Owner",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                user = json.loads(urlopen(user_request, timeout=5).read().decode("utf-8"))["user"]

                workspace_request = Request(
                    f"{base_url}/api/enterprise/workspaces",
                    data=json.dumps(
                        {
                            "name": "API workspace",
                            "owner_user_id": user["id"],
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                workspace = json.loads(urlopen(workspace_request, timeout=5).read().decode("utf-8"))["workspace"]

                access_request = Request(
                    f"{base_url}/api/enterprise/access/check",
                    data=json.dumps(
                        {
                            "user_id": user["id"],
                            "workspace_id": workspace["id"],
                            "permission": "workspace.read",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                access = json.loads(urlopen(access_request, timeout=5).read().decode("utf-8"))
                self.assertTrue(access["allowed"])

                sso_request = Request(
                    f"{base_url}/api/enterprise/sso",
                    data=json.dumps(
                        {
                            "provider": "oidc",
                            "issuer": "https://idp.example",
                            "client_id": "openagent",
                            "metadata": {"client_secret": "local-secret"},
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                sso = json.loads(urlopen(sso_request, timeout=5).read().decode("utf-8"))
                self.assertEqual(sso["config"]["provider"], "oidc")
                self.assertEqual(sso["config"]["metadata"]["client_secret"], "***")

                sso_token = make_hs256_jwt(
                    {
                        "iss": "https://idp.example",
                        "aud": "openagent",
                        "sub": "owner@example.com",
                        "exp": int(time.time()) + 3600,
                    },
                    "local-secret",
                )
                sso_validate_request = Request(
                    f"{base_url}/api/enterprise/sso/validate",
                    data=json.dumps({"provider": "oidc", "token": sso_token}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                sso_validation = json.loads(urlopen(sso_validate_request, timeout=5).read().decode("utf-8"))
                self.assertTrue(sso_validation["valid"])

                dr_request = Request(
                    f"{base_url}/api/enterprise/disaster-recovery",
                    data=json.dumps(
                        {
                            "name": "API recovery",
                            "rpo_minutes": 60,
                            "rto_minutes": 240,
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                dr = json.loads(urlopen(dr_request, timeout=5).read().decode("utf-8"))
                self.assertEqual(dr["plan"]["plan"]["rpo_minutes"], 60)

                enterprise_status = read_json(f"{base_url}/api/enterprise/status")
                self.assertEqual(enterprise_status["users"], 1)
                self.assertEqual(enterprise_status["workspaces"], 1)
                self.assertEqual(enterprise_status["disaster_recovery_plans"], 1)

                graphql_request = Request(
                    f"{base_url}/api/graphql",
                    data=json.dumps({"query": "{ health developerStatus }"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                graphql = json.loads(urlopen(graphql_request, timeout=5).read().decode("utf-8"))
                self.assertIn("health", graphql["data"])
                self.assertIn("developerStatus", graphql["data"])

                key_request = Request(
                    f"{base_url}/api/developer/api-keys",
                    data=json.dumps({"label": "API SDK", "scopes": ["api.read"]}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                api_key = json.loads(urlopen(key_request, timeout=5).read().decode("utf-8"))["api_key"]
                verify_request = Request(
                    f"{base_url}/api/developer/api-keys/verify",
                    data=json.dumps({"key": api_key["key"], "scope": "api.read"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                self.assertTrue(json.loads(urlopen(verify_request, timeout=5).read().decode("utf-8"))["valid"])

                webhook_request = Request(
                    f"{base_url}/api/developer/webhooks",
                    data=json.dumps({"event": "api.test", "target_url": "log://local"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                webhook = json.loads(urlopen(webhook_request, timeout=5).read().decode("utf-8"))["webhook"]
                dispatch_request = Request(
                    f"{base_url}/api/developer/webhooks/dispatch",
                    data=json.dumps({"event": "api.test", "payload": {"ok": True}}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                dispatch = json.loads(urlopen(dispatch_request, timeout=5).read().decode("utf-8"))
                self.assertEqual(dispatch["deliveries"][0]["webhook_id"], webhook["id"])

                plugin_request = Request(
                    f"{base_url}/api/developer/plugins",
                    data=json.dumps(
                        {
                            "id": "api-plugin",
                            "name": "API Plugin",
                            "version": "0.1.0",
                            "permissions": ["context.read"],
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                plugin = json.loads(urlopen(plugin_request, timeout=5).read().decode("utf-8"))["plugin"]
                self.assertEqual(plugin["id"], "api-plugin")

                tool_request = Request(
                    f"{base_url}/api/developer/tools",
                    data=json.dumps(
                        {
                            "tool_id": "api.echo",
                            "name": "API Echo",
                            "handler": "echo",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                tool = json.loads(urlopen(tool_request, timeout=5).read().decode("utf-8"))["tool"]
                invoke_request = Request(
                    f"{base_url}/api/developer/tools/invoke",
                    data=json.dumps({"tool_id": tool["id"], "input": {"hello": "world"}}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                invocation = json.loads(urlopen(invoke_request, timeout=5).read().decode("utf-8"))["invocation"]
                self.assertEqual(invocation["output"]["echo"]["hello"], "world")

                mcp_request = Request(
                    f"{base_url}/api/developer/mcp/servers",
                    data=json.dumps({"name": "API MCP", "command": "openagent-mcp"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                mcp_server = json.loads(urlopen(mcp_request, timeout=5).read().decode("utf-8"))["server"]
                self.assertEqual(mcp_server["name"], "API MCP")

                extension_request = Request(
                    f"{base_url}/api/developer/extensions",
                    data=json.dumps(
                        {
                            "package_type": "plugin",
                            "name": "api-plugin",
                            "version": "0.1.0",
                            "source": "local",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                package = json.loads(urlopen(extension_request, timeout=5).read().decode("utf-8"))["package"]
                self.assertEqual(package["package_type"], "plugin")

                sdk_request = Request(
                    f"{base_url}/api/developer/sdk",
                    data=json.dumps({"language": "python"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                sdk = json.loads(urlopen(sdk_request, timeout=5).read().decode("utf-8"))["artifact"]
                self.assertIn("openagent_client.py", sdk["files"])

                developer_status = read_json(f"{base_url}/api/developer/status")
                self.assertEqual(developer_status["api_keys"], 1)
                self.assertGreaterEqual(developer_status["plugins"], 1)
                self.assertGreaterEqual(developer_status["tools"], 4)

                secret_request = Request(
                    f"{base_url}/api/security/secrets",
                    data=json.dumps(
                        {
                            "name": "api-token",
                            "value": "local-secret",
                            "passphrase": "passphrase",
                            "description": "API smoke secret",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                secret = json.loads(urlopen(secret_request, timeout=5).read().decode("utf-8"))["secret"]
                self.assertTrue(secret["encrypted"])
                reveal_request = Request(
                    f"{base_url}/api/security/secrets/reveal",
                    data=json.dumps({"secret_id": secret["id"], "passphrase": "passphrase"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                revealed = json.loads(urlopen(reveal_request, timeout=5).read().decode("utf-8"))["secret"]
                self.assertEqual(revealed["value"], "local-secret")

                retention_request = Request(
                    f"{base_url}/api/security/retention",
                    data=json.dumps({"resource_type": "context_events", "retention_days": 365}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                retention = json.loads(urlopen(retention_request, timeout=5).read().decode("utf-8"))["policy"]
                retention_apply_request = Request(
                    f"{base_url}/api/security/retention/apply",
                    data=json.dumps({"policy_id": retention["id"], "dry_run": True}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                retention_result = json.loads(urlopen(retention_apply_request, timeout=5).read().decode("utf-8"))
                self.assertTrue(retention_result["dry_run"])

                share_request = Request(
                    f"{base_url}/api/security/shares",
                    data=json.dumps(
                        {
                            "resource_type": "workspace",
                            "resource_id": workspace["id"],
                            "grantee": "auditor@example.com",
                            "scope": "read",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                share = json.loads(urlopen(share_request, timeout=5).read().decode("utf-8"))["grant"]
                self.assertEqual(share["status"], "active")

                egress_request = Request(
                    f"{base_url}/api/security/egress/request",
                    data=json.dumps(
                        {
                            "requester": "api-test",
                            "purpose": "verify approval flow",
                            "destination": "https://api.example.com",
                            "resource_type": "workspace",
                            "resource_id": workspace["id"],
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                egress = json.loads(urlopen(egress_request, timeout=5).read().decode("utf-8"))["request"]
                egress_resolve_request = Request(
                    f"{base_url}/api/security/egress/resolve",
                    data=json.dumps(
                        {
                            "request_id": egress["id"],
                            "approved": False,
                            "approved_by": "owner@example.com",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                egress_resolved = json.loads(urlopen(egress_resolve_request, timeout=5).read().decode("utf-8"))["request"]
                self.assertEqual(egress_resolved["status"], "denied")

                seal_request = Request(
                    f"{base_url}/api/security/audit-chain/seal",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                sealed = json.loads(urlopen(seal_request, timeout=5).read().decode("utf-8"))
                self.assertGreaterEqual(sealed["sealed"], 1)
                self.assertTrue(read_json(f"{base_url}/api/security/audit-chain/verify")["ok"])

                profile_request = Request(
                    f"{base_url}/api/security/sandbox-profiles",
                    data=json.dumps(
                        {
                            "name": "api-read-only",
                            "permissions": ["context.read"],
                            "network_allowed": False,
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                profile = json.loads(urlopen(profile_request, timeout=5).read().decode("utf-8"))["profile"]
                sandbox_request = Request(
                    f"{base_url}/api/security/sandbox/evaluate",
                    data=json.dumps(
                        {
                            "profile_id": profile["id"],
                            "permissions": ["context.read"],
                            "network": True,
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                sandbox = json.loads(urlopen(sandbox_request, timeout=5).read().decode("utf-8"))
                self.assertFalse(sandbox["allowed"])

                privacy_report_request = Request(
                    f"{base_url}/api/security/privacy-report",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                privacy_report = json.loads(urlopen(privacy_report_request, timeout=5).read().decode("utf-8"))["report"]
                self.assertIn("data_inventory", privacy_report["report"])

                security_status = read_json(f"{base_url}/api/security/status")
                self.assertEqual(security_status["secrets"], 1)
                self.assertGreaterEqual(security_status["sealed_audit_events"], 1)

                hard_audit_request = Request(
                    f"{base_url}/api/capabilities/hard-audit",
                    data=json.dumps({"project": "server-hard-audit"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                hard_audit = json.loads(urlopen(hard_audit_request, timeout=15).read().decode("utf-8"))
                self.assertTrue(hard_audit["ok"], hard_audit)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_workflow_create_validation_returns_json_error(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                request = Request(
                    f"{base_url}/api/workflows",
                    data=json.dumps(
                        {
                            "name": "Bad workflow",
                            "project": "core",
                            "steps": [{"action": "note", "text": "not supported"}],
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(HTTPError) as ctx:
                    urlopen(request, timeout=5)
                self.assertEqual(ctx.exception.code, 400)
                payload = json.loads(ctx.exception.read().decode("utf-8"))
                ctx.exception.close()
                self.assertEqual(payload["error"]["message"], "Unsupported workflow action 'note'.")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


def read_json(url: str) -> dict[str, object]:
    return json.loads(urlopen(url, timeout=5).read().decode("utf-8"))

def make_hs256_jwt(claims: dict[str, object], secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = ".".join([base64url_json(header), base64url_json(claims)])
    signature = hmac.new(secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{base64url(signature)}"


def base64url_json(payload: dict[str, object]) -> str:
    return base64url(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def base64url(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


if __name__ == "__main__":
    unittest.main()
