from __future__ import annotations

from tempfile import TemporaryDirectory
import base64
import hashlib
import hmac
import json
import time
import unittest

from openagent.media import MediaManager
from openagent.models import AuditEvent
from openagent.production_runtime import ProductionRuntime
from openagent.security_privacy import SecurityPrivacyManager
from openagent.semantic_memory import SemanticMemoryManager
from openagent.storage import OpenAgentStore


class ProductionRuntimeTests(unittest.TestCase):
    def test_policy_queue_audit_vector_semantic_media_and_rerank(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            runtime = ProductionRuntime(store)

            deny = runtime.upsert_policy_rule(
                scope="external.model",
                effect="deny",
                conditions={"unless": "allow_external"},
                project="core",
            )
            self.assertEqual(deny["effect"], "deny")
            self.assertFalse(runtime.evaluate_policy(scope="external.model", project="core")["allowed"])
            self.assertTrue(
                runtime.evaluate_policy(
                    scope="external.model",
                    project="core",
                    context={"allow_external": True},
                )["allowed"]
            )

            job = runtime.enqueue_job(kind="noop", payload={"ok": True}, project="core")
            run = runtime.run_next_job(project="core")
            self.assertTrue(run["ran"])
            self.assertEqual(run["job"]["id"], job["id"])
            self.assertEqual(run["job"]["status"], "completed")

            store.add_audit_event(
                AuditEvent(
                    actor="test",
                    action="test.audit",
                    resource="resource:1",
                    project="core",
                )
            )
            anchor = runtime.anchor_audit_log(project="core")
            self.assertGreaterEqual(anchor["event_count"], 1)
            self.assertEqual(len(anchor["anchor_hash"]), 64)

            index = runtime.create_vector_index(name="production", project="core")
            vector = runtime.upsert_vector(
                index["id"],
                external_id="doc-1",
                content="Production vector database stores local private memory.",
            )
            self.assertEqual(vector["external_id"], "doc-1")
            results = runtime.vector_search(index["id"], query="private memory")
            self.assertEqual(results[0]["external_id"], "doc-1")
            self.assertNotIn("vector", results[0])

            semantic = SemanticMemoryManager(store)
            semantic.add_memory(
                title="Local preference",
                content="Awais prefers private local models.",
                project="core",
                memory_type="preference",
                tags=["privacy"],
            )
            science = runtime.semantic_memory_science(project="core")
            self.assertEqual(science["memory_count"], 1)
            self.assertGreaterEqual(science["average_confidence"], 0)

            reranked = runtime.rerank_sources(
                query="private local models",
                project="core",
                candidates=[
                    {"id": "cloud", "content": "Cloud hosted provider", "authority": 0.2},
                    {"id": "local", "content": "Private local models", "authority": 0.8},
                ],
            )
            self.assertEqual(reranked["results"][0]["id"], "local")

            media = MediaManager(store)
            asset = media.ingest_bytes(
                b"local image bytes",
                filename="photo.jpg",
                project="core",
                metadata={"ocr_text": "OpenAgent private screenshot"},
            )
            media_result = runtime.analyze_media_production(asset["id"])
            self.assertEqual(media_result["findings"]["ocr_text"], "OpenAgent private screenshot")

            actions = {event["action"] for event in store.list_audit_events(project="core", limit=100)}
            self.assertIn("production.policy.evaluate", actions)
            self.assertIn("production.vector.upsert", actions)
            self.assertIn("production.media.analyze", actions)

    def test_connector_oidc_crypto_automation_packaging_marketplace_ai_and_sync(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            runtime = ProductionRuntime(store)

            adapter = runtime.register_connector_adapter(
                platform="email",
                config={
                    "fixture_payload": {
                        "messages": [
                            {
                                "sender": "ops@example.com",
                                "subject": "Production sync",
                                "body": "Please follow up on the production sync.",
                            }
                        ]
                    }
                },
            )
            poll = runtime.poll_connector(adapter["id"], project="core")
            self.assertEqual(poll["sync_run"]["imported_messages"], 1)

            secret = b"jwks-secret"
            token = make_hs256_jwt(
                {
                    "iss": "https://idp.example",
                    "aud": "openagent",
                    "sub": "owner@example.com",
                    "exp": int(time.time()) + 600,
                },
                secret,
                kid="local",
            )
            jwks = {"keys": [{"kty": "oct", "kid": "local", "alg": "HS256", "k": b64url(secret)}]}
            validation = runtime.validate_oidc_jwks(
                token=token,
                jwks=jwks,
                issuer="https://idp.example",
                audience="openagent",
                project="core",
            )
            self.assertTrue(validation["valid"])

            envelope = runtime.encrypt_envelope(
                label="memory-vault",
                plaintext="private memory",
                passphrase="passphrase",
                recipients=["device-a"],
                project="core",
            )
            revealed = runtime.decrypt_envelope(envelope["id"], passphrase="passphrase")
            self.assertEqual(revealed["plaintext"], "private memory")

            key = runtime.register_hardware_key(
                name="security-key",
                public_key="local-public-key",
                project="core",
                enclave_hint="local-secure-enclave",
            )
            self.assertEqual(key["key_type"], "webauthn")

            desktop = runtime.run_desktop_automation(
                actions=[
                    {"action": "write_file", "path": "notes/status.txt", "content": "desktop automation wrote this"},
                    {"action": "read_file", "path": "notes/status.txt"},
                    {"action": "list_dir", "path": "notes"},
                ],
                project="core",
            )
            self.assertEqual(desktop["runtime"], "desktop")
            self.assertEqual(desktop["result"]["actions"][1]["content"], "desktop automation wrote this")
            self.assertIn("status.txt", desktop["result"]["actions"][2]["items"])
            browser = runtime.run_browser_automation(
                actions=[
                    {"action": "open", "url": "http://127.0.0.1"},
                    {"action": "fill", "selector": "email", "value": "owner@example.com"},
                    {"action": "assert_text", "text": "OpenAgent"},
                    {"action": "extract_title"},
                ],
                project="core",
                html='<title>OpenAgent</title><form><input name="email" value=""></form>',
            )
            self.assertEqual(browser["result"]["state"]["title"], "OpenAgent")
            self.assertEqual(browser["result"]["state"]["fields"]["email"], "owner@example.com")
            self.assertIn('value="owner@example.com"', browser["result"]["filled_html"])

            mobile_package = runtime.package_app(app_type="mobile", platform="pwa", project="core")
            desktop_package = runtime.package_app(app_type="desktop", platform="windows", project="core")
            self.assertTrue(mobile_package["package_path"])
            self.assertTrue((runtime.package_root / mobile_package["id"] / "manifest.webmanifest").exists())
            desktop_script = runtime.package_root / desktop_package["id"] / "openagent-desktop.ps1"
            desktop_cmd = runtime.package_root / desktop_package["id"] / "OpenAgent.cmd"
            self.assertTrue(desktop_script.exists())
            self.assertTrue(desktop_cmd.exists())
            self.assertEqual(desktop_package["manifest"]["app_type"], "desktop")
            self.assertEqual(desktop_package["manifest"]["entrypoint"], "openagent-desktop")
            self.assertFalse(desktop_package["manifest"]["browser_required"])
            self.assertNotIn("Start-Process", desktop_script.read_text(encoding="utf-8"))
            self.assertNotIn("127.0.0.1", desktop_script.read_text(encoding="utf-8"))

            agent = runtime.publish_agent_package(
                name="Community Research Agent",
                version="1.0.0",
                author="community",
                project="core",
                manifest={"agent_id": "community.research", "permissions": ["context.read"]},
            )
            self.assertEqual(agent["status"], "available")

            gpu = runtime.plan_gpu_management(model_id="openagent-local", project="core")
            self.assertIn("selected_adapter", gpu["plan"])
            finetune = runtime.create_finetune_job(
                model_id="openagent-local",
                dataset_ref="local://datasets/research.jsonl",
                project="core",
                job_type="lora",
            )
            self.assertEqual(finetune["job_type"], "lora")
            self.assertEqual(finetune["artifact"]["status"], "planned")

            proactive_job = runtime.create_proactive_job(
                name="Roadmap monitor",
                query="production",
                project="core",
            )
            proactive_run = runtime.run_proactive_job(proactive_job["id"])
            self.assertEqual(proactive_run["job_id"], proactive_job["id"])

            federated = runtime.aggregate_federated_round(
                objective="ranker",
                project="core",
                updates=[{"weight": 1.0, "bias": 0.5}, {"weight": 3.0, "bias": 1.5}],
            )
            self.assertEqual(federated["aggregate"]["weight"], 2.0)

            device_a = runtime.register_sync_device(name="Laptop", public_key="pub-a", project="core")
            device_b = runtime.register_sync_device(name="Phone", public_key="pub-b", project="core")
            bundle = runtime.create_sync_bundle(
                from_device_id=device_a["id"],
                to_device_id=device_b["id"],
                data={"events": [{"title": "Encrypted sync"}]},
                passphrase="sync-passphrase",
                project="core",
            )
            self.assertEqual(bundle["status"], "created")

            sdk = runtime.publish_sdk_package(language="python", version="0.10.0", project="core")
            self.assertEqual(sdk["status"], "published-local")

            security = SecurityPrivacyManager(store)
            profile = security.create_sandbox_profile(
                name="plugin-read",
                permissions=["context.read"],
                network_allowed=False,
            )
            sandbox = runtime.run_sandboxed_plugin(
                plugin_id="community.research",
                profile_id=profile["id"],
                input={"permissions": ["context.read"], "hello": "world"},
                project="core",
            )
            self.assertEqual(sandbox["status"], "completed")

            status = runtime.status(project="core")
            self.assertGreaterEqual(status["app_packages"], 2)
            self.assertGreaterEqual(status["sync_bundles"], 1)


def make_hs256_jwt(claims: dict[str, object], secret: bytes, *, kid: str) -> str:
    header = {"alg": "HS256", "kid": kid, "typ": "JWT"}
    signing_input = ".".join([b64url_json(header), b64url_json(claims)])
    signature = hmac.new(secret, signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{b64url(signature)}"


def b64url_json(payload: dict[str, object]) -> str:
    return b64url(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def b64url(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


if __name__ == "__main__":
    unittest.main()
