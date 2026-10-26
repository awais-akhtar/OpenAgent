from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone

from openagent.auth import AuthManager
from openagent.context import ContextEngine
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore
from openagent.calendar_system import CalendarSystem
from openagent.model_gateway import ModelResult, ProviderConfigurationError, StreamChunk
from openagent.models import AuditEvent
import openagent.codebase.service as codebase_service_module


def main() -> int:
    temporary = tempfile.TemporaryDirectory(prefix="openagent-react-python-")
    root = Path(temporary.name)
    os.environ["OPENAGENT_DEPLOYMENT_MODE"] = "personal_remote"
    os.environ["OPENAGENT_AUTH_ENFORCED"] = "1"
    os.environ["OPENAGENT_PERSONAL_REMOTE_HTTPS_TERMINATED"] = "1"
    os.environ["OPENAGENT_PERSONAL_REMOTE_HOSTS"] = "remote.test"
    os.environ["OPENAGENT_BOOTSTRAP_TOKEN"] = "batch2b-integration-bootstrap-secret-32-chars"
    os.environ["OPENAGENT_DATA_DIR"] = str(root / "data")
    store = OpenAgentStore(root / "data")
    ContextEngine(store).ingest(
        title="Research source-note smoke fixture",
        content="Synthetic local evidence for verifying persisted per-source Research notes. This record is confined to the temporary integration database.",
        type="note", source="integration-test", project="default",
    )
    store.add_audit_event(AuditEvent(
        actor="integration-fixture", action="security.audit.smoke",
        resource="synthetic-only", project="default", status="success",
        metadata={"private_marker": "must-not-be-rendered"}, id="integration-audit-smoke",
    ))
    # A due reminder gives the real React→Python smoke a safe, deterministic
    # record for exercising the acknowledgement lifecycle.
    due_start = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    CalendarSystem(store).create_event(
        "Batch 2B due reminder fixture", due_start,
        (datetime.now(timezone.utc) - timedelta(minutes=4)).isoformat(),
        project="default", reminders=[0],
    )
    # Inject exactly one TCP-close style interruption into a marked disposable
    # CodeQ SSE run. The real HTTP handler, event records and React subscriber
    # remain in use; the next request resumes from the supplied cursor.
    original_codeq_event_stream = codebase_service_module.CodebaseService.event_stream
    interrupted_codeq_streams: set[str] = set()

    def interrupt_marked_codeq_stream(service, run_id, *, owner="local", after=0, timeout_seconds=120):
        for event in original_codeq_event_stream(
            service, run_id, owner=owner, after=after, timeout_seconds=timeout_seconds,
        ):
            yield event
            if run_id not in interrupted_codeq_streams:
                run = service.get_run(run_id, owner=owner)
                if "[integration-reconnect]" in str(run.get("user_prompt", "")):
                    interrupted_codeq_streams.add(run_id)
                    raise ConnectionResetError("Synthetic transport interruption for React SSE reconnect coverage")

    codebase_service_module.CodebaseService.event_stream = interrupt_marked_codeq_stream
    auth = AuthManager(store)
    user = auth.register(
        "integration-owner", "integration-owner@example.test", "temporary-password-1",
        bootstrap_token=os.environ["OPENAGENT_BOOTSTRAP_TOKEN"],
    )
    user_id = user["user_id"]
    # A foreign installation's signed credential cannot resolve against this
    # one-owner store. Do not create a second account in the tested database.
    foreign_store = OpenAgentStore(root / "foreign-installation")
    foreign_auth = AuthManager(foreign_store)
    foreign_user = foreign_auth.register(
        "foreign-installation-owner", "foreign-owner@example.test", "temporary-password-2",
        bootstrap_token=os.environ["OPENAGENT_BOOTSTRAP_TOKEN"],
    )

    def token(name: str, scopes: list[str]) -> str:
        return auth.create_api_token(user_id, name, scopes=scopes)["token"]

    tokens = {
        "chat": token("smoke-chat", ["chat:read", "chat:write", "local_ai:read"]),
        # Home's attachment handoff composes Chat streaming with stored-media
        # upload. Keep that integration test credential separate from the
        # single-purpose Chat and Documents credentials above.
        "chatMedia": token("smoke-chat-media", ["chat:read", "chat:write", "local_ai:read", "media:read", "media:write"]),
        "codeq": token("smoke-codeq", ["codebase:read", "codebase:write", "local_ai:read", "terminal:read", "terminal:write"]),
        "research": token("smoke-research", ["knowledge:read", "knowledge:write"]),
        "agents": token("smoke-agents", ["agents:read", "agents:run"]),
        "agentsAdmin": token("smoke-agents-admin", ["agents:read", "agents:run", "agents:admin"]),
        "agentsRunOnly": token("smoke-agents-run-only", ["agents:run"]),
        "agentsReadOnly": token("smoke-agents-read-only", ["agents:read"]),
        "neitherAgentScope": token("smoke-neither-agent-scope", ["chat:read"]),
        "readonly": token("smoke-readonly", [
            "local_ai:read", "workspace:read", "calendar:read", "docs:read", "media:read",
        ]),
        "models": token("smoke-models", ["local_ai:read", "local_ai:admin"]),
        "personalOS": token("smoke-personal-os", ["personal_os:read", "personal_os:write"]),
        "workflows": token("smoke-workflows", ["workflows:read", "workflows:write"]),
        "scraper": token("smoke-scraper", ["scraper:read", "scraper:write"]),
        "documents": token("smoke-documents", ["docs:read", "docs:write", "media:read", "media:write"]),
        "workspace": token("smoke-workspace", ["workspace:read", "workspace:write"]),
        # The Knowledge screen reads/ingests indexed documents with
        # knowledge:* and manages semantic collections with semantic:*.
        # Keep this integration credential limited to those exact operations.
        "knowledge": token("smoke-knowledge", ["knowledge:read", "knowledge:write", "semantic:read", "semantic:write"]),
        "voice": token("smoke-voice", ["voice:read", "voice:write"]),
        "communication": token("smoke-communications", ["communication:read", "communication:write"]),
        "calendar": token("smoke-calendar", ["calendar:read", "calendar:write"]),
        # Fine-tuning Jobs composes the Fine-tuning contract with read-only
        # Local AI model/job catalogs. Keep the credential least-privilege but
        # grant the additional read scope the actual screen requests.
        "finetune": token("smoke-finetune", ["finetune:read", "finetune:write", "local_ai:read"]),
        "integrations": token("smoke-integrations", ["integrations:read", "integrations:admin"]),
        "assistant": token("smoke-assistant", ["assistant:read", "assistant:write"]),
        "twin": token("smoke-twin", ["twin:read", "twin:write"]),
        "securityAdmin": token("smoke-security-admin", ["security:admin"]),
        "securityRead": token("smoke-security-read", ["security:read"]),
        "otherAgentRead": foreign_auth.create_api_token(
            foreign_user["user_id"], "smoke-foreign-agent-history", scopes=["agents:read"],
        )["token"],
        "otherResearchRead": foreign_auth.create_api_token(
            foreign_user["user_id"], "smoke-foreign-research-read", scopes=["knowledge:read"],
        )["token"],
        "otherDocumentsRead": foreign_auth.create_api_token(
            foreign_user["user_id"], "smoke-foreign-documents-read", scopes=["docs:read", "media:read"],
        )["token"],
        "otherKnowledgeRead": foreign_auth.create_api_token(
            foreign_user["user_id"], "smoke-foreign-knowledge-read", scopes=["knowledge:read"],
        )["token"],
    }

    repository = root / "disposable-repository"
    repository.mkdir()
    (repository / "README.md").write_text("# Integration fixture\n\nDisposable local repository.\n", encoding="utf-8")
    (repository / "main.py").write_text("def answer():\n    return 'local fixture'\n", encoding="utf-8")

    # Keep the HTTP/API stack real while supplying a deterministic in-process
    # local model provider. This prevents external/network model side effects.
    import openagent.routes.runtime as runtime_module
    from openagent.http_security import RateLimit, SlidingWindowRateLimiter
    real_gateway_type = runtime_module.ModelGateway
    real_limiter_type = runtime_module.SlidingWindowRateLimiter

    def smoke_rate_limiter():
        # The smoke suite polls lifecycle state while exercising a single
        # synthetic workflow. Keep production throttling tests in the Python
        # unit suite, and prevent these safe local probes from consuming the
        # production per-minute external-request budget.
        limits = {name: RateLimit(1000, 60.0) for name in ("auth", "upload", "external", "model_job", "general")}
        return SlidingWindowRateLimiter(limits)

    class SyntheticLocalGateway:
        def __init__(self, *args, **kwargs):
            self.delegate = real_gateway_type(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.delegate, name)

        def complete(self, messages, *, provider_id="local", model=None, allow_external=False, **kwargs):
            prompt = "\n".join(str(message.get("content", "")) for message in messages if isinstance(message, dict))
            print(f"integration fixture completion request: {prompt[-160:]!r}", file=sys.stderr, flush=True)
            if "[integration-fail]" in prompt:
                raise ProviderConfigurationError("Synthetic local integration provider failure.")
            if "Run-only scope fixture" in prompt or "History integration fixture from the Agents workspace" in prompt:
                return ModelResult(provider="local", model="integration-fixture", content="Synthetic local agent response for the Batch 2A authorization test.", metadata={})
            if "[integration-agent-cancel]" in prompt:
                # Keep this disposable Agent job active until the real React
                # cancel control has submitted its authenticated request.
                print("integration Agent cancellation provider delay entered", file=sys.stderr, flush=True)
                time.sleep(8)
                return ModelResult(provider="local", model="integration-fixture", content="Synthetic local cancellation fixture.", metadata={})
            if "[integration-resume-fail]" in prompt:
                raise ProviderConfigurationError("Controlled local failure for resume integration coverage.")
            if "[integration-research-cancel]" in prompt:
                # Hold the fake local provider long enough for the browser to
                # submit cancellation before the worker reaches completion.
                time.sleep(8)
                return ModelResult(provider="local", model="integration-fixture", content='{"report":"Synthetic local cancellation fixture."}', metadata={})
            if "[integration-reconnect]" in prompt:
                # Keep a disposable CodeQ run active while Playwright interrupts
                # and restores the browser network for the real SSE consumer.
                time.sleep(25)
                return ModelResult(provider="local", model="integration-fixture", content="{}", metadata={})
            if "[integration-delay]" in prompt:
                time.sleep(8)
                return ModelResult(provider="local", model="integration-fixture", content="{}", metadata={})
            return self.delegate.complete(messages, provider_id=provider_id, model=model,
                                          allow_external=allow_external, **kwargs)

        def stream_complete(self, messages, *, provider_id="local", model=None, allow_external=False, **kwargs):
            prompt = str(messages[-1].get("content", "")) if messages else ""
            print(f"integration fixture provider: request={prompt[:60]!r}", file=sys.stderr, flush=True)
            if "[integration-fail]" in prompt:
                raise ProviderConfigurationError("Synthetic local integration provider failure.")
            if "[integration-agent-cancel]" in prompt:
                # Agent runs use the streaming provider path; hold this marked
                # fixture here until React submits the real cancel request.
                time.sleep(8)
            if "[integration-research-cancel]" in prompt:
                # Research synthesis uses the streaming gateway path. Keep its
                # deterministic local response in flight long enough for the
                # real React Cancel action to arrive before completion.
                time.sleep(8)
            if "[integration-delay]" in prompt:
                time.sleep(4)
            print("integration fixture provider: yielding deterministic answer", file=sys.stderr, flush=True)
            yield StreamChunk(provider="local", model="integration-fixture", content="Local integration response.", done=False, metadata={})
            yield StreamChunk(provider="local", model="integration-fixture", content="", done=True, metadata={})

        def stream_complete_vision(self, *args, **kwargs):
            yield StreamChunk(provider="local", model="integration-fixture", content="Local fixture image response.", done=True, metadata={})

    runtime_module.ModelGateway = SyntheticLocalGateway
    runtime_module.SlidingWindowRateLimiter = smoke_rate_limiter
    server = build_http_server("127.0.0.1", 0, store)
    runtime_module.SlidingWindowRateLimiter = real_limiter_type
    thread = threading.Thread(target=server.serve_forever, name="real-integration-http", daemon=True)
    thread.start()
    print(json.dumps({
        "baseUrl": f"http://127.0.0.1:{server.server_address[1]}",
        "tokens": tokens,
        "repoPath": str(repository),
        "ownerId": user_id,
        "ownerUsername": "integration-owner",
        "ownerPassword": "temporary-password-1",
    }), flush=True)
    try:
        for line in sys.stdin:
            if line.strip() == "STOP":
                break
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
