from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

from openagent.auth import AuthManager
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore
from openagent.model_gateway import ModelResult, ProviderConfigurationError, StreamChunk


def main() -> int:
    temporary = tempfile.TemporaryDirectory(prefix="openagent-react-python-")
    root = Path(temporary.name)
    os.environ["OPENAGENT_AUTH_ENFORCED"] = "1"
    os.environ["OPENAGENT_DATA_DIR"] = str(root / "data")
    store = OpenAgentStore(root / "data")
    auth = AuthManager(store)
    user = auth.register("integration-owner", "integration-owner@example.test", "temporary-password-1")
    user_id = user["user_id"]
    other_user = auth.register("integration-other", "integration-other@example.test", "temporary-password-2")

    def token(name: str, scopes: list[str]) -> str:
        return auth.create_api_token(user_id, name, scopes=scopes)["token"]

    tokens = {
        "chat": token("smoke-chat", ["chat:read", "chat:write", "local_ai:read"]),
        "codeq": token("smoke-codeq", ["codebase:read", "codebase:write", "local_ai:read"]),
        "research": token("smoke-research", ["knowledge:read", "knowledge:write"]),
        "agents": token("smoke-agents", ["agents:read", "agents:run"]),
        "readonly": token("smoke-readonly", [
            "local_ai:read", "workspace:read", "calendar:read", "docs:read", "media:read",
        ]),
        "otherAgentRead": auth.create_api_token(
            other_user["user_id"], "smoke-other-agent-history", scopes=["agents:read"],
        )["token"],
    }

    repository = root / "disposable-repository"
    repository.mkdir()
    (repository / "README.md").write_text("# Integration fixture\n\nDisposable local repository.\n", encoding="utf-8")
    (repository / "main.py").write_text("def answer():\n    return 'local fixture'\n", encoding="utf-8")

    # Keep the HTTP/API stack real while supplying a deterministic in-process
    # local model provider. This prevents external/network model side effects.
    import openagent.routes.runtime as runtime_module
    real_gateway_type = runtime_module.ModelGateway

    class SyntheticLocalGateway:
        def __init__(self, *args, **kwargs):
            self.delegate = real_gateway_type(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.delegate, name)

        def complete(self, messages, *, provider_id="local", model=None, allow_external=False, **kwargs):
            prompt = "\n".join(str(message.get("content", "")) for message in messages if isinstance(message, dict))
            print(f"integration fixture completion request: {prompt[-160:]!r}", file=sys.stderr, flush=True)
            if "[integration-resume-fail]" in prompt:
                raise ProviderConfigurationError("Controlled local failure for resume integration coverage.")
            if "[integration-research-cancel]" in prompt:
                # Hold the fake local provider long enough for the browser to
                # submit cancellation before the worker reaches completion.
                time.sleep(45)
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
            if "[integration-delay]" in prompt:
                time.sleep(4)
            print("integration fixture provider: yielding deterministic answer", file=sys.stderr, flush=True)
            yield StreamChunk(provider="local", model="integration-fixture", content="Local integration response.", done=False, metadata={})
            yield StreamChunk(provider="local", model="integration-fixture", content="", done=True, metadata={})

        def stream_complete_vision(self, *args, **kwargs):
            yield StreamChunk(provider="local", model="integration-fixture", content="Local fixture image response.", done=True, metadata={})

    runtime_module.ModelGateway = SyntheticLocalGateway
    server = build_http_server("127.0.0.1", 0, store)
    thread = threading.Thread(target=server.serve_forever, name="real-integration-http", daemon=True)
    thread.start()
    print(json.dumps({
        "baseUrl": f"http://127.0.0.1:{server.server_address[1]}",
        "tokens": tokens,
        "repoPath": str(repository),
        "ownerId": user_id,
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
