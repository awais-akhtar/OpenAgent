from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from openagent.communication import CommunicationHub
from openagent.mcp_runtime import MCPRuntime
from openagent.model_cookbook import ModelCookbook
from openagent.storage import OpenAgentStore


class _Response:
    def __init__(self, status: int, body: bytes = b"{}") -> None:
        self.status = status
        self.body = body
        self.headers: dict[str, str] = {"content-type": "application/json"}


class ServiceEgressTests(unittest.TestCase):
    def test_oauth_rejects_http_error_status(self) -> None:
        with TemporaryDirectory() as tmp:
            hub = CommunicationHub(OpenAgentStore(tmp))
            account = hub.configure_oauth_account(
                provider="generic",
                client_id="client",
                token_endpoint="https://oauth.example.test/token",
            )
            with patch.object(hub.egress, "fetch", return_value=_Response(401)) as fetch:
                with self.assertRaisesRegex(ValueError, "HTTP 401"):
                    hub.refresh_oauth_token(
                        account["id"],
                        allow_external=True,
                        authorization_code="code",
                    )
            self.assertTrue(fetch.call_args.kwargs["approved"])

    def test_mcp_rejects_http_error_status(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = MCPRuntime(OpenAgentStore(tmp))
            with patch.object(runtime.egress, "fetch", return_value=_Response(503)) as fetch:
                with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
                    runtime._invoke_http(
                        "http://127.0.0.1:8765/mcp",
                        "inspect",
                        {},
                        2.0,
                        10_000,
                        approved=True,
                    )
            self.assertTrue(fetch.call_args.kwargs["approved"])
            self.assertTrue(fetch.call_args.kwargs["allow_loopback"])

    def test_model_certification_rejects_http_error_status(self) -> None:
        with TemporaryDirectory() as tmp:
            cookbook = ModelCookbook(OpenAgentStore(tmp))
            with patch.object(cookbook.egress, "fetch", return_value=_Response(502)) as fetch:
                with self.assertRaisesRegex(ValueError, "HTTP 502"):
                    cookbook._endpoint_json(
                        "https://models.example.test/v1/models",
                        headers={"Accept": "application/json"},
                        timeout=2,
                        approved=True,
                        allow_loopback=False,
                    )
            self.assertTrue(fetch.call_args.kwargs["approved"])
            self.assertFalse(fetch.call_args.kwargs["allow_loopback"])


if __name__ == "__main__":
    unittest.main()
