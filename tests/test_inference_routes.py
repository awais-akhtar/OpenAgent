from __future__ import annotations

from http import HTTPStatus
import unittest

from openagent.model_gateway import ProviderUnavailableError
from openagent.routes.inference_routes import InferenceRouteDependencies, handle_post


class _UnavailableGateway:
    def warmup(self, **kwargs):
        raise ProviderUnavailableError("native worker failed")


class _AuditStore:
    def add_audit_event(self, event) -> None:
        raise AssertionError("failed warmup must not be recorded as successful")


class InferenceRouteTests(unittest.TestCase):
    def test_native_warmup_failure_returns_structured_service_unavailable(self) -> None:
        deps = InferenceRouteDependencies(
            gateway=_UnavailableGateway(),
            store=_AuditStore(),
            read_json=lambda: {"project": "default", "provider": "openvino-direct"},
        )

        result = handle_post("/api/inference/warmup", deps)

        self.assertIsNotNone(result)
        self.assertEqual(result.status, HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertEqual(result.payload["status"], "unavailable")
        self.assertIn("native worker failed", result.payload["error"])


if __name__ == "__main__":
    unittest.main()
