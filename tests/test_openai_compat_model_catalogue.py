"""Listing models must not report an unconfigured optional feature as a fault.

`/v1/models` is the first call an OpenAI-compatible client makes: it is how a
client discovers what the server can serve and, in many tools, how it decides
the server is alive at all. The native publisher is opt-in through
OPENAGENT_NATIVE_PUBLISHER_CONFIG and is the only source of a published model
id, so on a default install nothing is published. That used to be answered with
`503` and `"type": "server_error"`, which tells the client the server is
broken rather than that its catalogue is empty.
"""

from __future__ import annotations

import unittest
from http import HTTPStatus
from tempfile import TemporaryDirectory
from typing import Any

from openagent.routes.inference_routes import InferenceRouteDependencies, handle_get
from openagent.storage import OpenAgentStore


class _Gateway:
    """Only the provider registry matters to the catalogue branch."""

    def __init__(self, providers: dict[str, Any]) -> None:
        self.providers = providers


class _Publisher:
    def __init__(self, model: str) -> None:
        self._model = model

    def _configuration(self) -> dict[str, str]:
        return {"model": self._model}


class ModelCatalogueTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.store = OpenAgentStore(self._tmp.name)

    def _get(self, path: str, providers: dict[str, Any] | None = None):
        deps = InferenceRouteDependencies(
            gateway=_Gateway(providers or {}),
            store=self.store,
            read_json=lambda: {},
        )
        return handle_get(path, "", deps)

    def test_an_unconfigured_publisher_gives_an_empty_catalogue(self) -> None:
        result = self._get("/v1/models")
        self.assertEqual(result.status, HTTPStatus.OK)
        self.assertEqual(result.payload, {"object": "list", "data": []})

    def test_the_empty_catalogue_is_not_reported_as_a_server_error(self) -> None:
        result = self._get("/v1/models")
        self.assertNotIn("error", result.payload)
        self.assertLess(int(result.status), 400)

    def test_asking_for_one_model_without_a_publisher_is_a_model_not_found(self) -> None:
        result = self._get("/v1/models/anything")
        self.assertEqual(result.status, HTTPStatus.NOT_FOUND)
        self.assertEqual(result.payload["error"]["type"], "invalid_request_error")

    def test_a_configured_publisher_lists_its_model(self) -> None:
        result = self._get("/v1/models", {"native-publisher": _Publisher("local-published-model")})
        self.assertEqual(result.status, HTTPStatus.OK)
        self.assertEqual([entry["id"] for entry in result.payload["data"]], ["local-published-model"])
        self.assertEqual(result.payload["data"][0]["object"], "model")

    def test_a_configured_publisher_serves_its_own_model_by_id(self) -> None:
        result = self._get("/v1/models/local-published-model", {"native-publisher": _Publisher("local-published-model")})
        self.assertEqual(result.status, HTTPStatus.OK)
        self.assertEqual(result.payload["id"], "local-published-model")

    def test_a_configured_publisher_still_refuses_another_model_id(self) -> None:
        result = self._get("/v1/models/some-other-model", {"native-publisher": _Publisher("local-published-model")})
        self.assertEqual(result.status, HTTPStatus.NOT_FOUND)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
