from __future__ import annotations

import unittest

from openagent.routes import document_routes


class DocumentRouteContractTests(unittest.TestCase):
    def test_document_lifecycle_routes_are_declared(self) -> None:
        routes = {(route["method"], route["path"]) for route in document_routes.contract()["routes"]}

        self.assertIn(("POST", "/api/document"), routes)
        self.assertIn(("GET", "/api/documents/library"), routes)
        self.assertIn(("POST", "/api/document/{doc_id}/restore/{num}"), routes)


if __name__ == "__main__":
    unittest.main()
