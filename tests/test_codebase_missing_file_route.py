from __future__ import annotations

import unittest
from unittest.mock import Mock

from openagent.routes.codebase_routes import CodebaseRouteDependencies, handle_get


class CodebaseMissingFileRouteTests(unittest.TestCase):
    def test_missing_project_file_is_not_found(self) -> None:
        codebase = Mock()
        codebase.read_file.side_effect = FileNotFoundError("missing.txt")
        deps = CodebaseRouteDependencies(
            codebase=codebase,
            first=lambda values, key: values.get(key, [""])[0],
            parse_bool=lambda value, default=False: default,
            parse_int=lambda value, default=0: default,
            query_owner=lambda params: "local",
            payload_owner=lambda payload: "local",
            read_json=lambda: {},
        )

        result = handle_get(
            "/api/codebase/projects/project-1/file",
            "path=missing.txt",
            deps,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.status, 404)
        self.assertEqual(result.payload, {"error": "missing.txt"})


if __name__ == "__main__":
    unittest.main()
