"""Phase 3: the task dependency graph is validated and deadlocks are named."""

from __future__ import annotations

import unittest

from openagent.codebase.task_graph import validate_task_graph


def _task(task_id: str, *deps: str) -> dict:
    return {"id": task_id, "depends_on": list(deps)}


class TaskGraphValidationTests(unittest.TestCase):
    def test_acyclic_graph_is_valid(self) -> None:
        result = validate_task_graph([_task("A"), _task("B", "A"), _task("C", "A", "B")])
        self.assertTrue(result["valid"])
        self.assertEqual([], result["cycles"])

    def test_unknown_dependency_is_reported(self) -> None:
        result = validate_task_graph([_task("A", "missing")])
        self.assertFalse(result["valid"])
        self.assertEqual([{"task": "A", "missing": "missing"}], result["unknown_dependencies"])

    def test_cycle_is_named_not_hidden(self) -> None:
        result = validate_task_graph([_task("A", "B"), _task("B", "A")])
        self.assertFalse(result["valid"])
        self.assertTrue(result["cycles"])
        self.assertIn("A", result["cycles"][0])
        self.assertIn("B", result["cycles"][0])

    def test_self_dependency_is_a_cycle(self) -> None:
        result = validate_task_graph([_task("A", "A")])
        self.assertFalse(result["valid"])
        self.assertTrue(result["cycles"])

    def test_empty_graph_is_valid(self) -> None:
        self.assertTrue(validate_task_graph([])["valid"])


if __name__ == "__main__":
    unittest.main()
