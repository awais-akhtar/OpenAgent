"""Phase 3: only independent tasks with declared, disjoint write sets may run together."""

from __future__ import annotations

import unittest

from openagent.codebase.task_graph import independent_batch


class IndependentBatchTests(unittest.TestCase):
    def test_disjoint_declared_tasks_are_batched(self) -> None:
        ready = [{"id": "A"}, {"id": "B"}, {"id": "C"}]
        declared = {"A": ["ui/home.tsx"], "B": ["api/users.py"], "C": ["db/schema.sql"]}
        self.assertEqual(["A", "B", "C"], independent_batch(ready, declared))

    def test_overlapping_write_set_is_not_batched(self) -> None:
        ready = [{"id": "A"}, {"id": "B"}]
        declared = {"A": ["src/app.py"], "B": ["src/app.py", "src/extra.py"]}
        self.assertEqual(["A"], independent_batch(ready, declared))

    def test_undeclared_task_runs_alone(self) -> None:
        ready = [{"id": "A"}, {"id": "B"}]
        declared = {"A": ["a.py"]}
        self.assertEqual(["A"], independent_batch(ready, declared))

    def test_selection_is_deterministic_in_ready_order(self) -> None:
        ready = [{"id": "B"}, {"id": "A"}]
        declared = {"A": ["x.py"], "B": ["x.py"]}
        self.assertEqual(["B"], independent_batch(ready, declared))


if __name__ == "__main__":
    unittest.main()
