from __future__ import annotations

import unittest
from dataclasses import replace

from openagent.adaptive_engine.phase1_p02_route_bindings import (
    all_bindings,
    require_route_binding,
    validate_all_bindings,
)
from openagent.adaptive_engine.source_scope import source_matrix_rows


class Phase1P02RouteBindingTests(unittest.TestCase):
    def test_all_immutable_bindings_match_canonical_admitted_chains(self) -> None:
        bindings = all_bindings()

        admitted = {
            (row["family"], row["model_type"], row["backend"], row["arithmetic"],
             row["source_format"], row["persisted_format"])
            for row in source_matrix_rows(admitted_only=True)
        }
        self.assertEqual({binding.route_key for binding in bindings}, admitted)
        self.assertTrue(all(binding.validate() is True for binding in bindings))
        self.assertTrue(all(result == "ok" for result in validate_all_bindings().values()))

    def test_binding_with_changed_consumer_is_rejected(self) -> None:
        binding = all_bindings()[0]
        dimension = binding.callable_chain[0][0]
        altered = replace(
            binding,
            callable_chain=((dimension, ("unreviewed.fake_consumer",)),) + binding.callable_chain[1:],
        )

        with self.assertRaisesRegex(ValueError, "differs from its exact canonical"):
            altered.validate()

    def test_exact_key_lookup_does_not_fall_back_to_neighboring_tuple(self) -> None:
        binding = all_bindings()[0]

        with self.assertRaisesRegex(ValueError, "no P02 source-reviewed binding"):
            require_route_binding(
                family=binding.family,
                model_type=binding.model_type,
                backend=binding.backend,
                arithmetic=binding.arithmetic,
                source_format=binding.source_format,
                persisted_format="invented-format",
            )


if __name__ == "__main__":
    unittest.main()
