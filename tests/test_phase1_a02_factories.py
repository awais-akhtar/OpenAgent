"""Phase 1 A02 — Native family factory verification.

Tests that all priority model factories are implemented and importable.
Source-level verification; full end-to-end requires real checkpoints (A10).
"""

import unittest
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
from unittest.mock import patch


class Phase1A02FactoriesTest(unittest.TestCase):
    """Verify A02 factories exist and are callable."""

    def test_p02_owner_resolver_checks_python_symbols_not_only_module_names(self):
        from openagent.adaptive_engine.phase1_p02_route_bindings import (
            _resolve_callable_owner_source,
        )
        repository_root = Path(__file__).resolve().parents[1]
        self.assertTrue(_resolve_callable_owner_source(
            "native_checkpoint.read_native_text_checkpoint", repository_root))
        self.assertFalse(_resolve_callable_owner_source(
            "native_checkpoint.not_a_real_phase1_owner", repository_root))

    def test_exact_source_reviewed_bindings_cover_every_unique_admitted_tuple(self):
        """P02 bindings cover the canonical matrix without asserting execution."""
        from dataclasses import replace
        from openagent.adaptive_engine.phase1_p02_route_bindings import (
            source_reviewed_route_bindings,
            get_source_reviewed_route_binding,
            validate_source_reviewed_route_bindings,
            validate_callable_owner_sources,
        )
        from openagent.adaptive_engine.source_scope import source_matrix_rows

        rows = source_matrix_rows(admitted_only=True)
        expected = {
            (row["family"], row["model_type"], row["backend"], row["arithmetic"],
             row["source_format"], row["persisted_format"])
            for row in rows
        }
        bindings = source_reviewed_route_bindings()
        actual = {binding.route_key for binding in bindings}
        self.assertEqual(actual, expected)
        self.assertEqual(len(bindings), len(expected))
        self.assertTrue(validate_source_reviewed_route_bindings())
        self.assertGreaterEqual(validate_callable_owner_sources(Path(__file__).resolve().parents[1]), 170)
        self.assertTrue(all(binding.runtime_certified is False for binding in bindings))
        for binding in bindings:
            self.assertTrue(binding.validate())
            self.assertEqual(tuple(name for name, _ in binding.callable_chain),
                             binding.chain_order)
            self.assertTrue(all(binding.owners_for(name)
                                for name in binding.chain_order))
            self.assertEqual(get_source_reviewed_route_binding(
                family=binding.family, model_type=binding.model_type,
                backend=binding.backend, arithmetic=binding.arithmetic,
                source_format=binding.source_format,
                persisted_format=binding.persisted_format), binding)

        altered = replace(bindings[0], source_review="p02-forged-review")
        with self.assertRaisesRegex(ValueError, "exact canonical P02 row"):
            altered.validate()
        first_dimension, _ = bindings[0].callable_chain[0]
        altered_chain = ((first_dimension, ("forged.owner",)),) + bindings[0].callable_chain[1:]
        with self.assertRaisesRegex(ValueError, "exact canonical P02 row"):
            replace(bindings[0], callable_chain=altered_chain).validate()
        with self.assertRaises(KeyError):
            get_source_reviewed_route_binding(
                family="not-a-family", model_type="unknown", backend="native",
                arithmetic="fp32", source_format="F32",
                persisted_format="float32")
        with self.assertRaisesRegex(ValueError, "complete six-field"):
            get_source_reviewed_route_binding(
                family=None, model_type="unknown", backend="native",
                arithmetic="fp32", source_format="F32",
                persisted_format="float32")

    def test_canonical_verifier_reports_the_p02_binding_coverage(self):
        from phase1_verify import report_canonical_matrix

        output = StringIO()
        with redirect_stdout(output):
            source_audit_ok, counts = report_canonical_matrix()
        self.assertTrue(source_audit_ok)
        self.assertEqual(counts["verified-admitted"], 171)
        self.assertIn("P02 exact source-route bindings: PASS (162 unique admitted tuples)",
                      output.getvalue())
        self.assertIn("P02 callable owner source references: PASS (178 unique owners)",
                      output.getvalue())

    def test_all_factories_importable(self):
        """A02: All 5 priority model factories are importable and callable."""
        from openagent.adaptive_engine.execution_factories import (
            create_qwen35_native_session,
            create_qwen4_native_session,
            create_glm5_native_session,
            create_glm_dsa_native_session,
            create_deepseek_v41_native_session
        )

        factories = [
            create_qwen35_native_session,
            create_qwen4_native_session,
            create_glm5_native_session,
            create_glm_dsa_native_session,
            create_deepseek_v41_native_session,
        ]

        for factory in factories:
            self.assertTrue(callable(factory), f"{factory.__name__} is not callable")
            self.assertIsNotNone(factory.__doc__, f"{factory.__name__} missing docstring")
            self.assertIn('a02', factory.__doc__.lower(), f"{factory.__name__} not marked A02")

    def test_native_checkpoint_loader_exists(self):
        """A02: native_checkpoint loader function exists."""
        from openagent.adaptive_engine import native_checkpoint
        self.assertTrue(hasattr(native_checkpoint, 'read_native_text_checkpoint'))

    def test_native_wrapper_and_text_routes_have_distinct_factory_traces(self):
        """Wrapper checkpoints and their separate text-service identities stay distinct."""
        from openagent.adaptive_engine.device_routes import (
            QWEN35_NATIVE_TRACE, QWEN4_NATIVE_TRACE,
            GLM5_NATIVE_TRACE, GLM_DSA_NATIVE_TRACE,
            MIXTRAL_NATIVE_TRACE, OLMOE_NATIVE_TRACE,
            DEEPSEEK_V41_NATIVE_BF16_TRACE, K3_NATIVE_TRACE, INKLING_NATIVE_TRACE,
            ALL_ROUTES, validate_all_routes_complete,
        )
        from openagent.adaptive_engine.device_routes import (
            QWEN35_TEXT_NATIVE_TRACE, QWEN4_TEXT_NATIVE_TRACE, GLM5_TEXT_NATIVE_TRACE,
        )
        from openagent.adaptive_engine.source_scope import source_matrix_rows

        self.assertTrue(validate_all_routes_complete())
        canonical = {
            (row['family'], row['model_type'], row['backend'], row['arithmetic'],
             row['source_format'], row['persisted_format'])
            for row in source_matrix_rows()
        }
        for route in ALL_ROUTES:
            self.assertIn(
                (route.family, route.model_type, route.backend, route.arithmetic,
                 route.source_format, route.persisted_format), canonical,
                f"trace is not an exact canonical route: {route}")
        text_factories = (
            (QWEN35_TEXT_NATIVE_TRACE, "qwen3_5_moe_text", "create_qwen35_native_session"),
            (QWEN4_TEXT_NATIVE_TRACE, "qwen4_exp_text", "create_qwen4_native_session"),
            (GLM5_TEXT_NATIVE_TRACE, "glm5_next_text", "create_glm5_native_session"),
            (GLM_DSA_NATIVE_TRACE, "glm_moe_dsa", "create_glm_dsa_native_session"),
        )
        for route, model_type, factory_name in text_factories:
            self.assertEqual(route.abi.factory_model_type, model_type)
            self.assertIn(factory_name, route.abi.factory_function)
            self.assertIn("not wired", route.abi.cli_entry)
            self.assertIn("not wired", route.cache_tp.tp_owner)
            self.assertIn("NativeExecutionService.save_session_cache", route.recovery_writer)
            self.assertIn("NativeExecutionService.restore_session_cache", route.recovery_reader)
            self.assertNotIn("Qwen35CacheRecovery", route.cache_tp.cache_owner)
            self.assertNotIn("Qwen4CacheRecovery", route.cache_tp.cache_owner)

        wrapper_routes = (
            (QWEN35_NATIVE_TRACE, "qwen3_5_moe"),
            (QWEN4_NATIVE_TRACE, "qwen4_exp"),
            (GLM5_NATIVE_TRACE, "glm5_next"),
        )
        for route, model_type in wrapper_routes:
            self.assertEqual(route.model_type, model_type)
            self.assertEqual(route.abi.factory_model_type, model_type)
            self.assertIn("open_native_conditional_directory_service", route.abi.factory_function)
            self.assertIn("native_conditional_cli.PublisherSession", route.abi.cli_entry)
            self.assertIn("MultimodalSessionExecutor", route.consumer.tensor_consumer)

        self.assertIn("native_family_cli.PublisherSession", MIXTRAL_NATIVE_TRACE.loader.loader_function)
        self.assertIn("not established", MIXTRAL_NATIVE_TRACE.cache_tp.tp_owner)
        self.assertIn("no dedicated wholemodel_validation", MIXTRAL_NATIVE_TRACE.tooling)
        self.assertIn("universal_adapters.OlmoeNativeOwner", OLMOE_NATIVE_TRACE.cache_tp.cache_owner)
        self.assertIn("not tensor-parallel", OLMOE_NATIVE_TRACE.cache_tp.tp_owner)
        self.assertIn("olmoe_cli.main", OLMOE_NATIVE_TRACE.abi.cli_entry)
        self.assertIn("not exposed", OLMOE_NATIVE_TRACE.recovery_writer)
        self.assertIn("PublisherV41Execution", DEEPSEEK_V41_NATIVE_BF16_TRACE.cache_tp.cache_owner)
        self.assertNotIn("PublisherV41CheckpointOwner", DEEPSEEK_V41_NATIVE_BF16_TRACE.cache_tp.cache_owner)
        self.assertIn("K3Execution", K3_NATIVE_TRACE.cache_tp.cache_owner)
        self.assertIn("create_public_k3_execution_service", K3_NATIVE_TRACE.abi.factory_function)
        self.assertNotIn("K3CheckpointManager", K3_NATIVE_TRACE.cache_tp.cache_owner)
        self.assertIn("open_native_conditional_directory_service", INKLING_NATIVE_TRACE.abi.factory_function)
        self.assertIn("GroupedSnapshot", INKLING_NATIVE_TRACE.abi.continuation_format)
        self.assertNotIn("ContinuationSnapshot", INKLING_NATIVE_TRACE.abi.continuation_format)

    def test_qwen35_factory_binds_config_and_reloads_fresh_decoder_per_session(self):
        from openagent.adaptive_engine.execution_factories import create_qwen35_native_session

        class Snapshot:
            def assert_unchanged(self):
                return None

        class Decoder:
            def __init__(self, ordinal):
                self.ordinal = ordinal
                self.closed = 0

            @property
            def position(self):
                return 0

            def forward(self, token_ids, **kwargs):
                import numpy as np
                return np.zeros((len(token_ids), 8), dtype=np.float32)

            def snapshot(self):
                return self.ordinal

            def restore(self, receipt):
                if receipt != self.ordinal:
                    raise ValueError('foreign fake snapshot')

            def reset(self):
                return None

            def close(self):
                self.closed += 1

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / 'config.json'
            config.write_text(json.dumps({
                'model_type': 'qwen3_5_moe_text', 'vocab_size': 8,
                'max_position_embeddings': 16}), encoding='utf-8')
            library = root / 'decoder.dll'
            library.write_bytes(b'fixture library identity')
            decoders = []

            def load_decoder(*args, **kwargs):
                decoder = Decoder(len(decoders))
                decoders.append(decoder)
                return decoder

            with patch("openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission",
                       return_value=Snapshot()) as admission, \
                 patch("openagent.adaptive_engine.native_checkpoint.load_native_text_decoder",
                       side_effect=load_decoder) as loader:
                service = create_qwen35_native_session(
                    root, config, library=library, max_weight_bytes=1024,
                    max_bytes=1024, max_new_tokens=4)
                first_id = service.open_session()
                first = service._session(first_id).model
                service.close_session(first_id)
                second_id = service.open_session()
                second = service._session(second_id).model
                service.close_session(second_id)
                service.close()

            self.assertIsNot(first, second)
            self.assertEqual(first.closed, 1)
            self.assertEqual(second.closed, 1)
            self.assertEqual(loader.call_count, 2)
            admission.assert_called_once_with(
                root.resolve(), family='qwen35', model_type='qwen3_5_moe_text',
                backend='native', arithmetic='fp32', persisted_format='float32')

    def test_text_factory_rejects_foreign_config_and_wrapper_alias_before_loading(self):
        from openagent.adaptive_engine.execution_factories import create_qwen35_native_session

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'checkpoint'
            root.mkdir()
            config = root / 'config.json'
            config.write_text(json.dumps({
                'model_type': 'qwen3_5_moe_text', 'vocab_size': 8,
                'max_position_embeddings': 16}), encoding='utf-8')
            library = root / 'decoder.dll'
            library.write_bytes(b'fixture library identity')
            foreign = Path(temporary) / 'foreign.json'
            foreign.write_text(config.read_text(encoding='utf-8'), encoding='utf-8')
            with patch("openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission") as admit:
                with self.assertRaisesRegex(ValueError, "canonical config.json"):
                    create_qwen35_native_session(root, foreign, library=library)
                admit.assert_not_called()

            config.write_text(json.dumps({
                'model_type': 'qwen3_5_moe', 'vocab_size': 8,
                'max_position_embeddings': 16}), encoding='utf-8')
            with patch("openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission") as admit:
                with self.assertRaisesRegex(ValueError, "qwen3_5_moe_text"):
                    create_qwen35_native_session(root, config, library=library)
                admit.assert_not_called()

    def test_qwen4_and_glm_text_factories_use_exact_loader_identities(self):
        from openagent.adaptive_engine.execution_factories import (
            create_glm5_native_session, create_glm_dsa_native_session,
            create_qwen4_native_session)

        class Snapshot:
            def assert_unchanged(self):
                return None

        class Decoder:
            def __init__(self):
                self.closed = 0

            def forward(self, token_ids):
                return token_ids

            def snapshot(self):
                return object()

            def restore(self, receipt):
                return None

            def reset(self):
                return None

            def close(self):
                self.closed += 1

        factories = (
            (create_qwen4_native_session, "qwen4_exp_text", "qwen4",
             "max_position_embeddings", {}),
            (create_glm5_native_session, "glm5_next_text", "glm5-next",
             "seq_length", {"context_tokens": 16}),
            (create_glm_dsa_native_session, "glm_moe_dsa", "glm-dsa",
             "max_seq_len", {"max_snapshots": 4}),
        )
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            library = base / "decoder.dll"
            library.write_bytes(b"fixture library identity")
            with patch("openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission",
                       side_effect=lambda *args, **kwargs: Snapshot()) as admission, \
                 patch("openagent.adaptive_engine.native_checkpoint.load_native_text_decoder",
                       side_effect=lambda *args, **kwargs: Decoder()) as loader:
                for factory, model_type, family, context_key, expected_options in factories:
                    root = base / model_type
                    root.mkdir()
                    config = root / "config.json"
                    config.write_text(json.dumps({
                        "model_type": model_type, "vocab_size": 8,
                        context_key: 16}), encoding="utf-8")
                    service = factory(root, config, library=library,
                                      max_weight_bytes=1024, max_bytes=1024,
                                      max_new_tokens=4)
                    service.close()

            self.assertEqual(admission.call_count, 3)
            self.assertEqual(loader.call_count, 3)
            for call, (_, model_type, family, _, expected_options) in zip(
                    loader.call_args_list, factories):
                self.assertEqual(call.kwargs["model_type"], model_type)
                self.assertEqual(call.kwargs["library"], library.resolve())
                self.assertEqual(call.kwargs["max_weight_bytes"], 1024)
                for name, expected in expected_options.items():
                    self.assertEqual(call.kwargs[name], expected)
            for call, (_, model_type, family, _, _) in zip(admission.call_args_list, factories):
                self.assertEqual(call.kwargs["model_type"], model_type)
                self.assertEqual(call.kwargs["family"], family)

    def test_v41_factory_delegates_to_the_bank_aware_service(self):
        from openagent.adaptive_engine.execution_factories import create_deepseek_v41_native_session

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            config.write_text(json.dumps({"model_type": "deepseek_v41"}), encoding="utf-8")
            budgets = {
                "execution_library": "execution.dll", "mxfp4_library": "fp4.dll",
                "fp8_library": "fp8.dll", "context_tokens": 16,
                "max_weight_bytes": 1024, "runtime_max_bytes": 1024,
                "arithmetic_max_bytes": 1024, "max_expert_bytes": 1024,
                "max_projection_bytes": 1024, "max_new_tokens": 4,
                "max_snapshot_bytes": 2048,
            }
            sentinel = object()
            with patch("openagent.adaptive_engine.execution_factories.open_publisher_v41_service",
                       return_value=sentinel) as bank_factory:
                result = create_deepseek_v41_native_session(root, config, **budgets)

            self.assertIs(result, sentinel)
            bank_factory.assert_called_once_with(root.resolve(), **budgets)

            config.write_text(json.dumps({"model_type": "deepseek_v41_text"}), encoding="utf-8")
            with patch("openagent.adaptive_engine.execution_factories.open_publisher_v41_service") as bank_factory:
                with self.assertRaisesRegex(ValueError, "exact DeepSeek V4.1 model_type"):
                    create_deepseek_v41_native_session(root, config, **budgets)
                bank_factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
