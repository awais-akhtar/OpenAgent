import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from openagent.adaptive_engine.olmoe_cli import _kernel_resources, _close_kernel_resources, build_parser


class KernelSelectionTests(unittest.TestCase):
    def test_q8_0_cli_resolves_real_registered_backend(self):
        registry, client = _kernel_resources(Path('.'), 'q8_0-numpy', 1)
        self.assertIsNone(client)
        try:
            backend = registry.resolve(dtype='q8_0', preferred='q8_0-numpy')
            self.assertEqual(backend.info().name, 'q8_0-numpy')
            self.assertEqual(backend.info().capabilities.payload_dtypes, frozenset({'q8_0'}))
        finally:
            registry.close()

    def test_int4_default_paths_do_not_replace_int8_artifacts(self):
        from openagent.adaptive_engine.olmoe_cli import main

        with patch("openagent.adaptive_engine.olmoe_cli._pack", return_value=0) as pack:
            self.assertEqual(main(["pack", "model", "--dtype", "int4-row"]), 0)
            self.assertEqual(pack.call_args.args[1:3], ("openagent-experts.i4", "openagent-experts.i4.manifest.json"))
            self.assertEqual(main(["pack", "model"]), 0)
            self.assertEqual(pack.call_args.args[1:3], ("openagent-experts.i8", "openagent-experts.manifest.json"))

    def test_float32_cli_default_paths_match_the_persisted_format(self):
        from openagent.adaptive_engine.olmoe_cli import main

        with patch("openagent.adaptive_engine.olmoe_cli._pack", return_value=0) as pack:
            self.assertEqual(main(["pack", "model", "--dtype", "float32"]), 0)
        self.assertEqual(pack.call_args.args[1:3],
                         ("openagent-experts.f32", "openagent-experts.f32.manifest.json"))

    def test_every_advertised_packed_dtype_has_default_artifact_paths(self):
        import argparse
        from openagent.adaptive_engine.olmoe_cli import main

        root_parser = build_parser()
        subparsers = next(action for action in root_parser._actions
                          if isinstance(action, argparse._SubParsersAction))
        pack_parser = subparsers.choices["pack"]
        dtype_action = next(action for action in pack_parser._actions if action.dest == "dtype")
        with patch("openagent.adaptive_engine.olmoe_cli._pack", return_value=0) as pack:
            for dtype in dtype_action.choices:
                with self.subTest(dtype=dtype):
                    self.assertEqual(main(["pack", "model", "--dtype", dtype]), 0)
                    output, manifest = pack.call_args.args[1:3]
                    self.assertTrue(output)
                    self.assertTrue(manifest)

    def test_thread_tuning_arguments_preserve_explicit_workload(self):
        args = build_parser().parse_args([
            "tune", "model", "manifest", "hello", "profile.json",
            "--candidates", "1", "2", "--max-tokens", "4",
            "--warmup", "2", "--runs", "3",
        ])
        self.assertEqual(args.candidates, [1, 2])
        self.assertEqual(args.profile, Path("profile.json"))
        self.assertEqual((args.max_tokens, args.warmup, args.runs), (4, 2, 3))

    def test_vectorized_kernel_and_thread_limit_are_explicit(self):
        args = build_parser().parse_args([
            "generate", "model", "manifest", "hello",
            "--kernel", "vectorized-cpu-numba", "--blas-threads", "1",
        ])
        self.assertEqual(args.kernel, "vectorized-cpu-numba")
        self.assertEqual(args.blas_threads, 1)

    def test_direct_io_is_accepted_by_all_execution_commands(self):
        parser = build_parser()
        for command, arguments in (
            ("generate", ["model", "manifest", "hello"]),
            ("benchmark", ["model", "manifest", "hello"]),
            ("verify", ["model", "manifest", "oracle"]),
        ):
            with self.subTest(command=command):
                parsed = parser.parse_args([command, *arguments, "--io-mode", "direct", "--kernel", "torch-expert"])
                self.assertEqual(parsed.io_mode, "direct")
                self.assertEqual(parsed.kernel, "torch-expert")

    def test_unavailable_torch_is_not_silently_replaced(self):
        with patch("openagent.adaptive_engine.torch_expert.TorchExpertBackend") as backend:
            backend.return_value.info.return_value.available = False
            backend.return_value.info.return_value.unavailable_reason = "no accelerator"
            with self.assertRaisesRegex(RuntimeError, "no accelerator"):
                _kernel_resources(Path("."), "torch-expert", 1)

    def test_cpu_test_kernel_is_not_a_user_execution_backend(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            _kernel_resources(Path("."), "torch-cpu-test", 1)

    def test_client_cleanup_runs_after_kernel_cleanup_failure(self):
        kernels, client = Mock(), Mock()
        kernels.close.side_effect = RuntimeError("cleanup failed")
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            _close_kernel_resources(kernels, client)
        client.close.assert_called_once()
