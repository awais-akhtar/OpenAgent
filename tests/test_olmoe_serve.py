from __future__ import annotations

import io
import json
import threading
from types import ModuleType
import unittest
from unittest.mock import patch, MagicMock

from openagent.adaptive_engine.olmoe_service import OlmoeService
from openagent.adaptive_engine.olmoe_serve import JsonlServer, build_parser, main


class Decoder:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def generate(self, messages, *, max_tokens, on_token, cancelled):
        self.calls.append(tuple(messages))
        self.started.set()
        on_token("hello")
        while not self.release.wait(0.005):
            if cancelled():
                return
        if not cancelled():
            on_token(" world")


class JsonlServerTests(unittest.TestCase):
    def setUp(self):
        self.decoder = Decoder()
        self.service = OlmoeService(self.decoder)
        self.output = io.StringIO()
        self.server = JsonlServer(self.service, self.output, max_requests=2)

    def tearDown(self):
        self.decoder.release.set()
        self.service.shutdown()
        for reader in self.server._readers.values():
            reader.join(2)
            self.assertFalse(reader.is_alive())

    def records(self):
        with self.server._output_lock:
            return [json.loads(line) for line in self.output.getvalue().splitlines()]

    def wait_readers(self):
        for reader in self.server._readers.values():
            reader.join(2)
            self.assertFalse(reader.is_alive())

    def test_input_remains_usable_and_cancel_streams(self):
        self.server.dispatch({"action": "submit", "id": 1, "request_id": "r", "prompt": "hi"})
        self.assertTrue(self.decoder.started.wait(1))
        self.server.dispatch({"action": "status", "id": 2})
        self.server.dispatch({"action": "cancel", "id": 3, "request_id": "r"})
        self.wait_readers()
        records = self.records()
        self.assertEqual(records[0]["request_id"], "r")
        self.assertEqual(records[0]["id"], 1)
        self.assertTrue(any(item.get("kind") == "token" for item in records))
        self.assertTrue(any(item.get("kind") == "cancelled" for item in records))
        status = next(item for item in records if item.get("id") == 2)
        self.assertEqual(status["result"]["execution_mode"], "single-decoder-worker")

    def test_capacity_duplicate_and_serial_decoding(self):
        for rid in ("a", "b", "c", "a"):
            self.server.dispatch({"action": "submit", "request_id": rid, "prompt": rid})
        self.assertTrue(self.decoder.started.wait(1))
        self.assertEqual(len(self.server._readers), 2)
        self.assertEqual(len(self.decoder.calls), 1)
        self.assertEqual(sum(item.get("ok") is False for item in self.records()), 2)
        self.decoder.release.set()
        self.wait_readers()
        self.server.dispatch({"action": "status"})
        self.assertEqual(self.server._readers, {})

    def test_session_history_clear_and_generated_ids(self):
        self.server.dispatch({"action": "session", "create": True, "session_id": "s",
                              "messages": [{"role": "system", "content": "be brief"}]})
        self.decoder.release.set()
        self.server.dispatch({"action": "submit", "session_id": "s", "prompt": "hi"})
        self.wait_readers()
        self.server.dispatch({"action": "session", "session_id": "s"})
        snapshot = self.records()[-1]["result"]
        self.assertEqual(snapshot["revision"], 1)
        self.assertEqual(snapshot["messages"][-1]["content"], "hello world")
        self.server.dispatch({"action": "clear", "session_id": "s"})
        self.assertTrue(self.records()[-1]["result"])

    def test_invalid_commands_do_not_stop_input(self):
        self.server.run(io.StringIO('oops\n[]\n{"action":"submit","prompt":"x","max_tokens":true}\n'
                                    '{"action":"unknown"}\n{"action":"status","id":"ok"}\n'
                                    '{"action":"shutdown"}\n'))
        records = self.records()
        self.assertEqual(sum(item.get("ok") is False for item in records), 4)
        self.assertTrue(next(item for item in records if item.get("id") == "ok")["ok"])
        self.assertFalse(self.service.telemetry()["worker_alive"])

    def test_oversized_line_drained_and_eof_cancels(self):
        server = JsonlServer(self.service, self.output, max_line_chars=32)
        server.run(io.StringIO('x' * 100 + '\n{"action":"status"}\n'))
        self.assertFalse(self.records()[0]["ok"])
        self.assertTrue(self.records()[1]["ok"])

    def test_run_reads_cancel_while_generation_active(self):
        decoder = self.decoder

        class Input:
            def __init__(self):
                self.lines = iter(['{"action":"submit","request_id":"r","prompt":"hi"}\n',
                                   '{"action":"cancel","request_id":"r"}\n',
                                   '{"action":"shutdown"}\n'])
                self.count = 0

            def readline(self, limit):
                self.count += 1
                if self.count == 2:
                    assert decoder.started.wait(1)
                return next(self.lines, "")

        self.server.run(Input())
        self.assertTrue(any(item.get("kind") == "cancelled" for item in self.records()))
        self.assertEqual(self.server._readers, {})

    def test_parser_accepts_registered_entry_arguments(self):
        args = build_parser().parse_args(['--model', 'model', '--manifest', 'manifest',
                                         '--kernel', 'numpy-reference', '--cache-gb', '1'])
        self.assertEqual(args.kernel, 'numpy-reference')

    def test_help_does_not_load_model(self):
        with patch('sys.stdout', io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(['--help'])
        self.assertEqual(caught.exception.code, 0)

    def test_continuous_cli_selects_batched_service(self):
        runtime = ModuleType('openagent.adaptive_engine.expert_runtime')
        runtime.SparseExpertManifest = MagicMock()
        model_module = ModuleType('openagent.adaptive_engine.olmoe')
        model_module.OlmoeModel = MagicMock()
        continuous = ModuleType('openagent.adaptive_engine.olmoe_continuous')
        continuous.ContinuousOlmoeService = MagicMock()
        with patch.dict('sys.modules', {runtime.__name__: runtime,
                                       model_module.__name__: model_module,
                                       continuous.__name__: continuous}), \
                patch('openagent.adaptive_engine.olmoe_serve.OlmoeChatDecoder'), \
                patch('openagent.adaptive_engine.olmoe_serve.JsonlServer'), \
                patch('openagent.adaptive_engine.olmoe_serve.OlmoeService') as serial:
            self.assertEqual(main(['--model', 'm', '--manifest', 'manifest',
                                   '--continuous', '--batch-size', '3',
                                   '--active-kv-mb', '512', '--prompt-chunk-tokens', '8']), 0)
        serial.assert_not_called()
        kwargs = continuous.ContinuousOlmoeService.call_args.kwargs
        self.assertEqual(kwargs['microbatch_size'], 3)
        self.assertEqual(kwargs['active_kv_capacity_bytes'], 512 * 1024**2)
        self.assertEqual(kwargs['prompt_chunk_tokens'], 8)

    def test_main_loads_resources_and_closes_kernel_client(self):
        runtime = ModuleType('openagent.adaptive_engine.expert_runtime')
        runtime.SparseExpertManifest = MagicMock()
        model_module = ModuleType('openagent.adaptive_engine.olmoe')
        model_module.OlmoeModel = MagicMock()
        cli = ModuleType('openagent.adaptive_engine.olmoe_cli')
        client = MagicMock()
        kernels = MagicMock()
        cli._kernel_resources = MagicMock(return_value=(kernels, client))
        modules = {runtime.__name__: runtime, model_module.__name__: model_module,
                   cli.__name__: cli}
        with patch.dict('sys.modules', modules), \
                patch('openagent.adaptive_engine.olmoe_serve.OlmoeChatDecoder'), \
                patch('openagent.adaptive_engine.olmoe_serve.OlmoeService') as service, \
                patch('sys.stdin', io.StringIO('{"action":"shutdown"}\n')), \
                patch('sys.stdout', io.StringIO()):
            self.assertEqual(main(['--model', 'm', '--manifest', 'manifest',
                                   '--cache-gb', '2', '--kernel', 'numpy-reference']), 0)
        kwargs = model_module.OlmoeModel.call_args.kwargs
        self.assertEqual(kwargs['expert_cache_bytes'], 2 * 1024**3)
        self.assertIs(kwargs['kernels'], kernels)
        self.assertEqual(kwargs['preferred_kernel'], 'numpy-reference')
        client.close.assert_called_once()
        kernels.close.assert_called_once()
        service.return_value.shutdown.assert_any_call(wait=True, cancel_pending=True)


if __name__ == "__main__":
    unittest.main()
