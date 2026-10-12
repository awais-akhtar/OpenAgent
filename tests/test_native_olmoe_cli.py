"""Real compiled operator execution and preflight rejection."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.native_olmoe_cli import main
from openagent.adaptive_engine.olmoe import OlmoeModel
from tests.test_native_segment_executor import build_executor
from tests.test_olmoe_batch import checkpoint


class NativeOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiled_directory = tempfile.mkdtemp(prefix="oa-native-operator-")
        cls.library = build_executor(cls.compiled_directory)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.manifest = checkpoint(self.root, True)
        self.path = self.manifest.write(self.root / "manifest.json")
        self.args = [str(self.root), str(self.path), "--library", str(self.library),
            "--context-tokens", "8", "--cache-mib", "1", "--resident-mib", "1",
            "--transfer-mib", "1", "--staging-mib", "1"]

    def invoke(self, args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(args)
        return code, out.getvalue(), err.getvalue()

    def test_real_native_generation_matches_independent_numpy_model(self):
        prompt = [1, 3]
        with OlmoeModel(self.root, self.manifest, context_tokens=8, expert_cache_bytes=1024**2) as model:
            pending, expected = prompt, []
            for _ in range(2):
                token = int(np.argmax(model.forward(pending)))
                expected.append(token)
                pending = [token]
        oracle = self.root / "oracle.json"
        oracle.write_text(json.dumps(dict(prompt_ids=prompt, full_ids=prompt+expected)), encoding="utf-8")
        with patch.object(OlmoeModel, "forward", side_effect=AssertionError("no NumPy forward")):
            code, out, err = self.invoke(self.args+["--oracle", str(oracle)])
        self.assertEqual(code, 0, err)
        result = json.loads(out)
        self.assertEqual(result["generated_ids"], expected)
        self.assertTrue(result["matched"])
        self.assertTrue(result["status"]["sparse_streamed_experts"])
        self.assertFalse(result["upstream_abi_compatible"])

    def test_invalid_prompt_or_oracle_is_rejected_before_native_loading(self):
        invalid = ("[]", "[true]", "[-1]", "[1.2]", "[999999]", "{}")
        with patch("openagent.adaptive_engine.native_olmoe_cli.NativeLeasedOlmoeSegmentEngine") as engine:
            for payload in invalid:
                with self.subTest(payload=payload):
                    code, out, err = self.invoke(self.args+["--token-ids", payload])
                    self.assertEqual(code, 1)
                    self.assertFalse(out)
                    self.assertTrue(err)
            engine.assert_not_called()

    def test_malformed_oracle_objects_and_prefixes_are_rejected(self):
        path = self.root / "bad-oracle.json"
        with patch("openagent.adaptive_engine.native_olmoe_cli.NativeLeasedOlmoeSegmentEngine") as engine:
            for value in (None, "oracle", [], {}, {"prompt_ids": [1], "full_ids": [2, 3]},
                          {"prompt_ids": [1], "full_ids": [1]}):
                with self.subTest(value=value):
                    path.write_text(json.dumps(value), encoding="utf-8")
                    code, out, err = self.invoke(self.args+["--oracle", str(path)])
                    self.assertEqual(code, 1)
                    self.assertFalse(out)
                    self.assertTrue(err)
            engine.assert_not_called()

    def test_invalid_context_and_missing_library_have_no_fallback(self):
        for tail in (["--context-tokens", "999999", "--token-ids", "[1]"],
                     ["--library", str(self.root/"missing.dll"), "--token-ids", "[1]", "--max-tokens", "1"]):
            code, out, err = self.invoke(self.args+tail)
            self.assertEqual(code, 1)
            self.assertFalse(out)
            self.assertTrue(err)
