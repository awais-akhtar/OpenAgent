"""Actual compiled CED logits, ownership and atomic replay equation checks."""
import ctypes as C
from dataclasses import replace
import gc
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from openagent.adaptive_engine.native_deepseek_v41_ced import (
    CEDSnapshot, DeepSeekV41CEDConfig, NativeDeepSeekV41CED, ced_tensor_shapes,
)


def build(directory):
    source = Path(__file__).resolve().parents[1] / 'openagent/adaptive_engine/native_deepseek_v41_ced.c'
    output = Path(directory) / ('ced.dll' if os.name == 'nt' else 'ced.so')
    if os.name == 'nt':
        finder = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)')) / 'Microsoft Visual Studio/Installer/vswhere.exe'
        location = subprocess.check_output([str(finder), '-latest', '-products', '*',
            '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
            '-property', 'installationPath'], text=True).strip()
        if not location:
            raise RuntimeError('MSVC is required')
        setup = Path(location) / 'VC/Auxiliary/Build/vcvars64.bat'
        args = ['cl', '/nologo', '/W4', '/WX', '/O2', '/fp:precise', '/LD',
                str(source), '/link', '/OUT:' + str(output)]
        command = 'cmd.exe /d /s /c ""' + str(setup) + '" && ' + subprocess.list2cmdline(args) + '"'
    else:
        compiler = shutil.which('cc')
        if not compiler:
            raise RuntimeError('C compiler is required')
        command = [compiler, '-std=c99', '-Wall', '-Wextra', '-Werror', '-O2',
            '-shared', '-fPIC', str(source), '-o', str(output), '-lm']
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return output


def fixture(config):
    rng = np.random.default_rng(4141)
    result = {name: rng.normal(0, .25, shape).astype(np.float32)
              for name, shape in ced_tensor_shapes(config).items()}
    for name in result:
        if name.endswith('.norm'):
            result[name] = np.linspace(.8, 1.2, config.hidden, dtype=np.float32)
    return result


def analytical(c, weights, tokens, *, incorrect_decoder_source=False):
    h, length = c.hidden, len(tokens)
    local_k = np.zeros((c.layers, length, h), np.float32)
    local_v, raw, zraw = local_k.copy(), local_k.copy(), local_k.copy()
    encoder = np.zeros((length, h), np.float32)
    outputs, trace = [], []

    def norm(x, gain):
        a = x.astype(np.float64)
        return (a / np.sqrt(np.mean(a*a) + c.epsilon) * gain).astype(np.float32)

    def matrix(w, x):
        return (w.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)

    for t, token in enumerate(tokens):
        x = weights['embedding.weight'][token].copy()
        owner, main, keys, selected, candidates = -1, None, None, [], []
        trace = []
        for layer, mode in enumerate(c.modes):
            if layer == c.encoder:
                owner, main, keys, selected, candidates = -1, None, None, [], []
            w = {name[len(f'layers.{layer}.'):]: value for name, value in weights.items()
                 if name.startswith(f'layers.{layer}.')}
            a = norm(x, w['attention.norm'])
            q = matrix(w['attention.q'], a)
            local_k[layer, t] = matrix(w['attention.k'], a)
            local_v[layer, t] = matrix(w['attention.v'], a)
            if mode != 'swa':
                gq = matrix(w['global.q'], a)
                if mode == 'full':
                    source = x if layer < c.encoder or incorrect_decoder_source else encoder[t]
                    raw[layer, t] = matrix(w['global.kv'], source)
                    zraw[layer, t] = matrix(w['global.z'], source)
                    main = []
                    for group in range((t+1)//c.ratio):
                        z = zraw[layer, group*c.ratio:(group+1)*c.ratio].astype(np.float64)
                        p = np.exp(z - z.max(axis=0))
                        p /= p.sum(axis=0)
                        main.append((p * raw[layer, group*c.ratio:(group+1)*c.ratio]).sum(axis=0).astype(np.float32))
                    main = np.array(main, np.float32).reshape(-1, h)
                    keys = np.array([matrix(w['index.k'], v) for v in main], np.float32).reshape(-1, h)
                    owner = layer
                if mode != 'reuse':
                    iq = matrix(w['index.q'], a).astype(np.float64)
                    scores = np.maximum(keys.astype(np.float64) @ iq, 0)
                    eligible = [r for r in range(len(main)) if layer < c.encoder or mode == 'full'
                                or r//c.block_size in candidates]
                    selected = sorted(eligible, key=lambda r: (-scores[r], r))[:c.topk]
                    if layer >= c.encoder and mode == 'full':
                        block_scores = [max(scores[b:b+c.block_size])
                                        for b in range(0, len(main), c.block_size)]
                        candidates = sorted(range(len(block_scores)),
                            key=lambda b: (-block_scores[b], b))[:c.candidate_blocks]
            first = max(0, t+1-c.window)
            scores = list(local_k[layer, first:t+1].astype(np.float64) @ q.astype(np.float64))
            values = list(local_v[layer, first:t+1])
            if mode != 'swa':
                scores += list(main[selected].astype(np.float64) @ gq.astype(np.float64))
                values += list(main[selected])
            scores = np.array(scores) / np.sqrt(h)
            probabilities = np.exp(scores - scores.max())
            probabilities /= probabilities.sum()
            joined = (probabilities[:, None] * np.array(values)).sum(axis=0).astype(np.float32)
            x = (x.astype(np.float64) + matrix(w['attention.out'], joined)).astype(np.float32)
            a = norm(x, w['mlp.norm'])
            gate = matrix(w['mlp.gate'], a).astype(np.float64)
            up = matrix(w['mlp.up'], a).astype(np.float64)
            product = (gate / (1 + np.exp(-gate)) * up).astype(np.float32)
            x = (x.astype(np.float64) + matrix(w['mlp.down'], product)).astype(np.float32)
            if layer+1 == c.encoder:
                encoder[t] = x
            trace.append((owner if mode != 'swa' else -1, len(selected) if mode != 'swa' else 0,
                          *((selected + [-1]*c.topk)[:c.topk] if mode != 'swa' else [-1]*c.topk)))
        outputs.append(matrix(weights['head.weight'], norm(x, weights['final.norm'])))
    return np.array(outputs), tuple(trace)


class NativeCEDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.library = build(cls.directory.name)
        cls.addClassCleanup(gc.collect)

    def model(self, c=None, weights=None):
        c = c or DeepSeekV41CEDConfig()
        return NativeDeepSeekV41CED(c, weights if weights is not None else fixture(c), library=self.library)

    def test_full_logits_compression_and_cross_layer_selection(self):
        ids = [1, 5, 7, 2, 9, 3, 4]
        for ratio in (1, 2, 4):
            c = replace(DeepSeekV41CEDConfig(), ratio=ratio, candidate_blocks=1)
            w = fixture(c)
            model = self.model(c, w)
            expected, trace = analytical(c, w, ids)
            np.testing.assert_allclose(model.forward(ids), expected, atol=2e-5, rtol=2e-5)
            self.assertEqual(model.trace, trace)
            self.assertEqual(model.trace[2][0], 1)
            self.assertEqual(model.trace[4][0], 3)
            self.assertEqual(model.trace[5], model.trace[4])

    def test_encoder_source_is_not_decoder_hidden(self):
        c = DeepSeekV41CEDConfig(ratio=1,
            modes=('swa', 'full', 'reuse', 'swa', 'full', 'reuse'))
        w, ids = fixture(c), [1, 5, 7, 2]
        correct, _ = analytical(c, w, ids)
        incorrect, _ = analytical(c, w, ids, incorrect_decoder_source=True)
        self.assertGreater(np.max(np.abs(correct-incorrect)), .001)
        np.testing.assert_allclose(self.model(c, w).forward(ids), correct, atol=2e-5, rtol=2e-5)

    def test_chunks_snapshot_and_detached_configuration(self):
        model, ids = self.model(), [1, 5, 7, 2, 9, 3, 4]
        full = model.forward(ids)
        model.reset()
        chunks = [model.forward(ids[:2])]
        saved = model.snapshot()
        chunks.extend([model.forward(ids[2:3]), model.forward(ids[3:])])
        np.testing.assert_array_equal(np.concatenate(chunks), full)
        model.restore(saved)
        np.testing.assert_array_equal(model.forward(ids[2:]), full[2:])
        self.assertEqual(saved.tokens, (1, 5))
        with self.assertRaises(AttributeError):
            model.config = replace(model.config, hidden=1)
        with self.assertRaises(ValueError):
            model.weights[0] = 0

    def test_failure_capacity_restore_and_budget_rollback(self):
        model = self.model()
        model.forward([1, 5])
        before, trace = model.snapshot(), model.trace
        with self.assertRaisesRegex(ValueError, 'state unchanged'):
            model.forward([7, 2], fail_layer=5)
        self.assertEqual(model.snapshot(), before)
        self.assertEqual(model.trace, trace)
        for bad in ([True], [-1], [999], [0]*33, []):
            with self.assertRaises(ValueError):
                model.forward(bad)
            self.assertEqual(model.snapshot(), before)
        for bad in (CEDSnapshot('bad', (1,)), CEDSnapshot(before.identity, (999,)),
                    CEDSnapshot(before.identity, [1])):
            with self.assertRaises(ValueError):
                model.restore(bad)
            self.assertEqual(model.snapshot(), before)
        model.max_bytes = 1
        with self.assertRaisesRegex(ValueError, 'budget'):
            model.forward([7])
        self.assertEqual(model.snapshot(), before)

    def test_raw_native_outputs_remain_unchanged_on_late_failure(self):
        model = self.model()
        ids = np.array([1, 5], np.int32)
        output = np.full((2, model.config.vocabulary), -99, np.float32)
        trace = np.full((model.config.layers, 2+model.config.topk), -33, np.int32)
        status = model._run(C.byref(model._c), model._modes.ctypes.data, model._modes.size,
            model.weights.ctypes.data, model.weights.size, ids.ctypes.data, ids.size,
            output.ctypes.data, output.size, trace.ctypes.data, trace.size, model._admit(2), 5)
        self.assertEqual(status, 5)
        np.testing.assert_array_equal(output, -99)
        np.testing.assert_array_equal(trace, -33)

    def test_geometry_weights_and_phase_modes_fail_closed(self):
        for values in ({'hidden': True}, {'encoder': 6}, {'window': 33}, {'ratio': 0},
                       {'modes': ('reuse',)*6}, {'epsilon': np.nan}):
            with self.assertRaises(ValueError):
                replace(DeepSeekV41CEDConfig(), **values)
        c, w = DeepSeekV41CEDConfig(), fixture(DeepSeekV41CEDConfig())
        w['head.weight'][0, 0] = np.inf
        with self.assertRaises(ValueError):
            self.model(c, w)
        with self.assertRaises(ValueError):
            self.model(c, {})

    def test_reentrant_mutations_during_ingestion_preserve_history_and_trace(self):
        model = self.model()
        model.forward([1, 5])
        before, trace = model.snapshot(), model.trace
        actions = (model.reset, lambda: model.restore(before), lambda: model.forward([7]))
        for action in actions:
            for during_iter in (False, True):
                with self.subTest(action=action, during_iter=during_iter):
                    class Input:
                        def __iter__(self):
                            if during_iter:
                                action()
                            return self.values()

                        def values(self):
                            action()
                            yield 999

                    with self.assertRaisesRegex(RuntimeError, 'reentrant CED'):
                        model.forward(Input())
                    self.assertEqual(model.snapshot(), before)
                    self.assertEqual(model.trace, trace)
                    self.assertFalse(model._active_mutation)
        expected = self.model().forward([1, 5, 7])[-1:]
        np.testing.assert_array_equal(model.forward([7]), expected)
        model.restore(before)
        model.reset()
        self.assertEqual(model.length, 0)

    def test_handled_reentrant_attempt_does_not_disrupt_outer_forward(self):
        model = self.model()
        model.forward([1, 5])
        before = model.snapshot()

        def values():
            self.assertEqual(model.snapshot(), before)
            for action in (model.reset, lambda: model.restore(before), lambda: model.forward([9])):
                with self.assertRaisesRegex(RuntimeError, 'reentrant CED'):
                    action()
            yield 7

        expected = self.model().forward([1, 5, 7])[-1:]
        np.testing.assert_array_equal(model.forward(values()), expected)
        self.assertEqual(model.snapshot().tokens, (1, 5, 7))

    def test_native_config_alignment_range_and_null_rejected_before_access(self):
        model = self.model()
        config_type = type(model._c)
        ids = np.array([1], np.int32)
        output = np.full(model.config.vocabulary, -99, np.float32)
        trace = np.full((model.config.layers, 2+model.config.topk), -33, np.int32)
        before = bytes(model._c)
        wrapping = np.iinfo(np.uintp).max - C.alignment(config_type) + 1
        pointers = (0, C.addressof(model._c)+1, wrapping)
        for address in pointers:
            with self.subTest(address=address):
                config = C.cast(C.c_void_p(address), C.POINTER(config_type))
                status = model._run(config, model._modes.ctypes.data, model._modes.size,
                    model.weights.ctypes.data, model.weights.size, ids.ctypes.data, ids.size,
                    output.ctypes.data, output.size, trace.ctypes.data, trace.size, model._admit(1), -1)
                self.assertEqual(status, 1)
                self.assertEqual(bytes(model._c), before)
                np.testing.assert_array_equal(output, -99)
                np.testing.assert_array_equal(trace, -33)

    def test_native_config_output_and_trace_full_partial_overlap_rejected(self):
        c = DeepSeekV41CEDConfig(hidden=1, layers=2, encoder=1, vocabulary=2,
            capacity=1, window=1, ratio=1, topk=1, block_size=1, candidate_blocks=1,
            intermediate=1, modes=('full', 'full'))
        model = self.model(c)
        config_type = type(model._c)
        ids = np.array([1], np.int32)
        output = np.full(2, -99, np.float32)
        trace = np.full(6, -33, np.int32)
        backing = (C.c_uint64*32)()
        address = C.addressof(backing)+64
        C.memmove(address, C.byref(model._c), C.sizeof(config_type))
        config = C.cast(C.c_void_p(address), C.POINTER(config_type))
        before = bytes(backing)
        aliases = ((address, trace.ctypes.data), (output.ctypes.data, address),
                   (address-4, trace.ctypes.data),
                   (output.ctypes.data, address+C.sizeof(config_type)-4))
        for output_address, trace_address in aliases:
            with self.subTest(output=output_address, trace=trace_address):
                status = model._run(config, model._modes.ctypes.data, model._modes.size,
                    model.weights.ctypes.data, model.weights.size, ids.ctypes.data, ids.size,
                    output_address, output.size, trace_address, trace.size, model._admit(1), -1)
                self.assertEqual(status, 1)
                self.assertEqual(bytes(backing), before)
                np.testing.assert_array_equal(output, -99)
                np.testing.assert_array_equal(trace, -33)
