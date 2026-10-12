"""Independent Engram equations, native admission and transactional replay."""
import ctypes as C
import importlib.util
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("engram_fixture",
    ROOT / "openagent/adaptive_engine/native_deepseek_v41_engram.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(dim=5, branches=3):
    rng = random.Random(1729)
    heads = []
    for order in (2, 3, 4):
        for h in range(2):
            width, rows = h + 1, (7, 11)[h]
            heads.append(dict(order=order, width=width, rows=rows,
                seed=2**64 - 17 - order - h,
                multipliers=[2**64 - 5, 13 + h, 31 + order, 97],
                table=[rng.uniform(-2, 2) for _ in range(rows * width)]))
    memory = sum(h["width"] for h in heads)
    return dict(heads=heads, dim=dim, branches=branches, vocabulary=19, pad=18,
        key=[rng.uniform(-1, 1) for _ in range(branches * dim * memory)],
        value=[rng.uniform(-1, 1) for _ in range(dim * memory)],
        query_scale=[rng.uniform(.5, 1.5) for _ in range(branches * dim)],
        key_scale=[rng.uniform(.5, 1.5) for _ in range(branches * dim)],
        epsilon=1e-5, weight_budget=16 * 1024 * 1024)


def f32(v):
    return C.c_float(v).value


def scalar(config, tokens, hidden):
    """Specification from equations, not a call to the native implementation."""
    dim, branches = config["dim"], config["branches"]
    result, addresses = [], []
    for t in range(len(tokens)):
        memory = []
        for h in config["heads"]:
            context = [config["pad"] if p < 0 else tokens[p]
                       for p in range(t - h["order"] + 1, t + 1)]
            terms = [((token * multiplier) % 2**64)
                     for token, multiplier in zip(context, h["multipliers"])]
            hashed = h["seed"]
            for term in terms:
                hashed ^= term
            row = hashed % h["rows"]
            addresses.append(row)
            memory.extend(map(f32, h["table"][row*h["width"]:(row+1)*h["width"]]))
        m = len(memory)
        value = [f32(sum(f32(config["value"][j*m+z]) * memory[z] for z in range(m)))
                 for j in range(dim)]
        for b in range(branches):
            query = list(map(f32, hidden[(t*branches+b)*dim:(t*branches+b+1)*dim]))
            key = [f32(sum(f32(config["key"][(b*dim+j)*m+z]) * memory[z]
                           for z in range(m))) for j in range(dim)]
            qr = math.sqrt(sum(v*v for v in query)/dim + f32(config["epsilon"]))
            kr = math.sqrt(sum(v*v for v in key)/dim + f32(config["epsilon"]))
            logit = sum(query[j]/qr * f32(config["query_scale"][b*dim+j]) *
                key[j]/kr * f32(config["key_scale"][b*dim+j]) for j in range(dim))/math.sqrt(dim)
            gate = 1/(1+math.exp(-logit))
            result.extend(f32(gate*v) for v in value)
    return tuple(result), tuple(addresses)


class EngramEquations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        source = ROOT / "openagent/adaptive_engine/native_deepseek_v41_engram.c"
        cls.output = Path(cls.directory.name) / ("engram.dll" if os.name == "nt" else "engram.so")
        if os.name == "nt":
            vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
            location = subprocess.check_output([str(vswhere), "-latest", "-products", "*", "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()
            env = Path(location) / "VC/Auxiliary/Build/vcvars64.bat"
            args = ["cl", "/nologo", "/W4", "/WX", "/O2", "/fp:strict", "/LD", str(source),
                    "/link", "/OUT:" + str(cls.output)]
            command = 'cmd.exe /d /s /c ""' + str(env) + '" && ' + subprocess.list2cmdline(args) + '"'
        else:
            command = [shutil.which("cc") or "cc", "-std=c99", "-Wall", "-Wextra", "-Werror",
                       "-O2", "-shared", "-fPIC", str(source), "-o", str(cls.output), "-lm"]
        result = subprocess.run(command, cwd=cls.directory.name, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def model(self, **changes):
        config = fixture()
        config.update(changes)
        model = module.NativeDeepSeekV41Engram(self.output, **config)
        self.addCleanup(model.close)
        return model, config

    def assertNear(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertAlmostEqual(a, e, delta=2e-6 * max(1, abs(e)))

    def test_scalar_orders_hash_heads_padding_branches(self):
        model, config = self.model()
        tokens = [0, 1, 7, 2, 18, 3, 4, 4, 2]
        x = [math.sin(i) for i in range(len(tokens)*15)]
        actual, rows = model.forward(tokens, x)
        expected, expected_rows = scalar(config, tokens, x)
        self.assertEqual(rows, expected_rows)
        self.assertNear(actual, expected)
        self.assertNotEqual(actual[:5], actual[5:10])

    def test_numpy_independent_projection_and_gate(self):
        import numpy as np
        model, config = self.model()
        tokens = [1, 3, 5, 7, 9]
        x = np.arange(75, dtype=np.float32).reshape(5, 3, 5)/37 - 1
        # Construct addresses independently using vectorized unsigned multiplication/XOR.
        memories, rows = [], []
        padded = [config["pad"]]*3 + tokens
        for t in range(5):
            memory = []
            for h in config["heads"]:
                context = np.array(padded[t+4-h["order"]:t+4], dtype=np.uint64)
                terms = context * np.array(h["multipliers"][:h["order"]], dtype=np.uint64)
                hashed = np.bitwise_xor.reduce(terms, initial=np.uint64(h["seed"]))
                row = int(hashed % np.uint64(h["rows"]))
                rows.append(row)
                memory.extend(h["table"][row*h["width"]:(row+1)*h["width"]])
            memories.append(memory)
        memory = np.asarray(memories, dtype=np.float32).astype(np.float64)
        k = np.asarray(config["key"], dtype=np.float32).astype(np.float64).reshape(3, 5, 9)
        v = np.asarray(config["value"], dtype=np.float32).astype(np.float64).reshape(5, 9)
        keys = np.einsum("bdm,tm->tbd", k, memory).astype(np.float32).astype(np.float64)
        values = (memory @ v.T).astype(np.float32).astype(np.float64)
        q = x.astype(np.float64)
        eps = f32(config["epsilon"])
        q /= np.sqrt(np.mean(q*q, axis=-1, keepdims=True)+eps)
        keys /= np.sqrt(np.mean(keys*keys, axis=-1, keepdims=True)+eps)
        qs = np.asarray(config["query_scale"], dtype=np.float32).reshape(3, 5)
        ks = np.asarray(config["key_scale"], dtype=np.float32).reshape(3, 5)
        gates = 1/(1+np.exp(-np.sum(q*qs*keys*ks, axis=-1)/np.sqrt(5)))
        expected = (gates[..., None]*values[:, None, :]).astype(np.float32).ravel()
        actual, addresses = model.forward(tokens, x.ravel())
        self.assertEqual(addresses, tuple(rows))
        self.assertNear(actual, expected)

    def test_full_all_chunk_boundaries_and_rollback(self):
        model, _ = self.model()
        tokens = [0, 1, 4, 3, 4, 2, 8]
        x = [math.cos(i) for i in range(105)]
        expected = model.forward(tokens, x)
        for split in range(8):
            session = model.session()
            a = session.forward(tokens[:split], x[:split*15])
            b = session.forward(tokens[split:], x[split*15:])
            self.assertEqual((a[0]+b[0], a[1]+b[1]), expected)
            session.rollback(split)
            self.assertEqual(session.length, split)
            self.assertEqual(session.forward(tokens[split:], x[split*15:]), b)
        session = model.session()
        session.forward(tokens, x)
        session.rollback(2)
        new = session.forward([11, 12], x[:30])
        whole = model.forward(tokens[:2]+[11, 12], x[:30]+x[:30])
        self.assertEqual(new, (whole[0][30:], whole[1][12:]))

    def raw(self, model, **changes):
        state = module.State()
        check = model.library.oa_v41_engram_state_init
        self.assertEqual(check(C.byref(state), 10), 0)
        tokens = (C.c_uint32*2)(1, 2)
        x = (C.c_float*30)(*([1]*30))
        out = (C.c_float*30)(*([-77]*30))
        ids = (C.c_uint32*12)(*([99]*12))
        args = dict(c=C.byref(model._config), s=C.byref(state), tokens=tokens, nt=2,
                    hidden=x, nh=30, out=out, no=30, indices=ids, ni=12, budget=10000)
        args.update(changes)
        before = bytes(state), bytes(out), bytes(ids)
        status = model.library.oa_v41_engram_run(*args.values())
        self.assertEqual((bytes(state), bytes(out), bytes(ids)), before)
        return status

    def test_native_failure_atomicity_and_aliases(self):
        model, _ = self.model()
        for changes, status in [({"budget":0}, 2), ({"no":29}, 1), ({"ni":11}, 1),
            ({"nh":29}, 1), ({"nt":257}, 1), ({"tokens":None}, 1),
            ({"tokens":(C.c_uint32*2)(1, 19)}, 1),
            ({"hidden":(C.c_float*30)(*([float("nan")]*30))}, 3)]:
            self.assertEqual(self.raw(model, **changes), status)
        x = (C.c_float*30)(*([1]*30))
        self.assertEqual(self.raw(model, hidden=x, out=x), 1)
        model._config.weight_budget = 0
        self.assertEqual(self.raw(model), 2)

    def test_late_nonfinite_table_or_overflow_does_not_commit(self):
        model, config = self.model()
        session = model.session()
        _, rows = model.forward([1, 2], [1]*30)
        table = model._storage[0]
        table[rows[6]*config["heads"][0]["width"]] = float("inf")
        with self.assertRaises(ValueError):
            session.forward([1, 2], [1]*30)
        self.assertEqual(session.length, 0)
        model, _ = self.model(value=[3e38]*45)
        self.assertEqual(self.raw(model), 3)

    def test_state_capacity_empty_calls_and_close(self):
        model, _ = self.model()
        session = model.session(2)
        self.assertEqual(session.forward([], []), ((), ()))
        session.forward([1, 2], [1]*30)
        before = bytes(session._state)
        with self.assertRaises(ValueError):
            session.forward([3], [1]*15)
        self.assertEqual(bytes(session._state), before)
        with self.assertRaises(ValueError):
            session.rollback(3)
        session.rollback(0)
        model.close()
        with self.assertRaises(RuntimeError):
            session.forward([], [])

    def test_wrapper_bounds_and_owned_weight_copies(self):
        model, config = self.model()
        expected = model.forward([1], [0]*15)
        config["heads"][0]["table"][:] = [999]*14
        self.assertEqual(model.forward([1], [0]*15), expected)
        for changes in ({"dim":257}, {"branches":9}, {"pad":19}, {"epsilon":0},
                        {"weight_budget":1}, {"heads":[]}, {"key":[1]}):
            with self.assertRaises(ValueError):
                self.model(**changes)
        for tokens in ([True], [-1], [2**32], [1]*257):
            with self.assertRaises((ValueError, TypeError)):
                model.forward(tokens, [])

    def test_zero_query_half_gate_and_context_changes(self):
        model, config = self.model()
        zero = model.forward([1, 2], [0]*30)[0]
        self.assertNear(zero, scalar(config, [1, 2], [0]*30)[0])
        self.assertNotEqual(model.forward([1, 2], [1]*30)[0], zero)
        self.assertNotEqual(model.forward([1, 2], [-1]*30)[0], zero)

    def test_exact_staging_budget_and_readonly_native_weights(self):
        model, _ = self.model()
        before = tuple(bytes(v) for v in model._storage)
        # 30 output + 9 memory + 10 key/value floats; 12 uint32 addresses.
        with self.assertRaises(ValueError):
            model.forward([1, 2], [1]*30, budget=243)
        model.forward([1, 2], [1]*30, budget=244)
        self.assertEqual(tuple(bytes(v) for v in model._storage), before)

    def test_max_heads_width_memory_and_hidden_branch_bounds(self):
        for dim, branches, width, count in ((1, 1, 256, 24), (256, 8, 1, 3)):
            heads = [dict(order=2+i%3, width=width, rows=1, seed=i,
                     multipliers=[1, 2, 3, 4], table=[1]*width) for i in range(count)]
            m = width*count
            model, _ = self.model(heads=heads, dim=dim, branches=branches,
                key=[0]*(branches*dim*m), value=[1/m]*(dim*m),
                query_scale=[1]*(branches*dim), key_scale=[1]*(branches*dim))
            y, rows = model.forward([1], [0]*(dim*branches))
            self.assertNear(y, [.5]*(dim*branches))
            self.assertEqual(rows, (0,)*count)

    def test_native_malformed_geometry_state_and_norm(self):
        for field, bad in (("heads",25), ("dim",257), ("branches",9),
                           ("pad",19), ("reserved",1), ("epsilon",float("nan")),
                           ("key_count",1)):
            model, _ = self.model()
            setattr(model._config, field, bad)
            self.assertEqual(self.raw(model), 1)
        model, _ = self.model()
        for field, bad in (("order",1), ("width",257), ("rows",33554433), ("count",1)):
            h = model._config.head[0]
            previous = getattr(h, field)
            setattr(h, field, bad)
            self.assertEqual(self.raw(model), 1)
            setattr(h, field, previous)
        session = model.session()
        session._state.length = 4097
        before = bytes(session._state)
        with self.assertRaises(ValueError):
            session.forward([], [])
        self.assertEqual(bytes(session._state), before)

    def test_bounded_4096_history_and_256_chunk(self):
        heads = [dict(order=n, width=1, rows=1, seed=n,
                      multipliers=[1, 2, 3, 4], table=[1]) for n in (2, 3, 4)]
        model, _ = self.model(heads=heads, dim=1, branches=1, key=[0]*3,
            value=[1]*3, query_scale=[1], key_scale=[1])
        session = model.session()
        expected = model.forward([1]*256, [1]*256)
        for _ in range(16):
            self.assertEqual(session.forward([1]*256, [1]*256), expected)
        self.assertEqual(session.length, 4096)
        with self.assertRaises(ValueError):
            session.forward([1], [1])
        session.rollback(4095)
        self.assertEqual(session.forward([1], [1]), ((1.5,), (0, 0, 0)))

    def test_reentrant_close_during_packing_never_calls_unloaded_library(self):
        model, _ = self.model()
        session = model.session()
        def tokens():
            model.close()
            yield 1
        with self.assertRaises(RuntimeError):
            session.forward(tokens(), [1]*15)
        self.assertEqual(session.length, 0)


if __name__ == "__main__":
    unittest.main()
