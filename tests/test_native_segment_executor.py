"""Compile real C (MSVC mandatory on Windows) and test math and raw ABI."""
import ctypes as C
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.native_segment_executor import (
    NativeSegmentExecutor, NativeExecutionError, RangeConfig, RangeLayer, Tensor, _tensor)
from openagent.adaptive_engine.olmoe import OlmoeModel, _rms_norm, _softmax
from openagent.adaptive_engine.expert_kernels import _decode_payload
from tests.test_olmoe_batch import checkpoint


def build_executor(directory):
    source = Path(__file__).resolve().parents[1] / "openagent/adaptive_engine/native/segment_executor.c"
    output = Path(directory) / ("executor.dll" if os.name == "nt" else "executor.so")
    if os.name == "nt":
        cl = shutil.which("cl")
        args = ["cl", "/nologo", "/W4", "/WX", "/O2", "/fp:precise", "/LD",
                str(source), "/link", "/OUT:" + str(output)]
        if cl:
            args[0] = cl
            command = args
        else:
            vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
            if not vswhere.is_file():
                raise unittest.SkipTest("MSVC tools unavailable")
            location = subprocess.check_output([str(vswhere), "-latest", "-products", "*",
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath"], text=True).strip()
            if not location:
                raise unittest.SkipTest("MSVC tools unavailable")
            bat = Path(location) / "VC/Auxiliary/Build/vcvars64.bat"
            command = 'cmd.exe /d /s /c ""' + str(bat) + '" && ' + subprocess.list2cmdline(args) + '"'
    else:
        compiler = next((shutil.which(n) for n in ("cc", "gcc", "clang") if shutil.which(n)), None)
        if not compiler:
            raise unittest.SkipTest("C compiler unavailable")
        command = [compiler, "-std=c99", "-Wall", "-Wextra", "-Werror", "-O2", "-fPIC",
                   "-shared", str(source), "-o", str(output), "-lm"]
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=120)
    if result.returncode or not output.is_file():
        raise RuntimeError("Executor compilation failed:\n" + result.stdout + result.stderr)
    return output


def fixture(heads=4, kv_heads=2, dim=2, intermediate=7, experts=3, layers=2, topk=2, normalize=True):
    h, kv = heads * dim, kv_heads * dim
    cfg = dict(hidden=h, intermediate=intermediate, experts=experts, heads=heads,
               kv_heads=kv_heads, topk=topk, capacity=12, normalize_topk=normalize,
               epsilon=1e-5, theta=10000.)
    rng = np.random.default_rng(983)
    weights = []
    for layer_index in range(layers):
        w = {name: rng.uniform(.7, 1.3, width).astype(np.float32) for name, width in
             (("input_norm", h), ("q_norm", h), ("k_norm", kv), ("post_norm", h))}
        shapes = dict(q=(h,h), k=(kv,h), v=(kv,h), o=(h,h), router=(experts,h),
                      gate=(experts,intermediate,h), up=(experts,intermediate,h),
                      down=(experts,h,intermediate))
        w.update({name: rng.normal(0,.15,shape).astype(np.float32) for name, shape in shapes.items()})
        if layer_index == 0:
            w["router"].fill(0)  # Stable ties matter: expert 0 then expert 1.
        weights.append(w)
    return cfg, weights


class Reference:
    def __init__(self, cfg, weights):
        self.cfg, self.weights = cfg, weights
        c = SimpleNamespace(hidden_size=cfg["hidden"], num_attention_heads=cfg["heads"],
            num_key_value_heads=cfg["kv_heads"], rms_norm_eps=cfg["epsilon"], rope_theta=cfg["theta"])
        shape = (cfg["kv_heads"], cfg["capacity"], cfg["hidden"] // cfg["heads"])
        self.state = SimpleNamespace(config=c, _k=[np.zeros(shape,np.float32) for _ in weights],
                                     _v=[np.zeros(shape,np.float32) for _ in weights])
        self.position = 0

    def forward(self, hidden):
        x = hidden.copy()
        for layer_index,w in enumerate(self.weights):
            z = _rms_norm(x,w["input_norm"],self.cfg["epsilon"])
            x += OlmoeModel._attention(self.state,layer_index,w,z,self.position)
            z = _rms_norm(x,w["post_norm"],self.cfg["epsilon"])
            probabilities = _softmax(z @ w["router"].T)
            ids = np.argsort(-probabilities,axis=1,kind="stable")[:,:self.cfg["topk"]]
            p = np.take_along_axis(probabilities,ids,axis=1)
            if self.cfg["normalize_topk"]:
                p /= p.sum(axis=1,keepdims=True)
            for r in range(len(x)):
                for e,weight in zip(ids[r],p[r]):
                    g,u = w["gate"][e] @ z[r], w["up"][e] @ z[r]
                    activation = g * np.exp(-np.logaddexp(0,-g)) * u
                    x[r] += weight * (w["down"][e] @ activation)
        self.position += len(x)
        return x


class NativeExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Loaded Windows DLL remains locked until process exit.
        cls.build_directory = tempfile.mkdtemp(prefix="oa-native-executor-")
        cls.library = build_executor(cls.build_directory)
        cls.executor = NativeSegmentExecutor(cls.library)

    def make(self, **kwargs):
        cfg, weights = fixture(**kwargs)
        engine = self.executor.create_range(weights, **cfg)
        return engine, engine.create_session(), Reference(cfg,weights)

    def test_full_range_prefill_decode_and_kv_differential(self):
        for heads,kv,dim in ((4,2,2),(3,1,4),(2,2,6),(1,1,2)):
            for normalize in (True,False):
                with self.subTest(heads=heads,kv=kv,dim=dim,normalize=normalize):
                    engine, session, ref = self.make(heads=heads,kv_heads=kv,dim=dim,normalize=normalize)
                    rng = np.random.default_rng(213)
                    for rows in (3,1,2,1):
                        x = rng.normal(size=(rows,engine.config.hidden)).astype(np.float32)
                        expected = ref.forward(x)
                        # Neither transformer nor NumPy expert reference is called by adapter.
                        with patch.object(OlmoeModel,"_attention",side_effect=AssertionError("callback")):
                            actual = session.forward(x)
                        np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=4e-5)
                        self.assertEqual(session.position,ref.position)
                        for layer_index in range(engine.config.layers):
                            np.testing.assert_allclose(session._keys[layer_index,:,:session.position],
                                ref.state._k[layer_index][:,:ref.position],atol=2e-6,rtol=3e-5)
                            np.testing.assert_allclose(session._values[layer_index,:,:session.position],
                                ref.state._v[layer_index][:,:ref.position],atol=2e-6,rtol=3e-5)

    def test_chunking_range_composition_reset_and_isolation(self):
        cfg, weights = fixture()
        engine = self.executor.create_range(weights, **cfg)
        a,b = engine.create_session(),engine.create_session()
        x = np.random.default_rng(33).normal(size=(7,cfg["hidden"])).astype(np.float32)
        expected = a.forward(x)
        actual = np.concatenate([b.forward(x[:2]),b.forward(x[2:3]),b.forward(x[3:])])
        np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-5)
        parts = [self.executor.create_range([w],**cfg).create_session() for w in weights]
        np.testing.assert_allclose(parts[1].forward(parts[0].forward(x)),expected,atol=2e-6,rtol=2e-5)
        a.reset()
        self.assertEqual(b.position,7)
        np.testing.assert_array_equal(a.forward(x),expected)
        a.close()
        with self.assertRaises(RuntimeError):
            a.forward(x)

    def test_edges(self):
        rng = np.random.default_rng(71)
        w = rng.normal(size=(13,8)).astype(np.float32)
        norm = rng.uniform(.7,1.3,8).astype(np.float32)
        ids = np.array([4,0,12,4],np.int64)
        np.testing.assert_array_equal(self.executor.embeddings(w,ids),w[ids])
        x = w[ids]
        np.testing.assert_allclose(self.executor.final_head(x,norm,w),
            _rms_norm(x,norm,1e-5) @ w.T,atol=3e-6,rtol=3e-5)
        with self.assertRaises(NativeExecutionError):
            self.executor.embeddings(w,np.array([-1],np.int64))

    def test_loaded_checkpoint_embed_range_head_vs_olmoe_model(self):
        for normalize in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = checkpoint(root, normalize)
                with OlmoeModel(root,manifest,expert_cache_bytes=16384) as model:
                    layers = []
                    for layer_index,dense in enumerate(model.layers):
                        w = dict(dense)
                        banks = [[],[],[]]
                        # Exactly the model's quantized weights, materialized once.
                        # This is NOT native packed INT8 execution or disk streaming.
                        for e in range(3):
                            layer_id = f"model.layers.{layer_index}"
                            region = manifest.layers[layer_index].regions[str(e)]
                            with model.experts.store.acquire(model.experts.key(layer_id,str(e))) as payload:
                                g,u,d,gs,us,ds = _decode_payload(memoryview(bytes(payload)),8,6,"int8-row",region.size)
                                for bank,matrix,scale in zip(banks,(g,u,d),(gs,us,ds)):
                                    bank.append(matrix.astype(np.float32)*scale[:,None])
                        for name,bank in zip(("gate","up","down"),banks):
                            w[name] = np.stack(bank)
                        layers.append(w)
                    cfg = dict(hidden=8,intermediate=6,experts=3,heads=4,kv_heads=2,
                               topk=2,capacity=10,normalize_topk=normalize)
                    engine = self.executor.create_range(layers,**cfg)
                    session = engine.create_session()
                    for ids in ([1,4,2],[7],[8,3],[0]):
                        x = self.executor.embeddings(model.embed,np.array(ids,np.int64))
                        x = session.forward(x)
                        logits = self.executor.final_head(x,model.final_norm,model.lm_head)
                        np.testing.assert_allclose(logits[-1],model.forward(ids),atol=3e-6,rtol=4e-5)
                        for layer_index in range(2):
                            np.testing.assert_allclose(session._keys[layer_index,:,:session.position],
                                model._k[layer_index][:,:model.position],atol=2e-6,rtol=3e-5)
                    self.assertEqual(engine.status()["resident_weight_bytes"],
                                     sum(a.nbytes for a in engine._weights))
                    self.assertFalse(engine.status()["sparse_streamed_experts"])

    def raw(self, engine, session, x, out, **kwargs):
        return self.executor.lib.oa_executor_range(
            kwargs.get("config",C.byref(engine.config)), kwargs.get("layers",engine._layers),
            kwargs.get("layer_count",engine.config.layers), kwargs.get("input",_tensor(x)),
            kwargs.get("rows",x.shape[0]), kwargs.get("output",out.ctypes.data),
            kwargs.get("output_count",out.size), kwargs.get("keys",session._keys.ctypes.data),
            session._values.ctypes.data, kwargs.get("cache_count",session._keys.size),
            kwargs.get("position",C.byref(session._position)), kwargs.get("limit",self.executor.max_bytes))

    def test_raw_malformed_envelopes_are_transactional(self):
        engine,s,_ = self.make()
        x = np.ones((2,8),np.float32)
        s.forward(x)
        before = (s._keys.copy(),s._values.copy(),s.position)
        out = np.full_like(x,123.)
        cases = [(1,dict(config=None)),(1,dict(layers=None)),(1,dict(output=None)),
                 (1,dict(keys=None)),(1,dict(position=None)),(1,dict(layer_count=1)),
                 (1,dict(input=Tensor(None,x.size))), (1,dict(input=Tensor(x.ctypes.data,x.size-1))),
                 (1,dict(output_count=out.size-1)),(1,dict(cache_count=s._keys.size-1)),
                 (1,dict(rows=0)),(5,dict(rows=(1<<64)-1)),(5,dict(limit=4))]
        for status,kwargs in cases:
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.raw(engine,s,x,out,**kwargs),status)
                np.testing.assert_array_equal(out,123.)
                np.testing.assert_array_equal(s._keys,before[0])
                np.testing.assert_array_equal(s._values,before[1])
                self.assertEqual(s.position,before[2])
        for field,value,status in (("version",99,2),("reserved",1,1),("heads",0,1),
                                  ("kv_heads",3,1),("topk",4,1),("normalize_topk",2,1),
                                  ("hidden",(1<<64)-8,5),("epsilon",float("nan"),1),
                                  ("theta",0.,1),("capacity",1,5)):
            c = RangeConfig.from_buffer_copy(engine.config)
            setattr(c,field,value)
            with self.subTest(field=field):
                self.assertEqual(self.raw(engine,s,x,out,config=C.byref(c)),status)
                np.testing.assert_array_equal(out,123.)
                np.testing.assert_array_equal(s._keys,before[0])
                self.assertEqual(s.position,before[2])

    def test_late_overflow_rolls_back_entire_range_and_output(self):
        cfg, weights = fixture()
        # Overflow only at the second layer's expert activation, after staged KV.
        weights[1]["gate"].fill(np.finfo(np.float32).max)
        weights[1]["up"].fill(np.finfo(np.float32).max)
        engine = self.executor.create_range(weights,**cfg)
        s = engine.create_session()
        x,out = np.ones((2,8),np.float32),np.full((2,8),77.,np.float32)
        self.assertEqual(self.raw(engine,s,x,out),1)
        np.testing.assert_array_equal(out,77.)
        np.testing.assert_array_equal(s._keys,0.)
        np.testing.assert_array_equal(s._values,0.)
        self.assertEqual(s.position,0)

    def test_all_weight_lengths_and_nonfinite_inputs(self):
        engine,s,_ = self.make()
        x,out = np.ones((1,8),np.float32),np.full((1,8),91.,np.float32)
        for name,_ in RangeLayer._fields_:
            layers = (RangeLayer * engine.config.layers).from_buffer_copy(engine._layers)
            getattr(layers[1],name).count -= 1
            self.assertEqual(self.raw(engine,s,x,out,layers=layers),1,name)
            np.testing.assert_array_equal(out,91.)
            self.assertEqual(s.position,0)
        x[0,0] = np.nan
        self.assertEqual(self.raw(engine,s,x,out),1)
        np.testing.assert_array_equal(out,91.)

    def test_raw_edges_rollback_and_prefix_validation(self):
        w = np.ones((3,8),np.float32)
        x = np.ones((2,8),np.float32)
        ids = np.array([0,3],np.int64)
        out = np.full_like(x,18.)
        self.assertEqual(self.executor.lib.oa_executor_embed(_tensor(w),3,8,ids.ctypes.data,
            2,out.ctypes.data,out.size,self.executor.max_bytes),1)
        np.testing.assert_array_equal(out,18.)
        w.fill(np.finfo(np.float32).max)
        norm = np.full(8,2.,np.float32)
        logits = np.full((2,3),19.,np.float32)
        self.assertEqual(self.executor.lib.oa_executor_head(_tensor(x),2,8,_tensor(norm),
            _tensor(w),3,1e-5,logits.ctypes.data,logits.size,self.executor.max_bytes),1)
        np.testing.assert_array_equal(logits,19.)
        engine,s,_ = self.make()
        s.forward(x)
        s._keys[1,0,0,0] = np.nan
        before = s._keys.copy()
        self.assertEqual(self.raw(engine,s,x,out),1)
        np.testing.assert_array_equal(s._keys,before)
        np.testing.assert_array_equal(out,18.)
        self.assertEqual(s.position,2)

    def test_scratch_aggregate_limit_and_future_cache_untouched(self):
        engine,s,_ = self.make()
        x,out = np.ones((1,8),np.float32),np.full((1,8),21.,np.float32)
        # Each tensor fits 800 bytes; full transactional scratch does not.
        self.assertEqual(self.raw(engine,s,x,out,limit=800),5)
        np.testing.assert_array_equal(out,21.)
        self.assertEqual(s.position,0)
        s._keys.fill(np.nan)
        s._values.fill(np.nan)
        s.forward(x)  # No initialized prefix is required at position zero.
        self.assertTrue(np.isnan(s._keys[:,:,1:]).all())
        self.assertTrue(np.isnan(s._values[:,:,1:]).all())

    def test_invalid_adapter_shapes_and_budget(self):
        engine,s,_ = self.make()
        for x in (np.ones((1,8),np.float64),np.ones(8,np.float32),np.ones((0,8),np.float32),
                  np.ones((1,7),np.float32)):
            with self.assertRaises(ValueError):
                s.forward(x)
        with self.assertRaises(NativeExecutionError):
            s.forward(np.ones((13,8),np.float32))
        self.assertEqual(s.position,0)

    def test_atomic_decode_raw_math_and_head_failure_rollback(self):
        engine,session,reference = self.make()
        rng = np.random.default_rng(514)
        embedding = rng.normal(size=(13,8)).astype(np.float32)
        head = rng.normal(size=(13,8)).astype(np.float32)
        final_norm = np.ones(8,np.float32)
        for tokens in ([1,3,7],[9],[0,5]):
            ids = np.array(tokens,np.int64)
            expected = _rms_norm(reference.forward(embedding[ids]),final_norm,1e-5) @ head.T
            out = np.full((len(ids),13),87.,np.float32)
            status = self.executor._decode(C.byref(engine.config),engine._layers,engine.config.layers,
                _tensor(embedding),_tensor(final_norm),_tensor(head),13,ids.ctypes.data,len(ids),
                out.ctypes.data,out.size,session._keys.ctypes.data,session._values.ctypes.data,
                session._keys.size,C.byref(session._position),self.executor.max_bytes)
            self.assertEqual(status,0)
            np.testing.assert_allclose(out,expected,atol=4e-6,rtol=4e-5)
        before = session._keys.copy(),session._values.copy(),session.position
        ids = np.array([4],np.int64)
        out = np.full((1,13),87.,np.float32)
        args = [C.byref(engine.config),engine._layers,engine.config.layers,_tensor(embedding),
            _tensor(final_norm),_tensor(head),13,ids.ctypes.data,1,out.ctypes.data,out.size,
            session._keys.ctypes.data,session._values.ctypes.data,session._keys.size,
            C.byref(session._position),self.executor.max_bytes]
        for index,value,status in ((4,Tensor(final_norm.ctypes.data,7),1),
                                   (9,None,1),(10,out.size-1,1),(12,None,1),
                                   (13,session._keys.size-1,1),(14,None,1),(15,4,5)):
            malformed = list(args)
            malformed[index] = value
            self.assertEqual(self.executor._decode(*malformed),status)
            np.testing.assert_array_equal(out,87.)
            np.testing.assert_array_equal(session._keys,before[0])
            np.testing.assert_array_equal(session._values,before[1])
            self.assertEqual(session.position,before[2])
        # Finite edge weights overflow only after a successful private range call.
        enormous_norm = np.full(8,np.finfo(np.float32).max,np.float32)
        enormous_head = np.full((13,8),np.finfo(np.float32).max,np.float32)
        args[4],args[5] = _tensor(enormous_norm),_tensor(enormous_head)
        self.assertEqual(self.executor._decode(*args),1)
        np.testing.assert_array_equal(out,87.)
        np.testing.assert_array_equal(session._keys,before[0])
        np.testing.assert_array_equal(session._values,before[1])
        self.assertEqual(session.position,before[2])

    def test_atomic_decode_aggregate_scratch_exact_boundary(self):
        engine,session,_ = self.make()
        c = engine.config
        ids = np.array([1],np.int64)
        embedding = np.ones((13,8),np.float32)
        norm = np.ones(8,np.float32)
        out = np.full((1,13),28.,np.float32)
        n,cache,logits = 8,session._keys.size,13
        kv = c.kv_heads*(c.hidden//c.heads)
        outer = 2*n+2*cache+logits
        nested = n+2*cache+4*c.hidden+2*kv+c.capacity+c.experts+2*c.intermediate
        required = 4*(outer+max(nested,logits+c.hidden))
        for budget,status in ((required-1,5),(required,0)):
            actual = self.executor._decode(C.byref(c),engine._layers,c.layers,_tensor(embedding),
                _tensor(norm),_tensor(embedding),13,ids.ctypes.data,1,out.ctypes.data,out.size,
                session._keys.ctypes.data,session._values.ctypes.data,cache,
                C.byref(session._position),budget)
            self.assertEqual(actual,status)
            if status:
                np.testing.assert_array_equal(out,28.)
                self.assertEqual(session.position,0)


if __name__ == "__main__":
    unittest.main()
