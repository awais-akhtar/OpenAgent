"""Original FP32 K3 composition; does not import or execute vendor code."""
import copy
import ctypes as C
import _ctypes
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
import threading
import weakref

import numpy as np

from ..native_kimi import NativeKimiKDA
from ..native_mxfp4 import NativeMXFP4
from ..native_tensor_ops import NativeTensorOps
from .cache import MLALatentPages
from .vision import K3Vision, K3VisionConfig, vision_tensor_shapes
from .preprocessing import preprocess_images
from .vision_ops import VisionOps

REVISION = "f831ab66814297da540d832a5235f8e904f29d06"


def _release_libraries(libraries):
    release = _ctypes.FreeLibrary if hasattr(_ctypes, "FreeLibrary") else _ctypes.dlclose
    while libraries:
        release(libraries[-1]._handle);libraries.pop()


@dataclass(frozen=True)
class K3Config:
    hidden: int
    vocab: int
    layer_types: tuple
    heads: int
    nope: int
    rope: int
    value: int
    qrank: int
    kvrank: int
    kda_heads: int
    kda_dim: int
    kernel: int
    dense_intermediate: int
    latent: int
    expert_intermediate: int
    experts: int
    selected: int
    shared: int
    capacity: int
    dense_prefix: int = 1
    residual_block: int = 12
    epsilon: float = 1e-5
    projection_epsilon: float = 1e-6
    l2_epsilon: float = 1e-6
    situ_beta: float = 4.0
    situ_linear_beta: float = 25.0
    routed_scale: float = 1.0
    rotary_mode: str = "publisher-none"
    rotary_theta: float = 10000.0
    revision: str = REVISION
    expert_groups: int = 1
    selected_groups: int = 1
    expert_format: str = "fp32"
    vision: K3VisionConfig | None = None
    mla_mode: str = "publisher-expanded"

    def __post_init__(self):
        names = ("hidden", "vocab", "heads", "nope", "value", "qrank", "kvrank",
                 "kda_heads", "kda_dim", "kernel", "dense_intermediate", "latent",
                 "expert_intermediate", "experts", "selected", "capacity", "residual_block",
                 "expert_groups", "selected_groups")
        for name in names:
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError("positive integer required: " + name)
        if type(self.layer_types) is not tuple or not self.layer_types:
            raise ValueError("explicit layer schedule required")
        if any(type(x) is not str or x not in ("mla", "kda") for x in self.layer_types):
            raise ValueError("unknown mixer")
        if self.layer_types[-1] != "mla":
            raise ValueError("final global MLA required")
        for name in ("rope", "shared", "dense_prefix"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError("nonnegative integer required: " + name)
        if self.rope % 2 or self.dense_prefix > len(self.layer_types) or self.selected > self.experts:
            raise ValueError("invalid rotary dimension, dense prefix, or top-k")
        if self.kda_heads > 128 or self.kda_dim > 256 or self.kernel > 64:
            raise ValueError("native KDA geometry exceeded")
        if self.heads > 128 or self.nope + self.rope > 512 or self.value > 256 or self.capacity > 1048576:
            raise ValueError("native attention geometry exceeded")
        if self.kvrank > 65536:
            raise ValueError("native latent attention rank exceeded")
        if self.experts % self.expert_groups or self.selected_groups > self.expert_groups:
            raise ValueError("invalid expert groups")
        if self.expert_groups > self.selected_groups and (self.experts // self.expert_groups < 2 or
                self.selected > self.selected_groups * (self.experts // self.expert_groups)):
            raise ValueError("group selection cannot supply top-k")
        if self.expert_format not in ("fp32", "e2m1-low-first-e8m0-g32"):
            raise ValueError("unknown explicit expert packing contract")
        if self.expert_format != "fp32" and (self.latent % 32 or self.expert_intermediate % 32):
            raise ValueError("packed expert input dimensions must be divisible by 32")
        if self.vision is not None and (type(self.vision) is not K3VisionConfig or self.vision.text_hidden != self.hidden):
            raise ValueError("vision/projector text width must match decoder")
        for name in ("epsilon", "projection_epsilon", "l2_epsilon", "situ_beta",
                     "situ_linear_beta", "routed_scale", "rotary_theta"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v) or v <= 0:
                raise ValueError("positive finite scalar required: " + name)
        if self.revision != REVISION or self.rotary_mode not in ("publisher-none", "split-half"):
            raise ValueError("unresolved revision or rotary semantics")
        if self.mla_mode not in ("publisher-expanded", "compressed"):
            raise ValueError("unknown MLA evaluation policy")


def tensor_shapes(c):
    """Exact FP32 publisher-name inventory for the explicitly configured subset."""
    if type(c) is not K3Config:
        raise ValueError("K3Config required")
    h = c.hidden
    shapes = {"language_model.model.embed_tokens.weight": (c.vocab, h),
              "language_model.model.norm.weight": (h,),
              "language_model.model.output_attn_res_norm.weight": (h,),
              "language_model.model.output_attn_res_proj.weight": (1, h),
              "language_model.lm_head.weight": (c.vocab, h)}

    def mlp(p, width, middle, expert=False):
        g, u, d = ("w1", "w3", "w2") if expert else ("gate_proj", "up_proj", "down_proj")
        shapes[p + g + ".weight"] = (middle, width)
        shapes[p + u + ".weight"] = (middle, width)
        shapes[p + d + ".weight"] = (width, middle)
        if expert and c.expert_format != "fp32":
            for name in (g, u, d):
                rows, cols = shapes.pop(p + name + ".weight")
                shapes[p + name + ".weight_packed"] = (rows, cols // 2)
                shapes[p + name + ".weight_scale"] = (rows, cols // 32)

    for i, kind in enumerate(c.layer_types):
        p = f"language_model.model.layers.{i}."
        for name in ("input_layernorm", "post_attention_layernorm", "self_attention_res_norm", "mlp_res_norm"):
            shapes[p + name + ".weight"] = (h,)
        for name in ("self_attention_res_proj", "mlp_res_proj"):
            shapes[p + name + ".weight"] = (1, h)
        if kind == "mla":
            local = {"q_a_proj.weight": (c.qrank, h), "q_a_layernorm.weight": (c.qrank,),
                     "q_b_proj.weight": (c.heads * (c.nope + c.rope), c.qrank),
                     "kv_a_proj_with_mqa.weight": (c.kvrank + c.rope, h),
                     "kv_a_layernorm.weight": (c.kvrank,),
                     "kv_b_proj.weight": (c.heads * (c.nope + c.value), c.kvrank),
                     "g_proj.weight": (c.heads * c.value, h), "o_proj.weight": (h, c.heads * c.value)}
        else:
            w = c.kda_heads * c.kda_dim
            local = {name + ".weight": (w, h) for name in ("q_proj", "k_proj", "v_proj", "g_proj")}
            local.update({"f_a_proj.weight": (c.kda_dim, h), "f_b_proj.weight": (w, c.kda_dim),
                          "dt_bias": (w,), "A_log": (c.kda_heads,), "b_proj.weight": (c.kda_heads, h),
                          "o_norm.weight": (c.kda_dim,), "o_proj.weight": (h, w)})
            local.update({name + "_conv1d.weight": (w, 1, c.kernel) for name in ("q", "k", "v")})
        shapes.update({p + "self_attn." + name: shape for name, shape in local.items()})
        if i < c.dense_prefix:
            mlp(p + "mlp.", h, c.dense_intermediate)
        else:
            m = p + "block_sparse_moe."
            shapes.update({m + "gate.weight": (c.experts, h), m + "gate.e_score_correction_bias": (c.experts,),
                           m + "routed_expert_down_proj.weight": (c.latent, h),
                           m + "routed_expert_up_proj.weight": (h, c.latent),
                           m + "routed_expert_norm.weight": (c.latent,)})
            for e in range(c.experts):
                mlp(m + f"experts.{e}.", c.latent, c.expert_intermediate, True)
            if c.shared:
                mlp(m + "shared_experts.", h, c.shared * c.expert_intermediate)
    if c.vision is not None:
        shapes.update(vision_tensor_shapes(c.vision))
    return shapes


def _fp32(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    if not np.isfinite(a).all():
        raise ValueError("nonfinite arithmetic result")
    return a


@dataclass(frozen=True)
class K3Snapshot:
    identity: str
    position: int
    mla: tuple
    kda: tuple
    exact_evidence: str | None=None
    media_case: str | None=None


@dataclass(frozen=True)
class _ProjectionReference:
    shape: tuple[int, int]


class K3Execution:
    """Batch-one FP32 composition, with atomic whole-forward cache commit.

    Exact admission requires a pinned comparison receipt for this execution
    identity and BF16 arithmetic, within its recorded text-only boundaries.
    """
    published_checkpoint_supported = False
    implementation = "k3-explicit-fp32-native-recurrence-composition"

    def __init__(self, config, tensors, *, library, max_bytes=1024 * 1024 * 1024,
                 require_publisher_exact=False, mxfp4_library=None, attention_library=None,
                 operators_library=None, expert_provider=None, routing=None, fla_policy=None,
                 arithmetic=None, sharded_projections=(),exact_evidence=None,
                 native_projections=None,native_projection_work_bytes=64*1024**2):
        if type(require_publisher_exact) is not bool: raise ValueError('explicit exact mode boolean required')
        self.exact_mode_admission=None
        self._exact_evidence=None
        if exact_evidence is not None:
            if not require_publisher_exact:raise ValueError('exact evidence requires explicit exact execution policy')
            from .exact_admission import K3ExactEvidence
            if type(exact_evidence) is not K3ExactEvidence: raise ValueError('explicit pinned K3ExactEvidence required')
        if require_publisher_exact and (exact_evidence is None or arithmetic is None):
            raise ValueError('exact mode requires BF16 arithmetic and pinned whole-model comparison evidence')
        if routing is not None:
            from .routing import TorchK3Router
            if type(routing) is not TorchK3Router:
                raise ValueError('explicit TorchK3Router required')
        self._routing = routing
        if arithmetic is not None:
            from .bf16_arithmetic import K3BF16Arithmetic
            if type(arithmetic) is not K3BF16Arithmetic:
                raise ValueError('explicit K3BF16Arithmetic required')
            if config.mla_mode != 'publisher-expanded':
                raise ValueError('BF16 policy requires expanded MLA')
            if routing is None:
                raise ValueError('BF16 policy requires explicit PyTorch routing')
            if (routing.device != arithmetic.device
                    or routing.version != str(arithmetic.torch.__version__)):
                raise ValueError('routing and arithmetic device/version policies differ')
        self._arithmetic = arithmetic
        self._native_projections=native_projections
        self._native_projection_work_bytes=0
        if native_projections is not None:
            from ..native_original_projections import NativeOriginalProjections
            if (type(native_projections) is not NativeOriginalProjections or arithmetic is not None
                    or native_projections.reduction!='fp64' or type(native_projection_work_bytes) is not int
                    or native_projection_work_bytes<=0):
                raise ValueError('native K3 projections require an original FP64 dot owner and bounded scratch')
            self._native_projection_work_bytes=native_projection_work_bytes
            self.implementation='k3-native-original-dense-row-stream-composition'
        if arithmetic is not None:
            self.implementation='k3-bf16-device-text-moonvit-composition' if config.vision is not None else 'k3-bf16-device-text-composition'
        if fla_policy is not None:
            from .fla_mixer import FLAK3Policy
            from .metal_mixer import MetalK3Policy
            if type(fla_policy) not in (FLAK3Policy, MetalK3Policy):
                raise ValueError('explicit FLA or Metal KDA policy required')
        if arithmetic is not None:
            if 'kda' in config.layer_types and fla_policy is None:
                raise ValueError('hybrid BF16 execution requires an explicit FLA policy')
            if fla_policy is not None:
                target = arithmetic.torch.device(fla_policy.device)
                if target.type == 'cuda' and target.index is None:
                    target = arithmetic.torch.device('cuda', arithmetic.torch.cuda.current_device())
                if target.type == 'mps':
                    target = arithmetic.torch.device('mps:0')
                if target != arithmetic.device or fla_policy.torch_version != str(arithmetic.torch.__version__):
                    raise ValueError('FLA and BF16 arithmetic device/version policies differ')
        self._fla_policy = fla_policy
        self._cancel_local = threading.local()
        shapes = tensor_shapes(config)
        expert_shapes = {name: shape for name, shape in shapes.items() if ".experts." in name}
        if expert_provider is not None:
            if config.expert_format == "fp32" or not callable(getattr(expert_provider, "lease", None)):
                raise ValueError("provider requires explicit packed experts and lease(layer,expert)")
            identity = getattr(expert_provider, "identity", None)
            provider_budget = getattr(expert_provider, "max_expert_bytes", None)
            if not isinstance(identity, str) or not identity or type(provider_budget) is not int or provider_budget <= 0:
                raise ValueError("provider identity and positive max_expert_bytes required")
            if getattr(expert_provider, "tensor_shapes", None) != expert_shapes:
                raise ValueError("exact packed provider tensor inventory required")
            shapes = {name: shape for name, shape in shapes.items() if name not in expert_shapes}
        self._expert_provider = expert_provider
        if (type(sharded_projections) is not tuple or
                any(type(name) is not str for name in sharded_projections) or
                len(set(sharded_projections)) != len(sharded_projections)):
            raise ValueError('explicit unique sharded projection names required')
        sharded = set(sharded_projections)
        bank = native_projections if arithmetic is None else arithmetic._projection_backend
        allowed = {'q_proj','k_proj','v_proj','q_a_proj','q_b_proj','kv_a_proj_with_mqa',
                   'kv_b_proj','o_proj','f_a_proj','f_b_proj','b_proj','g_proj','gate_proj','up_proj',
                   'down_proj','w1','w2','w3','lm_head'}
        allowed.update({'wqkv','wo','fc0','fc1','0','2','embed_tokens',
                        'routed_expert_down_proj','routed_expert_up_proj'})
        for name in sharded:
            if (bank is None or name not in shapes or len(shapes[name]) != 2 or
                    not name.endswith('.weight') or name.split('.')[-2] not in allowed or
                    not bank.handles(name) or bank.shape(name) != shapes[name] or
                    (arithmetic is not None and bank.dtype_name != 'bfloat16')):
                raise ValueError('unadmitted rank-local projection reference: '+name)
            if name=='language_model.model.embed_tokens.weight' and not callable(getattr(bank,'gather_rows',None)):
                raise ValueError('original token table references require an admitted row gather owner')
            if native_projections is not None and config.mla_mode=='compressed' and name.endswith('.kv_b_proj.weight'):
                raise ValueError('original native KV projection streaming requires expanded MLA')
        if any(len(shape) == 2 and (shape[1] > 65536 or shape[0] > 1048576)
               for name, shape in shapes.items() if not name.endswith(("weight_packed", "weight_scale"))):
            raise ValueError("native projection geometry exceeded")
        if type(max_bytes) is not int or max_bytes <= 0 or set(tensors) != set(shapes)-sharded:
            raise ValueError("exact inventory and positive memory budget required")
        for name, shape in shapes.items():
            if name in sharded:
                continue
            a = np.asarray(tensors[name])
            dtype = np.uint8 if name.endswith(("weight_packed", "weight_scale")) else np.float32
            if a.shape != shape or a.dtype != dtype or not a.flags.c_contiguous or not np.isfinite(a).all():
                raise ValueError("invalid tensor: " + name)
            if name.endswith("weight_scale") and np.any(a == 255):
                raise ValueError("nonfinite E8M0 scales are unsupported")
        self.config, self.max_bytes = config, max_bytes
        self._weight_bytes = sum(np.asarray(tensors[n]).nbytes for n in shapes if n not in sharded)
        self._projection_scratch=0
        if sharded:
            from .publisher_projections import K3PublisherProjections
            if type(bank) is K3PublisherProjections or bank is native_projections:self._projection_scratch=bank.max_bytes
            else:self._weight_bytes += bank.max_bytes
        self._cache_bytes = sum(0 if kind == "mla" else 4 * (config.kda_heads * config.kda_dim ** 2 +
                                3 * (config.kernel - 1) * config.kda_heads * config.kda_dim)
                                for kind in config.layer_types)
        self._admit(1)
        self._weights = {n: np.frombuffer(np.asarray(tensors[n]).tobytes(), dtype=np.asarray(tensors[n]).dtype).reshape(s)
                         for n, s in shapes.items() if n not in sharded}
        self._weights.update({name:_ProjectionReference(shapes[name]) for name in sharded})
        digest = hashlib.sha256(repr(config).encode("ascii"))
        digest.update(('native-fp32' if arithmetic is None else arithmetic.identity).encode('ascii'))
        digest.update(repr(fla_policy).encode('utf-8'))
        digest.update(('strict-native-no-ties' if routing is None else routing.identity).encode('ascii'))
        if expert_provider is not None:
            digest.update(expert_provider.identity.encode("utf-8"))
        for name in sorted(self._weights):
            digest.update(name.encode("ascii"))
            if name in sharded:
                digest.update(('rank-local-reference:'+repr(shapes[name])+':'+bank.identity).encode('ascii'))
            else:
                digest.update(self._weights[name].tobytes())
        self._identity = digest.hexdigest()
        if require_publisher_exact:
            self.exact_mode_admission=exact_evidence.admit(self._identity,arithmetic.identity,
                libraries=(library,mxfp4_library,attention_library,operators_library))
            self._exact_evidence=exact_evidence
        self._lock = threading.RLock()
        self._busy, self._closed = False, False
        self._raw_libraries = []
        self._release = weakref.finalize(self, _release_libraries, self._raw_libraries)
        self._kda = {}
        for i, kind in enumerate(config.layer_types):
            if kind == "kda":
                p = f"language_model.model.layers.{i}.self_attn."
                taps = tuple(self._weights[p + n + "_conv1d.weight"].reshape(config.kda_heads, config.kda_dim, config.kernel)
                             for n in ("q", "k", "v"))
                if fla_policy is not None:
                    from .fla_mixer import FLAK3Mixer
                    from .metal_mixer import MetalK3Mixer, MetalK3Policy
                    mixer_type = MetalK3Mixer if type(fla_policy) is MetalK3Policy else FLAK3Mixer
                    pin = {} if type(fla_policy) is MetalK3Policy else {'fla_version': fla_policy.fla_version}
                    self._kda[i] = mixer_type(taps, self._weights[p+'o_norm.weight'],
                        self._weights[p+'A_log'], self._weights[p+'dt_bias'],
                        device=fla_policy.device, capacity=config.capacity,
                        max_bytes=fla_policy.max_bytes, torch_version=fla_policy.torch_version,
                        epsilon=config.epsilon, **pin)
                    continue
                self._kda[i] = NativeKimiKDA(taps, self._weights[p + "o_norm.weight"], library=library,
                                            capacity=config.capacity, epsilon=config.epsilon,
                                            l2_epsilon=config.l2_epsilon, q_scale=config.kda_dim ** -0.5,
                                            max_bytes=max_bytes)
                self._raw_libraries.append(self._kda[i].library)
        self._library=self._segment_library=self._ops=None
        self._mat=self._attention=self._latent_segment=self._segment=self._route=None
        if arithmetic is None:
            self._bind_cpu(library,attention_library,operators_library)
        self._packed = (NativeMXFP4(library=mxfp4_library or library, max_bytes=max_bytes)
                        if config.expert_format != 'fp32' and arithmetic is None else None)
        self._vision = None
        if config.vision is not None:
            inventory = vision_tensor_shapes(config.vision)
            self._vision = K3Vision(config.vision, {n: self._weights[n] for n in inventory},
                                    matrix=self._matrix, attention=self._attend_arrays,
                                    operators=self._ops, vision_ops=(VisionOps(self._segment_library, max_bytes) if arithmetic is None else None),
                                    max_bytes=max_bytes,arithmetic=arithmetic,
                                    check_cancelled=self._check_cancelled,_borrow_tensors=True,
                                    native_projections=native_projections,native_projection_work_bytes=native_projection_work_bytes)
        self.reset()

    def _bind_cpu(self,library,attention_library,operators_library):
        self._library = C.CDLL(str(Path(library).resolve(strict=True)))
        self._raw_libraries.append(self._library)
        fp,dp = C.POINTER(C.c_float),C.POINTER(C.c_double)
        self._mat = self._library.oa_kimi_matvec
        self._mat.argtypes,self._mat.restype = [C.c_uint64]*2+[fp]*3,C.c_int
        self._attention = self._library.oa_kimi_attention
        self._attention.argtypes,self._attention.restype = [C.c_uint64]*4+[fp]*4,C.c_int
        self._ops = NativeTensorOps(operators_library or library,max_scratch_bytes=self.max_bytes)
        self._segment_library = C.CDLL(str(Path(attention_library).resolve(strict=True))) if attention_library else self._library
        if self._segment_library is not self._library:self._raw_libraries.append(self._segment_library)
        self._latent_segment = getattr(self._segment_library,'oa_k3_latent_segment',None)
        if self._latent_segment is None and self.config.mla_mode=='compressed':
            raise ValueError('compressed MLA requires the explicit native attention library')
        if self._latent_segment is not None:
            self._latent_segment.argtypes = [C.c_uint64]*4+[C.c_double]+[fp]*3+[dp]*3
            self._latent_segment.restype = C.c_int
        self._segment = getattr(self._segment_library,'oa_k3_attention_segment',None)
        if self._segment is not None:
            self._segment.argtypes,self._segment.restype = [C.c_uint64]*4+[fp]*3+[dp]*3,C.c_int
        elif self.config.mla_mode=='publisher-expanded' or self.config.capacity>65536:
            raise ValueError('expanded MLA requires the explicit native attention library')
        self._route = getattr(self._segment_library,'oa_k3_grouped_route',None)
        if self._route is not None:
            self._route.argtypes = [C.c_uint64]*4+[C.c_double,fp,fp,C.POINTER(C.c_uint64),fp,C.c_uint64]
            self._route.restype = C.c_int
        elif self.config.expert_groups>1:
            raise ValueError('native grouped routing requires the explicit attention library')

    @contextmanager
    def _operation(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("K3 executor is closed")
            if self._busy:
                raise RuntimeError("reentrant K3 operation is not allowed")
            self._busy = True
            try:
                yield
            finally:
                self._busy = False

    def close(self):
        """Release executor-owned libraries and caches; never close caller provider."""
        with self._lock:
            if self._busy:
                raise RuntimeError("cannot close a busy K3 executor")
            if getattr(self,'_resources_released',False):
                return
            self._closed = True
            # Stop at a failed child release. Its library and borrowed provider
            # remain reachable for retry and cannot be unloaded underneath it.
            if self._fla_policy is not None:
                while self._kda:
                    index=next(reversed(self._kda))
                    self._kda[index].close();del self._kda[index]
            if getattr(self, "_ops", None) is not None:
                self._ops.close();self._ops=None
            if getattr(self, "_packed", None) is not None:
                self._packed._release();self._packed=None
            _release_libraries(self._raw_libraries)
            self._release.detach()
            self._mla, self._kda, self._weights, self._vision = {}, {}, {}, None
            self._comparison_observations=None
            self._expert_provider = None;self._resources_released=True

    def _admit(self, n):
        c = self.config
        scratch = 64 * (c.hidden * (len(c.layer_types) // c.residual_block + 3) + c.vocab + c.experts +
                        c.latent + c.dense_intermediate + c.expert_intermediate * max(1, c.shared) +
                        c.heads * (c.nope + c.rope + c.value + c.kvrank) + c.kda_heads * c.kda_dim + c.qrank + c.kvrank)
        live_pages = sum(p.nbytes for p in getattr(self, "_mla", {}).values())
        new_pages = 4 * n * (c.kvrank + c.rope) * c.layer_types.count("mla")
        tail_scratch = 8 * 256 * (c.kvrank + c.rope) * c.layer_types.count("mla")
        expanded_scratch = (16 * 256 * c.heads * (c.nope + c.rope + c.value)
                            if c.mla_mode == "publisher-expanded" else 0)
        provider_bytes = getattr(self._expert_provider, "max_expert_bytes", 0)
        device_scratch=(self._projection_scratch+self._native_projection_work_bytes+
                        getattr(self._arithmetic,'max_bytes',0)+getattr(self._routing,'max_bytes',0)
                        +getattr(self._fla_policy,'max_bytes',0)*c.layer_types.count('kda'))
        observation_bytes=getattr(getattr(self._cancel_local,'observer',None),'maximum',0)
        retained=getattr(self,'_comparison_observations',None)
        if isinstance(retained,dict):observation_bytes+=sum(value.nbytes for value in retained.values())
        required = (5 * self._weight_bytes + 2 * provider_bytes + 4 * self._cache_bytes +
                    3 * live_pages + 3 * new_pages + tail_scratch + scratch +
                    expanded_scratch+device_scratch+observation_bytes+8*c.capacity+4*n*(c.vocab+c.hidden))
        if required > self.max_bytes:
            raise ValueError("K3 resident/transaction budget exceeded")
        return required

    @staticmethod
    def _ptr(a):
        return a.ctypes.data_as(C.POINTER(C.c_float))

    def _linear(self, name, x):
        if self._native_projections is not None and self._native_projections.handles(name):
            return self._native_projections.linear(name,_fp32(x),max_work_bytes=self._native_projection_work_bytes,
                cancelled=lambda:bool(getattr(self._cancel_local,'event',None) is not None and self._cancel_local.event.is_set()))
        if self._arithmetic is not None:
            if name not in self._weights and '.experts.' in name:
                prefix = name.removesuffix('weight')
                packed = self._weights[prefix+'weight_packed']
                return self._arithmetic.packed_matrix(packed, self._weights[prefix+'weight_scale'],
                    _fp32(x), rows=packed.shape[0], cols=packed.shape[1]*2,
                    cancel=getattr(self._cancel_local, 'event', None))
            return self._arithmetic.named_matrix(name, self._weights[name], _fp32(x),
                                                router=name.endswith('gate.weight'),check_cancelled=self._check_cancelled)
        if name not in self._weights and self._packed is not None and ".experts." in name:
            prefix = name.removesuffix("weight")
            packed, scales = self._weights[prefix + "weight_packed"], self._weights[prefix + "weight_scale"]
            return self._packed.matvec(packed, scales, _fp32(x), rows=packed.shape[0], cols=packed.shape[1] * 2)
        return self._matrix(self._weights[name], x)

    def _matrix(self, w, x):
        if self._arithmetic is not None:
            return self._arithmetic.matrix(_fp32(w), _fp32(x),check_cancelled=self._check_cancelled)
        w = np.ascontiguousarray(w, dtype=np.float32)
        x = _fp32(x)
        out = np.empty(w.shape[0], np.float32)
        # Preserve the primitive's row arithmetic while avoiding its matrix-size
        # bound for dense prefixes, shared MLPs, embeddings and vocabulary heads.
        tile = min(1048576, 67108864 // w.shape[1])
        for start in range(0, w.shape[0], tile):
            rows = min(tile, w.shape[0] - start)
            if self._mat(rows, w.shape[1], self._ptr(w[start:start + rows]),
                         self._ptr(x), self._ptr(out[start:start + rows])):
                raise ValueError("native K3 projection failed")
        return out

    def _attend_pages(self, q, pages):
        if self._arithmetic is not None:
            if self._arithmetic.attention_mode=='paged-online-fp32':
                return self._arithmetic.attention_pages(_fp32(q),lambda:iter(pages),
                    check_cancelled=self._check_cancelled)
            keys=_fp32(np.concatenate([k for k,v in pages]))
            values=_fp32(np.concatenate([v for k,v in pages]))
            return self._arithmetic.attention(_fp32(q),keys,values)
        if self._segment is None:
            keys = _fp32(np.concatenate([k for k, v in pages]))
            values = _fp32(np.concatenate([v for k, v in pages]))
            out = np.empty((q.shape[0], values.shape[-1]), np.float32)
            if self._attention(len(keys), q.shape[0], q.shape[1], values.shape[-1],
                               *(self._ptr(a) for a in (q, keys, values, out))):
                raise ValueError("native attention failed")
            return out
        heads, key = q.shape
        value = pages[0][1].shape[-1]
        peak, denominator = np.full(heads, -np.inf, np.float64), np.zeros(heads, np.float64)
        numerator = np.zeros((heads, value), np.float64)
        dp = C.POINTER(C.c_double)
        for keys, values in pages:
            if self._segment(len(keys), heads, key, value, *(self._ptr(a) for a in (q, keys, values)),
                             *(a.ctypes.data_as(dp) for a in (peak, denominator, numerator))):
                raise ValueError("native page attention failed")
        return _fp32(numerator / denominator[:, None])

    def _attend_arrays(self, q, keys, values):
        if len(keys) > 65536 and self._segment is None and self._arithmetic is None:
            raise ValueError("large vision sequence requires page attention library")
        return self._attend_pages(q, tuple((keys[i:i + 256], values[i:i + 256]) for i in range(0, len(keys), 256)))

    def _norm(self, x, name, eps=None):
        eps = self.config.epsilon if eps is None else eps
        if self._arithmetic is not None:
            return self._arithmetic.norm(_fp32(x), self._weights[name], eps)
        return self._ops.normalize(_fp32(x), self._weights[name], epsilon=eps)

    def _residual(self, prefix, blocks, p):
        values = np.stack([*blocks, prefix])
        if self._arithmetic is not None:
            return self._arithmetic.residual(_fp32(values), self._weights[p+'_norm.weight'],
                self._weights[p+'_proj.weight'][0], self.config.epsilon)
        # Publisher scores normalize values, then use an FP32 gain/query product.
        score_weight = _fp32(self._weights[p + "_norm.weight"] * self._weights[p + "_proj.weight"][0])
        return self._ops.attnres(values, np.ones(self.config.hidden, np.float32),
                                 score_weight, epsilon=self.config.epsilon)

    def _mlp(self, x, p, expert=False):
        c = self.config
        g, u, d = ("w1", "w3", "w2") if expert else ("gate_proj", "up_proj", "down_proj")
        gate = self._linear(p + g + ".weight", x)
        up = self._linear(p + u + ".weight", x)
        active = (self._arithmetic.situ(gate, up, c.situ_beta, c.situ_linear_beta)
                  if self._arithmetic is not None else
                  self._ops.situ(gate, up, beta=c.situ_beta, linear_beta=c.situ_linear_beta))
        return self._linear(p + d + ".weight", active)

    def _moe(self, x, p):
        c = self.config
        logits = self._linear(p+'gate.weight', x)
        scores = (self._arithmetic.router_sigmoid(logits) if self._arithmetic is not None
                  else self._ops.sigmoid(logits))
        if self._routing is not None:
            correction = self._weights[p+'gate.e_score_correction_bias']
            if self._arithmetic is not None:
                correction = self._arithmetic.cast(correction)
            route = self._routing.route(scores, correction,
                selected=c.selected, groups=c.expert_groups, selected_groups=c.selected_groups,
                scale=c.routed_scale)
            self._observe_route(p,logits,route.ids,route.weights)
            return self._aggregate_experts(x, p, route.ids, route.weights)
        if self._route is not None:
            ids, weight = np.empty(c.selected, np.uint64), np.empty(c.selected, np.float32)
            status = self._route(c.experts, c.selected, c.expert_groups, c.selected_groups, c.routed_scale,
                                 self._ptr(scores), self._ptr(self._weights[p + "gate.e_score_correction_bias"]),
                                 ids.ctypes.data_as(C.POINTER(C.c_uint64)), self._ptr(weight), self.max_bytes)
            if status:
                raise ValueError("native grouped route failed; unresolved ties" if status == 4 else "native grouped route failed")
            self._observe_route(p,logits,ids,weight)
            return self._aggregate_experts(x, p, ids, weight)
        choice = _fp32(scores + self._weights[p + "gate.e_score_correction_bias"])
        if c.expert_groups > c.selected_groups:
            grouped = choice.reshape(c.expert_groups, -1)
            group_scores = _fp32(np.sort(grouped, axis=1)[:, -2:].sum(axis=1))
            if len(np.unique(group_scores)) != len(group_scores):
                raise ValueError("unresolved publisher group top-k tie semantics")
            group_ids = np.argsort(-group_scores, kind="stable")[:c.selected_groups]
            allowed = np.zeros(c.expert_groups, bool)
            allowed[group_ids] = True
            allowed = np.repeat(allowed, c.experts // c.expert_groups)
        else:
            allowed = np.ones(c.experts, bool)
        # Publisher topk does not specify tie order; do not silently invent it.
        if len(np.unique(choice[allowed])) != np.count_nonzero(allowed):
            raise ValueError("unresolved publisher top-k tie semantics")
        ids = np.argsort(-np.where(allowed, choice, -np.inf), kind="stable")[:c.selected]
        weight = scores[ids].copy()
        if c.selected > 1:
            weight /= weight.sum() + 1e-20
        weight = _fp32(weight * c.routed_scale)
        self._observe_route(p,logits,ids,weight)
        return self._aggregate_experts(x, p, ids, weight)

    def _aggregate_experts(self, x, p, ids, weight):
        c = self.config
        latent = self._linear(p + "routed_expert_down_proj.weight", x)
        routed = np.zeros(c.latent, np.float32)
        outputs = None
        if self._routing is not None:
            extra = 4*len(ids)*c.latent
            if self._admit(0)+extra > self.max_bytes:
                raise MemoryError('expert slot buffers exceed transaction admission')
            if 3*extra+8*len(ids)+8*c.latent > self._routing.max_bytes:
                raise MemoryError('publisher-order expert reduction exceeds scratch admission')
            outputs = np.empty((len(ids), c.latent), np.float32)
        order = range(len(ids)) if outputs is None else np.argsort(ids, kind='stable')
        for slot in order:
            e, w = ids[slot], weight[slot]
            if self._expert_provider is None:
                expert_output = self._mlp(latent, p + f"experts.{e}.", True)
            else:
                layer = int(p.split(".")[3])
                with self._expert_provider.lease(layer, int(e)) as tables:
                    expert_output = self._provider_mlp(latent, tables)
            if outputs is None:
                routed = _fp32(routed + w * expert_output)
            else:
                outputs[slot] = expert_output
        if outputs is not None:
            routed = self._routing.aggregate(outputs, weight)
        routed = self._norm(routed, p + "routed_expert_norm.weight")
        out = self._linear(p + "routed_expert_up_proj.weight", routed)
        return self._add(out, self._mlp(x, p+'shared_experts.')) if c.shared else out

    def _observe_route(self,prefix,logits,indices,weights):
        observer=getattr(self._cancel_local,'observer',None)
        if observer is not None:
            self._check_cancelled();observer.route(prefix,logits,indices,weights)

    @contextmanager
    def observation_scope(self,*,max_bytes,routing=True,projection_names=()):
        """Observe executed values for a later explicitly authorized comparison."""
        from contextlib import ExitStack
        from .observations import K3ObservationRecorder
        with self._lock:
            if self._closed or self._busy or getattr(self._cancel_local,'observer',None) is not None:
                raise RuntimeError('available independent K3 observation owner required')
            if projection_names and self._arithmetic is None:
                raise ValueError('BF16 projection boundary observations require the admitted arithmetic owner')
            recorder=K3ObservationRecorder(self.config,max_bytes=max_bytes,routing=routing,projection_names=projection_names)
            if any(name not in self._weights or len(self._weights[name].shape)!=2 for name in projection_names):
                raise ValueError('named local dense projection boundary inventory required')
            bank=None if self._arithmetic is None else self._arithmetic._projection_backend
            if (bank is not None and any(bank.handles(name) for name in projection_names) and
                    not getattr(bank,'boundary_observation_supported',False)):
                raise ValueError('borrowed-bank observation requires its own executed operand hook')
            self._cancel_local.observer=recorder
            try:
                self._admit(0)
                with ExitStack() as scopes:
                    if projection_names:scopes.enter_context(self._arithmetic.observation_scope(recorder))
                    bank_scope=getattr(bank,'boundary_observation_scope',None)
                    if projection_names and callable(bank_scope):scopes.enter_context(bank_scope(recorder))
                    yield recorder
            finally:self._cancel_local.observer=None

    def _add(self, left, right):
        return (self._arithmetic.add(_fp32(left), _fp32(right)) if self._arithmetic is not None
                else _fp32(left+right))

    @contextmanager
    def cancellation_scope(self, cancel):
        """Forward request cancellation to an explicitly scoped expert provider."""
        from contextlib import ExitStack
        if cancel is not None and not callable(getattr(cancel,'is_set',None)):
            raise ValueError('request cancellation must provide is_set')
        scope = getattr(self._expert_provider, 'cancellation_scope', None)
        bank=None if self._arithmetic is None else self._arithmetic._projection_backend
        projection_scope=getattr(bank,'cancellation_scope',None)
        previous = getattr(self._cancel_local, 'event', None)
        self._cancel_local.event = cancel
        try:
            with ExitStack() as owned:
                if callable(scope):owned.enter_context(scope(cancel))
                if callable(projection_scope):owned.enter_context(projection_scope(cancel))
                yield
        finally:
            self._cancel_local.event = previous

    def _provider_mlp(self, latent, tables):
        c = self.config
        shapes = {"w1": (c.expert_intermediate, c.latent), "w3": (c.expert_intermediate, c.latent),
                  "w2": (c.latent, c.expert_intermediate)}
        expected = {matrix + "." + suffix for matrix in shapes for suffix in ("weight_packed", "weight_scale")}
        if set(tables) != expected:
            raise ValueError("selected provider expert inventory mismatch")
        admitted = {}
        for matrix, (rows, cols) in shapes.items():
            for suffix, shape in (("weight_packed", (rows, cols // 2)), ("weight_scale", (rows, cols // 32))):
                name = matrix + "." + suffix
                a = np.asarray(tables[name])
                if a.shape != shape or a.dtype != np.uint8 or not a.flags.c_contiguous or a.flags.writeable:
                    raise ValueError("provider requires read-only contiguous packed tables")
                if suffix == "weight_scale" and np.any(a == 255):
                    raise ValueError("provider contains nonfinite scales")
                admitted[name] = a
        if sum(a.nbytes for a in admitted.values()) > self._expert_provider.max_expert_bytes:
            raise ValueError("provider exceeded selected-expert byte budget")

        def project(matrix, x):
            rows, cols = shapes[matrix]
            if self._arithmetic is not None:
                return self._arithmetic.packed_matrix(admitted[matrix+'.weight_packed'],
                    admitted[matrix+'.weight_scale'], _fp32(x), rows=rows, cols=cols,
                    cancel=getattr(self._cancel_local, 'event', None))
            return self._packed.matvec(admitted[matrix + ".weight_packed"], admitted[matrix + ".weight_scale"], x,
                                       rows=rows, cols=cols)

        gate, up = project("w1", latent), project("w3", latent)
        active = (self._arithmetic.situ(gate, up, c.situ_beta, c.situ_linear_beta)
                  if self._arithmetic is not None else
                  self._ops.situ(gate, up, beta=c.situ_beta, linear_beta=c.situ_linear_beta))
        return project('w2', active)

    def _rotate(self, x, position):
        c = self.config
        if c.rotary_mode == "publisher-none" or not c.rope:
            return x
        shape = x.shape
        rows = _fp32(x.reshape(-1, c.rope))
        positions = np.full(rows.shape[0], position, dtype=np.uint64)
        if self._arithmetic is not None:
            return self._arithmetic.rotate(rows,positions,rotary=c.rope,theta=c.rotary_theta).reshape(shape)
        return self._ops.rotate(rows, positions, rotary=c.rope, theta=c.rotary_theta).reshape(shape)

    def _mla_forward(self, x, p, cache, position, rotary_position):
        c = self.config
        q = self._linear(p + "q_a_proj.weight", x)
        q = self._norm(q, p + "q_a_layernorm.weight", c.projection_epsilon)
        q = self._linear(p + "q_b_proj.weight", q).reshape(c.heads, c.nope + c.rope)
        kv = self._linear(p + "kv_a_proj_with_mqa.weight", x)
        latent = self._norm(kv[:c.kvrank], p + "kv_a_layernorm.weight", c.projection_epsilon)
        expansion_name = p + "kv_b_proj.weight"
        expansion = (None if self._arithmetic is not None or
                     (self._native_projections is not None and self._native_projections.handles(expansion_name)) else
                     self._weights[expansion_name].reshape(c.heads, c.nope + c.value, c.kvrank))
        q[:, c.nope:] = self._rotate(q[:, c.nope:], rotary_position)
        if cache.length != position:
            raise ValueError("MLA cache position mismatch")
        cache.append(latent, self._rotate(kv[c.kvrank:], rotary_position))
        if c.mla_mode == "publisher-expanded":
            out = self._mla_expanded(q, expansion, cache, projection_name=expansion_name)
        else:
            out = self._mla_compressed(q, expansion, cache)
        gate_logits = self._linear(p+'g_proj.weight', x)
        if self._arithmetic is not None:
            out = self._arithmetic.multiply(_fp32(out.ravel()), self._arithmetic.gate(gate_logits))
        else:
            out = _fp32(out.ravel()*self._ops.sigmoid(gate_logits))
        return self._linear(p + "o_proj.weight", out)

    def _mla_expanded(self, q, expansion, cache, *, projection_name=None):
        """Restore the publisher's FP32 K/V projection before attention scoring."""
        c = self.config
        if self._arithmetic is not None:
            if projection_name is None:
                raise ValueError('identified expanded MLA projection required')
            def occupied_pages():
                for page in cache.pages:
                    self._check_cancelled()
                    expanded=self._arithmetic.named_matrix(projection_name,self._weights[projection_name],
                        _fp32(page[:,:c.kvrank]),check_cancelled=self._check_cancelled).reshape(
                            len(page),c.heads,c.nope+c.value)
                    keys=np.empty((len(page),c.heads,c.nope+c.rope),np.float32)
                    keys[:,:,:c.nope]=expanded[:,:,:c.nope]
                    keys[:,:,c.nope:]=page[:,None,c.kvrank:]
                    yield keys,_fp32(expanded[:,:,c.nope:])
            if self._arithmetic.attention_mode=='paged-online-fp32':
                return self._arithmetic.attention_pages(_fp32(q),occupied_pages,
                    check_cancelled=self._check_cancelled)
            required = 4*cache.length*c.heads*(c.nope+c.rope+c.value)
            if required*4 > self._arithmetic.max_bytes:
                raise MemoryError('BF16 expanded attention exceeds arithmetic admission')
            key_pages, value_pages = [], []
            for keys,values in occupied_pages():
                key_pages.append(keys)
                value_pages.append(values)
            return self._arithmetic.attention(_fp32(q), _fp32(np.concatenate(key_pages)),
                                              _fp32(np.concatenate(value_pages)))
        matrix = None if expansion is None else expansion.reshape(c.heads * (c.nope + c.value), c.kvrank)
        peak = np.full(c.heads, -np.inf, np.float64)
        denominator = np.zeros(c.heads, np.float64)
        numerator = np.zeros((c.heads, c.value), np.float64)
        dp = C.POINTER(C.c_double)
        for page in cache.pages:
            self._check_cancelled()
            if matrix is None:
                expanded=self._native_projections.linear(projection_name,_fp32(page[:,:c.kvrank]),
                    max_work_bytes=self._native_projection_work_bytes,
                    cancelled=lambda:bool(getattr(self._cancel_local,'event',None) is not None and self._cancel_local.event.is_set()))
            else:expanded = _fp32(np.stack([self._matrix(matrix, row[:c.kvrank]) for row in page]))
            expanded = expanded.reshape(len(page), c.heads, c.nope + c.value)
            keys = np.empty((len(page), c.heads, c.nope + c.rope), np.float32)
            keys[:, :, :c.nope] = expanded[:, :, :c.nope]
            keys[:, :, c.nope:] = page[:, None, c.kvrank:]
            values = _fp32(expanded[:, :, c.nope:])
            status = self._segment(len(page), c.heads, c.nope + c.rope, c.value,
                                   *(self._ptr(a) for a in (q, keys, values)),
                                   *(a.ctypes.data_as(dp) for a in (peak, denominator, numerator)))
            if status:
                raise ValueError("expanded native MLA attention failed")
        return _fp32(numerator / denominator[:, None])

    def _mla_compressed(self, q, expansion, cache):
        """Explicit reassociated policy; not the publisher projection graph."""
        c = self.config
        q_latent = _fp32(np.stack([self._matrix(expansion[h, :c.nope].T, q[h, :c.nope]) for h in range(c.heads)]))
        q_rope = _fp32(q[:, c.nope:])
        peak, denominator = np.full(c.heads, -np.inf, np.float64), np.zeros(c.heads, np.float64)
        numerator = np.zeros((c.heads, c.kvrank), np.float64)
        dp = C.POINTER(C.c_double)
        for page in cache.pages:
            status = self._latent_segment(len(page), c.heads, c.kvrank, c.rope, (c.nope + c.rope) ** -0.5,
                                          *(self._ptr(a) for a in (q_latent, q_rope, page)),
                                          *(a.ctypes.data_as(dp) for a in (peak, denominator, numerator)))
            if status:
                raise ValueError("compressed native MLA attention failed")
        weighted = _fp32(numerator / denominator[:, None])
        out = _fp32(np.stack([self._matrix(expansion[h, c.nope:], weighted[h]) for h in range(c.heads)]))
        return out

    def _kda_forward(self, x, p, mixer):
        c = self.config
        shape = (1, c.kda_heads, c.kda_dim)
        qkv = [self._linear(p + n + "_proj.weight", x).reshape(shape) for n in ("q", "k", "v")]
        decay = self._linear(p + "f_b_proj.weight", self._linear(p + "f_a_proj.weight", x))
        if self._fla_policy is not None:
            torch = mixer._torch
            def device_tensor(value, beta=False):
                value = value.reshape((1, 1, c.kda_heads) if beta else
                                      (1, 1, c.kda_heads, c.kda_dim))
                return torch.tensor(value, device=mixer.device, dtype=torch.float32)
            mixed = mixer.forward(*(device_tensor(v) for v in qkv), device_tensor(decay),
                device_tensor(self._linear(p+'b_proj.weight', x), beta=True),
                device_tensor(self._linear(p+'g_proj.weight', x)),
                cancel=getattr(self._cancel_local, 'event', None))
            return self._linear(p+'o_proj.weight', mixed.float().cpu().numpy().ravel())
        z = (decay.astype(np.float64) + self._weights[p + "dt_bias"]) * np.repeat(
            np.exp(self._weights[p + "A_log"].astype(np.float64)), c.kda_dim)
        if not np.isfinite(z).all():
            raise ValueError("nonfinite KDA gate logits")
        decay = _fp32(-5.0 * self._ops.sigmoid(_fp32(z))).reshape(shape)
        beta = self._ops.sigmoid(self._linear(p + "b_proj.weight", x)).reshape(1, c.kda_heads)
        gate = self._ops.sigmoid(self._linear(p + "g_proj.weight", x)).reshape(shape)
        mixed = mixer.forward(*qkv, decay, beta, gate)
        return self._linear(p + "o_proj.weight", mixed.ravel())

    def _check_cancelled(self):
        event = getattr(self._cancel_local, 'event', None)
        if event is not None and event.is_set():
            from ..execution_service import ExecutionCancelled
            raise ExecutionCancelled('K3 execution cancelled before transaction commit')

    def _kda_sequence(self, values, p, mixer):
        """Project one occupied prompt chunk and invoke FLA chunk recurrence."""
        if self._fla_policy is None or self._arithmetic is None:
            raise ValueError('chunk recurrence requires explicit FLA/BF16 policies')
        c, n = self.config, len(values)
        torch = mixer._torch
        def project(name, rows):
            self._check_cancelled()
            return self._linear_sequence(p+name+'.weight', _fp32(rows))
        def tensor(rows, beta=False):
            shape = (1, n, c.kda_heads) if beta else (1, n, c.kda_heads, c.kda_dim)
            return torch.tensor(rows.reshape(shape), device=mixer.device, dtype=torch.bfloat16)
        q, k, v = (tensor(project(name+'_proj', values)) for name in ('q', 'k', 'v'))
        decay = tensor(project('f_b_proj', project('f_a_proj', values)))
        beta = tensor(project('b_proj', values), beta=True).float()
        gate = tensor(project('g_proj', values))
        mixed = mixer.forward(q, k, v, decay, beta, gate,
            cancel=getattr(self._cancel_local, 'event', None))
        self._check_cancelled()
        return project('o_proj', mixed.float().cpu().numpy().reshape(n, -1))

    def _linear_sequence(self, name, values):
        self._check_cancelled()
        values = _fp32(values)
        if self._arithmetic is None:
            return _fp32(np.stack([self._linear(name, row) for row in values]))
        if name in self._weights:
            return self._arithmetic.named_matrix(name, self._weights[name], values,
                router=name.endswith('gate.weight'))
        if '.experts.' not in name:
            raise ValueError('missing dense batch projection weight')
        prefix = name.removesuffix('weight')
        packed = self._weights[prefix+'weight_packed']
        return self._arithmetic.packed_matrix(packed, self._weights[prefix+'weight_scale'],
            values, rows=packed.shape[0], cols=packed.shape[1]*2,
            cancel=getattr(self._cancel_local, 'event', None))

    def _mlp_sequence(self, values, p, expert=False):
        c = self.config
        gate_name, up_name, down_name = ('w1', 'w3', 'w2') if expert else ('gate_proj', 'up_proj', 'down_proj')
        gate = self._linear_sequence(p+gate_name+'.weight', values)
        up = self._linear_sequence(p+up_name+'.weight', values)
        active = self._arithmetic.situ(gate, up, c.situ_beta, c.situ_linear_beta)
        return self._linear_sequence(p+down_name+'.weight', active)

    def _moe_sequence(self, values, p):
        """Batch router, expert-sorted dispatch, and publisher slot reduction."""
        c, n = self.config, len(values)
        logits = self._linear_sequence(p+'gate.weight', values)
        scores = self._arithmetic.router_sigmoid(logits)
        correction = self._arithmetic.cast(self._weights[p+'gate.e_score_correction_bias'])
        route = self._routing.route_batch(scores, correction, selected=c.selected,
            groups=c.expert_groups, selected_groups=c.selected_groups, scale=c.routed_scale)
        self._observe_route(p,logits,route.ids,route.weights)
        extra = 16*n*c.selected*c.latent+16*n*c.latent
        if self._admit(n)+extra > self.max_bytes:
            raise MemoryError('batched expert dispatch exceeds transaction admission')
        latent = self._linear_sequence(p+'routed_expert_down_proj.weight', values)
        order = self._routing.expert_order(route.ids)
        flat_ids = route.ids.reshape(-1)
        outputs = np.empty((n*c.selected, c.latent), np.float32)
        start = 0
        while start < len(order):
            self._check_cancelled()
            expert = int(flat_ids[order[start]])
            stop = start+1
            while stop < len(order) and flat_ids[order[stop]] == expert:
                stop += 1
            slots = order[start:stop]
            incoming = _fp32(latent[slots//c.selected])
            if self._expert_provider is None:
                outgoing = self._mlp_sequence(incoming, p+f'experts.{expert}.', True)
            else:
                layer = int(p.split('.')[3])
                with self._expert_provider.lease(layer, expert) as tables:
                    outgoing = self._provider_mlp(incoming, tables)
            outputs[slots] = outgoing
            start = stop
        routed = self._routing.aggregate_batch(outputs.reshape(n, c.selected, c.latent), route.weights)
        routed = _fp32(np.stack([self._norm(row, p+'routed_expert_norm.weight') for row in routed]))
        result = self._linear_sequence(p+'routed_expert_up_proj.weight', routed)
        return self._add(result, self._mlp_sequence(values, p+'shared_experts.')) if c.shared else result

    def _prefill_layers(self, embedded, positions, mla, kda):
        """Layer-major prompt transaction with explicitly selected attention.

        FLA handles the prompt chunk in one recurrence call. Selected FA2 uses
        batched MLA; otherwise MLA retains its existing causal-prefix calls.
        All mutable caches are supplied by the caller's staged transaction.
        """
        c, n = self.config, len(embedded)
        extra = 64*n*(c.hidden*(len(c.layer_types)//c.residual_block+5)+c.kda_heads*c.kda_dim)
        if self._admit(n)+extra > self.max_bytes:
            raise MemoryError('layer-major prefill buffers exceed transaction admission')
        x, blocks = embedded, []
        def rows(function, values):
            output = []
            for row in values:
                self._check_cancelled()
                output.append(function(row))
            return _fp32(np.stack(output))
        def residual(values, prefix):
            output = []
            for t, row in enumerate(values):
                self._check_cancelled()
                output.append(self._residual(row, [b[t] for b in blocks], prefix))
            return _fp32(np.stack(output))
        for i, kind in enumerate(c.layer_types):
            self._check_cancelled()
            p = f'language_model.model.layers.{i}.'
            prefix = x
            mixed_input = residual(prefix, p+'self_attention_res') if blocks else prefix
            if i % c.residual_block == 0:
                blocks.append(prefix)
                prefix = None
            normalized = rows(lambda row: self._norm(row, p+'input_layernorm.weight'), mixed_input)
            if kind == 'kda':
                mixed = self._kda_sequence(normalized, p+'self_attn.', kda[i])
            elif self._arithmetic._attention_backend is not None:
                mixed = self._mla_sequence(normalized, p+'self_attn.', mla[i], positions)
            else:
                mixed_rows = []
                for t, row in enumerate(normalized):
                    self._check_cancelled()
                    mixed_rows.append(self._mla_forward(row, p+'self_attn.', mla[i],
                        self._position+t, int(positions[t])))
                mixed = _fp32(np.stack(mixed_rows))
            prefix = mixed if prefix is None else self._add(prefix, mixed)
            feed = rows(lambda row: self._norm(row, p+'post_attention_layernorm.weight'),
                        residual(prefix, p+'mlp_res'))
            out = (self._mlp_sequence(feed, p+'mlp.') if i < c.dense_prefix
                   else self._moe_sequence(feed, p+'block_sparse_moe.'))
            x = self._add(prefix, out)
        x = residual(x, 'language_model.model.output_attn_res')
        x = rows(lambda row: self._norm(row, 'language_model.model.norm.weight'), x)
        self._check_cancelled()
        return self._linear_sequence('language_model.lm_head.weight', x)

    def _mla_sequence(self, values, p, cache, positions):
        """Append a staged prompt chunk and apply bottom-right causal FA2."""
        c, n = self.config, len(values)
        if cache.length != self._position or len(positions) != n:
            raise ValueError('MLA prefill prefix position mismatch')
        def project(name, rows):
            self._check_cancelled()
            return self._linear_sequence(p+name+'.weight', _fp32(rows))
        def norm(name, rows):
            return _fp32(np.stack([self._norm(row, p+name+'.weight', c.projection_epsilon)
                                  for row in rows]))
        query = project('q_b_proj', norm('q_a_layernorm', project('q_a_proj', values)))
        query = query.reshape(n, c.heads, c.nope+c.rope)
        kv = project('kv_a_proj_with_mqa', values)
        latent = norm('kv_a_layernorm', kv[:, :c.kvrank])
        rotary_keys = np.empty((n, c.rope), np.float32)
        for t in range(n):
            self._check_cancelled()
            query[t, :, c.nope:] = self._rotate(query[t, :, c.nope:], int(positions[t]))
            rotary_keys[t] = self._rotate(kv[t, c.kvrank:], int(positions[t]))
        required = 24*(cache.length+n)*(c.kvrank+c.rope+c.heads*(c.nope+c.rope+c.value))
        if required > self._arithmetic.max_bytes:
            raise MemoryError('occupied MLA prefill expansion exceeds arithmetic admission')
        cache.append_many(latent, rotary_keys)
        joined = _fp32(np.concatenate(cache.pages))
        expanded = project('kv_b_proj', joined[:, :c.kvrank]).reshape(
            cache.length, c.heads, c.nope+c.value)
        keys = np.empty((cache.length, c.heads, c.nope+c.rope), np.float32)
        keys[:, :, :c.nope] = expanded[:, :, :c.nope]
        keys[:, :, c.nope:] = joined[:, None, c.kvrank:]
        attended = self._arithmetic.attention_sequence(query, keys, _fp32(expanded[:, :, c.nope:]))
        gated = self._arithmetic.multiply(attended.reshape(n, -1),
                                         self._arithmetic.gate(project('g_proj', values)))
        return project('o_proj', gated)

    @property
    def position(self):
        with self._operation():
            return self._position

    def write_cache(self, output, *, max_bytes, scratch_directory=None):
        from ..file_cache_transport import transport_cache
        from .checkpoints import save_k3_checkpoint
        return transport_cache(self,output,save_k3_checkpoint,max_bytes=max_bytes,
                               scratch_directory=scratch_directory)

    def restore_cache(self, source, *, max_bytes, scratch_directory=None):
        from ..file_cache_transport import transport_cache
        from .checkpoints import load_k3_checkpoint
        return transport_cache(self,source,load_k3_checkpoint,max_bytes=max_bytes,
                               restoring=True,scratch_directory=scratch_directory)

    def reset(self):
        with self._operation():
            self._admit(0)
            c = self.config
            mla = {i: MLALatentPages(c.kvrank, c.rope)
                   for i, kind in enumerate(c.layer_types) if kind == "mla"}
            staged = {i: m.fork() if self._fla_policy is not None else copy.copy(m)
                      for i, m in self._kda.items()}
            for m in staged.values():
                m.reset()
            self._mla, self._kda, self._position = mla, staged, 0
            self._exact_media_case=None
            self._comparison_observations=None

    def snapshot(self):
        with self._operation():
            self._admit(0)
            return K3Snapshot(self._identity, self._position,
                              tuple((i, pages.snapshot()) for i, pages in self._mla.items()),
                              tuple((i, m.snapshot()) for i, m in self._kda.items()),
                              None if self._exact_evidence is None else self._exact_evidence.identity,
                              self._exact_media_case)

    def restore(self, snapshot):
        with self._operation():
            if type(snapshot) is not K3Snapshot or snapshot.identity != self._identity:
                raise ValueError("foreign K3 snapshot")
            if type(snapshot.position) is not int or not 0 <= snapshot.position <= self.config.capacity:
                raise ValueError("invalid snapshot position")
            expected=None if self._exact_evidence is None else self._exact_evidence.identity
            if snapshot.exact_evidence!=expected or (expected is None and snapshot.media_case is not None):
                raise ValueError('snapshot exact evidence or media policy differs')
            if self._exact_evidence is not None:self._exact_evidence.admit_snapshot(snapshot.media_case,snapshot.position)
            self._admit(max(0, snapshot.position - self._position))
            mla, kda = {}, {}
            for i, pages in snapshot.mla:
                if i not in self._mla or i in mla:
                    raise ValueError("invalid MLA snapshot inventory")
                mla[i] = self._mla[i].from_snapshot(pages, snapshot.position)
            for i, item in snapshot.kda:
                if i not in self._kda or i in kda:
                    raise ValueError("invalid KDA snapshot inventory")
                if self._fla_policy is not None:
                    from .fla_mixer import FLAMixerSnapshot
                    if type(item) is not FLAMixerSnapshot or item.position != snapshot.position:
                        raise ValueError('invalid FLA snapshot position or layout')
                    m = self._kda[i].fork()
                    m.restore(item)
                    kda[i] = m
                    continue
                if type(item) is not tuple or len(item) != 3:
                    raise ValueError('invalid native KDA snapshot layout')
                position, state, history = item
                m = copy.copy(self._kda[i])
                if type(position) is not int or position != snapshot.position or type(history) is not tuple or len(history) != 3:
                    raise ValueError("invalid KDA snapshot position/history")
                for a, expected in zip((state, *history), (m._state, *m._history)):
                    if not isinstance(a, np.ndarray) or a.shape != expected.shape:
                        raise ValueError("invalid KDA snapshot shape")
                    NativeKimiKDA._buffer(a)
                m._state, m._history, m._position = state.copy(), tuple(a.copy() for a in history), position
                kda[i] = m
            if set(mla) != set(self._mla) or set(kda) != set(self._kda):
                raise ValueError("incomplete snapshot")
            self._mla, self._kda, self._position = mla, kda, snapshot.position
            self._exact_media_case=snapshot.media_case
            self._comparison_observations=None

    def forward(self, token_ids, *, position_ids=None, projected_media=None,
                media_token_id=None, pixel_values=None, grid_thws=None, images=None,
                attention_mask=None, media_items=None):
        """Return [expanded_tokens,vocab] logits, atomically updating caches.

        Pixels use MoonViT processor patch layout and expand media placeholders.
        projected_media remains a separate one-row-per-placeholder interface.
        A binary one-dimensional attention_mask removes padding before all model
        operations; returned rows correspond to occupied (then expanded) tokens.
        This is causal padding, not an arbitrary pairwise attention matrix.
        """
        with self._operation():
            self._comparison_observations=None
            c = self.config
            if isinstance(token_ids,(tuple,list,range)) and not 1<=len(token_ids)<=4096:
                raise ValueError('bounded original token sequence required before conversion')
            ids = np.asarray(token_ids)
            if self._exact_evidence is not None:
                self._exact_evidence.admit_call(position=self._position,tokens=ids.size,
                    has_media=any(value is not None for value in (projected_media,pixel_values,images,media_items)),
                    media_case=self._exact_media_case)
            if ids.ndim != 1 or ids.dtype.kind not in "iu" or not 1 <= len(ids) <= 4096:
                raise ValueError("one-dimensional integer token IDs required")
            ids = ids.copy()
            if attention_mask is not None:
                if isinstance(attention_mask,(tuple,list,range)) and len(attention_mask)!=len(ids):
                    raise ValueError('padding mask extent differs before conversion')
                mask = np.asarray(attention_mask)
                if (mask.shape != ids.shape or mask.dtype.kind not in 'biu' or
                        np.any((mask != 0) & (mask != 1))):
                    raise ValueError('binary one-dimensional padding mask required')
                occupied = np.array(mask, dtype=bool, copy=True)
                if position_ids is not None:
                    positions = np.asarray(position_ids)
                    if positions.shape == ids.shape and pixel_values is None and images is None and media_items is None:
                        position_ids = np.array(positions[occupied], copy=True)
                ids = ids[occupied]
                if not len(ids):
                    if any(value is not None for value in
                           (projected_media, pixel_values, grid_thws, images, media_items)):
                        raise ValueError('fully padded input cannot supply media')
                    return np.empty((0, c.vocab), np.float32)
            if np.any(ids >= c.vocab) or np.any(ids < 0):
                raise ValueError('occupied token IDs must belong to the vocabulary')
            if media_token_id is not None and (type(media_token_id) is not int or not 0 <= media_token_id < c.vocab):
                raise ValueError("invalid media placeholder ID")
            count = int(np.count_nonzero(ids == media_token_id)) if media_token_id is not None else 0
            if images is not None or pixel_values is not None or projected_media is not None or media_items is not None:
                if not count:
                    raise ValueError("media input requires an explicit matching placeholder")
            if media_items is not None:
                from .video_input import prepare_media
                if (self._vision is None or images is not None or pixel_values is not None or
                        grid_thws is not None or projected_media is not None):
                    raise ValueError('mixed decoded media requires vision and no other media input')
                if c.vision.patch != 14 or c.vision.merge != (2, 2):
                    raise ValueError('decoded media requires pinned patch14/merge2 geometry')
                if not isinstance(media_items, (tuple, list)) or len(media_items) != count:
                    raise ValueError('one decoded media item per occupied placeholder required')
                def cancelled():
                    self._check_cancelled()
                    return False
                prepared = prepare_media(media_items,max_bytes=self.max_bytes-self._admit(len(ids)),
                                         cancelled=cancelled)
                pixel_values,grid_thws = prepared.pixels,prepared.grids
            if images is not None:
                if self._vision is None or pixel_values is not None or grid_thws is not None or projected_media is not None:
                    raise ValueError("images require configured vision and no other media input")
                if c.vision.patch != 14 or c.vision.merge != (2, 2):
                    raise ValueError("raw image preprocessing requires pinned patch14/merge2")
                if not isinstance(images, (list, tuple)) or len(images) != count:
                    raise ValueError("one image per media placeholder required")
                def preprocessing_cancelled():
                    self._check_cancelled();return False
                pixel_values, grid_thws = preprocess_images(images,
                    max_bytes=self.max_bytes - self._admit(len(ids)),cancelled=preprocessing_cancelled)
            if pixel_values is not None and (self._vision is None or grid_thws is None or projected_media is not None):
                raise ValueError("pixels require configured MoonViT3d, grids and no projected_media")
            if pixel_values is None and grid_thws is not None:
                raise ValueError("grid_thws without pixels")
            n = len(ids)
            if pixel_values is not None:
                lengths = self._vision.projected_lengths(grid_thws)
                if count != len(lengths):
                    raise ValueError("one media sequence per placeholder required")
                n = len(ids) - count + sum(lengths)
            selected_case=self._exact_media_case;media_retained=0
            if self._exact_evidence is not None:
                has_media=any(value is not None for value in (projected_media,pixel_values,images,media_items))
                if has_media:
                    if not isinstance(pixel_values,np.ndarray) or grid_thws is None:
                        raise ValueError('exact media requires original typed processor buffers')
                    original_grids=np.asarray(grid_thws)
                    media_retained=2*(pixel_values.nbytes+original_grids.nbytes)
                    if 2*media_retained>self.max_bytes-self._admit(n):
                        raise MemoryError('exact immutable processor capture exceeds runtime admission')
                    pixel_values=np.frombuffer(pixel_values.tobytes(order='C'),dtype=pixel_values.dtype).reshape(pixel_values.shape)
                    grid_thws=np.frombuffer(original_grids.tobytes(order='C'),dtype=original_grids.dtype).reshape(original_grids.shape)
                    selected_case=self._exact_evidence.admit_media(position=self._position,token_ids=ids,
                        media_token_id=media_token_id,pixels=pixel_values,grids=grid_thws,expanded_tokens=n)
                else:
                    self._exact_evidence.admit_call(position=self._position,tokens=n,has_media=False,
                        media_case=selected_case)
            if self._position + n > c.capacity:
                raise ValueError("K3 cache capacity exceeded")
            self._admit(n)
            if isinstance(position_ids,(tuple,list,range)) and len(position_ids)!=n:
                raise ValueError('expanded rotary position extent differs before conversion')
            pos = np.arange(self._position, self._position + n) if position_ids is None else np.array(position_ids, copy=True)
            if pos.shape != (n,) or pos.dtype.kind not in "iu" or np.any(pos < 0) or np.any(pos >= c.capacity):
                raise ValueError("invalid rotary positions")
            if self._exact_evidence is not None and not np.array_equal(pos,np.arange(self._position,self._position+n)):
                raise ValueError('exact evidence requires its recorded contiguous occupied rotary positions')
            features = (self._vision.encode(pixel_values, grid_thws, max_bytes=self.max_bytes - self._admit(n)-media_retained)
                        if pixel_values is not None else None)
            embedding_name='language_model.model.embed_tokens.weight'
            if type(self._weights[embedding_name]) is _ProjectionReference:
                embedded=self._arithmetic._projection_backend.gather_rows(embedding_name,ids)
            else:embedded = self._weights[embedding_name][ids].copy()
            if self._arithmetic is not None:
                embedded = self._arithmetic.cast(embedded)
            if features is not None:
                rows, feature_index = [], 0
                for token, row in zip(ids, embedded):
                    if token == media_token_id:
                        rows.append(features[feature_index])
                        feature_index += 1
                    else:
                        rows.append(row[None, :])
                embedded = _fp32(np.concatenate(rows))
            elif projected_media is not None:
                if type(media_token_id) is not int or not 0 <= media_token_id < c.vocab:
                    raise ValueError("explicit media placeholder ID required")
                media = np.asarray(projected_media)
                if media.shape != (count, c.hidden) or count == 0:
                    raise ValueError("one projected row per media placeholder required")
                NativeKimiKDA._buffer(media)
                embedded[ids == media_token_id] = media
            elif media_token_id is not None and np.any(ids == media_token_id):
                raise ValueError("media placeholder lacks projected features")
            mla = {i: pages.fork() for i, pages in self._mla.items()}
            kda = {i: m.fork() if self._fla_policy is not None else copy.copy(m)
                   for i, m in self._kda.items()}
            if n > 1 and self._fla_policy is not None and self._arithmetic is not None:
                logits = self._prefill_layers(embedded, pos, mla, kda)
                self._check_cancelled()
                self._mla, self._kda, self._position = mla, kda, self._position+n
                self._exact_media_case=selected_case
                return logits
            logits = np.empty((n, c.vocab), np.float32)
            for t, x in enumerate(embedded):
                self._check_cancelled()
                blocks = []
                for i, kind in enumerate(c.layer_types):
                    p = f"language_model.model.layers.{i}."
                    prefix = x
                    mixed_input = self._residual(prefix, blocks, p + "self_attention_res") if blocks else prefix
                    if i % c.residual_block == 0:
                        blocks.append(prefix)
                        prefix = None
                    mixed_input = self._norm(mixed_input, p + "input_layernorm.weight")
                    a = p + "self_attn."
                    mixed = (self._mla_forward(mixed_input, a, mla[i], self._position + t, int(pos[t]))
                             if kind == "mla" else self._kda_forward(mixed_input, a, kda[i]))
                    prefix = mixed if prefix is None else self._add(prefix, mixed)
                    ff = self._norm(self._residual(prefix, blocks, p + "mlp_res"), p + "post_attention_layernorm.weight")
                    out = self._mlp(ff, p + "mlp.") if i < c.dense_prefix else self._moe(ff, p + "block_sparse_moe.")
                    x = self._add(prefix, out)
                x = self._residual(x, blocks, "language_model.model.output_attn_res")
                x = self._norm(x, "language_model.model.norm.weight")
                logits[t] = self._linear("language_model.lm_head.weight", x)
            self._check_cancelled()
            self._mla, self._kda, self._position = mla, kda, self._position + n
            self._exact_media_case=selected_case
            return logits

    def encode_media(self, pixel_values, grid_thws):
        """Encode processor patches through MoonViT and patchmergerv2, no cache mutation."""
        with self._operation():
            if self._vision is None:
                raise ValueError("MoonViT3d is not configured")
            return self._vision.encode(pixel_values, grid_thws, max_bytes=self.max_bytes - self._admit(0))
