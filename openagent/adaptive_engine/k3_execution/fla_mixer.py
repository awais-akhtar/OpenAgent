"""Opt-in projected K3 mixer using explicitly pinned installed FLA kernels."""
from dataclasses import dataclass
from importlib.metadata import version
from threading import RLock
from contextlib import contextmanager
import hashlib
import json
import copy
from .fla_contract import admit_fla_sources
from .precision import check_precision


@dataclass(frozen=True)
class FLAMixerSnapshot:
    identity: str
    position: int
    recurrent: object
    convolution: tuple


@dataclass(frozen=True)
class FLAK3Policy:
    device: str
    torch_version: str
    fla_version: str
    max_bytes: int

    def __post_init__(self):
        if (any(type(x) is not str or not x for x in
                (self.device, self.torch_version, self.fla_version))
                or type(self.max_bytes) is not int or self.max_bytes <= 0):
            raise ValueError('explicit device, dependency pins and FLA budget required')


class FLAK3Mixer:
    """Own BF16 convolutions and raw transposed FLA recurrent state.

    Inputs are already projected device tensors. Gate logits are raw, not
    pre-sigmoid probabilities or precomputed logarithmic decay. This component
    does not implement projections, MLA, routing, or whole-model admission.
    """

    def __init__(self, taps, norm, a_log, dt_bias, *, device, capacity,
                 max_bytes, torch_version, fla_version, epsilon=1e-5,
                 lower_bound=-5.0):
        import torch
        from fla.modules import ShortConvolution, FusedRMSNormGated
        from fla.ops.kda import chunk_kda, fused_recurrent_kda
        if str(torch.__version__) != torch_version or version('fla-core') != fla_version:
            raise ValueError('installed Torch/FLA versions differ from explicit pins')
        source_revision = admit_fla_sources(ShortConvolution, chunk_kda, fused_recurrent_kda)
        target = torch.device(device)
        if target.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('this FLA adapter requires an available CUDA device; no CPU fallback')
        if target.index is None:
            target = torch.device('cuda', torch.cuda.current_device())
        if target.index >= torch.cuda.device_count():
            raise ValueError('CUDA device index is unavailable')
        if any(type(x) is not int or x <= 0 for x in (capacity, max_bytes)):
            raise ValueError('positive capacity and memory budget required')
        if len(taps) != 3 or any(x.ndim != 3 or x.shape != taps[0].shape for x in taps):
            raise ValueError('three matching head/channel/tap tensors required')
        heads, channels, kernel = taps[0].shape
        if not heads or not channels or not kernel or norm.shape != (channels,):
            raise ValueError('invalid convolution or output norm geometry')
        if a_log.shape != (heads,) or dt_bias.shape != (heads*channels,):
            raise ValueError('invalid FLA gate parameter geometry')
        import math
        if (not math.isfinite(epsilon) or epsilon <= 0 or
                (lower_bound is not None and
                 (not math.isfinite(lower_bound) or lower_bound >= 0))):
            raise ValueError('invalid norm epsilon or safe-gate lower bound')
        self._torch, self.device = torch, target
        check_precision(torch, target)
        self.capacity, self.max_bytes = capacity, max_bytes
        self.heads, self.channels, self.kernel = heads, channels, kernel
        self._lock, self._closed = RLock(), False
        self._chunk, self._recurrent = chunk_kda, fused_recurrent_kda
        self.lower_bound = lower_bound
        weight_bytes = 2*(3*heads*channels*kernel+channels)+4*(heads+heads*channels)
        cache_bytes = 4*heads*channels*channels+6*heads*channels*kernel
        self._resident = weight_bytes+cache_bytes
        if 3*self._resident > max_bytes:
            raise MemoryError('FLA weight/cache staging exceeds admission')
        digest = hashlib.sha256(json.dumps([torch_version, fla_version, source_revision, str(target),
            capacity, epsilon, lower_bound], separators=(',', ':')).encode())
        for source in (*taps, norm, a_log, dt_bias):
            array = torch.as_tensor(source, dtype=torch.float32, device='cpu').contiguous()
            if not bool(torch.isfinite(array).all()):
                raise ValueError('finite FLA parameters required')
            digest.update(array.numpy().tobytes())
        self.identity = digest.hexdigest()
        self._conv = []
        for weights in taps:
            module = ShortConvolution(hidden_size=heads*channels,
                kernel_size=kernel, activation='silu', bias=False, backend='triton').to(target, dtype=torch.bfloat16)
            if module.backend != 'triton' or module.kernel_size != (kernel,):
                raise ValueError('FLA convolution backend or kernel differs from reviewed contract')
            with torch.no_grad():
                if module.weight.numel() != weights.size:
                    raise ValueError('installed FLA convolution weight geometry differs')
                module.weight.copy_(torch.as_tensor(weights, device=target,
                    dtype=torch.bfloat16).reshape(module.weight.shape))
            module.eval()
            self._conv.append(module)
        self._norm = FusedRMSNormGated(channels, eps=epsilon, activation='sigmoid').to(
            target, dtype=torch.bfloat16)
        with torch.no_grad():
            self._norm.weight.copy_(torch.as_tensor(norm, device=target, dtype=torch.bfloat16))
        self._norm.eval()
        self._a = torch.as_tensor(a_log, device=target, dtype=torch.float32).clone()
        self._dt = torch.as_tensor(dt_bias, device=target, dtype=torch.float32).clone()
        self._state, self._history, self.position = None, (None, None, None), 0

    def _open(self):
        if self._closed:
            raise RuntimeError('FLA mixer is closed')

    @contextmanager
    def _operation(self):
        with self._lock:
            self._open()
            if getattr(self, '_busy', False):
                raise RuntimeError('reentrant K3 mixer operation')
            self._busy = True
            try:
                yield
            finally:
                self._busy = False

    def forward(self, q, k, v, gate_logits, beta_logits, output_gate, *, cancel=None):
        with self._operation(), self._torch.inference_mode():
            self._open()
            check_precision(self._torch, self.device)
            torch = self._torch
            inputs = (q, k, v, gate_logits, beta_logits, output_gate)
            if not isinstance(q, torch.Tensor) or q.ndim != 4:
                raise ValueError('projected tensors must be [1,tokens,heads,channels]')
            n = q.shape[1]
            expected = (1, n, self.heads, self.channels)
            if not n or self.position+n > self.capacity:
                raise ValueError('FLA context capacity exceeded')
            for index, tensor in enumerate(inputs):
                shape = expected[:-1] if index == 4 else expected
                if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                        or tensor.device != self.device or tensor.dtype not in (torch.float32, torch.bfloat16)
                        or not bool(torch.isfinite(tensor).all())):
                    raise ValueError('invalid projected FLA tensor')
            if 3*self._resident + 48*n*self.heads*self.channels > self.max_bytes:
                raise MemoryError('FLA transaction exceeds explicit buffer admission')
            def check():
                if cancel is not None and cancel.is_set():
                    from ..execution_service import ExecutionCancelled
                    raise ExecutionCancelled('FLA mixer cancelled')
            check()
            staged = tuple(None if x is None else x.clone() for x in self._history)
            filtered, history = [], []
            for module, x, cache in zip(self._conv, inputs[:3], staged):
                y, state = module(x=x.to(torch.bfloat16).reshape(1, n, -1),
                                  cache=cache, output_final_state=True)
                filtered.append(y.reshape(expected))
                history.append(state)
                # FLA kernels cannot be interrupted once dispatched. Poll
                # between the three independent convolution calls so a
                # cancelled request does not continue into later kernels.
                check()
            initial = None if self._state is None else self._state.clone()
            kwargs = dict(q=filtered[0], k=filtered[1], v=filtered[2],
                g=gate_logits.to(torch.bfloat16), beta=beta_logits.float(),
                A_log=self._a, dt_bias=self._dt, initial_state=initial,
                output_final_state=True, use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                lower_bound=self.lower_bound, transpose_state_layout=True)
            if n == 1:
                result, recurrent = self._recurrent(**kwargs)
            else:
                result, recurrent = self._chunk(**kwargs, safe_gate=self.lower_bound is not None)
            check()
            result = self._norm(result, output_gate.to(torch.bfloat16))
            if (not bool(torch.isfinite(result).all()) or recurrent is None
                    or not bool(torch.isfinite(recurrent).all())):
                raise ValueError('nonfinite or missing FLA result/state')
            check()
            self._state, self._history = recurrent, tuple(history)
            self.position += n
            return result

    def snapshot(self):
        with self._operation():
            self._open()
            return FLAMixerSnapshot(self.identity, self.position,
                None if self._state is None else self._state.clone(),
                tuple(None if x is None else x.clone() for x in self._history))

    def fork(self):
        with self._operation():
            self._open()
            result = copy.copy(self)
            result._lock = RLock()
            result._busy = False
            result._state = None if self._state is None else self._state.clone()
            result._history = tuple(None if x is None else x.clone() for x in self._history)
            # Modules and immutable parameter tensors are shared; mutable
            # recurrent/convolution state belongs to the new transaction.
            return result

    def reset(self):
        with self._operation():
            self._open()
            self._state, self._history, self.position = None, (None, None, None), 0

    def restore(self, snapshot):
        with self._operation(), self._torch.inference_mode():
            self._open()
            if (type(snapshot) is not FLAMixerSnapshot or snapshot.identity != self.identity
                    or type(snapshot.position) is not int
                    or not 0 <= snapshot.position <= self.capacity
                    or type(snapshot.convolution) is not tuple or len(snapshot.convolution) != 3):
                raise ValueError('foreign or invalid FLA snapshot')
            if snapshot.position == 0:
                if snapshot.recurrent is not None or any(x is not None for x in snapshot.convolution):
                    raise ValueError('empty FLA snapshot contains occupied state')
                self._state, self._history, self.position = None, (None, None, None), 0
                return
            torch = self._torch
            expected = ((1, self.heads, self.channels, self.channels),
                        *((1, self.heads*self.channels, self.kernel),)*3)
            tensors = (snapshot.recurrent, *snapshot.convolution)
            for index, (tensor, shape) in enumerate(zip(tensors, expected)):
                if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                        or tensor.device != self.device
                        or tensor.dtype != (torch.float32 if index == 0 else torch.bfloat16)
                        or not bool(torch.isfinite(tensor).all())):
                    raise ValueError('FLA snapshot state layout differs from admitted contract')
            if 3*self._resident > self.max_bytes:
                raise MemoryError('FLA restore staging exceeds admission')
            copies = tuple(x.clone() for x in tensors)
            self._state, self._history, self.position = copies[0], copies[1:], snapshot.position

    def close(self):
        with self._lock:
            if getattr(self, '_busy', False):
                raise RuntimeError('cannot close an active K3 mixer')
            self._state, self._history, self._conv = None, (), []
            self._norm = self._a = self._dt = None
            self._closed = True
