"""Metal KDA recurrence with explicit BF16 boundaries and raw FP32 state."""
from dataclasses import dataclass
from threading import RLock
import hashlib
import math
from pathlib import Path
from .device_policy import admit_device
from .fla_mixer import FLAK3Mixer
from .precision import check_precision


@dataclass(frozen=True)
class MetalK3Policy:
    device: str
    torch_version: str
    max_bytes: int

    def __post_init__(self):
        if (not isinstance(self.device, str) or not self.device.startswith('mps')
                or not isinstance(self.torch_version, str) or not self.torch_version
                or type(self.max_bytes) is not int or self.max_bytes <= 0):
            raise ValueError('explicit Metal device, Torch pin and KDA budget required')


class MetalK3Mixer(FLAK3Mixer):
    """Own recurrence implementation; FLA snapshot layout is a transport contract.

    Immutable weights are shared by transactional forks. Convolution history
    and value/key transposed recurrent matrices belong to each fork.
    """
    def __init__(self, taps, norm, a_log, dt_bias, *, device, capacity,
                 max_bytes, torch_version, epsilon=1e-5, lower_bound=-5.0):
        import torch
        if str(torch.__version__) != torch_version:
            raise ValueError('installed Torch differs from explicit Metal pin')
        target = admit_device(torch, device)
        if target.type != 'mps':
            raise ValueError('Metal KDA requires Metal arithmetic')
        if any(type(x) is not int or x <= 0 for x in (capacity, max_bytes)):
            raise ValueError('positive capacity and buffer budget required')
        if len(taps) != 3 or any(x.ndim != 3 or x.shape != taps[0].shape for x in taps):
            raise ValueError('matching head/channel/convolution taps required')
        heads, channels, kernel = taps[0].shape
        if (min(heads, channels, kernel) <= 0 or norm.shape != (channels,)
                or a_log.shape != (heads,) or dt_bias.shape != (heads*channels,)
                or not math.isfinite(epsilon) or epsilon <= 0
                or (lower_bound is not None and
                    (not math.isfinite(lower_bound) or lower_bound >= 0))):
            raise ValueError('invalid Metal KDA geometry or gate policy')
        self._torch, self.device = torch, target
        self.heads, self.channels, self.kernel = heads, channels, kernel
        self.capacity, self.max_bytes = capacity, max_bytes
        self.epsilon, self.lower_bound = epsilon, lower_bound
        self._lock, self._closed = RLock(), False
        self._resident = 2*(3*heads*channels*kernel+channels)+4*(heads+heads*channels)+4*heads*channels**2+6*heads*channels*kernel
        if 3*self._resident > max_bytes:
            raise MemoryError('Metal KDA weight/state staging exceeds admission')
        digest = hashlib.sha256(repr(('metal-kda-fp32-recurrence-bf16-v1',
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            torch_version, str(target), capacity, epsilon, lower_bound)).encode())
        sources = []
        for source in (*taps, norm, a_log, dt_bias):
            host = torch.as_tensor(source, device='cpu', dtype=torch.float32).contiguous()
            if not bool(torch.isfinite(host).all()):
                raise ValueError('finite KDA source parameters required')
            digest.update(host.numpy().tobytes())
            sources.append(host)
        self.identity = digest.hexdigest()
        self._conv = tuple(x.to(target, dtype=torch.bfloat16) for x in sources[:3])
        self._norm = sources[3].to(target, dtype=torch.bfloat16)
        self._a, self._dt = (x.to(target) for x in sources[4:])
        self._state, self._history, self.position = None, (None, None, None), 0

    def forward(self, q, k, v, gate_logits, beta_logits, output_gate, *, cancel=None):
        torch = self._torch
        with self._operation(), torch.inference_mode():
            self._open()
            check_precision(torch, self.device)
            if not isinstance(q, torch.Tensor) or q.ndim != 4:
                raise ValueError('projected KDA tensors must be [1,tokens,heads,channels]')
            n = q.shape[1]
            shape = (1, n, self.heads, self.channels)
            if n <= 0 or self.position+n > self.capacity:
                raise ValueError('Metal KDA context capacity exceeded')
            for i, x in enumerate((q, k, v, gate_logits, beta_logits, output_gate)):
                if (not isinstance(x, torch.Tensor) or tuple(x.shape) != (shape[:-1] if i == 4 else shape)
                        or x.device != self.device or x.dtype not in (torch.float32, torch.bfloat16)
                        or not bool(torch.isfinite(x).all())):
                    raise ValueError('invalid projected Metal KDA tensor')
            if 3*self._resident+64*n*self.heads*self.channels > self.max_bytes:
                raise MemoryError('Metal KDA transaction exceeds buffer admission')
            def check():
                if cancel is not None and cancel.is_set():
                    from ..execution_service import ExecutionCancelled
                    raise ExecutionCancelled('Metal KDA cancelled')
            check()
            histories = [torch.zeros((1,self.heads*self.channels,self.kernel),
                device=self.device,dtype=torch.bfloat16) if x is None else x.clone()
                for x in self._history]
            state = (torch.zeros((1,self.heads,self.channels,self.channels),device=self.device)
                     if self._state is None else self._state.clone())
            outputs = []
            for token in range(n):
                check()
                filtered = []
                for i, (value, taps) in enumerate(zip((q,k,v), self._conv)):
                    history = torch.cat((histories[i][...,1:],
                        value[:,token].to(torch.bfloat16).reshape(1,-1,1)),dim=-1)
                    histories[i] = history
                    conv = (history.float()*taps.reshape(1,-1,self.kernel).float()).sum(-1)
                    filtered.append(torch.nn.functional.silu(conv).to(torch.bfloat16).float().reshape(1,self.heads,self.channels))
                qt, kt, vt = filtered
                qt = qt*torch.rsqrt(qt.square().sum(-1,keepdim=True)+1e-6)
                kt = kt*torch.rsqrt(kt.square().sum(-1,keepdim=True)+1e-6)
                raw = gate_logits[:,token].to(torch.bfloat16).float()+self._dt.reshape(1,self.heads,self.channels)
                if self.lower_bound is None:
                    decay = -self._a.exp().reshape(1,-1,1)*torch.nn.functional.softplus(raw)
                else:
                    decay = self.lower_bound*torch.sigmoid(self._a.exp().reshape(1,-1,1)*raw)
                state = state*decay.exp().unsqueeze(-2)
                prediction = (state*kt.unsqueeze(-2)).sum(-1)
                delta = (vt-prediction)*torch.sigmoid(beta_logits[:,token].float()).unsqueeze(-1)
                state = state+delta.unsqueeze(-1)*kt.unsqueeze(-2)
                out = (state*qt.unsqueeze(-2)).sum(-1)*(self.channels**-0.5)
                rounded = out.to(torch.bfloat16).float()
                normalized = rounded*torch.rsqrt(rounded.square().mean(-1,keepdim=True)+self.epsilon)
                out = normalized*self._norm.float()*torch.sigmoid(output_gate[:,token].to(torch.bfloat16).float())
                outputs.append(out.to(torch.bfloat16))
            result = torch.stack(outputs,dim=1)
            if not bool(torch.isfinite(result).all()) or not bool(torch.isfinite(state).all()):
                raise ArithmeticError('nonfinite Metal KDA state or result')
            check()
            self._state,self._history,self.position = state,tuple(histories),self.position+n
            return result
