"""Explicit device-local PyTorch routing for the pinned K3 text contract."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from .precision import check_precision


@dataclass(frozen=True)
class K3Route:
    ids: np.ndarray
    weights: np.ndarray


class TorchK3Router:
    """Use the selected device's top-k implementation and tie ordering.

    This avoids substituting stable sorting for unsorted publisher top-k. CPU
    and CUDA tie ordering need not agree. Version identity is recorded, not
    treated as numerical certification of the rest of the decoder.
    """

    def __init__(self, *, max_bytes, expected_torch_version=None, device='cpu'):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('positive routing scratch budget required')
        import torch
        version = str(torch.__version__)
        if expected_torch_version is not None and version != expected_torch_version:
            raise ValueError('installed PyTorch differs from pinned routing version')
        from .device_policy import admit_device
        target = admit_device(torch, device, allow_cpu=True)
        if target.type == 'cuda':
            if not torch.cuda.is_available():
                raise ValueError('requested CUDA routing device is unavailable')
            target = torch.device('cuda', torch.cuda.current_device() if target.index is None else target.index)
            if target.index >= torch.cuda.device_count():
                raise ValueError('requested CUDA routing index is unavailable')
        self.device = target
        self._torch = torch
        check_precision(torch, target)
        self.max_bytes = max_bytes
        self.version = version
        self.identity = hashlib.sha256(json.dumps({
            'torch': version, 'device': str(target), 'dtype': 'float32',
            'contract': 'k3-unsorted-topk-slot-order-fp32-reduction-v1',
        }, sort_keys=True).encode('ascii')).hexdigest()

    def route(self, scores, correction, *, selected, groups, selected_groups,
              scale, renormalize=True):
        scores = np.asarray(scores)
        if scores.ndim != 1:
            raise ValueError('single-token routing requires a score vector')
        routed = self.route_batch(scores[None], correction, selected=selected, groups=groups,
            selected_groups=selected_groups, scale=scale, renormalize=renormalize)
        return K3Route(routed.ids[0], routed.weights[0])

    def route_batch(self, scores, correction, *, selected, groups, selected_groups,
                    scale, renormalize=True):
        check_precision(self._torch, self.device)
        scores, correction = np.asarray(scores), np.asarray(correction)
        if (scores.ndim != 2 or not all(scores.shape) or correction.shape != (scores.shape[-1],)
                or scores.dtype != np.float32 or correction.dtype != np.float32
                or not np.isfinite(scores).all() or not np.isfinite(correction).all()
                or np.any(scores < 0)):
            raise ValueError('finite nonnegative FP32 scores and matching correction required')
        tokens, experts = scores.shape
        for name, value in (('selected', selected), ('groups', groups),
                            ('selected_groups', selected_groups)):
            if type(value) is not int or value <= 0:
                raise ValueError('positive route geometry required: '+name)
        if (experts % groups or selected > experts or selected_groups > groups
                or selected > selected_groups*(experts//groups)
                or (groups > selected_groups and experts//groups < 2)):
            raise ValueError('invalid grouped routing geometry')
        if (type(renormalize) is not bool or isinstance(scale, bool)
                or not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0):
            raise ValueError('explicit boolean normalization and positive finite scale required')
        # Covers explicit score copies, masks, group scores, top-k outputs and
        # returned arrays. PyTorch allocator overhead is not a process-RSS bound.
        required = tokens*(48*experts + 32*groups + 48*selected)
        if required > self.max_bytes:
            raise MemoryError('routing buffers exceed admission')
        torch = self._torch
        with torch.inference_mode():
            original = torch.tensor(scores, dtype=torch.float32, device=self.device)
            bias = torch.tensor(correction, dtype=torch.float32, device=self.device)[None]
            choice = original + bias
            if not bool(torch.isfinite(choice).all()):
                raise ValueError('corrected routing scores overflow')
            if groups > selected_groups:
                grouped = choice.reshape(tokens, groups, experts//groups)
                group_score = torch.topk(grouped, 2, dim=-1).values.sum(dim=-1)
                if not bool(torch.isfinite(group_score).all()):
                    raise ValueError('group scores overflow')
                chosen_groups = torch.topk(group_score, selected_groups, dim=-1, sorted=False).indices
                admitted = torch.zeros((tokens, groups), dtype=torch.bool, device=self.device)
                admitted.scatter_(1, chosen_groups, True)
                choice = choice.masked_fill(~admitted.repeat_interleave(experts//groups, dim=-1), -float('inf'))
            ids = torch.topk(choice, selected, sorted=False).indices
            weights = original.gather(1, ids)
            if selected > 1 and renormalize:
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
            weights = weights * scale
            if not bool(torch.isfinite(weights).all()):
                raise ValueError('routing weights overflow')
            result_ids = ids.cpu().numpy().astype(np.uint64, copy=True)
            result_weights = weights.cpu().numpy().copy()
        result_ids.flags.writeable = result_weights.flags.writeable = False
        return K3Route(result_ids, result_weights)

    def expert_order(self, ids):
        check_precision(self._torch, self.device)
        ids = np.asarray(ids)
        if ids.ndim != 2 or not all(ids.shape) or ids.dtype != np.uint64:
            raise ValueError('batched route IDs required')
        if ids.size*32 > self.max_bytes or np.any(ids > np.iinfo(np.int64).max):
            raise MemoryError('expert dispatch ordering exceeds admission')
        with self._torch.inference_mode():
            values = self._torch.tensor(ids.astype(np.int64), device=self.device)
            return values.flatten().argsort().cpu().numpy().copy()

    def aggregate_batch(self, outputs, weights):
        check_precision(self._torch, self.device)
        outputs, weights = np.asarray(outputs), np.asarray(weights)
        if (outputs.ndim != 3 or not all(outputs.shape) or weights.shape != outputs.shape[:2]
                or outputs.dtype != np.float32 or weights.dtype != np.float32
                or not np.isfinite(outputs).all() or not np.isfinite(weights).all()):
            raise ValueError('finite batched slot outputs and weights required')
        if outputs.size*12+weights.size*8 > self.max_bytes:
            raise MemoryError('batched expert reduction exceeds admission')
        torch = self._torch
        with torch.inference_mode():
            values = torch.tensor(outputs, device=self.device)
            gains = torch.tensor(weights, device=self.device)
            result = (values*gains[..., None]).sum(dim=1)
            if not bool(torch.isfinite(result).all()):
                raise ValueError('batched expert reduction overflow')
            return result.cpu().numpy().copy()

    def status(self):
        return {'implementation': 'torch-device-unsorted-topk', 'torch_version': self.version,
                'device': str(self.device),
                'identity': self.identity, 'certified': False}

    def aggregate(self, outputs, weights):
        check_precision(self._torch, self.device)
        outputs, weights = np.asarray(outputs), np.asarray(weights)
        if (outputs.ndim != 2 or not all(outputs.shape) or weights.shape != (outputs.shape[0],)
                or outputs.dtype != np.float32 or weights.dtype != np.float32
                or not np.isfinite(outputs).all() or not np.isfinite(weights).all()):
            raise ValueError('finite slot-ordered FP32 outputs and weights required')
        required = 12*outputs.size + 8*weights.size + 8*outputs.shape[1]
        if required > self.max_bytes:
            raise MemoryError('expert reduction exceeds routing scratch admission')
        torch = self._torch
        with torch.inference_mode():
            values = torch.tensor(outputs, dtype=torch.float32, device=self.device)
            gains = torch.tensor(weights, dtype=torch.float32, device=self.device)
            result = (values[None]*gains[None, :, None]).sum(dim=1)[0]
            if not bool(torch.isfinite(result).all()):
                raise ValueError('expert reduction overflow')
            return result.cpu().numpy().copy()
