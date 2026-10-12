"""Resident K3 projection banks with explicit BF16/FP32 model boundaries."""
import hashlib
import math
import threading

import numpy as np


class K3DeviceProjections:
    """Own admitted device matrices; activations and results remain on device.

    The caller pins Torch and device policy before construction. Router weights
    and products use FP32. Ordinary products round inputs/weights to BF16 and
    return BF16 boundary values represented as FP32 device tensors.
    """
    def __init__(self,torch,tensors,*,device,torch_version,max_bytes,router_names=()):
        if str(torch.__version__)!=torch_version or type(max_bytes) is not int or max_bytes<=0:
            raise ValueError('explicit Torch pin and resident projection budget required')
        from .device_policy import admit_device
        target = admit_device(torch,device)
        supplied = dict(tensors)
        routers = frozenset(router_names)
        if not supplied or not routers.issubset(supplied) or any(type(name) is not str or not name for name in supplied):
            raise ValueError('named complete projection inventory required')
        shapes = {}
        for name,value in supplied.items():
            value = np.asarray(value)
            if value.ndim!=2 or not all(value.shape) or value.dtype!=np.float32 or not np.isfinite(value).all():
                raise ValueError('finite exact FP32 projection source required: '+name)
            shapes[name] = value.shape
        charge = sum(math.prod(shape)*(4 if name in routers else 2) for name,shape in shapes.items())
        source = sum(math.prod(shape)*4 for shape in shapes.values())
        if charge+2*source>max_bytes:
            raise MemoryError('K3 resident projection source/device admission exceeded')
        digest = hashlib.sha256(repr((torch_version,str(target),'k3-resident-bf16-router-fp32-v1',sorted(routers))).encode())
        weights = {}
        for name in sorted(supplied):
            value = np.array(supplied[name],dtype=np.float32,copy=True,order='C')
            digest.update(name.encode());digest.update(memoryview(value).cast('B'))
            weights[name] = torch.tensor(value,device=target,dtype=torch.float32 if name in routers else torch.bfloat16)
        self.torch,self.device,self.max_bytes,self.weight_bytes = torch,target,max_bytes,charge
        self._weights,self._shapes,self._routers = weights,shapes,routers
        self.identity = digest.hexdigest()
        self._lock,self._closed = threading.RLock(),False

    def handles(self,name):
        with self._lock:
            return not self._closed and name in self._weights

    def shape(self,name):
        with self._lock:
            if self._closed:
                raise RuntimeError('K3 device projection owner closed')
            return self._shapes[name]

    def linear(self,name,value,*,cancel=None):
        with self._lock:
            if self._closed:
                raise RuntimeError('K3 device projection owner closed')
            weight = self._weights[name]
            torch = self.torch
            if value.ndim not in (1,2,3) or value.shape[-1]!=weight.shape[1] or value.device!=self.device or value.dtype!=torch.float32:
                raise ValueError('matching FP32 K3 device projection activations required')
            rows = value.numel()//weight.shape[1]
            if not rows or self.weight_bytes+rows*(weight.shape[1]*12+weight.shape[0]*12)>self.max_bytes:
                raise MemoryError('K3 resident projection activation/output exceeds admission')
            if cancel is not None and not callable(getattr(cancel,'is_set',None)):
                raise ValueError('projection cancellation must provide is_set')
            def check():
                if cancel is not None and cancel.is_set():
                    from ..execution_service import ExecutionCancelled
                    raise ExecutionCancelled('K3 resident device projection cancelled')
            check()
            with torch.inference_mode():
                operand = value if name in self._routers else value.to(torch.bfloat16)
                result = torch.nn.functional.linear(operand,weight).to(torch.float32)
                if not bool(torch.isfinite(result).all()):
                    raise ArithmeticError('nonfinite K3 device projection result')
                check()
                return result

    def close(self):
        with self._lock:
            self._weights.clear();self._shapes.clear()
            self._closed = True
