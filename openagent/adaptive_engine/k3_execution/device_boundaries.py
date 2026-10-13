"""Device-resident K3 arithmetic with explicit BF16 rounding boundaries."""
import math


class K3DeviceBoundaries:
    def __init__(self,torch,*,device,torch_version,max_bytes):
        if str(torch.__version__)!=torch_version or type(max_bytes) is not int or max_bytes<=0:
            raise ValueError('explicit Torch pin and device operation budget required')
        from .device_policy import admit_device
        self.torch,self.device,self.max_bytes = torch,admit_device(torch,device),max_bytes
        from .precision import check_precision
        check_precision(torch,self.device)

    def _admit(self,*values,extra=0):
        # Re-check per operation because callers may enter an autocast context
        # after this owner has been constructed.
        from .precision import check_precision
        check_precision(self.torch,self.device)
        if any(value.dtype!=self.torch.float32 or value.device!=self.device or not value.numel() for value in values):
            raise ValueError('nonempty matching FP32 device boundary tensors required')
        if sum(value.numel()*16 for value in values)+extra>self.max_bytes:
            raise MemoryError('K3 device boundary working set exceeds admission')
        if any(not bool(self.torch.isfinite(value).all()) for value in values):
            raise ValueError('finite K3 device boundary tensors required')

    def _bf(self,value):
        return value.to(self.torch.bfloat16)

    def _result(self,value):
        result = value.to(self.torch.float32)
        if not bool(self.torch.isfinite(result).all()):
            raise ArithmeticError('nonfinite K3 device boundary result')
        return result

    def cast(self,value):
        self._admit(value)
        with self.torch.inference_mode():
            return self._result(self._bf(value))

    def norm(self,value,weight,epsilon):
        self._admit(value,weight)
        if value.ndim<1 or weight.shape!=(value.shape[-1],):
            raise ValueError('matching K3 device norm width required')
        if isinstance(epsilon,bool) or not math.isfinite(epsilon) or epsilon<=0:
            raise ValueError('positive normalization epsilon required')
        with self.torch.inference_mode():
            rounded = self._bf(value).float()
            normalized = rounded*self.torch.rsqrt(rounded.square().mean(dim=-1,keepdim=True)+epsilon)
            return self._result(self._bf(normalized)*self._bf(weight))

    def residual(self,values,gain,query,epsilon):
        self._admit(values,gain,query)
        if values.ndim<2 or gain.shape!=(values.shape[-1],) or query.shape!=gain.shape:
            raise ValueError('matching K3 residual mixture geometry required')
        if isinstance(epsilon,bool) or not math.isfinite(epsilon) or epsilon<=0:
            raise ValueError('positive residual epsilon required')
        with self.torch.inference_mode():
            rows = self._bf(values).float()
            normalized = rows*self.torch.rsqrt(rows.square().mean(dim=-1,keepdim=True)+epsilon)
            scorer = self._bf(gain).float()*self._bf(query).float()
            scores = (normalized*scorer).sum(dim=-1)
            result = (scores.softmax(dim=-1)[...,None]*rows).sum(dim=-2)
            return self._result(self._bf(result))

    def situ(self,gate,up,beta,linear_beta):
        self._admit(gate,up)
        if gate.shape!=up.shape or any(isinstance(value,bool) or not math.isfinite(value) or value<=0 for value in (beta,linear_beta)):
            raise ValueError('matching Situ inputs and positive coefficients required')
        with self.torch.inference_mode():
            g,u = self._bf(gate).float(),self._bf(up).float()
            result = beta*self.torch.tanh(g/beta)*g.sigmoid()*linear_beta*self.torch.tanh(u/linear_beta)
            return self._result(self._bf(result))

    def add(self,left,right):
        self._admit(left,right)
        if left.shape!=right.shape:
            raise ValueError('matching K3 residual operands required')
        with self.torch.inference_mode():
            return self._result(self._bf(left)+self._bf(right))

    def multiply(self,left,right):
        self._admit(left,right)
        if left.shape!=right.shape:
            raise ValueError('matching K3 gated operands required')
        with self.torch.inference_mode():
            return self._result(self._bf(left)*self._bf(right))

    def gate(self,value):
        self._admit(value)
        with self.torch.inference_mode():
            return self._result(self._bf(value).sigmoid())

    def router_sigmoid(self,value):
        self._admit(value)
        with self.torch.inference_mode():
            return self._result(value.sigmoid())
