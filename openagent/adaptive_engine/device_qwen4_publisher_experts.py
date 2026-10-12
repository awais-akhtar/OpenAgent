"""Selected original Qwen4 experts with bounded CUDA/HIP/Metal row transfers."""
import hashlib
import threading
from .publisher_weight_mapping import expert_bindings
from .publisher_tensor_reader import TensorSlice
from .device_publisher_projections import DevicePublisherProjections
from .qwen4_streamed_experts import qwen4_metadata_fingerprint
from .device_execution_policy import admit_fp32

class DeviceQwen4PublisherExperts:
    @staticmethod
    def _configuration(config):
        return qwen4_metadata_fingerprint(config),config,tuple(range(config['num_hidden_layers'])),None
    def __init__(self,snapshot,config,*,torch,device,max_bytes,max_host_bytes,
                 max_cache_bytes=0,block_shape=None,prefix='model.language_model.'):
        self.fingerprint,geometry,self.layers,self.activation_limit=self._configuration(config)
        self.hidden_size=geometry['hidden_size'];self.intermediate_size=geometry['moe_intermediate_size']
        self.expert_count=geometry['num_experts'];self.selected_experts=geometry['num_experts_per_tok']
        if type(max_bytes) is not int or max_bytes<=0:
            raise ValueError('positive selected device expert admission required')
        self.max_bytes=max_bytes;self._lock=threading.RLock();self._closed=False;self._busy=False
        bindings={};h,m,e=self.hidden_size,self.intermediate_size,self.expert_count
        for layer in self.layers:
            gate,down=expert_bindings(snapshot,prefix=f'{prefix}layers.{layer}.mlp.experts.',
                expert_count=e,hidden_size=h,intermediate_size=m)
            for expert in range(e):
                if gate.operation=='identity':
                    source=gate.sources[0]
                    g=TensorSlice(source.source,(m,h),source.element_offset+expert*2*m*h)
                    u=TensorSlice(source.source,(m,h),source.element_offset+(expert*2+1)*m*h)
                else:
                    g,u=gate.sources[2*expert:2*expert+2]
                if down.operation=='identity':
                    source=down.sources[0];d=TensorSlice(source.source,(h,m),source.element_offset+expert*h*m)
                else:
                    d=down.sources[expert]
                for suffix,value in (('gate',g),('up',u),('down',d)):
                    bindings[f'{layer}/{expert}/{suffix}']=value
        self.bank=DevicePublisherProjections(snapshot,bindings,torch=torch,device=device,
            max_bytes=max_bytes//2,max_host_bytes=max_host_bytes,max_cache_bytes=max_cache_bytes,
            block_shape=block_shape)
        self.identity='openagent-device-qwen4-publisher-experts-v1:'+hashlib.sha256(
            (self.fingerprint+self.bank.identity).encode()).hexdigest()

    def execute(self,layer,expert,inputs,*,torch,device,cancel=None):
        admit_fp32(torch,device)
        with self._lock:
            if self._closed or self._busy:
                raise RuntimeError('selected device expert owner is closed or executing')
            if (type(layer) is not int or layer not in self.layers or type(expert) is not int
                    or not 0<=expert<self.expert_count or torch is not self.bank.torch
                    or torch.device(device)!=self.bank.device):
                raise ValueError('expert call differs from admitted model/device')
            if inputs.ndim!=2 or inputs.shape[1]!=self.hidden_size or not inputs.shape[0]:
                raise ValueError('nonempty selected expert row batch required')
            work=8*inputs.shape[0]*(3*self.hidden_size+6*self.intermediate_size)
            if work>self.max_bytes-self.bank.max_bytes:
                raise MemoryError('selected device expert activations exceed admission')
            self._busy=True
            try:
                prefix=f'{layer}/{expert}/'
                with torch.inference_mode():
                    gate=self.bank.linear(prefix+'gate',inputs,cancel=cancel)
                    up=self.bank.linear(prefix+'up',inputs,cancel=cancel)
                    if self.activation_limit is not None:
                        gate=gate.clamp_max(self.activation_limit)
                        up=up.clamp(-self.activation_limit,self.activation_limit)
                    activated=torch.nn.functional.silu(gate)*up
                    del gate,up
                    return self.bank.linear(prefix+'down',activated,cancel=cancel)
            finally:
                self._busy=False

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._busy:
                raise RuntimeError('selected expert execution still owns resources')
            self.bank.close();self._closed=True

__all__=['DeviceQwen4PublisherExperts']
