"""Original selected Inkling experts on CUDA, HIP and Metal devices."""
import hashlib
import json
from .device_qwen4_publisher_experts import DeviceQwen4PublisherExperts
from .independent_inkling import configuration

class DeviceInklingPublisherExperts(DeviceQwen4PublisherExperts):
    @staticmethod
    def _configuration(config):
        c,_=configuration(config)
        geometry=dict(hidden_size=c['hidden_size'],moe_intermediate_size=c['moe_intermediate_size'],
            num_experts=c['n_routed_experts'],num_experts_per_tok=c['num_experts_per_tok'])
        layers=tuple(index for index,kind in enumerate(c['mlp_layer_types']) if kind=='sparse')
        fingerprint=hashlib.sha256(json.dumps(dict(config),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        return fingerprint,geometry,layers,None
    def __init__(self,snapshot,config,**options):
        self.contract=configuration(config)[0]
        super().__init__(snapshot,config,**options)
        self.identity=self.identity.replace('qwen4-publisher-experts','inkling-publisher-experts')

__all__=['DeviceInklingPublisherExperts']
