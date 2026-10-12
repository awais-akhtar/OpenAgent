"""Selected original DSA experts with bounded CUDA/HIP/Metal row streaming."""
import hashlib
import json

from .device_qwen4_publisher_experts import DeviceQwen4PublisherExperts
from .independent_glm_dsa import IndependentGlmDsaDecoder


class DeviceGlmDsaExperts(DeviceQwen4PublisherExperts):
    @staticmethod
    def _configuration(config):
        c=IndependentGlmDsaDecoder(config,{},_inventory_only=True).config
        geometry=dict(hidden_size=c.hidden_size,moe_intermediate_size=c.moe_intermediate_size,
            num_experts=c.n_routed_experts,num_experts_per_tok=c.num_experts_per_tok)
        layers=tuple(index for index,kind in enumerate(c.mlp_layer_types) if kind=='sparse')
        fingerprint=hashlib.sha256(json.dumps(vars(c),sort_keys=True,
            separators=(',',':'),allow_nan=False).encode()).hexdigest()
        return fingerprint,geometry,layers,None

    def __init__(self,snapshot,config,*,prefix='model.',**options):
        self.contract=IndependentGlmDsaDecoder(config,{},_inventory_only=True).config
        if not any(kind=='sparse' for kind in self.contract.mlp_layer_types):
            raise ValueError('selected DSA expert owner requires configured routed layers')
        super().__init__(snapshot,config,prefix=prefix,**options)
        self.identity=self.identity.replace('qwen4-publisher-experts','glm-dsa-publisher-experts')


__all__=['DeviceGlmDsaExperts']
