"""Original selected Qwen3.5 expert rows with explicit family fingerprints."""
from copy import deepcopy
import hashlib
from .inkling_publisher_experts import InklingPublisherExperts
from .device_qwen4_publisher_experts import DeviceQwen4PublisherExperts
from .qwen35_streamed_experts import qwen35_metadata_fingerprint


class Qwen35PublisherExperts(InklingPublisherExperts):
    @staticmethod
    def _configuration(config):
        qwen35_metadata_fingerprint(config)
        value=deepcopy(dict(config));value['n_routed_experts']=value['num_experts']
        value['mlp_layer_types']=['sparse']*value['num_hidden_layers']
        return value

    def __init__(self,snapshot,config,**options):
        self.fingerprint=qwen35_metadata_fingerprint(config)
        super().__init__(snapshot,config,**options)
        self.hidden_size=config['hidden_size'];self.intermediate_size=config['moe_intermediate_size']
        self.expert_count=config['num_experts'];self.selected_experts=config['num_experts_per_tok']
        self.layer_ids=tuple(str(index) for index in range(config['num_hidden_layers']))
        self.identity='openagent-qwen35-publisher-experts-v1:'+hashlib.sha256(
            (self.fingerprint+self.identity).encode()).hexdigest()

    def execute(self,layer,expert,inputs,*,cancelled=None):
        return self.evaluate(layer,expert,inputs,cancelled=cancelled)


class DeviceQwen35PublisherExperts(DeviceQwen4PublisherExperts):
    @staticmethod
    def _configuration(config):
        return qwen35_metadata_fingerprint(config),config,tuple(range(config['num_hidden_layers'])),None

    def __init__(self,snapshot,config,**options):
        super().__init__(snapshot,config,**options)
        self.identity='openagent-device-qwen35-publisher-experts-v1:'+hashlib.sha256(self.identity.encode()).hexdigest()


__all__=['Qwen35PublisherExperts','DeviceQwen35PublisherExperts']
