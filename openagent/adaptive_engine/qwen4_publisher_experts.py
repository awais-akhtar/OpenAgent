"""Original Qwen4 split/FP8 experts through the native selected-expert ABI."""
from copy import deepcopy
from .inkling_publisher_experts import InklingPublisherExperts
from .qwen4_streamed_experts import qwen4_metadata_fingerprint

class Qwen4PublisherExperts(InklingPublisherExperts):
    @staticmethod
    def _configuration(config):
        qwen4_metadata_fingerprint(config)
        c=deepcopy(dict(config));c['n_routed_experts']=c['num_experts']
        c['mlp_layer_types']=['sparse']*c['num_hidden_layers']
        return c

    def __init__(self,snapshot,config,**options):
        self.family_config=deepcopy(dict(config))
        self.fingerprint=qwen4_metadata_fingerprint(config)
        super().__init__(snapshot,config,**options)
        self.hidden_size=config['hidden_size'];self.intermediate_size=config['moe_intermediate_size']
        self.expert_count=config['num_experts'];self.selected_experts=config['num_experts_per_tok']
        self.layer_ids=tuple(str(i) for i in range(config['num_hidden_layers']))
        self.identity=self.identity.replace('inkling-publisher','qwen4-publisher')

    def execute(self,layer,expert,inputs,*,cancelled=None):
        return self.evaluate(layer,expert,inputs,cancelled=cancelled)

__all__=['Qwen4PublisherExperts']
