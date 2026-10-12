"""Selected quantized DSA experts on the decoder's requested accelerator."""
from .torch_mixtral_experts import TorchMixtralExperts
from .independent_glm_dsa import IndependentGlmDsaDecoder
from .glm_dsa_streamed_experts import config_identity


class TorchGlmDsaExperts(TorchMixtralExperts):
    architecture='glm_moe_dsa'

    def __init__(self,manifest,store,config,**options):
        self.contract=IndependentGlmDsaDecoder(config,{},_inventory_only=True).config
        self._sparse_layers=tuple(i for i,kind in enumerate(self.contract.mlp_layer_types) if kind=='sparse')
        self._slots={layer:slot for slot,layer in enumerate(self._sparse_layers)}
        if manifest.fingerprint!=config_identity(config):
            raise ValueError('DSA packed manifest differs from exact checkpoint configuration')
        super().__init__(manifest,store,**options)
        c=self.contract
        if (self.hidden_size,self.intermediate_size,self.expert_count,self.selected_experts)!=(
                c.hidden_size,c.moe_intermediate_size,c.n_routed_experts,c.num_experts_per_tok):
            raise ValueError('DSA packed expert geometry differs from exact decoder')

    def _admit_layer_ids(self,ids):
        return ids==tuple(str(i) for i in self._sparse_layers)

    def _layer_slot(self,layer):
        if type(layer) is not int or layer not in self._slots:
            raise ValueError('exact original sparse DSA layer required')
        return self._slots[layer]
