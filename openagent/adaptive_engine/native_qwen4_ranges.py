"""Selected Qwen4 native ranges with hyper-channel and token-side inputs."""
import math

from .independent_qwen4 import IndependentQwen4TextDecoder
from .native_qwen4 import NativeQwen4TextDecoder


def qwen4_tensor_inventory(config, *, max_bytes):
    """Derive exact full-family shapes without reading or allocating weights."""
    contract = IndependentQwen4TextDecoder.__new__(IndependentQwen4TextDecoder)
    IndependentQwen4TextDecoder.__init__(contract,config,{},max_bytes=max_bytes,
                                        _inventory_only=True)
    shapes = dict(contract.tensor_shapes)
    shapes.update((name,tuple(value.shape)) for name,value in contract.integer_metadata.items())
    return shapes,contract.integer_metadata


class NativeQwen4Range(NativeQwen4TextDecoder):
    """Own only selected layers; retain original global layer indices.

    Hidden inputs/outputs include all hyper-connection channels. Token IDs are
    required for PLE even though embeddings and final head belong to the edge.
    KV, recurrent, convolution, PLE, masks and rotary positions commit together.
    """
    implementation = 'c-qwen4-selected-hidden-range'

    def __init__(self, config, tensors, *, begin, end, library, max_bytes,
                 max_snapshots=4,expert_provider=None,ple_provider=None,native_composition=False,
                 native_expert_bank=None,native_ple_bank=None):
        super().__init__(config,tensors,library=library,max_bytes=max_bytes,
                         max_snapshots=max_snapshots,layer_range=(begin,end),expert_provider=expert_provider,
                         ple_provider=ple_provider,native_composition=native_composition,
                         native_expert_bank=native_expert_bank,native_ple_bank=native_ple_bank)

    @classmethod
    def from_directory(cls, directory, *, begin, end, library, max_weight_bytes,
                       max_bytes, max_snapshots=4, read_limits=None, cancel=None,expert_provider=None,ple_provider=None,
                       native_composition=False,native_expert_bank=None,native_ple_bank=None):
        from .native_checkpoint import read_native_text_checkpoint, _finish_decoder
        if type(max_bytes) is not int or max_bytes<=0:
            raise ValueError('explicit Qwen4 range memory budget required')
        def select(config,inventory):
            shapes,metadata = qwen4_tensor_inventory(config,max_bytes=max_bytes)
            layers = config['num_hidden_layers']
            if type(begin) is not int or type(end) is not int or not 0<=begin<end<=layers:
                raise ValueError('valid Qwen4 selected layer interval required')
            if config.get('tie_word_embeddings',False) and 'lm_head.weight' not in inventory:
                shapes.pop('lm_head.weight')
            expected=set(shapes)
            if ple_provider is not None:
                for layer,(sources,starts) in ple_provider.bindings.items():
                    table=f'model.layers.{layer}.ple.ple_embedding.ngram_embedding.weight'
                    expected.discard(table);expected.update(sources)
                if dict(ple_provider.config)!=config:
                    raise ValueError('PLE checkpoint provider configuration differs')
            if set(inventory)!=expected:
                raise ValueError('exact complete Qwen4 source inventory required')
            for name,shape in shapes.items():
                if ple_provider is not None and name.endswith('ple.ple_embedding.ngram_embedding.weight'):
                    continue
                record=inventory[name]
                allowed=('I64',) if name in metadata else ('F32','F16','BF16')
                if (tuple(record['shape'])!=tuple(shape) or
                        record.get('dtype') not in allowed):
                    raise ValueError('Qwen4 source shape or dtype mismatch: '+name)
            prefixes = tuple(f'model.layers.{index}.' for index in range(begin,end))
            names = tuple(sorted(name for name in shapes if name.startswith(prefixes)))
            if ple_provider is not None:
                names=tuple(name for name in names if not name.endswith('ple.ple_embedding.ngram_embedding.weight'))
            if expert_provider is not None:
                names = tuple(name for name in names if not name.endswith(
                    ('mlp.experts.gate_up_proj','mlp.experts.down_proj')))
            size = sum(math.prod(shapes[name])*(8 if name in metadata else 4) for name in names)
            if (3*size+getattr(expert_provider,'max_bytes',0)+getattr(ple_provider,'max_bytes',0)
                    +getattr(native_expert_bank,'max_bytes',0)+getattr(native_ple_bank,'max_bytes',0)>max_bytes):
                raise MemoryError('Qwen4 range checkpoint conversion exceeds admission')
            return names
        config,weights = read_native_text_checkpoint(directory,model_type='qwen4_exp_text',
            max_weight_bytes=max_weight_bytes,read_limits=read_limits,cancel=cancel,
            tensor_selection=select,source_backend='public-range')
        model = cls(config,weights,begin=begin,end=end,library=library,max_bytes=max_bytes,
                    max_snapshots=max_snapshots,expert_provider=expert_provider,ple_provider=ple_provider,
                    native_composition=native_composition,native_expert_bank=native_expert_bank,native_ple_bank=native_ple_bank)
        return _finish_decoder(model,cancel)

    def forward(self, hidden, token_ids, *, attention_mask=None, position_ids=None, cancelled=None):
        return self._forward(token_ids,attention_mask=attention_mask,position_ids=position_ids,
                             incoming_hidden=hidden, cancelled=cancelled)

    def save_cache(self, destination, *, max_bytes):
        from .cache_file_recovery import save_executor_cache
        return save_executor_cache(self,destination,max_bytes=max_bytes)

    def load_cache(self, source, *, max_bytes):
        from .cache_file_recovery import load_executor_cache
        return load_executor_cache(self,source,max_bytes=max_bytes)

    def status(self):
        with self._lock:
            result=super().status()
            result.update(begin=self.layer_begin,end=self.layer_end,
                width=self.config.hc_count*self.config.hidden_size,token_side_input=True,certified=False)
            return result


__all__ = ['NativeQwen4Range','qwen4_tensor_inventory']
