"""Native Qwen4 hyper-channel embeddings and final mixer/head ownership."""
import math
import numpy as np

from .native_qwen4 import NativeQwen4TextDecoder
from .native_qwen4_ranges import qwen4_tensor_inventory


class NativeQwen4Edge(NativeQwen4TextDecoder):
    implementation = 'c-qwen4-hyper-channel-edge'

    def __init__(self, config, tensors, *, library, max_bytes):
        super().__init__(config,tensors,library=library,max_bytes=max_bytes,
                         max_snapshots=1,_edge_only=True)

    @classmethod
    def from_directory(cls, directory, *, library, max_weight_bytes, max_bytes,
                       read_limits=None, cancel=None,ple_provider=None):
        from .native_checkpoint import read_native_text_checkpoint, _finish_decoder
        if type(max_bytes) is not int or max_bytes<=0:
            raise ValueError('explicit Qwen4 edge memory budget required')
        def select(config,inventory):
            shapes,metadata = qwen4_tensor_inventory(config,max_bytes=max_bytes)
            if config.get('tie_word_embeddings',False) and 'lm_head.weight' not in inventory:
                shapes.pop('lm_head.weight')
            expected=set(shapes)
            if ple_provider is not None:
                if dict(ple_provider.config)!=config: raise ValueError('PLE edge checkpoint configuration differs')
                ple_provider.snapshot.assert_unchanged()
                for layer,(sources,starts) in ple_provider.bindings.items():
                    expected.discard(f'model.layers.{layer}.ple.ple_embedding.ngram_embedding.weight')
                    expected.update(sources)
                    for name in sources:
                        record=ple_provider.snapshot.tensors[name]
                        if name not in inventory or tuple(inventory[name]['shape'])!=record.shape or inventory[name]['dtype']!=record.dtype:
                            raise ValueError('PLE edge source inventory differs')
            if expected!=set(inventory):
                raise ValueError('exact complete Qwen4 source inventory required')
            for name,shape in shapes.items():
                if ple_provider is not None and name.endswith('ple.ple_embedding.ngram_embedding.weight'): continue
                record=inventory[name]
                allowed=('I64',) if name in metadata else ('F32','F16','BF16')
                if (tuple(record['shape'])!=tuple(shape) or
                        record.get('dtype') not in allowed):
                    raise ValueError('Qwen4 source shape or dtype mismatch: '+name)
            names = tuple(sorted(name for name in shapes if not name.startswith('model.layers.')))
            size = 4*sum(math.prod(shapes[name]) for name in names)
            if config.get('tie_word_embeddings',False) and 'lm_head.weight' not in shapes:
                size += 4*math.prod(shapes['model.embed_tokens.weight'])
            if 3*size>max_bytes:
                raise MemoryError('Qwen4 edge checkpoint conversion exceeds admission')
            return names
        config,weights = read_native_text_checkpoint(directory,model_type='qwen4_exp_text',
            max_weight_bytes=max_weight_bytes,read_limits=read_limits,cancel=cancel,
            tensor_selection=select,source_backend='public-range')
        return _finish_decoder(cls(config,weights,library=library,max_bytes=max_bytes),cancel)

    @property
    def resident_weight_bytes(self):
        return self.weight_bytes

    def embeddings(self, token_ids):
        with self._lock:
            self._check_open()
            ids = np.asarray(token_ids)
            if ids.ndim==1:
                ids = ids[None]
            c = self.config
            if (ids.ndim!=2 or not all(ids.shape) or ids.dtype.kind not in 'iu' or
                    ids.size>65536 or np.any(ids<0) or np.any(ids>=c.vocab_size)):
                raise ValueError('bounded Qwen4 edge token IDs required')
            needed = ids.size*(16+4*c.hidden_size*(1+c.hc_count))
            if self.weight_bytes+needed>self.max_bytes:
                raise MemoryError('Qwen4 edge embedding staging exceeds admission')
            return self._embed(ids)

    def final_head(self, hidden):
        with self._lock:
            self._check_open()
            values = np.asarray(hidden)
            c = self.config
            width = c.hc_count*c.hidden_size
            if (values.ndim not in (2,3) or values.dtype!=np.float32 or
                    values.shape[-1]!=width or not np.isfinite(values).all()):
                raise ValueError('finite FP32 Qwen4 hyper-channel hidden rows required')
            rows = values.size//width
            if not 0<rows<=65536:
                raise ValueError('Qwen4 edge row count exceeds admission')
            workspace = rows*(16*width+8*c.hc_lowrank+12*c.hidden_size+4*c.vocab_size)
            workspace += 8*(width+c.hc_lowrank)
            if self.weight_bytes+workspace>self.max_bytes:
                raise MemoryError('Qwen4 edge mixer/head staging exceeds admission')
            incoming = np.ascontiguousarray(values.reshape(1,rows,width))
            mixed,_ = self._mix(incoming,'model.hyper_connection_mixer.',False)
            output = self._head(mixed).reshape((*values.shape[:-1],c.vocab_size))
            if not np.isfinite(output).all():
                raise ArithmeticError('nonfinite native Qwen4 edge logits')
            return output

    def forward(self, *args, **kwargs):
        raise TypeError('Qwen4 edge exposes embeddings/final_head; use ranges for layer execution')


__all__ = ['NativeQwen4Edge']
