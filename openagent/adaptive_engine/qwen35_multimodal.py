"""Original Qwen3.5 wrapper composition with native vision and hybrid text.

The shared patch encoder implements the matching affine, normalization, grid
interpolation, spatial rotary attention and merger contract. Text execution is
the separate Qwen3.5 gated-delta/MoE owner; no Qwen4 text equations are selected.
The tiny NativeQwen35Vision fixture remains a separate explicit API.

Contract: Transformers v5.16.1 models/qwen3_5_moe/modeling_qwen3_5_moe.py;
publisher revision is recorded in publisher_family_contracts.PUBLISHER_VARIANTS.
https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
https://huggingface.co/Qwen/Qwen3.5-35B-A3B/tree/59d61f3ce65a6d9863b86d2e96597125219dc754
"""
import hashlib
import threading
from uuid import uuid4
import numpy as np

from .publisher_family_contracts import component_contract,partition_weights
from .qwen4_multimodal import Qwen4MultimodalDecoder
from .qwen4_vision import Qwen4VisionEncoder


class Qwen35VisionEncoder(Qwen4VisionEncoder):
    implementation='native-c-qwen35-publisher-variable-grid-vision-fp32'

    def __init__(self,config,tensors,**options):
        if config.get('model_type') not in ('qwen3_5_moe','qwen3_5_moe_vision'):
            raise ValueError('exact Qwen3.5 publisher vision identity required')
        super().__init__(config,tensors,**options)
        self.identity='openagent-qwen35-publisher-vision-v1:'+hashlib.sha256(self.identity.encode()).hexdigest()


class Qwen35MultimodalDecoder(Qwen4MultimodalDecoder):
    """Reuse media placement/ownership transport, with separate family owners."""
    implementation='native-c-qwen35-publisher-conditional-fp32'
    cache_magic=b'OAQ5MM01'

    def _positions(self,ids,types,image_grids,video_grids):
        if image_grids is not None:
            grids=np.asarray(image_grids)
            if grids.ndim!=2 or grids.shape[1]!=3 or np.any(grids[:,0]!=1):
                raise ValueError('Qwen3.5 still images require one temporal patch grid per image')
        return super()._positions(ids,types,image_grids,video_grids)

    def __init__(self,config,tensors,*,library,text_library,max_bytes,max_snapshots=4,
            arithmetic='fp32',context_tokens=None,text_device=None,expert_provider=None,own_expert_provider=False,
            projection_provider=None,own_projection_provider=False):
        self.components=component_contract(config,arithmetic=arithmetic)
        if self.components.variant.wrapper!='qwen3_5_moe':raise ValueError('Qwen3.5 publisher wrapper required')
        if type(max_bytes) is not int or max_bytes<=0 or type(max_snapshots) is not int or not 1<=max_snapshots<=64:
            raise ValueError('explicit bounded Qwen3.5 component and snapshot admissions required')
        self.config=self.components.original_config;self.max_bytes=max_bytes
        self.max_snapshots=max_snapshots;self._lock=threading.RLock()
        self._busy=self._closed=False;self._owner=uuid4().hex;self._snapshots={}
        self.rope_delta=0;self.text=self.vision=None
        self._native_store_runtime=self._native_experts=self._native_ple=None
        self._ple_provider=self._expert_provider=self._projection_provider=None
        self._own_ple_provider=self._own_expert_provider=self._own_projection_provider=False
        self._expert_provider=expert_provider;self._own_expert_provider=own_expert_provider
        self._projection_provider=projection_provider;self._own_projection_provider=own_projection_provider
        if projection_provider is not None and text_device is None:
            raise ValueError('original device projection bank requires explicit Qwen3.5 device text arithmetic')
        self.auxiliary={}
        text,vision,audio,auxiliary=partition_weights(tensors,self.components)
        if audio:raise ValueError('Qwen3.5 publisher does not own an audio tower')
        c=dict(self.components.text_config)
        if context_tokens is not None:
            if type(context_tokens) is not int or not 1<=context_tokens<=c['max_position_embeddings']:
                raise ValueError('bounded Qwen3.5 logical context required')
        physical_context=c['max_position_embeddings'] if context_tokens is None else context_tokens
        if text_device is None and physical_context>65536:
            raise ValueError('native Qwen3.5 kernel requires an explicit context of at most 65536 tokens')
        if self.components.vision_config['out_hidden_size']!=c['hidden_size']:
            raise ValueError('Qwen3.5 vision projection and language width differ')
        for name in ('image_token_id','video_token_id'):
            if type(self.config.get(name)) is not int or not 0<=self.config[name]<c['vocab_size']:
                raise ValueError('explicit Qwen3.5 modality token IDs required')
        if self.config['image_token_id']==self.config['video_token_id']:
            raise ValueError('distinct image and video token IDs required')
        amount=sum(np.asarray(value).nbytes for value in auxiliary.values())
        if amount*2>=max_bytes:raise MemoryError('Qwen3.5 auxiliary staging exceeds admission')
        for name,value in auxiliary.items():
            value=np.asarray(value)
            if value.dtype.kind!='f' or not np.isfinite(value).all():raise ValueError('finite owned auxiliary weights required')
            self.auxiliary[name]=np.frombuffer(value.tobytes(),dtype=value.dtype).reshape(value.shape)
        available=max_bytes-amount
        try:
            self.vision=Qwen35VisionEncoder(self.components.vision_config,vision,library=library,max_bytes=available//2)
            if text_device is None:
                from .native_qwen35 import NativeQwen35TextDecoder
                self.text=NativeQwen35TextDecoder(c,text,library=text_library,max_bytes=available-available//2,
                    max_snapshots=max_snapshots,expert_provider=expert_provider,context_tokens=physical_context)
            else:
                from .torch_qwen35_decoder import TorchQwen35Decoder
                self.text=TorchQwen35Decoder(c,text,device=text_device,max_bytes=available-available//2,
                    max_snapshots=max_snapshots,expert_provider=expert_provider,projection_provider=projection_provider,
                    context_tokens=physical_context)
                self.implementation='torch-qwen35-publisher-conditional-fp32';self.native_range_execution=False
            if self.text.mrope_section is None:raise ValueError('explicit publisher MRoPE sections required')
            digest=hashlib.sha256((self.components.identity+self.text.identity+self.vision.identity).encode())
            for name,value in sorted(self.auxiliary.items()):
                digest.update(name.encode());digest.update(memoryview(value).cast('B'))
            self.identity='openagent-qwen35-publisher-media-v1:'+digest.hexdigest()
            self.text.identity+=':wrapper:'+self.identity
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:
                primary.cleanup=self.close;primary.add_note('Qwen3.5 component cleanup retained: '+repr(cleanup))
            raise


__all__=['Qwen35VisionEncoder','Qwen35MultimodalDecoder']
