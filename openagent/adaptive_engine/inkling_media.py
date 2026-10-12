"""Independent Inkling HMLP and dMel tower composition using native arithmetic.

Input pixels are processor patches [patch,T,H,W,C]; audio is already discretized
dMel. No tensor aliases between Inkling and transformer vision architectures.
"""
import hashlib
import json
import math
import threading
import numpy as np
from .native_media_ops import NativeMediaOps
from .qwen4_vision import _cancel

def _factors(number):
    values=[];divisor=2
    while divisor*divisor<=number:
        while number%divisor==0:
            values.append(divisor);number//=divisor
        divisor+=1
    if number>1: values.append(number)
    return values[::-1]

def _assignment(cost):
    """Rectangular minimum-cost row assignment, deterministic leftmost ties."""
    n,m=cost.shape
    if n>m: raise ValueError('assignment needs at least one column per row')
    u=np.zeros(n+1,np.float64);v=np.zeros(m+1,np.float64)
    assigned=np.zeros(m+1,np.int64);previous=np.zeros(m+1,np.int64)
    for row in range(1,n+1):
        assigned[0]=row;column=0;distances=np.full(m+1,np.inf);used=np.zeros(m+1,bool)
        while True:
            used[column]=True;current=int(assigned[column]);best=np.inf;following=0
            for candidate in range(1,m+1):
                if used[candidate]: continue
                residual=float(cost[current-1,candidate-1])-u[current]-v[candidate]
                if residual<distances[candidate]:
                    distances[candidate]=residual;previous[candidate]=column
                if distances[candidate]<best:
                    best=distances[candidate];following=candidate
            for candidate in range(m+1):
                if used[candidate]: u[assigned[candidate]]+=best;v[candidate]-=best
                else: distances[candidate]-=best
            column=following
            if assigned[column]==0: break
        while column:
            prior=int(previous[column]);assigned[column]=assigned[prior];column=prior
    result=np.empty(n,np.int64)
    for column in range(1,m+1):
        if assigned[column]: result[assigned[column]-1]=column-1
    return result

def hmlp_scales(*,patch_size,temporal_patch_size,layers,channels):
    if any(type(v) is not int or not 1<=v<=4096 for v in (patch_size,temporal_patch_size,layers,channels)):
        raise ValueError('positive bounded Inkling HMLP geometry required')
    if patch_size<2 or temporal_patch_size<2:
        raise ValueError('HMLP publisher scale planner requires nontrivial spatial and temporal patches')
    scales=[(1,1,1,channels)];spatial=1;temporal=1
    for factor in _factors(patch_size):
        spatial*=factor;scales.append((1,spatial,spatial,64*math.ceil(spatial*spatial*channels/64)))
    for factor in _factors(temporal_patch_size):
        temporal*=factor;scales.append((temporal,spatial,spatial,64*math.ceil(temporal*spatial*spatial*channels/64)))
    reduction=np.asarray([t*h*w for t,h,w,_ in scales],np.float32)
    ideal=np.linspace(0,np.log(np.float32(patch_size**2*temporal_patch_size*channels)),layers+1,dtype=np.float32)
    cost=np.abs(ideal[:,None]-np.log(reduction)[None,:])
    indices=cost.argmin(axis=1) if layers>=len(scales) else _assignment(cost)
    indices[0]=0;indices[-1]=len(scales)-1
    chosen=[scales[int(i)] for i in indices]
    for a,b in zip(chosen,chosen[1:]):
        if any(b[i]%a[i] or b[i]<a[i] for i in range(3)):
            raise ValueError('publisher HMLP assignment has nonintegral folding')
    return tuple(chosen)

def media_shapes(vision_config,audio_config,*,text_hidden):
    """Plan the complete publisher towers without weights or native libraries."""
    if type(vision_config) is not dict or type(audio_config) is not dict:
        raise ValueError('complete Inkling vision and audio configurations required')
    if type(text_hidden) is not int or not 1<=text_hidden<=65536:
        raise ValueError('bounded Inkling text width required')
    v=vision_config;a=audio_config
    if v.get('vision_encoder_type','hmlp')!='hmlp' or v.get('use_vision_norm',True) is not True or a.get('use_audio_norm',True) is not True:
        raise ValueError('Inkling media tower switches differ from publisher')
    scales=hmlp_scales(patch_size=v['patch_size'],temporal_patch_size=v['temporal_patch_size'],
        layers=v.get('n_layers',v.get('num_hidden_layers')),channels=v.get('n_channels',v.get('num_channels',3)))
    shapes={};layers=[]
    for index,(before,after) in enumerate(zip(scales,scales[1:])):
        temporal=after[0]//before[0];spatial=after[1]//before[1]
        output=text_hidden if index==len(scales)-2 else after[3]
        name=f'model.vision_tower.encoder_layers.{index}.'
        shapes[name+'projection.weight']=(output,before[3]*temporal*spatial**2)
        if index<len(scales)-2: shapes[name+'layer_norm.weight']=(output,)
        layers.append((name,temporal,spatial,index<len(scales)-2))
    shapes['model.vision_tower.final_norm.weight']=(text_hidden,)
    bins=a['n_mel_bins'];vocabulary=a['mel_vocab_size']
    if type(bins) is not int or not 1<=bins<=512 or type(vocabulary) is not int or not 1<=vocabulary<=65536:
        raise ValueError('bounded Inkling dMel codebooks required')
    shapes['model.audio_tower.embed_audio_tokens.embed_audio_tokens.weight']=(bins*vocabulary,text_hidden)
    shapes['model.audio_tower.norm.weight']=(text_hidden,)
    return shapes,tuple(layers),scales

class InklingMediaEncoder:
    implementation='native-c-inkling-hmlp-dmel-fp32'
    def __init__(self,vision_config,audio_config,tensors,*,library,text_hidden,max_bytes,max_items=65536):
        if type(max_bytes) is not int or max_bytes<=0 or type(max_items) is not int or not 1<=max_items<=1048576:
            raise ValueError('finite explicit Inkling media admission required')
        self.vision_config=dict(vision_config);self.audio_config=dict(audio_config)
        self.hidden=text_hidden;self.max_bytes=max_bytes;self.max_items=max_items
        self._lock=threading.RLock();self._closed=False;self._busy=False
        v=self.vision_config;a=self.audio_config
        shapes,self.layers,self.scales=media_shapes(v,a,text_hidden=text_hidden)
        if set(tensors)!=set(shapes):
            raise ValueError('exact complete Inkling media weight inventory required')
        amount=sum(math.prod(shape)*4 for shape in shapes.values())
        if 2*amount>max_bytes: raise MemoryError('Inkling media weight staging exceeds admission')
        self.weight_bytes=amount;self.weights={}
        digest=hashlib.sha256(json.dumps(dict(vision=v,audio=a,text_hidden=text_hidden),sort_keys=True,allow_nan=False).encode())
        for name,shape in shapes.items():
            source=np.asarray(tensors[name])
            if source.shape!=shape or source.dtype.kind!='f' or not np.isfinite(source).all():
                raise ValueError('invalid Inkling media weight: '+name)
            array=np.frombuffer(np.asarray(source,dtype=np.float32).tobytes(order='C'),np.float32).reshape(shape)
            if not np.isfinite(array).all(): raise ValueError('Inkling media FP32 overflow')
            self.weights[name]=array;digest.update(name.encode());digest.update(memoryview(array).cast('B'))
        self.ops=NativeMediaOps(library);digest.update(self.ops.identity.encode())
        self.identity='openagent-inkling-media-v1:'+digest.hexdigest()

    def _check(self):
        if self._closed or self._busy: raise RuntimeError('Inkling media owner is closed or busy')

    def vision(self,pixel_values,*,cancelled=None):
        with self._lock:
            self._check();self._busy=True
            try:
                _cancel(cancelled);source=np.asarray(pixel_values);c=self.vision_config
                expected=(c['temporal_patch_size'],c['patch_size'],c['patch_size'],c.get('n_channels',c.get('num_channels',3)))
                if source.ndim!=5 or source.shape[1:]!=expected or not 0<source.shape[0]<=self.max_items or source.dtype!=np.float32:
                    raise ValueError('Inkling processor patch geometry differs')
                peak=max(math.prod(scale[:3])*scale[3] for scale in self.scales)
                scratch=source.nbytes*3+source.shape[0]*(8*peak+8*self.hidden)
                if self.weight_bytes+scratch>self.max_bytes: raise MemoryError('Inkling HMLP transaction exceeds admission')
                x=np.ascontiguousarray(source)
                for name,temporal,spatial,normalize in self.layers:
                    _cancel(cancelled);b,t,h,w,width=x.shape
                    if temporal>1 or spatial>1:
                        x=x.reshape(b,t//temporal,temporal,h//spatial,spatial,w//spatial,spatial,width)
                        x=np.ascontiguousarray(x.transpose(0,1,3,5,2,4,6,7)).reshape(b,t//temporal,h//spatial,w//spatial,-1)
                    x=self.ops.linear(x,self.weights[name+'projection.weight'])
                    if normalize:
                        x=self.ops.activation(self.ops.norm(x,self.weights[name+'layer_norm.weight'],rms=True),kind='gelu')
                x=self.ops.norm(x,self.weights['model.vision_tower.final_norm.weight'],rms=True)
                _cancel(cancelled);return x.reshape(source.shape[0],self.hidden)
            finally: self._busy=False

    def audio(self,audio_input_ids,*,mask=None,cancelled=None):
        with self._lock:
            self._check();self._busy=True
            try:
                _cancel(cancelled);ids=np.asarray(audio_input_ids);c=self.audio_config
                if ids.ndim not in (2,3) or ids.shape[-1]!=c['n_mel_bins'] or ids.dtype.kind not in 'iu':
                    raise ValueError('discrete Inkling audio frame ids required')
                if ids.size//c['n_mel_bins']>self.max_items: raise MemoryError('audio frame capacity exceeded')
                if mask is not None:
                    mask=np.asarray(mask)
                    if mask.shape!=ids.shape[:-1] or not np.isin(mask,[0,1]).all(): raise ValueError('audio frame mask differs')
                    ids=ids[mask.astype(bool)]
                else: ids=ids.reshape(-1,c['n_mel_bins'])
                if self.weight_bytes+ids.size*24+ids.shape[0]*self.hidden*16>self.max_bytes:
                    raise MemoryError('Inkling audio transaction exceeds admission')
                if not ids.shape[0]: return np.empty((0,self.hidden),np.float32)
                if np.any(ids<0) or np.any(ids>=c['mel_vocab_size']): raise ValueError('audio id exceeds dMel bin vocabulary')
                value=self.ops.audio_embed(np.ascontiguousarray(ids,dtype=np.int64),
                    self.weights['model.audio_tower.embed_audio_tokens.embed_audio_tokens.weight'],vocabulary=c['mel_vocab_size'])
                value=self.ops.norm(value,self.weights['model.audio_tower.norm.weight'],rms=True)
                _cancel(cancelled);return value
            finally: self._busy=False

    def close(self):
        with self._lock:
            if self._closed: return
            self._check();self.ops.close();self.weights={};self._closed=True

__all__=['InklingMediaEncoder','hmlp_scales']
