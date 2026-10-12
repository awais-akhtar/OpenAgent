"""Native GLM5-next vision blocks, spatial downsampling and publisher merger."""
import hashlib
import json
import math
import threading
from types import MappingProxyType
import numpy as np
from .native_media_ops import NativeMediaOps
from .publisher_glm5_config import VISION_DEFAULTS
from .qwen4_vision import _cancel

def vision_shapes(config):
    c=dict(VISION_DEFAULTS,**dict(config))
    for name in ('depth','hidden_size','num_heads','in_channels','patch_size','temporal_patch_size',
                 'spatial_merge_size','out_hidden_size','intermediate_size','projection_intermediate_size'):
        if type(c[name]) is not int or not 1<=c[name]<=1048576: raise ValueError('bounded GLM vision geometry required: '+name)
    h=c['hidden_size'];o=c['out_hidden_size'];m=c['intermediate_size'];z=c['projection_intermediate_size']
    if h%c['num_heads'] or (h//c['num_heads'])%4 or c['depth']>128 or c['spatial_merge_size']>16:
        raise ValueError('GLM vision head and merge geometry exceeds native bounds')
    if c['hidden_act']!='silu' or c['attention_dropout']!=0 or type(c['attention_bias']) is not bool:
        raise ValueError('GLM vision requires publisher SwiGLU and inference attention switches')
    if any(type(c[v]) not in (float,int) or not math.isfinite(c[v]) or c[v]<=0 for v in ('rms_norm_eps','swiglu_limit')):
        raise ValueError('finite positive GLM vision numerical boundaries required')
    shapes={'patch_embed.proj.weight':(h,c['in_channels'],c['temporal_patch_size'],c['patch_size'],c['patch_size']),
        'patch_embed.proj.bias':(h,), 'post_layernorm.weight':(h,),
        'downsample.weight':(o,h,c['spatial_merge_size'],c['spatial_merge_size']), 'downsample.bias':(o,),
        'merger.proj.weight':(o,o),'merger.post_projection_norm.weight':(o,),
        'merger.post_projection_norm.bias':(o,), 'merger.gate_proj.weight':(z,o),
        'merger.up_proj.weight':(z,o),'merger.down_proj.weight':(o,z)}
    for index in range(c['depth']):
        p=f'blocks.{index}.'
        for norm in ('norm1','norm2'): shapes[p+norm+'.weight']=(h,)
        for norm in ('q_norm','k_norm'): shapes[p+'attn.'+norm+'.weight']=(h//c['num_heads'],)
        for name,shape in (('attn.qkv',(3*h,h)),('attn.proj',(h,h)),
                           ('mlp.gate_proj',(m,h)),('mlp.up_proj',(m,h)),('mlp.down_proj',(h,m))):
            shapes[p+name+'.weight']=shape
            if c['attention_bias']: shapes[p+name+'.bias']=(shape[0],)
    return c,shapes

class Glm5VisionEncoder:
    implementation='native-c-glm5-next-variable-grid-vision-fp32'
    def __init__(self,config,tensors,*,library,max_bytes,max_patches=65536,prefix='model.visual.'):
        c,shapes=vision_shapes(config)
        if type(max_bytes) is not int or max_bytes<=0 or type(max_patches) is not int or not 1<=max_patches<=1048576:
            raise ValueError('explicit GLM vision memory and patch bounds required')
        selected={name[len(prefix):]:value for name,value in tensors.items() if name.startswith(prefix)}
        if set(selected)!=set(shapes): raise ValueError('complete original GLM vision tensor inventory required')
        amount=sum(math.prod(shape)*4 for shape in shapes.values())
        if 2*amount>max_bytes: raise MemoryError('GLM vision weight construction exceeds admission')
        self.config=MappingProxyType(c);self.weights={};self.max_bytes=max_bytes;self.max_patches=max_patches;self.weight_bytes=amount
        self._lock=threading.RLock();self._busy=False;self._closed=False;self.ops=None
        digest=hashlib.sha256(json.dumps(c,sort_keys=True,allow_nan=False).encode())
        for name,shape in shapes.items():
            value=np.asarray(selected[name])
            if value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all():
                raise ValueError('invalid GLM vision tensor: '+name)
            owned=np.frombuffer(value.tobytes(),np.float32).reshape(shape);self.weights[name]=owned
            digest.update(name.encode());digest.update(memoryview(owned).cast('B'))
        self.ops=NativeMediaOps(library);digest.update(self.ops.identity.encode())
        self.identity='openagent-glm5-vision-v1:'+digest.hexdigest()

    def _projection(self,x,name):
        weight=self.weights[name+'.weight']
        return self.ops.linear(x,weight.reshape(weight.shape[0],-1),self.weights.get(name+'.bias'))

    def _norm(self,x,name):
        return self.ops.norm(x,self.weights[name+'.weight'],rms=True,epsilon=self.config['rms_norm_eps'])

    def _mlp(self,x,name):
        gate=self._projection(x,name+'gate_proj');up=self._projection(x,name+'up_proj')
        return self._projection(self.ops.clamped_swiglu(gate,up,limit=self.config['swiglu_limit']),name+'down_proj')

    def forward(self,pixel_values,grid_thw,*,cancelled=None):
        with self._lock:
            if self._closed or self._busy: raise RuntimeError('GLM vision owner is closed or executing')
            c=self.config;pixels=np.asarray(pixel_values);grids=np.asarray(grid_thw);merge=c['spatial_merge_size']
            if (grids.ndim!=2 or grids.shape[1]!=3 or not grids.shape[0] or grids.dtype.kind not in 'iu' or
                    np.any(grids<=0) or np.any(grids>self.max_patches) or np.any(grids[:,1:]%merge)):
                raise ValueError('positive merge-aligned original GLM vision grids required')
            count=sum(math.prod(int(v) for v in row) for row in grids)
            patch_width=c['in_channels']*c['temporal_patch_size']*c['patch_size']**2
            if count>self.max_patches or pixels.shape!=(count,patch_width) or pixels.dtype!=np.float32:
                raise ValueError('GLM processor patches differ from the declared grid extent')
            workspace=4*count*(2*patch_width+24*c['hidden_size']+8*c['intermediate_size']+
                4*c['out_hidden_size']+8*c['projection_intermediate_size'])+128*count
            if self.weight_bytes+workspace>self.max_bytes: raise MemoryError('whole GLM vision transaction exceeds admission')
            if not np.isfinite(pixels).all(): raise ValueError('nonfinite original GLM processor pixels')
            self._busy=True
            try:
                _cancel(cancelled);positions=np.empty((count,2),np.int64);offsets=[0];cursor=0
                for temporal,height,width in grids:
                    for _ in range(int(temporal)):
                        for by in range(0,int(height),merge):
                            for bx in range(0,int(width),merge):
                                for dy in range(merge):
                                    for dx in range(merge): positions[cursor]=(by+dy,bx+dx);cursor+=1
                        offsets.append(cursor)
                x=self._projection(np.ascontiguousarray(pixels),'patch_embed.proj')
                heads=c['num_heads'];dim=c['hidden_size']//heads
                for index in range(c['depth']):
                    _cancel(cancelled);p=f'blocks.{index}.'
                    qkv=self._projection(self._norm(x,p+'norm1'),p+'attn.qkv').reshape(count,3,heads,dim)
                    qkv[:,0]=self._norm(np.ascontiguousarray(qkv[:,0]),p+'attn.q_norm')
                    qkv[:,1]=self._norm(np.ascontiguousarray(qkv[:,1]),p+'attn.k_norm')
                    qkv=qkv.reshape(count,3*c['hidden_size']);attention=np.empty_like(x)
                    for start,end in zip(offsets[:-1],offsets[1:]):
                        _cancel(cancelled)
                        attention[start:end]=self.ops.vision_attention(np.ascontiguousarray(qkv[start:end]),
                            np.ascontiguousarray(positions[start:end]),np.array([0,end-start],np.uint64),heads=heads)
                    x=x+self._projection(attention,p+'attn.proj')
                    x=x+self._mlp(self._norm(x,p+'norm2'),p+'mlp.')
                grouped=self._norm(x,'post_layernorm').reshape(-1,merge,merge,c['hidden_size'])
                channels=np.ascontiguousarray(grouped.transpose(0,3,1,2)).reshape(-1,c['hidden_size']*merge*merge)
                downsampled=self._projection(channels,'downsample')
                projected=self._projection(downsampled,'merger.proj')
                projected=self.ops.norm(projected,self.weights['merger.post_projection_norm.weight'],
                    self.weights['merger.post_projection_norm.bias'],epsilon=1e-5)
                projected=self._mlp(self.ops.activation(projected,kind='gelu'),'merger.')
                _cancel(cancelled);sizes=[math.prod(int(v) for v in row)//merge**2 for row in grids]
                return tuple(np.split(projected,np.cumsum(sizes)[:-1]))
            finally: self._busy=False

    def close(self):
        with self._lock:
            if self._busy: raise RuntimeError('GLM vision inference owns resources')
            if self._closed: return
            if self.ops is not None: self.ops.close();self.ops=None
            self.weights.clear();self._closed=True

__all__=['Glm5VisionEncoder','vision_shapes']
