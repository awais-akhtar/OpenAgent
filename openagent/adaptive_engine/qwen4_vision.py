"""Qwen4 publisher vision composition with variable grids and native arithmetic.

Checkpoint namespace: model.visual. Patch input is the processor's flattened
channel/temporal/spatial block order. Frames attend independently; spatial
merge groups stay contiguous throughout the transformer and final merger.
"""
from types import MappingProxyType
import hashlib
import json
import math
import threading
import numpy as np
from .native_media_ops import NativeMediaOps

def _cancel(callback):
    if callback is not None:
        if not callable(callback):
            raise TypeError('callable media cancellation required')
        if callback():
            raise InterruptedError('media execution cancelled before continuation commit')

def vision_shapes(config):
    c=dict(config)
    for name in ('hidden_size','depth','num_heads','intermediate_size','out_hidden_size',
                 'patch_size','temporal_patch_size','spatial_merge_size','in_channels','num_position_embeddings'):
        if type(c.get(name)) is not int or not 1<=c[name]<=1048576:
            raise ValueError('explicit bounded Qwen4 vision geometry required: '+name)
    h=c['hidden_size'];merge=c['spatial_merge_size'];merged=h*merge*merge
    side=math.isqrt(c['num_position_embeddings'])
    if (h%c['num_heads'] or (h//c['num_heads'])%4 or side*side!=c['num_position_embeddings'] or
            c.get('deepstack_visual_indexes',[]) or c.get('hidden_act') not in ('gelu','gelu_pytorch_tanh')):
        raise ValueError('unsupported Qwen4 vision head, learned grid or activation contract')
    shapes={'patch_embed.proj.weight':(h,c['in_channels'],c['temporal_patch_size'],c['patch_size'],c['patch_size']),
            'patch_embed.proj.bias':(h,), 'pos_embed.weight':(c['num_position_embeddings'],h),
            'merger.norm.weight':(h,), 'merger.norm.bias':(h,),
            'merger.linear_fc1.weight':(merged,merged),'merger.linear_fc1.bias':(merged,),
            'merger.linear_fc2.weight':(c['out_hidden_size'],merged),'merger.linear_fc2.bias':(c['out_hidden_size'],)}
    for layer in range(c['depth']):
        p=f'blocks.{layer}.'
        for norm in ('norm1','norm2'):
            shapes[p+norm+'.weight']=(h,);shapes[p+norm+'.bias']=(h,)
        shapes.update({p+'attn.qkv.weight':(3*h,h),p+'attn.qkv.bias':(3*h,),
            p+'attn.proj.weight':(h,h),p+'attn.proj.bias':(h,),
            p+'mlp.linear_fc1.weight':(c['intermediate_size'],h),p+'mlp.linear_fc1.bias':(c['intermediate_size'],),
            p+'mlp.linear_fc2.weight':(h,c['intermediate_size']),p+'mlp.linear_fc2.bias':(h,)})
    return shapes

class Qwen4VisionEncoder:
    implementation='native-c-qwen4-variable-grid-vision-fp32'
    def __init__(self,config,tensors,*,library,max_bytes,max_patches=65536,prefix='model.visual.'):
        if type(max_bytes) is not int or max_bytes<=0 or type(max_patches) is not int or not 1<=max_patches<=1048576:
            raise ValueError('explicit finite media budgets required')
        self.config=MappingProxyType(dict(config));shapes=vision_shapes(config)
        selected={name[len(prefix):]:value for name,value in tensors.items() if name.startswith(prefix)}
        if set(selected)!=set(shapes):
            raise ValueError('complete exact Qwen4 vision checkpoint tensor inventory required')
        amount=sum(math.prod(shape)*4 for shape in shapes.values())
        if 2*amount>max_bytes:
            raise MemoryError('vision construction exceeds resident weight admission')
        self.weights={}
        self._lock=threading.RLock();self._executing=False;self._closed=False
        self.max_bytes,self.max_patches,self.weight_bytes=max_bytes,max_patches,amount
        digest=hashlib.sha256(json.dumps(dict(config),sort_keys=True,allow_nan=False).encode())
        for name,shape in shapes.items():
            source=np.asarray(selected[name])
            if source.shape!=shape or source.dtype.kind!='f' or not np.isfinite(source).all():
                raise ValueError('invalid Qwen4 media tensor: '+name)
            array=np.array(source,dtype=np.float32,order='C',copy=True)
            if not np.isfinite(array).all():
                raise ValueError('media tensor exceeds FP32 range: '+name)
            array.flags.writeable=False;self.weights[name]=array
            digest.update(name.encode());digest.update(memoryview(array).cast('B'))
        self.identity='openagent-qwen4-vision-v1:'+digest.hexdigest()
        self.ops=NativeMediaOps(library)
        self.identity+=':arithmetic:'+self.ops.identity

    def _check(self):
        if self._closed or self._executing:
            raise RuntimeError('vision owner is closed or executing')

    def _projection(self,x,name):
        weight=self.weights[name+'.weight']
        return self.ops.linear(x,weight.reshape(weight.shape[0],-1),self.weights.get(name+'.bias'))

    def _norm(self,x,name):
        return self.ops.norm(x,self.weights[name+'.weight'],self.weights[name+'.bias'])

    def _layout(self,grids):
        c=self.config;merge=c['spatial_merge_size'];side=math.isqrt(c['num_position_embeddings'])
        positions=[];learned=[];offsets=[0]
        table=self.weights['pos_embed.weight'].reshape(side,side,-1)
        for temporal,height,width in grids:
            ys=np.linspace(0,side-1,int(height),dtype=np.float32)
            xs=np.linspace(0,side-1,int(width),dtype=np.float32)
            for _ in range(int(temporal)):
                for by in range(0,int(height),merge):
                    for bx in range(0,int(width),merge):
                        for dy in range(merge):
                            for dx in range(merge):
                                y,x=by+dy,bx+dx;fy,fx=ys[y],xs[x]
                                y0,x0=int(fy),int(fx);y1,x1=min(y0+1,side-1),min(x0+1,side-1)
                                wy,wx=np.float32(fy-y0),np.float32(fx-x0)
                                learned.append((1-wy)*((1-wx)*table[y0,x0]+wx*table[y0,x1])+
                                               wy*((1-wx)*table[y1,x0]+wx*table[y1,x1]))
                                positions.append((y,x))
                offsets.append(len(positions))
        return (np.asarray(positions,np.int64),np.asarray(learned,np.float32),np.asarray(offsets,np.uint64))

    def forward(self,pixel_values,grid_thw,*,cancelled=None):
        with self._lock:
            self._check();self._executing=True
            try:
                _cancel(cancelled)
                c=self.config;pixels=np.asarray(pixel_values);grids=np.asarray(grid_thw)
                if (grids.ndim!=2 or grids.shape[1]!=3 or not grids.shape[0] or grids.dtype.kind not in 'iu' or
                        np.any(grids<=0) or np.any(grids>self.max_patches) or
                        np.any(grids[:,1:]%c['spatial_merge_size'])):
                    raise ValueError('positive merge-aligned Qwen4 grids required')
                count=sum(math.prod(int(v) for v in row) for row in grids)
                patch_width=c['in_channels']*c['temporal_patch_size']*c['patch_size']**2
                if count>self.max_patches or pixels.shape!=(count,patch_width) or pixels.dtype!=np.float32:
                    raise ValueError('bounded processor patches and grid extent must agree')
                # Whole activation peak, positions, interpolation and bounded native row scores.
                scratch=4*count*(patch_width+16*c['hidden_size']+4*c['intermediate_size']+
                    4*c['hidden_size']*c['spatial_merge_size']**2+2*c['out_hidden_size'])+64*count
                if self.weight_bytes+scratch>self.max_bytes:
                    raise MemoryError('complete Qwen4 vision transaction exceeds admission')
                if not np.isfinite(pixels).all():
                    raise ValueError('nonfinite processor pixels')
                positions,learned,offsets=self._layout(grids)
                x=self._projection(np.ascontiguousarray(pixels),'patch_embed.proj')+learned
                for layer in range(c['depth']):
                    _cancel(cancelled);p=f'blocks.{layer}.'
                    qkv=self._projection(self._norm(x,p+'norm1'),p+'attn.qkv')
                    attention=np.empty_like(x)
                    for start,end in zip(offsets[:-1],offsets[1:]):
                        _cancel(cancelled);start,end=int(start),int(end)
                        attention[start:end]=self.ops.vision_attention(np.ascontiguousarray(qkv[start:end]),
                            np.ascontiguousarray(positions[start:end]),np.array([0,end-start],np.uint64),heads=c['num_heads'])
                    x=x+self._projection(attention,p+'attn.proj')
                    value=self._projection(self._norm(x,p+'norm2'),p+'mlp.linear_fc1')
                    x=x+self._projection(self.ops.activation(value,kind=c['hidden_act']),p+'mlp.linear_fc2')
                value=self._norm(x,'merger.norm').reshape(-1,c['hidden_size']*c['spatial_merge_size']**2)
                value=self.ops.activation(self._projection(value,'merger.linear_fc1'),kind='gelu')
                projected=self._projection(value,'merger.linear_fc2');_cancel(cancelled)
                sizes=[math.prod(int(v) for v in row)//c['spatial_merge_size']**2 for row in grids]
                return tuple(np.split(projected,np.cumsum(sizes)[:-1]))
            finally:
                self._executing=False

    def close(self):
        with self._lock:
            if self._closed: return
            self._check();self.ops.close();self.weights={};self._closed=True

__all__=['Qwen4VisionEncoder','vision_shapes']
