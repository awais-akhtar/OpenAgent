"""Bounded device operations for MoonViT3d and patchmergerv2.

The arithmetic owner supplies the pinned Torch/device policy. Every operation
releases its device temporaries before returning BF16-rounded FP32 host rows.
This policy is an execution implementation, not multimodal exact evidence.
"""
import math
import numpy as np


class K3BF16VisionOps:
    policy = 'moonvit3d-bf16-rms-fp32-complex-rope-eager-tiled-attention-v1'

    def __init__(self, arithmetic, *, check_cancelled):
        from .bf16_arithmetic import K3BF16Arithmetic
        if type(arithmetic) is not K3BF16Arithmetic or not callable(check_cancelled):
            raise ValueError('pinned K3 arithmetic and request cancellation required')
        self.arithmetic=arithmetic;self.check=check_cancelled
        self.torch=arithmetic.torch

    def _admit(self,*values,extra=0):
        self.check()
        return self.arithmetic._admit(*values,extra=extra)

    def normalize(self,x,weight,*,epsilon):
        x,weight=self._admit(x,weight,extra=np.asarray(x).size*16)
        if (x.ndim!=2 or weight.shape!=(x.shape[-1],) or not x.shape[0]
                or type(epsilon) not in (int,float) or not math.isfinite(epsilon) or epsilon<=0):
            raise ValueError('finite batched vision RMS geometry and epsilon required')
        with self.torch.inference_mode():
            value=self.arithmetic._tensor(x)
            fp=value.float()
            normalized=(fp*self.torch.rsqrt(fp.square().mean(-1,keepdim=True)+epsilon)).to(value.dtype)
            result=self.arithmetic._host(normalized*self.arithmetic._tensor(weight))
        self.check();return result

    def gelu(self,x,approximate):
        x,=self._admit(x,extra=np.asarray(x).size*8)
        if type(approximate) is not bool:raise ValueError('explicit vision GELU policy required')
        with self.torch.inference_mode():
            result=self.arithmetic._host(self.torch.nn.functional.gelu(
                self.arithmetic._tensor(x),approximate='tanh' if approximate else 'none'))
        self.check();return result

    def rotate(self,x,phase):
        x,phase=self._admit(x,phase,extra=np.asarray(x).size*32)
        if x.ndim!=3 or x.shape[-1]%2 or phase.shape!=(x.shape[0],x.shape[-1]//2):
            raise ValueError('matching vision complex-pair rotary rows required')
        with self.torch.inference_mode():
            value=self.arithmetic._tensor(x)
            pairs=value.float().reshape(*value.shape[:-1],value.shape[-1]//2,2)
            angle=self.torch.tensor(phase,device=self.arithmetic.device,dtype=self.torch.float32)[:,None]
            cosine,sine=angle.cos(),angle.sin()
            output=self.torch.stack((pairs[...,0]*cosine-pairs[...,1]*sine,
                                    pairs[...,0]*sine+pairs[...,1]*cosine),dim=-1)
            result=self.arithmetic._host(output.reshape(value.shape).to(value.dtype))
        self.check();return result

    def spatial_position(self,weight,height,width):
        weight,=self._admit(weight,extra=height*width*np.asarray(weight).shape[-1]*24)
        if weight.ndim!=3 or any(type(v) is not int or not 0<v<=512 for v in (height,width)):
            raise ValueError('bounded spatial position geometry required')
        with self.torch.inference_mode():
            value=self.arithmetic._tensor(weight)
            if tuple(value.shape[:2])!=(height,width):
                value=self.torch.nn.functional.interpolate(value.permute(2,0,1)[None],
                    size=(height,width),mode='bilinear',align_corners=False)[0].permute(1,2,0)
            result=self.arithmetic._host(value)
        self.check();return result

    def mean(self,x,axis):
        x,=self._admit(x,extra=np.asarray(x).size*8)
        if type(axis) is not int or not -x.ndim<=axis<x.ndim or not x.shape[axis]:
            raise ValueError('nonempty vision pooling axis required')
        with self.torch.inference_mode():
            result=self.arithmetic._host(self.arithmetic._tensor(x).mean(dim=axis))
        self.check();return result

    def attention(self,query,keys,values):
        """Bidirectional media-local attention without a full score matrix."""
        query,keys,values=self._admit(query,keys,values)
        if (query.ndim!=3 or keys.shape!=query.shape or values.shape!=query.shape
                or not all(query.shape)):
            raise ValueError('equal complete media-local vision Q/K/V required')
        length,heads,dim=query.shape
        fixed=sum(v.size*12 for v in (query,keys,values))+query.size*8+4096
        # BF16 scores/probabilities and FP32 softmax temporaries are included.
        per_row=heads*(length*24+dim*24)
        rows=min(length,256,(self.arithmetic.max_bytes-fixed)//per_row)
        if rows<=0:raise MemoryError('one vision attention query cannot fit device admission')
        result=np.empty(query.shape,np.float32)
        with self.torch.inference_mode():
            key=self.arithmetic._tensor(keys).permute(1,2,0)
            value=self.arithmetic._tensor(values).permute(1,0,2)
            for begin in range(0,length,rows):
                self.check();end=min(length,begin+rows)
                q=self.arithmetic._tensor(query[begin:end]).permute(1,0,2)
                score=self.torch.matmul(q,key)*dim**-.5
                probability=score.float().softmax(-1).to(self.torch.bfloat16)
                mixed=self.torch.matmul(probability,value).permute(1,0,2)
                result[begin:end]=self.arithmetic._host(mixed)
                del q,score,probability,mixed
        self.check();return result


__all__=['K3BF16VisionOps']
