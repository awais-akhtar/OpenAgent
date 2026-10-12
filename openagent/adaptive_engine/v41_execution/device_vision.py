"""Device vision and aligner equations with bounded host transport."""
import numpy as np
from .publisher_vision import PublisherV41Vision


class DevicePublisherV41Vision(PublisherV41Vision):
    def _norm(self,x,name):
        return self.provider.arithmetic.auxiliary_math('norm',x,
            self._tensor(name+'.weight',(self.width,)),epsilon=1e-6)

    def _linear(self,name,x,width,bias=False):
        out = super()._linear(name,x,width,False)
        if bias:
            out = self.provider.arithmetic.auxiliary_math('add',out,self._tensor(name+'.bias',(width,)))
        return out

    def encode_image(self,patches,n_h,n_w,*,workspace_bytes=None):
        budget = self.max_bytes if workspace_bytes is None else workspace_bytes
        if type(budget) is not int or not 0<budget<=self.max_bytes:
            raise ValueError('positive admitted device vision workspace required')
        if any(type(v) is not int or v<=0 for v in (n_h,n_w)):
            raise ValueError('positive image patch grid required')
        n,d = n_h*n_w,self.width//self.heads
        mid,r = self.config['intermediate_size'],self.config['downsample_ratio']
        retained = 4*n*(24*self.width+8*mid+3*self.config['patch_size']**2)
        if retained>=budget:
            raise MemoryError('device vision workspace exceeds admission')
        arithmetic = self.provider.arithmetic
        t = arithmetic.torch
        math = arithmetic.auxiliary_math
        patch = self.config['patch_size']
        x = self._linear('vision.patch_embed.proj',self._cast(self._array(patches,(n,3*patch*patch))),self.width,True)
        arithmetic._admit(x,extra=n*d*32)
        with t.inference_mode():
            inv = self.theta**(-t.arange(0,d//2,2,device=arithmetic.device,dtype=t.float32)/(d//2))
            yy,xx = t.meshgrid(t.arange(n_h,device=arithmetic.device),t.arange(n_w,device=arithmetic.device),indexing='ij')
            phase = t.cat((yy.reshape(-1,1)*inv,xx.reshape(-1,1)*inv),dim=1)
            cosine,sine = phase.cos()[:,None,:],phase.sin()[:,None,:]
            def rotate(value):
                arithmetic._admit(value)
                a,b = arithmetic._tensor(value).chunk(2,dim=-1)
                return arithmetic._host(t.cat((a*cosine-b*sine,b*cosine+a*sine),dim=-1).to(t.bfloat16))
            for layer in range(self.config['num_hidden_layers']):
                self.provider.check_cancelled()
                prefix = f'vision.blocks.{layer}.'
                qkv = self._linear(prefix+'attn.wqkv',self._norm(x,prefix+'norm1'),3*self.width,True)
                q,k,v = [row.reshape(n,self.heads,d) for row in np.split(qkv,3,axis=-1)]
                out = self._cast(arithmetic.attention_rows(rotate(q),(rotate(k),),(v,),max_bytes=budget-retained)).reshape(n,self.width)
                x = math('add',x,self._linear(prefix+'attn.wo',out,self.width,True))
                gate,up = np.split(self._linear(prefix+'mlp.w1',self._norm(x,prefix+'norm2'),2*mid),2,axis=-1)
                activation = math('multiply',math('silu',gate),up)
                x = math('add',x,self._linear(prefix+'mlp.w2',activation,self.width))
            x = self._norm(x,'vision.norm').reshape(n_h,n_w,self.width)
            hh,ww = (n_h+r-1)//r,(n_w+r-1)//r
            if 4*(4*hh*ww*self.width*r*r+8*hh*ww*self.hidden+n*self.width)>budget:
                raise MemoryError('device aligner workspace exceeds admission')
            x = np.pad(x,((0,hh*r-n_h),(0,ww*r-n_w),(0,0)))
            x = x.reshape(hh,r,ww,r,self.width).transpose(0,2,4,1,3).reshape(hh*ww,-1)
            x = self._linear('aligner.w1',x,self.hidden,True)
            return self._linear('aligner.w2',math('gelu',x),self.hidden,True)
