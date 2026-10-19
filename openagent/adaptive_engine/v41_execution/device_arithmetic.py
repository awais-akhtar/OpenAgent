"""Publisher V4.1 arithmetic on an explicitly pinned Torch accelerator.

Host arrays are transport boundaries. Neural reductions, activation rounding,
packed decoding and cache quantization execute on the selected device.
"""
import hashlib
import math
import threading
from pathlib import Path
from contextlib import contextmanager, ExitStack
import numpy as np
from .publisher_arithmetic import NativePublisherV41Arithmetic
from .publisher_cache import PackedPublisherRow, _FP4, _FP8, _FORMATS, packed_nbytes


class DevicePublisherV41Arithmetic(NativePublisherV41Arithmetic):
    @contextmanager
    def cancellation_scope(self,cancel):
        with super().cancellation_scope(cancel):
            providers = []
            seen = set()
            for provider in (self.fp8_provider, self.fp4_provider):
                if provider is not None and id(provider) not in seen:
                    seen.add(id(provider))
                    providers.append(provider)
            if not providers:
                yield self
            else:
                with ExitStack() as scopes:
                    for provider in providers:
                        scopes.enter_context(provider.cancellation_scope(cancel))
                    yield self

    def __init__(self,payload,*,device,torch_version,max_bytes,tokenizer=None,
                 compressed_ids=None,projection_backend=None,fp4_provider=None):
        import torch
        from ..k3_execution.device_policy import admit_device
        from ..k3_execution.precision import check_precision
        if str(torch.__version__) != torch_version or type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('explicit Torch pin and positive V4.1 arithmetic budget required')
        self.torch,self.device,self.max_bytes = torch,admit_device(torch,device),max_bytes
        check_precision(torch,self.device)
        self.payload,self.rules = payload,{r.sources[0]:r for r in payload.contracts}
        if payload.unsupported or len(self.rules) != len(payload.contracts):
            raise ValueError('complete supported original V4.1 source contracts required')
        self._lock,self._cancellation = threading.RLock(),threading.local()
        self._closed,self._release,self._library_identities = False,None,()
        self._fp4,self._fp8 = None,None
        self._projection_backend = projection_backend
        self._projection_identity = None if projection_backend is None else projection_backend.identity
        if projection_backend is not None and not callable(getattr(projection_backend,'bind_arithmetic',None)):
            # Distributed owners must declare the same arithmetic device.
            if getattr(projection_backend,'device',None) != self.device:
                raise ValueError('V4.1 collective and arithmetic devices differ')
        for name in payload.sharded_projections:
            if (projection_backend is None or not projection_backend.handles(name)
                    or projection_backend.shape(name) != self.rules[name].shape):
                raise ValueError('omitted V4.1 matrix requires its exact projection owner')
        self.fp4_provider = fp4_provider
        self.fp8_provider = payload.fp8_provider
        self._map,self._hash_metadata = None,{}
        eg = payload.profile.engram
        if eg.layers:
            if payload.engram_provider is None:
                raise ValueError('original selected Engram row owner required')
            if compressed_ids is None:
                if tokenizer is None:
                    raise ValueError('Engram tokenizer or explicit compressed vocabulary required')
                compressed_ids = self._compressed_map(tokenizer,payload.profile.vocabulary)
            ids = np.asarray(compressed_ids)
            if (ids.shape != (payload.profile.vocabulary,) or ids.dtype.kind not in 'iu'
                    or np.any(ids<0) or np.any(ids>=eg.compressed_vocabulary)
                    or np.unique(ids).size != eg.compressed_vocabulary):
                raise ValueError('exact publisher compressed tokenizer map required')
            self._map = np.frombuffer(ids.astype(np.uint64).tobytes(),np.uint64)
            self._make_hash_metadata()
        source_identity=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.collective_policy_identity=hashlib.sha256(repr(('v41-device-fp32-bf16-packed-v1',
            source_identity,torch_version,self.device.type,getattr(torch.version,'hip',None),max_bytes)).encode()).hexdigest()
        resident_identity=hashlib.sha256()
        for name,value in sorted(payload.tensors.items()):
            resident_identity.update(name.encode())
            resident_identity.update(memoryview(np.ascontiguousarray(value)).cast('B'))
        self.identity = hashlib.sha256(repr(('v41-device-fp32-bf16-packed-v1',source_identity,resident_identity.hexdigest(),
            torch_version,str(self.device),getattr(torch.version,'hip',None),payload.config_sha256,
            max_bytes,self._projection_identity,
            tuple(getattr(x,'identity',None) for x in (payload.expert_provider,
                payload.fp8_provider,payload.engram_provider,payload.embedding_provider,fp4_provider)),
            None if self._map is None else hashlib.sha256(self._map.tobytes()).hexdigest())).encode()).hexdigest()

    def _tensor(self,value):
        array = np.ascontiguousarray(value,dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError('finite V4.1 transport tensor required')
        return self.torch.tensor(array,device=self.device,dtype=self.torch.float32)

    def status(self):
        return dict(implementation='v41-device-publisher-arithmetic',identity=self.identity,
            device=str(self.device),torch_version=str(self.torch.__version__),
            projection_owner=self._projection_identity,closed=self._closed,
            numerical_parity_verified=False,platform_certified=False)

    def _host(self,value):
        if not bool(self.torch.isfinite(value).all()):
            raise ArithmeticError('nonfinite V4.1 device result')
        return value.float().cpu().numpy().copy()

    def _admit(self,*values,extra=0):
        from ..k3_execution.precision import check_precision
        self._ensure_open()
        check_precision(self.torch,self.device)
        if sum(np.asarray(v).size*24 for v in values)+extra > self.max_bytes:
            raise MemoryError('V4.1 device operation exceeds explicit buffer budget')

    def _nearest(self,value,levels):
        t = self.torch
        table = t.tensor(levels,device=self.device)
        # Bucketizing the monotonically increasing levels bounds scratch to O(N).
        hi = t.bucketize(value.contiguous(),table).clamp(max=len(levels)-1)
        lo = (hi-1).clamp(min=0)
        dl,dh = (value-table[lo]).abs(),(value-table[hi]).abs()
        return t.where((dh<dl)|((dh==dl)&((hi%2)==0)),hi,lo)

    def _quantize(self,row,mode):
        t = self.torch
        group,bits = (16,4) if mode==2 else (32,4 if mode==3 else 8)
        if row.numel()%group:
            raise ValueError('complete publisher activation scale groups required')
        blocks = row.reshape(-1,group)
        maximum = blocks.abs().amax(-1)
        if mode==2:
            scale_codes = self._nearest(maximum.clamp(min=6*2**-9)/6,_FP8)
            scales = t.tensor(_FP8,device=self.device)[scale_codes]
        else:
            raw = maximum.clamp(min=6*2**-126 if bits==4 else 1e-4)/(6 if bits==4 else 448)
            exponent = t.ceil(t.log2(raw))
            scale_codes = (exponent+127).long()
            scales = t.pow(t.tensor(2.,device=self.device),exponent)
            if bool(((scale_codes<0)|(scale_codes>=255)).any()):
                raise ValueError('publisher activation scale outside finite UE8M0 range')
        if not bool((scales>0).all()) or not bool(t.isfinite(scales).all()):
            raise ArithmeticError('nonfinite publisher activation scale')
        levels = _FP4 if bits==4 else _FP8
        codes = self._nearest((blocks/scales[:,None]).abs(),levels)
        codes = codes | (t.signbit(blocks).long()*(8 if bits==4 else 128))
        values = t.tensor(levels,device=self.device)[codes & (7 if bits==4 else 127)]
        values = t.where((codes & (8 if bits==4 else 128))!=0,-values,values)*scales[:,None]
        return values.reshape(row.shape),codes.reshape(-1),scale_codes

    def pack_row(self,row,encoding):
        row = np.asarray(row,dtype=np.float32).reshape(-1)
        packed_nbytes(int(row.size),encoding)
        self._admit(row)
        modes = {'fp8-e4m3-g32-ue8m0':1,'fp4-e2m1-g16-e4m3':2,'fp4-e2m1-g32-ue8m0':3}
        with self.torch.inference_mode():
            _,codes,scales = self._quantize(self._tensor(row),modes[encoding])
            if modes[encoding] != 1:
                codes = codes[::2]|(codes[1::2]<<4)
            return PackedPublisherRow(encoding,row.size,codes.to(self.torch.uint8).cpu().numpy().tobytes(),
                scales.to(self.torch.uint8).cpu().numpy().tobytes())

    def decode_row(self,row):
        if type(row) is not PackedPublisherRow:
            raise ValueError('complete immutable publisher packed row required')
        t = self.torch
        group,bits = _FORMATS[row.encoding]
        codes = t.tensor(list(row.values),device=self.device,dtype=t.int32)
        if bits==4:
            codes = t.stack((codes & 15,codes>>4),dim=-1).reshape(-1)
        table = t.tensor(_FP4 if bits==4 else _FP8,device=self.device)
        values = table[codes & (7 if bits==4 else 127)]
        values = t.where((codes & (8 if bits==4 else 128))!=0,-values,values)
        scales = t.tensor(list(row.scales),device=self.device,dtype=t.int32)
        scales = (t.tensor(_FP8,device=self.device)[scales] if row.encoding=='fp4-e2m1-g16-e4m3'
                  else t.pow(t.tensor(2.,device=self.device),(scales-127).float()))
        return (values.reshape(-1,group)*scales[:,None]).reshape(-1).to(t.bfloat16).float()

    def _kernel(self,op,x,w=(),z=(),*,a=0,b=0,c=0,p=0,n=None):
        with self._lock:
            result=self._device_kernel(op,x,w,z,a=a,b=b,c=c,p=p,n=n)
            self.check_cancelled()
            return result

    def _device_kernel(self,op,x,w=(),z=(),*,a=0,b=0,c=0,p=0,n=None):
        self._admit(x,w,z,extra=16*(np.asarray(x).size if n is None else n))
        t = self.torch
        with t.inference_mode():
            x,w,z = (self._tensor(v).reshape(-1) for v in (x,w,z))
            rms = lambda value: t.rsqrt(value.square().mean()+p)
            if op==0: out = w.reshape(a,b)@x
            elif op==1: out = x*rms(x)*w
            elif op==2: out = x+w
            elif op==4: out = (x.reshape(a,b)*t.softmax(w.reshape(a,b),dim=0)).sum(0)
            elif op==8 or op==22:
                gate = x.clamp(max=p) if p>0 else x
                up = w.clamp(min=-p,max=p) if p>0 else w
                out = t.nn.functional.silu(gate)*up
                if op==22: out = out*z[0]
            elif op==10: out = x+p*w
            elif op==11:
                out = t.stack(tuple(x[i:i+b].max() for i in range(0,a,b)))
            elif op==13 or (op==23 and c==0): out = (x.reshape(a,b)*w[:a,None]).sum(0)
            elif op==14 or op==23:
                out = w[a:a+a*a].reshape(a,a)@x.reshape(a,b)+w[a+a*a:,None]*z
            elif op==16:
                width = a*(a+2)
                if z.numel()!=width+5 or w.numel()!=width:
                    raise ValueError('exact publisher mHC parameters required')
                inverse = t.rsqrt(x.square().mean()+z[-2])
                pre = t.sigmoid(w[:a]*inverse*z[width]+z[:a])+z[-1]
                post = 2*t.sigmoid(w[a:2*a]*inverse*z[width+1]+z[a:2*a])
                matrix = t.softmax((w[2*a:]*inverse*z[width+2]+z[2*a:width]).reshape(a,a),dim=-1)+z[-1]
                for step in range(c):
                    if step: matrix = matrix/(matrix.sum(-1,keepdim=True)+z[-1])
                    matrix = matrix/(matrix.sum(0,keepdim=True)+z[-1])
                out = t.cat((pre,matrix.reshape(-1),post))
            elif op==17 or op==3:
                width = b
                rows = x.reshape(-1,width) if op==17 else x[-width:].reshape(1,width)
                d = t.arange(0,width,2,device=self.device,dtype=t.float32)
                theta = float(w[0]) if op==17 else p
                frequency = t.pow(t.tensor(theta,device=self.device),-d/width)
                if op==17 and float(w[1])>0:
                    orig,factor,fast,slow = (float(v) for v in w[1:5])
                    low = max(0,math.floor(width*math.log(orig/(fast*2*math.pi))/(2*math.log(theta))))
                    high = min(width-1,math.ceil(width*math.log(orig/(slow*2*math.pi))/(2*math.log(theta))))
                    ramp = ((d/2-low)/max(high-low,1e-3)).clamp(0,1)
                    frequency = frequency*(1-ramp+ramp/factor)
                angle = a*frequency*(c if op==17 else 1)
                even,odd = rows[:,::2],rows[:,1::2]
                rotated = t.stack((even*angle.cos()-odd*angle.sin(),even*angle.sin()+odd*angle.cos()),dim=-1).reshape(-1)
                out = rotated if op==17 else t.cat((x[:-width],rotated))
            elif op==18:
                scores = (x.reshape(a,b)@w.reshape(c,b).T)*p
                maximum = t.maximum(scores.amax(-1),z)
                ex = (scores-maximum[:,None]).exp()
                out = (ex@w.reshape(c,b))/(ex.sum(-1)+(z-maximum).exp())[:,None]
            elif op==19:
                out = (((x.reshape(a,b)@w.reshape(c,b).T)*p).clamp(min=0)*z[:,None]).sum(0)
            elif op==20:
                q,k = x.reshape(a,b),w.reshape(a,b)
                q = q*t.rsqrt(q.square().mean(-1,keepdim=True)+p)
                k = k*t.rsqrt(k.square().mean(-1,keepdim=True)+p)
                score = ((q*z[b:b+a*b].reshape(a,b))*(k*z[b+a*b:].reshape(a,b))).sum(-1)*(b**-0.5)
                score = t.where(t.signbit(score),-t.ones_like(score),t.ones_like(score))*score.abs().clamp(min=1e-6).sqrt()
                out = x.reshape(a,b)+t.sigmoid(score)[:,None]*z[:b]
            elif op==21:
                affinity = (t.nn.functional.softplus(x/z[0]).sqrt() if c==0 else
                            t.sigmoid(x/z[0]) if c==1 else t.softmax(x/z[0],dim=-1))
                ids = t.argsort(affinity+w,descending=True,stable=True)[:b]
                weights = affinity[ids]
                if bool(z[2]) and b>1: weights = weights/(weights.sum()+1e-20)
                out = t.cat((ids.float(),weights*z[1]))
            elif op==24:
                blocks = t.stack(tuple(x[i:i+a].max() for i in range(0,x.numel(),a)))
                blocks[-1] = t.finfo(t.float32).max
                picks = t.argsort(blocks,descending=True,stable=True)[:b]
                selected = t.zeros_like(blocks,dtype=t.bool)
                selected[picks] = True
                out = selected[t.arange(x.numel(),device=self.device)//a].float()
            elif op==25:
                ids = t.argsort(t.where(w!=0,x,t.full_like(x,-t.inf)),descending=True,stable=True)[:b]
                out = t.sort(t.where(w[ids]!=0,ids,t.full_like(ids,-1))).values.float()
            elif op==26:
                out = x.to(t.bfloat16).float() if a==0 else self._quantize(x,a)[0]
                if a in (1,2,3): out = out.to(t.bfloat16).float()
            else: raise ValueError('unsupported publisher device operation: '+str(op))
            expected = x.numel() if n is None else n
            if out.numel()!=expected:
                raise ValueError('publisher device result extent differs from operation contract')
            return self._host(out.reshape(-1))

    def _project(self,source,x):
        self._ensure_open()
        rule = self.rules[source]
        if len(rule.shape)!=2 or source.endswith('.wo_a.weight'):
            raise ValueError('ordinary original publisher matrix required')
        rows,cols = rule.shape
        x = np.ascontiguousarray(x,dtype=np.float32).reshape(-1)
        if x.size!=cols: raise ValueError('publisher input width differs')
        fp32 = (source=='head.weight' or source.endswith(('_fn','ffn.gate.weight')) or
            ('.attn.compressor.' in source and self.payload.profile.layers[int(source.split('.')[1])].compression_ratio>1))
        x = self._kernel(26,x,a=4) if rule.encoding in ('fp8-e4m3-g32x32','v41-fp4-g32') else x
        bank = self._projection_backend
        if bank is not None and bank.identity!=self._projection_identity:
            raise ValueError('publisher projection owner changed')
        if bank is not None and bank.handles(source):
            result = bank.linear(source,x)
        elif rule.encoding in ('fp8-e4m3-g32x32','v41-fp4-g32'):
            provider = self.payload.fp8_provider if rule.encoding=='fp8-e4m3-g32x32' else self.fp4_provider
            if provider is None:
                raise ValueError('bounded original packed row projection owner required')
            tile = min(provider.tile_rows(source),(self.max_bytes-48*cols-rows*16)//(cols*64+32))
            if rule.encoding=='fp8-e4m3-g32x32' and tile<rows:
                tile = tile//32*32
            if tile<=0:
                raise MemoryError('device packed projection cannot admit one source row block')
            result = np.empty(rows,np.float32)
            for start in range(0,rows,tile):
                self.check_cancelled()
                count = min(tile,rows-start)
                with provider.lease(source,row_start=start,row_count=count) as tables:
                    result[start:start+count] = self.packed_matvec(tables[rule.sources[0]],
                        tables[rule.sources[1]],x,rows=count,cols=cols,encoding=rule.encoding)
        else:
            matrix = self.payload.tensors[source]
            if np.asarray(matrix).shape!=(rows,cols) or np.asarray(matrix).dtype!=np.float32:
                raise ValueError('exact original resident publisher matrix required')
            tile=min(rows,(self.max_bytes-48*cols-rows*16)//(cols*32+32))
            if tile<=0:raise MemoryError('resident publisher projection cannot admit one complete device row')
            result=np.empty(rows,np.float32)
            with self.torch.inference_mode():
                value=self._tensor(x)
                if not fp32:value=value.to(self.torch.bfloat16)
                for begin in range(0,rows,tile):
                    self.check_cancelled()
                    end=min(rows,begin+tile)
                    self._admit(matrix[begin:end],x,extra=rows*16)
                    weight=self._tensor(matrix[begin:end])
                    if not fp32:weight=weight.to(self.torch.bfloat16)
                    result[begin:end]=self._host(weight@value)
                    del weight
        result = np.asarray(result)
        if result.shape!=(rows,) or result.dtype!=np.float32 or not np.isfinite(result).all():
            raise ArithmeticError('invalid publisher device projection')
        return result if fp32 else self.cast(result)

    def packed_matvec(self,packed,scales,x,*,rows,cols,encoding):
        with self._lock:
            self._ensure_open()
            return self._packed_matvec_open(packed,scales,x,rows=rows,cols=cols,encoding=encoding)

    def _packed_matvec_open(self,packed,scales,x,*,rows,cols,encoding):
        packed,scales,x=np.asarray(packed),np.asarray(scales),np.asarray(x)
        if (type(rows) is not int or type(cols) is not int or rows<=0 or cols<=0 or cols%32
                or packed.dtype!=np.uint8 or scales.dtype!=np.uint8
                or packed.ndim!=1 or scales.ndim!=1 or not packed.flags.c_contiguous
                or not scales.flags.c_contiguous or x.shape!=(cols,)
                or x.dtype!=np.float32 or not x.flags.c_contiguous or not np.isfinite(x).all()):
            raise ValueError('exact original packed device projection geometry and byte sources required')
        # Reject malformed format payloads before copying any bytes to the
        # selected device. These checks mirror the consumer's true tile layout,
        # not merely the nominal source tensor dimensions.
        if encoding=='v41-fp4-g32':
            if packed.size!=rows*cols//2 or scales.size!=rows*cols//32:
                raise ValueError('exact original group32 FP4 weight extent required')
            if bool((scales==255).any()):
                raise ValueError('nonfinite original FP4 scale')
        elif encoding=='fp8-e4m3-g32x32':
            if packed.size!=rows*cols or scales.size!=math.ceil(rows/32)*math.ceil(cols/32):
                raise ValueError('exact original tiled32 FP8 weight extent required')
            if bool(((packed&127)==127).any()) or bool((scales==255).any()):
                raise ValueError('nonfinite original FP8 weight or scale')
        else:
            raise ValueError('explicit original publisher packed encoding required')
        self._admit(packed,scales,x,extra=rows*cols*24)
        t = self.torch
        with t.inference_mode():
            codes = t.tensor(np.array(packed,copy=True),device=self.device,dtype=t.int32)
            scale = t.tensor(np.array(scales,copy=True),device=self.device,dtype=t.int32)
            if encoding=='v41-fp4-g32':
                codes = t.stack((codes&15,codes>>4),dim=-1).reshape(rows,cols)
                matrix = t.tensor(_FP4,device=self.device)[codes&7]
                matrix = t.where((codes&8)!=0,-matrix,matrix)
                matrix = matrix*t.pow(t.tensor(2.,device=self.device),(scale.reshape(rows,-1)-127).float()).repeat_interleave(32,dim=1)
            elif encoding=='fp8-e4m3-g32x32':
                codes = codes.reshape(rows,cols)
                matrix = t.tensor(_FP8,device=self.device)[codes&127]
                matrix = t.where((codes&128)!=0,-matrix,matrix)
                matrix = matrix*t.pow(t.tensor(2.,device=self.device),(scale.reshape(math.ceil(rows/32),-1)-127).float()).repeat_interleave(32,0).repeat_interleave(32,1)[:rows,:cols]
            return self._host(matrix@self._tensor(x))

    def bf16_matvec(self,weight,value):
        self._ensure_open()
        weight,value=np.asarray(weight),np.asarray(value)
        if weight.ndim!=2 or value.shape!=(weight.shape[1],):
            raise ValueError('complete BF16 matrix and vector geometry required')
        rows,columns=weight.shape
        tile=min(rows,(self.max_bytes-48*columns-16*rows)//(32*columns+32))
        if tile<=0:raise MemoryError('BF16 device projection cannot admit one complete row')
        result=np.empty(rows,np.float32)
        t=self.torch
        with t.inference_mode():
            x=self._tensor(value).to(t.bfloat16)
            for begin in range(0,rows,tile):
                self.check_cancelled()
                end=min(rows,begin+tile)
                self._admit(weight[begin:end],value,extra=16*rows)
                matrix=self._tensor(weight[begin:end]).to(t.bfloat16)
                result[begin:end]=self._host(matrix@x)
                del matrix
        return result

    def grouped_linear(self,source,x,*,groups):
        with self._lock:
            self._ensure_open()
            rule=self.rules[source]
            if not source.endswith('.wo_a.weight') or len(rule.shape)!=2:
                raise ValueError('original group-local wo_a projection required')
            rows,columns=rule.shape
            value=np.asarray(x)
            if (type(groups) is not int or groups<=0 or rows%groups or
                    value.dtype!=np.float32 or value.ndim not in (2,3)
                    or value.shape[-2:]!=(groups,columns) or not value.size):
                raise ValueError('exact finite grouped publisher inputs required')
            self._admit(value,extra=rows*(1 if value.ndim==2 else len(value))*16)
            bank=self._projection_backend
            if bank is not None and bank.identity!=self._projection_identity:
                raise ValueError('grouped projection owner changed')
            if bank is not None and bank.handles(source):
                result=np.asarray(bank.grouped_linear(source,value,groups=groups))
            else:
                weights=np.asarray(self.payload.tensors[source])
                if weights.shape!=(rows,columns):
                    raise ValueError('complete grouped original publisher matrix required')
                weights=weights.reshape(groups,rows//groups,columns)
                inputs=value[None] if value.ndim==2 else value
                result=np.stack(tuple(np.stack(tuple(self.bf16_matvec(weights[g],batch[g])
                    for g in range(groups))) for batch in inputs))
                if value.ndim==2:result=result[0]
            expected=(*value.shape[:-2],groups,rows//groups)
            if result.shape!=expected or result.dtype!=np.float32 or not np.isfinite(result).all():
                raise ArithmeticError('invalid grouped device publisher output')
            return result.reshape(*value.shape[:-2],rows)

    def _engram_rows(self,layer,tokens):
        self._ensure_open()
        eg = self.payload.profile.engram
        multipliers,primes,offsets = self._hash_metadata[layer]
        pad = int(self._map[eg.padding_token])
        lookback = [int(self._map[tokens[-1-i]]) if i<len(tokens) else pad for i in range(eg.maximum_order)]
        ids,hashed = [],0
        for order in range(eg.maximum_order):
            hashed ^= (lookback[order]*int(multipliers[order]))&((1<<64)-1)
            if order:
                for head in range(eg.heads_per_order):
                    slot = (order-1)*eg.heads_per_order+head
                    ids.append(hashed%int(primes[slot])+int(offsets[slot]))
        source = f'layers.{layer}.engram.embed.weight'
        with self.payload.engram_provider.lease_rows(source,tuple(ids)) as tables:
            self._admit(tables[source])
            return self.cast(np.array(tables[source],dtype=np.float32,copy=True).reshape(-1))

    def auxiliary_math(self,operation,*arrays,epsilon=1e-6,axis=-1):
        self._admit(*arrays)
        t = self.torch
        with t.inference_mode():
            values = [self._tensor(x) for x in arrays]
            if operation=='norm':
                out = values[0]*t.rsqrt(values[0].square().mean(-1,keepdim=True)+epsilon)*values[1]
            elif operation in ('add','add_fp32'): out = values[0]+values[1]
            elif operation=='multiply': out = values[0]*values[1]
            elif operation=='silu': out = t.nn.functional.silu(values[0])
            elif operation=='gelu': out = t.nn.functional.gelu(values[0],approximate='none')
            elif operation=='streams': out = (values[0]*values[1][...,None]).sum(axis)
            else: raise ValueError('unknown device auxiliary arithmetic')
            return self._host(out if operation=='add_fp32' else out.to(t.bfloat16))

    def attention_rows(self,q,key_banks,value_banks,*,sink=None,max_bytes,check_cancelled=None):
        q = np.asarray(q)
        keys,values = tuple(key_banks),tuple(value_banks)
        if (q.ndim!=3 or not all(q.shape) or len(keys)!=len(values) or not keys
                or any(np.asarray(k).ndim!=3 and np.asarray(k).ndim!=2 for k in keys)):
            raise ValueError('complete device auxiliary attention inventory required')
        tokens,heads,width = q.shape
        required = q.size*32+heads*width*32+heads*64
        if type(max_bytes) is not int or required>max_bytes:
            raise MemoryError('device auxiliary attention exceeds workspace')
        self._admit(q,extra=required)
        t = self.torch
        result = np.empty_like(q)
        with t.inference_mode():
            for token in range(tokens):
                query = self._tensor(q[token])
                maximum = t.full((heads,),-t.inf,device=self.device) if sink is None else self._tensor(sink)
                denominator = t.zeros_like(maximum) if sink is None else t.ones_like(maximum)
                numerator = t.zeros((heads,width),device=self.device)
                for kb,vb in zip(keys,values):
                    if len(kb)!=len(vb) or not len(kb):
                        raise ValueError('matching occupied auxiliary key/value rows required')
                    for kr,vr in zip(kb,vb):
                        self.check_cancelled()
                        if check_cancelled is not None: check_cancelled()
                        key,value = self._tensor(kr),self._tensor(vr)
                        if key.shape not in ((width,),(heads,width)) or value.shape!=key.shape:
                            raise ValueError('matching auxiliary attention head widths required')
                        score = (query*key).sum(-1)*(width**-0.5)
                        next_maximum = t.maximum(maximum,score)
                        prior,weight = (maximum-next_maximum).exp(),(score-next_maximum).exp()
                        numerator = numerator*prior[:,None]+weight[:,None]*value
                        denominator = denominator*prior+weight
                        maximum = next_maximum
                result[token] = self._host(numerator/denominator[:,None])
        return result

    def close(self):
        with self._lock:
            if getattr(self,'_borrowers',0):
                raise RuntimeError('publisher execution still borrows device arithmetic')
            self._closed = True
            self._map,self._hash_metadata = None,{}
