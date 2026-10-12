"""V4.1 incremental device backbone with original compact cache transport."""
import hashlib
import threading
from pathlib import Path
import numpy as np
from .publisher import PublisherV41Execution
from .device_arithmetic import DevicePublisherV41Arithmetic
from ..publisher_checkpoint.v41_profile import V41PublisherProfile


class DevicePublisherV41Execution(PublisherV41Execution):
    implementation = 'torch-device-publisher-v41-packed-cache-v1'

    def __init__(self,profile,provider,*,library=None,max_bytes=64*1024**2):
        arithmetic = getattr(provider,'arithmetic',None)
        if (type(arithmetic) is not DevicePublisherV41Arithmetic
                or not isinstance(profile,V41PublisherProfile) or profile != arithmetic.payload.profile
                or type(max_bytes) is not int or max_bytes<=0
                or profile.streams>8 or profile.hidden_size>16384
                or profile.query_rank>65536 or profile.expert_intermediate>65536
                or profile.shared_experts!=1 or profile.sinkhorn_iterations>64
                or profile.scoring_function!='sqrtsoftplus'
                or profile.routing_method!='noaux_tc' or not profile.normalize_routed_weights):
            raise ValueError('matching admitted V4.1 device arithmetic and publisher profile required')
        self.profile,self.provider,self.max_bytes = profile,provider,max_bytes
        self.arithmetic = arithmetic
        self._cancellation = threading.local()
        self._lock,self._busy,self._closed = threading.RLock(),False,False
        self._tokens,self._token_types = (),()
        self._state = self._empty()
        self._release = lambda: None
        self.identity = hashlib.sha256(repr((profile,provider.identity,
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'v41-device-packed-row-online-fp32-v1')).encode()).hexdigest()

    def _kernel(self,*args,**kwargs):
        self._cancel_rows()
        return self.arithmetic._kernel(*args,**kwargs)

    def _pack_row(self,row,encoding):
        self._cancel_rows()
        return self.arithmetic.pack_row(row,encoding)

    def _target_mean(self,residual):
        self.arithmetic._admit(residual)
        t = self.arithmetic.torch
        with t.inference_mode():
            value = self.arithmetic._tensor(residual).reshape(self.profile.streams,self.profile.hidden_size)
            return self.arithmetic._host(value.mean(0).to(t.bfloat16))

    def _packed_reduce(self,query,rows,auxiliary,*,heads,width,scale,indexing):
        rows = tuple(rows)
        q,aux = np.asarray(query),np.asarray(auxiliary)
        if (q.size!=heads*width or aux.size!=heads or not rows
                or not np.isfinite(q).all() or not np.isfinite(aux).all()
                or any(row.size!=width for row in rows)):
            raise ValueError('complete finite packed publisher reduction geometry required')
        required = q.nbytes*6+aux.nbytes*8+width*32+(len(rows)*8 if indexing else heads*width*16)
        if required>self.max_bytes:
            raise MemoryError('V4.1 packed reduction exceeds runtime admission')
        self.arithmetic._admit(q,aux,extra=required)
        t = self.arithmetic.torch
        with t.inference_mode():
            query = self.arithmetic._tensor(q).reshape(heads,width)
            auxiliary = self.arithmetic._tensor(aux).reshape(heads)
            if indexing:
                scores = t.empty(len(rows),device=self.arithmetic.device)
                for i,row in enumerate(rows):
                    self._cancel_rows()
                    key = self.arithmetic.decode_row(row)
                    scores[i] = ((query@key).clamp(min=0)*auxiliary).sum()*scale
                return self.arithmetic._host(scores)
            maximum = auxiliary.clone()
            denominator = t.ones_like(maximum)
            numerator = t.zeros((heads,width),device=self.arithmetic.device)
            for row in rows:
                self._cancel_rows()
                key = self.arithmetic.decode_row(row)
                score = (query@key)*scale
                next_maximum = t.maximum(maximum,score)
                prior,weight = (maximum-next_maximum).exp(),(score-next_maximum).exp()
                numerator = numerator*prior[:,None]+weight[:,None]*key
                denominator = denominator*prior+weight
                maximum = next_maximum
            self._cancel_rows()
            return self.arithmetic._host(numerator/denominator[:,None])
