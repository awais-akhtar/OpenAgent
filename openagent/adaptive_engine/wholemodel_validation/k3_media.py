"""Whole-model media observations with real expanded-prefix continuation.

This module supplies candidate execution and comparison code, never a default
oracle or fabricated parity evidence. It is invoked only by a later validation
phase using explicitly pinned candidate/oracle adapters and processor fixtures.
"""
from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping
import sys
import ast
import struct
import numpy as np
from .contracts import Observation
from .cache_readers import CacheContext
from .comparison import compare,Tolerance
from .runner import pinned_artifacts,metadata as adapter_metadata,digest_file
from .adapters import FreshOwner


@dataclass(frozen=True)
class K3MediaCase:
    token_ids: tuple
    media_token_id: int
    pixels: np.ndarray
    grids: np.ndarray
    expanded_tokens: int

    def admit(self,max_bytes):
        if (type(max_bytes) is not int or max_bytes<=0 or type(self.token_ids) is not tuple
                or not self.token_ids or any(type(v) is not int or v<0 for v in self.token_ids)
                or type(self.media_token_id) is not int or self.media_token_id<0
                or type(self.expanded_tokens) is not int or self.expanded_tokens<=0):
            raise ValueError('explicit occupied media comparison fixture required')
        if (type(self.pixels) is not np.ndarray or self.pixels.dtype!=np.float32 or self.pixels.ndim!=4
                or type(self.grids) is not np.ndarray or self.grids.dtype.kind not in 'iu'
                or self.grids.ndim!=2 or self.grids.shape[1]!=3 or not len(self.grids)
                or self.token_ids.count(self.media_token_id)!=len(self.grids)):
            raise ValueError('matching original processor patches and placeholder inventory required')
        if self.pixels.nbytes*3+self.grids.nbytes*3+len(self.token_ids)*128>max_bytes:
            raise MemoryError('complete media fixture exceeds admission')
        if not np.isfinite(self.pixels).all() or np.any(self.grids<=0):raise ValueError('finite positive processor fixture required')
        if sum(int(t)*int(h)*int(w) for t,h,w in self.grids)!=len(self.pixels):
            raise ValueError('processor patch rows differ from media grids')

    def identity(self):
        digest=hashlib.sha256(repr((self.token_ids,self.media_token_id,self.expanded_tokens)).encode())
        for value in (self.pixels,self.grids):
            digest.update(repr((value.shape,value.dtype.str)).encode())
            digest.update(memoryview(np.ascontiguousarray(value)).cast('B'))
        return digest.hexdigest()

    def freeze(self,max_bytes):
        """Detach both processor buffers into immutable bytes-owned storage."""
        self.admit(max_bytes)
        values=[np.frombuffer(value.tobytes(order='C'),dtype=value.dtype).reshape(value.shape)
                for value in (self.pixels,self.grids)]
        result=K3MediaCase(self.token_ids,self.media_token_id,*values,self.expanded_tokens)
        result.admit(max_bytes)
        if result.identity()!=self.identity():raise ValueError('processor fixture changed during capture')
        return result


class K3MediaAdapter(FreshOwner):
    def __init__(self,factory,metadata,cache_reader,*,max_bytes,observations=None):
        super().__init__()
        if not callable(factory) or not callable(cache_reader) or type(max_bytes) is not int or max_bytes<=0:
            raise ValueError('fresh complete media factory, observed reader and bounded admission required')
        self.factory=factory;self._metadata=copy.deepcopy(metadata);self.reader=cache_reader;self.max_bytes=max_bytes
        self.observations=copy.deepcopy(observations)
        self._metadata['execution_scope']='processor-media-expanded-prefix'

    def metadata(self):return copy.deepcopy(self._metadata)

    def new_session(self,*,seed):
        if type(seed) is not int or seed<0:raise ValueError('nonnegative integer seed required')
        executor=self.factory(seed=seed)
        self.claim(executor)
        try:return K3MediaSession(executor,self.reader,max_bytes=self.max_bytes,observations=self.observations)
        except BaseException as primary:
            try:executor.close()
            except BaseException as cleanup:
                primary.cleanup=executor.close;primary.add_note('media factory cleanup retained: '+repr(cleanup))
            raise


class K3MediaSession:
    def __init__(self,executor,cache_reader,*,max_bytes,observations=None):
        from ..k3_execution.executor import K3Execution
        if type(executor) is not K3Execution or executor.config.vision is None or not callable(cache_reader):
            raise ValueError('actual complete K3 vision executor and observed cache reader required')
        self.executor=executor;self.reader=cache_reader;self.max_bytes=max_bytes
        self.observations=copy.deepcopy(observations)
        if type(max_bytes) is not int or max_bytes<=0:raise ValueError('positive observation admission required')
        self._started=False;self.failed=False;self.closed=False

    @property
    def position(self):return self.executor.position

    def _active(self):
        if self.failed or self.closed:raise RuntimeError('failed or closed media session')

    def _transaction(self,operation):
        self._active()
        before=self.executor.snapshot()
        try:
            if self.observations is None:return operation()
            with self.executor.observation_scope(**self.observations) as recorder:
                result=operation();evidence=dict(result.evidence)
                evidence.update(recorder.export(rows=len(result.logits)))
                values=[result.logits,*result.cache.values(),*evidence.values()]
                if sum(value.nbytes+48*value.size for value in values)>self.max_bytes:
                    raise MemoryError('complete media routing/arithmetic observations exceed admission')
                return Observation(result.logits,result.cache,evidence)
        except BaseException as primary:
            try:self.executor.restore(before)
            except BaseException as rollback:
                self.failed=True
                primary.add_note('media observation rollback failed: '+repr(rollback))
            raise

    def _preflight(self,rows,*,feature_elements=0,fixture_bytes=0):
        """Reserve complete occupied cache exports before reader allocation."""
        config=self.executor.config
        position=self.position+rows
        if type(rows) is not int or rows<=0 or position>config.capacity:
            raise ValueError('occupied complete observation must fit executor capacity')
        cache_elements=0
        for kind in config.layer_types:
            if kind=='mla':cache_elements+=position*(config.kvrank+config.rope)
            else:
                taps=config.kernel if self.executor._fla_policy is not None else config.kernel-1
                cache_elements+=config.kda_heads*(config.kda_dim**2+3*config.kda_dim*taps)
        # Detached cache/logit arrays, snapshot state, paired comparison and
        # FP64 error/limit work. This bound covers both sides of the comparison.
        required=fixture_bytes+128*(cache_elements+rows*config.vocab+feature_elements)
        required+=256*(len(config.layer_types)+1)
        if required>self.max_bytes:raise MemoryError('complete media observation exceeds preflight admission')

    def _observe(self,logits,evidence=None):
        snapshot=self.executor.snapshot()
        cache=self.reader(CacheContext(snapshot,self.executor,None,self.executor.position,len(logits)))
        evidence={} if evidence is None else evidence
        if not isinstance(cache,Mapping) or not isinstance(evidence,Mapping):
            raise ValueError('named observed cache and evidence mappings required')
        values=[logits,*cache.values(),*evidence.values()]
        if any(type(value) is not np.ndarray or value.dtype.kind not in 'biuf' for value in values):
            raise ValueError('actual numeric observation arrays required')
        # Include detached observations plus comparator float64/error work.
        if sum(value.nbytes+value.size*48 for value in values)>self.max_bytes:
            raise MemoryError('complete observation and comparison work exceed admission')
        if any(not np.isfinite(value).all() for value in values):raise ValueError('nonfinite observation')
        return Observation(np.array(logits,copy=True),
            {key:np.array(value,copy=True) for key,value in cache.items()},
            {key:np.array(value,copy=True) for key,value in evidence.items()})

    def forward_media(self,case):
        self._active()
        case=case.freeze(self.max_bytes)
        if self._started or self.executor.position:raise ValueError('media fixture requires a fresh independent session')
        if (any(token>=self.executor.config.vocab for token in case.token_ids)
                or case.media_token_id>=self.executor.config.vocab
                or case.pixels.shape[1:]!=(3,self.executor.config.vision.patch,self.executor.config.vision.patch)):
            raise ValueError('processor patch geometry and vocabulary must match the model')
        expected=len(case.token_ids)-len(case.grids)+sum(self.executor._vision.projected_lengths(case.grids))
        if expected!=case.expanded_tokens:raise ValueError('declared expanded prefix differs from actual projector geometry')
        self._preflight(expected,feature_elements=(expected-len(case.token_ids)+len(case.grids))*
            self.executor.config.vision.text_hidden,
            fixture_bytes=3*(case.pixels.nbytes+case.grids.nbytes)+128*len(case.token_ids))
        # Observe the real encoder output separately from final logits. No
        # projected feature fixture substitutes for token-to-logit execution.
        def execute():
            features=self.executor.encode_media(case.pixels,case.grids)
            logits=self.executor.forward(case.token_ids,pixel_values=case.pixels,grid_thws=case.grids,
                media_token_id=case.media_token_id)
            if logits.shape!=(expected,self.executor.config.vocab) or self.executor.position!=expected:
                raise ValueError('actual complete media prefix geometry differs')
            return self._observe(logits,{f'vision.media.{i}':value for i,value in enumerate(features)})
        result=self._transaction(execute)
        self._started=True
        return result

    def forward(self,tokens):
        self._active()
        if not self._started:raise ValueError('media prefill must precede continuation')
        tokens=tuple(tokens)
        if not tokens or any(type(v) is not int or not 0<=v<self.executor.config.vocab for v in tokens):
            raise ValueError('occupied in-vocabulary continuation required')
        self._preflight(len(tokens))
        return self._transaction(lambda:self._observe(self.executor.forward(tokens)))
    def snapshot(self):
        self._active();return self.executor.snapshot()
    def restore(self,snapshot):
        self._active();self.executor.restore(snapshot);self._started=bool(self.position)
    def close(self):
        self.closed=True
        self.executor.close()


def run_media_comparison(candidate,oracle,case,*,offline_root,artifact_pins,cache_schema,
                         max_bytes,seed,generation_steps,tolerance,fixture_config):
    """Compare full media logits/features/caches, generation and restoration."""
    if type(tolerance) is not Tolerance or type(generation_steps) is not int or generation_steps<=0:
        raise ValueError('explicit comparison tolerances and positive generation coverage required')
    if type(seed) is not int or seed<0:raise ValueError('nonnegative integer comparison seed required')
    case=case.freeze(max_bytes)
    pins=pinned_artifacts({'artifacts':artifact_pins},offline_root)
    if not pins:raise ValueError('pinned source, checkpoint and processor fixture artifacts required')
    root=Path(offline_root).resolve(strict=True);fixture=(root/fixture_config).resolve(strict=True)
    if (not fixture.is_relative_to(root) or fixture.stat().st_size>65536 or
            not any(Path(row['resolved_path'])==fixture and row['kind']=='config' for row in pins)):
        raise ValueError('bounded pinned media fixture configuration required')
    from ..native_checkpoint import _pairs
    fixture_data=json.loads(fixture.read_text(encoding='utf-8'),object_pairs_hook=_pairs)
    if fixture_data.get('media_fixture_sha256')!=case.identity():
        raise ValueError('actual processor fixture differs from pinned media identity')
    spec={'artifacts':artifact_pins}
    metas=[adapter_metadata(candidate,spec),adapter_metadata(oracle,spec,oracle=True)]
    if any(meta.get('cache_projection') is not None for meta in metas):
        raise ValueError('media comparison requires complete physical caches without projected substitutes')
    if any(metas[0][key]!=metas[1][key] for key in
           ('model_type','config_sha256','weights_sha256','tokenizer_sha256','num_layers')):
        raise ValueError('candidate and oracle checkpoint identities must agree')
    if metas[0]['vocabulary_size']!=metas[1]['vocabulary_size'] or any(
            token>=metas[0]['vocabulary_size'] for token in case.token_ids):
        raise ValueError('media fixtures and both vocabularies must agree')
    if any(meta['cache_schema']!=cache_schema or meta.get('execution_scope')!='processor-media-expanded-prefix' for meta in metas):
        raise ValueError('both complete media adapters must declare identical observed cache schema and scope')
    if metas[0].get('empty_cache_shapes',{})!=metas[1].get('empty_cache_shapes',{}):
        raise ValueError('structural empty cache declarations differ')
    if metas[0].get('evidence_schema',{})!=metas[1].get('evidence_schema',{}):
        raise ValueError('executed routing/arithmetic observation schemas differ')
    sessions=[];checks=[]
    def observation(left,right,label,rows,media=False):
        if type(left) is not Observation or type(right) is not Observation:
            raise ValueError('actual detached complete observations required')
        if any(type(value.logits) is not np.ndarray or value.logits.dtype.kind!='f'
               or value.logits.shape!=(rows,metas[0]['vocabulary_size'])
               or not np.isfinite(value.logits).all() for value in (left,right)):
            raise ValueError('comparison omitted complete expanded token logits')
        if set(left.cache)!=set(cache_schema) or set(right.cache)!=set(cache_schema):
            raise ValueError('comparison omitted or invented logical cache leaves')
        arrays=[value for item in (left,right) for value in
                (item.logits,*item.cache.values(),*item.evidence.values())]
        if any(type(value) is not np.ndarray or value.dtype.kind not in 'biuf'
               or not np.isfinite(value).all() for value in arrays):
            raise ValueError('finite numeric detached observation arrays required')
        if sum(value.nbytes+48*value.size for value in arrays)>max_bytes:
            raise MemoryError('paired observations and comparator work exceed admission')
        checks.append(dict(label=label+'.logits',**compare(left.logits,right.logits,tolerance)))
        for name in sorted(cache_schema):
            pattern=metas[0].get('empty_cache_shapes',{}).get(name)
            allow_empty=False
            if pattern is not None:
                shape=np.asarray(left.cache[name]).shape
                allow_empty=(len(shape)==len(pattern) and all(p is None or p==size for p,size in zip(pattern,shape)))
            checks.append(dict(label=label+'.cache.'+name,**compare(left.cache[name],right.cache[name],tolerance,allow_empty=allow_empty)))
        required=set(metas[0].get('evidence_schema',{}))
        if media:required.update(f'vision.media.{i}' for i in range(len(case.grids)))
        if set(left.evidence)!=required or set(right.evidence)!=required:
            raise ValueError('complete observed projector/routing/arithmetic outputs required on both sides')
        for name in sorted(required):
            checks.append(dict(label=label+'.'+name,**compare(left.evidence[name],right.evidence[name],tolerance)))
    try:
        for adapter in (candidate,oracle):sessions.append(adapter.new_session(seed=seed))
        left,right=(session.forward_media(case) for session in sessions)
        observation(left,right,'media-prefill',case.expanded_tokens,True)
        tokens=[]
        for step in range(generation_steps):
            ctoken,otoken=int(np.argmax(left.logits[-1])),int(np.argmax(right.logits[-1]))
            checks.append(dict(label=f'greedy.{step}',passed=ctoken==otoken,candidate=ctoken,oracle=otoken))
            snapshots=[session.snapshot() for session in sessions]
            left,right=(session.forward((otoken,)) for session in sessions)
            observation(left,right,f'decode.{step}',1)
            for session,snapshot in zip(sessions,snapshots):session.restore(snapshot)
            restored=[session.forward((otoken,)) for session in sessions]
            observation(restored[0],left,f'candidate-restore.{step}',1)
            observation(restored[1],right,f'oracle-restore.{step}',1)
            tokens.append(otoken)
        if any(digest_file(row['resolved_path'])!=row['sha256'] for row in pins):
            raise ValueError('pinned artifacts changed during media comparison')
        executor=getattr(sessions[0],'executor',None)
        arithmetic=getattr(executor,'_arithmetic',None)
        runtime=dict(execution_identity=getattr(executor,'identity',None),
                     arithmetic_identity=getattr(arithmetic,'identity',None))
        return dict(schema='openagent.k3-media-wholemodel.v2',status='passed' if all(row['passed'] for row in checks) else 'failed',
                    checks=checks,generated_tokens=tokens,pins=pins,fixture_sha256=case.identity(),
                    adapter_metadata=metas,platform_certified=False,candidate_runtime=runtime,
                    fixture=dict(token_ids=list(case.token_ids),media_token_id=case.media_token_id,
                        expanded_tokens=case.expanded_tokens,grids=case.grids.tolist(),
                        pixels_shape=list(case.pixels.shape),pixels_dtype=case.pixels.dtype.str,
                        grids_dtype=case.grids.dtype.str))
    finally:
        primary=sys.exception()
        failures=[]
        for session in reversed(sessions):
            try:session.close()
            except BaseException as error:failures.append(error)
        if failures:
            error=RuntimeError('media comparison resource cleanup remains pending')
            error.pending_sessions=tuple(sessions)
            error.cleanup_failures=tuple(failures)
            if primary is not None:
                primary.pending_sessions=tuple(sessions)
                primary.add_note(str(error)+': '+repr(failures[0]))
            else:raise error from failures[0]


def _read_fixture_array(path,*,max_bytes,dtype_kind,dimensions):
    """Read a bounded, non-pickled NPY fixture without trusting array headers.

    Inspect the header and exact payload length before allocation. Object,
    structured, compressed and Fortran-ordered arrays are outside this contract.
    """
    path=Path(path)
    with path.open('rb') as stream:
        before=path.stat()
        if before.st_size>max_bytes+65536:raise MemoryError('fixture array exceeds admission')
        if stream.read(6)!=b'\x93NUMPY':raise ValueError('plain original NPY fixture required')
        version=stream.read(2)
        if version==b'\x01\x00':width,encoding=2,'latin1'
        elif version in (b'\x02\x00',b'\x03\x00'):width,encoding=4,'utf-8' if version==b'\x03\x00' else 'latin1'
        else:raise ValueError('unsupported fixture NPY version')
        raw=stream.read(width)
        if len(raw)!=width:raise ValueError('truncated fixture header')
        length=struct.unpack('<H' if width==2 else '<I',raw)[0]
        if not 0<length<=65536:raise ValueError('bounded fixture header required')
        raw=stream.read(length)
        if len(raw)!=length:raise ValueError('truncated fixture header')
        header=ast.literal_eval(raw.decode(encoding).strip())
        if type(header) is not dict or set(header)!={'descr','fortran_order','shape'}:
            raise ValueError('exact NPY fixture fields required')
        if type(header['descr']) is not str or header['fortran_order'] is not False:
            raise ValueError('scalar contiguous fixture dtype required')
        dtype=np.dtype(header['descr']);shape=header['shape']
        if (dtype.kind not in dtype_kind or dtype.hasobject or dtype.fields is not None
                or dtype.subdtype is not None or type(shape) is not tuple or len(shape)!=dimensions
                or any(type(size) is not int or size<=0 for size in shape)):
            raise ValueError('positive original fixture geometry and numeric dtype required')
        size=dtype.itemsize
        for extent in shape:
            if extent>max_bytes//size:raise MemoryError('fixture dimensions exceed admission')
            size*=extent
        if size>max_bytes or stream.tell()+size!=before.st_size:
            raise ValueError('fixture payload length differs from declared geometry')
        payload=stream.read(size)
        after=path.stat()
        if len(payload)!=size or stream.read(1) or (before.st_size,before.st_mtime_ns,before.st_ino)!=(
                after.st_size,after.st_mtime_ns,after.st_ino):
            raise ValueError('fixture changed while being captured')
    return np.frombuffer(payload,dtype=dtype).reshape(shape)


def load_media_fixture(*,offline_root,fixture_config,artifact_pins,max_bytes):
    """Load exact pinned processor output; never preprocess or invent pixels.

    The configuration and both array files are pinned as config artifacts.
    The digest additionally binds token order, placeholder ID and expanded
    prefix length. Processor source belongs in the independent source pins.
    """
    if type(max_bytes) is not int or max_bytes<=0:raise ValueError('positive fixture budget required')
    root=Path(offline_root).resolve(strict=True)
    pins=pinned_artifacts({'artifacts':artifact_pins},root)
    def resolve(name):
        if type(name) is not str or not name:raise ValueError('explicit relative fixture path required')
        path=(root/name).resolve(strict=True)
        if not path.is_relative_to(root) or not any(
                row['kind']=='config' and Path(row['resolved_path'])==path for row in pins):
            raise ValueError('every fixture file must be an admitted config artifact')
        return path
    config=resolve(fixture_config)
    if config.stat().st_size>65536:raise ValueError('bounded fixture configuration required')
    from ..native_checkpoint import _pairs
    data=json.loads(config.read_text(encoding='utf-8'),object_pairs_hook=_pairs)
    fields={'schema','token_ids','media_token_id','expanded_tokens','pixels','grids','media_fixture_sha256'}
    if type(data) is not dict or set(data)!=fields or data['schema']!='openagent.k3-media-fixture.v1':
        raise ValueError('exact versioned media fixture configuration required')
    if type(data['token_ids']) is not list:raise ValueError('ordered fixture tokens required')
    # Reserve three copies of both processor buffers, plus Python token storage.
    available=max_bytes-len(data['token_ids'])*128
    if available<=0:raise MemoryError('fixture token inventory exceeds admission')
    pixels=_read_fixture_array(resolve(data['pixels']),max_bytes=available//3,dtype_kind='f',dimensions=4)
    available-=pixels.nbytes*3
    if available<=0:raise MemoryError('fixture patches exhausted admission')
    grids=_read_fixture_array(resolve(data['grids']),max_bytes=available//3,dtype_kind='iu',dimensions=2)
    case=K3MediaCase(tuple(data['token_ids']),data['media_token_id'],pixels,grids,data['expanded_tokens'])
    case.admit(max_bytes)
    if case.identity()!=data['media_fixture_sha256']:raise ValueError('pinned processor fixture identity differs')
    if any(digest_file(row['resolved_path'])!=row['sha256'] for row in pins):
        raise ValueError('fixture artifacts changed during capture')
    return case


def create_k3_media_candidate_adapter(*,offline_root,options):
    """Use the actual publisher handoff, with an observed complete cache reader."""
    from .k3 import K3CandidateFactory,K3ExecutionCacheReader,k3_metadata,observation_options
    from .adapters import reader_from_options
    import inspect
    root=Path(offline_root).resolve(strict=True)
    pins=pinned_artifacts({'artifacts':options['_artifact_pins']},root)
    source=Path(inspect.getsourcefile(K3MediaSession)).resolve(strict=True)
    if not any(row['kind']=='source' and Path(row['resolved_path'])==source for row in pins):
        raise ValueError('actual complete media observation adapter source must be pinned')
    factory=K3CandidateFactory(root,options)
    if factory.config.vision is None:raise ValueError('complete publisher vision configuration required')
    fla=bool({'fla_policy','metal_policy'} & options.keys())
    metadata=k3_metadata(options['metadata'],factory.config,fla=fla)
    observations=observation_options(options,factory.config,metadata)
    if metadata.get('authoritative') is True:raise ValueError('candidate cannot claim authoritative oracle identity')
    reader=reader_from_options(options)
    if reader is None:
        reader=K3ExecutionCacheReader(factory.config,metadata['cache_schema'],fla=fla,evidence_schema=metadata.get('evidence_schema'))
    return K3MediaAdapter(factory,metadata,reader,max_bytes=options['max_observation_bytes'],observations=observations)


def run_media_configuration(configuration,*,offline_root):
    """Explicit phase-two entry point; this module never invokes it on import.

    Both adapter factories are supplied in the pinned configuration. No vendor
    oracle, tolerance, revision or processor input is chosen by this entry point.
    """
    from .runner import load_adapter
    if type(configuration) is not dict:raise ValueError('explicit complete media comparison configuration required')
    fields={'schema','artifacts','candidate','oracle','fixture_config','cache_schema',
            'max_observation_bytes','seed','generation_steps','tolerance'}
    if set(configuration)!=fields or configuration['schema']!='openagent.k3-media-comparison.v1':
        raise ValueError('exact versioned complete media comparison configuration required')
    pins=configuration['artifacts']
    pinned_artifacts({'artifacts':pins},offline_root)
    maximum=configuration['max_observation_bytes']
    case=load_media_fixture(offline_root=offline_root,fixture_config=configuration['fixture_config'],
        artifact_pins=pins,max_bytes=maximum)
    adapters=[]
    for role in ('candidate','oracle'):
        adapter=configuration[role]
        if type(adapter) is not dict or set(adapter)!={'factory','options'}:
            raise ValueError('explicit pinned factory and options required for '+role)
        spec=dict(adapter,artifacts=pins)
        adapters.append(load_adapter(spec,offline_root))
    policy=configuration['tolerance']
    if type(policy) is not dict or set(policy)!={'atol','rtol'}:
        raise ValueError('explicit media comparison tolerance required')
    return run_media_comparison(*adapters,case,offline_root=offline_root,artifact_pins=pins,
        cache_schema=configuration['cache_schema'],max_bytes=maximum,seed=configuration['seed'],
        generation_steps=configuration['generation_steps'],tolerance=Tolerance(**policy),
        fixture_config=configuration['fixture_config'])


__all__=['K3MediaCase','K3MediaSession','K3MediaAdapter','run_media_comparison','load_media_fixture',
         'create_k3_media_candidate_adapter','run_media_configuration']
