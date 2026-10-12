"""Opt-in Qwen3.5 media observations from the complete executed wrapper.

Fixtures contain the publisher processor's expanded token IDs and original
patch/grid buffers. No oracle, processor output or comparison evidence is
created automatically. Callers pin their processor, model and oracle artifacts.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import copy
import hashlib
import numpy as np

from .cache_readers import CacheContext,detached
from .contracts import Observation
from .qwen35_publisher_caches import Qwen35PublisherCacheReader
from .adapters import FreshOwner


@dataclass(frozen=True)
class Qwen35MediaCase:
    fixture_schema='openagent.qwen35-media-fixture.v1'
    token_ids: tuple
    pixel_values: object=None
    image_grid_thw: object=None
    pixel_values_videos: object=None
    video_grid_thw: object=None

    def freeze(self,*,max_bytes):
        if type(max_bytes) is not int or max_bytes<=0:
            raise ValueError('positive complete media fixture admission required')
        if (type(self.token_ids) is not tuple or not self.token_ids or
                any(type(token) is not int or token<0 for token in self.token_ids)):
            raise ValueError('original expanded integer processor token IDs required')
        buffers=(self.pixel_values,self.image_grid_thw,self.pixel_values_videos,self.video_grid_thw)
        if all(value is None for value in buffers):raise ValueError('occupied image or video fixture required')
        size=128*len(self.token_ids)
        for pixels,grids in (buffers[:2],buffers[2:]):
            if (pixels is None)!=(grids is None):raise ValueError('paired original patches and grids required')
            if pixels is None:continue
            if (type(pixels) is not np.ndarray or pixels.dtype!=np.float32 or pixels.ndim!=2 or
                    not all(pixels.shape) or type(grids) is not np.ndarray or grids.dtype.kind not in 'iu' or
                    grids.ndim!=2 or grids.shape[1]!=3 or not len(grids)):
                raise ValueError('original FP32 patch matrix and integer temporal/spatial grids required')
            size+=3*(pixels.nbytes+grids.nbytes)
        if size>max_bytes:raise MemoryError('complete processor fixture exceeds admission')
        for pixels,grids in (buffers[:2],buffers[2:]):
            if pixels is None:continue
            if not np.isfinite(pixels).all() or np.any(grids<=0):
                raise ValueError('finite original patches and positive grids required')
            if sum(int(t)*int(h)*int(w) for t,h,w in grids)!=len(pixels):
                raise ValueError('original patch rows differ from complete grids')
        original=self.identity()
        copied=[None if value is None else np.frombuffer(value.tobytes(order='C'),dtype=value.dtype).reshape(value.shape)
                for value in buffers]
        result=type(self)(self.token_ids,*copied)
        if original!=self.identity() or original!=result.identity():
            raise ValueError('processor fixture changed during immutable capture')
        return result

    def identity(self):
        digest=hashlib.sha256((self.fixture_schema+repr(self.token_ids)).encode())
        for value in (self.pixel_values,self.image_grid_thw,self.pixel_values_videos,self.video_grid_thw):
            if value is None:digest.update(b'absent');continue
            digest.update(repr((value.shape,value.dtype.str)).encode())
            digest.update(memoryview(np.ascontiguousarray(value)).cast('B'))
        return digest.hexdigest()


class Qwen35MediaSession:
    reader_class=Qwen35PublisherCacheReader

    def _wrapper_class(self):
        from ..qwen35_multimodal import Qwen35MultimodalDecoder
        return Qwen35MultimodalDecoder

    def _extra_cache_elements(self,position):return 0

    def __init__(self,executor,cache_reader,*,max_bytes,capture_routing=False):
        if (type(executor) is not self._wrapper_class() or
                type(cache_reader) is not self.reader_class or
                cache_reader.role!='candidate' or not cache_reader.media):
            raise ValueError('complete Qwen3.5 wrapper and media physical-cache reader required')
        if type(max_bytes) is not int or max_bytes<=0 or type(capture_routing) is not bool:
            raise ValueError('bounded observations and explicit routing policy required')
        if vars(cache_reader.config)!=vars(executor.text.config):
            raise ValueError('media cache reader must bind the executed text configuration exactly')
        if capture_routing and (not executor.routing_observation_supported or not hasattr(cache_reader,'evidence_schema')):
            raise ValueError('executed routing support and complete evidence schema required')
        self.executor=executor;self.reader=cache_reader;self.max_bytes=max_bytes
        self.capture_routing=capture_routing;self.closed=self.failed=False

    @property
    def position(self):return self.executor.position

    def _active(self):
        if self.closed or self.failed:raise RuntimeError('closed or quarantined media observation owner')

    def _preflight(self,rows,fixture_bytes=0):
        self._active();c=self.reader.config;position=self.position+rows
        capacity=getattr(self.executor.text,'capacity',getattr(self.executor.text,'max_tokens',c.max_position_embeddings))
        if not 0<rows or position>capacity:raise ValueError('occupied bounded media observation required')
        width=2*c.linear_num_key_heads*c.linear_key_head_dim+c.linear_num_value_heads*c.linear_value_head_dim
        elements=sum(width*c.linear_conv_kernel_dim+c.linear_num_value_heads*c.linear_key_head_dim*c.linear_value_head_dim
            if kind=='linear_attention' else 2*c.num_key_value_heads*position*c.head_dim
            for kind in c.layer_types)
        elements+=self._extra_cache_elements(position)
        elements+=rows*(c.vocab_size+c.hidden_size)
        if self.capture_routing:elements+=2*rows*c.num_hidden_layers*c.num_experts
        if fixture_bytes+128*elements>self.max_bytes:
            raise MemoryError('complete media logits, physical caches and observation work exceed admission')

    def _observe(self,output,rows,evidence):
        logits=output if isinstance(output,np.ndarray) else output.get('logits') if isinstance(output,Mapping) else getattr(output,'logits',None)
        if logits is None or tuple(logits.shape)!=(1,rows,self.reader.config.vocab_size):
            raise ValueError('complete expanded token-to-logit observation required')
        context=CacheContext(None,self.executor,output,self.position,rows)
        cache=self.reader(context)
        if self.capture_routing:evidence.update(self.reader.evidence(context))
        values=[logits,*cache.values(),*evidence.values()]
        if sum(int(np.prod(value.shape,dtype=object))*64 for value in values)>self.max_bytes:
            raise MemoryError('detached media observation exceeds admission')
        result=Observation(detached(logits[0]),cache,{name:detached(value) for name,value in evidence.items()})
        if any(not np.isfinite(value).all() for value in (result.logits,*result.cache.values(),*result.evidence.values())):
            raise ValueError('nonfinite executed media observation')
        return result

    def _transaction(self,operation):
        self._active();before=self.executor.snapshot();primary=None
        try:return operation()
        except BaseException as failure:
            primary=failure
            try:self.executor.restore(before)
            except BaseException as rollback:
                self.failed=True;primary.add_note('media observation rollback failed: '+repr(rollback))
            raise
        finally:
            try:self.executor.release_snapshot(before)
            except BaseException as cleanup:
                self.failed=True
                if primary is None:raise
                primary.add_note('media observation snapshot release failed: '+repr(cleanup))

    def forward_media(self,case):
        self._active()
        if not isinstance(case,Qwen35MediaCase) or self.position:
            raise ValueError('fresh independent session and original Qwen3.5 media fixture required')
        case=case.freeze(max_bytes=self.max_bytes)
        self._preflight(len(case.token_ids),3*sum(value.nbytes for value in
            (case.pixel_values,case.image_grid_thw,case.pixel_values_videos,case.video_grid_thw) if value is not None))
        def execute():
            evidence={};vision=self.executor.vision
            calls=[(name,len(grids)) for name,pixels,grids in
                (('image',case.pixel_values,case.image_grid_thw),('video',case.pixel_values_videos,case.video_grid_thw))
                if pixels is not None]
            observed=0;used=0;original=vision.forward
            had_override='forward' in vars(vision);override=vars(vision).get('forward')
            def capture(*args,**kwargs):
                nonlocal observed,used
                if observed>=len(calls):raise ValueError('unexpected executed media projector call')
                name,count=calls[observed];values=original(*args,**kwargs)
                if not isinstance(values,(tuple,list)) or len(values)!=count:
                    raise ValueError('complete executed per-grid projector outputs required')
                for index,value in enumerate(values):
                    if not isinstance(value,np.ndarray) or value.ndim!=2 or value.dtype.kind!='f':
                        raise ValueError('actual floating projector matrix required')
                    used+=64*value.size
                    if used>self.max_bytes:raise MemoryError('executed projector capture exceeds admission')
                    evidence[f'vision.{name}.{index}']=detached(value)
                observed+=1
                return values
            # Observe the exact results inserted by the wrapper, rather than
            # performing a separate vision execution for a comparison trace.
            with self.executor._lock:
                vision.forward=capture
                try:
                    output=self.executor.forward(case.token_ids,pixel_values=case.pixel_values,image_grid_thw=case.image_grid_thw,
                        pixel_values_videos=case.pixel_values_videos,video_grid_thw=case.video_grid_thw,capture_routing=self.capture_routing)
                finally:
                    if had_override:vision.forward=override
                    else:del vision.forward
            if observed!=len(calls):raise ValueError('missing executed media projector observations')
            return self._observe(output,len(case.token_ids),evidence)
        return self._transaction(execute)

    def forward(self,tokens):
        ids=tuple(tokens)
        if not self.position or not ids or any(type(token) is not int or not 0<=token<self.reader.config.vocab_size for token in ids):
            raise ValueError('in-vocabulary continuation after actual media prefill required')
        self._preflight(len(ids))
        return self._transaction(lambda:self._observe(self.executor.forward(ids,capture_routing=self.capture_routing),len(ids),{}))

    def snapshot(self):self._active();return self.executor.snapshot()
    def restore(self,snapshot):self._active();self.executor.restore(snapshot)
    def release_snapshot(self,snapshot):self._active();self.executor.release_snapshot(snapshot)
    def close(self):
        if not self.closed:self.executor.close();self.closed=True


class Qwen35MediaAdapter(FreshOwner):
    """Caller-pinned fresh-wrapper factory for later media comparisons."""
    reader_class=Qwen35PublisherCacheReader
    session_class=Qwen35MediaSession
    def __init__(self,factory,metadata,cache_reader,*,max_bytes,capture_routing=False):
        super().__init__()
        if not callable(factory) or type(cache_reader) is not self.reader_class or not cache_reader.media:
            raise ValueError('fresh wrapper factory and complete media schema required')
        if (cache_reader.role!='candidate' or type(max_bytes) is not int or max_bytes<=0 or
                type(capture_routing) is not bool):
            raise ValueError('candidate reader, positive admission and explicit capture policy required')
        if (not isinstance(metadata,Mapping) or metadata.get('cache_schema')!=cache_reader.schema or
                metadata.get('cache_projection') is not None):
            raise ValueError('complete physical media cache metadata without projection required')
        self.factory=factory;self.reader=cache_reader;self.max_bytes=max_bytes
        self.capture_routing=capture_routing;self._metadata=copy.deepcopy(dict(metadata))
        self._metadata['execution_scope']='processor-media-expanded-prefix'

    def metadata(self):return copy.deepcopy(self._metadata)

    def new_session(self,*,seed):
        if type(seed) is not int or seed<0:raise ValueError('nonnegative integer media session seed required')
        executor=self.factory(seed=seed)
        # A reused live owner still belongs to its first session. Reject it
        # before entering cleanup for a newly constructed session.
        self.claim(executor)
        try:
            if executor.position:raise ValueError('independent empty media wrapper required')
            return self.session_class(executor,self.reader,max_bytes=self.max_bytes,capture_routing=self.capture_routing)
        except BaseException as primary:
            try:executor.close()
            except BaseException as cleanup:
                primary.add_note('media factory cleanup retained: '+repr(cleanup));primary.cleanup=executor.close
            raise


def run_media_comparison(candidate,oracle,case,*,offline_root,artifact_pins,max_bytes,
                         seed,generation_steps,tolerance,fixture_config):
    """Observe full prefill, continuation and both owned restore trajectories.

    The unchanged oracle supplies the same explicit session interface. Actual
    pins and observations are mandatory; this report grants no certification.
    """
    from .comparison import Tolerance,compare
    from .runner import pinned_artifacts,metadata
    if (not isinstance(case,Qwen35MediaCase) or type(tolerance) is not Tolerance or
            type(seed) is not int or seed<0 or type(generation_steps) is not int or not 1<=generation_steps<=4096):
        raise ValueError('original media fixture, explicit tolerance and bounded generation trajectory required')
    case=case.freeze(max_bytes=max_bytes)
    specification={'artifacts':artifact_pins}
    pins=pinned_artifacts(specification,offline_root)
    if not pins:raise ValueError('explicit original processor, model and implementation artifact pins required')
    from pathlib import Path
    import json
    from ..segment_snapshot import _pairs
    root=Path(offline_root).resolve(strict=True);fixture=(root/fixture_config).resolve(strict=True)
    if (not fixture.is_relative_to(root) or not fixture.is_file() or fixture.stat().st_size>65536 or
            not any(Path(row['resolved_path'])==fixture and row['kind']=='config' for row in pins)):
        raise ValueError('bounded explicitly pinned original processor fixture configuration required')
    with fixture.open('rb') as stream:encoded=stream.read(65537)
    if len(encoded)>65536:raise ValueError('processor fixture configuration exceeds admission')
    declaration=json.loads(encoded,object_pairs_hook=_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if type(declaration) is not dict or declaration.get('media_fixture_sha256')!=case.identity():
        raise ValueError('actual original processor fixture differs from its artifact declaration')
    metas=[metadata(candidate,specification),metadata(oracle,specification,oracle=True)]
    if candidate is oracle or ((metas[0]['implementation'],metas[0]['implementation_revision'])==
                              (metas[1]['implementation'],metas[1]['implementation_revision'])):
        raise ValueError('independent candidate and unchanged oracle implementations required')
    for key in ('model_type','config_sha256','weights_sha256','tokenizer_sha256','num_layers',
                'vocabulary_size','cache_schema','empty_cache_shapes','evidence_schema'):
        if metas[0].get(key)!=metas[1].get(key):raise ValueError('candidate/oracle media contract differs: '+key)
    if any(meta.get('execution_scope')!='processor-media-expanded-prefix' or
           meta.get('cache_projection') is not None or 'rope_delta' not in meta['cache_schema'] for meta in metas):
        raise ValueError('complete physical media continuation schemas required')
    vocabulary=metas[0]['vocabulary_size']
    if any(token>=vocabulary for token in case.token_ids):raise ValueError('processor token outside shared vocabulary')
    fixture_bytes=3*sum(value.nbytes for value in
        (case.pixel_values,case.image_grid_thw,case.pixel_values_videos,case.video_grid_thw) if value is not None)
    fixture_bytes+=128*len(case.token_ids)
    checks=[];sessions=[];snapshots=[];prefix_snapshots=[];primary=None
    receipt_bytes=0;observation_work_bytes=0;retained_bytes=0
    def record(value):
        nonlocal receipt_bytes
        from .runner import safe_json
        size=len(json.dumps(safe_json(value),allow_nan=False,separators=(',',':')).encode())+256
        if size>max_bytes-receipt_bytes-fixture_bytes-observation_work_bytes-retained_bytes:
            raise MemoryError('media comparison receipt inventory exceeds admission')
        receipt_bytes+=size;checks.append(value)
    def observe(left,right,label,rows):
        nonlocal observation_work_bytes
        if any(type(item) is not Observation for item in (left,right)):
            raise ValueError('actual complete media observations required')
        for item in (left,right):
            if item.logits.shape!=(rows,vocabulary) or set(item.cache)!=set(metas[0]['cache_schema']):
                raise ValueError('complete expanded logits and physical cache leaves required')
            expected=set(metas[0].get('evidence_schema',{}))
            if label=='media-prefill':
                expected.update(f'vision.{name}.{index}' for name,grids in
                    (('image',case.image_grid_thw),('video',case.video_grid_thw)) if grids is not None
                    for index in range(len(grids)))
            if expected!=set(item.evidence):raise ValueError('complete executed projector/routing evidence required')
        if set(left.evidence)!=set(right.evidence):raise ValueError('media projector/routing inventories differ')
        arrays=[value for item in (left,right) for value in (item.logits,*item.cache.values(),*item.evidence.values())]
        if any(type(value) is not np.ndarray or value.dtype.kind not in 'biuf' or not np.isfinite(value).all() for value in arrays):
            raise ValueError('finite numeric complete observed media buffers required')
        if left.logits.dtype.kind!='f' or right.logits.dtype.kind!='f':
            raise ValueError('full floating token logits required')
        observation_work_bytes=sum(value.nbytes+48*value.size for value in arrays)
        if observation_work_bytes+receipt_bytes+fixture_bytes+retained_bytes>max_bytes:
            raise MemoryError('paired observations and comparator work exceed admission')
        record(dict(label=label+'.logits',**compare(left.logits,right.logits,tolerance)))
        for group in ('cache','evidence'):
            for name in sorted(getattr(left,group)):
                lvalue=getattr(left,group)[name];rvalue=getattr(right,group)[name]
                # Empty physical histories require a separately declared exact
                # structural shape; comparator admission remains fail-closed.
                from .cache_readers import declared_empty
                pattern=metas[0].get('empty_'+group+'_shapes',{}).get(name)
                empty=declared_empty(lvalue,pattern) and declared_empty(rvalue,pattern)
                record(dict(label=label+'.'+group+'.'+name,
                    **compare(lvalue,rvalue,tolerance,allow_empty=empty)))
    try:
        for adapter in (candidate,oracle):sessions.append(adapter.new_session(seed=seed))
        left,right=[session.forward_media(case) for session in sessions]
        observe(left,right,'media-prefill',len(case.token_ids))
        for session in sessions:prefix_snapshots.append(session.snapshot())
        continuation=[]
        for step in range(generation_steps):
            token=int(np.argmax(left.logits[-1]));oracle_token=int(np.argmax(right.logits[-1]))
            record(dict(label=f'decode.{step}.token',passed=token==oracle_token,
                candidate_token=token,oracle_token=oracle_token))
            # Both consume the same candidate token to compare like-for-like
            # physical trajectories even when a token decision differs.
            left,right=[session.forward((token,)) for session in sessions]
            continuation.append(token)
            observe(left,right,f'decode.{step}',1)
        for session in sessions:snapshots.append(session.snapshot())
        token=int(np.argmax(left.logits[-1]))
        expected=[session.forward((token,)) for session in sessions]
        observe(expected[0],expected[1],'restore-boundary',1)
        for index,session in enumerate(sessions):
            session.restore(snapshots[index]);actual=session.forward((token,))
            observe(actual,expected[index],('candidate' if index==0 else 'oracle')+'-restore',1)
        # Re-execute the same observed continuation from the occupied media
        # prefix in full and in bounded chunks. Both are actual owner calls;
        # no media state or routing decision is reconstructed from tokens.
        left=right=expected=actual=None
        for index,session in enumerate(sessions):
            session.restore(prefix_snapshots[index])
            full=session.forward(tuple(continuation))
            retained_bytes=sum(value.nbytes for value in (full.logits,*full.cache.values(),*full.evidence.values()))
            if retained_bytes+fixture_bytes+receipt_bytes>max_bytes:
                raise MemoryError('full media continuation observation exceeds admission')
            for chunk_size in sorted({1,max(1,len(continuation)//2)}):
                session.restore(prefix_snapshots[index]);logits=[];evidence={};chunk=None
                chunk_bytes=0
                for start in range(0,len(continuation),chunk_size):
                    chunk=session.forward(tuple(continuation[start:start+chunk_size]))
                    if type(chunk) is not Observation or set(chunk.evidence)!=set(full.evidence):
                        raise ValueError('complete chunked continuation observation required')
                    values=(chunk.logits,*chunk.evidence.values())
                    chunk_bytes+=sum(value.nbytes for value in values)
                    if retained_bytes+3*chunk_bytes+fixture_bytes+receipt_bytes+sum(value.nbytes for value in chunk.cache.values())>max_bytes:
                        raise MemoryError('chunked media continuation assembly exceeds admission')
                    logits.append(chunk.logits)
                    for name,value in chunk.evidence.items():evidence.setdefault(name,[]).append(value)
                joined=Observation(np.concatenate(logits,axis=0),chunk.cache,
                    {name:np.concatenate(values,axis=1) for name,values in evidence.items()})
                retained_bytes+=chunk_bytes
                observe(joined,full,('candidate' if index==0 else 'oracle')+f'-chunked.{chunk_size}',len(continuation))
                retained_bytes-=chunk_bytes
                joined=chunk=None;logits.clear();evidence.clear()
            full=None;retained_bytes=0
        if pins!=pinned_artifacts(specification,offline_root):raise ValueError('media artifacts changed during observations')
        return dict(schema=case.fixture_schema.replace('-fixture.','-comparison.'),fixture_sha256=case.identity(),
            metadata=metas,artifacts=pins,checks=checks,generation_steps=generation_steps,
            passed=all(row['passed'] for row in checks),numerical_parity_verified=False,platform_certified=False)
    except BaseException as failure:
        primary=failure;raise
    finally:
        cleanup=[]
        for index,session in enumerate(sessions):
            if index<len(prefix_snapshots):
                try:session.release_snapshot(prefix_snapshots[index])
                except BaseException as failure:cleanup.append(failure)
            if index<len(snapshots):
                try:session.release_snapshot(snapshots[index])
                except BaseException as failure:cleanup.append(failure)
            try:session.close()
            except BaseException as failure:cleanup.append(failure)
        if cleanup:
            if primary is None:raise RuntimeError('media comparison owner cleanup failed') from cleanup[0]
            for failure in cleanup:primary.add_note('media comparison cleanup failed: '+repr(failure))


def load_media_fixture(*,offline_root,fixture_config,artifact_pins,max_bytes,case_class=Qwen35MediaCase):
    from pathlib import Path
    import json
    from ..segment_snapshot import _pairs
    from .runner import pinned_artifacts
    from .k3_media import _read_fixture_array
    if type(max_bytes) is not int or max_bytes<=0:raise ValueError('positive complete fixture admission required')
    root=Path(offline_root).resolve(strict=True);pins=pinned_artifacts({'artifacts':artifact_pins},root)
    def resolve(name):
        if type(name) is not str or not name:raise ValueError('explicit original fixture path required')
        path=(root/name).resolve(strict=True)
        if not path.is_relative_to(root) or not any(row['kind']=='config' and Path(row['resolved_path'])==path for row in pins):
            raise ValueError('every original fixture buffer must be pinned as a config artifact')
        return path
    path=resolve(fixture_config)
    with path.open('rb') as stream:encoded=stream.read(65537)
    if len(encoded)>65536:raise ValueError('bounded original fixture declaration required')
    value=json.loads(encoded,object_pairs_hook=_pairs,
        parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))
    fields={'schema','token_ids','pixel_values','image_grid_thw','pixel_values_videos','video_grid_thw','media_fixture_sha256'}
    if type(value) is not dict or set(value)!=fields or value['schema']!=case_class.fixture_schema:
        raise ValueError('exact original Qwen3.5 processor fixture declaration required')
    if type(value['token_ids']) is not list:raise ValueError('ordered expanded processor token list required')
    available=max_bytes-128*len(value['token_ids']);buffers=[]
    for name in ('pixel_values','image_grid_thw','pixel_values_videos','video_grid_thw'):
        if value[name] is None:buffers.append(None);continue
        if available<3:raise MemoryError('original fixture buffers exceed aggregate admission')
        array=_read_fixture_array(resolve(value[name]),max_bytes=available//3,
            dtype_kind='iu' if 'grid' in name else 'f',dimensions=2)
        available-=3*array.nbytes;buffers.append(array)
    result=case_class(tuple(value['token_ids']),*buffers).freeze(max_bytes=max_bytes)
    if result.identity()!=value['media_fixture_sha256']:raise ValueError('original fixture differs from declared content identity')
    if pins!=pinned_artifacts({'artifacts':artifact_pins},root):raise ValueError('fixture artifacts changed during capture')
    return result


def run_media_configuration(configuration,*,offline_root):
    from .runner import load_adapter,pinned_artifacts
    from .comparison import Tolerance
    fields={'schema','artifacts','candidate','oracle','fixture_config','max_observation_bytes','seed','generation_steps','tolerance'}
    if (type(configuration) is not dict or set(configuration)!=fields or
            configuration['schema']!='openagent.qwen35-media-comparison-configuration.v1'):
        raise ValueError('exact pinned Qwen3.5 media comparison configuration required')
    pins=configuration['artifacts'];pinned_artifacts({'artifacts':pins},offline_root)
    case=load_media_fixture(offline_root=offline_root,fixture_config=configuration['fixture_config'],
        artifact_pins=pins,max_bytes=configuration['max_observation_bytes'])
    policy=configuration['tolerance']
    if type(policy) is not dict or set(policy)!={'atol','rtol'}:raise ValueError('explicit comparison tolerances required')
    adapters=[]
    for role in ('candidate','oracle'):
        spec=configuration[role]
        if type(spec) is not dict or set(spec)!={'factory','options'}:
            raise ValueError('explicit caller-pinned media adapter factory and options required')
        adapters.append(load_adapter(dict(spec,artifacts=pins),offline_root))
    return run_media_comparison(*adapters,case,offline_root=offline_root,artifact_pins=pins,
        max_bytes=configuration['max_observation_bytes'],seed=configuration['seed'],
        generation_steps=configuration['generation_steps'],tolerance=Tolerance(**policy),
        fixture_config=configuration['fixture_config'])


def create_candidate(*,offline_root,options):
    """Bind an explicit complete-wrapper loader to real media observations."""
    from pathlib import Path
    import inspect
    from .runner import pinned_artifacts
    from .adapters import resolve_callable
    root=Path(offline_root).resolve(strict=True)
    pins=pinned_artifacts({'artifacts':options['_artifact_pins']},root)
    loader=resolve_callable(options['decoder_factory'])
    loader_source=inspect.getsourcefile(loader)
    if loader_source is None:raise ValueError('inspectable original wrapper loader source required')
    for source in (Path(__file__).resolve(strict=True),Path(loader_source).resolve(strict=True)):
        if not any(row['kind']=='source' and Path(row['resolved_path'])==source for row in pins):
            raise ValueError('actual media session and wrapper loader implementation sources must be pinned')
    meta=copy.deepcopy(options['metadata'])
    if meta.get('authoritative') is True:raise ValueError('candidate cannot declare authoritative oracle identity')
    reader=Qwen35PublisherCacheReader(family='qwen3_5_moe_text',config=options['config'],role='candidate',
        max_export_bytes=options['max_cache_export_bytes'],media=True)
    routing=bool(meta.get('evidence_schema'))
    if routing and reader.enable_routing_observations(max_export_bytes=options['max_routing_export_bytes'])!=meta['evidence_schema']:
        raise ValueError('complete executed Qwen3.5 routing schema required')
    if meta.get('num_layers')!=reader.config.num_hidden_layers or meta.get('vocabulary_size')!=reader.config.vocab_size:
        raise ValueError('candidate media metadata differs from executed model geometry')
    decoder_options=copy.deepcopy(dict(options.get('decoder_options',{})))
    def fresh(*,seed):
        return loader(seed=seed,offline_root=str(root),options=copy.deepcopy(decoder_options))
    return Qwen35MediaAdapter(fresh,meta,reader,max_bytes=options['max_observation_bytes'],capture_routing=routing)


__all__=['Qwen35MediaCase','Qwen35MediaSession','Qwen35MediaAdapter','run_media_comparison',
         'load_media_fixture','run_media_configuration','create_candidate']
