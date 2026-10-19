"""Fail-closed admission from an explicitly pinned later-phase comparison.

An evidence receipt is an author's bounded comparison claim, not a universal
numerical or platform certificate. No evidence is manufactured by execution.
"""
from pathlib import Path
from types import MappingProxyType
import hashlib
import hmac
import json
import re
import os

from ..session_recovery import _path

def _tensor_observation(actual,expected,*,name,empty_shapes):
    """Admit canonical observed tensor descriptors, including structural zeros."""
    fields={'shape','dtype','sha256'}
    if type(actual) is not dict or type(expected) is not dict or set(actual)!=fields or set(expected)!=fields or actual!=expected:
        raise ValueError('identical complete canonical tensor descriptors required')
    shape=actual['shape'];dtype=actual['dtype'];pin=actual['sha256']
    if (type(shape) is not list or len(shape)>8 or any(type(size) is not int or not 0<=size<2**63 for size in shape)
            or type(dtype) is not str or re.fullmatch(r'(?:\|(?:[iu]1|b1)|[<>](?:[iu][248]|f[248]))',dtype) is None
            or type(pin) is not str or re.fullmatch('[0-9a-f]{64}',pin) is None):
        raise ValueError('bounded numeric canonical observation geometry required')
    if 0 in shape:
        leaf=name.split('.cache.',1)[1] if '.cache.' in name else None
        pattern=empty_shapes.get(leaf) if type(empty_shapes) is dict else None
        if (type(pattern) is not list or len(pattern)!=len(shape) or 0 not in pattern or
                any(value is not None and (type(value) is not int or value!=size) for value,size in zip(pattern,shape))):
            raise ValueError('empty observation requires an explicit structural physical-cache declaration')

def _observed_coverage(value, report):
    """Bind admission to exact recorded comparisons, never name substrings."""
    groups=value['coverage_checks']
    if type(groups) is not dict or set(groups)!=set(value['coverage']):
        raise ValueError('each exact coverage requirement needs named comparison bindings')
    models=report.get('models',{})
    if any(models.get(role,{}).get('cache_projection') is not None for role in ('candidate','oracle')):
        raise ValueError('exact admission requires complete physical caches without comparison projections')
    empty_shapes=models.get('candidate',{}).get('empty_cache_shapes',{})
    if empty_shapes!=models.get('oracle',{}).get('empty_cache_shapes',{}):
        raise ValueError('identical structural empty-cache declarations required')
    indexed={}
    for check in report['checks']:
        name=check.get('name')
        if type(name) is not str or not name or name in indexed:
            raise ValueError('unique named exact comparison records required')
        if check.get('projected') is True or check.get('numerically_certified') is False:
            raise ValueError('projected comparisons cannot admit exact mode')
        actual,expected=check.get('actual'),check.get('expected')
        if isinstance(actual,dict) or isinstance(expected,dict):
            if type(actual) is not dict or type(expected) is not dict:
                raise ValueError('complete exact tensor identities required')
            _tensor_observation(actual,expected,name=name,empty_shapes=empty_shapes)
            if (actual.get('shape')!=expected.get('shape') or actual.get('dtype')!=expected.get('dtype') or
                    actual.get('sha256')!=expected.get('sha256') or
                    type(actual.get('sha256')) is not str or
                    re.fullmatch('[0-9a-f]{64}',actual['sha256']) is None or
                    check.get('mismatches')!=0):
                raise ValueError('exact admission requires identical complete tensor observations')
            if check.get('exact') is not True and (check.get('atol')!=0 or check.get('rtol')!=0 or check.get('max_absolute')!=0):
                raise ValueError('exact admission requires zero-error numerical records')
        elif type(actual) is not int or type(expected) is not int or actual!=expected:
            raise ValueError('exact token observations must match integer values')
        indexed[name]=check
    selected={}
    for kind,names in groups.items():
        if (type(names) is not list or not names or len(names)>len(indexed) or
                any(type(name) is not str or name not in indexed for name in names) or
                len(set(names))!=len(names)):
            raise ValueError('coverage binds missing, duplicate or empty comparison records')
        selected[kind]=set(names)
    if any(not name.endswith('.logits') for name in selected['whole-model-logits']):
        raise ValueError('whole-model coverage must bind token logits')
    schemas=[report.get('models',{}).get(role,{}).get('cache_schema') for role in ('oracle','candidate')]
    if any(type(schema) is not dict or not schema for schema in schemas) or schemas[0]!=schemas[1]:
        raise ValueError('complete identical declared cache schemas required for exact mode')
    cache_names=selected['complete-cache']
    prefixes={name.split('.cache.',1)[0] for name in cache_names if '.cache.' in name}
    if not prefixes or any('.cache.' not in name for name in cache_names):
        raise ValueError('complete-cache coverage must bind physical cache comparisons')
    for prefix in prefixes:
        if {prefix+'.cache.'+leaf for leaf in schemas[0]}-cache_names:
            raise ValueError('exact cache coverage omits declared physical leaves')
    if any('.cross_evidence.' not in name or 'routing' not in name.split('.cross_evidence.',1)[1]
           for name in selected['routing']):
        raise ValueError('routing coverage requires cross-model route observations')
    if any(re.fullmatch(r'generation\.\d+\.token',name) is None for name in selected['generated-tokens']):
        raise ValueError('generation coverage requires executed token comparisons')
    if not any(name.startswith('schedule.') for name in selected['prefill-and-decode']) or not any(name.startswith('generation.') for name in selected['prefill-and-decode']):
        raise ValueError('prefill and incremental decode observations are both required')
    if any(not name.startswith('continuation.restore.') for name in selected['continuation-restore']):
        raise ValueError('restored continuation comparisons required')
    if any(not name.startswith('quantization.boundary.') for name in selected['quantization-boundaries']):
        raise ValueError('executed quantization boundary comparisons required')
    manifest=report.get('manifest',{})
    tokens=manifest.get('tokens') if type(manifest) is dict else None
    generated=report.get('generated_tokens',{})
    if (type(tokens) is not list or not tokens or type(generated) is not dict or
            set(generated)!={'oracle','candidate'} or type(generated['candidate']) is not list or
            not generated['candidate'] or generated['candidate']!=generated['oracle']):
        raise ValueError('recorded prompt and identical generated trajectory required')
    if any(type(token) is not int or token<0 for token in tokens+generated['candidate']):
        raise ValueError('recorded trajectories contain invalid token IDs')
    required_tokens={f'generation.{index}.token' for index in range(len(generated['candidate']))}
    if selected['generated-tokens']!=required_tokens:
        raise ValueError('exact generation coverage omits trajectory steps')
    if value['bounds']['prefill_tokens']>len(tokens) or value['bounds']['context_tokens']>len(tokens)+len(generated['candidate']):
        raise ValueError('exact admission bounds exceed observed token extents')

def _pairs(values):
    result={}
    for name,value in values:
        if name in result: raise ValueError('duplicate exact admission field')
        result[name]=value
    return result

def _media_coverage(report, value):
    """Admit only an observed processor fixture and its full continuation."""
    if (type(report) is not dict or report.get('schema')!='openagent.k3-media-wholemodel.v2'
            or report.get('status')!='passed' or report.get('errors',[])!=[]
            or report.get('candidate_runtime')!={
                'execution_identity':value['execution_identity'],'arithmetic_identity':value['arithmetic_identity']}):
        raise ValueError('observed matching complete error-free media comparison required')
    metadata=report.get('adapter_metadata')
    if (type(metadata) is not list or len(metadata)!=2 or any(type(row) is not dict for row in metadata)
            or metadata[1].get('authoritative') is not True or metadata[1].get('unchanged') is not True
            or not metadata[0].get('cache_schema') or metadata[0]['cache_schema']!=metadata[1].get('cache_schema')):
        raise ValueError('unchanged media oracle and identical complete cache schemas required')
    if any(row.get('cache_projection') is not None for row in metadata):
        raise ValueError('exact media admission requires complete physical caches without projections')
    for key in ('model_type','config_sha256','weights_sha256','tokenizer_sha256','num_layers','vocabulary_size'):
        if metadata[0].get(key)!=metadata[1].get(key):
            raise ValueError('media comparison checkpoint identities differ')
    empty_shapes=metadata[0].get('empty_cache_shapes',{})
    if empty_shapes!=metadata[1].get('empty_cache_shapes',{}):
        raise ValueError('identical media physical-cache structural declarations required')
    text=value['report'].get('models',{}).get('candidate',{})
    for key in ('model_type','config_sha256','weights_sha256','tokenizer_sha256','num_layers','vocabulary_size','cache_schema'):
        if metadata[0].get(key)!=text.get(key):
            raise ValueError('media and text exact comparison contracts differ')
    evidence=metadata[0].get('evidence_schema',{})
    if (type(evidence) is not dict or evidence!=metadata[1].get('evidence_schema',{}) or
            evidence!=text.get('evidence_schema',{}) or
            any(type(name) is not str or not name or (layer!='global' and
                (type(layer) is not int or not 0<=layer<metadata[0]['num_layers'])) for name,layer in evidence.items())):
        raise ValueError('identical complete executed route/arithmetic evidence schemas required')
    fixture=report.get('fixture');digest=report.get('fixture_sha256')
    required={'token_ids','media_token_id','expanded_tokens','grids','pixels_shape','pixels_dtype','grids_dtype'}
    if (type(fixture) is not dict or set(fixture)!=required or type(digest) is not str
            or re.fullmatch('[0-9a-f]{64}',digest) is None):
        raise ValueError('complete pinned processor fixture identity required')
    if (type(fixture['expanded_tokens']) is not int or fixture['expanded_tokens']<=0
            or type(fixture['token_ids']) is not list or not fixture['token_ids']
            or type(fixture['media_token_id']) is not int
            or any(type(t) is not int or not 0<=t<metadata[0]['vocabulary_size'] for t in fixture['token_ids'])
            or type(fixture['grids']) is not list or not fixture['grids']
            or any(type(grid) is not list or len(grid)!=3 or any(type(n) is not int or n<=0 for n in grid) for grid in fixture['grids'])
            or fixture['token_ids'].count(fixture['media_token_id'])!=len(fixture['grids'])):
        raise ValueError('occupied exact processor token and grid geometry required')
    pixels_shape=fixture['pixels_shape']
    if (type(pixels_shape) is not list or len(pixels_shape)!=4 or
            any(type(size) is not int or not 0<size<2**63 for size in pixels_shape) or
            pixels_shape[1]!=3 or pixels_shape[2]!=pixels_shape[3] or
            pixels_shape[0]!=sum(t*h*w for t,h,w in fixture['grids']) or
            fixture['pixels_dtype']!='<f4' or type(fixture['grids_dtype']) is not str or
            re.fullmatch(r'(?:\|[iu]1|[<>][iu][248])',fixture['grids_dtype']) is None):
        raise ValueError('complete original RGB patch and integer grid declarations required')
    tokens=report.get('generated_tokens')
    if type(tokens) is not list or not tokens or any(type(t) is not int or not 0<=t<metadata[0]['vocabulary_size'] for t in tokens):
        raise ValueError('observed media continuation trajectory required')
    indexed={}
    for check in report.get('checks',[]):
        if type(check) is not dict or check.get('passed') is not True:
            raise ValueError('every recorded media comparison must pass')
        label=check.get('label')
        if type(label) is not str or not label or label in indexed:
            raise ValueError('unique media comparison labels required')
        if check.get('projected') is True or check.get('numerically_certified') is False:
            raise ValueError('projected media comparisons cannot admit exact mode')
        if label.startswith('greedy.'):
            if type(check.get('candidate')) is not int or check['candidate']!=check.get('oracle'):
                raise ValueError('exact observed media token comparison required')
        else:
            actual,expected=check.get('actual'),check.get('expected')
            _tensor_observation(actual,expected,name=label,empty_shapes=empty_shapes)
            if (type(actual) is not dict or type(expected) is not dict or actual!=expected
                    or re.fullmatch('[0-9a-f]{64}',str(actual.get('sha256'))) is None
                    or check.get('mismatches')!=0 or (check.get('exact') is not True and
                        any(check.get(key)!=0 for key in ('atol','rtol','max_absolute')))):
                raise ValueError('identical complete media tensor observations required')
        indexed[label]=check
    prefixes=['media-prefill']
    for step,token in enumerate(tokens):
        name=f'greedy.{step}'
        if name not in indexed or indexed[name]['candidate']!=token:
            raise ValueError('media token trajectory was not compared')
        prefixes.extend((f'decode.{step}',f'candidate-restore.{step}',f'oracle-restore.{step}'))
    names={prefix+'.logits' for prefix in prefixes}
    names.update(prefix+'.cache.'+leaf for prefix in prefixes for leaf in metadata[0]['cache_schema'])
    names.update(prefix+'.'+leaf for prefix in prefixes for leaf in evidence)
    names.update(f'media-prefill.vision.media.{i}' for i in range(len(fixture['grids'])))
    if not names<=set(indexed):
        raise ValueError('media evidence omits full logits, projector, physical cache or restoration observations')
    for prefix in prefixes:
        observation=indexed[prefix+'.logits']['actual']
        expected_rows=fixture['expanded_tokens'] if prefix=='media-prefill' else 1
        if observation['shape']!=[expected_rows,metadata[0]['vocabulary_size']] or observation['dtype'] not in ('<f2','<f4','<f8'):
            raise ValueError('media exact evidence requires every expanded token and full vocabulary logits')
    features=[indexed[f'media-prefill.vision.media.{i}']['actual'] for i in range(len(fixture['grids']))]
    if (any(len(row['shape'])!=2 or row['dtype'] not in ('<f2','<f4','<f8') for row in features) or
            len({row['shape'][1] for row in features})!=1 or
            sum(row['shape'][0] for row in features)!=fixture['expanded_tokens']-len(fixture['token_ids'])+len(features)):
        raise ValueError('complete observed projector rows must account for every expanded media token')
    observed={row.get('resolved_path'):row.get('sha256') for row in report.get('pins',[]) if type(row) is dict}
    if any(observed.get(path)!=pin for path,pin in value['pins'].items()):
        raise ValueError('media comparison did not use every admitted source/runtime pin')
    return copy_fixture(fixture),digest,len(tokens)

def copy_fixture(value):
    return json.loads(json.dumps(value,allow_nan=False))

class K3ExactEvidence:
    def __init__(self,path,*,expected_sha256,max_bytes=16*1024**2):
        path=_path(path).resolve(strict=True)
        if path.is_symlink() or not path.is_file() or type(max_bytes) is not int or max_bytes<=0:
            raise ValueError('bounded regular exact-evidence artifact required')
        if type(expected_sha256) is not str or len(expected_sha256)!=64 or any(v not in '0123456789abcdef' for v in expected_sha256):
            raise ValueError('explicit lowercase SHA256 evidence pin required')
        before=path.stat()
        if before.st_size>max_bytes: raise MemoryError('exact-evidence artifact exceeds admission')
        def stamp(info):return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns
        with path.open('rb') as stream:
            if stamp(os.fstat(stream.fileno()))!=stamp(before):
                raise ValueError('exact-evidence artifact changed during opening')
            raw=stream.read(max_bytes+1)
            if stamp(os.fstat(stream.fileno()))!=stamp(before):
                raise ValueError('exact-evidence artifact changed during reading')
        after=path.stat()
        if stamp(before)!=stamp(after) or len(raw)!=before.st_size:
            raise ValueError('exact-evidence artifact changed during admission')
        if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(),expected_sha256):
            raise ValueError('exact-evidence artifact differs from explicit pin')
        value=json.loads(raw,object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        fields={'schema','execution_identity','arithmetic_identity','pins','bounds','coverage','coverage_checks','report'}
        schema=value.get('schema') if type(value) is dict else None
        media=schema in ('openagent.k3-exact-evidence.v3','openagent.k3-exact-evidence.v4')
        if schema=='openagent.k3-exact-evidence.v3':fields.add('media_report')
        if schema=='openagent.k3-exact-evidence.v4':fields.add('media_reports')
        if type(value) is not dict or set(value)!=fields or schema not in ('openagent.k3-exact-evidence.v2','openagent.k3-exact-evidence.v3','openagent.k3-exact-evidence.v4'):
            raise ValueError('unsupported exact-evidence schema')
        for name in ('execution_identity','arithmetic_identity'):
            if type(value[name]) is not str or not value[name]: raise ValueError('explicit execution policy identity required')
        bounds=value['bounds']
        if type(bounds) is not dict or set(bounds)!={'context_tokens','prefill_tokens','batch_size'} or bounds['batch_size']!=1 or any(type(v) is not int or v<=0 for v in bounds.values()):
            raise ValueError('explicit batch-one comparison extent required')
        coverage=value['coverage']
        required={'whole-model-logits','complete-cache','routing','generated-tokens',
                  'prefill-and-decode','continuation-restore','quantization-boundaries'}
        if type(coverage) is not list or len(set(coverage))!=len(coverage) or set(coverage)!=required:
            raise ValueError('whole-model exact evidence coverage is incomplete')
        report=value['report']
        if type(report) is not dict or report.get('schema')!='openagent.wholemodel-validation.v1' or report.get('status')!='passed' or report.get('errors')!=[]:
            raise ValueError('successful unchanged whole-model comparison report required')
        candidate=report.get('adapter_runtime',{}).get('candidate')
        if type(candidate) is not list or not candidate or any(type(row) is not dict or
                row.get('execution_identity')!=value['execution_identity'] or
                row.get('arithmetic_identity')!=value['arithmetic_identity'] for row in candidate):
            raise ValueError('report did not observe the admitted execution/precision identities')
        oracle=report.get('models',{}).get('oracle',{})
        if oracle.get('authoritative') is not True or oracle.get('unchanged') is not True:
            raise ValueError('exact evidence requires an unchanged authoritative oracle')
        for kind in ('logits','cache'):
            tolerance=report.get('tolerances',{}).get(kind,{})
            if tolerance.get('atol')!=0 or tolerance.get('rtol')!=0:
                raise ValueError('exact admission requires zero-tolerance recorded comparisons')
        checks=report.get('checks')
        if type(checks) is not list or not checks or any(type(check) is not dict or check.get('passed') is not True for check in checks):
            raise ValueError('all recorded whole-model comparisons must pass')
        _observed_coverage(value,report)
        pins=value['pins']
        if type(pins) is not dict or not pins or len(pins)>10000:
            raise ValueError('bounded original source/runtime artifact pins required')
        compared={row['resolved_path']:row['sha256'] for role in ('oracle','candidate')
            for row in report.get('artifacts',{}).get(role,[]) if type(row) is dict and
            type(row.get('resolved_path')) is str and type(row.get('sha256')) is str}
        if any(compared.get(name)!=digest for name,digest in pins.items()):
            raise ValueError('exact artifact pins were not used in the recorded comparison')
        self._media=None
        self._media_cases=()
        if media:
            reports=[value['media_report']] if schema=='openagent.k3-exact-evidence.v3' else value['media_reports']
            if type(reports) is not list or not 1<=len(reports)<=256:
                raise ValueError('one to 256 complete media case reports required')
            cases=tuple(_media_coverage(row,value) for row in reports)
            if len({digest for _,digest,_ in cases})!=len(cases):
                raise ValueError('duplicate exact media fixture identities')
            if any(fixture['expanded_tokens']>min(bounds['prefill_tokens'],bounds['context_tokens']) for fixture,_,_ in cases):
                raise ValueError('media cases exceed the declared observed exact admission bounds')
            self._media_cases=cases
            self._media=cases[0]
        self.pins=MappingProxyType(dict(pins));self.bounds=MappingProxyType(dict(bounds))
        self.execution_identity=value['execution_identity'];self.arithmetic_identity=value['arithmetic_identity']
        self.identity=schema.replace('openagent.k3-exact-evidence.','openagent-k3-exact-evidence-')+':'+expected_sha256

    @property
    def admits_media(self):return self._media is not None

    @property
    def media_cases(self):
        """Detached case inventory for callers; never mutable live admission."""
        return tuple(dict(fixture_sha256=digest,fixture=copy_fixture(fixture),
            prefill_tokens=fixture['expanded_tokens'],context_tokens=min(self.bounds['context_tokens'],fixture['expanded_tokens']+steps),
            observed_context_tokens=fixture['expanded_tokens']+steps)
            for fixture,digest,steps in self._media_cases)

    def admit(self,execution_identity,arithmetic_identity,*,libraries):
        if execution_identity!=self.execution_identity or arithmetic_identity!=self.arithmetic_identity:
            raise ValueError('evidence execution/precision policy differs from candidate')
        engine=Path(__file__).resolve().parents[1]
        required={str(path.resolve()) for path in engine.joinpath('k3_execution').glob('*.py')}
        required.update(str(engine.joinpath(name).resolve(strict=True)) for name in
            ('session_recovery.py','tensor_parallel.py','device_execution_policy.py'))
        required.update(str(Path(path).resolve(strict=True)) for path in libraries if path is not None)
        if not required<=set(self.pins): raise ValueError('evidence does not pin every K3 execution source and native library')
        for name,digest in self.pins.items():
            if type(name) is not str or type(digest) is not str or len(digest)!=64:
                raise ValueError('invalid exact source artifact pin')
            if not Path(name).is_absolute():raise ValueError('absolute exact source artifact required')
            path=_path(name)
            if not path.is_absolute() or path.is_symlink() or not path.is_file(): raise ValueError('regular absolute exact source artifact required')
            before=path.stat();observed=hashlib.sha256()
            def stamp(info):return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns
            with path.open('rb') as stream:
                if stamp(os.fstat(stream.fileno()))!=stamp(before):
                    raise ValueError('exact comparison artifact changed during opening: '+name)
                for chunk in iter(lambda:stream.read(1024**2),b''): observed.update(chunk)
                if stamp(os.fstat(stream.fileno()))!=stamp(before):
                    raise ValueError('exact comparison artifact changed during reading: '+name)
            after=path.stat()
            if stamp(before)!=stamp(after) or not hmac.compare_digest(observed.hexdigest(),digest):
                raise ValueError('exact comparison artifact changed: '+name)
        return self.identity

    def admit_snapshot(self,media_case,position):
        if type(position) is not int or position<0:
            raise ValueError('nonnegative exact continuation position required')
        context=self.bounds['context_tokens']
        if media_case is not None:
            matches=[(fixture,steps) for fixture,digest,steps in self._media_cases if digest==media_case]
            if type(media_case) is not str or len(matches)!=1:
                raise ValueError('unknown exact media continuation case')
            fixture,steps=matches[0]
            context=min(context,fixture['expanded_tokens']+steps)
            if position<fixture['expanded_tokens']:
                raise ValueError('exact media continuation precedes its observed prefill')
        if type(position) is not int or not 0<=position<=context:
            raise ValueError('continuation exceeds selected exact comparison extent')
        return context

    def admit_call(self,*,position,tokens,has_media,media_case=None):
        if type(has_media) is not bool:raise ValueError('explicit exact media-presence policy required')
        if has_media and self._media is None: raise ValueError('this exact evidence admits text only')
        context=self.admit_snapshot(media_case,position)
        if type(tokens) is not int or tokens<=0 or tokens>self.bounds['prefill_tokens'] or position+tokens>context:
            raise ValueError('request exceeds recorded exact comparison extent')

    def admit_media(self,*,position,token_ids,media_token_id,pixels,grids,expanded_tokens):
        if self._media is None:raise ValueError('pinned complete media evidence required')
        import numpy as np
        if type(position) is not int or position!=0 or pixels is None or grids is None:
            raise ValueError('media request differs from the recorded original processor fixture')
        if type(media_token_id) is not int or type(expanded_tokens) is not int or expanded_tokens<=0:
            raise ValueError('explicit integer media placeholder and expanded extent required')
        if isinstance(token_ids,np.ndarray):
            if token_ids.ndim!=1 or token_ids.dtype.kind not in 'iu' or token_ids.size>self.bounds['prefill_tokens']:
                raise ValueError('bounded original integer media token inventory required')
        elif (type(token_ids) not in (tuple,list) or len(token_ids)>self.bounds['prefill_tokens'] or
                any(type(token) is not int for token in token_ids)):
            raise ValueError('bounded original integer media token inventory required')
        ids=tuple(int(token) for token in token_ids)
        if not isinstance(pixels,np.ndarray):raise ValueError('original typed processor patch array required')
        if not isinstance(grids,np.ndarray):
            maximum=max(len(fixture['grids']) for fixture,_,_ in self._media_cases)
            if (type(grids) not in (tuple,list) or not 0<len(grids)<=maximum or
                    any(type(row) not in (tuple,list) or len(row)!=3 or
                        any(type(value) is not int for value in row) for row in grids)):
                raise ValueError('bounded original integer media grid inventory required')
        grids=np.asarray(grids)
        candidates=[(fixture,digest,steps) for fixture,digest,steps in self._media_cases
            if ids==tuple(fixture['token_ids']) and media_token_id==fixture['media_token_id']
            and expanded_tokens==fixture['expanded_tokens'] and list(pixels.shape)==fixture['pixels_shape']
            and pixels.dtype.str==fixture['pixels_dtype'] and grids.shape==(len(fixture['grids']),3)
            and grids.dtype.str==fixture['grids_dtype']]
        if not candidates:raise ValueError('media layout has no recorded exact evidence case')
        observed=hashlib.sha256(repr((ids,media_token_id,expanded_tokens)).encode())
        for array in (pixels,grids):
            observed.update(repr((array.shape,array.dtype.str)).encode())
            contiguous=np.ascontiguousarray(array)
            observed.update(memoryview(contiguous).cast('B'))
        matches=[(fixture,digest,steps) for fixture,digest,steps in candidates
                 if hmac.compare_digest(observed.hexdigest(),digest)]
        if len(matches)!=1:
            raise ValueError('media processor fixture has no unique exact evidence case')
        fixture,digest,steps=matches[0]
        if (list(ids)!=fixture['token_ids'] or media_token_id!=fixture['media_token_id']
                or expanded_tokens!=fixture['expanded_tokens'] or list(pixels.shape)!=fixture['pixels_shape']
                or pixels.dtype.str!=fixture['pixels_dtype'] or grids.dtype.str!=fixture['grids_dtype']
                or grids.tolist()!=fixture['grids']):
            raise ValueError('media processor layout differs from selected exact evidence')
        if expanded_tokens>min(self.bounds['prefill_tokens'],self.bounds['context_tokens'],fixture['expanded_tokens']+steps):
            raise ValueError('media prefill exceeds selected observed exact extent')
        return digest

__all__=['K3ExactEvidence']
