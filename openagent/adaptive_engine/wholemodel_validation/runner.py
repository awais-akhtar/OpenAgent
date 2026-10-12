"""Offline whole-model full/incremental/cache/greedy comparison harness."""
from contextlib import ExitStack
from dataclasses import asdict
from collections.abc import Mapping
import copy
import hashlib
import importlib
import json
import os
import stat
from pathlib import Path
import platform
import sys
import tempfile
import weakref

import numpy as np

from .comparison import Tolerance, canonical, compare, identity
from .contracts import Observation
from .cache_readers import declared_empty
from ..session_recovery import _path
from ..native_library_identity import _stamp


def digest_file(path):
    path=_path(path);before=path.stat()
    if not stat.S_ISREG(before.st_mode):raise ValueError('regular pinned artifact required')
    digest = hashlib.sha256()
    descriptor=os.open(path,os.O_RDONLY|getattr(os,'O_BINARY',0)|
        getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))
    try:
        if _stamp(os.fstat(descriptor))!=_stamp(before):
            raise ValueError('pinned artifact changed during opening')
        stream=os.fdopen(descriptor,'rb')
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        count=0
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            count+=len(block)
            if count>before.st_size:raise ValueError('pinned artifact grew during hashing')
            digest.update(block)
        if count!=before.st_size or _stamp(os.fstat(stream.fileno()))!=_stamp(before):
            raise ValueError('pinned artifact changed during hashing')
    if _stamp(_path(path).stat())!=_stamp(before):
        raise ValueError('pinned artifact path changed during hashing')
    return digest.hexdigest()


def pinned_artifacts(spec, root):
    root = _path(root).resolve(strict=True)
    rows = spec['artifacts']
    if (not isinstance(rows, list) or not rows or
            any(not isinstance(r, dict) for r in rows) or
            not {'source', 'config', 'weights', 'tokenizer'} <= {r['kind'] for r in rows}):
        raise ValueError('source/config/weights/tokenizer pins required')
    observed = []
    entries = set()
    paths = set()
    for row in rows:
        if not isinstance(row.get('path'), str) or not row['path']:
            raise ValueError('nonempty artifact path required')
        if row['kind'] not in {'source', 'config', 'weights', 'tokenizer', 'dependency'}:
            raise ValueError('unsupported artifact kind')
        pin = row['sha256']
        if not isinstance(pin, str) or len(pin) != 64 or any(c not in '0123456789abcdef' for c in pin):
            raise ValueError('artifact must have a lowercase SHA-256 pin')
        path = _path(root / row['path']).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError('artifact outside offline root')
        entry = (row['kind'], str(path).casefold())
        if entry in entries or entry[1] in paths:
            raise ValueError('duplicate or ambiguously classified artifact path')
        entries.add(entry)
        paths.add(entry[1])
        digest = digest_file(path)
        if digest != row['sha256']:
            raise ValueError('artifact hash mismatch: ' + row['path'])
        observed.append(dict(row, resolved_path=str(path), observed_sha256=digest))
    return observed


def checkpoint_identity(spec):
    """Match the complete checkpoint, not a metadata-selected single shard."""
    return {kind: sorted(row['sha256'] for row in spec['artifacts'] if row['kind'] == kind)
            for kind in ('config', 'weights', 'tokenizer')}


def load_adapter(spec, root):
    module, name = spec['factory'].split(':')
    if not module or not name:
        raise ValueError('explicit module:factory required')
    options = copy.deepcopy(dict(spec.get('options', {})))
    if '_artifact_pins' in options:
        raise ValueError('_artifact_pins is reserved for runner-supplied pins')
    options['_artifact_pins'] = [dict(row) for row in spec['artifacts']]
    return getattr(importlib.import_module(module), name)(
        offline_root=root, options=options)


def metadata(adapter, spec, oracle=False):
    data = dict(adapter.metadata())
    required = ('model_type', 'revision', 'config_sha256', 'weights_sha256',
                'tokenizer_sha256', 'vocabulary_size', 'num_layers', 'dtype',
                'implementation', 'implementation_revision', 'device',
                'dependencies', 'cache_schema')
    if any(key not in data for key in required):
        raise ValueError('incomplete adapter metadata')
    if oracle and (data.get('authoritative') is not True or data.get('unchanged') is not True):
        raise ValueError('oracle must declare authoritative and unchanged')
    for key in ('vocabulary_size', 'num_layers'):
        if type(data[key]) is not int or data[key] <= 0:
            raise ValueError('invalid model geometry: ' + key)
    for key in ('model_type', 'revision', 'dtype', 'implementation', 'implementation_revision', 'device'):
        if not isinstance(data[key], str) or not data[key]:
            raise ValueError('nonempty metadata required: ' + key)
    if (not isinstance(data['dependencies'], dict) or
            any(not isinstance(k, str) or not k or
                (v is not None and (not isinstance(v, str) or not v))
                for k,v in data['dependencies'].items())):
        raise ValueError('named dependency versions or explicit unavailable values required')
    schema = data['cache_schema']
    if not isinstance(schema, dict) or not schema or any(not isinstance(k, str) or not k for k in schema):
        raise ValueError('complete nonempty named cache schema required')
    if any(v != 'global' and (type(v) is not int or not 0 <= v < data['num_layers']) for v in schema.values()):
        raise ValueError('invalid cache layer identity')
    if not set(range(data['num_layers'])) <= set(schema.values()):
        raise ValueError('cache schema must cover every layer')
    evidence_schema = data.get('evidence_schema', {})
    if (not isinstance(evidence_schema, dict)
            or any(not isinstance(k, str) or not k for k in evidence_schema)
            or any(v != 'global' and (type(v) is not int or not 0 <= v < data['num_layers'])
                   for v in evidence_schema.values())):
        raise ValueError('invalid separate evidence schema')
    for name, pattern in data.get('empty_evidence_shapes', {}).items():
        if (name not in evidence_schema or not isinstance(pattern, list) or 0 not in pattern
                or any(p is not None and (type(p) is not int or p < 0) for p in pattern)):
            raise ValueError('invalid explicitly declared structural empty evidence shape')
    for name, pattern in data.get('empty_cache_shapes', {}).items():
        if (name not in schema or not isinstance(pattern, list) or 0 not in pattern
                or any(p is not None and (type(p) is not int or p < 0) for p in pattern)):
            raise ValueError('invalid explicitly declared structural empty cache shape')
    for kind in ('config', 'weights', 'tokenizer'):
        matches = [row for row in spec['artifacts'] if row['kind'] == kind
                   and row['sha256'] == data[kind + '_sha256']]
        if len(matches) != 1:
            raise ValueError('metadata digest must identify exactly one pinned ' + kind + ' artifact')
    json.dumps(data, allow_nan=False)
    return data


def observe(session, tokens, meta):
    tokens = tuple(tokens)
    if not tokens or any(type(t) is not int or not 0<=t<meta['vocabulary_size'] for t in tokens):
        raise ValueError('nonempty in-vocabulary observation tokens required')
    if getattr(session, 'failed', False) or getattr(session, 'closed', False):
        raise RuntimeError('closed or failed session cannot produce evidence')
    position = getattr(session, 'position', None)
    if position is not None and (type(position) is not int or position<0):
        raise ValueError('invalid session position')
    raw = session.forward(tuple(tokens))
    if getattr(session, 'failed', False) or getattr(session, 'closed', False):
        raise RuntimeError('session failed while producing evidence')
    if position is not None and getattr(session, 'position', None)!=position+len(tokens):
        raise ValueError('session did not advance by appended token count')
    if not isinstance(raw, Observation):
        raise TypeError('forward must return Observation')
    logits = canonical(raw.logits)
    if logits.shape != (len(tokens), meta['vocabulary_size']):
        raise ValueError('logits must be [appended tokens, vocabulary]')
    if logits.dtype.kind != 'f' or not np.isfinite(logits).all():
        raise ValueError('finite floating logits required')
    if not isinstance(raw.cache, Mapping) or set(raw.cache) != set(meta['cache_schema']):
        raise ValueError('missing or extra canonical cache leaves')
    cache = {key: canonical(value) for key, value in raw.cache.items()}
    if any((not value.size and not declared_empty(value, meta.get('empty_cache_shapes', {}).get(name)))
           or not np.isfinite(value).all() for name, value in cache.items()):
        raise ValueError('undeclared empty or nonfinite cache leaf')
    if not isinstance(raw.evidence, Mapping) or set(raw.evidence) != set(meta.get('evidence_schema', {})):
        raise ValueError('missing or extra separate evidence leaves')
    evidence = {key: canonical(value) for key, value in raw.evidence.items()}
    if any((not value.size and not declared_empty(value, meta.get('empty_evidence_shapes', {}).get(name)))
           or not np.isfinite(value).all() for name, value in evidence.items()):
        raise ValueError('undeclared empty or nonfinite separate evidence leaf')
    for layer in range(meta['num_layers']):
        if not any(value.size and meta['cache_schema'][key]==layer for key,value in cache.items()):
            raise ValueError('no nonempty continuation cache evidence for layer '+str(layer))
    return Observation(logits, cache, evidence)


def safe_json(value):
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.generic):
        return safe_json(value.item())
    return value


def checked_runtime(value):
    if not isinstance(value, (dict,list)) or not value:
        raise ValueError('nonempty runtime evidence required when exposed')
    if isinstance(value,list) and any(not isinstance(row,dict) or not row for row in value):
        raise ValueError('runtime records must be nonempty named mappings')
    json.dumps(value,allow_nan=False)
    def visit(node):
        if isinstance(node,dict):
            if (node.get('failed') is True or node.get('success') is False or
                    node.get('passed') is False or node.get('training') is True or
                    node.get('deterministic_algorithms') is False or
                    node.get('status') in ('failed','error','unsupported') or
                    node.get('error') or node.get('errors')):
                raise ValueError('adapter runtime reports failure or nondeterministic execution')
            for child in node.values():
                visit(child)
        elif isinstance(node,list):
            for child in node:
                visit(child)
    visit(value)
    return copy.deepcopy(value)


def run(manifest, *, base_directory, output_directory):
    """Write report.json and observations.npz. Missing evidence always fails."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(schema='openagent.wholemodel-validation.v1', status='failed',
                  certification=False, scope='exact pinned text-model case only',
                  manifest=manifest, checks=[], errors=[], artifacts={})
    arrays = {}
    adapters = {}
    root = None
    output_conflicts = set()
    try:
        encoded = json.dumps(manifest, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode('utf-8')
        report['manifest_sha256'] = hashlib.sha256(encoded).hexdigest()
        manifest = json.loads(encoded)
        report['manifest'] = manifest
        report['runtime'] = dict(python=sys.version, platform=platform.platform(),
                                 numpy=np.__version__)
        if manifest['schema'] != 'openagent.wholemodel-validation-input.v1':
            raise ValueError('unsupported input schema')
        root = (Path(base_directory) / manifest['offline_root']).resolve()
        seed = manifest['seed']
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('seed must be uint32')
        tokens = manifest['tokens']
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError('nonempty nonnegative integer token list required')
        steps = manifest['generation_steps']
        if type(steps) is not int or steps < 1:
            raise ValueError('generation_steps must be positive')
        log_tol = Tolerance(**manifest['tolerances']['logits'])
        cache_tol = Tolerance(**manifest['tolerances']['cache'])
        if any(type(value) not in (int,float) for tol in (log_tol,cache_tol) for value in (tol.atol,tol.rtol)):
            raise ValueError('numeric nonboolean tolerances required')
        report['tolerances'] = dict(logits=asdict(log_tol), cache=asdict(cache_tol))
        # Pin both roles before either can execute imported adapter code.
        for role in ('oracle', 'candidate'):
            report['artifacts'][role] = pinned_artifacts(manifest[role], root)
            output_conflicts.update(Path(row['resolved_path']) for row in report['artifacts'][role]
                if Path(row['resolved_path']) in
                   {(output/'report.json').resolve(),(output/'observations.npz').resolve()})
            if output_conflicts:
                raise ValueError('validation output would overwrite a pinned artifact')
        identities = {role: checkpoint_identity(manifest[role]) for role in ('oracle', 'candidate')}
        if identities['oracle'] != identities['candidate']:
            raise ValueError('complete config/weights/tokenizer pin sets differ between roles')
        report['checkpoint_identity'] = identities['oracle']
        for role in ('oracle', 'candidate'):
            adapters[role] = load_adapter(manifest[role], root)
        if adapters['oracle'] is adapters['candidate']:
            raise ValueError('independent adapters required')
        source_pins = {role: sorted(row['sha256'] for row in report['artifacts'][role]
                       if row['kind']=='source') for role in adapters}
        if source_pins['oracle']==source_pins['candidate']:
            raise ValueError('oracle and candidate have identical implementation source pin sets')
        metas = {role: metadata(adapter, manifest[role], role == 'oracle') for role, adapter in adapters.items()}
        report['models'] = metas
        if ((metas['oracle']['implementation'], metas['oracle']['implementation_revision']) ==
                (metas['candidate']['implementation'], metas['candidate']['implementation_revision'])):
            raise ValueError('oracle and candidate declare the same implementation identity')
        report['independence'] = dict(distinct_adapters=True, distinct_implementation_identities=True,
                                      source_sha256=source_pins,
                                      certification=False)
        projection = metas['candidate'].get('cache_projection')
        if projection is not None:
            if (not isinstance(projection, dict) or projection.get('certified') is not False
                    or projection.get('tolerance') != asdict(cache_tol)
                    or not isinstance(projection.get('leaves'), list)
                    or not projection['leaves']
                    or len(set(projection['leaves'])) != len(projection['leaves'])
                    or not set(projection['leaves']) <= set(metas['candidate']['cache_schema'])
                    or not metas['candidate'].get('evidence_schema')):
                raise ValueError('projection requires uncertified explicit matching cache tolerance and raw evidence')
            report['comparison_projection'] = projection
        for key in ('model_type', 'revision', 'config_sha256', 'weights_sha256',
                    'tokenizer_sha256', 'vocabulary_size', 'num_layers', 'cache_schema'):
            if metas['oracle'][key] != metas['candidate'][key]:
                raise ValueError('model identity/contract mismatch: ' + key)
        if metas['oracle'].get('text_model_type') != metas['candidate'].get('text_model_type'):
            raise ValueError('explicit inner text_model_type differs between roles')
        if metas['oracle'].get('cache_layout') != metas['candidate'].get('cache_layout'):
            raise ValueError('cache layouts differ; matching leaf names do not prove semantic equivalence')
        if metas['oracle'].get('empty_cache_shapes', {}) != metas['candidate'].get('empty_cache_shapes', {}):
            raise ValueError('structural empty cache declarations differ between roles')
        if any(t >= metas['oracle']['vocabulary_size'] for t in tokens):
            raise ValueError('input token outside vocabulary')
        schedules = [[len(tokens)], [1] * len(tokens)]
        cross_evidence = manifest.get('cross_evidence', [])
        if type(cross_evidence) is not list or len(cross_evidence)>65536:
            raise ValueError('bounded cross-model evidence contract list required')
        evidence_names = set()
        for contract in cross_evidence:
            if (type(contract) is not dict or set(contract)!={'name','oracle','candidate','tolerance'} or
                    any(type(contract[key]) is not str or not contract[key] for key in ('name','oracle','candidate')) or
                    contract['name'] in evidence_names):
                raise ValueError('unique explicit cross-model evidence mapping required')
            evidence_names.add(contract['name'])
            for role in ('oracle','candidate'):
                if contract[role] not in metas[role].get('evidence_schema',{}):
                    raise ValueError('cross-model evidence leaf is not exported by '+role)
            if type(contract['tolerance']) is not dict or set(contract['tolerance'])!={'atol','rtol'}:
                raise ValueError('explicit evidence tolerance required')
            if any(type(v) not in (int,float) for v in contract['tolerance'].values()):
                raise ValueError('numeric nonboolean evidence tolerance required')
            Tolerance(**contract['tolerance'])
        report['cross_evidence_contracts'] = cross_evidence
        for schedule in manifest.get('chunk_schedules', []):
            if not isinstance(schedule, list) or not schedule or any(type(n) is not int or n <= 0 for n in schedule) or sum(schedule) != len(tokens):
                raise ValueError('chunk schedule must partition prompt')
            schedules.append(schedule)

        def record(label, observation):
            arrays[label + '.logits'] = observation.logits
            for key, value in observation.cache.items():
                arrays[label + '.cache.' + key] = value
            for key, value in observation.evidence.items():
                arrays[label + '.evidence.' + key] = value

        def check(label, actual, expected, *, evidence_role=None):
            if set(actual.cache)!=set(expected.cache) or not expected.cache:
                raise ValueError('comparison requires identical nonempty cache inventories')
            report['checks'].append(dict(name=label + '.logits',
                                        **compare(actual.logits, expected.logits, log_tol)))
            if evidence_role is None:
                for contract in cross_evidence:
                    report['checks'].append(dict(name=label+'.cross_evidence.'+contract['name'],
                        projected=False,oracle_leaf=contract['oracle'],candidate_leaf=contract['candidate'],
                        **compare(actual.evidence[contract['candidate']],expected.evidence[contract['oracle']],
                                  Tolerance(**contract['tolerance']))))
            for key in sorted(expected.cache):
                report['checks'].append(dict(name=label + '.cache.' + key,
                    layer=metas['oracle']['cache_schema'][key],
                    projected=bool(projection and evidence_role != 'oracle' and key in projection['leaves']),
                    numerically_certified=False if projection and evidence_role != 'oracle' and key in projection['leaves'] else None,
                    **compare(actual.cache[key], expected.cache[key], cache_tol,
                              allow_empty=declared_empty(expected.cache[key],
                                  metas['oracle'].get('empty_cache_shapes', {}).get(key)))))
            if evidence_role is not None and expected.evidence:
                if set(actual.evidence) != set(expected.evidence):
                    raise ValueError('same-layout raw evidence inventory differs')
                for key in sorted(expected.evidence):
                    # Raw evidence is only compared when both observations expose
                    # it (same-role prefix checks), never against expanded oracle KV.
                    report['checks'].append(dict(name=label + '.evidence.' + key,
                        projected=False,
                        **compare(actual.evidence[key], expected.evidence[key],
                                  Tolerance(0, 0) if key.endswith('.weight') else cache_tol,
                                  allow_empty=declared_empty(expected.evidence[key],
                                      metas[evidence_role].get('empty_evidence_shapes', {}).get(key)))))

        sessions_seen = []
        owners_seen = []
        releasers = {}
        with ExitStack() as resources:
            def fresh(role):
                session = adapters[role].new_session(seed=seed)
                if any(session is previous for previous in sessions_seen):
                    raise ValueError('adapter reused a session')
                if not callable(getattr(session,'close',None)):
                    raise TypeError('session must expose close')
                sessions_seen.append(session)
                released = False
                def release_once():
                    nonlocal released
                    if not released:
                        released = True
                        try:
                            session.close()
                        except Exception as error:
                            report['status'] = 'failed'
                            report['errors'].append(dict(role=role,stage='session_close',
                                type=type(error).__name__,message=str(error)))
                releasers[id(session)] = release_once
                resources.callback(release_once)
                if not callable(getattr(session,'forward',None)):
                    raise TypeError('session must expose forward')
                if getattr(session,'position',0)!=0:
                    raise ValueError('fresh session is not at position zero')
                local_owners = []
                for name in ('model','executor'):
                    owner = getattr(session,name,None)
                    if owner is None or any(owner is previous for previous in local_owners):
                        continue
                    local_owners.append(owner)
                    if any(owner is reference() for reference in owners_seen):
                        raise ValueError('sessions share a model/executor owner')
                    try:
                        owners_seen.append(weakref.ref(owner))
                    except TypeError:
                        owners_seen.append(lambda owner=owner: owner)
                return session

            def full_prefix(role, prefix_tokens):
                session = fresh(role)
                try:
                    return observe(session, prefix_tokens, metas[role])
                finally:
                    releasers[id(session)]()

            baseline = {}
            for index, schedule in enumerate(schedules):
                sessions = {role: fresh(role) for role in adapters}
                accumulated = {role: [] for role in adapters}
                position = 0
                last = {}
                for chunk in schedule:
                    for role in adapters:
                        last[role] = observe(sessions[role], tokens[position:position + chunk], metas[role])
                        accumulated[role].append(last[role].logits)
                        record(f'schedule.{index}.position.{position + chunk}.{role}', last[role])
                    check(f'schedule.{index}.position.{position + chunk}.cross', last['candidate'], last['oracle'])
                    position += chunk
                    if index != 0:
                        # Full prefixes use fresh state, not incremental replay.
                        for role in adapters:
                            prefix = full_prefix(role, tokens[:position])
                            joined = Observation(np.concatenate(accumulated[role]), last[role].cache,
                                                 last[role].evidence)
                            record(f'schedule.{index}.prefix.{position}.{role}', prefix)
                            check(f'schedule.{index}.prefix.{position}.{role}', joined, prefix,
                                  evidence_role=role)
                for role in adapters:
                    joined = Observation(np.concatenate(accumulated[role]), last[role].cache,
                                         last[role].evidence)
                    if index == 0:
                        baseline[role] = joined
                    else:
                        check(f'schedule.{index}.full.{role}', joined, baseline[role], evidence_role=role)
                    releasers[id(sessions[role])]()

            # Independent greedy trajectories; argmax ties use lowest token ID.
            generated = {role: [] for role in adapters}
            report['generated_tokens'] = generated
            sessions = {role: fresh(role) for role in adapters}
            last = {role: observe(sessions[role], tokens, metas[role]) for role in adapters}
            for role in adapters:
                record('generation.prefill.' + role, last[role])
                check('generation.prefill.'+role,last[role],baseline[role],evidence_role=role)
            check('generation.prefill', last['candidate'], last['oracle'])
            for step in range(steps):
                for role in adapters:
                    generated[role].append(int(np.argmax(last[role].logits[-1])))
                report['checks'].append(dict(name=f'generation.{step}.token',
                    passed=generated['candidate'][-1] == generated['oracle'][-1],
                    actual=generated['candidate'][-1], expected=generated['oracle'][-1]))
                for role in adapters:
                    last[role] = observe(sessions[role], [generated[role][-1]], metas[role])
                    record(f'generation.{step}.{role}', last[role])
                    prefix = full_prefix(role, tokens + generated[role])
                    record(f'generation.{step}.prefix.{role}', prefix)
                    check(f'generation.{step}.prefix.{role}', last[role],
                          Observation(prefix.logits[-1:], prefix.cache, prefix.evidence),
                          evidence_role=role)
                check(f'generation.{step}.cross', last['candidate'], last['oracle'])

            # Separate shared-input rollout preserves numerical diagnostics even
            # when the independent candidate trajectory chooses a different token.
            shared = {role: fresh(role) for role in adapters}
            shared_last = {role: observe(shared[role], tokens, metas[role]) for role in adapters}
            for role in adapters:
                record('teacher_forced.prefill.' + role, shared_last[role])
                check('teacher_forced.prefill.'+role,shared_last[role],baseline[role],evidence_role=role)
            check('teacher_forced.prefill', shared_last['candidate'], shared_last['oracle'])
            for step, token in enumerate(generated['oracle']):
                for role in adapters:
                    shared_last[role] = observe(shared[role], [token], metas[role])
                    record(f'teacher_forced.{step}.{role}', shared_last[role])
                    prefix = full_prefix(role,tokens+generated['oracle'][:step+1])
                    record(f'teacher_forced.{step}.prefix.{role}',prefix)
                    check(f'teacher_forced.{step}.prefix.{role}',shared_last[role],
                          Observation(prefix.logits[-1:],prefix.cache,prefix.evidence),evidence_role=role)
                check(f'teacher_forced.{step}.cross', shared_last['candidate'], shared_last['oracle'])
            restore_case=manifest.get('continuation_restore',False)
            if type(restore_case) is not bool:
                raise ValueError('explicit continuation restoration comparison policy required')
            if restore_case:
                candidate=fresh('candidate')
                for method in ('snapshot','restore','release_snapshot'):
                    if not callable(getattr(candidate,method,None)):
                        raise ValueError('candidate lacks complete restoration protocol: '+method)
                observe(candidate,tokens,metas['candidate'])
                receipt=candidate.snapshot()
                try:
                    suffix=generated['oracle']
                    if not suffix:raise ValueError('restoration comparison requires generated continuation tokens')
                    first=observe(candidate,suffix,metas['candidate'])
                    candidate.restore(receipt)
                    restored=observe(candidate,suffix,metas['candidate'])
                    record('continuation.restore.candidate',restored)
                    check('continuation.restore.same_candidate',restored,first,evidence_role='candidate')
                    oracle_session=fresh('oracle')
                    observe(oracle_session,tokens,metas['oracle'])
                    tail=observe(oracle_session,suffix,metas['oracle'])
                    record('continuation.restore.oracle',tail)
                    check('continuation.restore.cross',restored,tail)
                finally:
                    candidate.release_snapshot(receipt)
        report['status'] = 'passed' if report['checks'] and all(c['passed'] for c in report['checks']) else 'failed'
    except Exception as error:
        detail = dict(type=type(error).__name__, message=str(error))
        if hasattr(error, 'unsupported_fields'):
            detail['unsupported_fields'] = list(error.unsupported_fields)
        report['errors'].append(detail)
    report['adapter_runtime'] = {}
    for role, adapter in adapters.items():
        try:
            getter = getattr(adapter, 'runtime_metadata', None)
            if getter is not None:
                if not callable(getter):
                    raise TypeError('runtime_metadata must be callable')
                report['adapter_runtime'][role] = checked_runtime(getter())
        except Exception as error:
            report['status'] = 'failed'
            report['errors'].append(dict(role=role,stage='runtime_metadata',type=type(error).__name__,message=str(error)))
    closed_adapters = set()
    for role,adapter in adapters.items():
        if id(adapter) in closed_adapters:
            continue
        closed_adapters.add(id(adapter))
        try:
            closer = getattr(adapter,'close',None)
            if closer is not None:
                if not callable(closer):
                    raise TypeError('adapter close must be callable')
                closer()
        except Exception as error:
            report['status'] = 'failed'
            report['errors'].append(dict(role=role,stage='adapter_close',type=type(error).__name__,message=str(error)))
    report['artifacts_after_cleanup'] = {}
    if root is not None:
        for role in report['artifacts']:
            try:
                observed = pinned_artifacts(manifest[role],root)
                if observed!=report['artifacts'][role]:
                    raise ValueError('artifact evidence changed during validation')
                report['artifacts_after_cleanup'][role] = observed
            except Exception as error:
                report['status'] = 'failed'
                report['errors'].append(dict(role=role,stage='artifact_recheck',type=type(error).__name__,message=str(error)))
    if arrays:
        artifact = output / 'observations.npz'
        temporary = None
        try:
            observations = {key: identity(value) for key,value in arrays.items()}
            with tempfile.NamedTemporaryFile(dir=output,suffix='.npz',delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream,**arrays)
            digest = digest_file(temporary)
            temporary.replace(artifact)
            report['observation_artifact'] = dict(path=str(artifact.resolve()),sha256=digest)
            report['observations'] = observations
        except Exception as error:
            report['status'] = 'failed'
            report['errors'].append(dict(stage='observations',type=type(error).__name__,message=str(error)))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if report['errors']:
        report['status'] = 'failed'
    report = safe_json(report)
    if (output/'report.json').resolve() in output_conflicts:
        return report
    temporary = None
    try:
        encoded = json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n'
        with tempfile.NamedTemporaryFile(dir=output,suffix='.json',mode='w',encoding='utf-8',delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
        temporary.replace(output/'report.json')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report
