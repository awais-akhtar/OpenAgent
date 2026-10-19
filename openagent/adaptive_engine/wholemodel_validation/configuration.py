"""Construct offline pinned comparison inputs without invoking model factories."""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import sys
from ..session_recovery import _path
from ..segment_snapshot import _write
from ..generation_checkpoint_files import _remove_staging


@dataclass(frozen=True)
class ArtifactSelection:
    kind: str
    path: str


def pin_artifacts(root, selections, *, max_files=200000, max_file_bytes=1 << 44):
    """Hash caller-selected source/dependency/checkpoint files with file guards."""
    root = _path(root).resolve(strict=True)
    if not root.is_dir() or any(type(v) is not int or v<=0 for v in (max_files,max_file_bytes)):
        raise ValueError('local offline root and positive artifact limits required')
    rows, seen = [],set()
    for selection in selections:
        if len(rows)>=max_files or type(selection) is not ArtifactSelection:
            raise ValueError('bounded explicit artifact selections required')
        if selection.kind not in ('source','config','weights','tokenizer','dependency'):
            raise ValueError('unsupported artifact role')
        if type(selection.path) is not str or not selection.path:
            raise ValueError('explicit nonempty artifact path required')
        path = _path(root/selection.path).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file() or path in seen:
            raise ValueError('unique regular artifacts inside the offline root required')
        seen.add(path)
        before = path.stat()
        stamp = lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if not 0<before.st_size<=max_file_bytes:
            raise MemoryError('artifact extent exceeds pinning admission')
        digest = hashlib.sha256()
        with path.open('rb') as source:
            opened = os.fstat(source.fileno())
            if stamp(opened)!=stamp(before):
                raise ValueError('artifact replaced during pinning')
            remaining = before.st_size
            while remaining:
                block = source.read(min(1024**2,remaining))
                if not block:
                    raise ValueError('artifact truncated during pinning')
                digest.update(block)
                remaining -= len(block)
            if source.read(1) or stamp(os.fstat(source.fileno()))!=stamp(before):
                raise ValueError('artifact changed during pinning')
        if stamp(_path(path).stat())!=stamp(before):
            raise ValueError('artifact replaced after pinning')
        rows.append(dict(kind=selection.kind,path=path.relative_to(root).as_posix(),sha256=digest.hexdigest()))
    if not {'source','config','weights','tokenizer','dependency'} <= {row['kind'] for row in rows}:
        raise ValueError('complete source/config/weights/tokenizer/dependency selections required')
    return sorted(rows,key=lambda row:(row['kind'],row['path']))


def comparison_configuration(*, offline_root, oracle_factory, candidate_factory,
                             oracle_artifacts, candidate_artifacts, tokens,
                             oracle_options=None, candidate_options=None,
                             logits_tolerance, cache_tolerance, routing_evidence=(),
                             generation_steps=8, seed=0, chunk_sizes=(2,7,32),continuation_restore=False,
                             execution_contract=None):
    """Create a complete comparison case with explicit logits/cache/routing policy.

    Artifact arguments are already pinned records, never guessed revisions.
    Construction performs no imports of factories and executes no comparisons.
    """
    for factory in (oracle_factory,candidate_factory):
        if type(factory) is not str or factory.count(':')!=1 or any(not s for s in factory.split(':')):
            raise ValueError('explicit module:factory required')
    if type(seed) is not int or not 0<=seed<2**32 or type(generation_steps) is not int or generation_steps<=0:
        raise ValueError('uint32 seed and positive generation steps required')
    ids = list(tokens)
    if not 0<len(ids)<=1048576 or any(type(t) is not int or t<0 for t in ids):
        raise ValueError('bounded nonempty integer prompt required')
    for tolerance in (logits_tolerance,cache_tolerance):
        if (type(tolerance) is not dict or set(tolerance)!={'atol','rtol'} or
                any(type(v) not in (float,int) or not math.isfinite(v) or v<0 for v in tolerance.values())):
            raise ValueError('explicit finite nonnegative numerical tolerances required')
    schedules = []
    for size in chunk_sizes:
        if type(size) is not int or size<=0:
            raise ValueError('positive prefill chunk sizes required')
        schedule = [size]*(len(ids)//size)
        if len(ids)%size:
            schedule.append(len(ids)%size)
        if schedule not in schedules:
            schedules.append(schedule)
    if type(continuation_restore) is not bool:
        raise ValueError('explicit continuation comparison policy required')
    if execution_contract is not None:
        required_contract = {'family', 'model_type', 'backend', 'arithmetic',
                             'source_format', 'persisted_format', 'media'}
        if (type(execution_contract) is not dict or set(execution_contract) != required_contract
                or any(type(execution_contract[key]) is not str or not execution_contract[key]
                       for key in required_contract - {'media'})
                or type(execution_contract['media']) is not bool):
            raise ValueError('exact family/model/backend/arithmetic/source/persisted/media contract required')
        execution_contract = dict(execution_contract)
    manifest = dict(schema='openagent.wholemodel-validation-input.v1',offline_root=str(offline_root),
        continuation_restore=continuation_restore,
        seed=seed,tokens=ids,generation_steps=generation_steps,chunk_schedules=schedules,
        tolerances=dict(logits=logits_tolerance,cache=cache_tolerance),cross_evidence=list(routing_evidence),
        oracle=dict(factory=oracle_factory,artifacts=list(oracle_artifacts),options=oracle_options or {}),
        candidate=dict(factory=candidate_factory,artifacts=list(candidate_artifacts),options=candidate_options or {}))
    for role in ('oracle','candidate'):
        kinds = set()
        paths = set()
        for row in manifest[role]['artifacts']:
            if type(row) is not dict or set(row)!={'kind','path','sha256'}:
                raise ValueError('complete exact artifact pin records required')
            pin = row['sha256']
            if type(pin) is not str or len(pin)!=64 or any(c not in '0123456789abcdef' for c in pin):
                raise ValueError('lowercase SHA-256 artifact pins required')
            if type(row['path']) is not str or not row['path'] or row['path'] in paths:
                raise ValueError('unique artifact paths required')
            kinds.add(row['kind'])
            paths.add(row['path'])
        if not {'source','config','weights','tokenizer','dependency'}<=kinds:
            raise ValueError('complete artifact categories required for '+role)
    if execution_contract is not None:
        # Only canonicalize records after their shape and required fields have
        # been admitted, so malformed input uses the explicit contract errors
        # above rather than leaking KeyError from the binding construction.
        # This remains an integrity join, not a runtime receipt or proof that
        # the artifacts implement the execution row.
        pin_records = {
            role: sorted((dict(row) for row in manifest[role]['artifacts']),
                         key=lambda row: (row['kind'], row['path'], row['sha256']))
            for role in ('oracle', 'candidate')
        }
        bound_payload = dict(contract=execution_contract,
                             oracle_factory=oracle_factory,
                             candidate_factory=candidate_factory,
                             artifact_pins=pin_records)
        binding_sha256 = hashlib.sha256(json.dumps(
            bound_payload, sort_keys=True, separators=(',', ':'),
            allow_nan=False).encode('utf-8')).hexdigest()
        # Adapters must still echo the exact row in metadata before comparison.
        manifest['execution_binding'] = dict(
            contract=execution_contract, oracle_factory=oracle_factory,
            candidate_factory=candidate_factory, artifact_pins=pin_records,
            binding_sha256=binding_sha256, receipt_required=True)
    for kind in ('config','weights','tokenizer'):
        pins = lambda role:sorted(row['sha256'] for row in manifest[role]['artifacts'] if row['kind']==kind)
        if pins('oracle')!=pins('candidate'):
            raise ValueError('oracle and candidate checkpoint artifact pins differ')
    return json.loads(json.dumps(manifest,allow_nan=False))


def write_configuration(manifest, destination):
    """Atomically publish a prepared case; do not execute its commands."""
    path = _path(destination)
    path.parent.mkdir(parents=True,exist_ok=True)
    raw = (json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if len(raw)>64*1024**2:
        raise MemoryError('comparison configuration exceeds publication admission')
    descriptor,name = tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
    temporary = Path(name)
    try:
        try:
            output=os.fdopen(descriptor,'wb')
        except BaseException as primary:
            try:
                os.close(descriptor)
            except BaseException as cleanup:
                primary.cleanup=lambda:os.close(descriptor)
                primary.add_note('comparison configuration descriptor cleanup retained: '+repr(cleanup))
            raise
        with output:
            _write(output,raw)
            output.flush()
            os.fsync(output.fileno())
        _path(path);os.replace(temporary,path)
    finally:
        _remove_staging(temporary,sys.exc_info()[1])
    return str(path.resolve())


__all__ = ['ArtifactSelection','pin_artifacts','comparison_configuration','write_configuration']
