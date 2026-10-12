"""Assemble existing pinned comparisons into a multi-case admission receipt.

This operator performs bounded artifact admission only. It never runs a model,
creates observations, changes tolerances or promotes a failed comparison.
"""
import copy
import hashlib
import hmac
import json
import os
from pathlib import Path
import tempfile

from .exact_admission import K3ExactEvidence,_pairs
from ..session_recovery import _path
from ..checkpoint_recovery import _sync_directory


def _read(path,pin,maximum):
    if type(pin) is not str or len(pin)!=64 or any(c not in '0123456789abcdef' for c in pin):
        raise ValueError('explicit lowercase comparison artifact SHA256 required')
    path=_path(path);before=path.stat()
    if not path.is_file() or not 0<before.st_size<=maximum:
        raise ValueError('bounded regular recorded comparison artifact required')
    stamp=lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
    with path.open('rb') as stream:
        if stamp(os.fstat(stream.fileno()))!=stamp(before):
            raise ValueError('recorded comparison artifact changed during opening')
        raw=stream.read(maximum+1)
        if stamp(os.fstat(stream.fileno()))!=stamp(before):
            raise ValueError('recorded comparison artifact changed during reading')
    after=path.stat()
    if stamp(before)!=stamp(after) or len(raw)!=before.st_size:
        raise ValueError('recorded comparison artifact changed during admission')
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(),pin):
        raise ValueError('recorded comparison differs from the explicit artifact pin')
    return json.loads(raw,object_pairs_hook=_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))),len(raw)


def assemble_media_evidence(text_evidence,destination,*,text_sha256,media_reports,
        max_bytes=16*1024**2):
    """media_reports contains ordered (path, SHA256) pairs for existing reports.

    All reports must match the text evidence's execution/arithmetic identities
    and original source pins, and contain complete zero-error observations.
    Per-case continuation coverage is derived from recorded observations.
    """
    if type(max_bytes) is not int or not 1<=max_bytes<=256*1024**2:
        raise ValueError('explicit bounded evidence assembly admission required')
    if type(media_reports) is not tuple or not 1<=len(media_reports)<=256 or any(
            type(row) is not tuple or len(row)!=2 for row in media_reports):
        raise ValueError('one to 256 explicitly pinned recorded media comparisons required')
    base,size=_read(text_evidence,text_sha256,max_bytes)
    if size*8>max_bytes:raise MemoryError('original evidence decoding exceeds assembly admission')
    if type(base) is not dict or base.get('schema')!='openagent.k3-exact-evidence.v2':
        raise ValueError('original pinned v2 text evidence required')
    # Validate the complete original text receipt before incorporating reports.
    K3ExactEvidence(text_evidence,expected_sha256=text_sha256,max_bytes=max_bytes)
    reports=[]
    for path,pin in media_reports:
        report,amount=_read(path,pin,max_bytes-size)
        size+=amount
        if size*8>max_bytes:raise MemoryError('evidence decoding and publication staging exceed admission')
        reports.append(report)
    value=copy.deepcopy(base);value['schema']='openagent.k3-exact-evidence.v4';value['media_reports']=reports
    encoded=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if size*8+len(encoded)*3>max_bytes:raise MemoryError('complete evidence publication exceeds admission')
    digest=hashlib.sha256(encoded).hexdigest();path=_path(destination);temporary=None;primary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
            temporary=Path(stream.name);stream.write(encoded);stream.flush();os.fsync(stream.fileno())
        admitted=K3ExactEvidence(temporary,expected_sha256=digest,max_bytes=max_bytes)
        cases=admitted.media_cases
        _path(path);os.replace(temporary,path);temporary=None;_sync_directory(path.parent)
        return dict(path=str(path),bytes=len(encoded),sha256=digest,identity=admitted.identity,
            cases=[dict(fixture_sha256=row['fixture_sha256'],prefill_tokens=row['prefill_tokens'],
                context_tokens=row['context_tokens']) for row in cases],
            observations_created=False,numerical_parity_verified=False,platform_certified=False)
    except BaseException as failure:
        primary=failure;raise
    finally:
        if temporary is not None:
            try:temporary.unlink(missing_ok=True)
            except BaseException as cleanup:
                if primary is None:raise
                primary.uncommitted_staging=(str(temporary),)
                primary.add_note('exact-evidence staging cleanup retained: '+repr(cleanup))


__all__=['assemble_media_evidence']
