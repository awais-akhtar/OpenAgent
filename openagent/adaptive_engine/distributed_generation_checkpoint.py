"""Collective publication of complete rank generation frames on shared storage.

Every frame retains physical cache, next logits and sampler state. A manifest
is published only after every rank writes the same logical boundary. Partial
directories remain uncommitted; publication never selects a partial rank set.
"""
from dataclasses import asdict
import hashlib
import json
import os
import stat
from pathlib import Path
import tempfile
import uuid

from .generation_checkpoint_files import AtomicGenerationCheckpoint
from .session_recovery import _path
from .segment_snapshot import _pairs


def signature(boundary):
    value=dict(tokens=list(boundary.tokens),dtype=boundary.dtype,
        logits_sha256=hashlib.sha256(boundary.logits).hexdigest(),
        sampler_sha256=hashlib.sha256(boundary.sampler).hexdigest(),
        policy=None if boundary.policy is None else asdict(boundary.policy),
        stops=list(boundary.stops),stopped=boundary.stopped,
        vocabulary=boundary.vocabulary,context=boundary.context,
        application_identity=boundary.application_identity)
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        allow_nan=False).encode()).hexdigest()


def _atomic_manifest(path,value):
    from .checkpoint_recovery import _sync_directory
    encoded=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(encoded)>65536:raise MemoryError('distributed generation manifest exceeds admission')
    path=_path(path);temporary=None;primary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
            temporary=Path(stream.name)
            stream.write(encoded);stream.flush();os.fsync(stream.fileno())
        _path(path);os.replace(temporary,path);temporary=None
        _sync_directory(path.parent)
    except BaseException as failure:
        primary=failure
        raise
    finally:
        if temporary is not None:
            try:temporary.unlink(missing_ok=True)
            except BaseException as failure:
                if primary is None:raise
                primary.add_note('distributed manifest staging cleanup remains pending at '+str(temporary)+': '+str(failure))
                primary.cleanup=lambda path=temporary:path.unlink(missing_ok=True)


def read_manifest(path):
    """Read one bounded regular manifest from a stable descriptor and path."""
    path=_path(path)
    def stamp(info):return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns
    before=path.stat()
    if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=65536:
        raise ValueError('bounded regular distributed manifest required')
    descriptor=os.open(path,os.O_RDONLY|getattr(os,'O_BINARY',0)|
        getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))
    try:
        opened=os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or stamp(opened)!=stamp(before):
            raise ValueError('distributed manifest changed during opening')
        stream=os.fdopen(descriptor,'rb')
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        raw=stream.read(65537)
        if stamp(os.fstat(stream.fileno()))!=stamp(before):
            raise ValueError('distributed manifest changed during reading')
    if stamp(_path(path).stat())!=stamp(before) or len(raw)!=before.st_size:
        raise ValueError('distributed manifest changed during admission')
    return path,raw


class CollectiveGenerationSink:
    def __init__(self,coordinator,directory,*,max_rank_bytes):
        if type(max_rank_bytes) is not int or not 128<=max_rank_bytes<=2**63-1:
            raise ValueError('bounded complete per-rank generation frame admission required')
        if type(coordinator.world_size) is not int or not 1<=coordinator.world_size<=64:
            raise ValueError('one to 64 generation publication ranks required')
        self.coordinator=coordinator;self.directory=_path(directory)
        self.max_rank_bytes=max_rank_bytes;self.index=0

    def __call__(self,boundary,write_frame):
        owner=self.coordinator;receipt=None;error=None;logical=None
        # A sink is recreated for each request. Its local index cannot name an
        # immutable recovery boundary across requests or replacement owners.
        reservation=None
        if owner.rank==0:
            try:
                self.directory.mkdir(parents=True,exist_ok=True)
                name=f'boundary-{self.index}-{uuid.uuid4().hex}'
                directory=_path(self.directory/name)
                directory.mkdir(exist_ok=False)
                from .checkpoint_recovery import _sync_directory
                _sync_directory(self.directory)
                reservation=(name,None)
            except BaseException as failure:
                reservation=(None,(str(failure) or type(failure).__name__)[:4096])
        reservations=[None]*owner.world_size
        owner.distributed.all_gather_object(reservations,reservation,group=owner.group)
        name,reservation_error=reservations[0]
        if reservation_error is not None:
            raise RuntimeError('distributed generation directory reservation failed: '+reservation_error)
        try:
            directory=_path(self.directory/name)
            logical=signature(boundary)
            store=AtomicGenerationCheckpoint(directory/f'rank-{owner.rank}.generation',
                max_bytes=self.max_rank_bytes)
            try:
                receipt=store(boundary,write_frame)
            finally:store.close()
        except BaseException as failure:
            error=(str(failure) or type(failure).__name__)[:4096]
        records=[None]*owner.world_size
        owner.distributed.all_gather_object(records,(logical,error,receipt),group=owner.group)
        failures=[f'rank {rank}: {row[1]}' for rank,row in enumerate(records) if row[1] is not None]
        if failures or any(row[0]!=records[0][0] for row in records):
            raise RuntimeError('distributed generation capture failed: '+
                ('; '.join(failures) if failures else 'logical generation boundaries differ'))
        error=None
        if owner.rank==0:
            try:
                manifest=dict(version=1,kind='generation',world_size=owner.world_size,
                    boundary_sha256=logical,context_tokens=boundary.position,
                    ranks=[dict(rank=rank,filename=f'rank-{rank}.generation',
                        bytes=row[2]['bytes'],sha256=row[2]['sha256'],
                        identity=row[2]['execution_identity']) for rank,row in enumerate(records)])
                _atomic_manifest(directory/'manifest.json',manifest)
                # This pointer contains an exact relative committed manifest and
                # its digest. The immutable boundary manifest is the recovery pin.
                _,raw=read_manifest(directory/'manifest.json')
                _atomic_manifest(self.directory/'latest.json',dict(version=1,
                    manifest=f'{name}/manifest.json',
                    sha256=hashlib.sha256(raw).hexdigest()))
            except BaseException as failure:
                error=(str(failure) or type(failure).__name__)[:4096]
        errors=[None]*owner.world_size
        owner.distributed.all_gather_object(errors,error,group=owner.group)
        if any(value is not None for value in errors):
            raise RuntimeError('distributed generation manifest publication failed: '+
                next(value for value in errors if value is not None))
        self.index+=1


def admit_manifest(path,*,world_size,max_rank_bytes,context):
    if (type(world_size) is not int or not 1<=world_size<=64 or
            type(max_rank_bytes) is not int or not 128<=max_rank_bytes<=2**63-1 or
            type(context) is not int or not 1<=context<=2**63-1):
        raise ValueError('bounded rank, generation frame and context admissions required')
    path,raw=read_manifest(path)
    value=json.loads(raw,object_pairs_hook=_pairs,
        parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))
    def pin(text):return type(text) is str and len(text)==64 and all(c in '0123456789abcdef' for c in text)
    if (type(value) is not dict or set(value)!={'version','kind','world_size','boundary_sha256','context_tokens','ranks'}
            or type(value['version']) is not int or value['version']!=1 or value['kind']!='generation'
            or type(value['world_size']) is not int or value['world_size']!=world_size
            or type(value['context_tokens']) is not int or not 1<=value['context_tokens']<=context
            or not pin(value['boundary_sha256']) or type(value['ranks']) is not list
            or len(value['ranks'])!=world_size):
        raise ValueError('distributed generation manifest geometry differs')
    for rank,row in enumerate(value['ranks']):
        if (type(row) is not dict or set(row)!={'rank','filename','bytes','sha256','identity'}
                or type(row['rank']) is not int or row['rank']!=rank
                or row['filename']!=f'rank-{rank}.generation'
                or type(row['bytes']) is not int or not 128<=row['bytes']<=max_rank_bytes
                or not pin(row['sha256']) or type(row['identity']) is not str or not 0<len(row['identity'])<=8192):
            raise ValueError('distributed generation rank receipt differs')
    return path,value


def restore_rank(coordinator,session,command):
    receipt=command['manifest']['ranks'][coordinator.rank]
    path=_path(Path(command['directory'])/receipt['filename'])
    expected=dict(path=str(path),bytes=receipt['bytes'],sha256=receipt['sha256'])
    store=AtomicGenerationCheckpoint(path,max_bytes=command['recovery_max_bytes'])
    try:
        with store.source(receipt=expected) as source:
            boundary=coordinator.service.restore_generation_checkpoint(session,source,
                max_bytes=command['recovery_max_bytes'],application_identity=command['application_identity'])
        if (boundary.identity!=receipt['identity'] or boundary.position!=command['manifest']['context_tokens']
                or signature(boundary)!=command['manifest']['boundary_sha256']):
            raise ValueError('restored generation boundary differs from committed rank manifest')
        return boundary
    finally:store.close()


__all__=['CollectiveGenerationSink','admit_manifest','restore_rank']
