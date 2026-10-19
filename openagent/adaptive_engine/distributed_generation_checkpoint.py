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
import shutil

from .generation_checkpoint_files import AtomicGenerationCheckpoint
from .session_recovery import _path
from .segment_snapshot import _pairs
from .distributed_execution import _failure_summary
from .owned_file_descriptor import close_after_failure,fdopen_owned


def signature(boundary):
    value=dict(tokens=list(boundary.tokens),dtype=boundary.dtype,
        logits_sha256=hashlib.sha256(boundary.logits).hexdigest(),
        sampler_sha256=hashlib.sha256(boundary.sampler).hexdigest(),
        policy=None if boundary.policy is None else asdict(boundary.policy),
        stops=list(boundary.stops),stopped=boundary.stopped,
        stop_prefixes=[list(prefix) for prefix in getattr(boundary, 'stop_prefixes', ())],
        vocabulary=boundary.vocabulary,context=boundary.context,
        application_identity=boundary.application_identity,
        rng=getattr(boundary,'rng',None),
        proposal_rng=getattr(boundary,'proposal_rng',None))
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
                primary.add_note('distributed manifest staging cleanup remains pending at '+str(temporary)+': '+
                                 _failure_summary(failure))
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
    except BaseException as primary:
        close_after_failure(descriptor,primary,owner='distributed generation manifest')
        raise
    stream=fdopen_owned(descriptor,'rb',owner='distributed generation manifest')
    with stream:
        raw=stream.read(65537)
        if stamp(os.fstat(stream.fileno()))!=stamp(before):
            raise ValueError('distributed manifest changed during reading')
    if stamp(_path(path).stat())!=stamp(before) or len(raw)!=before.st_size:
        raise ValueError('distributed manifest changed during admission')
    return path,raw


def _valid_reservation_record(rank, record, *, index):
    if rank != 0:
        return record is None
    if type(record) is not tuple or len(record) != 2:
        return False
    name, error = record
    if error is not None:
        return (name is None and type(error) is str and 0 < len(error) <= 4096)
    prefix=f'boundary-{index}-'
    suffix=name[len(prefix):] if type(name) is str and name.startswith(prefix) else ''
    return (len(suffix) == 32 and
            all(character in '0123456789abcdef' for character in suffix))


def _valid_collective_error(value):
    return value is None or (type(value) is str and 0 < len(value) <= 4096)


def _valid_generation_record(row, *, max_rank_bytes):
    if type(row) is not tuple or len(row) != 3:
        return False
    observed, failure, item = row
    if failure is not None:
        return (observed is None and item is None and type(failure) is str and
                0 < len(failure) <= 4096)
    if (type(observed) is not str or len(observed) != 64 or
            any(char not in '0123456789abcdef' for char in observed) or
            type(item) is not dict or
            set(item) != {'bytes', 'sha256', 'execution_identity'} or
            type(item['bytes']) is not int or item['bytes'].bit_length() > 63 or
            not 128 <= item['bytes'] <= max_rank_bytes or
            type(item['sha256']) is not str or len(item['sha256']) != 64 or
            any(char not in '0123456789abcdef' for char in item['sha256']) or
            type(item['execution_identity']) is not str or
            not 0 < len(item['execution_identity']) <= 8192 or
            '\x00' in item['execution_identity']):
        return False
    return True


def _poison_coordinator(owner):
    # These result-envelope failures make subsequent control schedules
    # untrustworthy. The coordinator is caller-owned and must be restarted.
    try:
        owner._failed = True
    except BaseException:
        pass


class CollectiveGenerationSink:
    def __init__(self,coordinator,directory,*,max_rank_bytes):
        if type(max_rank_bytes) is not int or not 128<=max_rank_bytes<=2**63-1:
            raise ValueError('bounded complete per-rank generation frame admission required')
        if type(coordinator.world_size) is not int or not 1<=coordinator.world_size<=64:
            raise ValueError('one to 64 generation publication ranks required')
        self.coordinator=coordinator;self.directory=_path(directory)
        self.max_rank_bytes=max_rank_bytes;self.index=0

    @staticmethod
    def _quarantine_uncommitted(path, *, manifest_written=False):
        """Remove only a just-created boundary known to lack a commit pointer."""
        path=_path(path)
        try:
            # Once publication of latest.json has begun, its atomic rename may
            # have committed even if the following directory fsync failed.
            # Before that point the unique generation directory cannot be
            # referenced by the commit pointer and is safe to remove, even if
            # its private manifest was already written.
            if path.is_dir() and not manifest_written:
                shutil.rmtree(path)
        except BaseException as error:
            return error
        return None

    def __call__(self,boundary,write_frame):
        owner=self.coordinator;receipt=None;error=None;logical=None
        # A sink is recreated for each request. Its local index cannot name an
        # immutable recovery boundary across requests or replacement owners.
        reservation=None;directory=None
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
                reservation=(None,_failure_summary(failure))
        reservations=[None]*owner.world_size
        owner.distributed.all_gather_object(reservations,reservation,group=owner.group)
        malformed=[rank for rank,row in enumerate(reservations)
                   if not _valid_reservation_record(rank,row,index=self.index)]
        if malformed:
            cleanup_error=None
            if owner.rank==0 and directory is not None:
                cleanup_error=self._quarantine_uncommitted(directory)
            cleanup=[None]*owner.world_size
            owner.distributed.all_gather_object(cleanup,
                None if cleanup_error is None else _failure_summary(cleanup_error),group=owner.group)
            _poison_coordinator(owner)
            if any(not _valid_collective_error(value) for value in cleanup):
                raise RuntimeError('malformed reservation and cleanup agreement records')
            cleanup_failures=[value for value in cleanup if value is not None]
            detail=('malformed distributed generation reservation record from rank(s) '+
                    ','.join(str(rank) for rank in malformed))
            if cleanup_failures:
                detail+='; reservation cleanup failed: '+'; '.join(cleanup_failures)
            raise RuntimeError(detail)
        name,reservation_error=reservations[0]
        if reservation_error is not None:
            cleanup_error=None
            if owner.rank==0 and directory is not None:
                cleanup_error=self._quarantine_uncommitted(directory)
            cleanup=[None]*owner.world_size
            owner.distributed.all_gather_object(cleanup,
                None if cleanup_error is None else _failure_summary(cleanup_error),group=owner.group)
            if any(not _valid_collective_error(value) for value in cleanup):
                _poison_coordinator(owner)
                raise RuntimeError('malformed reservation cleanup agreement record')
            cleanup_failures=[value for value in cleanup if value is not None]
            if cleanup_failures:
                _poison_coordinator(owner)
                raise RuntimeError('distributed generation reservation failed: '+reservation_error+
                    '; cleanup: '+'; '.join(cleanup_failures))
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
            error=_failure_summary(failure)
        records=[None]*owner.world_size
        owner.distributed.all_gather_object(records,(logical,error,receipt),group=owner.group)
        malformed = [rank for rank, row in enumerate(records)
                     if not _valid_generation_record(row,max_rank_bytes=self.max_rank_bytes)]
        for rank in malformed:
            records[rank] = (None, 'malformed generation publication record', None)
        if malformed:
            _poison_coordinator(owner)
        failures=[f'rank {rank}: {row[1]}' for rank,row in enumerate(records) if row[1] is not None]
        if (failures or records[0][0] is None or
                any(row[0] != records[0][0] for row in records)):
            cleanup_error=None
            if owner.rank==0:
                cleanup_error=self._quarantine_uncommitted(directory)
            cleanup=[None]*owner.world_size
            owner.distributed.all_gather_object(cleanup,
                None if cleanup_error is None else _failure_summary(cleanup_error),group=owner.group)
            if any(not _valid_collective_error(value) for value in cleanup):
                _poison_coordinator(owner)
                raise RuntimeError('malformed generation cleanup agreement record')
            if any(value is not None for value in cleanup):
                failures.append('uncommitted boundary cleanup failed: '+
                    next(value for value in cleanup if value is not None))
            raise RuntimeError('distributed generation capture failed: '+
                ('; '.join(failures) if failures else 'logical generation boundaries differ'))
        error=None;latest_publication_started=False
        if owner.rank==0:
            try:
                if any(row[2] is None for row in records):
                    raise ValueError('complete rank generation receipts required')
                manifest=dict(version=1,kind='generation',world_size=owner.world_size,
                    boundary_sha256=logical,context_tokens=boundary.position,
                    application_identity=boundary.application_identity,
                    ranks=[dict(rank=rank,filename=f'rank-{rank}.generation',
                        bytes=row[2]['bytes'],sha256=row[2]['sha256'],
                        identity=row[2]['execution_identity']) for rank,row in enumerate(records)])
                _atomic_manifest(directory/'manifest.json',manifest)
                # This pointer contains an exact relative committed manifest and
                # its digest. The immutable boundary manifest is the recovery pin.
                _,raw=read_manifest(directory/'manifest.json')
                latest_publication_started=True
                _atomic_manifest(self.directory/'latest.json',dict(version=1,
                    manifest=f'{name}/manifest.json',
                    sha256=hashlib.sha256(raw).hexdigest()))
            except BaseException as failure:
                error=_failure_summary(failure)
        errors=[None]*owner.world_size
        owner.distributed.all_gather_object(errors,error,group=owner.group)
        malformed_errors=[rank for rank,value in enumerate(errors)
                          if not _valid_collective_error(value)]
        if malformed_errors:
            _poison_coordinator(owner)
            errors=[value if _valid_collective_error(value) else
                    f'malformed generation manifest agreement record from rank {rank}'
                    for rank,value in enumerate(errors)]
        if any(value is not None for value in errors):
            cleanup_error=None
            if owner.rank==0:
                cleanup_error=self._quarantine_uncommitted(
                    directory,manifest_written=latest_publication_started)
            cleanup=[None]*owner.world_size
            owner.distributed.all_gather_object(cleanup,
                None if cleanup_error is None else _failure_summary(cleanup_error),group=owner.group)
            if any(not _valid_collective_error(value) for value in cleanup):
                _poison_coordinator(owner)
                raise RuntimeError('malformed manifest cleanup agreement record')
            if any(value is not None for value in cleanup):
                errors.append('uncommitted boundary cleanup failed: '+
                    next(value for value in cleanup if value is not None))
            raise RuntimeError('distributed generation manifest publication failed: '+
                next(value for value in errors if value is not None))
        self.index+=1


def admit_manifest(path,*,world_size,max_rank_bytes,context,application_identity=None):
    if (type(world_size) is not int or not 1<=world_size<=64 or
            type(max_rank_bytes) is not int or not 128<=max_rank_bytes<=2**63-1 or
            type(context) is not int or not 1<=context<=2**63-1):
        raise ValueError('bounded rank, generation frame and context admissions required')
    path,raw=read_manifest(path)
    value=json.loads(raw,object_pairs_hook=_pairs,
        parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))
    def pin(text):return type(text) is str and len(text)==64 and all(c in '0123456789abcdef' for c in text)
    if (type(value) is not dict or set(value)!={'version','kind','world_size','boundary_sha256','context_tokens','application_identity','ranks'}
            or type(value['version']) is not int or value['version']!=1 or value['kind']!='generation'
            or type(value['world_size']) is not int or value['world_size']!=world_size
            or type(value['context_tokens']) is not int or not 1<=value['context_tokens']<=context
            or not pin(value['boundary_sha256']) or
            type(value['application_identity']) is not str or
            not 0<len(value['application_identity'])<=8192 or
            '\x00' in value['application_identity'] or
            (application_identity is not None and value['application_identity'] != application_identity) or
            type(value['ranks']) is not list
            or len(value['ranks'])!=world_size):
        raise ValueError('distributed generation manifest geometry differs')
    for rank,row in enumerate(value['ranks']):
        if (type(row) is not dict or set(row)!={'rank','filename','bytes','sha256','identity'}
                or type(row['rank']) is not int or row['rank']!=rank
                or row['filename']!=f'rank-{rank}.generation'
                or type(row['bytes']) is not int or not 128<=row['bytes']<=max_rank_bytes
                or not pin(row['sha256']) or type(row['identity']) is not str or
                not 0<len(row['identity'])<=8192 or '\x00' in row['identity']):
            raise ValueError('distributed generation rank receipt differs')
    return path,value


def restore_rank(coordinator,session,command):
    if (type(command) is not dict or
            type(command.get('directory')) is not str or
            type(command.get('application_identity')) is not str or
            not 0<len(command['application_identity'])<=8192 or
            '\x00' in command['application_identity'] or
            type(command.get('recovery_max_bytes')) is not int or
            command['recovery_max_bytes']<128):
        raise ValueError('explicit distributed restore command identity and budget required')
    manifest=command.get('manifest')
    if (type(manifest) is not dict or manifest.get('kind')!='generation' or
            type(manifest.get('world_size')) is not int or
            manifest['world_size']!=coordinator.world_size or
            type(manifest.get('context_tokens')) is not int or
            manifest['context_tokens']<=0 or
            type(manifest.get('application_identity')) is not str or
            manifest['application_identity'] != command['application_identity'] or
            type(manifest.get('boundary_sha256')) is not str or
            len(manifest['boundary_sha256'])!=64 or
            any(char not in '0123456789abcdef' for char in manifest['boundary_sha256']) or
            type(manifest.get('ranks')) is not list or
            len(manifest['ranks'])!=coordinator.world_size):
        raise ValueError('foreign or stale distributed generation manifest')
    receipt=manifest['ranks'][coordinator.rank]
    if (type(receipt) is not dict or
            type(receipt.get('rank')) is not int or receipt['rank']!=coordinator.rank or
            type(receipt.get('filename')) is not str or
            receipt['filename']!=f'rank-{coordinator.rank}.generation' or
            type(receipt.get('bytes')) is not int or receipt['bytes']<128 or
            type(receipt.get('sha256')) is not str or len(receipt['sha256'])!=64 or
            any(char not in '0123456789abcdef' for char in receipt['sha256']) or
            type(receipt.get('identity')) is not str or
            not 0<len(receipt['identity'])<=8192 or '\x00' in receipt['identity']):
        raise ValueError('foreign distributed rank receipt')
    root_path=_path(command['directory'])
    try:
        root_info=root_path.lstat()
    except OSError as error:
        raise ValueError('stable committed distributed generation directory required') from error
    if not root_path.is_dir() or root_path.is_symlink() or not stat.S_ISDIR(root_info.st_mode):
        raise ValueError('stable committed distributed generation directory required')
    root=root_path.resolve(strict=True)
    # Re-admit the immutable on-disk manifest before trusting any caller
    # supplied receipt.  A locally valid rank entry is insufficient when the
    # command carries a stale or foreign view of the other ranks.
    _, committed = admit_manifest(
        root / 'manifest.json',
        world_size=coordinator.world_size,
        max_rank_bytes=command['recovery_max_bytes'],
        context=manifest['context_tokens'],
        application_identity=command['application_identity'],
    )
    if committed != manifest:
        raise ValueError('distributed generation manifest differs from committed directory')
    path=(root/receipt['filename']).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError('distributed rank frame escapes the committed manifest directory')
    if (type(receipt) is not dict or receipt.get('rank')!=coordinator.rank or
            receipt.get('filename')!=f'rank-{coordinator.rank}.generation'):
        raise ValueError('foreign distributed rank receipt')
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
