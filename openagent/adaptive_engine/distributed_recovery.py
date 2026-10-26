"""Bounded replacement-rank recovery from explicitly pinned committed caches.

The application supplies all rendezvous URLs, devices and factories. Recovery
launches fresh rank owners; it never reuses a poisoned process group. Resumed
application work may run more than once if a worker fails: the caller owns any
external side-effect deduplication and checkpoint publication policy.
"""
from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
from types import MappingProxyType

from .segment_snapshot import _pairs


def _pin(value):
    return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _failure_summary(failure):
    """Describe cleanup failures without invoking exception formatting hooks."""
    try:
        name = type(failure).__name__
    except BaseException:
        name = 'BaseException'
    if type(name) is not str or not name or len(name) > 128:
        name = 'BaseException'
    return name + ': failure details omitted'


@dataclass(frozen=True)
class RankRecoveryCheckpoint:
    key: str
    manifest_path: str
    manifest_sha256: str
    max_rank_bytes: int

    def admit(self,world_size):
        if type(world_size) is not int or not 1<=world_size<=64:
            raise ValueError('one to 64 explicitly admitted recovery ranks required')
        if (type(self.key) is not str or not 0<len(self.key)<=256 or '\x00' in self.key or
                type(self.manifest_path) is not str or not 0<len(self.manifest_path)<=4096 or not _pin(self.manifest_sha256) or
                type(self.max_rank_bytes) is not int or not 88<=self.max_rank_bytes<=2**63-1):
            raise ValueError('explicit named and pinned distributed recovery checkpoint required')
        original=Path(self.manifest_path)
        if not original.is_absolute() or original.is_symlink():
            raise ValueError('absolute regular committed distributed manifest required')
        from .distributed_generation_checkpoint import read_manifest
        path,raw=read_manifest(original)
        if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(),self.manifest_sha256):
            raise ValueError('distributed recovery manifest differs from operator pin')
        value=json.loads(raw,object_pairs_hook=_pairs,
            parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))
        if (type(value) is not dict or set(value)!={'version','world_size','context_tokens','history_sha256','model_identity','ranks'}
                or type(value['version']) is not int or value['version']!=1
                or type(value['world_size']) is not int or value['world_size']!=world_size
                or type(value['context_tokens']) is not int or not 0<=value['context_tokens']<=2**63-1
                or not _pin(value['history_sha256']) or type(value['model_identity']) is not str
                or not 0<len(value['model_identity'])<=4096 or '\x00' in value['model_identity']
                or type(value['ranks']) is not list
                or len(value['ranks'])!=world_size):
            raise ValueError('committed distributed rank geometry differs')
        names=set()
        for rank,row in enumerate(value['ranks']):
            if (type(row) is not dict or set(row)!={'rank','filename','bytes','sha256','identity'}
                    or type(row['rank']) is not int or row['rank']!=rank
                    or type(row['filename']) is not str or not row['filename']
                    or row['filename']!=f'rank-{rank}.cache' or row['filename'] in names
                    or row['filename'] in ('.','..') or ':' in row['filename']
                    or type(row['bytes']) is not int or not 88<=row['bytes']<=self.max_rank_bytes
                    or not _pin(row['sha256']) or type(row['identity']) is not str
                    or not 0<len(row['identity'])<=4096 or '\x00' in row['identity']):
                raise ValueError('exact per-rank checkpoint receipt required')
            names.add(row['filename'])
        return path,value


@dataclass(frozen=True)
class RankGenerationRecoveryCheckpoint(RankRecoveryCheckpoint):
    application_identity: str=''

    def admit(self,world_size):
        from .distributed_generation_checkpoint import admit_manifest,read_manifest
        if type(world_size) is not int or not 1<=world_size<=64:
            raise ValueError('one to 64 explicitly admitted recovery ranks required')
        if (type(self.key) is not str or not 0<len(self.key)<=256 or '\x00' in self.key or
                type(self.manifest_path) is not str or not 0<len(self.manifest_path)<=4096 or
                not Path(self.manifest_path).is_absolute() or not _pin(self.manifest_sha256) or
                type(self.max_rank_bytes) is not int or not 128<=self.max_rank_bytes<=2**63-1 or
                type(self.application_identity) is not str or
                not 0<len(self.application_identity)<=8192 or
                '\x00' in self.application_identity):
            raise ValueError('explicit named, pinned distributed generation checkpoint required')
        path,value=admit_manifest(self.manifest_path,world_size=world_size,
            max_rank_bytes=self.max_rank_bytes,context=2**63-1,
            application_identity=self.application_identity)
        _,raw=read_manifest(path)
        if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(),self.manifest_sha256):
            raise ValueError('distributed generation manifest differs from operator pin')
        parsed=json.loads(raw,object_pairs_hook=_pairs,
            parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))
        if parsed!=value:raise ValueError('generation manifest changed between geometry and pin admission')
        return path,value


@dataclass(frozen=True)
class _RecoveryDriver:
    checkpoints: tuple
    driver: object
    world_size: int
    attempt: int

    def __call__(self,coordinator):
        # Workers enter serve() before root broadcasts any restore commands.
        # No application request is dispatched until every restore succeeds.
        restored={}
        opened=[]
        try:
            for checkpoint in self.checkpoints:
                path,before=checkpoint.admit(self.world_size)
                session=coordinator.open_session()
                # DistributedExecutionService exposes positive integer
                # handles. Validate that contract and compare by value: two
                # equal integer handles are not guaranteed to be the same
                # Python object (notably once they exceed the small-int cache).
                if (type(session) is not int or session <= 0 or
                        any(session == prior for prior in opened)):
                    raise RuntimeError('replacement coordinator returned a foreign or reused session handle')
                opened.append(session)
                if type(checkpoint) is RankGenerationRecoveryCheckpoint:
                    coordinator.restore_generation_checkpoint(session,path,max_rank_bytes=checkpoint.max_rank_bytes,
                        application_identity=checkpoint.application_identity)
                else:
                    coordinator.load_session_cache(session,path,max_rank_bytes=checkpoint.max_rank_bytes)
                _,after=checkpoint.admit(self.world_size)
                if before!=after:
                    raise ValueError('committed manifest changed while replacement ranks restored')
                restored[checkpoint.key]=session
            # A later restore can take time or execute caller-controlled code.
            # Re-pin every checkpoint together at the handoff boundary so a
            # manifest changed after its individual restore check cannot
            # authorize application work on a mixed/stale checkpoint set.
            for checkpoint in self.checkpoints:
                checkpoint.admit(self.world_size)
            return self.driver(coordinator,MappingProxyType(restored),self.attempt)
        except BaseException as primary:
            for session in reversed(tuple(opened)):
                try:
                    coordinator.close_session(session)
                except BaseException as cleanup:
                    # Cleanup exceptions are arbitrary user/application
                    # objects. Calling repr() here can raise and abort this
                    # loop, leaving older opened sessions unclosed while
                    # masking the restore/driver failure that triggered it.
                    try:
                        primary.add_note('replacement recovery session cleanup retained: '+
                                         _failure_summary(cleanup))
                    except BaseException:
                        # A custom BaseException may also reject notes. Keep
                        # attempting teardown for the remaining owners.
                        pass
            raise


def launch_recovered_distributed_execution(service_factory,root_driver,*,checkpoints,
        rendezvous_urls,world_size,projection_backend,device_ids=(),timeout_seconds=120,
        max_control_bytes=1024*1024,max_recovery_boundary_bytes=64*1024**2):
    """Restore fresh ranks and retry worker failure using bounded explicit epochs.

    root_driver(coordinator, restored_sessions, attempt) receives logical handles
    only after all checkpoint restorations reach rank consensus. Each supplied
    rendezvous URL represents one attempt, and must identify fresh resources.
    Manifest/rank files remain caller-owned immutable committed artifacts.
    """
    if type(world_size) is not int or not 1<=world_size<=64:
        raise ValueError('one to 64 replacement ranks required')
    if (type(checkpoints) is not tuple or not 1<=len(checkpoints)<=256 or
            any(type(row) not in (RankRecoveryCheckpoint,RankGenerationRecoveryCheckpoint) for row in checkpoints)
            or len({row.key for row in checkpoints})!=len(checkpoints)):
        raise ValueError('bounded unique named recovery checkpoints required')
    if (type(rendezvous_urls) is not tuple or not 1<=len(rendezvous_urls)<=16 or
            any(type(url) is not str or len(url)>4096 or not url.startswith(('tcp://','file://')) for url in rendezvous_urls)
            or len(set(rendezvous_urls))!=len(rendezvous_urls)):
        raise ValueError('one to 16 distinct explicit replacement rendezvous URLs required')
    if not callable(service_factory) or not callable(root_driver):
        raise TypeError('explicit importable replacement factory and recovered driver required')
    for checkpoint in checkpoints:checkpoint.admit(world_size)
    from .distributed_launcher import launch_distributed_execution
    from torch.multiprocessing.spawn import ProcessRaisedException,ProcessExitedException
    failures=[]
    for attempt,url in enumerate(rendezvous_urls):
        # The previous spawn/join has terminated all workers before this epoch.
        # Re-admit immutable manifests rather than letting a retry select a
        # different continuation or silently replay already submitted prompts.
        for checkpoint in checkpoints:checkpoint.admit(world_size)
        try:
            launch_distributed_execution(service_factory,
                _RecoveryDriver(checkpoints,root_driver,world_size,attempt),world_size=world_size,
                init_method=url,projection_backend=projection_backend,device_ids=device_ids,
                timeout_seconds=timeout_seconds,max_control_bytes=max_control_bytes,
                max_recovery_boundary_bytes=max_recovery_boundary_bytes)
            return dict(schema='openagent.distributed-recovery-receipt.v1',attempt=attempt,
                world_size=world_size,checkpoints=[dict(key=row.key,manifest_sha256=row.manifest_sha256)
                    for row in checkpoints],source_only=False,numerical_parity_verified=False,platform_certified=False)
        except (ProcessRaisedException,ProcessExitedException) as failure:
            failures.append(failure)
    error=RuntimeError('replacement rank recovery exhausted all admitted rendezvous epochs')
    for attempt,failure in enumerate(failures):
        error.add_note('recovery attempt '+str(attempt)+': '+_failure_summary(failure))
    raise error from failures[-1]


__all__=['RankRecoveryCheckpoint','RankGenerationRecoveryCheckpoint','launch_recovered_distributed_execution']
