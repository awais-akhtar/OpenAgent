"""Ordered distributed control for a tensor-parallel execution service.

The caller starts ranks and a trusted Gloo control group separately from any
accelerator projection group. Rank zero drives requests; peers run serve().
Collective failure poisons the coordinator: all ranks must then be restarted.
"""
import threading
import json
import shutil
from dataclasses import asdict
import hashlib
import struct
from pathlib import Path
from uuid import uuid4


def _history_digest(tokens):
    digest = hashlib.sha256()
    for start in range(0,len(tokens),4096):
        chunk = tokens[start:start+4096]
        digest.update(struct.pack('<'+'Q'*len(chunk),*chunk))
    return digest.hexdigest()


class DistributedExecutionService:
    def __init__(self, service, *, control_group, max_control_bytes=1024*1024,
                 max_recovery_boundary_bytes=64*1024**2):
        import torch.distributed as distributed
        from .execution_service import NativeExecutionService
        if type(service) is not NativeExecutionService:
            raise ValueError('rank-local NativeExecutionService required')
        if not distributed.is_initialized():
            raise ValueError('caller-owned initialized control group required')
        if str(distributed.get_backend(control_group)).lower() != 'gloo':
            raise ValueError('CPU Gloo control group required')
        if type(max_control_bytes) is not int or max_control_bytes < 4096:
            raise ValueError('control message budget must be at least 4096 bytes')
        self.service, self.distributed, self.group = service, distributed, control_group
        self.rank = distributed.get_rank(control_group)
        self.world_size = distributed.get_world_size(control_group)
        self.root = distributed.get_global_rank(control_group, 0)
        self.max_control_bytes = max_control_bytes
        if type(max_recovery_boundary_bytes) is not int or max_recovery_boundary_bytes<=0:
            raise ValueError('positive retained generation boundary admission required')
        self.max_recovery_boundary_bytes=max_recovery_boundary_bytes
        self._lock = threading.Lock()
        self._id_lock = threading.Lock()
        self._sessions = {}
        self._generation_boundaries={}
        self._snapshots = {}
        self._next_snapshot = 1
        self._next_id, self._sequence = 1, 0
        self._closed = self._failed = False
        self._active_cancel = None
        self._active_event = None
        policies = [None]*self.world_size
        distributed.all_gather_object(policies, repr((service.limits,max_recovery_boundary_bytes)), group=control_group)
        if any(policy != policies[0] for policy in policies):
            raise ValueError('rank-local execution limits differ')

    def _broadcast(self, command):
        values = [command]
        self.distributed.broadcast_object_list(values, src=self.root, group=self.group)
        return values[0]

    def _execute(self, command):
        from .execution_service import ExecutionCancelled
        error, result, summary = None, None, None
        restore_rollback = None
        prior_generation_boundary = None
        try:
            if (type(command) is not dict or command.get('sequence') != self._sequence or
                    command.get('action') not in ('open', 'generate', 'reset', 'close_session', 'close',
                                                 'snapshot', 'restore', 'release_snapshot', 'save_cache', 'load_cache',
                                                 'resume_generation','restore_generation')):
                raise ValueError('distributed command sequence or action mismatch')
            action = command['action']
            if action == 'open':
                handle = command['session']
                if type(handle) is not int or handle <= 0 or handle in self._sessions:
                    raise ValueError('unique positive distributed session required')
                self._sessions[handle] = self.service.open_session()
                result, summary = handle, ('open', handle)
            elif action == 'close':
                self.service.close()
                self._sessions.clear()
                self._generation_boundaries.clear()
                self._snapshots.clear()
                self._closed = True
                summary = ('close',)
            else:
                session = self._sessions[command['session']]
                if action in ('generate','resume_generation','restore_generation'):
                    import torch
                    from .sampling import SamplingPolicy
                    latch = threading.Event()
                    def boundary(event=None):
                        signatures = [None]*self.world_size
                        signature = None if event is None else tuple(event.get(key) for key in
                            ('event','index','token_id','processed_tokens','total_tokens'))
                        self.distributed.all_gather_object(signatures,signature,group=self.group)
                        if any(value != signatures[0] for value in signatures):
                            raise RuntimeError('distributed generation event schedule or token differs')
                        callback_error = None
                        if self.rank == 0 and event is not None and self._active_event is not None:
                            try:
                                self._active_event(dict(event))
                            except BaseException as failure:
                                callback_error = (str(failure) or type(failure).__name__)[:1024]
                        callback_errors = [None]*self.world_size
                        self.distributed.all_gather_object(callback_errors,callback_error,group=self.group)
                        if any(value is not None for value in callback_errors):
                            raise RuntimeError('distributed root event handler failed: '+
                                               next(value for value in callback_errors if value is not None))
                        requested, cancellation_error = False, None
                        try:
                            requested = bool(self.rank == 0 and self._active_cancel is not None and
                                             self._active_cancel.is_set())
                        except BaseException as failure:
                            cancellation_error = (str(failure) or type(failure).__name__)[:1024]
                        cancellation_errors = [None]*self.world_size
                        self.distributed.all_gather_object(cancellation_errors,cancellation_error,group=self.group)
                        if any(value is not None for value in cancellation_errors):
                            raise RuntimeError('distributed cancellation owner failed: '+
                                               next(value for value in cancellation_errors if value is not None))
                        flag = torch.tensor([int(requested)],dtype=torch.int64,device='cpu')
                        self.distributed.all_reduce(flag,op=self.distributed.ReduceOp.MAX,group=self.group)
                        if int(flag.item()):
                            latch.set()
                    boundary()
                    recovered=None
                    if action in ('resume_generation','restore_generation'):
                        from .distributed_generation_checkpoint import restore_rank
                        recovery_error=None
                        try:
                            if action=='restore_generation':
                                prior_generation_boundary=self._generation_boundaries.get(command['session'])
                                restore_rollback=self.service.snapshot(session)
                            recovered=(restore_rank(self,session,command) if command.get('manifest') is not None
                                else self._generation_boundaries[command['session']])
                            if action=='restore_generation':
                                occupied=sum(len(row.logits)+len(row.sampler)+128*(len(row.tokens)+len(row.stops))+1024
                                    for key,row in self._generation_boundaries.items() if key!=command['session'])
                                occupied+=len(recovered.logits)+len(recovered.sampler)+128*(len(recovered.tokens)+len(recovered.stops))+1024
                                if occupied>self.max_recovery_boundary_bytes:
                                    raise MemoryError('retained distributed generation boundaries exceed admission')
                        except BaseException as failure:
                            recovery_error=(str(failure) or type(failure).__name__)[:4096]
                        errors=[None]*self.world_size
                        self.distributed.all_gather_object(errors,recovery_error,group=self.group)
                        if any(value is not None for value in errors):
                            raise RuntimeError('distributed generation restore failed: '+
                                next(value for value in errors if value is not None))
                    if action=='restore_generation':
                        from .distributed_generation_checkpoint import signature
                        self._generation_boundaries[command['session']]=recovered
                        result=dict(session=command['session'],context_tokens=recovered.position,
                            boundary_sha256=signature(recovered))
                        summary=('restore_generation',recovered.position,signature(recovered))
                    else:
                        sink=None
                        if command.get('checkpoint_directory') is not None:
                            from .distributed_generation_checkpoint import CollectiveGenerationSink
                            sink=CollectiveGenerationSink(self,command['checkpoint_directory'],
                                max_rank_bytes=command['checkpoint_max_bytes'])
                        policy = command.get('sampling')
                        sampling = None if policy is None else SamplingPolicy(**policy)
                        options=dict(max_new_tokens=command['maximum'],cancel=latch,on_event=boundary,
                            checkpoint_sink=sink,max_checkpoint_bytes=command.get('checkpoint_max_bytes',0),
                            application_identity=command.get('application_identity',''),
                            stop_prefixes=command.get('stop_prefixes',()))
                        if recovered is None:
                            result = self.service.generate(session, command['tokens'],
                                stop_ids=command['stops'],sampling=sampling,**options)
                        else:
                            result = self.service.resume_generation(session,recovered,**options)
                        self._generation_boundaries.pop(command['session'],None)
                        summary = ('generate', tuple(result.token_ids), result.context_tokens,
                                   result.finish_reason)
                elif action == 'reset':
                    self.service.reset(session)
                    self._generation_boundaries.pop(command['session'],None)
                    summary = ('reset', command['session'])
                elif action == 'save_cache':
                    from .cache_file_recovery import verify_service_cache_file
                    directory = Path(command['directory'])
                    filename = f'rank-{self.rank}.cache'
                    result = self.service.save_session_cache(session,directory/filename,
                        max_bytes=command['max_bytes'])
                    verify_service_cache_file(directory/filename,receipt=result,max_bytes=command['max_bytes'])
                    history = self.service._session(session).tokens
                    result = dict(rank=self.rank,filename=filename,bytes=result['bytes'],
                                  sha256=result['sha256'],identity=self.service.status(session).identity)
                    summary = ('save_cache',command['session'],len(history),_history_digest(history))
                elif action == 'load_cache':
                    from .cache_file_recovery import verify_service_cache_file
                    receipt = command['manifest']['ranks'][self.rank]
                    source = Path(command['directory'])/receipt['filename']
                    verify_service_cache_file(source,receipt=receipt,max_bytes=command['max_bytes'])
                    model = self.service._session(session).model
                    if (getattr(model,'identity',None) != receipt['identity'] or
                            getattr(model,'identity',None) != command['manifest']['model_identity']):
                        raise ValueError('distributed rank cache identity differs from executor')
                    restore_rollback = self.service.snapshot(session)
                    self.service.load_session_cache(session,source,max_bytes=command['max_bytes'])
                    history = self.service._session(session).tokens
                    if (len(history) != command['manifest']['context_tokens'] or
                            _history_digest(history) != command['manifest']['history_sha256']):
                        raise ValueError('restored rank history differs from distributed manifest')
                    result = len(history)
                    summary = ('load_cache',command['session'],len(history),_history_digest(history))
                elif action == 'snapshot':
                    handle = command['snapshot']
                    if type(handle) is not int or handle <= 0 or handle in self._snapshots:
                        raise ValueError('unique positive distributed snapshot required')
                    snapshot = self.service.snapshot(session)
                    self._snapshots[handle] = (command['session'], snapshot)
                    result = handle
                    summary = ('snapshot', handle, len(snapshot.token_ids), _history_digest(snapshot.token_ids))
                elif action in ('restore', 'release_snapshot'):
                    owner, snapshot = self._snapshots[command['snapshot']]
                    if owner != command['session']:
                        raise ValueError('distributed snapshot belongs to a different session')
                    if action == 'restore':
                        prior_generation_boundary=self._generation_boundaries.get(command['session'])
                        restore_rollback=self.service.snapshot(session)
                        self.service.restore(session, snapshot)
                        self._generation_boundaries.pop(command['session'],None)
                        summary = ('restore', command['snapshot'], _history_digest(snapshot.token_ids),
                                   self.service.status(session).context_tokens)
                    else:
                        self.service.release_snapshot(session, snapshot)
                        del self._snapshots[command['snapshot']]
                        summary = ('release_snapshot', command['snapshot'])
                else:
                    self.service.close_session(session)
                    del self._sessions[command['session']]
                    self._generation_boundaries.pop(command['session'],None)
                    self._snapshots = {key:value for key,value in self._snapshots.items()
                                       if value[0] != command['session']}
                    summary = ('close_session', command['session'])
        except ExecutionCancelled:
            try:
                status = self.service.status(self._sessions[command['session']])
                if status.closed or status.pending_cleanup_count:
                    error = 'cancelled executor is quarantined or has pending cleanup'
                else:
                    summary = ('cancelled',command['session'],status.context_tokens)
            except BaseException as failure:
                error = (str(failure) or type(failure).__name__)[:4096]
        except BaseException as failure:
            error = (str(failure) or type(failure).__name__)[:4096]
        records = [None]*self.world_size
        self.distributed.all_gather_object(records,
            (summary, error, result if command.get('action')=='save_cache' else None),group=self.group)
        malformed = [rank for rank,record in enumerate(records)
                     if (type(record) is not tuple or len(record) != 3 or
                         (record[0] is not None and type(record[0]) is not tuple) or
                         (record[1] is not None and type(record[1]) is not str) or
                         (command.get('action') == 'save_cache' and
                          (record[2] is not None and type(record[2]) is not dict)))]
        if malformed:
            for rank in malformed:
                records[rank] = (None, 'malformed collective result record', None)
        failures = [f'rank {rank}: {record[1]}' for rank, record in enumerate(records)
                    if record[1] is not None]
        mismatch = any(record[0] != records[0][0] for record in records)
        # Opening is a distributed transaction too: if only a subset of ranks
        # successfully created its local executor session, those sessions must
        # not remain reachable after the coordinator is poisoned.  Best-effort
        # local rollback is itself rank-agreed so every participant reaches the
        # same failure boundary before the caller restarts the group.
        if command.get('action') == 'open' and (failures or mismatch):
            cleanup_error = None
            handle = command.get('session')
            try:
                if type(handle) is int and handle in self._sessions:
                    self.service.close_session(self._sessions[handle])
                    del self._sessions[handle]
            except BaseException as failure:
                cleanup_error = (str(failure) or type(failure).__name__)[:4096]
            cleanups = [None]*self.world_size
            self.distributed.all_gather_object(cleanups,cleanup_error,group=self.group)
            failures.extend(f'rank {rank} open rollback: {failure}'
                            for rank,failure in enumerate(cleanups) if failure is not None)
        if command.get('action') in ('load_cache','restore_generation','restore'):
            cleanup_error = None
            try:
                if restore_rollback is not None:
                    if failures or mismatch:
                        self.service.restore(self._sessions[command['session']],restore_rollback)
                        self._generation_boundaries.pop(command['session'],None)
                        if prior_generation_boundary is not None:
                            self._generation_boundaries[command['session']]=prior_generation_boundary
                    self.service.release_snapshot(self._sessions[command['session']],restore_rollback)
            except BaseException as failure:
                cleanup_error = (str(failure) or type(failure).__name__)[:4096]
            cleanups = [None]*self.world_size
            self.distributed.all_gather_object(cleanups,cleanup_error,group=self.group)
            failures.extend(f'rank {rank} restore cleanup: {failure}' for rank,failure in enumerate(cleanups)
                            if failure is not None)
        if failures or mismatch:
            self._failed = True
            raise RuntimeError('distributed execution failed: '+
                               ('; '.join(failures) if failures else 'committed rank results differ'))
        if command.get('action')=='load_cache':
            self._generation_boundaries.pop(command['session'],None)
        self._sequence += 1
        if command.get('action')=='save_cache':
            identities = {record[2]['identity'] for record in records}
            if len(identities) != 1:
                raise RuntimeError('distributed cache model identities differ across ranks')
            result = dict(version=1,world_size=self.world_size,context_tokens=summary[2],
                          history_sha256=summary[3],model_identity=next(iter(identities)),
                          ranks=[record[2] for record in records])
        if summary is not None and summary[0]=='cancelled' and self.rank==0:
            raise ExecutionCancelled('distributed generation cancelled; all ranks rolled back')
        return result

    def _request(self, action, *, local_cancel=None, local_on_event=None, **fields):
        from .execution_service import ExecutionCancelled
        if self.rank != 0:
            raise RuntimeError('only control rank zero may issue requests; peers call serve')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('distributed coordinator already has an active request')
        try:
            if self._closed or self._failed:
                raise RuntimeError('distributed coordinator is closed or requires rank restart')
            command = dict(action=action, sequence=self._sequence, **fields)
            if len(json.dumps(command,separators=(',', ':'),allow_nan=False).encode('utf-8')) > self.max_control_bytes:
                raise MemoryError('distributed command exceeds control message admission')
            try:
                self._active_cancel = local_cancel
                self._active_event = local_on_event
                self._broadcast(command)
                return self._execute(command)
            except ExecutionCancelled:
                raise
            except BaseException:
                self._failed = True
                raise
        finally:
            self._active_cancel = None
            self._active_event = None
            self._lock.release()

    def open_session(self):
        with self._id_lock:
            handle = self._next_id
            self._next_id += 1
        return self._request('open', session=handle)

    def generate(self, session, tokens, *, max_new_tokens, stop_ids=(), stop_prefixes=(), sampling=None,
                 cancel=None, on_event=None, checkpoint_directory=None,checkpoint_max_bytes=0,
                 application_identity=''):
        from .sampling import SamplingPolicy
        if sampling is not None and type(sampling) is not SamplingPolicy:
            raise TypeError('explicit SamplingPolicy required')
        if cancel is not None and not callable(getattr(cancel,'is_set',None)):
            raise ValueError('cancellation must provide is_set')
        if on_event is not None and not callable(on_event):
            raise ValueError('generation event handler must be callable')
        if type(session) is not int or session not in self._sessions:
            raise ValueError('known distributed session required')
        if (type(tokens) not in (list, tuple) or not tokens or
                any(type(value) is not int or value < 0 for value in tokens) or
                type(stop_ids) not in (list, tuple) or
                any(type(value) is not int or value < 0 for value in stop_ids) or
                type(stop_prefixes) not in (list, tuple) or len(stop_prefixes)>64 or
                any(type(prefix) not in (list, tuple) or not prefix or len(prefix)>128 or
                    any(type(value) is not int or value < 0 for value in prefix)
                    for prefix in stop_prefixes) or
                len({tuple(prefix) for prefix in stop_prefixes}) != len(stop_prefixes) or
                type(max_new_tokens) is not int or max_new_tokens <= 0):
            raise ValueError('bounded token sequence, stop inventory and positive generation limit required')
        if (len(tokens)+len(stop_ids)+sum(len(prefix) for prefix in stop_prefixes)+max_new_tokens)*32+512 > self.max_control_bytes:
            raise MemoryError('distributed token command exceeds control message budget')
        limits = self.service.limits
        if sampling is not None and (sampling.seed.bit_length()>128 or sampling.top_k>limits.vocabulary):
            raise ValueError('distributed sampling seed or top-k exceeds admission')
        status = self.service.status(self._sessions[session])
        if (any(value >= limits.vocabulary for value in (*tokens, *stop_ids, *(token for prefix in stop_prefixes for token in prefix))) or
                max_new_tokens > limits.max_new_tokens or
                status.context_tokens+len(tokens)+max_new_tokens > limits.context):
            raise ValueError('distributed request exceeds vocabulary, context or generation admission')
        destination=self._generation_destination(checkpoint_directory,checkpoint_max_bytes,application_identity)
        try:
            return self._request('generate', session=session, tokens=list(tokens),
                                 local_cancel=cancel,
                                 local_on_event=on_event,
                                 maximum=max_new_tokens, stops=list(stop_ids),
                                 stop_prefixes=[list(prefix) for prefix in stop_prefixes],
                                 sampling=None if sampling is None else asdict(sampling),
                                 checkpoint_directory=destination,checkpoint_max_bytes=checkpoint_max_bytes,
                                 application_identity=application_identity)
        except BaseException:
            if destination is not None:
                shutil.rmtree(destination, ignore_errors=True)
            raise

    def _generation_destination(self,directory,max_bytes,application_identity):
        from .session_recovery import _path
        if self.rank!=0:raise RuntimeError('only root may issue distributed generation requests')
        if (type(application_identity) is not str or len(application_identity)>8192 or
                '\x00' in application_identity or
                (directory is not None and not application_identity)):
            raise ValueError('bounded generation application identity required')
        if directory is None:
            if type(max_bytes) is not int or max_bytes!=0:
                raise ValueError('generation publication budget requires a destination')
            return None
        if type(max_bytes) is not int or not 128<=max_bytes<=2**63-1:
            raise ValueError('explicit per-rank generation frame budget required')
        root=_path(directory)
        if len(str(root))>4096:raise ValueError('bounded distributed generation directory required')
        root.mkdir(parents=True,exist_ok=True)
        destination=root/('generation-'+uuid4().hex)
        destination.mkdir()
        return str(destination)

    def restore_generation_checkpoint(self,session,manifest_path,*,max_rank_bytes,application_identity=''):
        from .distributed_generation_checkpoint import admit_manifest
        if self.rank!=0:raise RuntimeError('only root may restore distributed generation')
        if type(session) is not int or session not in self._sessions:
            raise ValueError('known distributed session required')
        if type(max_rank_bytes) is not int or not 128<=max_rank_bytes<=2**63-1:
            raise ValueError('explicit per-rank generation recovery budget required')
        if (type(application_identity) is not str or not 0<len(application_identity)<=8192
                or '\x00' in application_identity):
            raise ValueError('bounded generation application identity required')
        path,manifest=admit_manifest(manifest_path,world_size=self.world_size,
            max_rank_bytes=max_rank_bytes,context=self.service.limits.context,
            application_identity=application_identity)
        return self._request('restore_generation',session=session,directory=str(path.parent),manifest=manifest,
            recovery_max_bytes=max_rank_bytes,application_identity=application_identity)

    def resume_generation(self,session,manifest_path=None,*,max_rank_bytes=0,max_new_tokens,
            cancel=None,on_event=None,checkpoint_directory=None,checkpoint_max_bytes=0,
            application_identity=''):
        """Restore all complete generation frames, then continue without replay.

        Every rank verifies the shared manifest's frame and logical-boundary
        digests before any new token is sampled. Restore failure poisons the
        coordinator and requires replacement ranks, as other collective failure.
        """
        from .distributed_generation_checkpoint import admit_manifest
        if self.rank!=0:raise RuntimeError('only root may resume distributed generation')
        if type(session) is not int or session not in self._sessions:
            raise ValueError('known distributed session required')
        if type(max_new_tokens) is not int or not 1<=max_new_tokens<=self.service.limits.max_new_tokens:
            raise ValueError('admitted positive continuation length required')
        if cancel is not None and not callable(getattr(cancel,'is_set',None)):
            raise ValueError('cancellation must provide is_set')
        if on_event is not None and not callable(on_event):raise TypeError('callable generation event handler required')
        if (type(application_identity) is not str or not 0<len(application_identity)<=8192
                or '\x00' in application_identity):
            raise ValueError('bounded generation application identity required')
        if manifest_path is None:
            if type(max_rank_bytes) is not int or max_rank_bytes!=0 or session not in self._generation_boundaries:
                raise ValueError('owned restored generation boundary required without an input manifest')
            manifest=None;directory=None;occupied=self._generation_boundaries[session].position
        else:
            if type(max_rank_bytes) is not int or not 128<=max_rank_bytes<=2**63-1:
                raise ValueError('explicit per-rank generation recovery budget required')
            path,manifest=admit_manifest(manifest_path,world_size=self.world_size,
                max_rank_bytes=max_rank_bytes,context=self.service.limits.context,
                application_identity=application_identity)
            directory=str(path.parent);occupied=manifest['context_tokens']
        if occupied+max_new_tokens>self.service.limits.context:
            raise ValueError('recovered generation exceeds context admission')
        destination=self._generation_destination(checkpoint_directory,checkpoint_max_bytes,application_identity)
        try:
            return self._request('resume_generation',session=session,directory=directory,manifest=manifest,
                recovery_max_bytes=max_rank_bytes,maximum=max_new_tokens,local_cancel=cancel,local_on_event=on_event,
                checkpoint_directory=destination,checkpoint_max_bytes=checkpoint_max_bytes,
                application_identity=application_identity)
        except BaseException:
            if destination is not None:
                shutil.rmtree(destination, ignore_errors=True)
            raise

    def reset_session(self, session):
        if session not in self._sessions:
            raise ValueError('unknown distributed session')
        self._request('reset', session=session)

    def snapshot(self, session):
        if type(session) is not int or session not in self._sessions:
            raise ValueError('known distributed session required')
        status = self.service.status(self._sessions[session])
        if status.snapshot_count >= self.service.limits.max_snapshots_per_session:
            raise MemoryError('distributed session snapshot inventory exceeds admission')
        with self._id_lock:
            handle = self._next_snapshot
            self._next_snapshot += 1
        return self._request('snapshot', session=session, snapshot=handle)

    def save_session_cache(self, session, directory, *, max_rank_bytes):
        """Publish a complete rank set on an explicitly shared filesystem.

        A new immutable generation directory is created for each save. Partial
        rank files are retained for inspection but have no commit manifest.
        """
        from .session_recovery import _path
        import os
        import tempfile
        if self.rank != 0:
            raise RuntimeError('only root may save distributed caches')
        if type(session) is not int or session not in self._sessions:
            raise ValueError('known distributed session required')
        if type(max_rank_bytes) is not int or not 88<=max_rank_bytes<=2**63-1:
            raise ValueError('explicit per-rank cache byte limit required')
        root = _path(directory)
        if len(str(root))>4096:
            raise ValueError('distributed cache directory exceeds path admission')
        root.mkdir(parents=True,exist_ok=True)
        generation = root/('generation-'+uuid4().hex)
        generation.mkdir()
        try:
            manifest = self._request('save_cache',session=session,directory=str(generation),max_bytes=max_rank_bytes)
        except BaseException:
            # No rank set can be committed after a failed collective request;
            # remove only this fresh reservation while preserving unrelated
            # generations under the caller-owned root.
            shutil.rmtree(generation, ignore_errors=True)
            raise
        encoded = json.dumps(manifest,sort_keys=True,separators=(',', ':'),allow_nan=False).encode('utf-8')
        if len(encoded)>65536:
            raise MemoryError('distributed cache manifest exceeds admission')
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=generation,delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            _path(generation)
            destination = generation/'manifest.json'
            os.replace(temporary,destination)
            temporary = None
            if os.name!='nt':
                descriptor=os.open(generation,os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            return str(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def restore(self, session, snapshot):
        self._check_snapshot(session,snapshot)
        self._request('restore',session=session,snapshot=snapshot)

    def load_session_cache(self, session, manifest_path, *, max_rank_bytes):
        """Restore a committed shared-filesystem rank set with rollback consensus."""
        from .distributed_generation_checkpoint import read_manifest
        from .segment_snapshot import _pairs
        if self.rank != 0:
            raise RuntimeError('only root may restore distributed caches')
        if type(session) is not int or session not in self._sessions:
            raise ValueError('known distributed session required')
        if type(max_rank_bytes) is not int or not 88<=max_rank_bytes<=2**63-1:
            raise ValueError('explicit per-rank cache byte limit required')
        path, encoded = read_manifest(manifest_path)
        manifest = json.loads(encoded,object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if type(manifest) is not dict or set(manifest)!={'version','world_size','context_tokens','history_sha256','model_identity','ranks'}:
            raise ValueError('invalid distributed cache manifest fields')
        if (type(manifest['version']) is not int or manifest['version']!=1 or
                type(manifest['world_size']) is not int or manifest['world_size']!=self.world_size or
                type(manifest['context_tokens']) is not int or
                not 0<=manifest['context_tokens']<=self.service.limits.context or
                type(manifest['model_identity']) is not str or
                not 0<len(manifest['model_identity'])<=4096 or
                type(manifest['ranks']) is not list or len(manifest['ranks'])!=self.world_size):
            raise ValueError('distributed cache geometry differs')
        def digest(value):
            return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)
        if not digest(manifest['history_sha256']):
            raise ValueError('invalid distributed cache history digest')
        for rank,receipt in enumerate(manifest['ranks']):
            if (type(receipt) is not dict or set(receipt)!={'rank','filename','bytes','sha256','identity'} or
                    type(receipt['rank']) is not int or receipt['rank']!=rank or
                    receipt['filename']!=f'rank-{rank}.cache' or
                    type(receipt['bytes']) is not int or not 88<=receipt['bytes']<=max_rank_bytes or
                    not digest(receipt['sha256']) or type(receipt['identity']) is not str or
                    not 0<len(receipt['identity'])<=4096):
                raise ValueError('invalid distributed rank cache receipt')
        current_identity = self.service.status(session).identity
        if manifest['model_identity'] != current_identity:
            raise ValueError('distributed cache model identity differs from target session')
        return self._request('load_cache',session=session,directory=str(path.parent),
                             manifest=manifest,max_bytes=max_rank_bytes)

    def release_snapshot(self, session, snapshot):
        self._check_snapshot(session,snapshot)
        self._request('release_snapshot',session=session,snapshot=snapshot)

    def _check_snapshot(self, session, snapshot):
        if (type(session) is not int or session not in self._sessions or
                type(snapshot) is not int or snapshot not in self._snapshots or
                self._snapshots[snapshot][0] != session):
            raise ValueError('owned distributed session snapshot required')

    def close_session(self, session):
        if type(session) is not int or session not in self._sessions:
            raise ValueError('unknown distributed session')
        self._request('close_session', session=session)

    def close(self):
        if self._closed:
            return
        self._request('close')

    def serve(self):
        if self.rank == 0:
            raise RuntimeError('rank zero issues requests instead of running the peer loop')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('peer control loop is already running')
        try:
            while not self._closed and not self._failed:
                try:
                    self._execute(self._broadcast(None))
                except BaseException:
                    self._failed = True
                    raise
        finally:
            self._lock.release()
