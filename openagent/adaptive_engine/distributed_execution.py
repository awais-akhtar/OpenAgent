"""Ordered distributed control for a tensor-parallel execution service.

The caller starts ranks and a trusted Gloo control group separately from any
accelerator projection group. Rank zero drives requests; peers run serve().
Collective failure poisons the coordinator: all ranks must then be restarted.
"""
import threading
import json
import shutil
from dataclasses import asdict, dataclass
from functools import wraps
import hashlib
import struct
from pathlib import Path
from uuid import uuid4

from .failure_reporting import add_failure_note, failure_summary


def _history_digest(tokens):
    digest = hashlib.sha256()
    for start in range(0,len(tokens),4096):
        chunk = tokens[start:start+4096]
        digest.update(struct.pack('<'+'Q'*len(chunk),*chunk))
    return digest.hexdigest()


def _required_service_snapshot(service, session):
    """Require a concrete local rollback receipt before a rank mutates state."""
    from .execution_service import ExecutionSessionSnapshot
    receipt = service.snapshot(session)
    # A session id alone is not an ownership boundary: callers may reuse ids
    # across service instances, and a stale receipt can name an earlier
    # history for the same live session. Bind the temporary rollback receipt
    # to this service, this executor identity, and the exact current tokens
    # before any distributed operation relies on it.
    try:
        local = service._session(session)
        service_id = service._identity
        identity = local.identity
        tokens = local.tokens
    except BaseException as failure:
        raise RuntimeError(
            'distributed rank snapshot owner identity is unavailable') from failure
    if (type(receipt) is not ExecutionSessionSnapshot or
            type(service_id) is not str or not service_id or
            receipt.service_id != service_id or receipt.session_id != session or
            type(receipt.snapshot_id) is not str or not receipt.snapshot_id or
            type(identity) is not str or not identity or receipt.identity != identity or
            type(tokens) is not tuple or receipt.token_ids != tokens):
        raise RuntimeError(
            'distributed rank snapshot did not provide a valid service-, model-, '
            'and history-bound rollback receipt')
    return receipt


def _failure_summary(failure, *, limit=4096):
    """Return bounded rank-error text without calling arbitrary __str__ code."""
    try:
        name=type(failure).__name__
    except BaseException:
        name='BaseException'
    if type(name) is not str or not name or len(name)>128:
        name='BaseException'
    # Agreement relies on every surviving rank reaching the next collective.
    # Exception stringification is arbitrary application code and can itself
    # raise, so keep the exchanged diagnostic stable and side-effect-free.
    return (name+': failure details omitted')[:limit]


def _valid_result_summary(action, value):
    """Validate the bounded primitive summary exchanged after each request.

    These records are received from every process-group peer. Exact builtin
    fields avoid invoking arbitrary ``__eq__`` implementations while ranks
    compare summaries; malformed nested values become an agreed operation
    failure instead of a rank-local exception between collectives.
    """
    def integer(item, minimum=0):
        return type(item) is int and minimum <= item <= 2**63-1

    def digest(item):
        return (type(item) is str and len(item) == 64 and
                all(character in '0123456789abcdef' for character in item))

    if type(value) is not tuple or not value or type(value[0]) is not str:
        return False
    tag = value[0]
    if action == 'open':
        return (len(value) == 4 and tag == 'open' and integer(value[1], 1) and
                type(value[2]) is str and 0 < len(value[2]) <= 4096 and
                '\x00' not in value[2] and type(value[3]) is tuple and
                len(value[3]) == 2 and all(type(item) is str and item for item in value[3]))
    if action == 'close':
        return len(value) == 1 and value[0] == 'close'
    if action in ('generate', 'resume_generation') and tag == 'cancelled':
        return (len(value) == 4 and integer(value[1], 1) and integer(value[2]) and
                digest(value[3]))
    if action in ('generate', 'resume_generation'):
        return (len(value) == 4 and tag == 'generate' and type(value[1]) is tuple and
                all(integer(item) for item in value[1]) and integer(value[2]) and
                type(value[3]) is str and 0 < len(value[3]) <= 64)
    if action in ('reset',):
        return (len(value) == 4 and tag == 'reset' and integer(value[1], 1) and
                integer(value[2]) and digest(value[3]))
    if action == 'save_cache':
        return (len(value) == 4 and tag == 'save_cache' and integer(value[1], 1) and
                integer(value[2]) and digest(value[3]))
    if action == 'load_cache':
        return (len(value) == 4 and tag == 'load_cache' and integer(value[1], 1) and
                integer(value[2]) and digest(value[3]))
    if action == 'snapshot':
        return (len(value) == 4 and tag == 'snapshot' and integer(value[1], 1) and
                integer(value[2]) and digest(value[3]))
    if action == 'restore':
        return (len(value) == 4 and tag == 'restore' and integer(value[1], 1) and
                digest(value[2]) and integer(value[3]))
    if action in ('restore_generation',):
        return (len(value) == 3 and tag == 'restore_generation' and
                integer(value[1]) and digest(value[2]))
    if action == 'cancelled':
        return (len(value) == 4 and tag == 'cancelled' and integer(value[1], 1) and
                integer(value[2]) and digest(value[3]))
    if action == 'release_snapshot':
        return len(value) == 2 and tag == 'release_snapshot' and integer(value[1], 1)
    if action == 'close_session':
        return len(value) == 2 and tag == 'close_session' and integer(value[1], 1)
    return False


def _collective_cancel_requested(distributed, torch, group, world_size, requested):
    """Agree local cancellation-flag setup before entering its tensor collective.

    Tensor construction is rank-local and can fail (for example, under host
    memory pressure).  Building the flag immediately before ``all_reduce``
    would let the other ranks block in that reduction while the failed rank
    unwinds toward the result agreement.  Exchange bounded setup diagnostics
    first so every surviving rank either reduces or skips the reduction.
    """
    flag = None
    allocation_error = None
    try:
        flag = torch.tensor([int(bool(requested))], dtype=torch.int64, device='cpu')
    except BaseException as failure:
        allocation_error = _failure_summary(failure, limit=1024)
    allocation_errors = [None] * world_size
    distributed.all_gather_object(allocation_errors, allocation_error, group=group)
    if any(value is not None for value in allocation_errors):
        raise RuntimeError('distributed cancellation flag allocation failed: ' +
            next(f'rank {rank}: {value}' for rank, value in enumerate(allocation_errors)
                 if value is not None))

    distributed.all_reduce(flag, op=distributed.ReduceOp.MAX, group=group)
    value = None
    read_error = None
    try:
        value = flag.item()
        if type(value) not in (int, bool) or int(value) not in (0, 1):
            raise ValueError('cancellation reduction returned a non-boolean flag')
        value = bool(value)
    except BaseException as failure:
        read_error = _failure_summary(failure, limit=1024)
    read_errors = [None] * world_size
    distributed.all_gather_object(read_errors, read_error, group=group)
    if any(item is not None for item in read_errors):
        raise RuntimeError('distributed cancellation flag read failed: ' +
            next(f'rank {rank}: {item}' for rank, item in enumerate(read_errors)
                 if item is not None))
    return value


@dataclass(frozen=True)
class _RequestRejected:
    action: str
    message: str


def _synchronize_root_preflight(method):
    """Make root-only validation failures visible to every waiting rank.

    A public request method can fail while parsing a manifest or validating
    local arguments before its normal command broadcast. Without a rejection
    envelope, peers remain blocked in ``serve()`` waiting for that command.
    Once a collective has advanced or the coordinator is poisoned, do not send
    a second command.
    """
    @wraps(method)
    def synchronized(self, *args, **kwargs):
        sequence = self._sequence
        try:
            return method(self, *args, **kwargs)
        except BaseException as failure:
            if (self.rank == 0 and not self._closed and not self._failed and
                    self._sequence == sequence):
                try:
                    try:
                        message = _failure_summary(failure)
                    except BaseException:
                        message = type(failure).__name__
                    self._reject_preflight(method.__name__, message)
                except BaseException as rejection:
                    # The rejection acknowledgement is expected for ordinary
                    # validation errors. Preserve process-level interrupts
                    # after peers have received the synchronized rejection;
                    # otherwise Ctrl+C/SystemExit before broadcast can leave
                    # every peer blocked in serve().
                    if not isinstance(failure, Exception):
                        try:
                            add_failure_note(failure, 'distributed preflight was synchronized: '+failure_summary(rejection))
                        except BaseException:
                            pass
                        raise failure from rejection
                    try:
                        add_failure_note(failure, 'distributed preflight synchronization result: '+failure_summary(rejection))
                    except BaseException:
                        pass
                    raise rejection from failure
            raise
    return synchronized


class DistributedExecutionService:
    def __init__(self, service, *, control_group, max_control_bytes=1024*1024,
                 max_recovery_boundary_bytes=64*1024**2):
        import torch.distributed as distributed
        from .execution_service import NativeExecutionService
        if not distributed.is_initialized():
            raise ValueError('caller-owned initialized control group required')
        if str(distributed.get_backend(control_group)).lower() != 'gloo':
            raise ValueError('CPU Gloo control group required')
        # Every coordinator control message uses object collectives, which do
        # not expose a per-call Work handle on the supported Torch range.
        # Require the caller-owned group itself to enforce a short finite
        # timeout before the constructor enters its first rank agreement.
        from .tensor_parallel import require_group_timeout
        control_timeout_seconds = require_group_timeout(control_group, 300)
        rank = distributed.get_rank(control_group)
        world_size = distributed.get_world_size(control_group)
        root = distributed.get_global_rank(control_group, 0)

        # Do not let a rank-local admission error escape before peers enter the
        # constructor's agreement collective.  Otherwise a bad service type,
        # malformed budget, or unreadable limits value on one rank can strand
        # every other rank in all_gather_object before a coordinator exists.
        local_errors = []
        if type(service) is not NativeExecutionService:
            local_errors.append('rank-local NativeExecutionService required')
        if type(max_control_bytes) is not int or max_control_bytes < 4096:
            local_errors.append('control message budget must be at least 4096 bytes')
        if (type(max_recovery_boundary_bytes) is not int or
                max_recovery_boundary_bytes <= 0):
            local_errors.append('positive retained generation boundary admission required')
        limits_identity = None
        if type(service) is NativeExecutionService:
            try:
                limits_identity = repr(service.limits)
                if not limits_identity or len(limits_identity) > 8192:
                    raise ValueError('rank-local execution limits identity exceeds admission')
            except BaseException as failure:
                local_errors.append('execution limits unavailable: '+_failure_summary(failure,limit=1024))
        policy = (max_control_bytes if type(max_control_bytes) is int else None,
                  max_recovery_boundary_bytes if type(max_recovery_boundary_bytes) is int else None,
                  limits_identity, control_timeout_seconds)
        admissions = [None]*world_size
        distributed.all_gather_object(admissions,
            (tuple(local_errors), policy), group=control_group)
        rank_errors = [f'rank {peer}: {message}'
            for peer, (errors, _) in enumerate(admissions) for message in errors]
        if rank_errors:
            raise ValueError('distributed coordinator admission failed: '+'; '.join(rank_errors))
        policies = [record[1] for record in admissions]
        if any(value != policies[0] for value in policies):
            raise ValueError('rank-local execution limits, coordinator budgets, or control-group timeouts differ')

        self.service, self.distributed, self.group = service, distributed, control_group
        self.rank, self.world_size, self.root = rank, world_size, root
        self.control_timeout_seconds = control_timeout_seconds
        self.max_control_bytes = max_control_bytes
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

    def _poison_local_state(self, failure):
        """Retire rank-local owners after a control collective is broken.

        A failed Gloo/NCCL control collective cannot be safely retried: peers
        may have observed different command boundaries.  Previously the peer
        loop only flipped ``_failed`` and left local sessions, snapshots, and
        generation boundaries reachable until process teardown.  Close the
        service once, retain any cleanup failure on the original exception,
        and make the restart requirement explicit.
        """
        self._failed = True
        self._active_cancel = None
        self._active_event = None
        self._generation_boundaries.clear()
        self._snapshots.clear()
        self._sessions.clear()
        try:
            self.service.close()
        except BaseException as cleanup:
            add_failure_note(failure, 'distributed transport-loss cleanup retained: '+
                             failure_summary(cleanup))
            try:
                setattr(failure, 'cleanup', self.service.close)
            except BaseException:
                pass

    def _broadcast(self, command):
        values = [command]
        try:
            self.distributed.broadcast_object_list(values, src=self.root, group=self.group)
        except BaseException as failure:
            self._poison_local_state(failure)
            raise
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
                                                 'resume_generation','restore_generation','reject')):
                raise ValueError('distributed command sequence or action mismatch')
            action = command['action']
            if action == 'reject':
                rejected_action, rejection = command.get('rejected_action'), command.get('rejection')
                if (type(rejected_action) is not str or not rejected_action or
                        len(rejected_action) > 64 or type(rejection) is not str or
                        not rejection or len(rejection) > 1024):
                    raise ValueError('bounded distributed preflight rejection required')
                summary = ('request_rejected', rejected_action, rejection)
            elif action == 'open':
                handle = command['session']
                if type(handle) is not int or handle <= 0 or handle in self._sessions:
                    raise ValueError('unique positive distributed session required')
                local_session = self.service.open_session()
                self._sessions[handle] = local_session
                # A matching control-group size and request schedule do not
                # prove that ranks opened the same model/bank layout. Compare
                # the executor identity and implementation before publishing
                # the distributed handle. A mismatch is handled by the open
                # transaction rollback below, before any generation collective
                # can consume incompatible rank-local state.
                local_model = self.service._session(local_session).model
                identity = getattr(local_model, 'identity', None)
                if (type(identity) is not str or not 0 < len(identity) <= 4096 or
                        '\x00' in identity):
                    raise ValueError('distributed executor requires a bounded stable model identity')
                implementation = (type(local_model).__module__, type(local_model).__qualname__)
                result, summary = handle, ('open', handle, identity, implementation)
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
                                callback_error = _failure_summary(failure,limit=1024)
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
                            cancellation_error = _failure_summary(failure,limit=1024)
                        cancellation_errors = [None]*self.world_size
                        self.distributed.all_gather_object(cancellation_errors,cancellation_error,group=self.group)
                        if any(value is not None for value in cancellation_errors):
                            raise RuntimeError('distributed cancellation owner failed: '+
                                               next(value for value in cancellation_errors if value is not None))
                        if _collective_cancel_requested(self.distributed, torch, self.group,
                                                        self.world_size, requested):
                            latch.set()
                    boundary()
                    recovered=None
                    if action in ('resume_generation','restore_generation'):
                        from .distributed_generation_checkpoint import restore_rank
                        recovery_error=None
                        try:
                            if (action=='restore_generation' or
                                    (action=='resume_generation' and command.get('manifest') is not None)):
                                prior_generation_boundary=self._generation_boundaries.get(command['session'])
                                restore_rollback=_required_service_snapshot(self.service,session)
                            recovered=(restore_rank(self,session,command) if command.get('manifest') is not None
                                else self._generation_boundaries[command['session']])
                            if action=='restore_generation':
                                occupied=sum(len(row.logits)+len(row.sampler)+128*(len(row.tokens)+len(row.stops))+1024
                                    for key,row in self._generation_boundaries.items() if key!=command['session'])
                                occupied+=len(recovered.logits)+len(recovered.sampler)+128*(len(recovered.tokens)+len(recovered.stops))+1024
                                if occupied>self.max_recovery_boundary_bytes:
                                    raise MemoryError('retained distributed generation boundaries exceed admission')
                        except BaseException as failure:
                            recovery_error=_failure_summary(failure)
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
                        # A rank can report a local failure after another rank
                        # has already committed generated tokens. Retain the
                        # pre-generation physical state so result disagreement
                        # below can roll back every surviving rank before the
                        # coordinator is poisoned. Resume consumes a retained
                        # generation boundary, so preserve that metadata too.
                        if restore_rollback is None:
                            prior_generation_boundary=self._generation_boundaries.get(command['session'])
                            snapshot_error = None
                            try:
                                restore_rollback=_required_service_snapshot(self.service,session)
                            except BaseException as failure:
                                snapshot_error=_failure_summary(failure,limit=2048)
                            snapshot_errors=[None]*self.world_size
                            self.distributed.all_gather_object(snapshot_errors,snapshot_error,
                                                              group=self.group)
                            if any(value is not None for value in snapshot_errors):
                                raise RuntimeError('distributed generation rollback snapshot admission failed: '+
                                    '; '.join(f'rank {rank}: {failure}'
                                        for rank,failure in enumerate(snapshot_errors)
                                        if failure is not None))
                        if recovered is None:
                            result = self.service.generate(session, command['tokens'],
                                stop_ids=command['stops'],sampling=sampling,**options)
                        else:
                            result = self.service.resume_generation(session,recovered,**options)
                        self._generation_boundaries.pop(command['session'],None)
                        summary = ('generate', tuple(result.token_ids), result.context_tokens,
                                   result.finish_reason)
                elif action == 'reset':
                    # Reset is destructive on each local executor. Retain a
                    # rank-local physical snapshot until the result agreement
                    # below, so a peer-local reset failure cannot leave the
                    # surviving ranks at different generation boundaries.
                    prior_generation_boundary=self._generation_boundaries.get(command['session'])
                    restore_rollback=_required_service_snapshot(self.service,session)
                    self.service.reset(session)
                    self._generation_boundaries.pop(command['session'],None)
                    status=self.service.status(session)
                    history=self.service._session(session).tokens
                    if status.context_tokens!=0 or history:
                        raise RuntimeError('rank reset did not reach the empty history boundary')
                    summary = ('reset', command['session'], status.context_tokens,
                               _history_digest(history))
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
                    from .distributed_generation_checkpoint import read_manifest
                    from .segment_snapshot import _pairs
                    def admit_current_manifest():
                        _, current_bytes = read_manifest(Path(command['directory'])/'manifest.json')
                        current = json.loads(current_bytes, object_pairs_hook=_pairs,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                        if current != command['manifest']:
                            raise ValueError('distributed cache manifest changed during restore')
                    # The coordinator's initial parse is not a pin: shared
                    # storage can change between root admission and rank use.
                    # Re-admit before consuming a rank receipt and after the
                    # physical install; the latter is covered by the existing
                    # collective rollback snapshot.
                    admit_current_manifest()
                    receipt = command['manifest']['ranks'][self.rank]
                    source = Path(command['directory'])/receipt['filename']
                    verify_service_cache_file(source,receipt=receipt,max_bytes=command['max_bytes'])
                    model = self.service._session(session).model
                    if (getattr(model,'identity',None) != receipt['identity'] or
                            getattr(model,'identity',None) != command['manifest']['model_identity']):
                        raise ValueError('distributed rank cache identity differs from executor')
                    restore_rollback = _required_service_snapshot(self.service,session)
                    # Pin the exact receipt during the same stable read that
                    # installs physical executor state. The earlier readback
                    # check alone leaves a path-swap window before this open.
                    self.service.load_session_cache(session,source,
                        max_bytes=command['max_bytes'],expected_file_receipt=receipt)
                    history = self.service._session(session).tokens
                    if (len(history) != command['manifest']['context_tokens'] or
                            _history_digest(history) != command['manifest']['history_sha256']):
                        raise ValueError('restored rank history differs from distributed manifest')
                    admit_current_manifest()
                    result = len(history)
                    summary = ('load_cache',command['session'],len(history),_history_digest(history))
                elif action == 'snapshot':
                    handle = command['snapshot']
                    if type(handle) is not int or handle <= 0 or handle in self._snapshots:
                        raise ValueError('unique positive distributed snapshot required')
                    snapshot = _required_service_snapshot(self.service,session)
                    self._snapshots[handle] = (command['session'], snapshot)
                    result = handle
                    summary = ('snapshot', handle, len(snapshot.token_ids), _history_digest(snapshot.token_ids))
                elif action in ('restore', 'release_snapshot'):
                    owner, snapshot = self._snapshots[command['snapshot']]
                    if owner != command['session']:
                        raise ValueError('distributed snapshot belongs to a different session')
                    if action == 'restore':
                        prior_generation_boundary=self._generation_boundaries.get(command['session'])
                        restore_rollback=_required_service_snapshot(self.service,session)
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
                    # Equal lengths are not enough to establish that every
                    # rank rolled cancellation back to the same logical
                    # boundary. Compare the restored token history as well;
                    # the normal result-agreement path will poison the group
                    # if any rank retained a different prefix.
                    history = self.service._session(
                        self._sessions[command['session']]).tokens
                    summary = ('cancelled', command['session'],
                               status.context_tokens, _history_digest(history))
            except BaseException as failure:
                error = _failure_summary(failure)
        except BaseException as failure:
            error = _failure_summary(failure)
        records = [None]*self.world_size
        self.distributed.all_gather_object(records,
            (summary, error, result if command.get('action')=='save_cache' else None),group=self.group)
        action = command.get('action')
        def valid_cache_receipt(rank, receipt):
            return (type(receipt) is dict and
                    set(receipt) == {'rank', 'filename', 'bytes', 'sha256', 'identity'} and
                    type(receipt['rank']) is int and receipt['rank'] == rank and
                    receipt['filename'] == f'rank-{rank}.cache' and
                    type(receipt['bytes']) is int and 88 <= receipt['bytes'] <= command['max_bytes'] and
                    type(receipt['sha256']) is str and len(receipt['sha256']) == 64 and
                    all(character in '0123456789abcdef' for character in receipt['sha256']) and
                    type(receipt['identity']) is str and 0 < len(receipt['identity']) <= 4096 and
                    '\x00' not in receipt['identity'])
        malformed = []
        for rank, record in enumerate(records):
            invalid = (type(record) is not tuple or len(record) != 3)
            if not invalid:
                summary_value, failure_value, receipt_value = record
                invalid = (failure_value is not None and
                           (type(failure_value) is not str or len(failure_value) > 4096))
                if not invalid and summary_value is not None:
                    invalid = not _valid_result_summary(action, summary_value)
                if not invalid and summary_value is None and failure_value is None:
                    invalid = True
                if not invalid and action == 'save_cache':
                    invalid = (receipt_value is None if failure_value is not None else
                               not valid_cache_receipt(rank, receipt_value))
                elif not invalid:
                    invalid = receipt_value is not None
            if invalid:
                malformed.append(rank)
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
                cleanup_error = _failure_summary(failure)
            cleanups = [None]*self.world_size
            self.distributed.all_gather_object(cleanups,cleanup_error,group=self.group)
            failures.extend(f'rank {rank} open rollback: {failure}'
                            for rank,failure in enumerate(cleanups) if failure is not None)
        # A failed close is not physically rollbackable: some ranks may have
        # already released their model while others still own a live or
        # quarantined executor.  Retry cleanup on every rank that still has
        # the handle before poisoning the group, rather than leaving surviving
        # rank-local sessions reachable until process teardown.
        if command.get('action') == 'close_session' and (failures or mismatch):
            cleanup_error = None
            handle = command.get('session')
            try:
                if type(handle) is int and handle in self._sessions:
                    self.service.close_session(self._sessions[handle])
                    del self._sessions[handle]
                    self._generation_boundaries.pop(handle, None)
                    self._snapshots = {key: value for key, value in self._snapshots.items()
                                       if value[0] != handle}
            except BaseException as failure:
                cleanup_error = _failure_summary(failure)
            cleanups = [None]*self.world_size
            self.distributed.all_gather_object(cleanups, cleanup_error, group=self.group)
            failures.extend(f'rank {rank} close rollback: {failure}'
                            for rank, failure in enumerate(cleanups) if failure is not None)
        # Whole-service close has no physical rollback: a rank may have
        # already released some owners while another still retains failed
        # cleanup. Retry local cleanup on every rank whose first attempt did
        # not complete, then agree before peers leave their control loops.
        # The original collective still fails and requires a fresh group.
        if command.get('action') == 'close' and (failures or mismatch):
            cleanup_error = None
            try:
                if not self._closed:
                    self.service.close()
                    self._sessions.clear()
                    self._generation_boundaries.clear()
                    self._snapshots.clear()
                    self._closed = True
            except BaseException as failure:
                cleanup_error = _failure_summary(failure)
            cleanups = [None]*self.world_size
            self.distributed.all_gather_object(cleanups, cleanup_error, group=self.group)
            failures.extend(f'rank {rank} service-close retry: {failure}'
                            for rank, failure in enumerate(cleanups) if failure is not None)
        # Releasing a rank-local snapshot is not rollbackable: one rank may
        # have released its physical receipt while another retained it (or
        # quarantined its executor during release). Since the collective will
        # poison the coordinator below, close the affected local session on
        # every surviving rank so its remaining snapshot/cache owners are not
        # stranded behind an unusable distributed handle.
        if command.get('action') == 'release_snapshot' and (failures or mismatch):
            cleanup_error = None
            handle = command.get('session')
            try:
                if type(handle) is int and handle in self._sessions:
                    self.service.close_session(self._sessions[handle])
                    del self._sessions[handle]
                    self._generation_boundaries.pop(handle, None)
                    self._snapshots = {key: value for key, value in self._snapshots.items()
                                       if value[0] != handle}
            except BaseException as failure:
                cleanup_error = _failure_summary(failure)
            cleanups = [None]*self.world_size
            self.distributed.all_gather_object(cleanups, cleanup_error, group=self.group)
            failures.extend(f'rank {rank} snapshot-release cleanup: {failure}'
                            for rank, failure in enumerate(cleanups) if failure is not None)
        # Snapshot creation can succeed on only a subset of ranks (or produce
        # different logical boundaries). The group is unusable after that
        # disagreement, so retire every newly-created physical receipt before
        # returning the collective error. If receipt retirement itself fails,
        # tear down that rank's session as the only safe owner-level cleanup.
        if command.get('action') == 'snapshot' and (failures or mismatch):
            cleanup_error = None
            handle = command.get('session')
            snapshot_handle = command.get('snapshot')
            try:
                saved = self._snapshots.get(snapshot_handle)
                if saved is not None and saved[0] == handle and handle in self._sessions:
                    try:
                        self.service.release_snapshot(self._sessions[handle], saved[1])
                        del self._snapshots[snapshot_handle]
                    except BaseException as release_failure:
                        # A failed release may have partially retired the
                        # executor receipt. Do not retry that ambiguous handle;
                        # close the owning session and remove all coordinator
                        # references only if teardown completes.
                        try:
                            self.service.close_session(self._sessions[handle])
                            del self._sessions[handle]
                            self._generation_boundaries.pop(handle, None)
                            self._snapshots = {key: value for key, value in self._snapshots.items()
                                               if value[0] != handle}
                        except BaseException as close_failure:
                            add_failure_note(close_failure, 'snapshot receipt retirement also failed: '+failure_summary(release_failure))
                            raise close_failure
            except BaseException as failure:
                cleanup_error = _failure_summary(failure)
            cleanups = [None]*self.world_size
            self.distributed.all_gather_object(cleanups, cleanup_error, group=self.group)
            failures.extend(f'rank {rank} snapshot-create cleanup: {failure}'
                            for rank, failure in enumerate(cleanups) if failure is not None)
        if (command.get('action') in ('load_cache','restore_generation','restore','reset',
                                      'generate','resume_generation','close_session') or
                (command.get('action')=='resume_generation' and command.get('manifest') is not None)):
            cleanup_errors = []
            try:
                if restore_rollback is not None:
                    if failures or mismatch:
                        try:
                            self.service.restore(self._sessions[command['session']],restore_rollback)
                            self._generation_boundaries.pop(command['session'],None)
                            if prior_generation_boundary is not None:
                                self._generation_boundaries[command['session']]=prior_generation_boundary
                        except BaseException as failure:
                            cleanup_errors.append('physical rollback: '+
                                _failure_summary(failure,limit=2048))
                    # Rollback and snapshot retirement are independent cleanup
                    # operations.  In particular, a failed restore must not
                    # skip release of the temporary pre-operation snapshot.
                    try:
                        self.service.release_snapshot(self._sessions[command['session']],restore_rollback)
                    except BaseException as failure:
                        cleanup_errors.append('rollback snapshot release: '+
                            _failure_summary(failure,limit=2048))
                if cleanup_errors or failures or mismatch:
                    # Even a successful rollback does not make this
                    # rank-local session reachable again: the collective
                    # below poisons the coordinator, so no later
                    # close_session request can retire its physical state
                    # or caller-owned snapshots. Tear down the owner on
                    # every rank after disagreement/failure, including a
                    # failure before a rollback snapshot could be acquired.
                    # Keep rollback diagnostics if teardown also fails.
                    try:
                        handle = command['session']
                        if handle in self._sessions:
                            self.service.close_session(self._sessions[handle])
                            del self._sessions[handle]
                            self._generation_boundaries.pop(handle,None)
                            self._snapshots = {key:value for key,value in self._snapshots.items()
                                               if value[0] != handle}
                    except BaseException as failure:
                        cleanup_errors.append('unknown-boundary session teardown: '+
                            _failure_summary(failure,limit=2048))
            except BaseException as failure:
                cleanup_errors.append('rollback setup: '+
                    _failure_summary(failure,limit=2048))
            cleanup_error = '; '.join(cleanup_errors)[:4096] or None
            cleanups = [None]*self.world_size
            self.distributed.all_gather_object(cleanups,cleanup_error,group=self.group)
            cleanup_failures = [f'rank {rank} restore cleanup: {failure}'
                                for rank,failure in enumerate(cleanups) if failure is not None]
            failures.extend(cleanup_failures)
            if cleanup_failures:
                # A temporary snapshot may have been retired on some ranks
                # but not others. Once that cleanup disagreement is known,
                # retire the owning session on every surviving rank instead
                # of leaving a partially usable distributed handle behind.
                teardown_error = None
                try:
                    handle = command['session']
                    if handle in self._sessions:
                        self.service.close_session(self._sessions[handle])
                        del self._sessions[handle]
                        self._generation_boundaries.pop(handle, None)
                        self._snapshots = {key: value for key, value in self._snapshots.items()
                                           if value[0] != handle}
                except BaseException as failure:
                    teardown_error = _failure_summary(failure,limit=2048)
                teardowns = [None]*self.world_size
                self.distributed.all_gather_object(teardowns, teardown_error, group=self.group)
                failures.extend(f'rank {rank} cleanup-disagreement teardown: {failure}'
                                for rank, failure in enumerate(teardowns) if failure is not None)
        if failures or mismatch:
            self._failed = True
            raise RuntimeError('distributed execution failed: '+
                               ('; '.join(failures) if failures else 'committed rank results differ'))
        if summary is not None and summary[0] == 'request_rejected':
            self._sequence += 1
            return _RequestRejected(summary[1], summary[2])
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
                result = self._execute(command)
            except ExecutionCancelled:
                raise
            except BaseException as failure:
                # Any exception escaping a collective request makes the rank
                # schedule ambiguous: peers may have entered a later
                # collective while this rank failed in an earlier one.  A
                # boolean poison flag alone left this rank's service, sessions
                # and retained physical banks reachable behind an unusable
                # coordinator.  Retire those owners just as the worker does
                # after transport loss, while preserving the original failure
                # if cleanup itself also fails.
                self._poison_local_state(failure)
                raise
            if type(result) is _RequestRejected:
                raise ValueError(f'distributed {result.action} request rejected before execution: {result.message}')
            return result
        finally:
            self._active_cancel = None
            self._active_event = None
            self._lock.release()

    def _reject_preflight(self, action, message):
        # JSON's default ensure_ascii escaping can multiply non-ASCII text.
        # Bound the encoded reason so the rejection command always fits even
        # the minimum admitted control envelope.
        message = ((message or 'invalid request').encode('ascii', 'backslashreplace')
                   .decode('ascii')[:1024])
        return self._request('reject', rejected_action=action[:64], rejection=message)

    @_synchronize_root_preflight
    def open_session(self):
        with self._id_lock:
            handle = self._next_id
            self._next_id += 1
        return self._request('open', session=handle)

    @_synchronize_root_preflight
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

    @_synchronize_root_preflight
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

    @_synchronize_root_preflight
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

    @_synchronize_root_preflight
    def reset_session(self, session):
        # `bool` is an `int` subclass and aliases handles 0/1 in dictionaries.
        # Keep reset's ownership check consistent with every other public
        # operation so `reset_session(True)` cannot reset session 1.
        if type(session) is not int or session not in self._sessions:
            raise ValueError('unknown distributed session')
        self._request('reset', session=session)

    @_synchronize_root_preflight
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

    @_synchronize_root_preflight
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
        temporary = None
        published = False
        try:
            encoded = json.dumps(manifest,sort_keys=True,separators=(',', ':'),allow_nan=False).encode('utf-8')
            if len(encoded)>65536:
                raise MemoryError('distributed cache manifest exceeds admission')
            with tempfile.NamedTemporaryFile(dir=generation,delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            _path(generation)
            destination = generation/'manifest.json'
            os.replace(temporary,destination)
            temporary = None
            published = True
            if os.name!='nt':
                descriptor=os.open(generation,os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            return str(destination)
        except BaseException as primary:
            # Before the manifest rename, this fresh generation is only an
            # uncommitted set of rank files and must not be mistaken for a
            # usable recovery point. Once renamed, retain it if a later fsync
            # reports uncertainty: publication may already be durable.
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except BaseException as cleanup:
                    add_failure_note(primary, 'distributed cache manifest staging cleanup failed: '+failure_summary(cleanup))
            if not published:
                try:
                    shutil.rmtree(generation)
                except BaseException as cleanup:
                    add_failure_note(primary, 'uncommitted distributed cache generation cleanup failed: '+failure_summary(cleanup))
            raise

    @_synchronize_root_preflight
    def restore(self, session, snapshot):
        self._check_snapshot(session,snapshot)
        self._request('restore',session=session,snapshot=snapshot)

    @_synchronize_root_preflight
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
                '\x00' in manifest['model_identity'] or
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
                    not 0<len(receipt['identity'])<=4096 or '\x00' in receipt['identity']):
                raise ValueError('invalid distributed rank cache receipt')
        current_identity = self.service.status(session).identity
        if manifest['model_identity'] != current_identity:
            raise ValueError('distributed cache model identity differs from target session')
        return self._request('load_cache',session=session,directory=str(path.parent),
                             manifest=manifest,max_bytes=max_rank_bytes)

    @_synchronize_root_preflight
    def release_snapshot(self, session, snapshot):
        self._check_snapshot(session,snapshot)
        self._request('release_snapshot',session=session,snapshot=snapshot)

    def _check_snapshot(self, session, snapshot):
        if (type(session) is not int or session not in self._sessions or
                type(snapshot) is not int or snapshot not in self._snapshots or
                self._snapshots[snapshot][0] != session):
            raise ValueError('owned distributed session snapshot required')

    @_synchronize_root_preflight
    def close_session(self, session):
        if type(session) is not int or session not in self._sessions:
            raise ValueError('unknown distributed session')
        self._request('close_session', session=session)

    @_synchronize_root_preflight
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
                except BaseException as failure:
                    if not self._failed:
                        self._poison_local_state(failure)
                    raise
        finally:
            self._lock.release()
