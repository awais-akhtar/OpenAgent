"""Explicit spawned rank lifecycle for OpenAgent distributed execution.

Invoke from an importable application's guarded __main__ entry point. Factories
and root drivers must be top-level spawn-picklable callables. This module never
selects a checkpoint, accelerator, port, or tensor partition implicitly.
"""
from datetime import timedelta
import math
import pickle


def _cleanup_failure_message(failures):
    """Build bounded teardown diagnostics without invoking exception __str__."""
    from .distributed_execution import _failure_summary
    return '; '.join(_failure_summary(failure, limit=1024) for failure in failures)


def _worker(rank, world_size, service_factory, root_driver, init_method,
            projection_backend, device_ids, timeout_seconds, max_control_bytes,max_recovery_boundary_bytes):
    import torch
    import torch.distributed as distributed
    from .distributed_execution import DistributedExecutionService
    service = coordinator = projection_group = None
    control_ready = False
    primary_error = None
    try:
        if projection_backend == 'nccl':
            torch.cuda.set_device(device_ids[rank])
        distributed.init_process_group(backend='gloo',init_method=init_method,
            rank=rank,world_size=world_size,timeout=timedelta(seconds=timeout_seconds))
        control_ready = True
        projection_group = distributed.new_group(ranks=list(range(world_size)),
            backend=projection_backend,timeout=timedelta(seconds=timeout_seconds))
        service = service_factory(rank,world_size,projection_group)
        coordinator = DistributedExecutionService(service,
            control_group=distributed.group.WORLD,max_control_bytes=max_control_bytes,
            max_recovery_boundary_bytes=max_recovery_boundary_bytes)
        if rank == 0:
            driver_error = None
            try:
                root_driver(coordinator)
            except BaseException as error:
                driver_error = error
                raise
            finally:
                # Healthy peers are waiting on the ordered control channel.
                # A poisoned collective is shut down locally below instead.
                if not coordinator._failed and not coordinator._closed:
                    try:
                        coordinator.close()
                    except BaseException as error:
                        if driver_error is None:
                            raise
                        driver_error.add_note(
                            'ordered rank shutdown failed: '+_cleanup_failure_message((error,)))
        else:
            coordinator.serve()
    except BaseException as error:
        primary_error = error
        raise
    finally:
        failures = []
        if service is None and primary_error is not None:
            cleanup = getattr(primary_error, 'cleanup', None)
            if callable(cleanup):
                try:
                    cleanup()
                except BaseException as error:
                    failures.append(error)
        if service is not None:
            try:
                service.close()
            except BaseException as error:
                failures.append(error)
        if projection_group is not None:
            try:
                distributed.destroy_process_group(projection_group)
            except BaseException as error:
                failures.append(error)
        if control_ready:
            try:
                distributed.destroy_process_group()
            except BaseException as error:
                failures.append(error)
        if failures:
            message = _cleanup_failure_message(failures)
            if primary_error is not None:
                primary_error.add_note('rank shutdown failures: '+message)
            else:
                raise RuntimeError('rank shutdown incomplete: '+message) from failures[0]


def launch_distributed_execution(service_factory, root_driver, *, world_size,
        init_method, projection_backend, device_ids=(), timeout_seconds=120,
        max_control_bytes=1024*1024,max_recovery_boundary_bytes=64*1024**2):
    """Spawn ranks, join them, and propagate worker/cleanup failures.

    service_factory(rank, world_size, projection_group) constructs an owned
    NativeExecutionService and may attach rank-local projection banks. It must
    own/close any resources it creates beyond caller-borrowed process groups.
    root_driver(coordinator) issues the actual session requests on rank zero.
    Driver results must be persisted or forwarded by the driver's application;
    multiprocessing does not return its Python value to the launching process.

    The rendezvous URL and device assignment are operator supplied. All ranks
    must use compatible partition contracts. This launcher does not shard
    otherwise resident model inventories or certify any accelerator path.
    """
    if type(world_size) is not int or not 1<=world_size<=64:
        raise ValueError('world_size must contain one to 64 ranks')
    if projection_backend not in ('gloo','nccl'):
        raise ValueError('explicit Gloo or NCCL projection backend required')
    if (type(init_method) is not str or len(init_method)>4096 or
            not init_method.startswith(('tcp://','file://'))):
        raise ValueError('explicit bounded TCP or file rendezvous URL required')
    if (type(timeout_seconds) not in (int,float) or not math.isfinite(timeout_seconds) or
            not 1<=timeout_seconds<=3600):
        raise ValueError('process-group timeout must be between one second and one hour')
    if type(max_control_bytes) is not int or not 4096<=max_control_bytes<=64*1024*1024:
        raise ValueError('bounded distributed control-message budget required')
    if type(max_recovery_boundary_bytes) is not int or max_recovery_boundary_bytes<=0:
        raise ValueError('positive retained generation boundary admission required')
    if type(device_ids) is not tuple or any(type(value) is not int or value<0 for value in device_ids):
        raise ValueError('explicit tuple of nonnegative device indices required')
    if projection_backend=='nccl':
        if len(device_ids)!=world_size or len(set(device_ids))!=world_size:
            raise ValueError('NCCL ranks require distinct explicitly assigned devices')
    elif device_ids:
        raise ValueError('CPU Gloo launcher does not accept CUDA device assignments')
    if not callable(service_factory) or not callable(root_driver):
        raise TypeError('importable service factory and root driver required')
    # This checks transportability only; it never invokes either callable.
    pickle.dumps((service_factory,root_driver),protocol=pickle.HIGHEST_PROTOCOL)
    import torch.multiprocessing as multiprocessing
    return multiprocessing.spawn(_worker,nprocs=world_size,join=True,
        args=(world_size,service_factory,root_driver,init_method,projection_backend,
              device_ids,float(timeout_seconds),max_control_bytes,max_recovery_boundary_bytes))
