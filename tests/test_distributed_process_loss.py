"""Bounded real Gloo behavior when a launched rank exits unexpectedly."""
import importlib.util
import os
import socket
import time
import unittest

import numpy as np

from openagent.adaptive_engine.distributed_launcher import launch_distributed_execution


def _exit_second_rank(rank, world_size, _group):
    if rank == 1:
        # Simulate abrupt process loss after process-group creation but before
        # the coordinator's constructor agreement. The surviving rank must
        # leave its first control collective within the configured timeout.
        os._exit(73)
    from openagent.adaptive_engine.execution_service import (
        ExecutionLimits, NativeExecutionService)
    return NativeExecutionService(lambda: None, ExecutionLimits(1, 1))


def _unexpected_driver(_coordinator):
    raise AssertionError("driver must not run after a rank is lost during admission")


class _OperationLossExecutor:
    def __init__(self, rank):
        self.rank = rank
        self.identity = 'process-loss-fixture-v1'
        self.tokens = ()

    @property
    def position(self):
        return len(self.tokens)

    def forward(self, tokens):
        if self.rank == 1:
            os._exit(74)
        self.tokens += tuple(tokens)
        return np.zeros((len(tokens), 4), dtype=np.float32)

    def snapshot(self):
        return type('Snapshot', (), {
            'identity': self.identity, 'tokens': self.tokens,
            'position': len(self.tokens),
        })()

    def restore(self, snapshot):
        self.tokens = tuple(snapshot.tokens)

    def reset(self):
        self.tokens = ()

    def close(self):
        pass


def _operation_loss_service_factory(rank, _world_size, _projection_group):
    from openagent.adaptive_engine.execution_service import (
        ExecutionLimits, NativeExecutionService)
    return NativeExecutionService(lambda: _OperationLossExecutor(rank),
                                  ExecutionLimits(4, 32, max_new_tokens=2))


def _operation_loss_driver(coordinator):
    session = coordinator.open_session()
    coordinator.generate(session, [1], max_new_tokens=1)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                     "real Gloo process-loss check requires CPU Torch")
class DistributedProcessLossTests(unittest.TestCase):
    def test_rank_exit_during_admission_is_bounded_by_group_timeout(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        timeout_seconds = 4
        started = time.monotonic()
        with self.assertRaises(Exception):
            launch_distributed_execution(
                _exit_second_rank, _unexpected_driver, world_size=2,
                init_method=f"tcp://127.0.0.1:{port}",
                projection_backend="gloo", timeout_seconds=timeout_seconds,
                max_control_bytes=4096, max_recovery_boundary_bytes=4096)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, timeout_seconds * 3,
                        "a lost rank left the surviving worker blocked beyond the timeout bound")

    def test_rank_exit_during_generation_poisons_group_within_timeout(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        timeout_seconds = 4
        started = time.monotonic()
        with self.assertRaises(Exception):
            launch_distributed_execution(
                _operation_loss_service_factory, _operation_loss_driver, world_size=2,
                init_method=f"tcp://127.0.0.1:{port}",
                projection_backend="gloo", timeout_seconds=timeout_seconds,
                max_control_bytes=4096, max_recovery_boundary_bytes=4096)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, timeout_seconds * 3,
                        "a lost rank left a live generation collective blocked beyond the timeout bound")


if __name__ == "__main__":
    unittest.main()
