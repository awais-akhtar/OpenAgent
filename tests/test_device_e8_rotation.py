import importlib.util
import struct
import unittest
from types import SimpleNamespace

import numpy as np

from openagent.adaptive_engine.device_e8_rotation import rotate_device_e8
from openagent.adaptive_engine.device_packed_decode import decode_device_expert
from openagent.adaptive_engine.e8_lattice_expert import (
    DTYPE, LatticeContract, hadamard_transform, root_codebook,
)
from openagent.adaptive_engine.execution_service import ExecutionCancelled


class _Tensor:
    def __init__(self, value, *, device=None, dtype=None):
        self.value = np.asarray(value)
        self.device = device or SimpleNamespace(type='cpu')
        self.dtype = dtype if dtype is not None else _NumpyTorch.float32

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def shape(self):
        return self.value.shape

    @property
    def T(self):
        return _Tensor(self.value.T, device=self.device, dtype=self.dtype)

    def numel(self):
        return self.value.size

    def __getitem__(self, key):
        return _Tensor(self.value[key], device=self.device, dtype=self.dtype)

    def __setitem__(self, key, value):
        self.value[key] = value.value if isinstance(value, _Tensor) else value

    def all(self):
        return _Tensor(np.asarray(self.value.all()), device=self.device, dtype=bool)

    def item(self):
        return self.value.item()


class _NumpyTorch:
    float32 = object()

    @staticmethod
    def isfinite(value):
        return _Tensor(np.isfinite(value.value), device=value.device, dtype=bool)

    @staticmethod
    def tensor(value, *, device, dtype):
        return _Tensor(value, device=device, dtype=dtype)

    @staticmethod
    def empty_like(value):
        return _Tensor(np.empty_like(value.value), device=value.device,
                       dtype=value.dtype)

    @staticmethod
    def matmul(left, right):
        return _Tensor(left.value @ right.value, device=left.device,
                       dtype=left.dtype)


class DeviceE8RotationSourceContractTest(unittest.TestCase):
    def test_numpy_backed_tensor_obeys_forward_transform_contract(self):
        transform = hadamard_transform()
        source = np.arange(2 * 24, dtype=np.float32).reshape(2, 24) / 7
        values = _Tensor(source)
        actual = rotate_device_e8(_NumpyTorch, values, transform=transform,
                                  max_bytes=values.numel() * 8 + 8192)
        expected = source.reshape(2, 3, 8) @ transform.T
        np.testing.assert_allclose(actual.value.reshape(2, 3, 8), expected,
                                   rtol=1e-6, atol=1e-6)

    def test_decoded_codebook_weights_pair_with_one_activation_rotation(self):
        contract = LatticeContract(root_codebook(), hadamard_transform())
        records = []
        expected_matrices = []
        for matrix_index in range(3):
            matrix = np.empty((8, 8), np.float32)
            for row in range(8):
                index = 1 + (row + matrix_index) % (len(contract.codebook) - 1)
                scale = (-0.125 if (row + matrix_index) % 2 else 0.25)
                records.append(struct.pack('<Hf', index, scale))
                matrix[row] = contract.codebook[index] * np.float32(scale)
            expected_matrices.append(matrix)
        payload = b''.join(records)
        decoded = decode_device_expert(payload, hidden=8, intermediate=8,
            dtype=DTYPE, max_bytes=1 << 20, lattice_contract=contract)
        for actual, expected in zip(decoded, expected_matrices):
            np.testing.assert_array_equal(actual, expected)

        values = np.arange(16, dtype=np.float32).reshape(2, 8) / 7
        rotated = rotate_device_e8(_NumpyTorch, _Tensor(values),
            transform=contract.transform, max_bytes=1 << 20).value
        for actual, codebook_weights in zip(decoded, expected_matrices):
            persisted_weights = codebook_weights @ contract.transform
            reference = values @ persisted_weights.T
            device_path = rotated @ actual.T
            np.testing.assert_allclose(device_path, reference, rtol=2e-6, atol=2e-6)

    def test_budget_and_cancellation_fail_before_rotation(self):
        values = _Tensor(np.ones((2, 16), dtype=np.float32))
        with self.assertRaisesRegex(MemoryError, 'buffers exceed admission'):
            rotate_device_e8(_NumpyTorch, values, transform=hadamard_transform(),
                             max_bytes=128)

        class Cancelled:
            def is_set(self):
                return True

        with self.assertRaises(ExecutionCancelled):
            rotate_device_e8(_NumpyTorch, values, transform=hadamard_transform(),
                             max_bytes=8192, cancel=Cancelled())


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional Torch runtime unavailable')
class DeviceE8RotationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def test_matches_persisted_forward_transform_group_by_group(self):
        torch = self.torch
        transform = hadamard_transform()
        values = torch.arange(2 * 24, dtype=torch.float32).reshape(2, 24) / 7
        actual = rotate_device_e8(torch, values, transform=transform,
                                  max_bytes=values.numel() * 8 + 8192)
        expected = values.numpy().reshape(2, 3, 8) @ transform.T
        np.testing.assert_allclose(actual.numpy().reshape(2, 3, 8), expected,
                                   rtol=1e-6, atol=1e-6)

    def test_rejects_insufficient_work_and_nonorthogonal_contract(self):
        torch = self.torch
        values = torch.ones((2, 16), dtype=torch.float32)
        with self.assertRaisesRegex(MemoryError, 'buffers exceed admission'):
            rotate_device_e8(torch, values, transform=hadamard_transform(),
                             max_bytes=128)
        invalid = np.eye(8, dtype=np.float32)
        invalid[0, 0] = 2
        with self.assertRaisesRegex(ValueError, 'orthogonal transform required'):
            rotate_device_e8(torch, values, transform=invalid, max_bytes=8192)

    def test_cancellation_is_checked_before_device_allocation(self):
        torch = self.torch
        values = torch.ones((1, 8), dtype=torch.float32)

        class Cancelled:
            def is_set(self):
                return True

        with self.assertRaises(ExecutionCancelled):
            rotate_device_e8(torch, values, transform=hadamard_transform(),
                             max_bytes=8192, cancel=Cancelled())


if __name__ == '__main__':
    unittest.main()
