"""Numerical boundary checks; these are not whole-model certification."""
import importlib.util
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is not installed')
class K3TorchArithmeticContracts(unittest.TestCase):
    def setUp(self):
        import torch
        from openagent.adaptive_engine.k3_execution.bf16_arithmetic import K3BF16Arithmetic
        self.torch = torch
        self.arithmetic = K3BF16Arithmetic(device='cpu', torch_version=str(torch.__version__),
                                         max_bytes=4*1024*1024)

    def test_router_ties_preserve_batched_topk_slots(self):
        from openagent.adaptive_engine.k3_execution.routing import TorchK3Router
        torch = self.torch
        router = TorchK3Router(max_bytes=65536, expected_torch_version=str(torch.__version__))
        scores = np.full(16, .5, dtype=np.float32)
        correction = np.zeros_like(scores)
        actual = router.route(scores, correction, selected=4, groups=4, selected_groups=2, scale=1.25)
        original = torch.tensor(scores)[None]
        choice = original+torch.tensor(correction)[None]
        grouped = choice.reshape(1, 4, 4).topk(2, dim=-1).values.sum(-1)
        groups = grouped.topk(2, dim=-1, sorted=False).indices
        mask = torch.zeros((1, 4), dtype=torch.bool).scatter_(1, groups, True)
        ids = choice.masked_fill(~mask.repeat_interleave(4, dim=-1), -float('inf')).topk(
            4, dim=-1, sorted=False).indices
        weights = original.gather(1, ids)
        weights = weights/(weights.sum(-1, keepdim=True)+1e-20)*1.25
        np.testing.assert_array_equal(actual.ids, ids[0].numpy())
        np.testing.assert_array_equal(actual.weights, weights[0].numpy())

    def test_eager_attention_bf16_boundaries(self):
        torch = self.torch
        rng = np.random.default_rng(731)
        query = rng.normal(size=(2, 8)).astype(np.float32)
        keys = rng.normal(size=(5, 2, 8)).astype(np.float32)
        values = rng.normal(size=(5, 2, 4)).astype(np.float32)
        q = torch.tensor(query, dtype=torch.bfloat16)[None, :, None]
        k = torch.tensor(keys, dtype=torch.bfloat16).permute(1, 0, 2)[None]
        v = torch.tensor(values, dtype=torch.bfloat16).permute(1, 0, 2)[None]
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k)*8**(-.5)
        probs = scores.float().softmax(-1).to(torch.bfloat16)
        expected = torch.einsum('bhqk,bhkd->bhqd', probs, v)[0, :, 0].float().numpy()
        np.testing.assert_array_equal(self.arithmetic.attention(query, keys, values), expected)

    def test_packed_bf16_scale_boundary(self):
        torch = self.torch
        packed = np.full((3, 16), 0x21, dtype=np.uint8)
        scales = np.array([[0], [127], [128]], dtype=np.uint8)
        vector = np.linspace(-1, 1, 32, dtype=np.float32)
        decoded = torch.tensor([.5, 1.]*16, dtype=torch.bfloat16).repeat(3, 1)
        scale = torch.pow(2., torch.tensor(scales.astype(np.int32)-127, dtype=torch.bfloat16))
        expected = torch.nn.functional.linear(torch.tensor(vector, dtype=torch.bfloat16),
            decoded*scale).float().numpy()
        np.testing.assert_array_equal(self.arithmetic.packed_matrix(packed, scales, vector,
            rows=3, cols=32), expected)

    def test_ambient_autocast_is_rejected(self):
        with self.torch.autocast('cpu', dtype=self.torch.bfloat16):
            with self.assertRaisesRegex(ValueError, 'autocast'):
                self.arithmetic.cast(np.ones(8, np.float32))

    def test_batched_router_and_expert_order(self):
        from openagent.adaptive_engine.k3_execution.routing import TorchK3Router
        torch = self.torch
        router = TorchK3Router(max_bytes=65536, expected_torch_version=str(torch.__version__))
        scores = np.array([[.2, .4, .4, .1], [.8, .1, .3, .8]], np.float32)
        correction = np.array([0, .1, 0, -.1], np.float32)
        route = router.route_batch(scores, correction, selected=2, groups=1, selected_groups=1, scale=1)
        source = torch.tensor(scores)
        ids = (source+torch.tensor(correction)).topk(2, dim=-1, sorted=False).indices
        weights = source.gather(1, ids)
        weights /= weights.sum(-1, keepdim=True)+1e-20
        np.testing.assert_array_equal(route.ids, ids.numpy())
        np.testing.assert_array_equal(route.weights, weights.numpy())
        np.testing.assert_array_equal(router.expert_order(route.ids), ids.flatten().argsort().numpy())
        outputs = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        expected = (torch.tensor(outputs)*weights[..., None]).sum(1).numpy()
        np.testing.assert_array_equal(router.aggregate_batch(outputs, route.weights), expected)

    def test_packed_batch_projection_preserves_row_tile_output_layout(self):
        torch = self.torch
        packed = np.full((3, 16), 0x21, dtype=np.uint8)
        scales = np.array([[126], [127], [128]], dtype=np.uint8)
        values = np.arange(64, dtype=np.float32).reshape(2, 32)/32
        decoded = torch.tensor([.5, 1.]*16, dtype=torch.bfloat16).repeat(3, 1)
        scale = torch.pow(2., torch.tensor(scales.astype(np.int32)-127, dtype=torch.bfloat16))
        expected = torch.nn.functional.linear(torch.tensor(values, dtype=torch.bfloat16),
                                              decoded*scale).float().numpy()
        np.testing.assert_array_equal(self.arithmetic.packed_matrix(packed, scales, values,
            rows=3, cols=32), expected)
