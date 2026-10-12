import unittest

from openagent.adaptive_engine.olmoe import OlmoeConfig, validate_olmoe_tensors


class TensorContractTests(unittest.TestCase):
    def setUp(self):
        self.config = OlmoeConfig(4, 3, 1, 2, 2, 1, 1, 8, 16, 1e-5, 10000, False, 7)
        shapes = {
            "model.embed_tokens.weight": (8, 4),
            "lm_head.weight": (8, 4),
            "model.norm.weight": (4,),
        }
        prefix = "model.layers.0."
        for name in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
            shapes[prefix + name + ".weight"] = (4,)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            shapes[prefix + "self_attn." + name + ".weight"] = (4, 4)
        shapes[prefix + "mlp.gate.weight"] = (1, 4)
        for name, shape in (("gate_proj", (3, 4)), ("up_proj", (3, 4)), ("down_proj", (4, 3))):
            shapes[prefix + "mlp.experts.0." + name + ".weight"] = shape
        self.tensors = {name: (None, {"shape": list(shape), "dtype": "F32"}, 0) for name, shape in shapes.items()}

    def test_complete_contract(self):
        validate_olmoe_tensors(self.config, self.tensors)

    def test_same_count_with_wrong_name_is_rejected(self):
        self.tensors["wrong"] = self.tensors.pop("lm_head.weight")
        with self.assertRaisesRegex(ValueError, "tensor names"):
            validate_olmoe_tensors(self.config, self.tensors)

    def test_wrong_shape_is_rejected(self):
        self.tensors["lm_head.weight"][1]["shape"] = [4, 8]
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_olmoe_tensors(self.config, self.tensors)

    def test_unsupported_dtype_is_rejected(self):
        self.tensors["lm_head.weight"][1]["dtype"] = "I64"
        with self.assertRaisesRegex(ValueError, "dtype"):
            validate_olmoe_tensors(self.config, self.tensors)
