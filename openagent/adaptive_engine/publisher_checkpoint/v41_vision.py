"""Pinned V4.1 ViT and aligner parameter declarations, not a vision executor."""
from collections.abc import Mapping

from .contracts import V41_REVISION, V41_SOURCE, V41_CONVERT, geometry, reject

SOURCE = f"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/{V41_REVISION}/inference/vision.py"


def add_vision_contracts(config, hidden, add):
    vision = config.get("vision_config")
    if vision is None:
        return
    if not isinstance(vision, Mapping) or vision.get("model_type") != "deepseek_v41_vision":
        reject("vision_config", "exact V4.1 vision identity required")
    layers = geometry(vision, "num_hidden_layers", 4096)
    width, heads, mid, patch, ratio = (geometry(vision, key) for key in (
        "hidden_size", "num_attention_heads", "intermediate_size", "patch_size", "downsample_ratio"))
    if width % heads or (width // heads) % 4:
        reject("vision_config", "vision heads must support the declared two-axis rotary geometry")

    def dense(name, shape):
        add(name, "v41." + name, shape, evidence=(SOURCE, V41_SOURCE, V41_CONVERT))

    dense("vision.patch_embed.proj.weight", (width, 3 * patch * patch))
    dense("vision.patch_embed.proj.bias", (width,))
    for layer in range(layers):
        prefix = f"vision.blocks.{layer}."
        for name in ("norm1", "norm2"):
            dense(prefix + name + ".weight", (width,))
        for name, rows in (("wqkv", 3 * width), ("wo", width)):
            dense(prefix + "attn." + name + ".weight", (rows, width))
            dense(prefix + "attn." + name + ".bias", (rows,))
        dense(prefix + "mlp.w1.weight", (2 * mid, width))
        dense(prefix + "mlp.w2.weight", (width, mid))
    dense("vision.norm.weight", (width,))
    for name, columns in (("w1", width * ratio * ratio), ("w2", hidden)):
        dense("aligner." + name + ".weight", (hidden, columns))
        dense("aligner." + name + ".bias", (hidden,))
    for name in ("image_start", "image_end", "image_newline"):
        dense(name, (hidden,))
