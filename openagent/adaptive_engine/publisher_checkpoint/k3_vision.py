"""Publisher MoonViT3d/patchmergerv2 metadata, independent of native imports."""
from collections.abc import Mapping
import math
from .contracts import UnsupportedContract, UnsupportedPublisherContract
from .storage import positive


def vision_geometry(config):
    vision = config.get("vision_config")
    if not isinstance(vision, Mapping):
        raise ValueError("explicit K3 vision_config required")
    policy = {"activation_func": "gelu_pytorch_tanh", "attn_bias": False,
              "linear_bias": False, "patch_embed_proj_bias": False,
              "merge_type": "sd2_tpool", "mlp_type": "mlp2", "norm_type": "rmsnorm",
              "pos_emb_interpolation_mode": "bilinear", "pos_emb_type": "divided_fixed",
              "mm_projector_type": "patchmergerv2", "projector_hidden_act": "gelu"}
    if any(type(vision.get(k)) is not type(v) or vision.get(k) != v for k, v in policy.items()):
        raise UnsupportedPublisherContract((UnsupportedContract(
            "vision_configuration", "vision_config", "only the explicit bias-free MoonViT3d/patchmergerv2 profile is mapped"),))
    names = {"hidden": "vt_hidden_size", "intermediate": "vt_intermediate_size",
             "qkv_hidden": "qkv_hidden_size", "heads": "vt_num_attention_heads",
             "layers": "vt_num_hidden_layers", "patch": "patch_size",
             "position_height": "init_pos_emb_height", "position_width": "init_pos_emb_width",
             "position_time": "init_pos_emb_time", "text_hidden": "text_hidden_size"}
    geometry = {k: positive(vision.get(v), "vision_config." + v) for k, v in names.items()}
    merge = vision.get("merge_kernel_size")
    if type(merge) is not tuple or len(merge) != 2 or any(type(i) is not int or i <= 0 for i in merge):
        raise ValueError("explicit positive two-axis vision merge required")
    if (geometry["layers"] > 4096 or geometry["text_hidden"] != config["text_config"]["hidden_size"]
            or vision.get("mm_hidden_size") != geometry["hidden"]
            or geometry["qkv_hidden"] % geometry["heads"]):
        raise ValueError("vision geometry does not match wrapper/projector")
    geometry["merge"] = merge
    epsilon = vision.get("projector_ln_eps")
    if type(epsilon) not in (float, int) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("positive finite explicit projector_ln_eps required")
    geometry["projector_epsilon"] = epsilon
    return geometry


def vision_shapes(geometry):
    h, q, mid = geometry["hidden"], geometry["qkv_hidden"], geometry["intermediate"]
    merged = h * geometry["merge"][0] * geometry["merge"][1]
    shapes = {
        "vision_tower.patch_embed.proj.weight": (h, 3, geometry["patch"], geometry["patch"]),
        "vision_tower.patch_embed.pos_emb.weight": (geometry["position_height"], geometry["position_width"], h),
        "vision_tower.encoder.final_layernorm.weight": (h,),
        "mm_projector.proj.0.weight": (merged, merged),
        "mm_projector.proj.2.weight": (geometry["text_hidden"], merged),
        "mm_projector.post_norm.weight": (geometry["text_hidden"],),
    }
    for i in range(geometry["layers"]):
        p = f"vision_tower.encoder.blocks.{i}."
        shapes.update({p + "norm0.weight": (h,), p + "norm1.weight": (h,),
                       p + "wqkv.weight": (3*q, h), p + "wo.weight": (h, q),
                       p + "mlp.fc0.weight": (mid, h), p + "mlp.fc1.weight": (h, mid)})
    return shapes
