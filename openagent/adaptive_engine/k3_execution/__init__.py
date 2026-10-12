"""Explicit K3 execution, independent of publisher checkpoint admission."""
from .executor import K3Config, K3Execution, K3Snapshot, tensor_shapes
from .vision import K3VisionConfig, vision_tensor_shapes
from .preprocessing import preprocess_images
from .exact_admission import K3ExactEvidence
from .video_input import K3VideoInput, K3PreparedMedia, prepare_media
from .packed import MappedMXFP4Experts, PackedTensorSpan
from .checkpoints import save_k3_checkpoint, load_k3_checkpoint
from .batch import K3BatchExecution, K3BatchResult, K3BatchSnapshot
from .publisher_projections import K3PublisherProjections
from .metal_mixer import MetalK3Policy, MetalK3Mixer

__all__ = ["K3Config", "K3Execution", "K3Snapshot", "K3ExactEvidence", "tensor_shapes", "K3VisionConfig", "vision_tensor_shapes",
           "preprocess_images", "K3VideoInput", "K3PreparedMedia", "prepare_media", "MappedMXFP4Experts", "PackedTensorSpan",
           "save_k3_checkpoint", "load_k3_checkpoint",
           "K3BatchExecution", "K3BatchResult", "K3BatchSnapshot", "K3PublisherProjections", "MetalK3Policy", "MetalK3Mixer"]
