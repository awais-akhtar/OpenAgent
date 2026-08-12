"""Local fusion of scientific text, sensor data, images, and simulations."""

from .fusion import MultimodalScientificContext
from .image import analyse_image_bytes, observe_with_gateway

__all__ = ["MultimodalScientificContext", "analyse_image_bytes", "observe_with_gateway"]
