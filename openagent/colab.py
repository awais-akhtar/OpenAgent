"""Convenience import surface for launching OpenAgent from Google Colab."""

from .runtimes.colab import ColabLaunch, detect_colab, detect_hardware, launch, recommend_model

__all__ = ["ColabLaunch", "detect_colab", "detect_hardware", "launch", "recommend_model"]