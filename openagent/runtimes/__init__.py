"""Environment-specific launch helpers sharing the core OpenAgent runtime."""

from .colab import ColabLaunch, detect_colab, launch, recommend_model

__all__ = ["ColabLaunch", "detect_colab", "launch", "recommend_model"]