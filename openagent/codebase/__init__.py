"""Autonomous software-engineering workspaces for OpenAgent."""

from .service import CodebaseService
from .frameworks import framework_catalog, toolchain_preflight
from .sdk import CodeQSDK, DeclarativeFrameworkAdapter, FrameworkAdapter, GenerationContract, GenerationRequest, GeneratorRegistry

__all__ = [
    "CodebaseService", "CodeQSDK", "DeclarativeFrameworkAdapter", "FrameworkAdapter",
    "GenerationContract", "GenerationRequest", "GeneratorRegistry", "framework_catalog", "toolchain_preflight",
]
