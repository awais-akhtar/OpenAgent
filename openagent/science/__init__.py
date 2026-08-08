"""Scientific workflows built on OpenAgent's existing runtime and tools."""

from .agent import ScientificAgent
from .experiment import ExperimentPlanner
from .engineering.optimizer import EngineeringOptimizer, DesignProblem
from .memory import ScientificMemory
from .papers import PaperSearch, PaperRecord
from .safety import check_citations, check_consistency, detect_unsupported_claims
from .materials.predictor import MaterialPropertyPredictor
from .tools import COMSOLTool, FenicsTool, LAMMPSTool, OpenFOAMTool, discover_scientific_tools
from .execution import ScientificCodeRunner
from .computing.neural_operator import DeepONet, FNO, PINN
from .verification.scientific_verifier import ScientificVerifier

__all__ = [
    "DesignProblem",
    "EngineeringOptimizer",
    "ExperimentPlanner",
    "ScientificMemory",
    "ScientificCodeRunner",
    "PaperSearch",
    "PaperRecord",
    "check_citations",
    "check_consistency",
    "detect_unsupported_claims",
    "FNO",
    "DeepONet",
    "PINN",
    "FenicsTool",
    "OpenFOAMTool",
    "COMSOLTool",
    "LAMMPSTool",
    "discover_scientific_tools",
    "MaterialPropertyPredictor",
    "ScientificAgent",
    "ScientificVerifier",
]
