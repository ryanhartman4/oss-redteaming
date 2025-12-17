"""
Small Model Red Teaming

A comprehensive AI redteaming framework for testing small language models
across multiple safety and alignment dimensions.
"""

__version__ = "0.1.1"

# Core imports
from .deceptive_alignment import DeceptiveAlignmentTest
from .eval_awareness import GradientDeveloperPrompt, MetaAwareness, AwarenessImpactTest
from .set_up import Model, PreGenAnalyzer

__all__ = [
    "DeceptiveAlignmentTest",
    "GradientDeveloperPrompt",
    "MetaAwareness",
    "AwarenessImpactTest",
    "Model",
    "PreGenAnalyzer",
    "__version__",
]
