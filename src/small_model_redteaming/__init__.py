"""
Small Model Red Teaming

A comprehensive AI redteaming framework for testing small language models
across multiple safety and alignment dimensions.
"""

__version__ = "0.1.2"

# Core imports
from .deceptive_alignment import DeceptiveAlignmentTest
from .eval_awareness import GradientDeveloperPrompt, MetaAwareness, AwarenessImpactTest
from .set_up import Model, PreGenAnalyzer
from .data_exfiltration import DataExfiltrationTest
from .visualizer import (
    plot_awareness_scores,
    plot_agreement_comparison,
    plot_confidence_distribution,
    plot_deceptive_alignment_results,
    plot_exfiltration_results,
    plot_preference_heatmap,
    plot_test_comparison,
)

__all__ = [
    # Core classes
    "DeceptiveAlignmentTest",
    "GradientDeveloperPrompt",
    "MetaAwareness",
    "AwarenessImpactTest",
    "Model",
    "PreGenAnalyzer",
    # Data exfiltration
    "DataExfiltrationTest",
    # Visualization functions
    "plot_awareness_scores",
    "plot_agreement_comparison",
    "plot_confidence_distribution",
    "plot_deceptive_alignment_results",
    "plot_exfiltration_results",
    "plot_preference_heatmap",
    "plot_test_comparison",
    # Version
    "__version__",
]
