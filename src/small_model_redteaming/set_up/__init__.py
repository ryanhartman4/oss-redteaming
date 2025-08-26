"""
Model utilities and interfaces for the small model redteaming framework.

This module provides the core Model class for API interactions and the 
PreGenAnalyzer class that implements the lm_eval.LM interface for 
standardized evaluation tasks.
"""

from .model_utils import Model, PreGenAnalyzer

__all__ = [
    "Model",
    "PreGenAnalyzer",
]
