"""Securitization Comparison Package.

A package for comparing traditional securitization with blockchain-based securitization
using real-world mortgage loan data.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .data_loader import load_fannie_mae
from .models.base import SecuritizationModel, FACTORS
from .models.traditional import TraditionalModel
from .models.blockchain import BlockchainModel
from .viz import plot_scores
from .statistical_analysis import StatisticalAnalyzer, run_statistical_analysis

__all__ = [
    "load_fannie_mae",
    "SecuritizationModel",
    "FACTORS",
    "TraditionalModel", 
    "BlockchainModel",
    "plot_scores",
    "StatisticalAnalyzer",
    "run_statistical_analysis",
] 