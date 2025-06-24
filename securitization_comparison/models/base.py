"""Base securitization model class."""

from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd

FACTORS = ["Transparency", "Liquidity", "Systemic Risk", "Governance",
           "Auditing", "Interoperability"]


class SecuritizationModel(ABC):
    """Abstract base for any securitization evaluation.
    
    This class provides a common interface for evaluating different 
    securitization approaches (traditional vs blockchain-based) across
    multiple standardized factors.
    """

    def __init__(self, loans_df: pd.DataFrame) -> None:
        """Initialize the model with loan data.
        
        Args:
            loans_df: Cleaned DataFrame containing loan information.
        """
        self.loans = loans_df

    @abstractmethod
    def compute_scores(self) -> Dict[str, float]:
        """Return a 0-100 score for each factor in FACTORS.
        
        Returns:
            Dictionary mapping each factor name to its score (0-100).
            
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        pass

    def validate_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Validate that all factors are present and scores are in valid range.
        
        Args:
            scores: Dictionary of factor scores to validate.
            
        Returns:
            The validated scores dictionary.
            
        Raises:
            ValueError: If scores are invalid.
        """
        # Check all factors are present
        missing_factors = set(FACTORS) - set(scores.keys())
        if missing_factors:
            raise ValueError(f"Missing factor scores: {missing_factors}")
        
        # Check scores are in valid range
        for factor, score in scores.items():
            if not 0 <= score <= 100:
                raise ValueError(f"Score for {factor} must be between 0-100, got {score}")
        
        return scores 