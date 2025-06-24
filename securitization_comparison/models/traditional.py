"""Traditional securitization model implementation."""

import logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from .base import SecuritizationModel, FACTORS

logger = logging.getLogger(__name__)


class TraditionalModel(SecuritizationModel):
    """Traditional securitization evaluation model.
    
    This model evaluates traditional securitization approaches using
    heuristics based on historical market characteristics and regulatory
    frameworks.
    """

    def __init__(self, loans_df: pd.DataFrame, config_path: Optional[str] = None) -> None:
        """Initialize the traditional model.
        
        Args:
            loans_df: Cleaned DataFrame containing loan information.
            config_path: Optional path to YAML configuration file.
        """
        super().__init__(loans_df)
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
            
        Returns:
            Configuration dictionary.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "_traditional_weighting.yaml"
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values.
        
        Returns:
            Default configuration dictionary.
        """
        return {
            "transparency": {
                "base_score": 30,
                "disclosure_weight": 70,
                "typical_disclosure_rate": 0.4
            },
            "liquidity": {
                "cpr_alpha": 5.0,
                "cpr_scale": 0.1
            },
            "systemic_risk": {
                "concentration_weight": 0.6,
                "dispersion_weight": 0.4
            },
            "governance": {
                "fixed_score": 50
            },
            "auditing": {
                "regulated_score": 60
            },
            "interoperability": {
                "proprietary_penalty": 60
            }
        }

    def compute_scores(self) -> Dict[str, float]:
        """Compute scores for all factors using traditional model heuristics.
        
        Returns:
            Dictionary of factor scores (0-100).
        """
        scores = {}
        
        scores["Transparency"] = self._compute_transparency()
        scores["Liquidity"] = self._compute_liquidity()
        scores["Systemic Risk"] = self._compute_systemic_risk()
        scores["Governance"] = self._compute_governance()
        scores["Auditing"] = self._compute_auditing()
        scores["Interoperability"] = self._compute_interoperability()
        
        return self.validate_scores(scores)

    def _compute_transparency(self) -> float:
        """Compute transparency score.
        
        Transparency = base_score + (% fields publicly disclosed × weight)
        
        Returns:
            Transparency score (0-100).
        """
        config = self.config["transparency"]
        base = config["base_score"]
        weight = config["disclosure_weight"]
        disclosure_rate = config["typical_disclosure_rate"]
        
        # In traditional securitization, disclosure is limited
        # Calculate based on available data fields vs total possible fields
        total_fields = len(self.loans.columns)
        public_fields = max(1, total_fields // 3)  # Assume 1/3 are publicly disclosed
        disclosure_pct = public_fields / total_fields
        
        score = base + (disclosure_rate * weight)
        return min(100, max(0, score))

    def _compute_liquidity(self) -> float:
        """Compute liquidity score using CPR (Conditional Prepayment Rate) proxy.
        
        Uses sigmoid function scaled to 0-100.
        
        Returns:
            Liquidity score (0-100).
        """
        config = self.config["liquidity"]
        alpha = config["cpr_alpha"]
        scale = config["cpr_scale"]
        
        # Estimate CPR from loan characteristics
        if "orig_rate" in self.loans.columns and "orig_term" in self.loans.columns:
            # Higher rates and shorter terms typically mean lower liquidity
            avg_rate = self.loans["orig_rate"].mean()
            avg_term = self.loans["orig_term"].mean()
            
            # Normalize and compute proxy CPR
            rate_norm = min(avg_rate / 10.0, 1.0)  # Normalize by 10%
            term_norm = min(avg_term / 360.0, 1.0)  # Normalize by 30 years
            
            cpr_proxy = (rate_norm + (1 - term_norm)) / 2
        else:
            # Fallback to random estimation if columns not available
            cpr_proxy = np.random.beta(2, 5)  # Conservative estimate
        
        # Apply sigmoid transformation
        sigmoid_value = 1 / (1 + np.exp(-alpha * (cpr_proxy - scale)))
        return sigmoid_value * 100

    def _compute_systemic_risk(self) -> float:
        """Compute systemic risk score.
        
        Inverse of collateral dispersion + origination balance concentration.
        
        Returns:
            Systemic risk score (0-100).
        """
        config = self.config["systemic_risk"]
        conc_weight = config["concentration_weight"]
        disp_weight = config["dispersion_weight"]
        
        # Calculate concentration risk
        if "orig_bal" in self.loans.columns:
            balance_gini = self._calculate_gini_coefficient(self.loans["orig_bal"])
            concentration_risk = balance_gini * 100
        else:
            concentration_risk = 50  # Default moderate concentration
        
        # Calculate geographic dispersion risk
        if "state" in self.loans.columns:
            state_counts = self.loans["state"].value_counts()
            state_concentration = (state_counts.iloc[0] / len(self.loans)) * 100
            dispersion_risk = min(state_concentration, 100)
        else:
            dispersion_risk = 30  # Default moderate dispersion
        
        # Combine risks (higher values = higher risk, so invert for score)
        total_risk = (conc_weight * concentration_risk + 
                     disp_weight * dispersion_risk)
        
        # Invert risk to get score (lower risk = higher score)
        return 100 - min(total_risk, 100)

    def _calculate_gini_coefficient(self, values: pd.Series) -> float:
        """Calculate Gini coefficient for measuring concentration.
        
        Args:
            values: Series of values to calculate Gini coefficient for.
            
        Returns:
            Gini coefficient (0-1).
        """
        values_clean = values.dropna().sort_values()
        n = len(values_clean)
        
        if n <= 1:
            return 0.0
        
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * values_clean)) / (n * values_clean.sum()) - (n + 1) / n
        return max(0, min(1, gini))

    def _compute_governance(self) -> float:
        """Compute governance score.
        
        Returns fixed score as placeholder.
        
        Returns:
            Governance score (0-100).
        """
        return float(self.config["governance"]["fixed_score"])

    def _compute_auditing(self) -> float:
        """Compute auditing score.
        
        Returns score based on regulated audit assumptions.
        
        Returns:
            Auditing score (0-100).
        """
        return float(self.config["auditing"]["regulated_score"])

    def _compute_interoperability(self) -> float:
        """Compute interoperability score.
        
        Returns score accounting for proprietary data standards.
        
        Returns:
            Interoperability score (0-100).
        """
        penalty = self.config["interoperability"]["proprietary_penalty"]
        return 100 - penalty 