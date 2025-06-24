"""Blockchain/tokenized securitization model implementation."""

import logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from .base import SecuritizationModel, FACTORS

logger = logging.getLogger(__name__)


class BlockchainModel(SecuritizationModel):
    """Blockchain/tokenized securitization evaluation model.
    
    This model evaluates blockchain-based securitization approaches using
    assumptions typical of on-chain SPVs and decentralized finance protocols.
    """

    def __init__(self, loans_df: pd.DataFrame, config_path: Optional[str] = None) -> None:
        """Initialize the blockchain model.
        
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
            config_path = Path(__file__).parent / "_blockchain_weighting.yaml"
        
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
                "on_chain_score": 90
            },
            "liquidity": {
                "amm_depth_shape": 2.0,
                "amm_depth_scale": 0.8,
                "min_score": 20,
                "max_score": 95
            },
            "systemic_risk": {
                "oracle_dependency_weight": 0.4,
                "smart_contract_weight": 0.6,
                "base_oracle_risk": 15,
                "base_contract_risk": 10
            },
            "governance": {
                "participation_shape": 1.5,
                "participation_scale": 0.6,
                "min_score": 30,
                "max_score": 90
            },
            "auditing": {
                "real_time_score": 80
            },
            "interoperability": {
                "erc_compliance_score": 85
            }
        }

    def compute_scores(self) -> Dict[str, float]:
        """Compute scores for all factors using blockchain model assumptions.
        
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
        
        Assumes high transparency due to on-chain visibility.
        
        Returns:
            Transparency score (0-100).
        """
        return float(self.config["transparency"]["on_chain_score"])

    def _compute_liquidity(self) -> float:
        """Compute liquidity score using simulated AMM depth.
        
        Uses gamma distribution to simulate AMM liquidity depth.
        
        Returns:
            Liquidity score (0-100).
        """
        config = self.config["liquidity"]
        shape = config["amm_depth_shape"]
        scale = config["amm_depth_scale"]
        min_score = config["min_score"]
        max_score = config["max_score"]
        
        # Simulate AMM depth based on pool characteristics
        # Higher loan diversity typically means better AMM performance
        if len(self.loans) > 1000:
            pool_factor = 1.2  # Large pools get bonus
        elif len(self.loans) > 100:
            pool_factor = 1.0
        else:
            pool_factor = 0.8  # Small pools penalized
        
        # Generate random AMM depth score using gamma distribution
        np.random.seed(42)  # For reproducibility
        raw_score = np.random.gamma(shape, scale) * pool_factor
        
        # Normalize to score range
        normalized = min(max(raw_score, 0), 1)
        return min_score + (normalized * (max_score - min_score))

    def _compute_systemic_risk(self) -> float:
        """Compute systemic risk score.
        
        Risk = 100 - (oracle dependency score + smart contract risk score)
        
        Returns:
            Systemic risk score (0-100).
        """
        config = self.config["systemic_risk"]
        oracle_weight = config["oracle_dependency_weight"]
        contract_weight = config["smart_contract_weight"]
        base_oracle_risk = config["base_oracle_risk"]
        base_contract_risk = config["base_contract_risk"]
        
        # Calculate oracle dependency risk
        # More diverse data sources = lower oracle risk
        if "state" in self.loans.columns:
            state_diversity = len(self.loans["state"].unique())
            oracle_risk = base_oracle_risk * max(0.5, 1 - (state_diversity / 50))
        else:
            oracle_risk = base_oracle_risk
        
        # Calculate smart contract risk
        # Larger loan pools assumed to have more battle-tested contracts
        contract_complexity = min(20, len(self.loans) / 1000)  # Scale based on pool size
        contract_risk = base_contract_risk + max(0, 10 - contract_complexity)
        
        # Combine risks
        total_risk = (oracle_weight * oracle_risk + 
                     contract_weight * contract_risk)
        
        # Convert to score (100 - risk)
        return max(0, 100 - total_risk)

    def _compute_governance(self) -> float:
        """Compute governance score based on on-chain voting participation.
        
        Uses beta distribution to simulate participation metrics.
        
        Returns:
            Governance score (0-100).
        """
        config = self.config["governance"]
        shape = config["participation_shape"]
        scale = config["participation_scale"]
        min_score = config["min_score"]
        max_score = config["max_score"]
        
        # Simulate governance participation based on pool characteristics
        # Larger pools typically have better governance participation
        if len(self.loans) > 5000:
            participation_bonus = 0.2
        elif len(self.loans) > 1000:
            participation_bonus = 0.1
        else:
            participation_bonus = 0.0
        
        # Generate participation rate using beta distribution
        np.random.seed(123)  # For reproducibility
        base_participation = np.random.beta(shape, scale)  # Beta(1.5, 3) for realistic participation
        participation_rate = min(1.0, base_participation + participation_bonus)
        
        # Convert to score
        return min_score + (participation_rate * (max_score - min_score))

    def _compute_auditing(self) -> float:
        """Compute auditing score.
        
        Assumes high score due to real-time proof-of-reserve capabilities.
        
        Returns:
            Auditing score (0-100).
        """
        return float(self.config["auditing"]["real_time_score"])

    def _compute_interoperability(self) -> float:
        """Compute interoperability score.
        
        Assumes high score due to ERC-20/ERC-4626 standard compliance.
        
        Returns:
            Interoperability score (0-100).
        """
        return float(self.config["interoperability"]["erc_compliance_score"]) 