"""Unit tests for securitization models."""

import pytest
import pandas as pd
import numpy as np
from typing import Dict

from securitization_comparison.models.base import SecuritizationModel, FACTORS
from securitization_comparison.models.traditional import TraditionalModel
from securitization_comparison.models.blockchain import BlockchainModel


@pytest.fixture
def sample_loan_data() -> pd.DataFrame:
    """Create sample loan data for testing.
    
    Returns:
        Sample DataFrame with loan data.
    """
    np.random.seed(42)
    n_loans = 1000
    
    return pd.DataFrame({
        "loan_id": [f"LOAN_{i:06d}" for i in range(n_loans)],
        "orig_bal": np.random.lognormal(mean=12, sigma=0.5, size=n_loans),
        "orig_rate": np.random.normal(4.5, 1.0, n_loans).clip(2, 10),
        "orig_term": np.random.choice([180, 240, 300, 360], n_loans),
        "state": np.random.choice(['CA', 'TX', 'NY', 'FL', 'IL'], n_loans),
        "credit_score": np.random.normal(720, 50, n_loans).clip(300, 850),
        "dti": np.random.normal(25, 10, n_loans).clip(5, 60),
        "channel": np.random.choice(['R', 'B', 'C'], n_loans)
    })


class TestSecuritizationModel:
    """Test the base SecuritizationModel class."""
    
    def test_validate_scores_valid(self, sample_loan_data: pd.DataFrame) -> None:
        """Test score validation with valid scores."""
        class MockModel(SecuritizationModel):
            def compute_scores(self) -> Dict[str, float]:
                return {factor: 50.0 for factor in FACTORS}
        
        model = MockModel(sample_loan_data)
        scores = model.compute_scores()
        validated = model.validate_scores(scores)
        
        assert validated == scores
        assert all(0 <= score <= 100 for score in validated.values())
    
    def test_validate_scores_missing_factors(self, sample_loan_data: pd.DataFrame) -> None:
        """Test score validation with missing factors."""
        class MockModel(SecuritizationModel):
            def compute_scores(self) -> Dict[str, float]:
                return {"Transparency": 50.0}  # Missing other factors
        
        model = MockModel(sample_loan_data)
        scores = model.compute_scores()
        
        with pytest.raises(ValueError, match="Missing factor scores"):
            model.validate_scores(scores)
    
    def test_validate_scores_out_of_range(self, sample_loan_data: pd.DataFrame) -> None:
        """Test score validation with out-of-range scores."""
        class MockModel(SecuritizationModel):
            def compute_scores(self) -> Dict[str, float]:
                scores = {factor: 50.0 for factor in FACTORS}
                scores["Transparency"] = 150.0  # Invalid score
                return scores
        
        model = MockModel(sample_loan_data)
        scores = model.compute_scores()
        
        with pytest.raises(ValueError, match="Score for Transparency must be between 0-100"):
            model.validate_scores(scores)


class TestTraditionalModel:
    """Test the TraditionalModel class."""
    
    def test_initialization(self, sample_loan_data: pd.DataFrame) -> None:
        """Test model initialization."""
        model = TraditionalModel(sample_loan_data)
        assert model.loans is not None
        assert len(model.loans) == len(sample_loan_data)
        assert model.config is not None
    
    def test_compute_scores_structure(self, sample_loan_data: pd.DataFrame) -> None:
        """Test that compute_scores returns valid structure."""
        model = TraditionalModel(sample_loan_data)
        scores = model.compute_scores()
        
        # Check all factors are present
        assert set(scores.keys()) == set(FACTORS)
        
        # Check all scores are in valid range
        for factor, score in scores.items():
            assert 0 <= score <= 100, f"Score for {factor} is {score}, should be 0-100"
            assert isinstance(score, (int, float)), f"Score for {factor} should be numeric"
    
    def test_compute_scores_consistency(self, sample_loan_data: pd.DataFrame) -> None:
        """Test that scores are consistent across multiple calls."""
        model = TraditionalModel(sample_loan_data)
        scores1 = model.compute_scores()
        scores2 = model.compute_scores()
        
        for factor in FACTORS:
            assert abs(scores1[factor] - scores2[factor]) < 1e-6, \
                f"Scores for {factor} should be consistent"
    
    def test_transparency_calculation(self, sample_loan_data: pd.DataFrame) -> None:
        """Test transparency score calculation."""
        model = TraditionalModel(sample_loan_data)
        transparency_score = model._compute_transparency()
        
        assert 0 <= transparency_score <= 100
        # Traditional should have moderate transparency
        assert 20 <= transparency_score <= 80
    
    def test_liquidity_calculation(self, sample_loan_data: pd.DataFrame) -> None:
        """Test liquidity score calculation."""
        model = TraditionalModel(sample_loan_data)
        liquidity_score = model._compute_liquidity()
        
        assert 0 <= liquidity_score <= 100
        assert isinstance(liquidity_score, (int, float))
    
    def test_systemic_risk_calculation(self, sample_loan_data: pd.DataFrame) -> None:
        """Test systemic risk score calculation."""
        model = TraditionalModel(sample_loan_data)
        risk_score = model._compute_systemic_risk()
        
        assert 0 <= risk_score <= 100
        assert isinstance(risk_score, (int, float))
    
    def test_gini_coefficient(self, sample_loan_data: pd.DataFrame) -> None:
        """Test Gini coefficient calculation."""
        model = TraditionalModel(sample_loan_data)
        
        # Test with equal values (should be 0)
        equal_values = pd.Series([100, 100, 100, 100])
        gini_equal = model._calculate_gini_coefficient(equal_values)
        assert abs(gini_equal) < 1e-6
        
        # Test with highly unequal values (should be close to 1)
        unequal_values = pd.Series([1, 1, 1, 1000])
        gini_unequal = model._calculate_gini_coefficient(unequal_values)
        assert 0.5 < gini_unequal <= 1.0
        
        # Test with sample data
        gini_sample = model._calculate_gini_coefficient(sample_loan_data["orig_bal"])
        assert 0 <= gini_sample <= 1.0


class TestBlockchainModel:
    """Test the BlockchainModel class."""
    
    def test_initialization(self, sample_loan_data: pd.DataFrame) -> None:
        """Test model initialization."""
        model = BlockchainModel(sample_loan_data)
        assert model.loans is not None
        assert len(model.loans) == len(sample_loan_data)
        assert model.config is not None
    
    def test_compute_scores_structure(self, sample_loan_data: pd.DataFrame) -> None:
        """Test that compute_scores returns valid structure."""
        model = BlockchainModel(sample_loan_data)
        scores = model.compute_scores()
        
        # Check all factors are present
        assert set(scores.keys()) == set(FACTORS)
        
        # Check all scores are in valid range
        for factor, score in scores.items():
            assert 0 <= score <= 100, f"Score for {factor} is {score}, should be 0-100"
            assert isinstance(score, (int, float)), f"Score for {factor} should be numeric"
    
    def test_transparency_score(self, sample_loan_data: pd.DataFrame) -> None:
        """Test that blockchain transparency is high."""
        model = BlockchainModel(sample_loan_data)
        transparency_score = model._compute_transparency()
        
        assert 80 <= transparency_score <= 100  # Should be high for blockchain
    
    def test_liquidity_calculation(self, sample_loan_data: pd.DataFrame) -> None:
        """Test liquidity score calculation."""
        model = BlockchainModel(sample_loan_data)
        liquidity_score = model._compute_liquidity()
        
        assert 0 <= liquidity_score <= 100
        assert isinstance(liquidity_score, (int, float))
    
    def test_systemic_risk_calculation(self, sample_loan_data: pd.DataFrame) -> None:
        """Test systemic risk score calculation."""
        model = BlockchainModel(sample_loan_data)
        risk_score = model._compute_systemic_risk()
        
        assert 0 <= risk_score <= 100
        assert isinstance(risk_score, (int, float))
    
    def test_governance_calculation(self, sample_loan_data: pd.DataFrame) -> None:
        """Test governance score calculation."""
        model = BlockchainModel(sample_loan_data)
        governance_score = model._compute_governance()
        
        assert 0 <= governance_score <= 100
        assert isinstance(governance_score, (int, float))
    
    def test_auditing_score(self, sample_loan_data: pd.DataFrame) -> None:
        """Test that blockchain auditing score is high."""
        model = BlockchainModel(sample_loan_data)
        auditing_score = model._compute_auditing()
        
        assert 70 <= auditing_score <= 100  # Should be high for blockchain
    
    def test_interoperability_score(self, sample_loan_data: pd.DataFrame) -> None:
        """Test that blockchain interoperability score is high."""
        model = BlockchainModel(sample_loan_data)
        interop_score = model._compute_interoperability()
        
        assert 70 <= interop_score <= 100  # Should be high for blockchain


class TestModelComparison:
    """Test comparison between traditional and blockchain models."""
    
    def test_different_scores(self, sample_loan_data: pd.DataFrame) -> None:
        """Test that traditional and blockchain models produce different scores."""
        trad_model = TraditionalModel(sample_loan_data)
        chain_model = BlockchainModel(sample_loan_data)
        
        trad_scores = trad_model.compute_scores()
        chain_scores = chain_model.compute_scores()
        
        # At least some scores should be different
        differences = [abs(trad_scores[f] - chain_scores[f]) for f in FACTORS]
        assert sum(differences) > 10, "Models should produce meaningfully different scores"
    
    def test_blockchain_advantages(self, sample_loan_data: pd.DataFrame) -> None:
        """Test that blockchain has expected advantages in certain areas."""
        trad_model = TraditionalModel(sample_loan_data)
        chain_model = BlockchainModel(sample_loan_data)
        
        trad_scores = trad_model.compute_scores()
        chain_scores = chain_model.compute_scores()
        
        # Blockchain should typically score higher on transparency
        assert chain_scores["Transparency"] > trad_scores["Transparency"]
        
        # Blockchain should typically score higher on interoperability
        assert chain_scores["Interoperability"] > trad_scores["Interoperability"]
    
    def test_score_ranges_realistic(self, sample_loan_data: pd.DataFrame) -> None:
        """Test that scores are in realistic ranges."""
        trad_model = TraditionalModel(sample_loan_data)
        chain_model = BlockchainModel(sample_loan_data)
        
        trad_scores = trad_model.compute_scores()
        chain_scores = chain_model.compute_scores()
        
        # No model should dominate completely (all scores 90+)
        trad_high_scores = sum(1 for score in trad_scores.values() if score > 90)
        chain_high_scores = sum(1 for score in chain_scores.values() if score > 90)
        
        assert trad_high_scores < len(FACTORS), "Traditional model shouldn't dominate all factors"
        assert chain_high_scores < len(FACTORS), "Blockchain model shouldn't dominate all factors"
        
        # No model should fail completely (all scores <20)
        trad_low_scores = sum(1 for score in trad_scores.values() if score < 20)
        chain_low_scores = sum(1 for score in chain_scores.values() if score < 20)
        
        assert trad_low_scores < len(FACTORS) - 1, "Traditional model shouldn't fail most factors"
        assert chain_low_scores < len(FACTORS) - 1, "Blockchain model shouldn't fail most factors"


# Integration test with smaller dataset
def test_integration_small_dataset() -> None:
    """Test integration with a small dataset."""
    # Create minimal dataset
    small_data = pd.DataFrame({
        "loan_id": ["LOAN_001", "LOAN_002", "LOAN_003"],
        "orig_bal": [300000, 400000, 500000],
        "orig_rate": [3.5, 4.0, 4.5],
        "orig_term": [360, 360, 300],
        "state": ["CA", "TX", "NY"]
    })
    
    # Test both models work with small dataset
    trad_model = TraditionalModel(small_data)
    chain_model = BlockchainModel(small_data)
    
    trad_scores = trad_model.compute_scores()
    chain_scores = chain_model.compute_scores()
    
    # Basic validation
    assert len(trad_scores) == len(FACTORS)
    assert len(chain_scores) == len(FACTORS)
    
    for scores in [trad_scores, chain_scores]:
        for factor, score in scores.items():
            assert 0 <= score <= 100, f"Invalid score for {factor}: {score}" 