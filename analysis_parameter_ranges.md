# 📊 Statistical Analysis Parameter Ranges

This document shows exactly how much each parameter is being varied in the Monte Carlo statistical analysis.

## 🏛️ Traditional Securitization Parameters

| Category | Parameter | Baseline | Min | Max | Range Width | % Below Baseline | % Above Baseline |
|----------|-----------|----------|-----|-----|-------------|------------------|------------------|
| **Transparency** | base_score | 30 | 10 | 70 | 60 | -67% | +133% |
| | disclosure_weight | 70 | 30 | 100 | 70 | -57% | +43% |
| | typical_disclosure_rate | 0.4 | 0.1 | 0.8 | 0.7 | -75% | +100% |
| **Liquidity** | cpr_alpha | 5.0 | 1.0 | 10.0 | 9.0 | -80% | +100% |
| | cpr_scale | 0.1 | 0.01 | 0.3 | 0.29 | -90% | +200% |
| **Systemic Risk** | concentration_weight | 0.6 | 0.2 | 0.9 | 0.7 | -67% | +50% |
| | dispersion_weight | 0.4 | 0.1 | 0.8 | 0.7 | -75% | +100% |
| **Governance** | fixed_score | 50 | 20 | 90 | 70 | -60% | +80% |
| **Auditing** | regulated_score | 60 | 30 | 95 | 65 | -50% | +58% |
| **Interoperability** | proprietary_penalty | 60 | 20 | 100 | 80 | -67% | +67% |

## ⛓️ Blockchain Securitization Parameters

| Category | Parameter | Baseline | Min | Max | Range Width | % Below Baseline | % Above Baseline |
|----------|-----------|----------|-----|-----|-------------|------------------|------------------|
| **Transparency** | on_chain_score | 90 | 60 | 100 | 40 | -33% | +11% |
| **Liquidity** | amm_depth_shape | 2.0 | 0.5 | 5.0 | 4.5 | -75% | +150% |
| | amm_depth_scale | 0.8 | 0.2 | 1.5 | 1.3 | -75% | +88% |
| | min_score | 20 | 5 | 50 | 45 | -75% | +150% |
| | max_score | 95 | 60 | 100 | 40 | -37% | +5% |
| **Systemic Risk** | oracle_dependency_weight | 0.4 | 0.1 | 0.7 | 0.6 | -75% | +75% |
| | smart_contract_weight | 0.6 | 0.3 | 0.9 | 0.6 | -50% | +50% |
| | base_oracle_risk | 15 | 5 | 40 | 35 | -67% | +167% |
| | base_contract_risk | 10 | 2 | 30 | 28 | -80% | +200% |
| **Governance** | participation_shape | 1.5 | 0.5 | 3.0 | 2.5 | -67% | +100% |
| | participation_scale | 0.6 | 0.2 | 1.2 | 1.0 | -67% | +100% |
| | min_score | 30 | 10 | 60 | 50 | -67% | +100% |
| | max_score | 90 | 60 | 100 | 40 | -33% | +11% |
| **Auditing** | real_time_score | 80 | 50 | 100 | 50 | -38% | +25% |
| **Interoperability** | erc_compliance_score | 85 | 50 | 100 | 50 | -41% | +18% |

## 🎯 Key Insights

### **Maximum Variation Achieved**
- **Largest parameter swings**: Some parameters vary by ±200% from baseline
- **Comprehensive coverage**: Every parameter tested across wide ranges
- **Realistic bounds**: All ranges stay within meaningful limits (0-100 for scores, 0-1 for weights)

### **Most Sensitive Parameters**
- **Traditional liquidity** (`cpr_scale`): 90% below to 200% above baseline
- **Blockchain systemic risk** (`base_contract_risk`, `base_oracle_risk`): Up to 200% variation
- **Traditional transparency** (`base_score`): 67% below to 133% above baseline

### **Conservative Parameters**
- **Transparency scores**: Limited by realistic bounds (can't exceed 100)
- **Weight parameters**: Limited by sum-to-one constraints in some cases

## 🔬 What This Means for Analysis

**Every simulation** randomly selects values within these ranges, meaning:

1. **Worst-case scenarios**: Some simulations test extremely pessimistic assumptions
2. **Best-case scenarios**: Other simulations test very optimistic assumptions  
3. **Realistic coverage**: Most simulations fall between these extremes
4. **True sensitivity**: Results show robustness across full parameter uncertainty

**Example simulation scenario:**
- Traditional transparency could be as low as 10 (very opaque) or as high as 70 (very transparent)
- Blockchain liquidity could have AMM depth shape of 0.5 (poor liquidity) or 5.0 (excellent liquidity)
- Risk parameters could vary dramatically, testing both conservative and aggressive assumptions

This ensures the win probability calculation is based on **maximum realistic parameter uncertainty**, not just small variations around baseline values. 