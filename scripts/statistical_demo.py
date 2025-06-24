#!/usr/bin/env python3
"""Demonstration script for statistical analysis functionality."""

import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from securitization_comparison.data_loader import load_fannie_mae
from securitization_comparison.statistical_analysis import run_statistical_analysis
import logging

def main():
    """Run statistical analysis demonstration."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("🎲 SECURITIZATION STATISTICAL ANALYSIS DEMO")
    print("=" * 60)
    
    # Load data
    print("📊 Loading sample data...")
    try:
        df = load_fannie_mae(data_dir="data", sample_size=5000)
        print(f"✅ Loaded {len(df):,} loans for analysis")
    except Exception as e:
        print(f"⚠️  Could not load real data ({e}), generating sample data...")
        import pandas as pd
        import numpy as np
        
        # Generate sample data for demo
        np.random.seed(42)
        n_loans = 5000
        df = pd.DataFrame({
            'loan_id': [f'DEMO_{i:06d}' for i in range(n_loans)],
            'orig_bal': np.random.lognormal(mean=12, sigma=0.5, size=n_loans),
            'orig_rate': np.random.normal(4.5, 1.0, n_loans).clip(2, 10),
            'orig_term': np.random.choice([180, 240, 300, 360], n_loans),
            'state': np.random.choice(['CA', 'TX', 'NY', 'FL', 'IL'], n_loans),
            'credit_score': np.random.normal(720, 50, n_loans).clip(300, 850),
            'dti': np.random.normal(25, 10, n_loans).clip(5, 60),
        })
        print(f"✅ Generated {len(df):,} synthetic loans for demo")
    
    # Run statistical analysis
    print("\n🔬 Starting Monte Carlo Analysis...")
    print("This will run 500 simulations with varying parameters")
    print("Each simulation tests different weight combinations from the YAML configs")
    print("-" * 60)
    
    try:
        results = run_statistical_analysis(
            df,
            n_simulations=500,  # Use fewer for demo speed
            n_threads=4,
            output_dir="reports"
        )
        
        # Display results
        print("\n📈 RESULTS SUMMARY")
        print("=" * 60)
        
        summary = results["summary"]
        print(f"Total Simulations: {summary['total_simulations']}")
        print(f"Blockchain Wins: {summary['blockchain_wins']} ({summary['win_probability_percent']:.1f}%)")
        print(f"Traditional Wins: {summary['traditional_wins']} ({100-summary['win_probability_percent']:.1f}%)")
        print(f"Win Probability: {summary['win_probability']:.3f}")
        
        # Score statistics
        stats = results["score_statistics"]
        print(f"\n📊 SCORE STATISTICS")
        print("-" * 30)
        print(f"Mean Difference (Blockchain - Traditional): {stats['mean_difference']:+.1f}")
        print(f"Standard Deviation: {stats['std_difference']:.1f}")
        print(f"Score Range: [{stats['min_difference']:+.1f}, {stats['max_difference']:+.1f}]")
        
        # Confidence interval
        ci = results["confidence_interval_95"]
        print(f"95% Confidence Interval: [{ci[0]:+.1f}, {ci[1]:+.1f}]")
        
        # Factor analysis
        print(f"\n🔍 FACTOR-LEVEL ANALYSIS")
        print("-" * 40)
        factor_analysis = results["factor_analysis"]
        
        print(f"{'Factor':<20} {'Win Rate':<10} {'Avg Diff':<10}")
        print("-" * 40)
        for factor, factor_stats in factor_analysis.items():
            win_rate = factor_stats["blockchain_win_rate"] * 100
            avg_diff = factor_stats["mean_difference"]
            print(f"{factor:<20} {win_rate:>6.1f}%   {avg_diff:>+6.1f}")
        
        # Interpretation
        print(f"\n💡 INTERPRETATION")
        print("-" * 30)
        win_prob = summary['win_probability_percent']
        
        if win_prob > 70:
            print("🟢 Strong evidence that blockchain consistently outperforms traditional")
        elif win_prob > 60:
            print("🟡 Moderate evidence that blockchain tends to outperform traditional")
        elif win_prob < 30:
            print("🔴 Strong evidence that traditional consistently outperforms blockchain")
        elif win_prob < 40:
            print("🟡 Moderate evidence that traditional tends to outperform blockchain")
        else:
            print("⚪ Mixed results - performance varies based on parameter assumptions")
        
        if abs(stats['mean_difference']) < 5:
            print("📊 Close competition - small average score differences")
        elif abs(stats['mean_difference']) > 20:
            print("📊 Clear performance gap - significant average score differences")
        
        if stats['std_difference'] > 15:
            print("📈 High variability - results depend heavily on parameter assumptions")
        elif stats['std_difference'] < 5:
            print("📈 Consistent results - stable outcome across parameter ranges")
        
        print(f"\n📁 Detailed results exported to: reports/")
        print("  - statistical_analysis_summary.yaml")
        print("  - monte_carlo_results.csv")
        
    except Exception as e:
        print(f"❌ Statistical analysis failed: {e}")
        return 1
    
    print("\n✅ Demo completed successfully!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    exit(main()) 