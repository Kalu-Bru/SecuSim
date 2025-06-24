#!/usr/bin/env python3
"""Generate sample mortgage loan data for testing the securitization comparison."""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_sample_data(n_loans: int = 10000) -> pd.DataFrame:
    """Generate sample mortgage loan data."""
    np.random.seed(42)
    
    data = {
        'loan_id': [f'LOAN_{i:06d}' for i in range(n_loans)],
        'original_loan_amount': np.random.normal(300000, 100000, n_loans).clip(50000, 1000000),
        'loan_to_value_ratio': np.random.normal(75, 15, n_loans).clip(10, 100),
        'debt_to_income_ratio': np.random.normal(30, 10, n_loans).clip(5, 60),
        'credit_score': np.random.normal(720, 80, n_loans).clip(300, 850),
        'current_loan_balance': np.random.normal(250000, 80000, n_loans).clip(0, 800000),
        'monthly_payment': np.random.normal(1500, 500, n_loans).clip(200, 5000),
        'prepayment_penalty_flag': np.random.choice(['Y', 'N'], n_loans, p=[0.2, 0.8]),
        'property_type': np.random.choice(['SF', 'CO', 'PU', 'MH'], n_loans, p=[0.7, 0.2, 0.05, 0.05]),
        'occupancy_status': np.random.choice(['P', 'S', 'I'], n_loans, p=[0.8, 0.1, 0.1]),
        'loan_purpose': np.random.choice(['P', 'C', 'R'], n_loans, p=[0.6, 0.3, 0.1]),
        'number_of_borrowers': np.random.choice([1, 2, 3, 4], n_loans, p=[0.3, 0.6, 0.08, 0.02]),
        'first_time_homebuyer_flag': np.random.choice(['Y', 'N'], n_loans, p=[0.3, 0.7]),
        'servicer_name': np.random.choice(['WELLS FARGO', 'CHASE', 'BANK OF AMERICA', 'QUICKEN'], n_loans),
        'current_interest_rate': np.random.normal(4.5, 1.0, n_loans).clip(2.0, 8.0),
        'zero_balance_code': np.random.choice(['01', '02', '03', '06', '09', '15', '16'], n_loans, p=[0.02, 0.02, 0.85, 0.03, 0.05, 0.02, 0.01]),
        'zero_balance_date': pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(0, 365, n_loans), unit='D')
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    Path('data').mkdir(exist_ok=True)
    
    # Generate sample data
    df = generate_sample_data()
    
    # Save as CSV
    output_path = Path('data/sample_loan_data.csv')
    df.to_csv(output_path, index=False)
    
    print(f'✅ Created sample dataset with {len(df):,} loans')
    print(f'📁 Saved to: {output_path}')
    print(f'📊 File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB')
    print(f'🔍 Sample of the data:')
    print(df.head().to_string()) 