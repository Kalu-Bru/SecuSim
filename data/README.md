# Fannie Mae Data Setup

This directory should contain the Fannie Mae Single-Family Loan Performance data used for the securitization comparison analysis.

## Data Source

**Fannie Mae Single-Family Loan Performance Data**
- Website: https://loanperformancedata.fanniemae.com/lppub/index.html
- Access: Free, no authentication required
- License: See Fannie Mae's terms of use

## Download Instructions

1. **Visit the Data Portal**
   - Go to https://loanperformancedata.fanniemae.com/lppub/index.html
   - Click "Agree" to accept the terms of use

2. **Select Data**
   - Choose "Single-Family Loan Performance Data"
   - Select one or more years of data (recommend starting with recent years)
   - Each year contains quarterly data files

3. **Download Format**
   - Data is provided in ZIP files containing pipe-delimited text files
   - Each ZIP contains:
     - `*_Acquisition_*.txt` - Loan origination data
     - `*_Performance_*.txt` - Monthly loan performance data

4. **Extract to This Directory**
   ```bash
   # Extract ZIP files to the data/ directory
   cd data/
   unzip "*.zip"
   ```

## Expected File Structure

After downloading and extracting, you should have files like:
```
data/
├── README.md (this file)
├── historical_data_2023Q1.zip
├── historical_data_2023Q2.zip
├── Acquisition_2023Q1.txt
├── Performance_2023Q1.txt
├── Acquisition_2023Q2.txt
├── Performance_2023Q2.txt
└── fannie_mae_cached.parquet (created after first run)
```

## Data Dictionary

### Acquisition File Key Columns
- `LOAN_ID` - Unique loan identifier
- `ORIG_RT` - Original interest rate
- `ORIG_AMT` - Original loan amount
- `ORIG_TRM` - Original loan term in months
- `ORIG_DTE` - Origination date (MM/YYYY)
- `STATE` - Property state
- `CSCORE_B` - Borrower credit score
- `DTI` - Debt-to-income ratio

### Performance File Key Columns
- `LOAN_ID` - Matches acquisition file
- `Monthly_Rpt_Prd` - Reporting period (MM/YYYY)
- `LAST_UPB` - Current unpaid principal balance
- `Delq_Status` - Delinquency status
- `Loan_Age` - Age of loan in months

## Data Usage

The data loader (`securitization_comparison.data_loader.load_fannie_mae()`) will:

1. **Auto-detect** ZIP files and extract them if needed
2. **Parse** pipe-delimited text files with proper data types
3. **Clean** data by removing invalid records and capping extreme values
4. **Cache** processed data as Parquet for faster subsequent loads
5. **Sample** data if requested for quick iteration

## Data Size Considerations

- **Full dataset**: Several million loans per year, ~1-5GB per year
- **Recommended sample**: 25,000-100,000 loans for development
- **Memory usage**: ~1-2GB RAM for 100k loans with full features

## Troubleshooting

### "No acquisition files found"
- Ensure ZIP files are extracted in the `data/` directory
- Check that `.txt` files exist and are not corrupted
- Verify file naming matches expected patterns

### "Memory error" 
- Reduce sample size using `--sample-size` parameter
- Use `--no-cache` to avoid loading full dataset

### "Data parsing errors"
- Download fresh data files if corruption suspected
- Check Fannie Mae website for data format changes
- Enable debug logging with `--debug` flag

## Privacy and Compliance

- Data is anonymized by Fannie Mae (no PII)
- Follow Fannie Mae's terms of use for data redistribution
- Do not commit raw data files to version control (already in .gitignore)

## Alternative Data Sources

If Fannie Mae data is unavailable, you can create synthetic data for testing:

```python
from securitization_comparison.data_loader import load_fannie_mae
import pandas as pd
import numpy as np

# Create synthetic data
np.random.seed(42)
n_loans = 10000

synthetic_data = pd.DataFrame({
    "loan_id": [f"SYNTH_{i:06d}" for i in range(n_loans)],
    "orig_bal": np.random.lognormal(mean=12, sigma=0.5, size=n_loans),
    "orig_rate": np.random.normal(4.5, 1.0, n_loans).clip(2, 10),
    "orig_term": np.random.choice([180, 240, 300, 360], n_loans),
    "state": np.random.choice(['CA', 'TX', 'NY', 'FL', 'IL'], n_loans),
    # ... add more columns as needed
})
``` 