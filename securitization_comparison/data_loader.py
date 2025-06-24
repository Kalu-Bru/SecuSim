"""Data loader for Fannie Mae loan performance data."""

import logging
import zipfile
from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Column mappings for Fannie Mae data
# Acquisition file columns (subset of most relevant)
ACQUISITION_COLUMNS = {
    "LOAN_ID": "loan_id",
    "ORIG_CHN": "channel", 
    "SELLER_NAME": "seller_name",
    "ORIG_RT": "orig_rate",
    "ORIG_AMT": "orig_bal",
    "ORIG_TRM": "orig_term",
    "ORIG_DTE": "orig_date",
    "FRST_DTE": "first_pay_date",
    "OLTV": "oltv",
    "OCLTV": "ocltv", 
    "NUM_BO": "num_borrowers",
    "DTI": "dti",
    "CSCORE_B": "credit_score",
    "FTHB_FLG": "first_time_buyer",
    "PURPOSE": "purpose",
    "PROP_TYP": "property_type",
    "NUM_UNIT": "num_units",
    "OCC_STAT": "occupancy_status",
    "STATE": "state",
    "ZIP_3": "zip_3",
    "MI_PCT": "mi_pct",
    "PRODUCT_TYPE": "product_type",
    "CSCORE_C": "co_credit_score"
}

# Performance file columns (subset for delinquency metrics)
PERFORMANCE_COLUMNS = {
    "LOAN_ID": "loan_id",
    "Monthly_Rpt_Prd": "report_period",
    "Servicer_Name": "servicer_name", 
    "LAST_RT": "current_rate",
    "LAST_UPB": "current_upb",
    "Loan_Age": "loan_age",
    "Months_To_Legal_Mat": "months_to_maturity",
    "Adj_Month_To_Mat": "adj_months_to_maturity",
    "Maturity_Date": "maturity_date",
    "MSA": "msa",
    "Delq_Status": "delinquency_status",
    "MOD_FLAG": "modification_flag",
    "Zero_Bal_Code": "zero_balance_code",
    "ZB_DTE": "zero_balance_date"
}


def load_fannie_mae(
    data_dir: str = "data", 
    sample_size: Optional[int] = None,
    use_cache: bool = True,
    force_refresh: bool = False
) -> pd.DataFrame:
    """Load Fannie Mae loan data with optional sampling and caching.
    
    Args:
        data_dir: Directory containing the data files.
        sample_size: Number of loans to sample. If None, loads all data.
        use_cache: Whether to use cached Parquet file if available.
        force_refresh: Force refresh of cache even if it exists.
        
    Returns:
        DataFrame with cleaned loan data.
        
    Raises:
        FileNotFoundError: If data files are not found.
        ValueError: If data cannot be loaded or parsed.
    """
    data_path = Path(data_dir)
    cache_file = data_path / "fannie_mae_cached.parquet"
    
    # Try to load from cache first
    if use_cache and cache_file.exists() and not force_refresh:
        logger.info(f"Loading cached data from {cache_file}")
        df = pd.read_parquet(cache_file)
        
        if sample_size is not None and len(df) > sample_size:
            logger.info(f"Sampling {sample_size} loans from cached data")
            return df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        return df
    
    # Load fresh data
    logger.info("Loading fresh data from source files")
    
    # Find acquisition and performance files
    acquisition_files = list(data_path.glob("*Acquisition*.txt"))
    performance_files = list(data_path.glob("*Performance*.txt"))
    
    if not acquisition_files:
        # Try to find ZIP files and extract
        zip_files = list(data_path.glob("*.zip"))
        if zip_files:
            logger.info("Found ZIP files, extracting...")
            _extract_zip_files(zip_files, data_path)
            acquisition_files = list(data_path.glob("*Acquisition*.txt"))
            performance_files = list(data_path.glob("*Performance*.txt"))
    
    if not acquisition_files:
        raise FileNotFoundError(
            f"No acquisition files found in {data_path}. "
            "Please download Fannie Mae data and extract to this directory. "
            "See data/README.md for instructions."
        )
    
    # Load acquisition data
    logger.info(f"Loading {len(acquisition_files)} acquisition file(s)")
    acquisition_df = _load_acquisition_files(acquisition_files, sample_size)
    
    # Load performance data if available
    if performance_files:
        logger.info(f"Loading {len(performance_files)} performance file(s)")
        performance_df = _load_performance_files(performance_files, acquisition_df["loan_id"])
        
        # Merge acquisition and performance data
        df = _merge_loan_data(acquisition_df, performance_df)
    else:
        logger.warning("No performance files found, using acquisition data only")
        df = acquisition_df
    
    # Clean and validate data
    df = _clean_loan_data(df)
    
    # Cache the result
    if use_cache:
        logger.info(f"Caching data to {cache_file}")
        data_path.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file, compression="snappy")
    
    logger.info(f"Loaded {len(df)} loans with {len(df.columns)} columns")
    return df


def _extract_zip_files(zip_files: List[Path], extract_dir: Path) -> None:
    """Extract ZIP files to the specified directory.
    
    Args:
        zip_files: List of ZIP file paths.
        extract_dir: Directory to extract files to.
    """
    for zip_file in zip_files:
        logger.info(f"Extracting {zip_file}")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)


def _load_acquisition_files(files: List[Path], sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load and combine acquisition files.
    
    Args:
        files: List of acquisition file paths.
        sample_size: Optional sample size for quick iteration.
        
    Returns:
        Combined acquisition DataFrame.
    """
    dfs = []
    
    for file in tqdm(files, desc="Loading acquisition files"):
        try:
            # Fannie Mae files are pipe-delimited with no header
            df = pd.read_csv(
                file,
                sep="|",
                header=None,
                names=list(ACQUISITION_COLUMNS.keys()),
                dtype={
                    "LOAN_ID": str,
                    "ORIG_RT": "float64",
                    "ORIG_AMT": "float64",
                    "ORIG_TRM": "Int64",
                    "OLTV": "Int64",
                    "OCLTV": "Int64",
                    "NUM_BO": "Int64",
                    "DTI": "Int64",
                    "CSCORE_B": "Int64",
                    "CSCORE_C": "Int64",
                    "NUM_UNIT": "Int64",
                    "MI_PCT": "Int64"
                },
                low_memory=False
            )
            
            # Rename columns
            df = df.rename(columns=ACQUISITION_COLUMNS)
            
            # Convert date columns
            df["orig_date"] = pd.to_datetime(df["orig_date"], format="%m/%Y", errors="coerce")
            df["first_pay_date"] = pd.to_datetime(df["first_pay_date"], format="%m/%Y", errors="coerce")
            
            dfs.append(df)
            
        except Exception as e:
            logger.warning(f"Failed to load {file}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No acquisition files could be loaded")
    
    # Combine all files
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Sample if requested
    if sample_size is not None and len(combined_df) > sample_size:
        logger.info(f"Sampling {sample_size} loans from {len(combined_df)} total")
        combined_df = combined_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    return combined_df


def _load_performance_files(files: List[Path], loan_ids: pd.Series) -> pd.DataFrame:
    """Load and combine performance files.
    
    Args:
        files: List of performance file paths.
        loan_ids: Series of loan IDs to filter for.
        
    Returns:
        Combined performance DataFrame.
    """
    dfs = []
    loan_id_set = set(loan_ids)
    
    for file in tqdm(files, desc="Loading performance files"):
        try:
            # Load in chunks to handle large files
            chunk_dfs = []
            
            for chunk in pd.read_csv(
                file,
                sep="|",
                header=None,
                names=list(PERFORMANCE_COLUMNS.keys()),
                dtype={
                    "LOAN_ID": str,
                    "LAST_RT": "float64",
                    "LAST_UPB": "float64",
                    "Loan_Age": "Int64",
                    "Months_To_Legal_Mat": "Int64",
                    "Adj_Month_To_Mat": "Int64"
                },
                chunksize=10000,
                low_memory=False
            ):
                # Filter for relevant loan IDs
                chunk = chunk[chunk["LOAN_ID"].isin(loan_id_set)]
                
                if not chunk.empty:
                    # Rename columns
                    chunk = chunk.rename(columns=PERFORMANCE_COLUMNS)
                    
                    # Convert date columns
                    chunk["report_period"] = pd.to_datetime(chunk["report_period"], format="%m/%Y", errors="coerce")
                    chunk["maturity_date"] = pd.to_datetime(chunk["maturity_date"], format="%m/%Y", errors="coerce")
                    chunk["zero_balance_date"] = pd.to_datetime(chunk["zero_balance_date"], format="%m/%Y", errors="coerce")
                    
                    chunk_dfs.append(chunk)
            
            if chunk_dfs:
                dfs.append(pd.concat(chunk_dfs, ignore_index=True))
                
        except Exception as e:
            logger.warning(f"Failed to load {file}: {e}")
            continue
    
    if not dfs:
        logger.warning("No performance files could be loaded")
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


def _merge_loan_data(acquisition_df: pd.DataFrame, performance_df: pd.DataFrame) -> pd.DataFrame:
    """Merge acquisition and performance data.
    
    Args:
        acquisition_df: Acquisition data.
        performance_df: Performance data.
        
    Returns:
        Merged DataFrame with loan-level aggregated performance metrics.
    """
    if performance_df.empty:
        return acquisition_df
    
    # Aggregate performance data by loan
    perf_agg = performance_df.groupby("loan_id").agg({
        "current_upb": "last",  # Most recent balance
        "loan_age": "max",  # Maximum age observed
        "delinquency_status": ["max", "nunique"],  # Worst delinquency and variety
        "modification_flag": "any",  # Ever modified
        "report_period": "count"  # Number of payment records
    }).round(2)
    
    # Flatten column names
    perf_agg.columns = [
        "last_upb", "max_loan_age", "max_delinquency", 
        "delinquency_variety", "ever_modified", "payment_count"
    ]
    
    # Merge with acquisition data
    merged_df = acquisition_df.merge(perf_agg, on="loan_id", how="left")
    
    return merged_df


def _clean_loan_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate loan data.
    
    Args:
        df: Raw loan DataFrame.
        
    Returns:
        Cleaned DataFrame.
    """
    logger.info("Cleaning loan data")
    
    # Remove loans with missing critical fields
    required_fields = ["loan_id", "orig_bal"]
    before_count = len(df)
    
    for field in required_fields:
        if field in df.columns:
            df = df.dropna(subset=[field])
    
    logger.info(f"Removed {before_count - len(df)} loans with missing critical fields")
    
    # Cap extreme values
    if "orig_rate" in df.columns:
        df["orig_rate"] = df["orig_rate"].clip(0, 20)  # Cap at 20%
    
    if "orig_bal" in df.columns:
        df["orig_bal"] = df["orig_bal"].clip(1000, 10_000_000)  # Cap at $10M
    
    if "dti" in df.columns:
        df["dti"] = df["dti"].clip(0, 100)  # Cap DTI at 100%
    
    # Create derived features
    if "orig_date" in df.columns:
        df["orig_year"] = df["orig_date"].dt.year
        df["orig_month"] = df["orig_date"].dt.month
    
    if "orig_bal" in df.columns:
        # Create loan size categories
        df["loan_size_category"] = pd.cut(
            df["orig_bal"],
            bins=[0, 200_000, 400_000, 800_000, float("inf")],
            labels=["Small", "Medium", "Large", "Jumbo"]
        )
    
    return df 