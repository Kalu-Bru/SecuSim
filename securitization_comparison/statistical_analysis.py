"""Statistical analysis module for securitization comparison.

This module performs sensitivity analysis and Monte Carlo simulations
to determine the probability that blockchain securitization outperforms
traditional securitization across different parameter ranges.
"""

import logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings

from .models.traditional import TraditionalModel
from .models.blockchain import BlockchainModel

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class StatisticalAnalyzer:
    """Performs statistical analysis on securitization model comparisons."""
    
    def __init__(self, loans_df: pd.DataFrame, n_simulations: int = 1000):
        """Initialize the statistical analyzer.
        
        Args:
            loans_df: DataFrame containing loan data.
            n_simulations: Number of Monte Carlo simulations to run.
        """
        self.loans = loans_df
        self.n_simulations = n_simulations
        self.results = []
        self.parameter_ranges = self._load_parameter_ranges()
    
    def _load_parameter_ranges(self) -> Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]:
        """Load parameter ranges for sensitivity analysis.
        
        Returns:
            Dictionary containing parameter ranges for both models.
        """
        # return {
        #     "traditional": {
        #         "transparency": {
        #             "base_score": (10, 70),  # Much wider: 67% below to 133% above baseline (30)
        #             "disclosure_weight": (30, 100),  # Much wider: 57% below to 43% above baseline (70)
        #             "typical_disclosure_rate": (0.1, 0.8)  # Much wider: 75% below to 100% above baseline (0.4)
        #         },
        #         "liquidity": {
        #             "cpr_alpha": (1.0, 10.0),  # Much wider: 80% below to 100% above baseline (5.0)
        #             "cpr_scale": (0.01, 0.3)  # Much wider: 90% below to 200% above baseline (0.1)
        #         },
        #         "systemic_risk": {
        #             "concentration_weight": (0.2, 0.9),  # Much wider: 67% below to 50% above baseline (0.6)
        #             "dispersion_weight": (0.1, 0.8)  # Much wider: 75% below to 100% above baseline (0.4)
        #         },
        #         "governance": {
        #             "fixed_score": (20, 90)  # Much wider: 60% below to 80% above baseline (50)
        #         },
        #         "auditing": {
        #             "regulated_score": (30, 95)  # Much wider: 50% below to 58% above baseline (60)
        #         },
        #         "interoperability": {
        #             "proprietary_penalty": (20, 100)  # Much wider: 67% below to 67% above baseline (60)
        #         }
        #     },
        #     "blockchain": {
        #         "transparency": {
        #             "on_chain_score": (60, 100)  # Much wider: 33% below to 11% above baseline (90)
        #         },
        #         "liquidity": {
        #             "amm_depth_shape": (0.5, 5.0),  # Much wider: 75% below to 150% above baseline (2.0)
        #             "amm_depth_scale": (0.2, 1.5),  # Much wider: 75% below to 88% above baseline (0.8)
        #             "min_score": (5, 50),  # Much wider: 75% below to 150% above baseline (20)
        #             "max_score": (60, 100)  # Much wider: 37% below to 5% above baseline (95)
        #         },
        #         "systemic_risk": {
        #             "oracle_dependency_weight": (0.1, 0.7),  # Much wider: 75% below to 75% above baseline (0.4)
        #             "smart_contract_weight": (0.3, 0.9),  # Much wider: 50% below to 50% above baseline (0.6)
        #             "base_oracle_risk": (5, 40),  # Much wider: 67% below to 167% above baseline (15)
        #             "base_contract_risk": (2, 30)  # Much wider: 80% below to 200% above baseline (10)
        #         },
        #         "governance": {
        #             "participation_shape": (0.5, 3.0),  # Much wider: 67% below to 100% above baseline (1.5)
        #             "participation_scale": (0.2, 1.2),  # Much wider: 67% below to 100% above baseline (0.6)
        #             "min_score": (10, 60),  # Much wider: 67% below to 100% above baseline (30)
        #             "max_score": (60, 100)  # Much wider: 33% below to 11% above baseline (90)
        #         },
        #         "auditing": {
        #             "real_time_score": (50, 100)  # Much wider: 38% below to 25% above baseline (80)
        #         },
        #         "interoperability": {
        #             "erc_compliance_score": (50, 100)  # Much wider: 41% below to 18% above baseline (85)
        #         }
        #     }
        # }

        return {
            "traditional": {
                "transparency": {
                    "base_score": (0, 100),  # Much wider: 67% below to 133% above baseline (30)
                    "disclosure_weight": (0, 100),  # Much wider: 57% below to 43% above baseline (70)
                    "typical_disclosure_rate": (0.09, 0.91)  # Much wider: 75% below to 100% above baseline (0.4)
                },
                "liquidity": {
                    "cpr_alpha": (0.5, 10.0),  # Much wider: 80% below to 100% above baseline (5.0)
                    "cpr_scale": (0.01, 0.5)  # Much wider: 90% below to 200% above baseline (0.1)
                },
                "systemic_risk": {
                    "concentration_weight": (0.1, 2.0),  # Much wider: 67% below to 50% above baseline (0.6)
                    "dispersion_weight": (0.1, 2.0)  # Much wider: 75% below to 100% above baseline (0.4)
                },
                "governance": {
                    "fixed_score": (0, 100)  # Much wider: 60% below to 80% above baseline (50)
                },
                "auditing": {
                    "regulated_score": (0, 100)  # Much wider: 50% below to 58% above baseline (60)
                },
                "interoperability": {
                    "proprietary_penalty": (0, 100)  # Much wider: 67% below to 67% above baseline (60)
                }
            },
            "blockchain": {
                "transparency": {
                    "on_chain_score": (0, 100)  # Much wider: 33% below to 11% above baseline (90)
                },
                "liquidity": {
                    "amm_depth_shape": (0.5, 5.0),  # Much wider: 75% below to 150% above baseline (2.0)
                    "amm_depth_scale": (0.1, 3.0),  # Much wider: 75% below to 88% above baseline (0.8)
                    "min_score": (10, 30),  # Much wider: 75% below to 150% above baseline (20)
                    "max_score": (90, 100)  # Much wider: 37% below to 5% above baseline (95)
                },
                "systemic_risk": {
                    "oracle_dependency_weight": (0.1, 2.0),  # Much wider: 75% below to 75% above baseline (0.4)
                    "smart_contract_weight": (0.1, 2.0),  # Much wider: 50% below to 50% above baseline (0.6)
                    "base_oracle_risk": (1, 100),  # Much wider: 67% below to 167% above baseline (15)
                    "base_contract_risk": (1, 100)  # Much wider: 80% below to 200% above baseline (10)
                },
                "governance": {
                    "participation_shape": (0.1, 5.0),  # Much wider: 67% below to 100% above baseline (1.5)
                    "participation_scale": (0.1, 2.0),  # Much wider: 67% below to 100% above baseline (0.6)
                    "min_score": (0, 30),  # Much wider: 67% below to 100% above baseline (30)
                    "max_score": (90, 100)  # Much wider: 33% below to 11% above baseline (90)
                },
                "auditing": {
                    "real_time_score": (0, 100)  # Much wider: 38% below to 25% above baseline (80)
                },
                "interoperability": {
                    "erc_compliance_score": (0, 100)  # Much wider: 41% below to 18% above baseline (85)
                }
            }
        }
    
    def _generate_random_config(self, model_type: str) -> Dict[str, Any]:
        """Generate a random configuration within parameter ranges.
        
        Args:
            model_type: Either 'traditional' or 'blockchain'.
            
        Returns:
            Random configuration dictionary.
        """
        config = {}
        ranges = self.parameter_ranges[model_type]
        
        for category, params in ranges.items():
            config[category] = {}
            for param, (min_val, max_val) in params.items():
                # Generate random value within range
                config[category][param] = np.random.uniform(min_val, max_val)
        
        return config
    
    def _run_single_simulation(self, sim_id: int) -> Dict[str, Any]:
        """Run a single simulation with random parameters.
        
        Args:
            sim_id: Simulation identifier.
            
        Returns:
            Dictionary containing simulation results.
        """
        try:
            # Generate random configurations
            trad_config = self._generate_random_config("traditional")
            blockchain_config = self._generate_random_config("blockchain")
            
            # Create temporary config files
            trad_config_path = f"/tmp/trad_config_{sim_id}.yaml"
            blockchain_config_path = f"/tmp/blockchain_config_{sim_id}.yaml"
            
            with open(trad_config_path, 'w') as f:
                yaml.dump(trad_config, f)
            with open(blockchain_config_path, 'w') as f:
                yaml.dump(blockchain_config, f)
            
            # Create models with random configs
            trad_model = TraditionalModel(self.loans, config_path=trad_config_path)
            blockchain_model = BlockchainModel(self.loans, config_path=blockchain_config_path)
            
            # Compute scores
            trad_scores = trad_model.compute_scores()
            blockchain_scores = blockchain_model.compute_scores()
            
            # Calculate totals
            trad_total = sum(trad_scores.values())
            blockchain_total = sum(blockchain_scores.values())
            
            # Clean up temp files
            Path(trad_config_path).unlink(missing_ok=True)
            Path(blockchain_config_path).unlink(missing_ok=True)
            
            return {
                "simulation_id": sim_id,
                "traditional_total": trad_total,
                "blockchain_total": blockchain_total,
                "blockchain_wins": blockchain_total > trad_total,
                "difference": blockchain_total - trad_total,
                "traditional_scores": trad_scores,
                "blockchain_scores": blockchain_scores,
                "traditional_config": trad_config,
                "blockchain_config": blockchain_config
            }
            
        except Exception as e:
            logger.warning(f"Simulation {sim_id} failed: {e}")
            return {
                "simulation_id": sim_id,
                "error": str(e),
                "blockchain_wins": False,
                "difference": 0
            }
    
    def run_monte_carlo_analysis(self, n_threads: int = 4) -> Dict[str, Any]:
        """Run Monte Carlo analysis across parameter ranges.
        
        Args:
            n_threads: Number of threads for parallel processing.
            
        Returns:
            Dictionary containing analysis results.
        """
        logger.info(f"Running Monte Carlo analysis with {self.n_simulations} simulations")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Run simulations in parallel
        results = []
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            # Submit all simulations
            futures = {
                executor.submit(self._run_single_simulation, i): i 
                for i in range(self.n_simulations)
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=self.n_simulations, 
                             desc="Running simulations"):
                result = future.result()
                if 'error' not in result:
                    results.append(result)
        
        self.results = results
        return self._analyze_results()
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze the Monte Carlo simulation results.
        
        Returns:
            Dictionary containing statistical analysis.
        """
        if not self.results:
            return {"error": "No valid simulation results"}
        
        # Basic statistics
        blockchain_wins = sum(r["blockchain_wins"] for r in self.results)
        total_sims = len(self.results)
        win_probability = blockchain_wins / total_sims
        
        # Score differences
        differences = [r["difference"] for r in self.results]
        mean_difference = np.mean(differences)
        std_difference = np.std(differences)
        
        # Score distributions
        trad_totals = [r["traditional_total"] for r in self.results]
        blockchain_totals = [r["blockchain_total"] for r in self.results]
        
        # Factor-level analysis
        factor_analysis = self._analyze_factor_performance()
        
        # Confidence intervals
        confidence_interval = self._calculate_confidence_interval(differences)
        
        analysis = {
            "summary": {
                "total_simulations": total_sims,
                "blockchain_wins": blockchain_wins,
                "traditional_wins": total_sims - blockchain_wins,
                "win_probability": win_probability,
                "win_probability_percent": win_probability * 100
            },
            "score_statistics": {
                "mean_difference": mean_difference,
                "std_difference": std_difference,
                "min_difference": min(differences),
                "max_difference": max(differences),
                "median_difference": np.median(differences)
            },
            "distributions": {
                "traditional_scores": {
                    "mean": np.mean(trad_totals),
                    "std": np.std(trad_totals),
                    "min": min(trad_totals),
                    "max": max(trad_totals)
                },
                "blockchain_scores": {
                    "mean": np.mean(blockchain_totals),
                    "std": np.std(blockchain_totals),
                    "min": min(blockchain_totals),
                    "max": max(blockchain_totals)
                }
            },
            "confidence_interval_95": confidence_interval,
            "factor_analysis": factor_analysis
        }
        
        return analysis
    
    def _analyze_factor_performance(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance by individual factors.
        
        Returns:
            Dictionary containing factor-level statistics.
        """
        from .models.base import FACTORS
        
        factor_stats = {}
        
        for factor in FACTORS:
            trad_scores = [r["traditional_scores"][factor] for r in self.results 
                          if factor in r["traditional_scores"]]
            blockchain_scores = [r["blockchain_scores"][factor] for r in self.results
                               if factor in r["blockchain_scores"]]
            
            if trad_scores and blockchain_scores:
                differences = [b - t for b, t in zip(blockchain_scores, trad_scores)]
                blockchain_wins = sum(1 for d in differences if d > 0)
                
                factor_stats[factor] = {
                    "blockchain_win_rate": blockchain_wins / len(differences),
                    "mean_difference": np.mean(differences),
                    "std_difference": np.std(differences),
                    "traditional_mean": np.mean(trad_scores),
                    "blockchain_mean": np.mean(blockchain_scores)
                }
        
        return factor_stats
    
    def _calculate_confidence_interval(self, differences: List[float], 
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for score differences.
        
        Args:
            differences: List of score differences.
            confidence: Confidence level (default 0.95 for 95%).
            
        Returns:
            Tuple containing (lower_bound, upper_bound).
        """
        import scipy.stats as stats
        
        mean_diff = np.mean(differences)
        sem = stats.sem(differences)  # Standard error of mean
        
        # Calculate confidence interval
        alpha = 1 - confidence
        t_critical = stats.t.ppf(1 - alpha/2, len(differences) - 1)
        margin_error = t_critical * sem
        
        return (mean_diff - margin_error, mean_diff + margin_error)
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame.
        
        Returns:
            DataFrame containing all simulation results.
        """
        if not self.results:
            return pd.DataFrame()
        
        # Flatten results for DataFrame
        df_data = []
        for result in self.results:
            if 'error' not in result:
                row = {
                    'simulation_id': result['simulation_id'],
                    'traditional_total': result['traditional_total'],
                    'blockchain_total': result['blockchain_total'],
                    'difference': result['difference'],
                    'blockchain_wins': result['blockchain_wins']
                }
                
                # Add individual factor scores
                for factor, score in result['traditional_scores'].items():
                    row[f'traditional_{factor.lower().replace(" ", "_")}'] = score
                for factor, score in result['blockchain_scores'].items():
                    row[f'blockchain_{factor.lower().replace(" ", "_")}'] = score
                
                df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    def export_results(self, output_dir: str = "reports") -> Dict[str, str]:
        """Export results to files.
        
        Args:
            output_dir: Directory to save results.
            
        Returns:
            Dictionary containing file paths.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # Export summary analysis
        analysis = self._analyze_results()
        analysis_file = output_path / "statistical_analysis_summary.yaml"
        with open(analysis_file, 'w') as f:
            yaml.dump(analysis, f, default_flow_style=False)
        files['analysis'] = str(analysis_file)
        
        # Export detailed results
        df = self.get_results_dataframe()
        if not df.empty:
            results_file = output_path / "monte_carlo_results.csv"
            df.to_csv(results_file, index=False)
            files['results'] = str(results_file)
        
        logger.info(f"Results exported to {output_dir}")
        return files


def run_statistical_analysis(loans_df: pd.DataFrame, 
                           n_simulations: int = 1000,
                           n_threads: int = 4,
                           output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Run complete statistical analysis.
    
    Args:
        loans_df: DataFrame containing loan data.
        n_simulations: Number of Monte Carlo simulations.
        n_threads: Number of threads for parallel processing.
        output_dir: Optional directory to save results.
        
    Returns:
        Dictionary containing analysis results.
    """
    analyzer = StatisticalAnalyzer(loans_df, n_simulations)
    analysis = analyzer.run_monte_carlo_analysis(n_threads)
    
    if output_dir:
        analyzer.export_results(output_dir)
    
    return analysis 