#!/usr/bin/env python3
"""Main comparison script for traditional vs blockchain securitization."""

import argparse
import logging
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from securitization_comparison.data_loader import load_fannie_mae
from securitization_comparison.models.traditional import TraditionalModel
from securitization_comparison.models.blockchain import BlockchainModel
from securitization_comparison.viz import plot_scores, create_summary_table
from securitization_comparison.statistical_analysis import run_statistical_analysis


def setup_logging(debug: bool = False) -> None:
    """Set up logging configuration.
    
    Args:
        debug: Whether to enable debug logging.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('securitization_comparison.log')
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description='Compare traditional and blockchain-based securitization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=25000,
        help='Number of loans to sample for analysis'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing Fannie Mae data files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports/figures',
        help='Directory to save output visualizations'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable data caching'
    )
    
    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Force refresh of cached data'
    )
    
    parser.add_argument(
        '--traditional-config',
        type=str,
        help='Path to traditional model configuration file'
    )
    
    parser.add_argument(
        '--blockchain-config',
        type=str,
        help='Path to blockchain model configuration file'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation and display'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--statistical-analysis',
        action='store_true',
        help='Run Monte Carlo statistical analysis'
    )
    
    parser.add_argument(
        '--n-simulations',
        type=int,
        default=1000,
        help='Number of Monte Carlo simulations for statistical analysis'
    )
    
    parser.add_argument(
        '--n-threads',
        type=int,
        default=4,
        help='Number of threads for parallel processing'
    )
    
    return parser.parse_args()


def run_comparison(args: argparse.Namespace) -> None:
    """Run the securitization comparison analysis.
    
    Args:
        args: Parsed command line arguments.
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load data
        logger.info(f"Loading Fannie Mae data with sample size: {args.sample_size}")
        df = load_fannie_mae(
            data_dir=args.data_dir,
            sample_size=args.sample_size,
            use_cache=not args.no_cache,
            force_refresh=args.force_refresh
        )
        
        logger.info(f"Loaded {len(df)} loans with {len(df.columns)} features")
        
        # Create models
        logger.info("Initializing securitization models")
        traditional_model = TraditionalModel(df, config_path=args.traditional_config)
        blockchain_model = BlockchainModel(df, config_path=args.blockchain_config)
        
        # Compute scores
        logger.info("Computing traditional securitization scores")
        traditional_scores = traditional_model.compute_scores()
        
        logger.info("Computing blockchain securitization scores")
        blockchain_scores = blockchain_model.compute_scores()
        
        # Display results
        print("\n" + "="*80)
        print("SECURITIZATION COMPARISON RESULTS")
        print("="*80)
        
        print(f"\nDataset: {len(df):,} loans")
        print(f"Data source: {args.data_dir}")
        
        print("\nTraditional Securitization Scores:")
        for factor, score in traditional_scores.items():
            print(f"  {factor:20}: {score:6.1f}")
        
        print("\nBlockchain Securitization Scores:")
        for factor, score in blockchain_scores.items():
            print(f"  {factor:20}: {score:6.1f}")
        
        print("\nScore Differences (Blockchain - Traditional):")
        total_traditional = sum(traditional_scores.values())
        total_blockchain = sum(blockchain_scores.values())
        
        for factor in traditional_scores:
            diff = blockchain_scores[factor] - traditional_scores[factor]
            symbol = "+" if diff > 0 else ""
            print(f"  {factor:20}: {symbol}{diff:6.1f}")
        
        total_diff = total_blockchain - total_traditional
        symbol = "+" if total_diff > 0 else ""
        print(f"\n  {'TOTAL':20}: {symbol}{total_diff:6.1f}")
        
        # Determine overall winner
        if total_diff > 0:
            winner = "Blockchain"
        elif total_diff < 0:
            winner = "Traditional"
        else:
            winner = "Tie"
        
        print(f"\nOverall Winner: {winner}")
        print("="*80)
        
        # Generate visualizations and summary
        if not args.no_plots:
            logger.info("Generating visualizations")
            plot_scores(
                traditional_scores,
                blockchain_scores,
                output_dir=args.output_dir,
                show_plots=False,  # Don't show plots in script mode
                save_plots=True
            )
            
            # Create summary table
            logger.info("Creating summary table")
            create_summary_table(
                traditional_scores,
                blockchain_scores,
                output_dir=args.output_dir,
                save_table=True
            )
            
            print(f"\nVisualization files saved to: {args.output_dir}")
        
        # Run statistical analysis if requested
        if args.statistical_analysis:
            logger.info("Starting Monte Carlo statistical analysis")
            print("\n" + "="*80)
            print("STATISTICAL ANALYSIS")
            print("="*80)
            print(f"Running {args.n_simulations} Monte Carlo simulations...")
            print("This may take several minutes depending on sample size and number of simulations.")
            
            statistical_results = run_statistical_analysis(
                df, 
                n_simulations=args.n_simulations,
                n_threads=args.n_threads,
                output_dir=args.output_dir
            )
            
            # Display statistical results
            summary = statistical_results["summary"]
            print(f"\nMonte Carlo Results ({summary['total_simulations']} simulations):")
            print(f"  Blockchain wins: {summary['blockchain_wins']} ({summary['win_probability_percent']:.1f}%)")
            print(f"  Traditional wins: {summary['traditional_wins']} ({100-summary['win_probability_percent']:.1f}%)")
            print(f"  Win probability: {summary['win_probability']:.3f}")
            
            stats = statistical_results["score_statistics"]
            print(f"\nScore Difference Statistics:")
            print(f"  Mean difference: {stats['mean_difference']:+.1f}")
            print(f"  Standard deviation: {stats['std_difference']:.1f}")
            print(f"  Range: [{stats['min_difference']:+.1f}, {stats['max_difference']:+.1f}]")
            
            ci = statistical_results["confidence_interval_95"]
            print(f"  95% Confidence Interval: [{ci[0]:+.1f}, {ci[1]:+.1f}]")
            
            # Factor-level analysis
            print(f"\nFactor-Level Win Rates:")
            factor_analysis = statistical_results["factor_analysis"]
            for factor, stats in factor_analysis.items():
                win_rate = stats["blockchain_win_rate"] * 100
                mean_diff = stats["mean_difference"]
                print(f"  {factor:20}: {win_rate:5.1f}% ({mean_diff:+5.1f} avg diff)")
            
            print("\nStatistical results exported to:", args.output_dir)
            print("="*80)
        
        logger.info("Comparison completed successfully")
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        print(f"\nError: {e}")
        print("\nPlease ensure you have downloaded the Fannie Mae data.")
        print("See data/README.md for download instructions.")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    args = parse_arguments()
    setup_logging(debug=args.debug)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting securitization comparison")
    
    run_comparison(args)


if __name__ == "__main__":
    main() 