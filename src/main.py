#!/usr/bin/env python
"""
Forex Options Pricing and Risk Management System for EUR/TND

This script runs the forex options pricing and analysis system. It orchestrates the
various components: market data generation, options portfolio creation, pricing,
risk analysis, model evaluation, and visualization.
"""

from forex_options.visualization import VisualizationManager
from forex_options.evaluation import ModelEvaluator
from forex_options.portfolio import PortfolioManager
from forex_options.options_gen import OptionsGenerator
from forex_options.market_data import MarketDataGenerator
import os
import sys
import argparse
import yaml
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_config(config_path):
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the configuration file

    Returns
    -------
    dict
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return {}


def setup_directories(config):
    """
    Setup necessary directories.

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    dict
        Dictionary with directory paths
    """
    dirs = {
        'data': config.get('data_dir', 'data'),
        'results': config.get('results_dir', 'results'),
        'plots': config.get('plots_dir', 'plots')
    }

    for directory in dirs.values():
        os.makedirs(directory, exist_ok=True)

    return dirs


def run_market_data(config, dirs):
    """
    Run market data generation or loading.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    dirs : dict
        Directory paths

    Returns
    -------
    MarketDataGenerator
        Market data generator object
    """
    print("Initializing market data...")

    # Extract market data parameters
    market_config = config.get('market_data', {})
    base_eur_tnd_rate = market_config.get('base_eur_tnd_rate', 3.35)
    eur_rate_mean = market_config.get('eur_rate_mean', 0.03)
    tnd_rate_mean = market_config.get('tnd_rate_mean', 0.08)

    # Initialize market data generator
    market_data = MarketDataGenerator(
        start_date='2023-01-01',
        end_date='2024-12-31',
        base_eur_tnd_rate=base_eur_tnd_rate,
        eur_rate_mean=eur_rate_mean,
        tnd_rate_mean=tnd_rate_mean
    )

    # Check or generate data
    data_exists = market_data.check_or_generate_data(dirs['data'])
    if data_exists:
        print("Market data loaded/generated successfully.")
    else:
        print("Failed to load or generate market data.")
        sys.exit(1)

    return market_data


def run_options_generation(config, dirs, market_data):
    """
    Run options portfolio generation or loading.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    dirs : dict
        Directory paths
    market_data : MarketDataGenerator
        Market data generator object

    Returns
    -------
    OptionsGenerator
        Options generator object
    """
    print("Initializing options portfolio...")

    # Extract options parameters
    portfolio_config = config.get('portfolio', {})
    simulation_year = config.get('simulation_year', 2024)
    n_options = portfolio_config.get('n_options', 30)
    max_notional = portfolio_config.get('max_notional_eur', 10000000)

    # Initialize options generator
    options_generator = OptionsGenerator(
        market_data,
        simulation_year=simulation_year,
        max_notional=max_notional
    )

    # Check or generate options
    options_exist = options_generator.check_or_generate_options(dirs['data'])
    if options_exist:
        print(
            f"Options portfolio with {len(options_generator.options_portfolio)} options loaded/generated successfully.")
    else:
        print("Failed to load or generate options portfolio.")
        sys.exit(1)

    return options_generator


def run_portfolio_pricing(config, dirs, market_data, options_generator):
    """
    Run portfolio pricing.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    dirs : dict
        Directory paths
    market_data : MarketDataGenerator
        Market data generator object
    options_generator : OptionsGenerator
        Options generator object

    Returns
    -------
    PortfolioManager
        Portfolio manager object
    """
    print("Pricing options portfolio...")

    # Initialize portfolio manager
    portfolio_manager = PortfolioManager(market_data, options_generator)

    # Set pricing date (mid-year)
    pricing_date = pd.Timestamp(f"{config.get('simulation_year', 2024)}-06-01")

    # Extract model parameters
    models = config.get('models', ['black_scholes', 'merton_jump', 'sabr'])
    model_params = config.get('model_params', {})

    # Price portfolio
    try:
        results = portfolio_manager.price_portfolio(
            pricing_date, models, model_params)
        print(
            f"Portfolio pricing completed with {len(results)} results across {len(models)} models.")

        # Calculate portfolio risk
        portfolio_risk = portfolio_manager.calculate_portfolio_risk()
        for model, risk in portfolio_risk.items():
            print(
                f"{model} portfolio value: €{risk.total_value_eur:,.2f}, delta: {risk.total_delta:,.2f}")

        # Save results
        portfolio_manager.save_pricing_results(dirs['results'])
        print(f"Pricing results saved to {dirs['results']}/")
    except Exception as e:
        print(f"Error pricing portfolio: {e}")
        # Create an empty portfolio manager if pricing fails
        return portfolio_manager

    return portfolio_manager


def run_model_evaluation(config, dirs, market_data, options_generator, portfolio_manager):
    """
    Run model evaluation and backtesting.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    dirs : dict
        Directory paths
    market_data : MarketDataGenerator
        Market data generator object
    options_generator : OptionsGenerator
        Options generator object
    portfolio_manager : PortfolioManager
        Portfolio manager object

    Returns
    -------
    ModelEvaluator
        Model evaluator object
    """
    # Check if backtest is enabled
    backtest_config = config.get('backtest', {})
    if not backtest_config.get('enabled', False):
        print("Backtesting is disabled in configuration. Skipping...")
        return None

    print("Running model backtesting...")

    # Initialize model evaluator
    model_evaluator = ModelEvaluator(
        market_data, options_generator, portfolio_manager)

    # Extract backtest parameters
    start_date = pd.Timestamp(backtest_config.get(
        'start_date', f"{config.get('simulation_year', 2024)}-01-01"))
    end_date = pd.Timestamp(backtest_config.get(
        'end_date', f"{config.get('simulation_year', 2024)}-12-31"))
    frequency = backtest_config.get('frequency', 'M')

    # Extract model parameters
    models = config.get('models', ['black_scholes', 'merton_jump', 'sabr'])
    model_params = config.get('model_params', {})

    # Run backtest
    try:
        eval_results, portfolio_metrics = model_evaluator.run_backtest(
            start_date=start_date,
            end_date=end_date,
            freq=frequency,
            models=models,
            model_params=model_params
        )

        print(
            f"Backtest completed with {len(eval_results)} evaluation results across {len(models)} models.")

        # Calculate performance metrics
        performance_metrics = model_evaluator.calculate_performance_metrics()
        print("Model performance metrics:")
        for _, row in performance_metrics.iterrows():
            print(f"{row['model']}: avg value: €{row['avg_option_value_eur']:,.2f}, "
                  f"delta: {row['avg_delta']:.4f}")

        # Calculate model rankings
        rankings = model_evaluator.calculate_model_rankings(
            performance_metrics)
        if 'overall_rank' in rankings.columns:
            print("Model rankings:")
            for _, row in rankings.sort_values('overall_rank').iterrows():
                print(f"{row['model']}: rank {row['overall_rank']:.1f}")

        # Save evaluation results
        model_evaluator.save_evaluation_results(dirs['results'])
        print(f"Evaluation results saved to {dirs['results']}/")
    except Exception as e:
        print(f"Error running backtest: {e}")
        # Return the model evaluator even if the backtest fails
        return model_evaluator

    return model_evaluator


def run_visualizations(config, dirs, market_data, options_generator, portfolio_manager, model_evaluator):
    """
    Generate visualizations.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    dirs : dict
        Directory paths
    market_data : MarketDataGenerator
        Market data generator object
    options_generator : OptionsGenerator
        Options generator object
    portfolio_manager : PortfolioManager
        Portfolio manager object
    model_evaluator : ModelEvaluator
        Model evaluator object
    """
    print("Generating visualizations...")

    # Initialize visualization manager
    viz_manager = VisualizationManager(dirs['plots'])

    # Generate all plots
    viz_manager.generate_all_plots(
        market_data=market_data,
        options_generator=options_generator,
        portfolio_manager=portfolio_manager,
        model_evaluator=model_evaluator
    )


def run_system(config_path):
    """
    Run the entire forex options pricing and analysis system.

    Parameters
    ----------
    config_path : str
        Path to the configuration file
    """
    print("Starting Forex Options Pricing and Risk Management System for EUR/TND")
    print("=" * 80)

    # Load configuration
    config = load_config(config_path)

    # Setup directories
    dirs = setup_directories(config)

    # Initialize and run components
    market_data = run_market_data(config, dirs)
    options_generator = run_options_generation(config, dirs, market_data)
    portfolio_manager = run_portfolio_pricing(
        config, dirs, market_data, options_generator)
    model_evaluator = run_model_evaluation(
        config, dirs, market_data, options_generator, portfolio_manager)
    run_visualizations(config, dirs, market_data,
                       options_generator, portfolio_manager, model_evaluator)

    print("=" * 80)
    print("Processing complete!")
    print(f"Results stored in:")
    print(f"  - Data: {dirs['data']}/")
    print(f"  - Results: {dirs['results']}/")
    print(f"  - Plots: {dirs['plots']}/")


def parse_args():
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Forex Options Pricing and Risk Management System for EUR/TND')
    parser.add_argument('--config', '-c',
                        default='configs/simulation_config.yaml',
                        help='Path to configuration file')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_system(args.config)
