"""
Model evaluation module for forex options.

This module provides functionality for evaluating and comparing different
option pricing models through backtesting and performance metrics.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm


@dataclass
class ModelPerformanceMetrics:
    """Data class for storing model performance metrics."""
    model: str
    avg_option_value_eur: float
    atm_avg_value_eur: float
    avg_delta: float
    atm_avg_delta: float
    avg_gamma: Optional[float]
    portfolio_pnl_volatility: Optional[float]
    avg_portfolio_value_eur: float
    avg_portfolio_delta: float


class ModelEvaluator:
    """
    Evaluates and compares different option pricing models.

    This class provides functionality for backtesting models over a time period
    and calculating various performance metrics.
    """

    def __init__(self, market_data, options_generator, portfolio_manager):
        """
        Initialize the model evaluator.

        Parameters
        ----------
        market_data : object
            Object containing market data
        options_generator : object
            Object containing the options portfolio
        portfolio_manager : object
            Object for portfolio management
        """
        self.market_data = market_data
        self.options_generator = options_generator
        self.portfolio_manager = portfolio_manager
        self.evaluation_results = None
        self.portfolio_metrics = None

    def run_backtest(self, start_date, end_date, freq='W', models=None, model_params=None):
        """
        Run a backtest of the models over a time period.

        Parameters
        ----------
        start_date : datetime
            Start date for backtest
        end_date : datetime
            End date for backtest
        freq : str
            Frequency for evaluation ('D' for daily, 'W' for weekly, 'M' for monthly)
        models : list
            List of models to evaluate
        model_params : dict
            Dictionary with model parameters

        Returns
        -------
        tuple
            (evaluation_results, portfolio_metrics) - DataFrames with results
        """
        if models is None:
            models = ['black_scholes', 'merton_jump', 'sabr']

        # Generate evaluation dates
        eval_dates = pd.date_range(start=start_date, end=end_date, freq=freq)

        # Store results for each date
        all_results = []
        portfolio_metrics = []

        # Run evaluation for each date
        for eval_date in tqdm(eval_dates, desc="Running backtest"):
            # Price portfolio
            try:
                pricing_results = self.portfolio_manager.price_portfolio(
                    eval_date, models, model_params)

                # Store individual option results
                for _, row in pricing_results.iterrows():
                    result = {
                        'eval_date': eval_date,
                        'option_id': row['option_id'],
                        'model': row['model'],
                        'price': row['price'],
                        'option_value_eur': row['option_value_eur'],
                        'delta': row.get('delta', np.nan),
                        'gamma': row.get('gamma', np.nan),
                        'vega': row.get('vega', np.nan),
                        'theta': row.get('theta', np.nan),
                        'spot': row['spot'],
                        'strike': row['strike'],
                        'T': row['T']
                    }
                    all_results.append(result)

                # Calculate portfolio metrics
                portfolio_risk = self.portfolio_manager.calculate_portfolio_risk(
                    pricing_results)

                for model, risk in portfolio_risk.items():
                    metric = {
                        'eval_date': eval_date,
                        'model': model,
                        'total_value_eur': risk.total_value_eur,
                        'total_delta': risk.total_delta,
                        'total_gamma': risk.total_gamma,
                        'total_vega': risk.total_vega,
                        'total_theta': risk.total_theta,
                        'count': risk.count
                    }
                    portfolio_metrics.append(metric)

            except Exception as e:
                print(f"Error evaluating models on {eval_date}: {e}")

        # Create DataFrames
        self.evaluation_results = pd.DataFrame(all_results)
        self.portfolio_metrics = pd.DataFrame(portfolio_metrics)

        return self.evaluation_results, self.portfolio_metrics

    def calculate_pnl_time_series(self):
        """
        Calculate P&L time series from backtest results.

        Returns
        -------
        pd.DataFrame
            DataFrame with P&L time series
        """
        if self.portfolio_metrics is None or self.portfolio_metrics.empty:
            raise ValueError(
                "No backtest results available. Run backtest first.")

        # Initialize results
        pnl_series = []

        # Calculate P&L for each model
        for model in self.portfolio_metrics['model'].unique():
            model_metrics = self.portfolio_metrics[self.portfolio_metrics['model'] == model].sort_values(
                'eval_date')

            if len(model_metrics) <= 1:
                continue

            # Calculate P&L series
            dates = model_metrics['eval_date'].values[1:]
            values = model_metrics['total_value_eur'].values
            pnl = np.diff(values)

            for i, date in enumerate(dates):
                pnl_series.append({
                    'date': date,
                    'model': model,
                    'portfolio_value': values[i+1],
                    'pnl': pnl[i],
                    'pnl_pct': pnl[i] / values[i] if values[i] > 0 else np.nan
                })

        return pd.DataFrame(pnl_series)

    def calculate_option_pnl_attribution(self):
        """
        Calculate P&L attribution for each option in the backtest.

        Returns
        -------
        pd.DataFrame
            DataFrame with P&L attribution by option
        """
        if self.evaluation_results is None or self.evaluation_results.empty:
            raise ValueError(
                "No backtest results available. Run backtest first.")

        # Initialize results
        pnl_attribution = []

        # Calculate P&L for each model and option
        for model in self.evaluation_results['model'].unique():
            model_results = self.evaluation_results[self.evaluation_results['model'] == model]

            # For each option
            for option_id in model_results['option_id'].unique():
                option_results = model_results[model_results['option_id'] == option_id].sort_values(
                    'eval_date')

                if len(option_results) <= 1:
                    continue

                # Calculate P&L series
                dates = option_results['eval_date'].values[1:]
                values = option_results['option_value_eur'].values
                deltas = option_results['delta'].values[:-1]
                spots = option_results['spot'].values

                pnl = np.diff(values)
                spot_changes = np.diff(spots)

                # Calculate P&L attribution
                for i, date in enumerate(dates):
                    # Skip if delta is not available
                    if pd.isna(deltas[i]):
                        continue

                    # Calculate P&L components
                    delta_pnl = deltas[i] * spot_changes[i]
                    other_pnl = pnl[i] - delta_pnl

                    pnl_attribution.append({
                        'date': date,
                        'model': model,
                        'option_id': option_id,
                        'option_value': values[i+1],
                        'total_pnl': pnl[i],
                        'delta_pnl': delta_pnl,
                        'other_pnl': other_pnl,
                        'delta_pct': delta_pnl / pnl[i] if pnl[i] != 0 else np.nan,
                        'other_pct': other_pnl / pnl[i] if pnl[i] != 0 else np.nan
                    })

        return pd.DataFrame(pnl_attribution)

    def calculate_performance_metrics(self):
        """
        Calculate performance metrics for each model.

        Returns
        -------
        pd.DataFrame
            DataFrame with performance metrics
        """
        if self.evaluation_results is None or self.portfolio_metrics is None:
            raise ValueError(
                "No evaluation results available. Run backtest first.")

        # Calculate metrics for each model
        model_metrics = []

        for model in self.evaluation_results['model'].unique():
            model_results = self.evaluation_results[self.evaluation_results['model'] == model]
            model_portfolio = self.portfolio_metrics[self.portfolio_metrics['model'] == model]

            # Calculate PnL volatility (standard deviation of daily changes)
            pnl_volatility = np.nan
            if len(model_portfolio) > 1:
                sorted_portfolio = model_portfolio.sort_values('eval_date')
                pnl_volatility = sorted_portfolio['total_value_eur'].diff(
                ).std()

            # Calculate average Delta and Gamma
            avg_delta = model_results['delta'].mean()
            avg_gamma = model_results['gamma'].mean(
            ) if 'gamma' in model_results.columns else np.nan

            # Calculate average option value
            avg_value = model_results['option_value_eur'].mean()

            # Calculate metrics for ATM options
            atm_mask = (model_results['strike'] /
                        model_results['spot']).between(0.98, 1.02)
            atm_results = model_results[atm_mask]

            atm_avg_value = atm_results['option_value_eur'].mean(
            ) if not atm_results.empty else np.nan
            atm_avg_delta = atm_results['delta'].mean(
            ) if not atm_results.empty else np.nan

            # Store metrics
            metric = ModelPerformanceMetrics(
                model=model,
                avg_option_value_eur=avg_value,
                atm_avg_value_eur=atm_avg_value,
                avg_delta=avg_delta,
                atm_avg_delta=atm_avg_delta,
                avg_gamma=avg_gamma,
                portfolio_pnl_volatility=pnl_volatility,
                avg_portfolio_value_eur=model_portfolio['total_value_eur'].mean(
                ),
                avg_portfolio_delta=model_portfolio['total_delta'].mean()
            )

            model_metrics.append(vars(metric))

        return pd.DataFrame(model_metrics)

    def calculate_model_error_metrics(self, reference_model='black_scholes'):
        """
        Calculate error metrics between models.

        Parameters
        ----------
        reference_model : str
            Model to use as reference for comparison

        Returns
        -------
        pd.DataFrame
            DataFrame with error metrics
        """
        if self.evaluation_results is None:
            raise ValueError(
                "No evaluation results available. Run backtest first.")

        if reference_model not in self.evaluation_results['model'].unique():
            raise ValueError(
                f"Reference model {reference_model} not found in results")

        # Calculate error metrics for each model and evaluation date
        error_metrics = []

        # Group by evaluation date
        for eval_date in self.evaluation_results['eval_date'].unique():
            date_results = self.evaluation_results[self.evaluation_results['eval_date'] == eval_date]

            ref_results = date_results[date_results['model']
                                       == reference_model]
            ref_results = ref_results.set_index('option_id')

            for model in date_results['model'].unique():
                if model == reference_model:
                    continue

                model_results = date_results[date_results['model'] == model]
                model_results = model_results.set_index('option_id')

                # Find common options
                common_options = set(ref_results.index) & set(
                    model_results.index)

                if not common_options:
                    continue

                # Calculate error metrics
                price_diff = [(model_results.loc[opt, 'price'] - ref_results.loc[opt, 'price'])
                              for opt in common_options]

                price_pct_diff = []
                for opt in common_options:
                    if ref_results.loc[opt, 'price'] > 0:
                        price_pct_diff.append((model_results.loc[opt, 'price'] /
                                               ref_results.loc[opt, 'price'] - 1) * 100)

                # Calculate metrics
                mse = np.mean(np.square(price_diff)) if price_diff else np.nan
                rmse = np.sqrt(mse) if not np.isnan(mse) else np.nan
                mae = np.mean(np.abs(price_diff)) if price_diff else np.nan
                mape = np.mean(np.abs(price_pct_diff)
                               ) if price_pct_diff else np.nan

                error_metrics.append({
                    'eval_date': eval_date,
                    'model': model,
                    'reference_model': reference_model,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape,
                    'options_count': len(common_options)
                })

        return pd.DataFrame(error_metrics)

    def calculate_model_rankings(self, metrics=None):
        """
        Calculate overall model rankings based on performance metrics.

        Parameters
        ----------
        metrics : pd.DataFrame
            Performance metrics (if None, calculate from backtest results)

        Returns
        -------
        pd.DataFrame
            DataFrame with model rankings
        """
        if metrics is None:
            metrics = self.calculate_performance_metrics()

        if metrics is None or metrics.empty:
            raise ValueError("No performance metrics available")

        # Define ranking criteria (lower is better for some metrics)
        ranking_criteria = {
            'portfolio_pnl_volatility': 'asc',  # Lower volatility is better
            'avg_option_value_eur': 'desc',     # Higher average value is better
            'avg_portfolio_value_eur': 'desc'   # Higher portfolio value is better
        }

        # Calculate rankings
        rankings = pd.DataFrame({'model': metrics['model']})

        for criterion, direction in ranking_criteria.items():
            if criterion in metrics.columns:
                # Skip criterion if it contains NaN values
                if metrics[criterion].isna().all():
                    continue

                # Calculate ranking (1 is best)
                if direction == 'asc':
                    ranks = metrics[criterion].rank()
                else:  # desc
                    ranks = metrics[criterion].rank(ascending=False)

                rankings[f'rank_{criterion}'] = ranks

        # Calculate average ranking
        ranking_cols = [
            col for col in rankings.columns if col.startswith('rank_')]
        if ranking_cols:
            rankings['avg_rank'] = rankings[ranking_cols].mean(axis=1)
            rankings['overall_rank'] = rankings['avg_rank'].rank()

        return rankings

    def save_evaluation_results(self, output_dir='results'):
        """
        Save evaluation results to CSV files.

        Parameters
        ----------
        output_dir : str
            Directory to save the CSV files
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.evaluation_results is not None and not self.evaluation_results.empty:
            self.evaluation_results.to_csv(os.path.join(
                output_dir, 'model_evaluation_results.csv'), index=False)

        if self.portfolio_metrics is not None and not self.portfolio_metrics.empty:
            self.portfolio_metrics.to_csv(os.path.join(
                output_dir, 'portfolio_metrics_timeseries.csv'), index=False)

        # Save performance metrics
        try:
            performance_metrics = self.calculate_performance_metrics()
            performance_metrics.to_csv(os.path.join(
                output_dir, 'model_performance_metrics.csv'), index=False)

            # Save rankings
            rankings = self.calculate_model_rankings(performance_metrics)
            rankings.to_csv(os.path.join(
                output_dir, 'model_rankings.csv'), index=False)

            # Save PnL time series
            pnl_series = self.calculate_pnl_time_series()
            pnl_series.to_csv(os.path.join(
                output_dir, 'pnl_time_series.csv'), index=False)

        except Exception as e:
            print(f"Error saving evaluation results: {e}")

    def load_evaluation_results(self, input_dir='results'):
        """
        Load evaluation results from CSV files.

        Parameters
        ----------
        input_dir : str
            Directory where CSV files are stored

        Returns
        -------
        bool
            True if files were loaded, False otherwise
        """
        try:
            eval_path = os.path.join(input_dir, 'model_evaluation_results.csv')
            metrics_path = os.path.join(
                input_dir, 'portfolio_metrics_timeseries.csv')

            if os.path.exists(eval_path) and os.path.exists(metrics_path):
                self.evaluation_results = pd.read_csv(eval_path)
                self.evaluation_results['eval_date'] = pd.to_datetime(
                    self.evaluation_results['eval_date'])

                self.portfolio_metrics = pd.read_csv(metrics_path)
                self.portfolio_metrics['eval_date'] = pd.to_datetime(
                    self.portfolio_metrics['eval_date'])

                return True
            else:
                return False
        except Exception as e:
            print(f"Error loading evaluation results: {e}")
            return False
