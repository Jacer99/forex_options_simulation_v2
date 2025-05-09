"""
Visualization module for forex options.

This module provides functions for visualizing market data, options portfolio,
pricing results, and model evaluation metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D


class VisualizationManager:
    """
    Manages visualizations for the forex options pricing system.
    """

    def __init__(self, output_dir='plots'):
        """Initialize the visualization manager."""
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _save_plot(self, filename):
        """Helper to save plot to the output directory and close it."""
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def plot_market_data(self, market_data, plot_types=None):
        """Plot market data visualizations."""
        if plot_types is None:
            plot_types = ['exchange_rate', 'volatility',
                          'interest_rates', 'rate_differential']

        if market_data.eur_tnd_daily is None or market_data.eur_rates_monthly is None or market_data.tnd_rates_monthly is None:
            raise ValueError(
                "Market data not available. Generate or load data first.")

        # Plot exchange rates
        if 'exchange_rate' in plot_types:
            plt.figure(figsize=(12, 6))
            plt.plot(
                market_data.eur_tnd_daily['Date'], market_data.eur_tnd_daily['EUR/TND'])
            plt.title('EUR/TND Exchange Rate')
            plt.xlabel('Date')
            plt.ylabel('Rate')
            plt.grid(True)
            self._save_plot('eur_tnd_rate.png')

        # Plot volatility
        if 'volatility' in plot_types:
            plt.figure(figsize=(12, 6))
            for window in [5, 21, 63]:
                vol_col = f'{window}d_Volatility'
                if vol_col in market_data.eur_tnd_daily.columns:
                    plt.plot(market_data.eur_tnd_daily['Date'],
                             market_data.eur_tnd_daily[vol_col],
                             label=f'{window}-day volatility')
            plt.title('EUR/TND Realized Volatility')
            plt.xlabel('Date')
            plt.ylabel('Annualized Volatility')
            plt.legend()
            plt.grid(True)
            self._save_plot('eur_tnd_volatility.png')

        # Plot interest rates
        if 'interest_rates' in plot_types:
            plt.figure(figsize=(12, 6))
            plt.plot(market_data.eur_rates_monthly['Date'],
                     market_data.eur_rates_monthly['EUR_Rate'],
                     label='EUR Interest Rate')
            plt.plot(market_data.tnd_rates_monthly['Date'],
                     market_data.tnd_rates_monthly['TND_Rate'],
                     label='TND Interest Rate')
            plt.title('Interest Rates')
            plt.xlabel('Date')
            plt.ylabel('Rate')
            plt.legend()
            plt.grid(True)
            self._save_plot('interest_rates.png')

        # Plot interest rate differential
        if 'rate_differential' in plot_types:
            common_dates = pd.merge(
                market_data.eur_rates_monthly[['Date']],
                market_data.tnd_rates_monthly[['Date']],
                on='Date'
            )['Date']

            eur_rates = market_data.eur_rates_monthly[market_data.eur_rates_monthly['Date'].isin(
                common_dates)]
            tnd_rates = market_data.tnd_rates_monthly[market_data.tnd_rates_monthly['Date'].isin(
                common_dates)]

            differential = tnd_rates['TND_Rate'].values - \
                eur_rates['EUR_Rate'].values

            plt.figure(figsize=(12, 6))
            plt.plot(common_dates, differential)
            plt.title('TND-EUR Interest Rate Differential')
            plt.xlabel('Date')
            plt.ylabel('Rate Differential')
            plt.grid(True)
            self._save_plot('interest_rate_differential.png')

    def plot_options_portfolio(self, options_generator, plot_types=None):
        """Plot options portfolio visualizations."""
        if plot_types is None:
            plot_types = ['maturity', 'moneyness', 'notional',
                          'issue_dates', 'cumulative_notional']

        if options_generator.options_portfolio is None:
            raise ValueError(
                "Options portfolio not available. Generate or load options first.")

        portfolio = options_generator.options_portfolio

        # Plot maturity distribution
        if 'maturity' in plot_types:
            plt.figure(figsize=(10, 6))
            sns.histplot(portfolio['Tenor'], bins=20)
            plt.title('Distribution of Option Maturities')
            plt.xlabel('Tenor (days)')
            plt.ylabel('Count')
            plt.grid(True)
            self._save_plot('option_maturities.png')

        # Plot moneyness distribution
        if 'moneyness' in plot_types:
            plt.figure(figsize=(10, 6))
            sns.histplot(portfolio['Moneyness'], bins=20)
            plt.title('Distribution of Option Moneyness')
            plt.xlabel('Moneyness (Strike/Spot)')
            plt.ylabel('Count')
            plt.axvline(x=1, color='red', linestyle='--', label='At-the-money')
            plt.legend()
            plt.grid(True)
            self._save_plot('option_moneyness.png')

        # Plot notional distribution
        if 'notional' in plot_types:
            plt.figure(figsize=(10, 6))
            sns.histplot(portfolio['NotionalEUR'], bins=20)
            plt.title('Distribution of Option Notional Amounts')
            plt.xlabel('Notional (EUR)')
            plt.ylabel('Count')
            plt.grid(True)
            self._save_plot('option_notionals.png')

        # Plot issue dates
        if 'issue_dates' in plot_types:
            plt.figure(figsize=(10, 6))
            sns.histplot(portfolio['IssueDate'], bins=12)
            plt.title('Distribution of Option Issue Dates')
            plt.xlabel('Issue Date')
            plt.ylabel('Count')
            plt.grid(True)
            self._save_plot('option_issue_dates.png')

        # Plot cumulative notional
        if 'cumulative_notional' in plot_types:
            sorted_portfolio = portfolio.sort_values('IssueDate')
            cum_notional = sorted_portfolio['NotionalEUR'].cumsum()

            plt.figure(figsize=(10, 6))
            plt.plot(sorted_portfolio['IssueDate'], cum_notional)
            plt.title('Cumulative Notional Amount')
            plt.xlabel('Issue Date')
            plt.ylabel('Cumulative Notional (EUR)')
            plt.grid(True)
            self._save_plot('cumulative_notional.png')

    def plot_pricing_results(self, portfolio_manager, plot_types=None):
        """Plot pricing results visualizations."""
        if plot_types is None:
            plot_types = ['price_distribution', 'portfolio_greeks', 'exposure_maturity',
                          'exposure_moneyness', 'prices_vs_moneyness', 'greeks_vs_moneyness',
                          'model_comparison']

        if portfolio_manager.portfolio_results is None:
            raise ValueError(
                "Portfolio results not available. Run price_portfolio first.")

        results = portfolio_manager.portfolio_results

        # Plot price distribution by model
        if 'price_distribution' in plot_types:
            plt.figure(figsize=(12, 6))
            for model in results['model'].unique():
                model_results = results[results['model'] == model]
                sns.histplot(
                    model_results['option_value_eur'], kde=True, label=model, alpha=0.5)

            plt.title('Option Price Distribution by Model')
            plt.xlabel('Option Value (EUR)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            self._save_plot('price_distribution_by_model.png')

        # Plot portfolio greeks by model
        if 'portfolio_greeks' in plot_types:
            port_risk = portfolio_manager.calculate_portfolio_risk()

            metrics = ['total_value_eur', 'total_delta',
                       'total_gamma', 'total_vega', 'total_theta']
            for metric in metrics:
                # Skip metrics not available for all models
                if not all(hasattr(port_risk[model], metric) and
                           getattr(port_risk[model], metric) is not None
                           for model in port_risk):
                    continue

                plt.figure(figsize=(10, 6))
                values = [getattr(port_risk[model], metric)
                          for model in port_risk]
                plt.bar(list(port_risk.keys()), values)
                plt.title(
                    f'Portfolio {metric.replace("_", " ").title()} by Model')
                plt.xlabel('Model')
                plt.ylabel(metric.replace('_', ' ').title())
                plt.grid(True, axis='y')
                self._save_plot(f'portfolio_{metric}_by_model.png')

        # Plot exposure by maturity
        if 'exposure_maturity' in plot_types:
            exp_maturity = portfolio_manager.calculate_exposure_by_maturity()
            exp_mat_df = pd.DataFrame(exp_maturity).T

            plt.figure(figsize=(12, 6))
            exp_mat_df.plot(kind='bar', stacked=True, figsize=(12, 6))
            plt.title('Portfolio Exposure by Maturity')
            plt.xlabel('Model')
            plt.ylabel('Exposure (EUR)')
            plt.grid(True, axis='y')
            self._save_plot('exposure_by_maturity.png')

        # Plot exposure by moneyness
        if 'exposure_moneyness' in plot_types:
            exp_moneyness = portfolio_manager.calculate_exposure_by_moneyness()
            exp_mon_df = pd.DataFrame(exp_moneyness).T

            plt.figure(figsize=(12, 6))
            exp_mon_df.plot(kind='bar', stacked=True, figsize=(12, 6))
            plt.title('Portfolio Exposure by Moneyness')
            plt.xlabel('Model')
            plt.ylabel('Exposure (EUR)')
            plt.grid(True, axis='y')
            self._save_plot('exposure_by_moneyness.png')

        # Scatter plot of option prices vs. moneyness by model
        if 'prices_vs_moneyness' in plot_types:
            plt.figure(figsize=(12, 6))
            for model in results['model'].unique():
                model_results = results[results['model'] == model]
                moneyness = model_results['strike'] / model_results['spot']
                plt.scatter(
                    moneyness, model_results['price'], alpha=0.7, label=model)

            plt.title('Option Prices vs. Moneyness by Model')
            plt.xlabel('Moneyness (Strike/Spot)')
            plt.ylabel('Option Price')
            plt.legend()
            plt.grid(True)
            self._save_plot('prices_vs_moneyness.png')

        # Plot Greeks vs. moneyness for each model
        if 'greeks_vs_moneyness' in plot_types:
            for model in results['model'].unique():
                model_results = results[results['model'] == model]
                moneyness = model_results['strike'] / model_results['spot']

                # Create a plot for each Greek
                for greek in ['delta', 'gamma', 'vega', 'theta']:
                    if greek in model_results and not model_results[greek].isna().all():
                        plt.figure(figsize=(10, 6))
                        plt.scatter(moneyness, model_results[greek], alpha=0.7)
                        plt.title(
                            f'{greek.capitalize()} vs. Moneyness ({model})')
                        plt.xlabel('Moneyness (Strike/Spot)')
                        plt.ylabel(greek.capitalize())
                        plt.grid(True)
                        self._save_plot(f'{greek}_vs_moneyness_{model}.png')

        # Plot model comparison
        if 'model_comparison' in plot_types:
            try:
                model_comparison = portfolio_manager.calculate_model_comparison()

                if not model_comparison.empty:
                    # Plot price differences
                    plt.figure(figsize=(10, 6))
                    metrics = ['mean_price_diff_eur',
                               'median_price_diff_eur', 'max_price_diff_eur']

                    x = np.arange(len(model_comparison))
                    width = 0.25

                    for i, metric in enumerate(metrics):
                        plt.bar(x + i*width, model_comparison[metric], width,
                                label=metric.replace('_', ' ').title())

                    plt.xlabel('Model')
                    plt.ylabel('Price Difference (EUR)')
                    plt.title('Price Differences vs. Reference Model')
                    plt.xticks(x + width, model_comparison['model'])
                    plt.legend()
                    plt.grid(True, axis='y')
                    self._save_plot('model_price_differences.png')

                    # Plot percentage differences
                    plt.figure(figsize=(10, 6))
                    plt.bar(model_comparison['model'],
                            model_comparison['mean_price_diff_pct'])
                    plt.xlabel('Model')
                    plt.ylabel('Mean Price Difference (%)')
                    plt.title('Mean Price Difference vs. Reference Model (%)')
                    plt.grid(True, axis='y')
                    plt.gca().yaxis.set_major_formatter(PercentFormatter())
                    self._save_plot('model_price_diff_percent.png')
            except Exception as e:
                print(f"Error generating model comparison plots: {e}")

    def plot_volatility_surface(self, market_data, pricing_date, alpha=0.2, beta=0.5, rho=-0.3, nu=0.4):
        """Plot SABR volatility surface."""
        try:
            from .pricing import SABR

            # Get market data for the pricing date
            market_info = market_data.get_market_data(pricing_date)
            spot = market_info['spot_rate']
            r_d = market_info['eur_rate']
            r_f = market_info['tnd_rate']

            # Parameters for the surface
            strikes = np.linspace(0.8 * spot, 1.2 * spot, 20)
            tenors = np.linspace(0.1, 1.0, 10)

            # Create mesh grid
            K, T = np.meshgrid(strikes, tenors)
            implied_vols = np.zeros_like(K)

            # Calculate implied volatilities
            for i in range(tenors.shape[0]):
                for j in range(strikes.shape[0]):
                    # Forward price
                    F = spot * np.exp((r_d - r_f) * tenors[i])
                    # SABR implied volatility
                    implied_vols[i, j] = SABR.implied_vol(
                        F, strikes[j], tenors[i], alpha, beta, rho, nu)

            # Plot the surface
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(
                K, T, implied_vols, cmap='viridis', edgecolor='none')
            ax.set_xlabel('Strike')
            ax.set_ylabel('Tenor (years)')
            ax.set_zlabel('Implied Volatility')
            ax.set_title('SABR Implied Volatility Surface')
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            self._save_plot('sabr_vol_surface.png')

            # Plot volatility smiles for different tenors
            plt.figure(figsize=(12, 6))
            for i, tenor in enumerate([0, 2, 5, 9]):  # Sample tenors
                if tenor < len(tenors):
                    plt.plot(
                        strikes, implied_vols[tenor, :], label=f'T = {tenors[tenor]:.2f}')
            plt.axvline(x=spot, color='black', linestyle='--', label='Spot')
            plt.title('SABR Volatility Smiles for Different Tenors')
            plt.xlabel('Strike')
            plt.ylabel('Implied Volatility')
            plt.legend()
            plt.grid(True)
            self._save_plot('sabr_vol_smiles.png')
        except Exception as e:
            print(f"Error generating SABR volatility surface: {e}")

    def plot_evaluation_results(self, model_evaluator, plot_types=None):
        """Plot model evaluation results."""
        if plot_types is None:
            plot_types = ['portfolio_value', 'portfolio_delta', 'error_metrics',
                          'pnl_time_series', 'performance_metrics', 'model_rankings']

        if model_evaluator.evaluation_results is None or model_evaluator.portfolio_metrics is None:
            raise ValueError(
                "Evaluation results not available. Run backtest first.")

        # Plot portfolio value over time by model
        if 'portfolio_value' in plot_types:
            plt.figure(figsize=(12, 6))
            for model in model_evaluator.portfolio_metrics['model'].unique():
                model_data = model_evaluator.portfolio_metrics[
                    model_evaluator.portfolio_metrics['model'] == model].sort_values('eval_date')
                plt.plot(model_data['eval_date'],
                         model_data['total_value_eur'], label=model)
            plt.title('Portfolio Value Over Time by Model')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value (EUR)')
            plt.legend()
            plt.grid(True)
            self._save_plot('portfolio_value_time_series.png')

        # Plot portfolio delta over time by model
        if 'portfolio_delta' in plot_types:
            plt.figure(figsize=(12, 6))
            for model in model_evaluator.portfolio_metrics['model'].unique():
                model_data = model_evaluator.portfolio_metrics[
                    model_evaluator.portfolio_metrics['model'] == model].sort_values('eval_date')
                plt.plot(model_data['eval_date'],
                         model_data['total_delta'], label=model)
            plt.title('Portfolio Delta Over Time by Model')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Delta')
            plt.legend()
            plt.grid(True)
            self._save_plot('portfolio_delta_time_series.png')

        # Plot error metrics
        if 'error_metrics' in plot_types:
            try:
                error_metrics = model_evaluator.calculate_model_error_metrics()
                if not error_metrics.empty:
                    # Calculate average error metrics by model
                    avg_metrics = error_metrics.groupby('model').mean()

                    # Plot RMSE
                    plt.figure(figsize=(10, 6))
                    plt.bar(avg_metrics.index, avg_metrics['rmse'])
                    plt.title('Average RMSE by Model')
                    plt.xlabel('Model')
                    plt.ylabel('RMSE')
                    plt.grid(True, axis='y')
                    self._save_plot('model_rmse.png')

                    # Plot MAPE
                    plt.figure(figsize=(10, 6))
                    plt.bar(avg_metrics.index, avg_metrics['mape'])
                    plt.title('Average MAPE by Model')
                    plt.xlabel('Model')
                    plt.ylabel('MAPE (%)')
                    plt.grid(True, axis='y')
                    plt.gca().yaxis.set_major_formatter(PercentFormatter())
                    self._save_plot('model_mape.png')

                    # Plot error metrics over time
                    for model in error_metrics['model'].unique():
                        model_errors = error_metrics[error_metrics['model'] == model].sort_values(
                            'eval_date')
                        plt.figure(figsize=(12, 6))
                        plt.plot(model_errors['eval_date'],
                                 model_errors['rmse'], label='RMSE')
                        plt.plot(model_errors['eval_date'],
                                 model_errors['mae'], label='MAE')
                        plt.title(f'Error Metrics Over Time - {model}')
                        plt.xlabel('Date')
                        plt.ylabel('Error')
                        plt.legend()
                        plt.grid(True)
                        self._save_plot(f'error_metrics_{model}.png')
            except Exception as e:
                print(f"Error generating error metrics plots: {e}")

        # Plot PnL time series
        if 'pnl_time_series' in plot_types:
            try:
                pnl_series = model_evaluator.calculate_pnl_time_series()
                if not pnl_series.empty:
                    # Plot PnL time series
                    plt.figure(figsize=(12, 6))
                    for model in pnl_series['model'].unique():
                        model_pnl = pnl_series[pnl_series['model'] == model].sort_values(
                            'date')
                        plt.plot(model_pnl['date'],
                                 model_pnl['pnl'], label=model)
                    plt.title('Portfolio PnL Over Time by Model')
                    plt.xlabel('Date')
                    plt.ylabel('PnL (EUR)')
                    plt.legend()
                    plt.grid(True)
                    self._save_plot('pnl_time_series.png')

                    # Plot cumulative PnL
                    plt.figure(figsize=(12, 6))
                    for model in pnl_series['model'].unique():
                        model_pnl = pnl_series[pnl_series['model'] == model].sort_values(
                            'date')
                        plt.plot(
                            model_pnl['date'], model_pnl['pnl'].cumsum(), label=model)
                    plt.title('Cumulative Portfolio PnL by Model')
                    plt.xlabel('Date')
                    plt.ylabel('Cumulative PnL (EUR)')
                    plt.legend()
                    plt.grid(True)
                    self._save_plot('cumulative_pnl.png')

                    # Plot PnL distribution
                    plt.figure(figsize=(12, 6))
                    for model in pnl_series['model'].unique():
                        model_pnl = pnl_series[pnl_series['model'] == model]
                        sns.histplot(model_pnl['pnl'],
                                     kde=True, label=model, alpha=0.5)
                    plt.title('PnL Distribution by Model')
                    plt.xlabel('PnL (EUR)')
                    plt.ylabel('Frequency')
                    plt.legend()
                    plt.grid(True)
                    self._save_plot('pnl_distribution.png')
            except Exception as e:
                print(f"Error generating PnL plots: {e}")

        # Plot performance metrics
        if 'performance_metrics' in plot_types:
            try:
                performance_metrics = model_evaluator.calculate_performance_metrics()
                if not performance_metrics.empty:
                    # Plot key performance metrics
                    metrics = ['avg_option_value_eur',
                               'portfolio_pnl_volatility', 'avg_portfolio_value_eur']
                    for metric in metrics:
                        if metric in performance_metrics and not performance_metrics[metric].isna().all():
                            plt.figure(figsize=(10, 6))
                            plt.bar(
                                performance_metrics['model'], performance_metrics[metric])
                            plt.title(
                                f'{metric.replace("_", " ").title()} by Model')
                            plt.xlabel('Model')
                            plt.ylabel(metric.replace('_', ' ').title())
                            plt.grid(True, axis='y')
                            self._save_plot(f'{metric}_by_model.png')
            except Exception as e:
                print(f"Error generating performance metrics plots: {e}")

        # Plot model rankings
        if 'model_rankings' in plot_types:
            try:
                rankings = model_evaluator.calculate_model_rankings()
                if not rankings.empty and 'overall_rank' in rankings:
                    plt.figure(figsize=(10, 6))
                    plt.bar(rankings['model'], rankings['overall_rank'])
                    plt.title('Overall Model Ranking')
                    plt.xlabel('Model')
                    plt.ylabel('Rank (Lower is Better)')
                    plt.grid(True, axis='y')
                    self._save_plot('model_overall_ranking.png')

                    # Plot breakdown of rankings
                    ranking_cols = [
                        col for col in rankings.columns if col.startswith('rank_')]
                    if ranking_cols:
                        plt.figure(figsize=(12, 6))
                        x = np.arange(len(rankings))
                        width = 0.8 / len(ranking_cols)
                        for i, col in enumerate(ranking_cols):
                            plt.bar(x + i*width, rankings[col], width,
                                    label=col.replace('rank_', '').replace('_', ' ').title())
                        plt.xlabel('Model')
                        plt.ylabel('Rank (Lower is Better)')
                        plt.title('Model Rankings by Criterion')
                        plt.xticks(x + width*len(ranking_cols) /
                                   2, rankings['model'])
                        plt.legend()
                        plt.grid(True, axis='y')
                        self._save_plot('model_rankings_breakdown.png')
            except Exception as e:
                print(f"Error generating model ranking plots: {e}")

    def plot_model_comparison_report(self, portfolio_manager, model_evaluator=None):
        """Generate a comprehensive model comparison report."""
        if portfolio_manager.portfolio_results is None:
            raise ValueError(
                "Portfolio results not available. Run price_portfolio first.")

        # Calculate model comparison
        comparison = portfolio_manager.calculate_model_comparison()
        if comparison.empty:
            print("No model comparison data available.")
            return

        # Create a comprehensive comparison figure
        fig = plt.figure(figsize=(15, 12))
        fig.suptitle('Model Comparison Report', fontsize=16)

        # 1. Price differences
        ax1 = fig.add_subplot(221)
        metrics = ['mean_price_diff_eur', 'median_price_diff_eur']
        for metric in metrics:
            ax1.bar(comparison['model'] + ' - ' + metric, comparison[metric])
        ax1.set_title('Price Differences vs. Reference Model')
        ax1.set_ylabel('Difference (EUR)')
        ax1.grid(True, axis='y')
        plt.xticks(rotation=45, ha='right')

        # 2. Percentage differences
        ax2 = fig.add_subplot(222)
        ax2.bar(comparison['model'], comparison['mean_price_diff_pct'])
        ax2.set_title('Mean Price Difference (%)')
        ax2.set_ylabel('Difference (%)')
        ax2.yaxis.set_major_formatter(PercentFormatter())
        ax2.grid(True, axis='y')

        # 3. Greeks differences
        ax3 = fig.add_subplot(223)
        greek_metrics = ['mean_delta_diff',
                         'mean_gamma_diff', 'mean_vega_diff']
        greek_metrics = [m for m in greek_metrics if m in comparison.columns]
        for metric in greek_metrics:
            if not comparison[metric].isna().all():
                ax3.bar(comparison['model'] + ' - ' +
                        metric, comparison[metric])
        ax3.set_title('Greeks Differences')
        ax3.grid(True, axis='y')
        plt.xticks(rotation=45, ha='right')

        # 4. Model info
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        model_info = "Model Information:\n\n"

        for model in comparison['model'].unique():
            model_info += f"- {model}:\n"
            model_info += f"  - Mean price diff: {comparison[comparison['model'] == model]['mean_price_diff_eur'].values[0]:.4f} EUR\n"
            model_info += f"  - Mean price diff %: {comparison[comparison['model'] == model]['mean_price_diff_pct'].values[0]:.2f}%\n"

            # Add performance metrics if available
            if model_evaluator is not None and model_evaluator.evaluation_results is not None:
                try:
                    performance = model_evaluator.calculate_performance_metrics()
                    if not performance.empty and model in performance['model'].values:
                        model_perf = performance[performance['model']
                                                 == model].iloc[0]
                        model_info += f"  - Avg portfolio value: {model_perf['avg_portfolio_value_eur']:.2f} EUR\n"
                        if not pd.isna(model_perf['portfolio_pnl_volatility']):
                            model_info += f"  - PnL volatility: {model_perf['portfolio_pnl_volatility']:.2f} EUR\n"
                except Exception:
                    pass
            model_info += "\n"

        ax4.text(0, 1, model_info, fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        self._save_plot('model_comparison_report.png')

    def generate_all_plots(self, market_data, options_generator, portfolio_manager, model_evaluator=None):
        """Generate all plots in one go."""
        try:
            print("Generating market data plots...")
            self.plot_market_data(market_data)
        except Exception as e:
            print(f"Error generating market data plots: {e}")

        try:
            print("Generating options portfolio plots...")
            self.plot_options_portfolio(options_generator)
        except Exception as e:
            print(f"Error generating options portfolio plots: {e}")

        try:
            print("Generating pricing results plots...")
            self.plot_pricing_results(portfolio_manager)
        except Exception as e:
            print(f"Error generating pricing results plots: {e}")

        try:
            print("Generating volatility surface...")
            self.plot_volatility_surface(
                market_data, pricing_date=pd.Timestamp('2024-06-01'))
        except Exception as e:
            print(f"Error generating volatility surface: {e}")

        if model_evaluator is not None:
            try:
                print("Generating evaluation results plots...")
                self.plot_evaluation_results(model_evaluator)
            except Exception as e:
                print(f"Error generating evaluation results plots: {e}")

            try:
                print("Generating model comparison report...")
                self.plot_model_comparison_report(
                    portfolio_manager, model_evaluator)
            except Exception as e:
                print(f"Error generating model comparison report: {e}")

        print(f"All plots saved to {self.output_dir}/")
