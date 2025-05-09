"""
Visualization module for forex options.

This module provides functions for visualizing market data, options portfolio,
pricing results, and model evaluation metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple, Union


class VisualizationManager:
    """
    Manages visualizations for the forex options pricing system.

    This class provides methods for generating various plots and charts
    to visualize market data, option portfolios, pricing results, and
    model evaluation metrics.
    """

    def __init__(self, output_dir='plots'):
        """
        Initialize the visualization manager.

        Parameters
        ----------
        output_dir : str
            Directory to save plots
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def plot_market_data(self, market_data, plot_types=None):
        """
        Plot market data visualizations.

        Parameters
        ----------
        market_data : object
            Object containing market data
        plot_types : list, optional
            List of plot types to generate, by default all plots
        """
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
            plt.savefig(os.path.join(self.output_dir, 'eur_tnd_rate.png'))
            plt.close()

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
            plt.savefig(os.path.join(
                self.output_dir, 'eur_tnd_volatility.png'))
            plt.close()

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
            plt.savefig(os.path.join(self.output_dir, 'interest_rates.png'))
            plt.close()

        # Plot interest rate differential
        if 'rate_differential' in plot_types:
            # Ensure dates match for proper calculation
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
            plt.savefig(os.path.join(self.output_dir,
                        'interest_rate_differential.png'))
            plt.close()

        # Plot return distribution
        if 'return_distribution' in plot_types:
            plt.figure(figsize=(10, 6))
            sns.histplot(market_data.eur_tnd_daily['Return'], kde=True)
            plt.title('EUR/TND Daily Return Distribution')
            plt.xlabel('Return')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir,
                        'eur_tnd_return_distribution.png'))
            plt.close()

            # QQ plot for returns
            plt.figure(figsize=(10, 6))
            import scipy.stats as stats
            stats.probplot(
                market_data.eur_tnd_daily['Return'].dropna(), dist="norm", plot=plt)
            plt.title('EUR/TND Returns QQ Plot')
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir,
                        'eur_tnd_return_qqplot.png'))
            plt.close()

    def plot_model_comparison_visualizations(self, portfolio_manager, output_dir='plots'):
        """
        Generate visualizations specifically for comparing pricing models with emphasis on jump effects.

        Parameters
        ----------
        portfolio_manager : object
            Object containing portfolio pricing results
        output_dir : str
            Directory to save plots
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if portfolio_manager.portfolio_results is None:
            raise ValueError(
                "No portfolio results available. Run price_portfolio first.")

        results = portfolio_manager.portfolio_results

        # 1. Plot price differences across models by moneyness
        plt.figure(figsize=(12, 8))

        # Reference model (usually Black-Scholes)
        ref_model = 'black_scholes'
        models = results['model'].unique()

        for model in models:
            if model == ref_model:
                continue

            # Calculate price differences by moneyness
            moneyness_bins = np.linspace(0.8, 1.2, 20)
            moneyness_centers = (moneyness_bins[:-1] + moneyness_bins[1:]) / 2

            price_diffs = []

            for i in range(len(moneyness_bins) - 1):
                lower = moneyness_bins[i]
                upper = moneyness_bins[i+1]

                # Filter options by moneyness
                mask = (results['strike'] / results['spot']
                        ).between(lower, upper)

                # Get results for both models
                ref_prices = results[(results['model'] ==
                                      ref_model) & mask]['price']
                model_prices = results[(
                    results['model'] == model) & mask]['price']

                # Match options by ID
                common_options = set(ref_prices.index) & set(
                    model_prices.index)

                if common_options:
                    # Calculate average price difference
                    diff = np.mean([model_prices.loc[opt] - ref_prices.loc[opt]
                                   for opt in common_options])
                    price_diffs.append(diff)
                else:
                    price_diffs.append(np.nan)

            plt.plot(moneyness_centers, price_diffs,
                     label=f'{model} - {ref_model}')

        plt.title('Average Price Differences vs. Black-Scholes by Moneyness')
        plt.xlabel('Moneyness (Strike/Spot)')
        plt.ylabel('Price Difference')
        plt.axhline(y=0, color='k', linestyle='--')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'price_diff_by_moneyness.png'))
        plt.close()

        # 2. Plot price differences across models by maturity
        plt.figure(figsize=(12, 8))

        for model in models:
            if model == ref_model:
                continue

            # Calculate price differences by maturity (in days)
            maturity_bins = np.array([0, 30, 60, 90, 180, 270, 360])
            maturity_centers = (maturity_bins[:-1] + maturity_bins[1:]) / 2

            price_diffs = []

            for i in range(len(maturity_bins) - 1):
                lower = maturity_bins[i]
                upper = maturity_bins[i+1]

                # Filter options by maturity (T in years)
                mask = (results['T'] * 365).between(lower, upper)

                # Get results for both models
                ref_prices = results[(results['model'] ==
                                      ref_model) & mask]['price']
                model_prices = results[(
                    results['model'] == model) & mask]['price']

                # Match options by ID
                common_options = set(ref_prices.index) & set(
                    model_prices.index)

                if common_options:
                    # Calculate average price difference
                    diff = np.mean([model_prices.loc[opt] - ref_prices.loc[opt]
                                   for opt in common_options])
                    price_diffs.append(diff)
                else:
                    price_diffs.append(np.nan)

            plt.plot(maturity_centers, price_diffs,
                     label=f'{model} - {ref_model}')

        plt.title('Average Price Differences vs. Black-Scholes by Maturity')
        plt.xlabel('Maturity (days)')
        plt.ylabel('Price Difference')
        plt.axhline(y=0, color='k', linestyle='--')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'price_diff_by_maturity.png'))
        plt.close()

        # 3. Heatmap of Jump Model vs Black-Scholes price differences
        if 'merton_jump' in models:
            plt.figure(figsize=(14, 10))

            # Create a grid of moneyness vs maturity
            moneyness_bins = np.linspace(0.8, 1.2, 10)
            maturity_bins = np.array([0, 30, 60, 90, 180, 270, 360])

            # Initialize heatmap data
            heatmap_data = np.zeros(
                (len(moneyness_bins)-1, len(maturity_bins)-1))

            for i in range(len(moneyness_bins) - 1):
                for j in range(len(maturity_bins) - 1):
                    m_lower = moneyness_bins[i]
                    m_upper = moneyness_bins[i+1]
                    t_lower = maturity_bins[j]
                    t_upper = maturity_bins[j+1]

                    # Filter options by moneyness and maturity
                    m_mask = (results['strike'] / results['spot']
                              ).between(m_lower, m_upper)
                    t_mask = (results['T'] * 365).between(t_lower, t_upper)
                    mask = m_mask & t_mask

                    # Get results for both models
                    bs_prices = results[(results['model'] ==
                                         ref_model) & mask]['price']
                    mj_prices = results[(results['model'] ==
                                         'merton_jump') & mask]['price']

                    # Match options by ID
                    common_options = set(
                        bs_prices.index) & set(mj_prices.index)

                    if common_options:
                        # Calculate average percentage price difference
                        pct_diffs = [100 * (mj_prices.loc[opt] / bs_prices.loc[opt] - 1)
                                     for opt in common_options]
                        heatmap_data[i, j] = np.mean(pct_diffs)
                    else:
                        heatmap_data[i, j] = np.nan

            # Create heatmap
            plt.figure(figsize=(14, 10))
            ax = sns.heatmap(heatmap_data,
                             xticklabels=[
                                 f"{m}-{m+1}" for m in maturity_bins[:-1]],
                             yticklabels=[
                                 f"{m:.2f}-{m+0.04:.2f}" for m in moneyness_bins[:-1]],
                             cmap='coolwarm', center=0, annot=True, fmt='.2f')
            plt.title(
                'Merton Jump-Diffusion vs Black-Scholes Price Difference (%)')
            plt.xlabel('Maturity (days)')
            plt.ylabel('Moneyness (Strike/Spot)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'jump_vs_bs_heatmap.png'))
            plt.close()

        # 4. Overlay volatility smiles for different models
        plt.figure(figsize=(12, 8))

        # Calculate implied volatilities for a range of strikes
        spot = results['spot'].iloc[0]  # Use the first spot rate
        strikes = np.linspace(0.8 * spot, 1.2 * spot, 20)
        T = 0.5  # 6 months
        r_d = results['eur_rate'].iloc[0]  # Domestic rate
        r_f = results['tnd_rate'].iloc[0]  # Foreign rate

        # Get implied volatilities for each model
        if 'black_scholes' in models:
            bs_vols = []
            for K in strikes:
                # Use Black-Scholes implied vol directly
                bs_vol = results[results['model'] ==
                                 'black_scholes']['volatility'].iloc[0]
                bs_vols.append(bs_vol)
            plt.plot(strikes / spot, bs_vols, label='Black-Scholes')

        if 'merton_jump' in models:
            mj_vols = []
            for K in strikes:
                # Calculate Merton Jump implied vol (back out from price)
                model_params = portfolio_manager.portfolio_manager.model_params.get(
                    'merton_jump', {})
                lam = model_params.get('lambda', 1.0)
                mu_j = model_params.get('mu_j', -0.05)
                sigma_j = model_params.get('sigma_j', 0.08)

                # Get base volatility
                base_vol = results[results['model'] ==
                                   'merton_jump']['volatility'].iloc[0]

                # Price option with jump model
                from forex_options.pricing import MertonJumpDiffusion
                mj_price = MertonJumpDiffusion.price(
                    spot, K, T, r_d, r_f, base_vol, lam, mu_j, sigma_j, 'call'
                ).price

                # Back out implied volatility from Black-Scholes
                from forex_options.pricing import BlackScholesFX

                def objective(sigma):
                    bs_price = BlackScholesFX.price(
                        spot, K, T, r_d, r_f, sigma, 'call').price
                    return (bs_price - mj_price) ** 2

                from scipy.optimize import minimize_scalar
                result = minimize_scalar(
                    objective, bounds=(0.01, 1.0), method='bounded')
                mj_vols.append(result.x)

            plt.plot(strikes / spot, mj_vols, label='Merton Jump-Diffusion')

        if 'sabr' in models:
            sabr_vols = []
            for K in strikes:
                # Calculate SABR implied vol
                model_params = portfolio_manager.portfolio_manager.model_params.get('sabr', {
                })
                alpha = results[results['model'] ==
                                'sabr']['volatility'].iloc[0]
                beta = model_params.get('beta', 0.5)
                rho = model_params.get('rho', -0.3)
                nu = model_params.get('nu', 0.4)

                # Get SABR implied vol
                from forex_options.pricing import SABR
                F = spot * np.exp((r_d - r_f) * T)
                sabr_vol = SABR.implied_vol(F, K, T, alpha, beta, rho, nu)
                sabr_vols.append(sabr_vol)

            plt.plot(strikes / spot, sabr_vols, label='SABR')

        plt.title('Implied Volatility Smile Comparison')
        plt.xlabel('Moneyness (Strike/Spot)')
        plt.ylabel('Implied Volatility')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'model_vol_smile_comparison.png'))
        plt.close()

    def plot_options_portfolio(self, options_generator, plot_types=None):
        """
        Plot options portfolio visualizations.

        Parameters
        ----------
        options_generator : object
            Object containing options portfolio
        plot_types : list, optional
            List of plot types to generate, by default all plots
        """
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
            plt.savefig(os.path.join(self.output_dir, 'option_maturities.png'))
            plt.close()

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
            plt.savefig(os.path.join(self.output_dir, 'option_moneyness.png'))
            plt.close()

        # Plot notional distribution
        if 'notional' in plot_types:
            plt.figure(figsize=(10, 6))
            sns.histplot(portfolio['NotionalEUR'], bins=20)
            plt.title('Distribution of Option Notional Amounts')
            plt.xlabel('Notional (EUR)')
            plt.ylabel('Count')
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, 'option_notionals.png'))
            plt.close()

        # Plot issue dates
        if 'issue_dates' in plot_types:
            plt.figure(figsize=(10, 6))
            sns.histplot(portfolio['IssueDate'], bins=12)
            plt.title('Distribution of Option Issue Dates')
            plt.xlabel('Issue Date')
            plt.ylabel('Count')
            plt.grid(True)
            plt.savefig(os.path.join(
                self.output_dir, 'option_issue_dates.png'))
            plt.close()

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
            plt.savefig(os.path.join(self.output_dir,
                                     'cumulative_notional.png'))
            plt.close()

        # Plot tenor vs. moneyness scatter
        if 'tenor_moneyness' in plot_types:
            plt.figure(figsize=(10, 6))
            plt.scatter(portfolio['Tenor'], portfolio['Moneyness'],
                        alpha=0.6, s=portfolio['NotionalEUR']/20000)
            plt.title('Option Tenor vs. Moneyness')
            plt.xlabel('Tenor (days)')
            plt.ylabel('Moneyness (Strike/Spot)')
            plt.axhline(y=1, color='red', linestyle='--', label='At-the-money')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(
                self.output_dir, 'tenor_vs_moneyness.png'))
            plt.close()

    def plot_pricing_results(self, portfolio_manager, plot_types=None):
        """
        Plot pricing results visualizations.

        Parameters
        ----------
        portfolio_manager : object
            Object containing portfolio pricing results
        plot_types : list, optional
            List of plot types to generate, by default all plots
        """
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
            plt.savefig(os.path.join(self.output_dir,
                                     'price_distribution_by_model.png'))
            plt.close()

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

                # Create bar plot
                values = [getattr(port_risk[model], metric)
                          for model in port_risk]
                plt.bar(list(port_risk.keys()), values)

                plt.title(
                    f'Portfolio {metric.replace("_", " ").title()} by Model')
                plt.xlabel('Model')
                plt.ylabel(metric.replace('_', ' ').title())
                plt.grid(True, axis='y')
                plt.savefig(os.path.join(self.output_dir,
                                         f'portfolio_{metric}_by_model.png'))
                plt.close()

        # Plot exposure by maturity
        if 'exposure_maturity' in plot_types:
            exp_maturity = portfolio_manager.calculate_exposure_by_maturity()

            plt.figure(figsize=(12, 6))

            # Convert to DataFrame for easier plotting
            exp_mat_df = pd.DataFrame(exp_maturity).T

            # Stacked bar chart
            exp_mat_df.plot(kind='bar', stacked=True, figsize=(12, 6))

            plt.title('Portfolio Exposure by Maturity')
            plt.xlabel('Model')
            plt.ylabel('Exposure (EUR)')
            plt.grid(True, axis='y')
            plt.savefig(os.path.join(self.output_dir,
                                     'exposure_by_maturity.png'))
            plt.close()

        # Plot exposure by moneyness
        if 'exposure_moneyness' in plot_types:
            exp_moneyness = portfolio_manager.calculate_exposure_by_moneyness()

            plt.figure(figsize=(12, 6))

            # Convert to DataFrame for easier plotting
            exp_mon_df = pd.DataFrame(exp_moneyness).T

            # Stacked bar chart
            exp_mon_df.plot(kind='bar', stacked=True, figsize=(12, 6))

            plt.title('Portfolio Exposure by Moneyness')
            plt.xlabel('Model')
            plt.ylabel('Exposure (EUR)')
            plt.grid(True, axis='y')
            plt.savefig(os.path.join(self.output_dir,
                                     'exposure_by_moneyness.png'))
            plt.close()

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
            plt.savefig(os.path.join(self.output_dir,
                                     'prices_vs_moneyness.png'))
            plt.close()

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
                        plt.savefig(os.path.join(self.output_dir,
                                                 f'{greek}_vs_moneyness_{model}.png'))
                        plt.close()

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
                    plt.savefig(os.path.join(self.output_dir,
                                             'model_price_differences.png'))
                    plt.close()

                    # Plot percentage differences
                    plt.figure(figsize=(10, 6))
                    plt.bar(model_comparison['model'],
                            model_comparison['mean_price_diff_pct'])
                    plt.xlabel('Model')
                    plt.ylabel('Mean Price Difference (%)')
                    plt.title('Mean Price Difference vs. Reference Model (%)')
                    plt.grid(True, axis='y')
                    plt.gca().yaxis.set_major_formatter(PercentFormatter())
                    plt.savefig(os.path.join(self.output_dir,
                                             'model_price_diff_percent.png'))
                    plt.close()
            except Exception as e:
                print(f"Error generating model comparison plots: {e}")

    def plot_volatility_surface(self, market_data, pricing_date, alpha=0.2, beta=0.5, rho=-0.3, nu=0.4):
        """
        Plot SABR volatility surface.

        Parameters
        ----------
        market_data : object
            Object containing market data
        pricing_date : datetime
            Date for which to generate the surface
        alpha : float, optional
            SABR alpha parameter, by default 0.2
        beta : float, optional
            SABR beta parameter, by default 0.5
        rho : float, optional
            SABR rho parameter, by default -0.3
        nu : float, optional
            SABR nu parameter, by default 0.4
        """
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

            plt.savefig(os.path.join(self.output_dir, 'sabr_vol_surface.png'))
            plt.close()

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
            plt.savefig(os.path.join(self.output_dir, 'sabr_vol_smiles.png'))
            plt.close()
        except Exception as e:
            print(f"Error generating SABR volatility surface: {e}")


def plot_evaluation_results(self, model_evaluator, plot_types=None):

"""
Plot model evaluation results.

Parameters
----------
model_evaluator : object
    Object containing model evaluation results
plot_types : list, optional
    List of plot types to generate, by default all plots
"""
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
            model_evaluator.portfolio_metrics['model'] == model]
        model_data = model_data.sort_values('eval_date')

        plt.plot(model_data['eval_date'],
                 model_data['total_value_eur'], label=model)

    plt.title('Portfolio Value Over Time by Model')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (EUR)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(self.output_dir,
                             'portfolio_value_time_series.png'))
    plt.close()

# Plot portfolio delta over time by model
if 'portfolio_delta' in plot_types:
    plt.figure(figsize=(12, 6))

    for model in model_evaluator.portfolio_metrics['model'].unique():
        model_data = model_evaluator.portfolio_metrics[
            model_evaluator.portfolio_metrics['model'] == model]
        model_data = model_data.sort_values('eval_date')

        plt.plot(model_data['eval_date'],
                 model_data['total_delta'], label=model)

    plt.title('Portfolio Delta Over Time by Model')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Delta')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(self.output_dir,
                             'portfolio_delta_time_series.png'))
    plt.close()

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
            plt.savefig(os.path.join(
                self.output_dir, 'model_rmse.png'))
            plt.close()

            # Plot MAPE
            plt.figure(figsize=(10, 6))
            plt.bar(avg_metrics.index, avg_metrics['mape'])
            plt.title('Average MAPE by Model')
            plt.xlabel('Model')
            plt.ylabel('MAPE (%)')
            plt.grid(True, axis='y')
            plt.gca().yaxis.set_major_formatter(PercentFormatter())
            plt.savefig(os.path.join(
                self.output_dir, 'model_mape.png'))
            plt.close()

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
                plt.savefig(os.path.join(self.output_dir,
                                         f'error_metrics_{model}.png'))
                plt.close()
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
            plt.savefig(os.path.join(
                self.output_dir, 'pnl_time_series.png'))
            plt.close()

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
            plt.savefig(os.path.join(
                self.output_dir, 'cumulative_pnl.png'))
            plt.close()

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
            plt.savefig(os.path.join(
                self.output_dir, 'pnl_distribution.png'))
            plt.close()
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
                    plt.savefig(os.path.join(
                        self.output_dir, f'{metric}_by_model.png'))
                    plt.close()
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
            plt.savefig(os.path.join(self.output_dir,
                                     'model_overall_ranking.png'))
            plt.close()

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
                plt.savefig(os.path.join(self.output_dir,
                                         'model_rankings_breakdown.png'))
                plt.close()
    except Exception as e:
        print(f"Error generating model ranking plots: {e}")

# Plot PnL time series
    # add all valid plot types
    plot_types = ['pnl_time_series', 'other_plot_types_that_should_be_here']
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
                plt.savefig(os.path.join(
                    self.output_dir, 'pnl_time_series.png'))
                plt.close()

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
                plt.savefig(os.path.join(
                    self.output_dir, 'cumulative_pnl.png'))
                plt.close()

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
                plt.savefig(os.path.join(
                    self.output_dir, 'pnl_distribution.png'))
                plt.close()
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
                        plt.savefig(os.path.join(
                            self.output_dir, f'{metric}_by_model.png'))
                        plt.close()
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
                plt.savefig(os.path.join(self.output_dir,
                            'model_overall_ranking.png'))
                plt.close()

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
                    plt.savefig(os.path.join(self.output_dir,
                                'model_rankings_breakdown.png'))
                    plt.close()
        except Exception as e:
            print(f"Error generating model ranking plots: {e}")


def plot_model_comparison_report(self, portfolio_manager, model_evaluator=None):
    """
    Generate a comprehensive model comparison report.

    Parameters
    ----------
    portfolio_manager : object
        Object containing portfolio pricing results
    model_evaluator : object, optional
        Object containing model evaluation results
    """
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
    plt.savefig(os.path.join(self.output_dir,
                'model_comparison_report.png'))
    plt.close()


def generate_all_plots(self, market_data, options_generator, portfolio_manager, model_evaluator=None):
    """
    Generate all plots in one go.

    Parameters
    ----------
    market_data : object
        Object containing market data
    options_generator : object
        Object containing options portfolio
    portfolio_manager : object
        Object containing portfolio pricing results
    model_evaluator : object, optional
        Object containing model evaluation results
    """
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
