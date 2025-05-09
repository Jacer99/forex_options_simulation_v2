"""
Portfolio management module for forex options.

This module provides functionality for managing a portfolio of European options
on the EUR/TND currency pair, including pricing, risk calculation, and exposure analysis.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional

from .pricing import PricingEngine


@dataclass
class PortfolioRisk:
    """Data class for storing portfolio risk metrics."""
    total_value_eur: float
    total_delta: float
    total_gamma: Optional[float] = None
    total_vega: Optional[float] = None
    total_theta: Optional[float] = None
    total_rho_d: Optional[float] = None
    total_rho_f: Optional[float] = None
    count: int = 0


class PortfolioManager:
    """
    Manages a portfolio of forex options, handles pricing and risk management.

    This class provides functionality for pricing a portfolio of options using
    different models, calculating portfolio-level risk metrics, and analyzing
    exposure across different dimensions.
    """

    def __init__(self, market_data, options_generator):
        """
        Initialize the portfolio manager.

        Parameters
        ----------
        market_data : object
            Object containing market data
        options_generator : object
            Object containing the options portfolio
        """
        self.market_data = market_data
        self.options_generator = options_generator
        self.pricing_engine = PricingEngine()
        self.portfolio_results = None

    # Modifications to src/forex_options/portfolio.py

    def price_portfolio(self, pricing_date, models=None, model_params=None):
        """
        Price the entire portfolio using specified models.

        Parameters
        ----------
        pricing_date : datetime
            Date for which to price the portfolio
        models : list
            List of models to use for pricing
        model_params : dict
            Dictionary with model parameters

        Returns
        -------
        pd.DataFrame
            DataFrame with pricing results
        """
        if models is None:
            models = ['black_scholes', 'merton_jump', 'sabr']

        if model_params is None:
            model_params = {
                'merton_jump': {
                    'lambda': 1.0,
                    'mu_j': -0.05,
                    'sigma_j': 0.08
                },
                'sabr': {
                    'beta': 0.5,
                    'rho': -0.3,
                    'nu': 0.4
                }
            }

        # Get options portfolio
        portfolio = self.options_generator.options_portfolio

        if portfolio is None or portfolio.empty:
            print("No options portfolio available")
            self.portfolio_results = pd.DataFrame()
            return self.portfolio_results

        # Initialize results
        results = []

        # Get market data for pricing date
        try:
            market_data = self.market_data.get_market_data(pricing_date)
        except Exception as e:
            print(
                f"Error getting market data for pricing date {pricing_date}: {e}")
            # Return an empty DataFrame instead of failing
            self.portfolio_results = pd.DataFrame()
            return self.portfolio_results

        # Process each option
        valid_options_count = 0
        for _, option in portfolio.iterrows():
            # Skip options that haven't been issued yet
            if option['IssueDate'] > pricing_date:
                continue

            valid_options_count += 1

            # Extract option details
            issue_date = option['IssueDate']
            maturity_date = option['MaturityDate']
            strike = option['StrikePrice']
            spot = market_data['spot_rate']
            eur_rate = market_data['eur_rate']
            tnd_rate = market_data['tnd_rate']
            vol = market_data['volatility']

            # Calculate time to maturity in years
            if pricing_date > maturity_date:
                T = 0  # Option has expired
            else:
                T = (maturity_date - pricing_date).days / 365.0

            option_type = option['Type'].lower()

            # Price with each model
            for model in models:
                try:
                    # Prepare parameters for pricing
                    params = {
                        'S': spot,
                        'K': strike,
                        'T': T,
                        'r_d': eur_rate,
                        'r_f': tnd_rate,
                        'sigma': vol,
                        'option_type': option_type
                    }

                    # Add model-specific parameters
                    if model == 'merton_jump' or model == 'merton_jump_mc':
                        params.update(model_params.get('merton_jump', {}))
                    elif model == 'sabr':
                        params.update({
                            'alpha': vol,  # Use historical volatility as initial alpha
                            **model_params.get('sabr', {})
                        })

                    # Price the option
                    pricing_result = self.pricing_engine.price_option(
                        model, params)

                    # Calculate option value in EUR
                    notional = option['NotionalEUR']
                    contract_size = notional / \
                        option['SpotRate']  # Number of TND units
                    option_value_eur = pricing_result.price * contract_size

                    # Create result record
                    result = {
                        'pricing_date': pricing_date,
                        'option_id': option['OptionID'],
                        'model': model,
                        'issue_date': issue_date,
                        'maturity_date': maturity_date,
                        'T': T,
                        'spot': spot,
                        'strike': strike,
                        'price': pricing_result.price,
                        'delta': pricing_result.delta,
                        'gamma': pricing_result.gamma,
                        'vega': pricing_result.vega,
                        'theta': pricing_result.theta,
                        'rho_d': pricing_result.rho_d,
                        'rho_f': pricing_result.rho_f,
                        'eur_rate': eur_rate,
                        'tnd_rate': tnd_rate,
                        'volatility': vol,
                        'notional_eur': notional,
                        'option_value_eur': option_value_eur
                    }

                    results.append(result)
                except Exception as e:
                    print(
                        f"Error pricing option {option['OptionID']} with model {model}: {e}")

        # Check if we managed to price any options
        if not valid_options_count:
            print(f"No valid options to price for date {pricing_date}")

        # Convert to DataFrame
        if results:
            self.portfolio_results = pd.DataFrame(results)
            print(
                f"Successfully priced {len(self.portfolio_results) // len(models)} options with {len(models)} models.")
        else:
            # Create empty DataFrame with expected columns
            self.portfolio_results = pd.DataFrame(columns=[
                'pricing_date', 'option_id', 'model', 'issue_date', 'maturity_date',
                'T', 'spot', 'strike', 'price', 'delta', 'gamma', 'vega', 'theta',
                'rho_d', 'rho_f', 'eur_rate', 'tnd_rate', 'volatility',
                'notional_eur', 'option_value_eur'
            ])
            print(
                f"Warning: No options could be priced for date {pricing_date}")

        return self.portfolio_results

    def calculate_portfolio_risk(self, results=None):

        if results is None:
            results = self.portfolio_results

        if results is None or results.empty:
            print(
                "Warning: No portfolio results available. Returning empty risk metrics.")
            # Return empty risk metrics for available models
            empty_risk = {}
            if hasattr(self, 'pricing_engine') and hasattr(self.pricing_engine, 'models'):
                for model in self.pricing_engine.models.keys():
                    empty_risk[model] = PortfolioRisk(
                        total_value_eur=0.0,
                        total_delta=0.0,
                        count=0
                    )
            return empty_risk

    # Calculate portfolio metrics for each model
        portfolio_risk = {}

        for model in results['model'].unique():
            model_results = results[results['model'] == model]

            # Calculate total portfolio value
            total_value = model_results['option_value_eur'].sum()

            # Aggregate Greeks
            total_delta = (model_results['delta']
                           * model_results['notional_eur']).sum()

            # Optional Greeks (may not be available for all models)
            total_gamma = None
            if 'gamma' in model_results and not model_results['gamma'].isna().all():
                total_gamma = (
                    model_results['gamma'] * model_results['notional_eur']).sum()

            total_vega = None
            if 'vega' in model_results and not model_results['vega'].isna().all():
                total_vega = (model_results['vega'] *
                              model_results['notional_eur']).sum()

            total_theta = None
            if 'theta' in model_results and not model_results['theta'].isna().all():
                total_theta = (
                    model_results['theta'] * model_results['notional_eur']).sum()

            total_rho_d = None
            if 'rho_d' in model_results and not model_results['rho_d'].isna().all():
                total_rho_d = (
                    model_results['rho_d'] * model_results['notional_eur']).sum()

            total_rho_f = None
            if 'rho_f' in model_results and not model_results['rho_f'].isna().all():
                total_rho_f = (
                    model_results['rho_f'] * model_results['notional_eur']).sum()

        # Store results
            portfolio_risk[model] = PortfolioRisk(
                total_value_eur=total_value,
                total_delta=total_delta,
                total_gamma=total_gamma,
                total_vega=total_vega,
                total_theta=total_theta,
                total_rho_d=total_rho_d,
                total_rho_f=total_rho_f,
                count=len(model_results)
            )

        return portfolio_risk

    def calculate_exposure_by_maturity(self, results=None):
        """
        Calculate exposure by maturity buckets.

        Parameters
        ----------
        results : pd.DataFrame
            DataFrame with pricing results (if None, use self.portfolio_results)

        Returns
        -------
        dict
            Dictionary with exposure by maturity for each model
        """
        if results is None:
            results = self.portfolio_results

        if results is None or results.empty:
            print("Warning: No portfolio results available for exposure calculation.")
            return {}

        # Define maturity buckets (in months)
        buckets = [0, 1, 3, 6, 12]
        bucket_labels = ['0-1m', '1-3m', '3-6m', '6-12m', '>12m']

        # Calculate exposure by maturity
        exposure_by_maturity = {}

        for model in results['model'].unique():
            model_results = results[results['model'] == model]

            # Convert time to maturity to months
            model_results['T_months'] = model_results['T'] * 12

            # Initialize buckets
            bucket_exposure = {label: 0 for label in bucket_labels}

            # Assign exposure to buckets
            for i in range(len(buckets)):
                if i < len(buckets) - 1:
                    mask = (model_results['T_months'] >= buckets[i]) & (
                        model_results['T_months'] < buckets[i+1])
                else:
                    mask = model_results['T_months'] >= buckets[i]

                bucket_exposure[bucket_labels[i]
                                ] = model_results.loc[mask, 'option_value_eur'].sum()

            exposure_by_maturity[model] = bucket_exposure

        return exposure_by_maturity

    def calculate_exposure_by_moneyness(self, results=None):
        """
        Calculate exposure by moneyness.

        Parameters
        ----------
        results : pd.DataFrame
            DataFrame with pricing results (if None, use self.portfolio_results)

        Returns
        -------
        dict
            Dictionary with exposure by moneyness for each model
        """
        if results is None:
            results = self.portfolio_results

        if results is None or results.empty:
            print("Warning: No portfolio results available for moneyness calculation.")
            return {}

        # Define moneyness buckets
        buckets = [0, 0.95, 0.98, 1.02, 1.05, float('inf')]
        bucket_labels = ['Deep ITM', 'ITM', 'ATM', 'OTM', 'Deep OTM']

        # Calculate exposure by moneyness
        exposure_by_moneyness = {}

        for model in results['model'].unique():
            model_results = results[results['model'] == model]

            # Calculate moneyness (strike/spot for call options)
            model_results['moneyness'] = model_results['strike'] / \
                model_results['spot']

            # Initialize buckets
            bucket_exposure = {label: 0 for label in bucket_labels}

            # Assign exposure to buckets
            for i in range(len(buckets) - 1):
                mask = (model_results['moneyness'] >= buckets[i]) & (
                    model_results['moneyness'] < buckets[i+1])
                bucket_exposure[bucket_labels[i]
                                ] = model_results.loc[mask, 'option_value_eur'].sum()

            exposure_by_moneyness[model] = bucket_exposure

        return exposure_by_moneyness

    def calculate_pnl(self, results_t1, results_t0=None):
        """
        Calculate P&L between two pricing dates.

        Parameters
        ----------
        results_t1 : pd.DataFrame
            Pricing results at time t1
        results_t0 : pd.DataFrame
            Pricing results at time t0 (if None, use self.portfolio_results)

        Returns
        -------
        pd.DataFrame
            DataFrame with P&L analysis
        """
        if results_t0 is None:
            results_t0 = self.portfolio_results

        if results_t0 is None or results_t0.empty:
            raise ValueError("No initial pricing results available")

        if results_t1 is None or results_t1.empty:
            raise ValueError("No final pricing results available")

        # Ensure we're comparing the same options
        common_options = set(results_t0['option_id']) & set(
            results_t1['option_id'])

        if not common_options:
            raise ValueError("No common options between the two pricing sets")

        # Initialize PnL analysis
        pnl_analysis = []

        # For each model, calculate PnL
        for model in results_t0['model'].unique():
            if model not in results_t1['model'].unique():
                continue

            model_t0 = results_t0[results_t0['model'] == model]
            model_t1 = results_t1[results_t1['model'] == model]

            # Calculate PnL for each option
            for option_id in common_options:
                opt_t0 = model_t0[model_t0['option_id'] == option_id]
                opt_t1 = model_t1[model_t1['option_id'] == option_id]

                if opt_t0.empty or opt_t1.empty:
                    continue

                # Extract values
                t0_value = opt_t0['option_value_eur'].values[0]
                t1_value = opt_t1['option_value_eur'].values[0]
                t0_spot = opt_t0['spot'].values[0]
                t1_spot = opt_t1['spot'].values[0]
                t0_delta = opt_t0['delta'].values[0]

                # Calculate PnL
                total_pnl = t1_value - t0_value
                delta_pnl = t0_delta * \
                    (t1_spot - t0_spot) * \
                    opt_t0['notional_eur'].values[0] / t0_spot
                theta_pnl = total_pnl - delta_pnl

                # Store analysis
                pnl_analysis.append({
                    'model': model,
                    'option_id': option_id,
                    'date_t0': opt_t0['pricing_date'].values[0],
                    'date_t1': opt_t1['pricing_date'].values[0],
                    'value_t0': t0_value,
                    'value_t1': t1_value,
                    'total_pnl': total_pnl,
                    'delta_pnl': delta_pnl,
                    'theta_pnl': theta_pnl,
                    'spot_t0': t0_spot,
                    'spot_t1': t1_spot,
                    'delta_t0': t0_delta
                })

        return pd.DataFrame(pnl_analysis)

    def calculate_model_comparison(self, reference_model='black_scholes'):
        """
        Compare pricing and risk metrics across models.

        Parameters
        ----------
        reference_model : str
            Model to use as reference for comparison

        Returns
        -------
        pd.DataFrame
            DataFrame with model comparison metrics
        """
        if self.portfolio_results is None or self.portfolio_results.empty:
            raise ValueError(
                "No portfolio results available. Run price_portfolio first.")

        # Get unique options and models
        options = self.portfolio_results['option_id'].unique()
        models = self.portfolio_results['model'].unique()

        if reference_model not in models:
            raise ValueError(
                f"Reference model {reference_model} not found in results")

        # Extract reference model results
        ref_results = self.portfolio_results[self.portfolio_results['model']
                                             == reference_model]
        ref_results = ref_results.set_index('option_id')

        # Initialize comparison data
        comparison_data = []

        # Compare each model against the reference
        for model in models:
            if model == reference_model:
                continue

            model_results = self.portfolio_results[self.portfolio_results['model'] == model]
            model_results = model_results.set_index('option_id')

            # Calculate price differences
            common_options = set(ref_results.index) & set(model_results.index)

            if not common_options:
                continue

            # Price difference metrics
            price_diff = [(model_results.loc[opt, 'option_value_eur'] - ref_results.loc[opt, 'option_value_eur'])
                          for opt in common_options]

            price_diff_pct = []
            for opt in common_options:
                if ref_results.loc[opt, 'option_value_eur'] > 0:
                    price_diff_pct.append(model_results.loc[opt, 'option_value_eur'] /
                                          ref_results.loc[opt, 'option_value_eur'] - 1)

            # Delta difference metrics
            delta_diff = [(model_results.loc[opt, 'delta'] - ref_results.loc[opt, 'delta'])
                          for opt in common_options]

            # Risk metrics differences
            gamma_diff = []
            if 'gamma' in model_results and 'gamma' in ref_results:
                for opt in common_options:
                    if pd.notna(model_results.loc[opt, 'gamma']) and pd.notna(ref_results.loc[opt, 'gamma']):
                        gamma_diff.append(
                            model_results.loc[opt, 'gamma'] - ref_results.loc[opt, 'gamma'])

            vega_diff = []
            if 'vega' in model_results and 'vega' in ref_results:
                for opt in common_options:
                    if pd.notna(model_results.loc[opt, 'vega']) and pd.notna(ref_results.loc[opt, 'vega']):
                        vega_diff.append(
                            model_results.loc[opt, 'vega'] - ref_results.loc[opt, 'vega'])

            # Compile metrics
            comparison_data.append({
                'model': model,
                'reference_model': reference_model,
                'options_count': len(common_options),
                'mean_price_diff_eur': np.mean(price_diff) if price_diff else np.nan,
                'median_price_diff_eur': np.median(price_diff) if price_diff else np.nan,
                'max_price_diff_eur': np.max(price_diff) if price_diff else np.nan,
                'mean_price_diff_pct': np.mean(price_diff_pct) * 100 if price_diff_pct else np.nan,
                'median_price_diff_pct': np.median(price_diff_pct) * 100 if price_diff_pct else np.nan,
                'mean_delta_diff': np.mean(delta_diff) if delta_diff else np.nan,
                'mean_gamma_diff': np.mean(gamma_diff) if gamma_diff else np.nan,
                'mean_vega_diff': np.mean(vega_diff) if vega_diff else np.nan
            })

        return pd.DataFrame(comparison_data)

    def save_pricing_results(self, output_dir='results'):
        """
        Save pricing results to CSV files.

        Parameters
        ----------
        output_dir : str
            Directory to save the CSV files
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.portfolio_results is not None and not self.portfolio_results.empty:
            # Save overall results
            self.portfolio_results.to_csv(os.path.join(
                output_dir, 'portfolio_pricing_results.csv'), index=False)

            # Save results by model
            for model in self.portfolio_results['model'].unique():
                model_results = self.portfolio_results[self.portfolio_results['model'] == model]
                model_results.to_csv(os.path.join(
                    output_dir, f'{model}_pricing_results.csv'), index=False)

    def load_pricing_results(self, input_dir='results'):
        """
        Load pricing results from CSV file.

        Parameters
        ----------
        input_dir : str
            Directory where CSV files are stored

        Returns
        -------
        bool
            True if the file was loaded, False otherwise
        """
        try:
            results_path = os.path.join(
                input_dir, 'portfolio_pricing_results.csv')

            if os.path.exists(results_path):
                self.portfolio_results = pd.read_csv(results_path)
                self.portfolio_results['pricing_date'] = pd.to_datetime(
                    self.portfolio_results['pricing_date'])
                self.portfolio_results['issue_date'] = pd.to_datetime(
                    self.portfolio_results['issue_date'])
                self.portfolio_results['maturity_date'] = pd.to_datetime(
                    self.portfolio_results['maturity_date'])
                return True
            else:
                return False
        except Exception as e:
            print(f"Error loading pricing results: {e}")
            return False
