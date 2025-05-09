"""
Unit tests for the portfolio management module.

This module tests the portfolio management functionality for forex options.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from forex_options.market_data import MarketDataGenerator
from forex_options.options_gen import OptionsGenerator
from forex_options.portfolio import PortfolioManager


class TestPortfolioManager(unittest.TestCase):
    """Test cases for the PortfolioManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Initialize market data generator with a reduced date range for testing
        self.market_data = MarketDataGenerator(
            start_date='2024-01-01',
            end_date='2024-12-31',
            base_eur_tnd_rate=3.35,
            eur_rate_mean=0.03,
            tnd_rate_mean=0.08
        )

        # Generate market data
        self.market_data.generate_eur_tnd_daily(seed=42)
        self.market_data.generate_interest_rates(seed=42)

        # Initialize options generator
        self.options_generator = OptionsGenerator(
            self.market_data,
            simulation_year=2024,
            max_notional=5_000_000  # Reduced for testing
        )

        # Generate options portfolio with fewer options for testing
        self.options_generator.generate_options_portfolio(
            n_options=10, seed=42)

        # Initialize portfolio manager
        self.portfolio_manager = PortfolioManager(
            self.market_data, self.options_generator)

    def test_portfolio_pricing(self):
        """Test portfolio pricing functionality."""
        # Price portfolio with a specific date
        pricing_date = pd.Timestamp('2024-06-01')
        models = ['black_scholes', 'merton_jump']  # Reduced set for testing
        model_params = {
            'merton_jump': {
                'lambda': 1.0,
                'mu_j': -0.05,
                'sigma_j': 0.08
            }
        }

        results = self.portfolio_manager.price_portfolio(
            pricing_date, models, model_params)

        # Check results
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)

        # Check columns
        expected_columns = ['pricing_date', 'option_id', 'model', 'price', 'delta',
                            'option_value_eur', 'notional_eur']
        for col in expected_columns:
            self.assertIn(col, results.columns)

        # Check models
        self.assertEqual(set(results['model'].unique()), set(models))

        # Check values
        self.assertTrue((results['price'] >= 0).all())
        self.assertTrue((results['option_value_eur'] >= 0).all())

        # Delta should be between 0 and 1 for call options
        self.assertTrue((results['delta'] >= 0).all())
        self.assertTrue((results['delta'] <= 1).all())

    def test_portfolio_risk_calculation(self):
        """Test portfolio risk calculation."""
        # First price the portfolio
        pricing_date = pd.Timestamp('2024-06-01')
        models = ['black_scholes', 'merton_jump']
        self.portfolio_manager.price_portfolio(pricing_date, models)

        # Calculate portfolio risk
        portfolio_risk = self.portfolio_manager.calculate_portfolio_risk()

        # Check risk metrics
        self.assertIsNotNone(portfolio_risk)
        self.assertEqual(set(portfolio_risk.keys()), set(models))

        for model, risk in portfolio_risk.items():
            # Check attributes
            self.assertTrue(hasattr(risk, 'total_value_eur'))
            self.assertTrue(hasattr(risk, 'total_delta'))
            self.assertTrue(hasattr(risk, 'count'))

            # Check values
            self.assertGreater(risk.total_value_eur, 0)
            self.assertEqual(risk.count, len(self.portfolio_manager.portfolio_results[
                self.portfolio_manager.portfolio_results['model'] == model]))

    def test_exposure_calculations(self):
        """Test exposure calculation by maturity and moneyness."""
        # First price the portfolio
        pricing_date = pd.Timestamp('2024-06-01')
        models = ['black_scholes']
        self.portfolio_manager.price_portfolio(pricing_date, models)

        # Calculate exposure by maturity
        exp_maturity = self.portfolio_manager.calculate_exposure_by_maturity()

        # Check exposure by maturity
        self.assertIsNotNone(exp_maturity)
        self.assertEqual(set(exp_maturity.keys()), set(models))

        for model, buckets in exp_maturity.items():
            # Check buckets
            self.assertIsInstance(buckets, dict)
            self.assertGreater(len(buckets), 0)

            # Sum of exposure should match total portfolio value
            model_results = self.portfolio_manager.portfolio_results[
                self.portfolio_manager.portfolio_results['model'] == model]
            total_value = model_results['option_value_eur'].sum()
            bucket_sum = sum(buckets.values())
            self.assertAlmostEqual(bucket_sum, total_value, places=2)

        # Calculate exposure by moneyness
        exp_moneyness = self.portfolio_manager.calculate_exposure_by_moneyness()

        # Check exposure by moneyness
        self.assertIsNotNone(exp_moneyness)
        self.assertEqual(set(exp_moneyness.keys()), set(models))

        for model, buckets in exp_moneyness.items():
            # Check buckets
            self.assertIsInstance(buckets, dict)
            self.assertGreater(len(buckets), 0)

            # Sum of exposure should match total portfolio value
            model_results = self.portfolio_manager.portfolio_results[
                self.portfolio_manager.portfolio_results['model'] == model]
            total_value = model_results['option_value_eur'].sum()
            bucket_sum = sum(buckets.values())
            self.assertAlmostEqual(bucket_sum, total_value, places=2)

    def test_model_comparison(self):
        """Test model comparison functionality."""
        # First price the portfolio with multiple models
        pricing_date = pd.Timestamp('2024-06-01')
        models = ['black_scholes', 'merton_jump', 'sabr']
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
        self.portfolio_manager.price_portfolio(
            pricing_date, models, model_params)

        # Calculate model comparison with black_scholes as reference
        comparison = self.portfolio_manager.calculate_model_comparison(
            reference_model='black_scholes')

        # Check comparison results
        self.assertIsNotNone(comparison)
        self.assertGreater(len(comparison), 0)

        # Check columns
        expected_columns = ['model', 'reference_model', 'options_count',
                            'mean_price_diff_eur', 'mean_price_diff_pct']
        for col in expected_columns:
            self.assertIn(col, comparison.columns)

        # Check reference model is correct
        self.assertEqual(comparison['reference_model'].unique()[
                         0], 'black_scholes')

        # Check models (reference model should not be in the comparison)
        self.assertEqual(set(comparison['model']), set(
            models) - {'black_scholes'})

        # Check option counts
        for _, row in comparison.iterrows():
            self.assertGreater(row['options_count'], 0)


if __name__ == '__main__':
    unittest.main()
