"""
Unit tests for the market data module.

This module tests the market data generation and handling functionality.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from forex_options.market_data import MarketDataGenerator


class TestMarketDataGenerator(unittest.TestCase):
    """Test cases for the MarketDataGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.start_date = '2023-01-01'
        self.end_date = '2024-12-31'
        self.base_eur_tnd_rate = 3.35
        self.eur_rate_mean = 0.03
        self.tnd_rate_mean = 0.08

        # Initialize market data generator
        self.market_data = MarketDataGenerator(
            start_date=self.start_date,
            end_date=self.end_date,
            base_eur_tnd_rate=self.base_eur_tnd_rate,
            eur_rate_mean=self.eur_rate_mean,
            tnd_rate_mean=self.tnd_rate_mean
        )

    def test_exchange_rate_generation(self):
        """Test exchange rate time series generation."""
        # Generate data
        eur_tnd_daily = self.market_data.generate_eur_tnd_daily(seed=42)

        # Check basic properties
        self.assertIsNotNone(eur_tnd_daily)
        self.assertGreater(len(eur_tnd_daily), 0)

        # Check columns
        self.assertIn('Date', eur_tnd_daily.columns)
        self.assertIn('EUR/TND', eur_tnd_daily.columns)
        self.assertIn('Return', eur_tnd_daily.columns)

        # Check first value
        self.assertEqual(
            eur_tnd_daily['EUR/TND'].iloc[0], self.base_eur_tnd_rate)

        # Check volatility calculation
        self.assertIn('21d_Volatility', eur_tnd_daily.columns)

        # Check for non-negative rates
        self.assertTrue((eur_tnd_daily['EUR/TND'] > 0).all())

    def test_interest_rate_generation(self):
        """Test interest rate time series generation."""
        # Generate data
        eur_rates, tnd_rates = self.market_data.generate_interest_rates(
            seed=42)

        # Check basic properties
        self.assertIsNotNone(eur_rates)
        self.assertIsNotNone(tnd_rates)
        self.assertGreater(len(eur_rates), 0)
        self.assertGreater(len(tnd_rates), 0)

        # Check columns
        self.assertIn('Date', eur_rates.columns)
        self.assertIn('EUR_Rate', eur_rates.columns)
        self.assertIn('Date', tnd_rates.columns)
        self.assertIn('TND_Rate', tnd_rates.columns)

        # Check first values
        self.assertEqual(eur_rates['EUR_Rate'].iloc[0], self.eur_rate_mean)
        self.assertEqual(tnd_rates['TND_Rate'].iloc[0], self.tnd_rate_mean)

        # Check for non-negative rates
        self.assertTrue((eur_rates['EUR_Rate'] > 0).all())
        self.assertTrue((tnd_rates['TND_Rate'] > 0).all())

        # Check TND rates are higher than EUR rates (structural feature of the economy)
        dates = pd.merge(eur_rates, tnd_rates, on='Date')['Date']
        eur_subset = eur_rates[eur_rates['Date'].isin(dates)]
        tnd_subset = tnd_rates[tnd_rates['Date'].isin(dates)]

        for i in range(len(dates)):
            self.assertGreater(
                tnd_subset['TND_Rate'].iloc[i], eur_subset['EUR_Rate'].iloc[i])

    def test_get_market_data(self):
        """Test retrieving market data for a specific date."""
        # Generate data
        self.market_data.generate_eur_tnd_daily(seed=42)
        self.market_data.generate_interest_rates(seed=42)

        # Get data for a specific date
        test_date = pd.Timestamp('2024-06-01')
        market_info = self.market_data.get_market_data(test_date)

        # Check returned data
        self.assertIsNotNone(market_info)
        self.assertEqual(market_info['date'], test_date)
        self.assertIn('spot_rate', market_info)
        self.assertIn('eur_rate', market_info)
        self.assertIn('tnd_rate', market_info)
        self.assertIn('volatility', market_info)

        # Check types
        self.assertIsInstance(market_info['spot_rate'], float)
        self.assertIsInstance(market_info['eur_rate'], float)
        self.assertIsInstance(market_info['tnd_rate'], float)
        self.assertIsInstance(market_info['volatility'], float)

        # Check values are reasonable
        self.assertGreater(market_info['spot_rate'], 0)
        self.assertGreater(market_info['eur_rate'], 0)
        self.assertGreater(market_info['tnd_rate'], 0)
        self.assertGreater(market_info['volatility'], 0)

        # Check TND rate > EUR rate
        self.assertGreater(market_info['tnd_rate'], market_info['eur_rate'])

    def test_handling_nonexistent_date(self):
        """Test handling of non-existent dates in the data."""
        # Generate data
        self.market_data.generate_eur_tnd_daily(seed=42)
        self.market_data.generate_interest_rates(seed=42)

        # Get data for a date before the start date
        with self.assertRaises(ValueError):
            self.market_data.get_market_data(pd.Timestamp('2022-12-01'))

    def test_historical_volatility_calculation(self):
        """Test historical volatility calculation."""
        # Generate data
        self.market_data.generate_eur_tnd_daily(seed=42)

        # Calculate volatility for a date
        test_date = pd.Timestamp('2024-06-01')
        volatility = self.market_data.calculate_historical_volatility(
            test_date, window_size=21)

        # Check volatility
        self.assertIsNotNone(volatility)
        self.assertIsInstance(volatility, float)
        self.assertGreater(volatility, 0)
        self.assertLess(volatility, 1)  # Volatility should be reasonable

        # Compare with different window sizes
        vol_10 = self.market_data.calculate_historical_volatility(
            test_date, window_size=10)
        vol_63 = self.market_data.calculate_historical_volatility(
            test_date, window_size=63)

        # Both should give reasonable values
        self.assertGreater(vol_10, 0)
        self.assertGreater(vol_63, 0)

        # Shorter window size typically gives more volatile estimates
        # but this is not guaranteed, so we don't assert it


if __name__ == '__main__':
    unittest.main()
