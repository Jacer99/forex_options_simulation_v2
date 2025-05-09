"""
Market data handling module for the forex options pricing system.

This module provides functionality for loading, generating, and handling market data
for the EUR/TND currency pair and related interest rates.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


class MarketDataGenerator:
    """
    Generates and manages market data for EUR/TND forex pair and interest rates.

    This class handles the creation of realistic market data, including exchange rates
    with appropriate volatility patterns and interest rates for both currencies.
    """

    def __init__(self, start_date='2023-01-01', end_date='2024-12-31',
                 base_eur_tnd_rate=3.35, eur_rate_mean=0.03, tnd_rate_mean=0.08):
        """
        Initialize the market data generator with parameters.

        Parameters
        ----------
        start_date : str
            Start date for data generation in format 'YYYY-MM-DD'
        end_date : str
            End date for data generation in format 'YYYY-MM-DD'
        base_eur_tnd_rate : float
            Base rate for EUR/TND
        eur_rate_mean : float
            Mean interest rate for EUR
        tnd_rate_mean : float
            Mean interest rate for TND
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.base_eur_tnd_rate = base_eur_tnd_rate
        self.eur_rate_mean = eur_rate_mean
        self.tnd_rate_mean = tnd_rate_mean

        # Generate date range for daily and monthly data
        self.daily_dates = pd.date_range(
            start=self.start_date, end=self.end_date, freq='D')
        self.monthly_dates = pd.date_range(
            start=self.start_date, end=self.end_date, freq='MS')

        # Placeholder for generated data
        self.eur_tnd_daily = None
        self.eur_rates_monthly = None
        self.tnd_rates_monthly = None

    def generate_eur_tnd_daily(self, volatility=0.08, seed=42):
        """
        Generate daily EUR/TND exchange rates with realistic patterns.

        Parameters
        ----------
        volatility : float
            Annualized volatility for the exchange rate
        seed : int
            Random seed for reproducibility

        Returns
        -------
        pd.DataFrame
            DataFrame with daily EUR/TND rates
        """
        np.random.seed(seed)

        # Number of trading days
        n_days = len(self.daily_dates)

        # Daily volatility
        daily_vol = volatility / np.sqrt(252)

        # Generate log returns with some autocorrelation and mean reversion
        z = np.random.normal(0, 1, n_days)
        log_returns = np.zeros(n_days)

        # Add some autocorrelation, mean reversion and mild seasonality
        mean_reversion = 0.03
        autocorr = 0.2

        for i in range(1, n_days):
            month = self.daily_dates[i].month
            # Add mild seasonality effect (summer euros influx for tourism)
            seasonal_effect = 0.0002 * np.sin(2 * np.pi * (month - 6) / 12)
            log_returns[i] = seasonal_effect + mean_reversion * (0 - log_returns[i-1]) + \
                autocorr * log_returns[i-1] + daily_vol * z[i]

        # Add occasional small jumps (political/economic events)
        n_jumps = int(n_days * 0.02)  # 2% of days have jumps
        jump_days = np.random.choice(range(n_days), n_jumps, replace=False)
        jump_sizes = np.random.normal(0, daily_vol * 3, n_jumps)
        for i, day in enumerate(jump_days):
            log_returns[day] += jump_sizes[i]

        # Convert log returns to prices
        prices = np.zeros(n_days)
        prices[0] = self.base_eur_tnd_rate
        for i in range(1, n_days):
            prices[i] = prices[i-1] * np.exp(log_returns[i])

        # Create DataFrame
        self.eur_tnd_daily = pd.DataFrame({
            'Date': self.daily_dates,
            'EUR/TND': prices,
            'Return': log_returns
        })

        # Calculate realized volatility over rolling windows
        window_sizes = [5, 10, 21, 63]
        for window in window_sizes:
            vol_col = f'{window}d_Volatility'
            self.eur_tnd_daily[vol_col] = self.eur_tnd_daily['Return'].rolling(
                window=window).std() * np.sqrt(252)

        return self.eur_tnd_daily

    def generate_interest_rates(self, eur_vol=0.012, tnd_vol=0.025, seed=42):
        """
        Generate monthly interest rates for EUR and TND with realistic patterns.

        Parameters
        ----------
        eur_vol : float
            Volatility of EUR interest rates
        tnd_vol : float
            Volatility of TND interest rates
        seed : int
            Random seed for reproducibility

        Returns
        -------
        tuple
            (eur_rates_df, tnd_rates_df) - DataFrames with monthly interest rates
        """
        np.random.seed(seed)

        n_months = len(self.monthly_dates)

        # Generate EUR rates with mean reversion
        eur_rates = np.zeros(n_months)
        eur_rates[0] = self.eur_rate_mean

        # Generate TND rates with mean reversion and correlation with EUR
        tnd_rates = np.zeros(n_months)
        tnd_rates[0] = self.tnd_rate_mean

        # Parameters
        eur_mean_reversion = 0.1
        tnd_mean_reversion = 0.05
        correlation = 0.4

        for i in range(1, n_months):
            # Correlated normal random variables
            z1 = np.random.normal(0, 1)
            z2 = correlation * z1 + \
                np.sqrt(1 - correlation**2) * np.random.normal(0, 1)

            # Mean reversion model for EUR rates
            eur_rates[i] = eur_rates[i-1] + eur_mean_reversion * (self.eur_rate_mean - eur_rates[i-1]) + \
                eur_vol * z1

            # Mean reversion model for TND rates
            tnd_rates[i] = tnd_rates[i-1] + tnd_mean_reversion * (self.tnd_rate_mean - tnd_rates[i-1]) + \
                tnd_vol * z2

            # Ensure TND rates remain higher than EUR rates
            tnd_rates[i] = max(tnd_rates[i], eur_rates[i] + 0.02)

            # Ensure rates don't go negative (for EUR) or too low (for TND)
            eur_rates[i] = max(eur_rates[i], 0.001)
            tnd_rates[i] = max(tnd_rates[i], 0.02)

        # Create DataFrames
        self.eur_rates_monthly = pd.DataFrame({
            'Date': self.monthly_dates,
            'EUR_Rate': eur_rates
        })

        self.tnd_rates_monthly = pd.DataFrame({
            'Date': self.monthly_dates,
            'TND_Rate': tnd_rates
        })

        return self.eur_rates_monthly, self.tnd_rates_monthly

    def save_data(self, output_dir='data'):
        """
        Save the generated data to CSV files.

        Parameters
        ----------
        output_dir : str
            Directory to save the CSV files
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.eur_tnd_daily is not None:
            self.eur_tnd_daily.to_csv(os.path.join(
                output_dir, 'eur_tnd_daily_rates.csv'), index=False)

        if self.eur_rates_monthly is not None:
            self.eur_rates_monthly.to_csv(os.path.join(
                output_dir, 'eur_interest_rates_monthly.csv'), index=False)

        if self.tnd_rates_monthly is not None:
            self.tnd_rates_monthly.to_csv(os.path.join(
                output_dir, 'tnd_interest_rates_monthly.csv'), index=False)

    def load_data(self, input_dir='data'):
        """
        Load market data from CSV files.

        Parameters
        ----------
        input_dir : str
            Directory to load data from

        Returns
        -------
        bool
            True if data was loaded successfully, False otherwise
        """
        try:
            self.eur_tnd_daily = pd.read_csv(
                os.path.join(input_dir, 'eur_tnd_daily_rates.csv'))
            self.eur_tnd_daily['Date'] = pd.to_datetime(
                self.eur_tnd_daily['Date'], dayfirst=True)

            self.eur_rates_monthly = pd.read_csv(os.path.join(
                input_dir, 'eur_interest_rates_monthly.csv'))
            self.eur_rates_monthly['Date'] = pd.to_datetime(
                self.eur_rates_monthly['Date'], dayfirst=True)

            self.tnd_rates_monthly = pd.read_csv(os.path.join(
                input_dir, 'tnd_interest_rates_monthly.csv'))
            self.tnd_rates_monthly['Date'] = pd.to_datetime(
                self.tnd_rates_monthly['Date'], dayfirst=True)

            # Check and fix data issues
            if self.check_and_fix_data():
                print("Fixed issues in the loaded data.")
                # Re-save the fixed data
                self.save_data(input_dir)

            return True
        except Exception as e:
            print(f"Error loading data files: {e}")
        return False

    def check_or_generate_data(self, output_dir='data'):
        """
        Check if data files exist, if not generate them.

        Parameters
        ----------
        output_dir : str
            Directory for data files

        Returns
        -------
        bool
            True if data is available (loaded or generated), False otherwise
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        eur_tnd_path = os.path.join(output_dir, 'eur_tnd_daily_rates.csv')
        eur_rates_path = os.path.join(
            output_dir, 'eur_interest_rates_monthly.csv')
        tnd_rates_path = os.path.join(
            output_dir, 'tnd_interest_rates_monthly.csv')

        if os.path.exists(eur_tnd_path) and os.path.exists(eur_rates_path) and os.path.exists(tnd_rates_path):
            load_success = self.load_data(output_dir)
            if load_success:
                return True
            print("Failed to load existing data files. Regenerating...")

        # Generate new data
        print("Generating new market data...")
        self.generate_eur_tnd_daily()
        self.generate_interest_rates()

        # Ensure 'Return' column and volatility exists
        if 'Return' not in self.eur_tnd_daily.columns:
            self.eur_tnd_daily['Return'] = self.eur_tnd_daily['EUR/TND'].pct_change().fillna(0)

        # Calculate volatility
        for window in [5, 21, 63]:
            vol_col = f'{window}d_Volatility'
            self.eur_tnd_daily[vol_col] = self.eur_tnd_daily['Return'].rolling(
                window=window).std() * np.sqrt(252)

        self.save_data(output_dir)
        return True

    # Modifications to src/forex_options/market_data.py

    def calculate_historical_volatility(self, date, window_size=21):
        """
        Calculate historical volatility for a specific date.

        Parameters
        ----------
        date : datetime
            Date for which to calculate volatility
        window_size : int
            Window size in days for historical volatility calculation

        Returns
        -------
        float
            Annualized historical volatility
        """
        if self.eur_tnd_daily is None:
            raise ValueError(
                "No market data available. Load or generate data first.")

        # Find data up to the specified date
        mask = self.eur_tnd_daily['Date'] <= date
        data = self.eur_tnd_daily[mask].tail(window_size)

        if len(data) < window_size:
            print(
                f"Warning: Not enough data for {window_size}-day volatility. Using available data ({len(data)} days).")
            if len(data) == 0:
                return 0.15  # Default volatility if no data available

        # Ensure 'Return' column exists in data
        if 'Return' not in data.columns:
            print(f"Warning: 'Return' column not found. Calculating returns now.")
            # Calculate returns if not present
            if len(data) > 1:
                pct_change = data['EUR/TND'].pct_change().fillna(0)
                # Create a copy to avoid SettingWithCopyWarning
                data_copy = data.copy()
                data_copy['Return'] = pct_change
                data = data_copy
            else:
                # Default volatility for single data point
                return 0.15

        # Calculate annualized volatility
        return data['Return'].std() * np.sqrt(252)

    def get_market_data(self, date, volatility_window=21):
        """
        Get all market data for a specific date.

        Parameters
        ----------
        date : datetime
            Date for which to get market data
        volatility_window : int
            Window size for historical volatility calculation

        Returns
        -------
        dict
            Dictionary with all market data for the date
        """
        try:
            # Get spot rate
            try:
                spot_rate = self.get_spot_rate(date)
                if spot_rate <= 0:
                    raise ValueError(f"Invalid spot rate: {spot_rate}")
            except Exception as e:
                print(f"Error getting spot rate for {date}: {e}")
                raise ValueError(f"Invalid spot rate for {date}")

            # Get interest rates
            try:
                eur_rate, tnd_rate = self.get_interest_rates(date)
                # Validate interest rates (ensure they're not 0 or NaN)
                if eur_rate <= 0 or tnd_rate <= 0 or pd.isna(eur_rate) or pd.isna(tnd_rate):
                    raise ValueError(
                        f"Invalid interest rates: EUR={eur_rate}, TND={tnd_rate}")
            except Exception as e:
                print(f"Error getting interest rates for {date}: {e}")
                raise ValueError(f"Invalid interest rates for {date}")

            # Calculate historical volatility
            try:
                volatility = self.calculate_historical_volatility(
                    date, volatility_window)
                if pd.isna(volatility) or volatility <= 0:
                    volatility = 0.15  # Default volatility if calculation fails
                    print(
                        f"Warning: Using default volatility (15%) for {date}")
            except Exception as e:
                volatility = 0.15  # Default volatility if calculation fails
                print(
                    f"Warning: Using default volatility (15%) for {date}: {e}")

            return {
                'date': date,
                'spot_rate': spot_rate,
                'eur_rate': eur_rate,
                'tnd_rate': tnd_rate,
                'volatility': volatility
            }
        except Exception as e:
            raise ValueError(f"Error getting market data for {date}: {e}")

    def check_and_fix_data(self):
        fixed = False

    # Check exchange rate data
        if self.eur_tnd_daily is not None:
            # Add Return column if missing
            if 'Return' not in self.eur_tnd_daily.columns:
                print("Adding 'Return' column to exchange rate data...")
                self.eur_tnd_daily['Return'] = self.eur_tnd_daily['EUR/TND'].pct_change().fillna(0)
                fixed = True

        # Calculate volatility columns if missing
            for window in [5, 21, 63]:
                vol_col = f'{window}d_Volatility'
                if vol_col not in self.eur_tnd_daily.columns:
                    print(
                        f"Adding '{vol_col}' column to exchange rate data...")
                    self.eur_tnd_daily[vol_col] = self.eur_tnd_daily['Return'].rolling(
                        window=window).std() * np.sqrt(252)
                    fixed = True

    # Check interest rate data
        if self.eur_rates_monthly is not None:
            # Remove rows with invalid rates (0 or NaN)
            if 'EUR_Rate' in self.eur_rates_monthly.columns:
                invalid_mask = (self.eur_rates_monthly['EUR_Rate'] <= 0) | pd.isna(
                    self.eur_rates_monthly['EUR_Rate'])
                if invalid_mask.any():
                    print(
                        f"Removing {invalid_mask.sum()} rows with invalid EUR interest rates...")
                    self.eur_rates_monthly = self.eur_rates_monthly[~invalid_mask].reset_index(
                        drop=True)
                    fixed = True

        if self.tnd_rates_monthly is not None:
            # Remove rows with invalid rates (0 or NaN)
            if 'TND_Rate' in self.tnd_rates_monthly.columns:
                invalid_mask = (self.tnd_rates_monthly['TND_Rate'] <= 0) | pd.isna(
                    self.tnd_rates_monthly['TND_Rate'])
                if invalid_mask.any():
                    print(
                        f"Removing {invalid_mask.sum()} rows with invalid TND interest rates...")
                    self.tnd_rates_monthly = self.tnd_rates_monthly[~invalid_mask].reset_index(
                        drop=True)
                    fixed = True

        return fixed

    def get_interest_rates(self, date):
        """
        Get interest rates for a specific date.

        Parameters
        ----------
        date : datetime
            Date for which to get interest rates

        Returns
        -------
        tuple
            (eur_rate, tnd_rate) - Interest rates for the date
        """
        if self.eur_rates_monthly is None or self.tnd_rates_monthly is None:
            raise ValueError(
                "No interest rate data available. Load or generate data first.")

        # Convert to the start of the month
        month_start = pd.Timestamp(date.year, date.month, 1)

        # Find the closest previous month in data
        eur_mask = self.eur_rates_monthly['Date'] <= month_start
        tnd_mask = self.tnd_rates_monthly['Date'] <= month_start

        if not eur_mask.any() or not tnd_mask.any():
            raise ValueError(
                f"No interest rate data available for or before {date}")

        eur_rate = self.eur_rates_monthly[eur_mask]['EUR_Rate'].iloc[-1]
        tnd_rate = self.tnd_rates_monthly[tnd_mask]['TND_Rate'].iloc[-1]

        return eur_rate, tnd_rate

    def get_spot_rate(self, date):
        """
        Get spot exchange rate for a specific date.

        Parameters
        ----------
        date : datetime
            Date for which to get spot rate

        Returns
        -------
        float
            EUR/TND spot rate for the date
        """
        if self.eur_tnd_daily is None:
            raise ValueError(
                "No exchange rate data available. Load or generate data first.")

        # Find the closest previous date in data
        mask = self.eur_tnd_daily['Date'] <= date

        if not mask.any():
            raise ValueError(
                f"No exchange rate data available for or before {date}")

        return self.eur_tnd_daily[mask]['EUR/TND'].iloc[-1]

    def get_market_data(self, date, volatility_window=21):
        """
        Get all market data for a specific date.

        Parameters
        ----------
        date : datetime
            Date for which to get market data
        volatility_window : int
            Window size for historical volatility calculation

        Returns
        -------
        dict
            Dictionary with all market data for the date
        """
        try:
            # Get spot rate
            spot_rate = self.get_spot_rate(date)

            # Get interest rates
            eur_rate, tnd_rate = self.get_interest_rates(date)

            # Calculate historical volatility
            volatility = self.calculate_historical_volatility(
                date, volatility_window)

            return {
                'date': date,
                'spot_rate': spot_rate,
                'eur_rate': eur_rate,
                'tnd_rate': tnd_rate,
                'volatility': volatility
            }
        except Exception as e:
            raise ValueError(f"Error getting market data for {date}: {e}")

    def plot_market_data(self, output_dir='plots'):
        """
        Plot the market data for visualization.

        Parameters
        ----------
        output_dir : str
            Directory to save plots
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.eur_tnd_daily is not None:
            # Plot exchange rates
            plt.figure(figsize=(12, 6))
            plt.plot(self.eur_tnd_daily['Date'], self.eur_tnd_daily['EUR/TND'])
            plt.title('EUR/TND Exchange Rate')
            plt.xlabel('Date')
            plt.ylabel('Rate')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'eur_tnd_rate.png'))
            plt.close()

            # Plot volatility
            plt.figure(figsize=(12, 6))
            for window in [5, 21, 63]:
                vol_col = f'{window}d_Volatility'
                plt.plot(
                    self.eur_tnd_daily['Date'], self.eur_tnd_daily[vol_col], label=f'{window}-day volatility')
            plt.title('EUR/TND Realized Volatility')
            plt.xlabel('Date')
            plt.ylabel('Annualized Volatility')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'eur_tnd_volatility.png'))
            plt.close()

        if self.eur_rates_monthly is not None and self.tnd_rates_monthly is not None:
            # Plot interest rates
            plt.figure(figsize=(12, 6))
            plt.plot(
                self.eur_rates_monthly['Date'], self.eur_rates_monthly['EUR_Rate'], label='EUR Interest Rate')
            plt.plot(
                self.tnd_rates_monthly['Date'], self.tnd_rates_monthly['TND_Rate'], label='TND Interest Rate')
            plt.title('Interest Rates')
            plt.xlabel('Date')
            plt.ylabel('Rate')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'interest_rates.png'))
            plt.close()

            # Plot interest rate differential
            dates = self.eur_rates_monthly['Date']
            differential = self.tnd_rates_monthly['TND_Rate'].values - \
                self.eur_rates_monthly['EUR_Rate'].values

            plt.figure(figsize=(12, 6))
            plt.plot(dates, differential)
            plt.title('TND-EUR Interest Rate Differential')
            plt.xlabel('Date')
            plt.ylabel('Rate Differential')
            plt.grid(True)
            plt.savefig(os.path.join(
                output_dir, 'interest_rate_differential.png'))
            plt.close()
