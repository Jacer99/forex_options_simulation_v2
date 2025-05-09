"""
Options portfolio generation module.

This module provides functionality for generating a realistic portfolio of
European options on the EUR/TND currency pair.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class OptionsGenerator:
    """
    Generates a realistic portfolio of European options on EUR/TND.

    This class handles the creation of a portfolio of options with varying strikes,
    maturities, and notional amounts that reflect typical market conditions.
    """

    def __init__(self, market_data, simulation_year=2024, max_notional=10_000_000):
        """
        Initialize the options generator.

        Parameters
        ----------
        market_data : object
            Object containing market data
        simulation_year : int
            Year to simulate options for
        max_notional : float
            Maximum total notional amount in EUR
        """
        self.market_data = market_data
        self.simulation_year = simulation_year
        self.max_notional = max_notional
        self.options_portfolio = None

    def generate_options_portfolio(self, n_options=30, seed=42):
        """
        Generate a realistic portfolio of European options.

        Parameters
        ----------
        n_options : int
            Number of options to generate
        seed : int
            Random seed for reproducibility

        Returns
        -------
        pd.DataFrame
            DataFrame with the generated options
        """
        np.random.seed(seed)

        # Filter market data for the simulation year
        year_mask = self.market_data.eur_tnd_daily['Date'].dt.year == self.simulation_year
        year_data = self.market_data.eur_tnd_daily[year_mask]

        if year_data.empty:
            raise ValueError(
                f"No market data available for the simulation year {self.simulation_year}")

        # Available dates in the simulation year
        available_dates = year_data['Date'].tolist()

        # Generate options
        options = []
        total_notional = 0

        while len(options) < n_options and total_notional < self.max_notional:
            # Select random issue date
            # Ensure some time before year end
            issue_date = np.random.choice(available_dates[:-30])

            # Determine maturity (between 5 days and 1 year)
            min_days = 5
            max_days = min(
                360, (pd.Timestamp(f"{self.simulation_year}-12-31") - issue_date).days)

            if max_days <= min_days:
                continue

            maturity_days = np.random.randint(min_days, max_days + 1)
            maturity_date = issue_date + pd.Timedelta(days=maturity_days)

            # Get spot rate at issue date
            spot_rate = year_data[year_data['Date']
                                  == issue_date]['EUR/TND'].values[0]

            # Generate strike price with skew (more OTM calls)
            # Beta distribution for asymmetric strikes
            strike_skew = np.random.beta(2, 5)
            # Strikes between 90% and 110% of spot
            strike_percent = 0.9 + 0.2 * strike_skew
            strike = round(spot_rate * strike_percent, 4)

            # Generate notional amount (between €100,000 and €1,000,000)
            notional = np.random.randint(100_000, 1_000_001)

            # Ensure we don't exceed max notional
            if total_notional + notional > self.max_notional:
                notional = self.max_notional - total_notional

            if notional < 100_000:
                break

            # Create option record
            option = {
                'OptionID': f'OPT-{self.simulation_year}-{len(options) + 1:03d}',
                'Type': 'Call',  # As per requirement, only European calls
                'Currency': 'EUR/TND',
                'IssueDate': issue_date,
                'MaturityDate': maturity_date,
                'Tenor': maturity_days,
                'SpotRate': spot_rate,
                'StrikePrice': strike,
                'NotionalEUR': notional,
                'Moneyness': strike / spot_rate
            }

            options.append(option)
            total_notional += notional

        # Create portfolio DataFrame
        self.options_portfolio = pd.DataFrame(options)

        return self.options_portfolio

    def save_options(self, output_dir='data'):
        """
        Save the generated options portfolio to a CSV file.

        Parameters
        ----------
        output_dir : str
            Directory to save the CSV file
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.options_portfolio is not None:
            self.options_portfolio.to_csv(os.path.join(
                output_dir, 'options_portfolio.csv'), index=False)

    def load_options(self, input_dir='data'):
        try:
            self.options_portfolio = pd.read_csv(
                os.path.join(input_dir, 'options_portfolio.csv'))
            self.options_portfolio['IssueDate'] = pd.to_datetime(
                self.options_portfolio['IssueDate'], dayfirst=True)
            self.options_portfolio['MaturityDate'] = pd.to_datetime(
                self.options_portfolio['MaturityDate'], dayfirst=True)
            return True
        except Exception as e:
            print(f"Error loading options portfolio file: {e}")
        return False

    def check_or_generate_options(self, output_dir='data'):
        """
        Check if options portfolio file exists, if not generate it.

        Parameters
        ----------
        output_dir : str
            Directory for data files

        Returns
        -------
        bool
            True if options data is available (loaded or generated), False otherwise
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        options_path = os.path.join(output_dir, 'options_portfolio.csv')

        if os.path.exists(options_path):
            return self.load_options(output_dir)
        else:
            self.generate_options_portfolio()
            self.save_options(output_dir)
            return True

    def plot_options_distribution(self, output_dir='plots'):
        """
        Plot various distributions of the options portfolio.

        Parameters
        ----------
        output_dir : str
            Directory to save plots
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.options_portfolio is None:
            raise ValueError(
                "No options portfolio available. Generate or load options first.")

        # Plot maturity distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.options_portfolio['Tenor'], bins=20)
        plt.title('Distribution of Option Maturities')
        plt.xlabel('Tenor (days)')
        plt.ylabel('Count')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'option_maturities.png'))
        plt.close()

        # Plot moneyness distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.options_portfolio['Moneyness'], bins=20)
        plt.title('Distribution of Option Moneyness')
        plt.xlabel('Moneyness (Strike/Spot)')
        plt.ylabel('Count')
        plt.axvline(x=1, color='red', linestyle='--', label='At-the-money')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'option_moneyness.png'))
        plt.close()

        # Plot notional distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.options_portfolio['NotionalEUR'], bins=20)
        plt.title('Distribution of Option Notional Amounts')
        plt.xlabel('Notional (EUR)')
        plt.ylabel('Count')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'option_notionals.png'))
        plt.close()

        # Plot issue dates
        plt.figure(figsize=(10, 6))
        sns.histplot(self.options_portfolio['IssueDate'], bins=12)
        plt.title('Distribution of Option Issue Dates')
        plt.xlabel('Issue Date')
        plt.ylabel('Count')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'option_issue_dates.png'))
        plt.close()

        # Plot cumulative notional
        sorted_portfolio = self.options_portfolio.sort_values('IssueDate')
        cum_notional = sorted_portfolio['NotionalEUR'].cumsum()

        plt.figure(figsize=(10, 6))
        plt.plot(sorted_portfolio['IssueDate'], cum_notional)
        plt.title('Cumulative Notional Amount')
        plt.xlabel('Issue Date')
        plt.ylabel('Cumulative Notional (EUR)')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'cumulative_notional.png'))
        plt.close()
