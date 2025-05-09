# Forex Options Pricing and Risk Management System for EUR/TND

A comprehensive system for pricing, analyzing, and managing a portfolio of European options on the EUR/TND currency pair, tailored for the Tunisian market.

## Features

- **Multiple Pricing Models**
  - Black-Scholes (Garman-Kohlhagen) for FX options
  - Merton Jump-Diffusion with analytical and Monte Carlo approaches
  - SABR Stochastic Volatility Model for handling volatility smile/skew

- **Market Data Handling**
  - Generation of realistic EUR/TND exchange rates with appropriate properties
  - Interest rate simulation for both currencies
  - Historical volatility calculation

- **Options Portfolio Management**
  - Generation of realistic option portfolios
  - Portfolio-level risk aggregation (Delta, Gamma, Vega, Theta)
  - Exposure analysis by maturity and moneyness

- **Model Evaluation Framework**
  - Backtest capability over specified time periods
  - Performance metrics calculation and comparison
  - PnL attribution analysis

- **Visualization System**
  - Market data visualization
  - Portfolio exposure and risk visualizations
  - Volatility smile/surface plots
  - Model comparison charts

## Project Structure

```
forex_options_simulation/
├── data/                     # Data files
├── src/                      # Main source code
│   ├── forex_options/        # Core forex options simulation module
│   ├── tests/                # Unit and integration tests
│   └── main.py               # Entry point for CLI or demo
├── notebooks/                # Jupyter notebooks for analysis
├── configs/                  # Configuration files
├── results/                  # Output results
├── plots/                    # Generated plots
├── requirements.txt          # Python dependencies
└── README.md                 # Project overview, setup, usage
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/forex_options_simulation.git
cd forex_options_simulation
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the simulation

```bash
python src/main.py
```

### Configuration

The simulation parameters can be configured in `configs/simulation_config.yaml`.

### Jupyter Notebook

For interactive analysis, you can use the provided Jupyter notebooks:

```bash
jupyter notebook notebooks/forex_options_analysis.ipynb
```

## Market Context

This system is specifically designed for the EUR/TND currency pair, taking into account:

- The Tunisian Dinar's semi-managed float regime
- Structural characteristics including higher interest rates in Tunisia
- Asymmetric volatility patterns typical in emerging market currencies
- Potential for market jumps from political/economic events

## License

[MIT License](LICENSE)