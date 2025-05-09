"""
Pricing models for forex options.

This module implements various pricing models for European options on the EUR/TND 
currency pair, including Black-Scholes (Garman-Kohlhagen), Merton Jump-Diffusion, 
and SABR stochastic volatility models.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import numba as nb
from dataclasses import dataclass


@dataclass
class PricingResult:
    """Data class for storing option pricing results."""
    price: float
    delta: float = None
    gamma: float = None
    vega: float = None
    theta: float = None
    rho_d: float = None  # Domestic interest rate sensitivity
    rho_f: float = None  # Foreign interest rate sensitivity
    error: float = None  # For Monte Carlo methods


class BlackScholesFX:
    """
    Garman-Kohlhagen model for pricing forex options.

    This is an extension of the Black-Scholes model for foreign exchange options.
    """

    @staticmethod
    def price(S, K, T, r_d, r_f, sigma, option_type='call'):
        """
        Calculate option price and Greeks using the Garman-Kohlhagen model.

        Parameters
        ----------
        S : float
            Spot exchange rate (price of one unit of foreign currency in domestic currency)
        K : float
            Strike price
        T : float
            Time to maturity in years
        r_d : float
            Domestic risk-free interest rate (EUR)
        r_f : float
            Foreign risk-free interest rate (TND)
        sigma : float
            Volatility of the exchange rate
        option_type : str
            'call' or 'put'

        Returns
        -------
        PricingResult
            Object containing option price and Greeks
        """
        # Handle edge cases
        if T <= 0:
            if option_type.lower() == 'call':
                price = max(0, S - K)
                delta = 1.0 if S > K else 0.0
            else:
                price = max(0, K - S)
                delta = -1.0 if S < K else 0.0

            return PricingResult(
                price=price,
                delta=delta,
                gamma=0.0,
                vega=0.0,
                theta=0.0,
                rho_d=0.0,
                rho_f=0.0
            )

        # Calculate d1 and d2
        d1 = (np.log(S/K) + (r_d - r_f + 0.5 * sigma**2) * T) / \
            (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Option price calculation
        if option_type.lower() == 'call':
            price = S * np.exp(-r_f * T) * norm.cdf(d1) - \
                K * np.exp(-r_d * T) * norm.cdf(d2)

            # Greeks for call
            delta = np.exp(-r_f * T) * norm.cdf(d1)
            rho_d = K * T * np.exp(-r_d * T) * norm.cdf(d2)
            rho_f = -S * T * np.exp(-r_f * T) * norm.cdf(d1)
        else:  # put
            price = K * np.exp(-r_d * T) * norm.cdf(-d2) - \
                S * np.exp(-r_f * T) * norm.cdf(-d1)

            # Greeks for put
            delta = -np.exp(-r_f * T) * norm.cdf(-d1)
            rho_d = -K * T * np.exp(-r_d * T) * norm.cdf(-d2)
            rho_f = S * T * np.exp(-r_f * T) * norm.cdf(-d1)

        # Common Greeks
        gamma = np.exp(-r_f * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * np.exp(-r_f * T) * norm.pdf(d1) * np.sqrt(T)
        theta = -(S * sigma * np.exp(-r_f * T) * norm.pdf(d1)) / (2 * np.sqrt(T)) - \
            r_d * K * np.exp(-r_d * T) * norm.cdf(d2) + \
            r_f * S * np.exp(-r_f * T) * norm.cdf(d1)

        return PricingResult(
            price=price,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho_d=rho_d,
            rho_f=rho_f
        )


class MertonJumpDiffusion:
    """
    Merton Jump Diffusion model for pricing forex options.

    This model extends the Black-Scholes model by adding a jump component
    to account for sudden market movements.
    """

    @staticmethod
    def price(S, K, T, r_d, r_f, sigma, lam, mu_j, sigma_j, option_type='call', n_terms=20):
        """
        Calculate option price and Greeks using the Merton Jump Diffusion model.

        Parameters
        ----------
        S : float
            Spot exchange rate
        K : float
            Strike price
        T : float
            Time to maturity in years
        r_d : float
            Domestic risk-free interest rate
        r_f : float
            Foreign risk-free interest rate
        sigma : float
            Diffusion volatility
        lam : float
            Jump intensity (average number of jumps per year)
        mu_j : float
            Average jump size (in logarithmic terms)
        sigma_j : float
            Jump size volatility
        option_type : str
            'call' or 'put'
        n_terms : int
            Number of terms to include in the series approximation

        Returns
        -------
        PricingResult
            Object containing option price and Greeks
        """
        # Handle edge cases
        if T <= 0:
            if option_type.lower() == 'call':
                price = max(0, S - K)
                delta = 1.0 if S > K else 0.0
            else:
                price = max(0, K - S)
                delta = -1.0 if S < K else 0.0

            return PricingResult(
                price=price,
                delta=delta,
                gamma=0.0,
                vega=0.0,
                theta=0.0,
                rho_d=0.0,
                rho_f=0.0
            )

        # Expected jump size factor
        k = np.exp(mu_j + 0.5 * sigma_j**2) - 1

        # Adjusted parameters for the Merton model
        lam_prime = lam * (1 + k)
        r_prime = r_d - r_f - lam * k

        # Initialize price and greeks
        price = 0
        delta = 0
        gamma = 0
        vega = 0
        theta = 0
        rho_d = 0
        rho_f = 0

        # Sum over Poisson probabilities
        for n in range(n_terms):
            # Probability of n jumps during time T
            poisson_prob = np.exp(-lam * T) * \
                (lam * T)**n / np.math.factorial(n)

            # Adjusted volatility with jumps
            sigma_n = np.sqrt(sigma**2 + n * sigma_j**2 / T)

            # Black-Scholes price with adjusted volatility
            bs = BlackScholesFX.price(
                S, K, T, r_prime + lam_prime, r_f, sigma_n, option_type
            )

            # Accumulate price and Greeks
            price += poisson_prob * bs.price
            delta += poisson_prob * bs.delta
            gamma += poisson_prob * bs.gamma
            vega += poisson_prob * bs.vega
            theta += poisson_prob * bs.theta
            rho_d += poisson_prob * bs.rho_d
            rho_f += poisson_prob * bs.rho_f

        return PricingResult(
            price=price,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho_d=rho_d,
            rho_f=rho_f
        )

    @staticmethod
    @nb.njit
    def _simulate_path(S, T, r_d, r_f, sigma, lam, mu_j, sigma_j, n_steps, seed):
        """
        Simulate a single price path using the Merton Jump Diffusion model.

        Parameters
        ----------
        S : float
            Spot exchange rate
        T : float
            Time to maturity in years
        r_d : float
            Domestic risk-free interest rate
        r_f : float
            Foreign risk-free interest rate
        sigma : float
            Diffusion volatility
        lam : float
            Jump intensity (average number of jumps per year)
        mu_j : float
            Average jump size (in logarithmic terms)
        sigma_j : float
            Jump size volatility
        n_steps : int
            Number of time steps in the simulation
        seed : int
            Random seed

        Returns
        -------
        float
            Final price at maturity
        """
        # Set random seed
        np.random.seed(seed)

        # Time step
        dt = T / n_steps

        # Expected jump size
        k = np.exp(mu_j + 0.5 * sigma_j**2) - 1

        # Drift adjustment for risk-neutrality
        drift = r_d - r_f - 0.5 * sigma**2 - lam * k

        # Initialize price
        price = S

        # Simulate path
        for _ in range(n_steps):
            # Diffusion component
            z = np.random.normal(0, 1)
            diffusion = drift * dt + sigma * np.sqrt(dt) * z

            # Jump component
            n_jumps = np.random.poisson(lam * dt)
            jumps = 0

            if n_jumps > 0:
                for _ in range(n_jumps):
                    jump_size = np.random.normal(mu_j, sigma_j)
                    jumps += jump_size

            # Update price
            price *= np.exp(diffusion + jumps)

        return price

    @staticmethod
    def price_mc(S, K, T, r_d, r_f, sigma, lam, mu_j, sigma_j,
                 option_type='call', n_steps=100, n_sims=10000, seed=42):
        """
        Calculate option price using Monte Carlo simulation for the Merton Jump Diffusion model.

        Parameters
        ----------
        S : float
            Spot exchange rate
        K : float
            Strike price
        T : float
            Time to maturity in years
        r_d : float
            Domestic risk-free interest rate
        r_f : float
            Foreign risk-free interest rate
        sigma : float
            Diffusion volatility
        lam : float
            Jump intensity (average number of jumps per year)
        mu_j : float
            Average jump size (in logarithmic terms)
        sigma_j : float
            Jump size volatility
        option_type : str
            'call' or 'put'
        n_steps : int
            Number of time steps in each simulation
        n_sims : int
            Number of simulations
        seed : int
            Random seed for reproducibility

        Returns
        -------
        PricingResult
            Object containing option price and standard error
        """
        # Handle edge cases
        if T <= 0:
            if option_type.lower() == 'call':
                return PricingResult(price=max(0, S - K))
            else:
                return PricingResult(price=max(0, K - S))

        # Initialize array for final prices
        final_prices = np.zeros(n_sims)

        # Run simulations in parallel using numba
        for i in range(n_sims):
            final_prices[i] = MertonJumpDiffusion._simulate_path(
                S, T, r_d, r_f, sigma, lam, mu_j, sigma_j, n_steps, seed + i
            )

        # Calculate payoff
        if option_type.lower() == 'call':
            payoffs = np.maximum(final_prices - K, 0)
        else:  # put
            payoffs = np.maximum(K - final_prices, 0)

        # Calculate present value
        discount_factor = np.exp(-r_d * T)
        option_prices = discount_factor * payoffs

        # Calculate price and standard error
        price = np.mean(option_prices)
        se = np.std(option_prices) / np.sqrt(n_sims)

        return PricingResult(price=price, error=se)


class SABR:
    """
    SABR stochastic volatility model for pricing forex options.

    This model is particularly effective for handling volatility smile/skew
    effects observed in forex option markets.
    """

    @staticmethod
    def implied_vol(F, K, T, alpha, beta, rho, nu):
        """
        Calculate implied volatility using the SABR model.

        Parameters
        ----------
        F : float
            Forward price
        K : float
            Strike price
        T : float
            Time to maturity in years
        alpha : float
            Initial volatility level
        beta : float
            CEV parameter (0=normal, 1=lognormal)
        rho : float
            Correlation between volatility and price
        nu : float
            Volatility of volatility

        Returns
        -------
        float
            SABR implied volatility
        """
        # Handle numerical issues with very small time to maturity
        if T < 1e-6:
            return alpha / ((F*K)**((1-beta)/2))

        # Handle at-the-money case separately for numerical stability
        if abs(F - K) < 1e-6:
            # ATM formula
            ATM = alpha * (1 + ((1 - beta)**2/24) * alpha**2/(F**(2-2*beta)) +
                           0.25 * rho * beta * nu * alpha / (F**(1-beta)) +
                           (2 - 3 * rho**2) * nu**2 / 24) * T
            return ATM / (F**(1-beta))

        # For strikes away from ATM
        log_FK = np.log(F/K)

        z = (nu/alpha) * (F*K)**((1-beta)/2) * log_FK

        # Handle small values of z
        if abs(z) < 1e-6:
            x_z = 1 - 0.5 * rho * z + (2 - 3*rho**2)/12 * z**2
        else:
            x_z = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))

        # Implied volatility formula
        first_term = alpha / ((F*K)**((1-beta)/2) * (1 + (1-beta)**2/24 * log_FK**2 +
                                                     (1-beta)**4/1920 * log_FK**4))

        second_term = z/x_z

        third_term = 1 + ((1-beta)**2/24 * alpha**2/((F*K)**(1-beta)) +
                          0.25*rho*beta*nu*alpha/((F*K)**((1-beta)/2)) +
                          (2-3*rho**2)/24*nu**2) * T

        return first_term * second_term * third_term

    @staticmethod
    def price(S, K, T, r_d, r_f, alpha, beta, rho, nu, option_type='call'):
        """
        Calculate option price using the SABR model for implied volatility.

        Parameters
        ----------
        S : float
            Spot exchange rate
        K : float
            Strike price
        T : float
            Time to maturity in years
        r_d : float
            Domestic risk-free interest rate
        r_f : float
            Foreign risk-free interest rate
        alpha : float
            Initial volatility level
        beta : float
            CEV parameter (0=normal, 1=lognormal)
        rho : float
            Correlation between volatility and price
        nu : float
            Volatility of volatility
        option_type : str
            'call' or 'put'

        Returns
        -------
        PricingResult
            Object containing option price and approximate Greeks
        """
        # Handle edge cases
        if T <= 0:
            if option_type.lower() == 'call':
                return PricingResult(price=max(0, S - K), delta=1.0 if S > K else 0.0)
            else:
                return PricingResult(price=max(0, K - S), delta=-1.0 if S < K else 0.0)

        # Calculate forward price
        F = S * np.exp((r_d - r_f) * T)

        # Calculate implied volatility using SABR
        implied_vol = SABR.implied_vol(F, K, T, alpha, beta, rho, nu)

        # Price the option using Black-Scholes with the SABR implied volatility
        return BlackScholesFX.price(S, K, T, r_d, r_f, implied_vol, option_type)

    @staticmethod
    def calibrate(market_data, initial_params=None, bounds=None):
        """
        Calibrate SABR model parameters to market data.

        Parameters
        ----------
        market_data : list
            List of dictionaries with market data (strike, price, etc.)
        initial_params : tuple
            Initial parameters (alpha, beta, rho, nu) for optimization
        bounds : tuple
            Bounds for parameters

        Returns
        -------
        tuple
            Calibrated parameters (alpha, beta, rho, nu)
        """
        # Default initial parameters and bounds
        if initial_params is None:
            initial_params = (0.2, 0.5, -0.3, 0.3)

        if bounds is None:
            bounds = ((0.01, 1.0), (0.01, 0.99), (-0.999, 0.999), (0.01, 1.0))

        # Extract data
        forward = market_data[0]['forward']
        maturities = np.array([d['T'] for d in market_data])
        strikes = np.array([d['K'] for d in market_data])
        market_vols = np.array([d['vol'] for d in market_data])
        weights = np.array([d.get('weight', 1.0) for d in market_data])

        # Define the objective function - MSE between market and model implied vols
        def objective(params):
            alpha, beta, rho, nu = params
            model_vols = np.array([SABR.implied_vol(forward, K, T, alpha, beta, rho, nu)
                                  for K, T in zip(strikes, maturities)])
            return np.sum(weights * (model_vols - market_vols) ** 2)

        # Run optimization
        result = minimize(objective, initial_params,
                          method='L-BFGS-B', bounds=bounds)

        return result.x


class PricingEngine:
    """
    Unified interface for pricing options using different models.

    This class provides a convenient way to price options with different
    models and retrieve results in a consistent format.
    """

    def __init__(self):
        """Initialize the pricing engine."""
        self.models = {
            'black_scholes': self.price_black_scholes,
            'merton_jump': self.price_merton_jump,
            'merton_jump_mc': self.price_merton_jump_mc,
            'sabr': self.price_sabr
        }

    def price_option(self, model_name, params):
        """
        Price an option using the specified model.

        Parameters
        ----------
        model_name : str
            Name of the model to use
        params : dict
            Parameters for the model

        Returns
        -------
        PricingResult
            Object containing option price and Greeks
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        return self.models[model_name](params)

    @staticmethod
    def price_black_scholes(params):
        """Price an option using the Black-Scholes model."""
        return BlackScholesFX.price(
            params['S'],
            params['K'],
            params['T'],
            params['r_d'],
            params['r_f'],
            params['sigma'],
            params.get('option_type', 'call')
        )

    @staticmethod
    def price_merton_jump(params):
        """Price an option using the Merton Jump Diffusion model."""
        return MertonJumpDiffusion.price(
            params['S'],
            params['K'],
            params['T'],
            params['r_d'],
            params['r_f'],
            params['sigma'],
            params['lambda'],
            params['mu_j'],
            params['sigma_j'],
            params.get('option_type', 'call'),
            params.get('n_terms', 20)
        )

    @staticmethod
    def price_merton_jump_mc(params):
        """Price an option using the Merton Jump Diffusion model with Monte Carlo simulation."""
        return MertonJumpDiffusion.price_mc(
            params['S'],
            params['K'],
            params['T'],
            params['r_d'],
            params['r_f'],
            params['sigma'],
            params['lambda'],
            params['mu_j'],
            params['sigma_j'],
            params.get('option_type', 'call'),
            params.get('n_steps', 100),
            params.get('n_sims', 10000),
            params.get('seed', 42)
        )

    @staticmethod
    def price_sabr(params):
        """Price an option using the SABR model."""
        return SABR.price(
            params['S'],
            params['K'],
            params['T'],
            params['r_d'],
            params['r_f'],
            params['alpha'],
            params['beta'],
            params['rho'],
            params['nu'],
            params.get('option_type', 'call')
        )
