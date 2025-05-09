"""
Unit tests for the pricing module.

This module tests the pricing models to ensure they are working correctly.
"""

import unittest
import numpy as np
from forex_options.pricing import BlackScholesFX, MertonJumpDiffusion, SABR


class TestBlackScholesFX(unittest.TestCase):
    """Test cases for the Black-Scholes (Garman-Kohlhagen) pricing model."""

    def test_atm_call_option(self):
        """Test ATM call option pricing."""
        # Parameters
        S = 3.35  # Spot rate
        K = 3.35  # Strike (ATM)
        T = 0.5   # 6 months to maturity
        r_d = 0.03  # EUR rate
        r_f = 0.08  # TND rate
        sigma = 0.15  # Volatility

        # Price the option
        result = BlackScholesFX.price(S, K, T, r_d, r_f, sigma, 'call')

        # For ATM options with forward rate F = S*exp((r_d-r_f)*T),
        # the price should be approximately F * 0.4 * sigma * sqrt(T)
        F = S * np.exp((r_d - r_f) * T)
        expected_price = F * 0.4 * sigma * np.sqrt(T)

        # Check if the price is close to the expected value
        self.assertAlmostEqual(result.price, expected_price, delta=0.01)

        # Delta for ATM call should be close to 0.5 * exp(-r_f * T)
        expected_delta = 0.5 * np.exp(-r_f * T)
        self.assertAlmostEqual(result.delta, expected_delta, delta=0.05)

    def test_itm_call_option(self):
        """Test ITM call option pricing."""
        # Parameters
        S = 3.5   # Spot rate
        K = 3.0   # Strike (ITM)
        T = 0.25  # 3 months to maturity
        r_d = 0.03  # EUR rate
        r_f = 0.08  # TND rate
        sigma = 0.10  # Volatility

        # Price the option
        result = BlackScholesFX.price(S, K, T, r_d, r_f, sigma, 'call')

        # The price should be higher than the intrinsic value
        intrinsic = S - K
        self.assertGreater(result.price, intrinsic)

        # Delta for deep ITM call should be close to exp(-r_f * T)
        self.assertGreater(result.delta, 0.7)

    def test_otm_call_option(self):
        """Test OTM call option pricing."""
        # Parameters
        S = 3.0   # Spot rate
        K = 3.5   # Strike (OTM)
        T = 0.75  # 9 months to maturity
        r_d = 0.03  # EUR rate
        r_f = 0.08  # TND rate
        sigma = 0.20  # Volatility

        # Price the option
        result = BlackScholesFX.price(S, K, T, r_d, r_f, sigma, 'call')

        # The price should be positive but less than the strike
        self.assertGreater(result.price, 0)
        self.assertLess(result.price, K)

        # Delta for OTM call should be between 0 and 0.5
        self.assertGreater(result.delta, 0)
        self.assertLess(result.delta, 0.5)

    def test_put_call_parity(self):
        """Test put-call parity relationship."""
        # Parameters
        S = 3.35  # Spot rate
        K = 3.35  # Strike
        T = 0.5   # 6 months to maturity
        r_d = 0.03  # EUR rate
        r_f = 0.08  # TND rate
        sigma = 0.15  # Volatility

        # Price call and put options
        call_result = BlackScholesFX.price(S, K, T, r_d, r_f, sigma, 'call')
        put_result = BlackScholesFX.price(S, K, T, r_d, r_f, sigma, 'put')

        # Put-call parity: C - P = S * exp(-r_f * T) - K * exp(-r_d * T)
        lhs = call_result.price - put_result.price
        rhs = S * np.exp(-r_f * T) - K * np.exp(-r_d * T)

        # Check if the relationship holds
        self.assertAlmostEqual(lhs, rhs, delta=1e-10)

    def test_expired_option(self):
        """Test pricing of an expired option."""
        # Parameters
        S = 3.35  # Spot rate
        K = 3.20  # Strike
        T = 0.0   # Expired option
        r_d = 0.03  # EUR rate
        r_f = 0.08  # TND rate
        sigma = 0.15  # Volatility

        # Price call option
        call_result = BlackScholesFX.price(S, K, T, r_d, r_f, sigma, 'call')

        # For expired options, price should be the intrinsic value
        expected_price = max(0, S - K)
        self.assertEqual(call_result.price, expected_price)

        # Price put option
        put_result = BlackScholesFX.price(S, K, T, r_d, r_f, sigma, 'put')

        # For expired options, price should be the intrinsic value
        expected_price = max(0, K - S)
        self.assertEqual(put_result.price, expected_price)


class TestMertonJumpDiffusion(unittest.TestCase):
    """Test cases for the Merton Jump Diffusion pricing model."""

    def test_no_jumps_equals_black_scholes(self):
        """Test that with zero jump intensity, the price matches Black-Scholes."""
        # Parameters
        S = 3.35  # Spot rate
        K = 3.35  # Strike
        T = 0.5   # 6 months to maturity
        r_d = 0.03  # EUR rate
        r_f = 0.08  # TND rate
        sigma = 0.15  # Volatility
        lam = 0.0  # Zero jump intensity
        mu_j = -0.05  # Average jump size (irrelevant with zero intensity)
        sigma_j = 0.1  # Jump volatility (irrelevant with zero intensity)

        # Price with Merton Jump Diffusion
        mjd_result = MertonJumpDiffusion.price(
            S, K, T, r_d, r_f, sigma, lam, mu_j, sigma_j, 'call')

        # Price with Black-Scholes
        bs_result = BlackScholesFX.price(S, K, T, r_d, r_f, sigma, 'call')

        # Prices should be equal
        self.assertAlmostEqual(mjd_result.price, bs_result.price, delta=1e-10)

        # Greeks should be equal
        self.assertAlmostEqual(mjd_result.delta, bs_result.delta, delta=1e-10)
        self.assertAlmostEqual(mjd_result.gamma, bs_result.gamma, delta=1e-10)
        self.assertAlmostEqual(mjd_result.vega, bs_result.vega, delta=1e-10)

    def test_negative_jumps_decrease_price(self):
        """Test that negative jumps decrease the call option price."""
        # Parameters
        S = 3.35  # Spot rate
        K = 3.35  # Strike
        T = 0.5   # 6 months to maturity
        r_d = 0.03  # EUR rate
        r_f = 0.08  # TND rate
        sigma = 0.15  # Volatility

        # Price with Black-Scholes
        bs_result = BlackScholesFX.price(S, K, T, r_d, r_f, sigma, 'call')

        # Price with negative jumps
        lam = 1.0  # 1 jump per year on average
        mu_j = -0.05  # -5% average jump size
        sigma_j = 0.1  # Jump volatility
        mjd_result = MertonJumpDiffusion.price(
            S, K, T, r_d, r_f, sigma, lam, mu_j, sigma_j, 'call')

        # With negative jumps, call price should be lower
        self.assertLess(mjd_result.price, bs_result.price)

    def test_positive_jumps_increase_price(self):
        """Test that positive jumps increase the call option price."""
        # Parameters
        S = 3.35  # Spot rate
        K = 3.35  # Strike
        T = 0.5   # 6 months to maturity
        r_d = 0.03  # EUR rate
        r_f = 0.08  # TND rate
        sigma = 0.15  # Volatility

        # Price with Black-Scholes
        bs_result = BlackScholesFX.price(S, K, T, r_d, r_f, sigma, 'call')

        # Price with positive jumps
        lam = 1.0  # 1 jump per year on average
        mu_j = 0.05  # +5% average jump size
        sigma_j = 0.1  # Jump volatility
        mjd_result = MertonJumpDiffusion.price(
            S, K, T, r_d, r_f, sigma, lam, mu_j, sigma_j, 'call')

        # With positive jumps, call price should be higher
        self.assertGreater(mjd_result.price, bs_result.price)

    def test_monte_carlo_consistency(self):
        """Test that Monte Carlo prices are consistent with analytical prices."""
        # Parameters
        S = 3.35  # Spot rate
        K = 3.35  # Strike
        T = 0.5   # 6 months to maturity
        r_d = 0.03  # EUR rate
        r_f = 0.08  # TND rate
        sigma = 0.15  # Volatility
        lam = 1.0  # 1 jump per year on average
        mu_j = -0.05  # -5% average jump size
        sigma_j = 0.1  # Jump volatility

        # Price with analytical formula
        analytical_result = MertonJumpDiffusion.price(
            S, K, T, r_d, r_f, sigma, lam, mu_j, sigma_j, 'call')

        # Price with Monte Carlo
        mc_result = MertonJumpDiffusion.price_mc(S, K, T, r_d, r_f, sigma, lam, mu_j, sigma_j,
                                                 'call', n_steps=100, n_sims=10000, seed=42)

        # Prices should be close (within a few standard errors)
        self.assertAlmostEqual(
            mc_result.price, analytical_result.price, delta=3*mc_result.error)


class TestSABR(unittest.TestCase):
    """Test cases for the SABR stochastic volatility model."""

    def test_flat_volatility_surface(self):
        """Test that with nu=0, volatility surface is flat."""
        # Parameters
        F = 3.35  # Forward price
        alpha = 0.15  # Initial volatility
        beta = 0.5  # CEV parameter
        rho = -0.3  # Correlation
        nu = 0.0  # Volatility of volatility (zero)
        T = 0.5  # 6 months to maturity

        # Calculate implied volatility for different strikes
        K1 = 3.00
        K2 = 3.35
        K3 = 3.70

        vol1 = SABR.implied_vol(F, K1, T, alpha, beta, rho, nu)
        vol2 = SABR.implied_vol(F, K2, T, alpha, beta, rho, nu)
        vol3 = SABR.implied_vol(F, K3, T, alpha, beta, rho, nu)

        # With nu=0, volatilities should match a CEV model pattern
        # For beta=0.5, this should have a specific skew

        # ATM volatility should be close to alpha
        self.assertAlmostEqual(vol2, alpha, delta=0.0001)

        # For beta=0.5, ITM vol should be lower than ATM vol
        self.assertLess(vol1, vol2)

        # For beta=0.5, OTM vol should be higher than ATM vol
        self.assertGreater(vol3, vol2)

    def test_increased_volatility_of_volatility(self):
        """Test that increasing nu increases the curvature of the volatility smile."""
        # Parameters
        F = 3.35  # Forward price
        alpha = 0.15  # Initial volatility
        beta = 0.5  # CEV parameter
        rho = 0.0  # Correlation (zero to isolate the vol-of-vol effect)
        T = 0.5  # 6 months to maturity

        # Strikes
        K1 = 3.00  # ITM
        K2 = 3.35  # ATM
        K3 = 3.70  # OTM

        # Calculate implied vols with low nu
        nu_low = 0.2
        vol1_low = SABR.implied_vol(F, K1, T, alpha, beta, rho, nu_low)
        vol2_low = SABR.implied_vol(F, K2, T, alpha, beta, rho, nu_low)
        vol3_low = SABR.implied_vol(F, K3, T, alpha, beta, rho, nu_low)

        # Calculate implied vols with high nu
        nu_high = 0.6
        vol1_high = SABR.implied_vol(F, K1, T, alpha, beta, rho, nu_high)
        vol2_high = SABR.implied_vol(F, K2, T, alpha, beta, rho, nu_high)
        vol3_high = SABR.implied_vol(F, K3, T, alpha, beta, rho, nu_high)

        # Higher vol-of-vol should give more pronounced smile
        # Calculate smile curvature (difference between wings and ATM)
        curvature_low = (vol1_low + vol3_low) / 2 - vol2_low
        curvature_high = (vol1_high + vol3_high) / 2 - vol2_high

        # Higher nu should give more curvature
        self.assertGreater(curvature_high, curvature_low)

    def test_negative_correlation_effect(self):
        """Test that negative correlation creates downward sloping skew."""
        # Parameters
        F = 3.35  # Forward price
        alpha = 0.15  # Initial volatility
        beta = 0.5  # CEV parameter
        T = 0.5  # 6 months to maturity
        nu = 0.4  # Vol of vol

        # Strikes
        K1 = 3.00  # ITM
        K2 = 3.35  # ATM
        K3 = 3.70  # OTM

        # Calculate implied vols with negative correlation
        rho_neg = -0.7
        vol1_neg = SABR.implied_vol(F, K1, T, alpha, beta, rho_neg, nu)
        vol2_neg = SABR.implied_vol(F, K2, T, alpha, beta, rho_neg, nu)
        vol3_neg = SABR.implied_vol(F, K3, T, alpha, beta, rho_neg, nu)

        # Calculate implied vols with positive correlation
        rho_pos = 0.7
        vol1_pos = SABR.implied_vol(F, K1, T, alpha, beta, rho_pos, nu)
        vol2_pos = SABR.implied_vol(F, K2, T, alpha, beta, rho_pos, nu)
        vol3_pos = SABR.implied_vol(F, K3, T, alpha, beta, rho_pos, nu)

        # Calculate skew (difference between ITM and OTM vols)
        skew_neg = vol1_neg - vol3_neg
        skew_pos = vol1_pos - vol3_pos

        # Negative correlation should give downward sloping skew (ITM > OTM)
        self.assertGreater(skew_neg, 0)

        # Positive correlation should give upward sloping skew (ITM < OTM)
        self.assertLess(skew_pos, 0)

    def test_sabr_option_pricing(self):
        """Test SABR option pricing by comparing with Black-Scholes."""
        # Parameters
        S = 3.35  # Spot rate
        K = 3.35  # Strike (ATM)
        T = 0.5  # 6 months to maturity
        r_d = 0.03  # EUR rate
        r_f = 0.08  # TND rate
        alpha = 0.15  # Initial volatility
        beta = 1.0  # CEV parameter (1.0 makes it lognormal)
        rho = 0.0  # Correlation (zero to isolate vol-of-vol effect)
        nu = 0.0  # Vol of vol (zero to match Black-Scholes)

        # Price with SABR
        sabr_result = SABR.price(
            S, K, T, r_d, r_f, alpha, beta, rho, nu, 'call')

        # Price with Black-Scholes
        bs_result = BlackScholesFX.price(S, K, T, r_d, r_f, alpha, 'call')

        # With beta=1.0 and nu=0.0, SABR should match Black-Scholes
        self.assertAlmostEqual(sabr_result.price, bs_result.price, delta=1e-4)
        self.assertAlmostEqual(sabr_result.delta, bs_result.delta, delta=1e-4)


if __name__ == '__main__':
    unittest.main()
