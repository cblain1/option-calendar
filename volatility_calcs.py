import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import warnings # To warn about extrapolation


import math
import datetime

from scipy.optimize import minimize, Bounds
import scipy.stats as stats
from scipy.interpolate import interp1d # For percentile finding
try:
    from scipy.integrate import simpson
except ImportError:
    from scipy.integrate import simps as simpson

from scipy.optimize import brentq

import traceback # For detailed error printing

def get_risk_free_rate():
    """Fetches the 13-week Treasury Bill yield (^IRX) as a proxy for the risk-free rate."""
    try:
        irx = yf.Ticker("^IRX")
        data = irx.history(period="5d")
        if data.empty or 'Close' not in data.columns:
             print("Warning: Could not fetch ^IRX data reliably. Using default 3.0%.")
             return 0.03
        last_rate = data['Close'].iloc[-1]
        if pd.isna(last_rate) or not isinstance(last_rate, (int, float)):
             print(f"Warning: Invalid rate value ({last_rate}) fetched for ^IRX. Using default 3.0%.")
             return 0.03
        return last_rate / 100.0
    except Exception as e:
        print(f"Warning: Failed to fetch risk-free rate: {e}. Using default 3.0%.")
        return 0.03


def black_scholes_call(S, K, T, r, sigma):
    """Calculates the Black-Scholes price for a European call option."""
    if sigma <= 1e-6 or T <= 1e-6:
        return max(S - K * np.exp(-r * T), 0)
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        return call_price
    except (ValueError, OverflowError, ZeroDivisionError):
        return np.nan

def black_scholes_put(S, K, T, r, sigma):
    """Calculates the Black-Scholes price for a European put option."""
    if sigma <= 1e-6 or T <= 1e-6:
        return max(K * np.exp(-r * T) - S, 0)
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        put_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        return put_price
    except (ValueError, OverflowError, ZeroDivisionError):
        return np.nan

def black_scholes_greeks_call(S, K, T, r, sigma):
    """Calculates Delta, Gamma, Vega, Theta for a European call option."""
    if sigma <= 1e-6 or T <= 1e-6 or S <= 0 or K <= 0:
        return {'delta': np.nan, 'gamma': np.nan, 'vega': np.nan, 'theta': np.nan}
    try:
        with np.errstate(divide='ignore', invalid='ignore'): # Suppress numpy warnings locally
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            pdf_d1 = stats.norm.pdf(d1)
            cdf_d1 = stats.norm.cdf(d1)
            cdf_d2 = stats.norm.cdf(d2)

            vega = (S * pdf_d1 * np.sqrt(T)) / 100.0 # Per 1% change
            delta = cdf_d1
            gamma = pdf_d1 / (S * sigma * np.sqrt(T))
            theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) \
                     - r * K * np.exp(-r * T) * cdf_d2) / 365.25 # Per day

        greeks = {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}
        # Check results after calculation block
        if any(not np.isfinite(g) for g in greeks.values()):
             return {'delta': np.nan, 'gamma': np.nan, 'vega': np.nan, 'theta': np.nan}
        return greeks
    except (ValueError, ZeroDivisionError, OverflowError):
        return {'delta': np.nan, 'gamma': np.nan, 'vega': np.nan, 'theta': np.nan}

def black_scholes_greeks_put(S, K, T, r, sigma):
    """Calculates Delta, Gamma, Vega, Theta for a European put option."""
    greeks = black_scholes_greeks_call(S, K, T, r, sigma) # Calculate call greeks first
    if pd.isna(greeks['delta']):
        return greeks # Return NaNs if call greeks failed
    # Put Delta = Call Delta - 1 (Correct formula)
    greeks['delta'] -= 1.0
    # Put Theta = Call Theta + r * K * exp(-rT)
    try:
        greeks['theta'] += (r * K * np.exp(-r * T)) / 365.25
    except (ValueError, OverflowError):
         greeks['theta'] = np.nan
    # Gamma and Vega are the same for calls and puts
    # Re-check for NaN/Inf after adjustment
    if any(not np.isfinite(g) for g in greeks.values()):
        return {'delta': np.nan, 'gamma': np.nan, 'vega': np.nan, 'theta': np.nan}
    return greeks

def binomial_american_option_pricing(option_type, S, K, T, r, sigma, n):
    """
    Calculates the price of an American option using the binomial option pricing model (CRR).
    Includes robustness checks for parameters.
    """
    # --- Input Validation ---
    if not all(isinstance(i, (int, float)) for i in [S, K, T, r, sigma]) or not isinstance(n, int):
        return np.nan
    if S <= 0 or K <= 0 or T < 0 or sigma <= 1e-8 or n <= 0:
        return np.nan
    if T == 0:
         if option_type.lower() == 'call': return max(S - K, 0)
         else: return max(K - S, 0)

    # --- Binomial Tree Parameters (CRR) ---
    dt = T / n
    if dt <= 0: return np.nan

    try:
        sigma_sqrt_dt = sigma * math.sqrt(dt)
        if abs(sigma_sqrt_dt) < 1e-9: return np.nan # Avoid instability

        u = math.exp(sigma_sqrt_dt)
        # Check if u is finite and not too large/small
        if not np.isfinite(u) or u < 1e-9 or u > 1e9: return np.nan
        if abs(u-1.0) < 1e-9: return np.nan # Avoid division by zero later

        d = 1.0 / u
        ert = math.exp(r * dt)
        if not np.isfinite(ert): return np.nan

        # --- CRITICAL ROBUSTNESS CHECKS ---
        if abs(u - d) < 1e-9: return np.nan
        # Check arbitrage condition strictly, allowing for float precision
        if not (d < ert + 1e-9 and ert < u + 1e-9): return np.nan

        p = (ert - d) / (u - d)
        # Check probability validity strictly, allowing for float precision
        if not (-1e-9 <= p <= 1.0 + 1e-9): return np.nan

        p = max(0.0, min(1.0, p)) # Clamp p strictly between 0 and 1

    except (ValueError, OverflowError, ZeroDivisionError):
        return np.nan

    # --- Calculations ---
    try:
        # Use numpy arrays for efficiency
        j_vals = np.arange(n + 1)
        stock_prices = S * (u**(n - j_vals)) * (d**j_vals)

        if option_type.lower() == 'call':
            option_values = np.maximum(stock_prices - K, 0.0)
        elif option_type.lower() == 'put':
            option_values = np.maximum(K - stock_prices, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        discount_factor = math.exp(-r * dt)
        option_values_mutable = option_values.copy()

        # Backward Induction - Optimized slightly
        for j in range(n - 1, -1, -1):
            idx = np.arange(j + 1)
            current_stock_prices = S * (u**(j-idx)) * (d**idx)
            # Expected value next period, discounted (Continuation Value)
            continuation_values = discount_factor * (p * option_values_mutable[idx] + (1.0 - p) * option_values_mutable[idx + 1])

            # Intrinsic Value at current node
            if option_type.lower() == 'call':
                intrinsic_values = np.maximum(current_stock_prices - K, 0.0)
            else: # put
                intrinsic_values = np.maximum(K - current_stock_prices, 0.0)

            # American Value: Max of intrinsic or continuation
            option_values_mutable[:j+1] = np.maximum(intrinsic_values, continuation_values)

        result_price = option_values_mutable[0]
        # Final check if price is valid
        if not np.isfinite(result_price): return np.nan
        return result_price

    except Exception: # Catch any unexpected calculation error
        return np.nan


def binomial_implied_volatility(option_type, market_price, S, K, T, r, n,
                                vol_min=0.005, vol_max=5.0, # Adjusted vol_min
                                max_iter=100, tol=1e-6):
    """Calculates implied volatility using the American Binomial Model."""
    if T <= 1e-6: return np.nan

    def objective_func(sigma):
        if sigma < vol_min: return 1e6 # Penalize below min
        try:
            price = binomial_american_option_pricing(option_type, S, K, T, r, sigma, n)
            if np.isnan(price): return 1e6 # Penalize if binomial fails
            return price - market_price
        except Exception:
            return 1e6

    try:
        # Basic price checks
        if market_price < 0: return np.nan
        intrinsic_call = max(0, S - K) # Intrinsic at T=0 for American
        intrinsic_put = max(0, K - S)
        # Check against simple bounds (allow small tolerance)
        if option_type.lower() == 'call' and market_price < intrinsic_call - tol: return np.nan
        if option_type.lower() == 'put' and market_price < intrinsic_put - tol: return np.nan
        if option_type.lower() == 'call' and market_price > S + tol : return np.nan
        if option_type.lower() == 'put' and market_price > K + tol : return np.nan

        iv = brentq(objective_func, vol_min, vol_max, xtol=tol, rtol=tol, maxiter=max_iter)
        # Check if result is too close to boundary
        if abs(iv - vol_min) < tol or abs(iv - vol_max) < tol: iv = np.nan
        return iv
    except ValueError: # Brentq fails (no sign change)
        return np.nan
    except Exception:
        return np.nan


def sabr_implied_volatility(K, F, T, alpha, beta, rho, sigma0):
    """
    Calculates SABR implied volatility using Hagan's 2002 approximation.
    Args:
        K (float or np.array): Strike price(s)
        F (float): Forward price
        T (float): Time to expiration
        alpha (float): Vol-of-vol parameter (nu)
        beta (float): Exponent parameter
        rho (float): Correlation parameter
        sigma0 (float): Initial/Forward volatility parameter (ATM vol guess)

    Returns:
        float or np.array: Implied volatility, or NaN on error/invalid inputs.
    """
    # Input validation
    if T <= 1e-6 or alpha < 0 or abs(rho) > 1 or sigma0 <= 0 or F <= 0:
        return np.nan if isinstance(K, (int, float)) else np.full_like(K, np.nan, dtype=float)

    K = np.maximum(K, 1e-9) # Ensure K is positive for logs/powers

    logFK = np.log(F / K)
    FKbeta = (F * K)**((1 - beta) / 2.0)
    a = (1 - beta)**2 * sigma0**2 / (24.0 * FKbeta**2)
    b = 0.25 * rho * beta * alpha * sigma0 / FKbeta
    c = (2.0 - 3.0 * rho**2) * alpha**2 / 24.0
    d = sigma0 * FKbeta * logFK

    # Handle ATM case (K == F) separately to avoid division by zero / log(1)
    # Need to handle array vs scalar K
    is_atm = np.isclose(F, K, rtol=1e-6, atol=1e-6) # Check if K is close to F

    # --- Calculate z and x(z) ---
    # Avoid division by zero if sigma0 or alpha is tiny
    if sigma0 < 1e-8 or alpha < 1e-8:
        # If vol or vol-of-vol is zero, IV is just sigma0 (or NaN if alpha=0?)
        # Let's return sigma0, assuming alpha > 0 was checked earlier
         return sigma0 if isinstance(K, (int, float)) else np.full_like(K, sigma0, dtype=float)

    # Calculate z/x(z) term carefully
    z = (alpha / sigma0) * FKbeta * logFK
    # Use limit for x(z) as z -> 0 (for ATM case) which is 1
    # Use formula otherwise
    # Need to calculate sqrt term safely
    sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z**2)
    xz = np.log((sqrt_term + z - rho) / (1.0 - rho))

    # Handle potential NaNs if sqrt_term calculation failed
    if np.isscalar(xz) and not np.isfinite(xz): return np.nan
    if not np.isscalar(xz): xz[~np.isfinite(xz)] = 1e9 # Assign large number if invalid? Or check inputs

    # Ratio z/x(z) - use limit of 1 at the money
    z_over_xz = np.ones_like(z) # Initialize with ATM limit
    non_atm_mask = ~is_atm & (np.abs(xz) > 1e-8) # Where xz is not zero and not ATM
    z_over_xz[non_atm_mask] = z[non_atm_mask] / xz[non_atm_mask]

    # --- Combine terms ---
    multiplier = alpha * logFK / d if abs(beta - 1.0) < 1e-6 else z_over_xz # Hagan note for beta=1
    term1 = (1.0 - beta)**2 / 24.0 * (sigma0**2 / FKbeta**2)
    term2 = 0.25 * (rho * beta * alpha / FKbeta) * sigma0
    term3 = (2.0 - 3.0 * rho**2) / 24.0 * alpha**2

    # Correction term (multiply by T)
    correction = (term1 + term2 + term3) * T

    # Final IV calculation
    iv = (sigma0 / FKbeta) * multiplier * (1.0 + correction)

    # Apply ATM simplification directly where K approx F
    # Simplified ATM IV approx: sigma0 * (1 + ((1-beta)^2/24)*sigma0^2 + (rho*beta*alpha/4)*sigma0 + ((2-3*rho^2)/24)*alpha^2) * T
    # This is roughly sigma0 * (1 + correction_at_atm * T)
    if np.any(is_atm):
         atm_correction = ((1-beta)**2/24)*sigma0**2 + (rho*beta*alpha/4)*sigma0 + ((2-3*rho**2)/24)*alpha**2
         atm_iv = sigma0 * (1 + atm_correction * T)
         if np.isscalar(is_atm): iv = atm_iv
         else: iv[is_atm] = atm_iv

    # Return NaN if calculation resulted in invalid number
    if np.isscalar(iv) and (not np.isfinite(iv) or iv < 0): return np.nan
    if not np.isscalar(iv): iv[~np.isfinite(iv) | (iv < 0)] = np.nan
    return iv


def sabr_objective_function(params, market_k, market_iv, weights, F, T, beta):
    """Objective function for SABR calibration."""
    alpha, rho, sigma0 = params # Parameters to optimize
    # Ensure parameters are valid during optimization steps
    alpha = max(alpha, 1e-8)
    rho = max(min(rho, 0.9999), -0.9999)
    sigma0 = max(sigma0, 1e-8)

    # Convert k back to K for SABR formula
    strikes = F * np.exp(market_k)

    sabr_iv = sabr_implied_volatility(strikes, F, T, alpha, beta, rho, sigma0)

    # Handle cases where SABR calculation fails
    if np.isscalar(sabr_iv) and pd.isna(sabr_iv): return 1e12 # Return large error
    if not np.isscalar(sabr_iv):
        if np.isnan(sabr_iv).any():
             # Penalize based on number of NaNs? Or just return large error?
             # print("Warning: NaNs in SABR IV during optimization") # Debug
             return 1e12 # Force optimizer away if calculation fails

    # Calculate weighted squared error
    error = np.sum(weights * (sabr_iv - market_iv)**2)
    return error

def fit_sabr_model(chain_df_with_greeks, S, r, T, F, fit_thresholds, beta=0.5): # Add beta, default 0.5
    """Fits the SABR model to the filtered market data."""
    print(f"--- Debug: Entering fit_sabr_model (beta={beta}) ---")
    # --- Filtering (same as fit_svi_model) ---
    if chain_df_with_greeks.empty: print("Debug: Input chain_df empty."); return {'success': False}
    if 'spread_pct' not in chain_df_with_greeks.columns: chain_df_with_greeks['spread_pct'] = (chain_df_with_greeks['ask'] - chain_df_with_greeks['bid']) / chain_df_with_greeks['midPrice']
    fitter_filter = ((chain_df_with_greeks['volume'] >= fit_thresholds['min_volume']) & (chain_df_with_greeks['openInterest'] >= fit_thresholds['min_oi']) & (chain_df_with_greeks['spread_pct'].notna()) & (chain_df_with_greeks['spread_pct'] <= fit_thresholds['max_spread_pct']) & (chain_df_with_greeks['vega'].notna()) & (chain_df_with_greeks['vega'] >= fit_thresholds['min_vega']) & (chain_df_with_greeks['marketIV'].notna()))
    fit_data = chain_df_with_greeks[fitter_filter].copy()
    print(f"Debug: Options remaining after filters for SABR fit: {len(fit_data)}")
    if len(fit_data) < 4: print(f"Error: Insufficient points ({len(fit_data)}) for SABR calibration. Need >= 4."); return {'success': False}
    print(f"Using {len(fit_data)} points for SABR calibration.")

    market_k = fit_data['k'].values
    market_iv = fit_data['marketIV'].values
    market_vega = fit_data['vega'].values
    market_spread = fit_data['ask'] - fit_data['bid']
    weights = market_vega / np.maximum(market_spread, 0.01)**2; weights = np.maximum(weights, 1e-6); weights = weights / np.sum(weights)

    # --- Initial Guesses & Bounds for SABR ---
    # Guess sigma0 from ATM IV
    atm_iv_idx = np.argmin(np.abs(market_k))
    sigma0_guess = market_iv[atm_iv_idx] if not pd.isna(market_iv[atm_iv_idx]) else 0.20
    alpha_guess = 0.3 # Vol-of-vol guess
    rho_guess = -0.5 # Correlation guess
    p0 = [alpha_guess, rho_guess, sigma0_guess]

    # Bounds: alpha > 0, -1 < rho < 1, sigma0 > 0
    param_bounds = [(1e-6, np.inf), (-0.9999, 0.9999), (1e-6, np.inf)]

    # --- Optimization ---
    result = minimize(sabr_objective_function, p0,
                      args=(market_k, market_iv, weights, F, T, beta), # Pass F, T, beta
                      method='L-BFGS-B',
                      bounds=param_bounds,
                      options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8})

    if not result.success:
        print(f"Warning: SABR optimization failed. Message: {result.message}")
        return {'success': False}

    optimal_params = result.x # alpha, rho, sigma0
    print(f"Optimal SABR Params (alpha, rho, sigma0): {np.round(optimal_params, 4)}")

    # --- Calculate Residuals ---
    fitted_strikes = F * np.exp(market_k)
    sabr_iv_fit = sabr_implied_volatility(fitted_strikes, F, T, optimal_params[0], beta, optimal_params[1], optimal_params[2])
    residuals = market_iv - sabr_iv_fit

    residuals_df = fit_data[['strike', 'type', 'k', 'marketIV']].copy(); residuals_df['fitted_iv'] = sabr_iv_fit; residuals_df['residual'] = residuals
    residuals_df['is_liquid'] = ((fit_data['volume'] >= LIQUIDITY_THRESHOLDS['min_volume']) & (fit_data['openInterest'] >= LIQUIDITY_THRESHOLDS['min_oi']) & (((fit_data['ask'] - fit_data['bid']) / fit_data['midPrice']) <= LIQUIDITY_THRESHOLDS['max_spread_pct']))

    # --- Optional Plotting ---
    try:
        plt.figure(figsize=(10, 6)); plt.scatter(market_k, market_iv, marker='.', label='Market IV (Fit Points)')
        plot_k = np.linspace(min(market_k), max(market_k), 100); plot_strikes = F * np.exp(plot_k)
        plot_iv = sabr_implied_volatility(plot_strikes, F, T, optimal_params[0], beta, optimal_params[1], optimal_params[2])
        plt.plot(plot_k, plot_iv, color='green', label=f'SABR Fit IV (beta={beta})'); plt.title(f'SABR Fit vs Market IV'); plt.xlabel('Log-Moneyness k=log(K/F)'); plt.ylabel('Implied Volatility'); plt.grid(True); plt.legend(); plt.show(block=False); plt.pause(0.1)
    except Exception as plot_e: print(f"Warning: Could not generate SABR fit plot: {plot_e}")

    fit_k_range = (market_k.min(), market_k.max())
    return {'success': True, 'params': optimal_params, 'beta': beta, 'residuals_df': residuals_df, 'fit_k_range': fit_k_range}

def analyze_sabr_skew_slope(sabr_params, beta, F, T):
    """Calculates the slope of the SABR IV curve around ATM."""
    alpha, rho, sigma0 = sabr_params
    k1 = -0.05; k2 = 0.05
    K1 = F * math.exp(k1); K2 = F * math.exp(k2)
    try:
        iv1 = sabr_implied_volatility(K1, F, T, alpha, beta, rho, sigma0)
        iv2 = sabr_implied_volatility(K2, F, T, alpha, beta, rho, sigma0)
        if pd.isna(iv1) or pd.isna(iv2): return {'atm_slope': np.nan}
        slope = (iv2 - iv1) / (k2 - k1) if abs(k2 - k1) > 1e-9 else 0.0
        return {'atm_slope': slope if np.isfinite(slope) else np.nan}
    except Exception as e: print(f"Error calculating SABR slope: {e}"); return {'atm_slope': np.nan}

def calculate_pdf_cdf_from_sabr(sabr_params, beta, S, r, T, F, fit_data_k_range,
                                pricing_model='bs', n_steps=BINOMIAL_STEPS, # Use BS/Binomial for pricing
                                num_points=300):
    """Calculates the implied PDF and CDF from fitted SABR parameters."""
    print(f"Calculating PDF/CDF from SABR parameters using model: {pricing_model}...")
    alpha, rho, sigma0 = sabr_params
    if T <= 1e-6: print("Error: T too small."); return {'success': False}

    try: # Wrap major parts
        min_k_fit, max_k_fit = fit_data_k_range
        k_range_extension = 0.2
        dense_k = np.linspace(min_k_fit - k_range_extension, max_k_fit + k_range_extension, num_points)
        dense_strikes = F * np.exp(dense_k)

        # --- Calculate Smooth IVs using SABR ---
        smooth_iv_sabr = sabr_implied_volatility(dense_strikes, F, T, alpha, beta, rho, sigma0)

        # --- Generate Smooth Prices using CHOSEN MODEL (BS or Binomial) with SABR IVs ---
        print(f"Generating smooth prices using {pricing_model} model with SABR IVs...")
        smooth_prices_list = []; valid_strike_indices = []
        for i, (k_strike, iv) in enumerate(zip(dense_strikes, smooth_iv_sabr)):
            if pd.isna(iv) or iv <= 1e-6 or pd.isna(k_strike) or k_strike <= 0: continue
            if pricing_model.lower() == 'binomial': price = binomial_american_option_pricing('put', S, k_strike, T, r, iv, n_steps)
            else: price = black_scholes_put(S, k_strike, T, r, iv) # Default to BS
            if pd.notna(price): smooth_prices_list.append(price); valid_strike_indices.append(i)
        smooth_prices = np.array(smooth_prices_list); dense_strikes = dense_strikes[valid_strike_indices]

        # --- Reuse existing PDF/CDF logic ---
        valid_indices = ~np.isnan(smooth_prices) & ~np.isnan(dense_strikes) & (dense_strikes > 0)
        dense_strikes = dense_strikes[valid_indices]; smooth_prices = smooth_prices[valid_indices]
        if len(dense_strikes) < 3: print("Error: Not enough valid points for differentiation."); return {'success': False}
        print(f"Generated {len(dense_strikes)} smooth price points for differentiation.")
        # ... (Finite difference loop - SAME AS calculate_pdf_cdf_from_svi) ...
        pdf_values = []; pdf_strikes = []
        for i in range(1, len(dense_strikes) - 1):
            if dense_strikes[i] <= dense_strikes[i-1] or dense_strikes[i+1] <= dense_strikes[i]: continue
            dK_plus = dense_strikes[i+1] - dense_strikes[i]; dK_minus = dense_strikes[i] - dense_strikes[i-1]
            if dK_plus <= 1e-8 or dK_minus <= 1e-8 or (dK_plus + dK_minus) <= 1e-8: continue
            try:
                price_prev, price_curr, price_next = smooth_prices[i-1], smooth_prices[i], smooth_prices[i+1]
                if not all(np.isfinite([price_prev, price_curr, price_next])): continue
                term1 = (price_next - price_curr) / dK_plus; term2 = (price_curr - price_prev) / dK_minus
                second_derivative = 2 * (term1 - term2) / (dK_plus + dK_minus)
                if second_derivative > 1e-9:
                    pdf = math.exp(r * T) * second_derivative
                    if np.isfinite(pdf) and pdf > 0: pdf_values.append(pdf); pdf_strikes.append(dense_strikes[i])
            except Exception: continue
        if not pdf_strikes: print("Error: No valid PDF values calculated."); return {'success': False}
        pdf_strikes = np.array(pdf_strikes); pdf_values = np.array(pdf_values)
        # ... (CDF Integration - SAME AS calculate_pdf_cdf_from_svi) ...
        sort_indices = np.argsort(pdf_strikes); pdf_strikes = pdf_strikes[sort_indices]; pdf_values = pdf_values[sort_indices]
        if len(pdf_strikes) < 2: print("Error: Not enough PDF points for CDF integration."); return {'success': False}
        cdf_values = np.array([simpson(pdf_values[:k+1], x=pdf_strikes[:k+1]) for k in range(len(pdf_strikes))]); cdf_strikes = pdf_strikes
        # ... (Normalization - SAME AS calculate_pdf_cdf_from_svi) ...
        integral_value = simpson(pdf_values, x=pdf_strikes) if len(pdf_strikes) >= 2 else 0.0; print(f"Integral of PDF (SABR): {integral_value:.4f}")
        if integral_value > 1e-6:
             if not np.isclose(integral_value, 1.0, atol=0.05, rtol=0.05): print(f"Note: Normalizing CDF (Integral={integral_value:.4f}).")
             cdf_values_normalized = cdf_values / integral_value
        else: print(f"Warning: Integral near zero ({integral_value:.4f}). Cannot normalize."); cdf_values_normalized = cdf_values
        cdf_values_final = np.maximum.accumulate(cdf_values_normalized); cdf_values_final = np.clip(cdf_values_final, 0, 1.0)
        print("PDF/CDF calculation from SABR complete.")
        return {'success': True, 'pdf_k': pdf_strikes, 'pdf_v': pdf_values, 'cdf_k': cdf_strikes, 'cdf_v': cdf_values_final, 'integral': integral_value}

    except Exception as e: print(f"Error in calculate_pdf_cdf_from_sabr: {e}"); traceback.print_exc(); return {'success': False}



def implied_volatility(option_type, market_price, S, K, T, r, model='bs', n=BINOMIAL_STEPS):
    """
    Calculates implied volatility using Black-Scholes or American Binomial Model.
    Handles errors and returns NaN if calculation fails.
    """
    global _iv_debug_counter
    debug_print_flag = False
    # Limit overall debug prints for performance
    should_print_input = (_iv_debug_counter < 5)
    should_print_fail = (_iv_debug_counter < 15)

    if should_print_input:
        print(f"Debug IV Input: model={model}, type={option_type}, P={market_price:.4f}, S={S:.2f}, K={K:.2f}, T={T:.4f}, r={r:.4f}")
        _iv_debug_counter += 1

    # --- Model Selection ---
    if model.lower() == 'binomial':
        iv = binomial_implied_volatility(option_type, market_price, S, K, T, r, n)
        if pd.isna(iv) and should_print_fail:
             print(f"   -> Binomial IV calculation failed for K={K}.")
             _iv_debug_counter += 1 # Increment only on printed failure
        return iv

    elif model.lower() == 'bs':
        vol_min, vol_max = 1e-4, 5.0
        tol = 1e-6

        def objective_func_bs(sigma):
            if sigma < vol_min: return 1e6
            try:
                if option_type.lower() == 'call': price = black_scholes_call(S, K, T, r, sigma)
                else: price = black_scholes_put(S, K, T, r, sigma)
                if np.isnan(price): return 1e6
                return price - market_price
            except Exception: return 1e6

        try:
            if market_price < 0: return np.nan
            intrinsic_call = max(0, S - K * np.exp(-r*T))
            intrinsic_put = max(0, K * np.exp(-r*T) - S)
            if option_type.lower() == 'call' and market_price < intrinsic_call - tol: return np.nan
            if option_type.lower() == 'put' and market_price < intrinsic_put - tol: return np.nan

            iv = brentq(objective_func_bs, vol_min, vol_max, xtol=tol, rtol=tol, maxiter=100)
            # Check if result is too close to boundary
            if abs(iv - vol_min) < tol or abs(iv - vol_max) < tol: iv = np.nan
            if pd.isna(iv) and should_print_fail:
                 print(f"   -> BS IV calculation failed (Brentq) for K={K}.")
                 _iv_debug_counter += 1
            return iv
        except ValueError: # Brentq fails
            if should_print_fail:
                print(f"Debug IV Brentq Fail (BS): K={K}, P={market_price:.4f}. Error: No sign change.")
                try:
                    f_min = objective_func_bs(vol_min)
                    f_max = objective_func_bs(vol_max)
                    print(f"     f({vol_min:.4f})={f_min:.4f}, f({vol_max:.4f})={f_max:.4f}")
                except Exception as bound_e: print(f"     Error checking bounds: {bound_e}")
                _iv_debug_counter += 1
            return np.nan
        except Exception as e:
            if should_print_fail: print(f"   -> BS IV calculation failed (Other Error) for K={K}: {e}"); _iv_debug_counter += 1
            return np.nan
    else:
        raise ValueError("Invalid model specified for IV calculation. Choose 'bs' or 'binomial'.")

def yang_zhang(price_data, window=30, trading_periods=252, return_last_only=True):
    """
    Calculates the Yang-Zhang historical volatility estimator.

    This estimator uses Open, High, Low, and Close prices and is generally
    considered more efficient than close-to-close volatility.

    Args:
        price_data (pd.DataFrame): DataFrame containing historical price data.
                                   Must include 'Open', 'High', 'Low', 'Close' columns.
                                   Index should be datetime-like.
        window (int, optional): Rolling window size. Defaults to 30.
        trading_periods (int, optional): Number of trading periods in a year
                                         for annualization. Defaults to 252.
        return_last_only (bool, optional): If True, returns only the last calculated
                                           volatility value. If False, returns the
                                           entire series. Defaults to True.

    Returns:
        float or pd.Series or None: The calculated annualized volatility.
                                    Returns float if return_last_only is True.
                                    Returns pd.Series if return_last_only is False.
                                    Returns None if input data is insufficient or invalid.

    Raises:
        KeyError: If required columns ('Open', 'High', 'Low', 'Close') are missing.
        TypeError: If input is not a pandas DataFrame.
    """
    if not isinstance(price_data, pd.DataFrame):
        raise TypeError("Input 'price_data' must be a pandas DataFrame.")

    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in price_data.columns for col in required_cols):
        raise KeyError(f"DataFrame missing one or more required columns: {required_cols}")

    if len(price_data) < window + 1: # Need at least window+1 periods for shift and rolling
         print(f"Warning: Insufficient data ({len(price_data)} rows) for window size {window}. Returning None.")
         return None

    # Ensure numeric types
    df = price_data[required_cols].astype(float)

    log_ho = (df['High'] / df['Open']).apply(np.log)
    log_lo = (df['Low'] / df['Open']).apply(np.log)
    log_co = (df['Close'] / df['Open']).apply(np.log)

    log_oc = (df['Open'] / df['Close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2

    log_cc = (df['Close'] / df['Close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    # Calculate rolling variances (sum / (window - 1)) - adjust sum to mean for direct variance?
    # Using sum / (window - 1) is equivalent to variance calculation excluding current point maybe?
    # Let's stick to the original implementation's structure for now.
    # Note: Pandas .var() uses N-1 by default (ddof=1)

    close_vol_term = log_cc_sq.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
    open_vol_term = log_oc_sq.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
    window_rs_term = rs.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))

    # Ensure terms are valid numbers before proceeding
    # Drop initial NaNs created by shift and rolling
    valid_indices = close_vol_term.dropna().index
    if len(valid_indices) == 0:
         print(f"Warning: No valid volatility calculations possible after rolling window {window}. Returning None.")
         return None

    # Recalculate on potentially reduced index if needed, or just filter
    close_vol_term = close_vol_term.loc[valid_indices]
    open_vol_term = open_vol_term.loc[valid_indices]
    window_rs_term = window_rs_term.loc[valid_indices]


    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))

    # Ensure terms used in the sum are not negative (they shouldn't be variances/rs sums)
    # However, floating point issues could arise? Add clip just in case.
    variance_estimate = (open_vol_term + k * close_vol_term + (1 - k) * window_rs_term).clip(lower=0)

    result = variance_estimate.apply(np.sqrt) * np.sqrt(trading_periods)

    if return_last_only:
        # Check if result series is empty after all calculations
        return result.iloc[-1] if not result.empty else None
    else:
        return result # Already dropped NaNs via valid_indices filtering
    

def build_term_structure(days, ivs):
    """
    Builds a term structure function using linear interpolation of IV points.

    Handles extrapolation by returning the IV of the nearest endpoint (flat).

    Args:
        days (list or np.ndarray): Sequence of days-to-expiration (DTE).
        ivs (list or np.ndarray): Sequence of corresponding implied volatilities.

    Returns:
        function: A function that takes a DTE (float or int) and returns
                  the interpolated or extrapolated IV (float). Returns None
                  if input data is insufficient or invalid.

    Raises:
        ValueError: If 'days' and 'ivs' have different lengths or contain non-numeric data.
    """
    if len(days) != len(ivs):
        raise ValueError("Input 'days' and 'ivs' must have the same length.")
    if len(days) < 2:
        warnings.warn(f"Insufficient data points ({len(days)}) for interpolation. Need at least 2. Returning None.")
        return None # Cannot interpolate with less than 2 points

    try:
        # Convert to numpy arrays and ensure numeric type
        days_arr = np.array(days, dtype=float)
        ivs_arr = np.array(ivs, dtype=float)
    except ValueError:
         raise ValueError("Input 'days' and 'ivs' must contain numeric data.")

    # Check for NaNs or Infs
    if np.isnan(days_arr).any() or np.isnan(ivs_arr).any() or \
       np.isinf(days_arr).any() or np.isinf(ivs_arr).any():
        warnings.warn("Input data contains NaNs or Infs. Attempting to remove.")
        mask = ~(np.isnan(days_arr) | np.isnan(ivs_arr) | np.isinf(days_arr) | np.isinf(ivs_arr))
        days_arr = days_arr[mask]
        ivs_arr = ivs_arr[mask]
        if len(days_arr) < 2:
             warnings.warn(f"Insufficient valid data points ({len(days_arr)}) after removing NaNs/Infs. Returning None.")
             return None


    # Sort arrays based on days (DTE)
    sort_idx = days_arr.argsort()
    days_sorted = days_arr[sort_idx]
    ivs_sorted = ivs_arr[sort_idx]

    # Check for duplicate days, which interp1d doesn't like
    unique_days, unique_idx = np.unique(days_sorted, return_index=True)
    if len(unique_days) < len(days_sorted):
        warnings.warn("Duplicate DTE values found. Using IV from the first occurrence.")
        days_sorted = days_sorted[unique_idx]
        ivs_sorted = ivs_sorted[unique_idx]
        if len(days_sorted) < 2:
             warnings.warn(f"Insufficient unique data points ({len(days_sorted)}) after removing duplicates. Returning None.")
             return None

    # Create the interpolation function (bounds_error=False prevents error on extrapolation)
    # We handle extrapolation manually in the wrapper.
    try:
        spline = interp1d(days_sorted, ivs_sorted, kind='linear', bounds_error=False) # No fill_value needed here
    except ValueError as e:
         # Catch potential errors during spline creation (e.g., not enough points if somehow checks failed)
         warnings.warn(f"Error creating interpolation spline: {e}. Returning None.")
         return None

    min_day = days_sorted[0]
    max_day = days_sorted[-1]
    min_iv = ivs_sorted[0]
    max_iv = ivs_sorted[-1]

    def term_spline_wrapper(dte):
        """Wrapper function to handle extrapolation."""
        try:
             dte_f = float(dte)
        except (ValueError, TypeError):
             warnings.warn(f"Invalid input DTE: {dte}. Must be numeric. Returning None.")
             return None

        if dte_f < min_day:
            # warnings.warn(f"DTE {dte_f} is below the minimum input DTE ({min_day}). Extrapolating flatly.")
            return min_iv # Flat extrapolation - return IV of the earliest date
        elif dte_f > max_day:
            # warnings.warn(f"DTE {dte_f} is above the maximum input DTE ({max_day}). Extrapolating flatly.")
            return max_iv # Flat extrapolation - return IV of the latest date
        else:
            # Interpolate within the range
            interpolated_iv = spline(dte_f)
            # interp1d might return a 0-d array, ensure it's a float
            return float(interpolated_iv)

    return term_spline_wrapper