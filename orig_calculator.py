import FreeSimpleGUI as sg
import yfinance as yf
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import numpy as np
import threading
#from ibapi.client import IB

# Assume these imports are at the top of the file where compute_recommendation lives
import pandas as pd # Assuming pandas is still used for dataframes
from dateutil.parser import parse # Useful for robust date parsing if needed

# Import the NEW functions (we will define these later)
from option_data_api import (
    fetch_option_chains_list,       # Replaces yf.Ticker() partially
    fetch_detailed_option_chain,  # Replaces stock.option_chain(exp_date)
    get_current_underlying_price, # Replaces the old way of getting price
    fetch_historical_price_data,  # Replaces stock.history()
    get_nearby_expirations,        # The function we developed earlier
    get_risk_free_rate,
)

from ibapi.client import *
from ibapi.wrapper import *
import time
import threading

import connection

# Assume these are available from elsewhere (or need to be created/verified)
from volatility_calcs import build_term_structure, yang_zhang

# --- Constants ---
TARGET_DAYS = [30, 60, 90, 180] # Target days out for analysis summary (can be adjusted)
PLOT_MIN_DAYS = 20 # Min days out for expirations included in surface plot
PLOT_MAX_DAYS = 400 # Max days out for expirations included in surface plot
LIQUIDITY_THRESHOLDS = {'min_volume': 10, 'min_oi': 100, 'max_spread_pct': 0.25}
SVI_FIT_THRESHOLDS = {'min_volume': 10, 'min_oi': 100, 'max_spread_pct': 0.20, 'min_vega': 0.005} # Adjusted min_vega
NEAR_MONEY_PCT = 0.05 # For liquidity concentration calc
RELATIVE_VOL_SD = 1.5 # Number of standard deviations for cheap/expensive vol
_iv_debug_counter = 0 # Global counter for limited printing

# --- Binomial Steps Constant ---
# WARNING: Higher n increases accuracy but drastically increases runtime!
BINOMIAL_STEPS = 75 # Adjust as needed (50-100 is typical)

def filter_dates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)
    
    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)

    arr = []
    for i, date in enumerate(sorted_dates):
        if date >= cutoff_date:
            arr = [d.strftime("%Y-%m-%d") for d in sorted_dates[:i+1]]  
            break
    
    if len(arr) > 0:
        if arr[0] == today.strftime("%Y-%m-%d"):
            return arr[1:]
        return arr

    raise ValueError("No date 45 days or more in the future found.")


def get_current_price(ticker):
    todays_data = ticker.history(period='1d')
    return todays_data['Close'].iloc[0]

 #--- Refactored Function ---

def compute_recommendation(app, ticker):
    try:
        ticker = ticker.strip().upper()
        if not ticker:
            return "Error: No stock symbol provided."

        # 1. Fetch the list of available option chain summaries from the new API
        try:
            # This function should return a list of chain summaries (like the user provided)
            # Each summary should contain at least 'exchange' and 'expirations'
            all_chains_summary = fetch_option_chains_list(ticker)
            if not all_chains_summary:
                return f"Error: No option chain summary found for stock symbol '{ticker}' via API."
        except Exception as e:
            # Log the specific error e if needed
            return f"Error: Failed to fetch option chain list for '{ticker}'. API Error: {e}"

        # 2. Find the desired chain (e.g., 'SMART') and its expirations
        target_exchange = 'SMART' # Or make this configurable
        selected_chain_summary = None
        for chain_summary in all_chains_summary:
             # Adapt access based on actual structure (dict vs object)
            try:
                exchange = getattr(chain_summary, 'exchange', chain_summary.get('exchange'))
                if exchange == target_exchange:
                    selected_chain_summary = chain_summary
                    break
            except (AttributeError, TypeError, KeyError):
                 print(f"Warning: Could not read exchange from chain summary item: {chain_summary}")
                 continue # Skip malformed items

        if selected_chain_summary is None:
            return f"Error: Could not find option chain summary for exchange '{target_exchange}' for '{ticker}'."

        # Extract expirations (adapt access based on actual structure)
        try:
            raw_expirations = getattr(selected_chain_summary, 'expirations', selected_chain_summary.get('expirations'))
            if not raw_expirations:
                 return f"Error: No expirations found in the '{target_exchange}' chain summary for '{ticker}'."
        except (AttributeError, TypeError, KeyError):
             return f"Error: Could not access expirations in the '{target_exchange}' chain summary for '{ticker}'."

        # 3. Filter expiration dates to the desired range (e.g., 0-45 DTE)
        # We might need slightly more than 45 days for the spline calculation, let's use 60?
        # Or stick to 45 if that's the strict requirement. Let's use 60 for buffer.
        days_filter = 60 # Increased slightly for calculation buffer
        try:
            # Uses the function we developed, assuming YYYYMMDD input format from API
            # Output format is YYYY-MM-DD
            filtered_exp_dates = get_nearby_expirations(raw_expirations, days_out=days_filter)
            if not filtered_exp_dates:
                 return f"Error: No expiration dates found within the next {days_filter} days for '{ticker}' on {target_exchange}."
        except Exception as e:
            # Log specific error e
            return f"Error: Failed to filter expiration dates for '{ticker}'. Error: {e}"

        # 4. Get the current underlying price
        try:
            underlying_price = get_current_underlying_price(ticker)
            if underlying_price is None or not isinstance(underlying_price, (int, float)) or underlying_price <= 0:
                raise ValueError("Invalid underlying price received.")
        except Exception as e:
            # Log specific error e
            return f"Error: Unable to retrieve underlying stock price for '{ticker}'. API Error: {e}"

        # 5. Fetch detailed option chain data ONLY for filtered dates
        #    and calculate metrics
        atm_iv = {}
        straddle = None
        first_chain = True # Flag to calculate straddle only for the first (nearest) date

        for exp_date_str in sorted(filtered_exp_dates): # Ensure processing nearest date first
            try:
                # This function needs to return an object/dict with .calls and .puts DataFrames
                chain_details = fetch_detailed_option_chain(ticker, exp_date_str, exchange=target_exchange)

                # --- Crucial Step: Adapt column names if necessary ---
                # Assuming the returned dataframes have columns:
                # 'strike', 'impliedVolatility', 'bid', 'ask'
                calls = chain_details.calls
                puts = chain_details.puts
                # -------------------------------------------------------

                if calls.empty or puts.empty:
                    print(f"Warning: Empty calls or puts found for {ticker} {exp_date_str} on {target_exchange}. Skipping.")
                    continue

                # Find ATM options (ensure underlying_price is float)
                underlying_price_f = float(underlying_price)
                call_diffs = (calls['strike'].astype(float) - underlying_price_f).abs()
                call_idx = call_diffs.idxmin()
                call_iv = calls.loc[call_idx, 'impliedVolatility']

                put_diffs = (puts['strike'].astype(float) - underlying_price_f).abs()
                put_idx = put_diffs.idxmin()
                put_iv = puts.loc[put_idx, 'impliedVolatility']

                # Check if IVs are valid numbers before averaging
                if pd.isna(call_iv) or pd.isna(put_iv):
                     print(f"Warning: NaN IV found for ATM options {ticker} {exp_date_str}. Skipping IV calc for this date.")
                     continue # Skip this date if IV is bad

                atm_iv_value = (float(call_iv) + float(put_iv)) / 2.0
                atm_iv[exp_date_str] = atm_iv_value # Keyed by YYYY-MM-DD string

                # Calculate straddle for the first (nearest) expiration date
                if first_chain:
                    call_bid = calls.loc[call_idx, 'bid']
                    call_ask = calls.loc[call_idx, 'ask']
                    put_bid = puts.loc[put_idx, 'bid']
                    put_ask = puts.loc[put_idx, 'ask']

                    # Ensure bids/asks are valid numbers
                    call_bid_f = float(call_bid) if pd.notna(call_bid) else None
                    call_ask_f = float(call_ask) if pd.notna(call_ask) else None
                    put_bid_f = float(put_bid) if pd.notna(put_bid) else None
                    put_ask_f = float(put_ask) if pd.notna(put_ask) else None

                    call_mid = (call_bid_f + call_ask_f) / 2.0 if call_bid_f is not None and call_ask_f is not None else None
                    put_mid = (put_bid_f + put_ask_f) / 2.0 if put_bid_f is not None and put_ask_f is not None else None

                    if call_mid is not None and put_mid is not None:
                        straddle = (call_mid + put_mid)
                    first_chain = False # Don't calculate straddle again

            except Exception as e:
                print(f"Warning: Failed to process chain for {ticker} {exp_date_str}. Error: {e}. Skipping date.")
                continue # Skip this date if processing fails

        # 6. Check if we gathered enough IV data
        if not atm_iv or len(atm_iv) < 2 : # Need at least 2 points for spline/slope usually
            return f"Error: Could not determine enough valid ATM IVs for '{ticker}' within {days_filter} days on {target_exchange}."

        # 7. Calculate DTEs and build term structure
        today = datetime.date.today()
        dtes = []
        ivs = []
        # Ensure dates are sorted for DTE calculation
        for exp_date_str in sorted(atm_iv.keys()):
            iv = atm_iv[exp_date_str]
            try:
                # Use dateutil.parser for potentially more robust parsing if needed
                exp_date_obj = datetime.datetime.strptime(exp_date_str, "%Y-%m-%d").date()
                # exp_date_obj = parse(exp_date_str).date() # Alternative parsing
                days_to_expiry = (exp_date_obj - today).days
                if days_to_expiry >= 0: # Only include future dates
                    dtes.append(days_to_expiry)
                    ivs.append(iv)
            except ValueError:
                 print(f"Warning: Could not parse date {exp_date_str} for DTE calculation.")
                 continue

        if len(dtes) < 2: # Check again after filtering DTEs
            return f"Error: Not enough valid future expiration dates with IV found for term structure calculation for '{ticker}'."

        try:
            term_spline = build_term_structure(dtes, ivs)
        except Exception as e:
             return f"Error: Failed to build IV term structure for '{ticker}'. Error: {e}"

        # 8. Calculate final metrics using the term structure and historical data
        try:
            # Ensure 45 is within or extrapolated from the DTE range used for the spline
            if 45 < dtes[0]: # Basic check if 45 DTE is before the first available date
                 ts_slope_0_45 = None # Cannot calculate slope reliably
                 print("Warning: 45 DTE is before the first available DTE, cannot calculate 0-45 slope.")
            else:
                 # Ensure we don't divide by zero if dtes[0] == 45
                 denominator = (45 - dtes[0])
                 ts_slope_0_45 = (term_spline(45) - term_spline(dtes[0])) / denominator if denominator != 0 else 0.0

            # Ensure 30 is within or extrapolated from the DTE range
            iv30 = term_spline(30) if 30 >= dtes[0] else None # Don't calculate if 30 DTE is too soon

            # Fetch historical data
            price_history = fetch_historical_price_data(ticker, period='3mo') # Needs DataFrame return
            if price_history is None or price_history.empty:
                 raise ValueError("Failed to fetch valid historical price data.")

            rv30 = yang_zhang(price_history) # Needs DataFrame input
            if rv30 is None or rv30 <= 0:
                 raise ValueError("Failed to calculate valid historical volatility (RV30).")

            # Calculate IV/RV ratio only if both are valid
            iv30_rv30 = iv30 / rv30 if iv30 is not None else None

            # Calculate Avg Volume (ensure 'Volume' column exists)
            if 'Volume' not in price_history.columns:
                 raise ValueError("Historical data missing 'Volume' column.")
            avg_volume_series = price_history['Volume'].rolling(30).mean().dropna()
            avg_volume = avg_volume_series.iloc[-1] if not avg_volume_series.empty else 0

            # Calculate Expected Move
            expected_move = str(round(straddle / underlying_price_f * 100, 2)) + "%" if straddle is not None else "N/A"

            # Return results (use None for metrics that couldn't be calculated)
            return {
                'avg_volume': avg_volume >= 1500000 if avg_volume is not None else None,
                'iv30_rv30': iv30_rv30 >= 1.25 if iv30_rv30 is not None else None,
                'ts_slope_0_45': ts_slope_0_45 <= -0.00406 if ts_slope_0_45 is not None else None,
                'expected_move': expected_move
             }

        except Exception as e:
             # Log specific error e
             return f"Error: Failed during final metric calculation for '{ticker}'. Error: {e}"

    except Exception as e:
        # General catch-all, should ideally log the full traceback
        print(f"CRITICAL ERROR in compute_recommendation for {ticker}: {e}")
        # Re-raise or return a standard error format
        # raise Exception(f'Error occurred processing {ticker}') # Original way
        return f"FATAL Error processing {ticker}: Check logs." # Alternative return
        
def main_gui():
    main_layout = [
        [sg.Text("Enter Stock Symbol:"), sg.Input(key="stock", size=(20, 1), focus=True)],
        [sg.Button("Submit", bind_return_key=True), sg.Button("Exit")],
        [sg.Text("", key="recommendation", size=(50, 1))]
    ]
    
    window = sg.Window("Earnings Position Checker", main_layout)
    
    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break

        if event == "Submit":
            window["recommendation"].update("")
            stock = values.get("stock", "")

            loading_layout = [[sg.Text("Loading...", key="loading", justification="center")]]
            loading_window = sg.Window("Loading", loading_layout, modal=True, finalize=True, size=(275, 200))

            result_holder = {}

            def worker():
                try:
                    result = compute_recommendation(stock)
                    result_holder['result'] = result
                except Exception as e:
                    result_holder['error'] = str(e)

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

            while thread.is_alive():
                event_load, _ = loading_window.read(timeout=100)
                if event_load == sg.WINDOW_CLOSED:
                    break
            thread.join(timeout=1)

            if 'error' in result_holder:
                loading_window.close()
                window["recommendation"].update(f"Error: {result_holder['error']}")
            elif 'result' in result_holder:
                loading_window.close()
                result = result_holder['result']

                avg_volume_bool    = result['avg_volume']
                iv30_rv30_bool     = result['iv30_rv30']
                ts_slope_bool      = result['ts_slope_0_45']
                expected_move      = result['expected_move']
                
                if avg_volume_bool and iv30_rv30_bool and ts_slope_bool:
                    title = "Recommended"
                    title_color = "#006600"
                elif ts_slope_bool and ((avg_volume_bool and not iv30_rv30_bool) or (iv30_rv30_bool and not avg_volume_bool)):
                    title = "Consider"
                    title_color = "#ff9900"
                else:
                    title = "Avoid"
                    title_color = "#800000"
                
                result_layout = [
                    [sg.Text(title, text_color=title_color, font=("Helvetica", 16))],
                    [sg.Text(f"avg_volume: {'PASS' if avg_volume_bool else 'FAIL'}", text_color="#006600" if avg_volume_bool else "#800000")],
                    [sg.Text(f"iv30_rv30: {'PASS' if iv30_rv30_bool else 'FAIL'}", text_color="#006600" if iv30_rv30_bool else "#800000")],
                    [sg.Text(f"ts_slope_0_45: {'PASS' if ts_slope_bool else 'FAIL'}", text_color="#006600" if ts_slope_bool else "#800000")],
                    [sg.Text(f"Expected Move: {expected_move}", text_color="blue")],
                    [sg.Button("OK")]
                ]
                
                result_window = sg.Window("Recommendation", result_layout, modal=True, finalize=True, size=(275, 200))
                while True:
                    event_result, _ = result_window.read(timeout=100)
                    if event_result in (sg.WINDOW_CLOSED, "OK"):
                        break
                result_window.close()
    
    window.close()

def gui():
    main_gui()

    # --- Main Execution Block ---
if __name__ == "__main__":
    
    app = connection.ConnectionApp()
    
    try:
        print("Connecting to IB Gateway/TWS...")
        app.connect("127.0.0.1", 7496, clientId=1) # Adjust port/clientId

        # Start the message processing loop in a separate thread
        api_thread = threading.Thread(target=app.run, daemon=True)
        api_thread.start()

        # Wait for connection and nextValidId
        if not app.connection_event.wait(timeout=10):
             print("Failed to connect or get nextValidId from IB.")
             # app.disconnect() # Disconnect might hang if never connected
             raise ConnectionError("IB connection timeout")

        print("Connection successful. Ready to process.")

        # --- Run Analysis ---
        ticker = "NVDA" # Or get from input
        result = compute_recommendation(app, ticker) # Pass the connected app instance
        print("\n--- Result ---")
        print(result)
        print("--------------")

    except ConnectionError as ce:
         print(f"Connection Error: {ce}")
    except Exception as e:
         print(f"An error occurred: {e}")
         import traceback
         traceback.print_exc()
    finally:
        if app.isConnected():
            print("Disconnecting...")
            app.disconnect()
            # Give thread time to potentially finish processing disconnect
            time.sleep(1)
        print("Program finished.")

    gui()