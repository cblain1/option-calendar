# --- Imports ---
import yfinance as yf # Keep for YangZhang initially, can remove later if RV calc changes
import numpy as np
import pandas as pd
import math
import datetime
from datetime import date, timedelta
import warnings
from scipy.optimize import minimize, Bounds, brentq
import scipy.stats as stats
from scipy.interpolate import interp1d

import ibapi
try:
    from scipy.integrate import simpson
except ImportError:
    from scipy.integrate import simps as simpson
# import matplotlib.pyplot as plt # Optional plotting
import traceback
import threading
import time # For sleep

# --- Use PySimpleGUI if available ---
try:
    import PySimpleGUI as sg
except ImportError:
    import FreeSimpleGUI as sg

# --- IBKR Imports ---
from ibapi import wrapper
from ibapi import client
from ibapi.contract import Contract as IBContract
from ibapi.common import *
from ibapi.ticktype import TickTypeEnum
from ibapi.ticktype import TickType

# --- Suppress warnings ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
pd.options.mode.chained_assignment = None

# --- Constants ---
TARGET_DAYS = [30, 60]
AVG_VOLUME_THRESHOLD = 1500000
IV_RV_THRESHOLD = 1.25
MODIFIED_SLOPE_THRESHOLD = -0.10 # EXAMPLE - NEEDS VALIDATION
SVI_FIT_THRESHOLDS = {'min_volume': 10, 'min_oi': 100, 'max_spread_pct': 0.20, 'min_vega': 0.005}
BINOMIAL_STEPS = 50
_iv_debug_counter = 0
IB_HOST = '127.0.0.1' # Standard localhost
IB_PORT = 7496 # Default TWS Paper Trading port - CHANGE IF NEEDED
IB_CLIENT_ID = 1 # Choose a unique client ID

# --- Helper Functions ---

def get_risk_free_rate():
    """Fetches the current risk-free rate using Yahoo Finance (^IRX - 13-week Treasury Bill)."""
    irx = yf.Ticker("^IRX")
    hist = irx.history(period="1d")
    if not hist.empty:
        rate = hist['Close'].iloc[0] / 100.0  # Divide by 100 to get decimal form
        return rate
    else:
        warnings.warn("Could not fetch risk-free rate from yfinance. Returning default 0.02.")
        return 0.02  # Default risk-free rate if fetch fails


# --- Pricing/Greeks/IV/SVI/YZ Functions ---
# (Keep all the calculation functions: black_scholes_..., binomial_...,
#  implied_volatility, calculate_forward_price, yang_zhang,
#  svi_total_variance, svi_objective_function, fit_svi_model_single_exp,
#  analyze_svi_skew_slope as defined in the previous fully corrected script)
# Ensure they are present here...
# --- Example Placeholder ---
def black_scholes_call(S, K, T, r, sigma): return 1.0 # Placeholder
def black_scholes_put(S, K, T, r, sigma): return 1.0 # Placeholder
def black_scholes_greeks_call(S, K, T, r, sigma): return {'delta': 0.5, 'gamma': 0.1, 'vega': 0.1, 'theta': -0.01} # Placeholder
def black_scholes_greeks_put(S, K, T, r, sigma): return {'delta': -0.5, 'gamma': 0.1, 'vega': 0.1, 'theta': -0.01} # Placeholder
def implied_volatility(option_type, market_price, S, K, T, r, model='bs', n=BINOMIAL_STEPS): return 0.25 # Placeholder
def calculate_forward_price(S, r, T, q=0.0): return S * math.exp(r * T) # Placeholder
def yang_zhang(price_data, window=30, trading_periods=252, return_last_only=True): return 0.20 # Placeholder
def svi_total_variance(k, params): return 0.1 # Placeholder
def svi_objective_function(params, market_k, market_w, weights): return 1.0 # Placeholder
def fit_svi_model_single_exp(chain_df_with_greeks, S, r, T, F, fit_thresholds): return [0.1, 0.1, -0.5, 0.0, 0.1] # Placeholder
def analyze_svi_skew_slope(svi_params, F, T): return {'atm_slope': -0.2} # Placeholder
def calculate_chain_iv_greeks(chain_df, S, r, T, q, pricing_model='bs'): # Placeholder, replace with your actual function
    chain_df['F'] = calculate_forward_price(S, r, T, q)
    chain_df['marketIV'] = 0.25
    chain_df['deltaCall'] = 0.5
    chain_df['gammaCall'] = 0.1
    chain_df['vegaCall'] = 0.1
    chain_df['thetaCall'] = -0.01
    chain_df['deltaPut'] = -0.5
    chain_df['gammaPut'] = 0.1
    chain_df['vegaPut'] = 0.1
    chain_df['thetaPut'] = -0.01
    return chain_df
class IBapi(wrapper.EWrapper, client.EClient):
    def __init__(self):
        wrapper.EWrapper.__init__(self)
        client.EClient.__init__(self, self)
        self.data = {}
        self.price_data = {}
        self.hist_data = {}
        self.option_chain_params = {}
        self.option_chain_data = {}
        self.contract_details = {}
        self.reqIdCounter = 0
        self.lock = threading.Lock() # Lock for thread-safe data access

    def nextReqId(self):
        with self.lock:
            self.reqIdCounter += 1
            return self.reqIdCounter

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson):
        if reqId != -1:
            print(f"Error id:{reqId}, code:{errorCode}, msg:{errorString}")

    def contractDetails(self, reqId, contractDetails):
        with self.lock:
            if reqId not in self.contract_details:
                self.contract_details[reqId] = []
            self.contract_details[reqId].append(contractDetails)

    def contractDetailsEnd(self, reqId):
        print(f"Contract details received for reqId {reqId}")

    # def tickGeneric(self, reqId, tickType, value):
    #     if reqId not in self.data:
    #         self.data[reqId] = {}
    #     self.data[reqId][TickTypeEnum.to_str(tickType)] = value

    # def tickPrice(self, reqId: TickerId, tickType: TickType, price: float, attrib: TickAttrib):
    #     if reqId not in self.data:
    #          self.data[reqId] = {}
    #     print(f"TickTypeEnum object: {TickTypeEnum}")  # Debugging print statement
    #     print(f"TickType value: {tickType}, type: {type(tickType)}") # Print tickType value and its type
    #     try:
    #         tick_type_str = IBapi.ticktype.TickTypeEnum(tickType).name  # Fully qualified name
    #         self.data[reqId][tick_type_str] = price
    #     # Optional: Explore TickAttrib attributes
    #         if attrib.canAutoExecute:
    #             print(f"Tick can auto-execute for ReqId: {reqId}, TickType: {tick_type_str}")
    #     except Exception as e:
    #         print(f"Error in tickPrice: {e}") # Catch any potential errors during Enum conversion
    #         traceback.print_exc() # Print full traceback for more details

    # def tickSize(self, reqId, tickType, size):
    #     if reqId not in self.data:
    #         self.data[reqId] = {}
    #     self.data[reqId][TickTypeEnum.to_str(tickType)] = size

    def tickPrice(self, reqId, tickType, price, attrib):
        print(f"reqId: {reqId}, tickType: {TickTypeEnum.toStr(tickType)}, price: {price}, attrib: {attrib}")
      
    def tickSize(self, reqId, tickType, size):
        print(f"reqId: {reqId}, tickType: {TickTypeEnum.toStr(tickType)}, size: {size}")

    def tickOptionComputation(self, reqId, tickType, tickAttrib, impliedVol, delta, optPrice, pvDividend, gamma, vega, theta, undPrice):
         if reqId not in self.data:
            self.data[reqId] = {}
         if tickType == TickTypeEnum.MODEL_OPTION: # Model based Greeks and IV
             self.data[reqId]['modelGreeks'] = {'impliedVol': impliedVol, 'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}

    def historicalData(self, reqId, bar):
        print(bar)
        if reqId not in self.hist_data:
            self.hist_data[reqId] = []
        self.hist_data[reqId].append(bar)

    def historicalDataEnd(self, reqId, start, end):
        print(f"Historical data received for reqId {reqId}")

    def securityDefinitionOptionParameter(self, reqId, exchange, underlyingConId, tradingClass, multiplier, expirations, strikes):
        if reqId not in self.option_chain_params:
            self.option_chain_params[reqId] = []
        self.option_chain_params[reqId].append({'exchange': exchange, 'underlyingConId': underlyingConId, 'tradingClass': tradingClass, 'multiplier': multiplier, 'expirations': expirations, 'strikes': strikes})

    def marketDataType(self, reqId, marketDataType):
        if reqId not in self.data:
            self.data[reqId] = {}
        self.data[reqId]['marketDataType'] = marketDataType # Store market data type

    def get_contract_details(self, contract):
        reqId = self.nextReqId()
        self.contract_details[reqId] = []  # <--- **Correct Initialization: Empty List**
        self.reqContractDetails(reqId, contract)
        start_time = time.time()
        while reqId not in self.contract_details or not self.contract_details[reqId]: # Wait for contract details - check if list is empty, not if it's None
         time.sleep(0.1)
        if time.time() - start_time > 10: # Timeout after 10 seconds
            self.cancelContractDetails(reqId)
            del self.contract_details[reqId]
            raise TimeoutError(f"Timeout getting contract details for {contract.symbol}")
        details = self.contract_details.pop(reqId) # Get and remove data
        if not details:
          raise ValueError(f"Could not qualify contract for {contract.symbol}")
        return details[0].contract # Return the qualified contract

    # def get_market_data(self, contract):
    #     reqId = self.nextReqId()
    #     self.data[reqId] = {} # Initialize data storage for this reqId
    #     self.reqMktData(reqId, contract, '', True, False, []) # Request snapshot and model greeks
    #     start_time = time.time()
    #     while not self.has_required_ticks(reqId, {'Last', 'Bid', 'Ask', 'Close', 'Volume', 'OpenInterest', 'modelGreeks'}): # Wait for essential ticks
    #         time.sleep(5)
    #         if time.time() - start_time > 5:
    #             self.cancelMktData(reqId)
    #             self.data.pop(reqId, None) # Clean up on timeout
    #             raise TimeoutError(f"Timeout fetching market data for {contract.symbol}")
    #     market_data = self.data.pop(reqId, {}) # Get and remove data
    #     self.cancelMktData(reqId) # Cancel market data request after snapshot
    #     return market_data
    def get_ib_current_price_historical(ib: ibapi, contract: IBContract):
        """Gets the last historical closing price for a contract as a fallback."""
        print(f"Fetching historical closing price for {contract.symbol}...")
        try:
            hist_df = get_ib_historical_data(ib, contract, duration='1 D', bar_size='1 day') # Request last day's daily bar
            if not hist_df.empty:
                last_close = hist_df['Close'].iloc[-1] # Get the last closing price
                print(f"Historical closing price for {contract.symbol}: {last_close:.2f}")
                return last_close
            else:
                print(f"Warning: No historical data returned for {contract.symbol} to get closing price.")
                return None # Or raise an exception if you prefer
        except Exception as e:
            print(f"Error fetching historical closing price for {contract.symbol}: {e}")
        return None # Or raise exception

    def get_market_data(self, contract):
        reqId = self.nextReqId()
        self.data[reqId] = {}
        print(f"DEBUG: get_market_data - Requesting market data for reqId {reqId}, contract {contract.symbol}") # Debug - Request log
        self.reqMktData(reqId, contract, '456', True, False, []) # Request snapshot and Model Greeks
        start_time = time.time()
        timeout_seconds = 15 # Increased timeout for testing
        print(f"DEBUG: get_market_data - Waiting for market data, timeout = {timeout_seconds} seconds, reqId {reqId}") # Debug - Timeout log
        
        while not self.has_any_ticks(reqId): # DEBUG - Simplified check: ANY ticks received?
            time.sleep(0.1)
            elapsed_time = time.time() - start_time
            print(f"DEBUG: get_market_data - Waiting... elapsed time {elapsed_time:.2f}s, reqId {reqId}") # Debug - Wait loop log
            if elapsed_time > timeout_seconds:
                print(f"DEBUG: get_market_data - Timeout reached after {elapsed_time:.2f} seconds for reqId {reqId}, contract {contract.symbol}") # Debug - Timeout log
            #  print(f"DEBUG: get_market_data - Cancelling market data for reqId {reqId}, contract {contract.symbol}") # Debug - Cancel log - COMMENTED OUT
            # time.sleep(0.1) # Debug - Small delay before cancel - COMMENTED OUT
            # self.cancelMktData(reqId) # COMMENTED OUT
                self.data.pop(reqId, None)
                raise TimeoutError(f"Timeout fetching market data for {contract.symbol}")

            print(f"DEBUG: get_market_data - Market data received for reqId {reqId}, contract {contract.symbol} in {time.time() - start_time:.2f} seconds.") # Debug - Success log
            market_data = self.data.pop(reqId, {})
    # print(f"DEBUG: get_market_data - Cancelling market data after success for reqId {reqId}, contract {contract.symbol}") # Debug - Cancel log - COMMENTED OUT
    # time.sleep(0.1) # Debug - Small delay before cancel - COMMENTED OUT
    # self.cancelMktData(reqId) # COMMENTED OUT
        return market_data

    def get_historical_data(self, contract, duration: str = '4 M', bar_size: str = '1 day'):
        reqId = self.nextReqId()
        self.hist_data[reqId] = None # Initialize
        queryTime = datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")
        self.reqHistoricalData(6, contract, queryTime, duration, bar_size, "TRADES", 1, 1, False, [])

        start_time = time.time()
        while reqId not in self.hist_data or self.hist_data[reqId] is None:
            time.sleep(0.1)
            if time.time() - start_time > 20: # Increased timeout for historical data
                self.cancelHistoricalData(reqId)
                del self.hist_data[reqId]
                raise TimeoutError(f"Timeout fetching historical data for {contract.symbol}")

        bars = self.hist_data.pop(reqId) # Get and remove data
        if not bars:
            raise ValueError(f"No historical data returned for {contract.symbol}")

        df = pd.DataFrame([{'date': bar.date, 'Open': bar.open, 'High': bar.high, 'Low': bar.low, 'Close': bar.close, 'Volume': bar.volume} for bar in bars])
        if not df.empty:
            df['Date'] = pd.to_datetime(df['date']).dt.date # Ensure correct date format
            df.set_index('date', inplace=True)
            return df
        else:
            raise ValueError(f"Historical data conversion failed for {contract.symbol}")

    def get_option_expirations(self, stock_contract):
        reqId = self.nextReqId()
        self.option_chain_params[reqId] = None # Initialize
        self.reqSecDefOptParams(reqId, stock_contract.symbol, '', stock_contract.secType, stock_contract.conId)

        start_time = time.time()
        while reqId not in self.option_chain_params or self.option_chain_params[reqId] is None:
            time.sleep(0.1)
            if time.time() - start_time > 10:
                self.cancelSecDefOptParamsReq(reqId)
                del self.option_chain_params[reqId]
                raise TimeoutError("Timeout fetching option chain parameters.")

        params_list = self.option_chain_params.pop(reqId) # Get and remove data
        if not params_list:
            raise ValueError("No option chain parameters found.")

        all_expirations = set()
        for params in params_list:
            all_expirations.update(params['expirations'])
        return sorted(list(all_expirations))


    def get_option_chain_data(self, stock_contract, expiration: str, S):
        reqIdStart = self.nextReqId() # Starting reqId for option chain requests
        option_data = []
        strikes = self._get_option_strikes_for_expiration(stock_contract, expiration)
        if not strikes:
            raise ValueError(f"No strikes found for expiration {expiration}")

        contracts = []
        for strike in strikes:
            for right in ['C', 'P']:
                option_contract = IBContract()
                option_contract.symbol = stock_contract.symbol
                option_contract.secType = "OPT"
                option_contract.exchange = "SMART" # Or "BEST"
                option_contract.currency = "USD"
                option_contract.lastTradeDateOrContractMonth = expiration
                option_contract.strike = strike
                option_contract.right = right
                option_contract.tradingClass = stock_contract.symbol # Important for option chain filtering
                contracts.append(option_contract)

        print(f"Requesting market data for {len(contracts)} option contracts...")
        for contract in contracts:
            reqId = self.nextReqId()
            self.data[reqId] = {} # Initialize data for each option contract
            self.reqMktData(reqId, contract, '', True, False, []) # Request market data

        wait_time = 10 # seconds to wait for all option data
        start_time = time.time()
        received_count = 0
        expected_count = len(contracts)

        while received_count < expected_count and time.time() - start_time < wait_time:
            received_count = sum(1 for reqId in self.data if self.has_required_ticks(reqId, {'Bid', 'Ask', 'Last', 'Close', 'Volume', 'OpenInterest', 'modelGreeks'}))
            time.sleep(0.2) # Polling interval

        print(f"Received data for {received_count} of {expected_count} option contracts.")

        for reqId in list(self.data.keys()): # Iterate over a copy to allow deletion
            if self.has_required_ticks(reqId, {'Bid', 'Ask', 'Last', 'Close', 'Volume', 'OpenInterest', 'modelGreeks'}):
                market_data = self.data.pop(reqId)
                contract = None # Need to reconstruct contract info from reqId - not directly available in tick callbacks
                # Very inefficient, but no direct way to link reqId back to contract in ibapi.
                # In real application, consider managing reqId to contract mapping externally if performance is critical.
                for temp_contract in contracts:
                    temp_reqId = self.make_reqId_for_contract(temp_contract) # Dummy reqId creation - this is not the actual reqId used by TWS
                    if temp_reqId == reqIdStart: # Very basic, incorrect matching - needs proper reqId management
                        contract = temp_contract # Potentially incorrect contract association
                        break # Exit after finding *a* contract - may not be the *correct* one

                if contract: # Only process if we *think* we found a matching contract
                    iv = market_data.get('modelGreeks', {}).get('impliedVol', np.nan)
                    oi = market_data.get('OpenInterest', 0)
                    volume = market_data.get('Volume', 0)
                    last_price = market_data.get('Last', market_data.get('Close', np.nan))

                    option_data.append({
                        'contractSymbol': contract.localSymbol if hasattr(contract, 'localSymbol') else f"{contract.symbol}{expiration}{contract.right}{strike}", # Approximation
                        'strike': contract.strike,
                        'type': 'call' if contract.right == 'C' else 'put',
                        'lastPrice': last_price if pd.notna(last_price) else np.nan,
                        'bid': market_data.get('Bid', np.nan),
                        'ask': market_data.get('Ask', np.nan),
                        'change': market_data.get('Last', 0) - market_data.get('Close', 0), # Approximation
                        'percentChange': (market_data.get('Last', 0) / market_data.get('Close', 1) - 1) * 100, # Approximation
                        'volume': volume,
                        'openInterest': oi,
                        'impliedVolatility': iv,
                        'inTheMoney': (contract.right == 'C' and contract.strike < S) or (contract.right == 'P' and contract.strike > S),
                        'contractSize': 'REGULAR',
                        'currency': 'USD'
                    })
                else:
                    print(f"Warning: No contract match found for reqId {reqId}. Data potentially discarded.")
            else:
                self.data.pop(reqId, None) # Clean up incomplete data

        chain_df = pd.DataFrame(option_data)

        chain_df['midPrice'] = (chain_df['bid'] + chain_df['ask']) / 2.0
        missing_iv_filter = chain_df['impliedVolatility'].isna()
        print(f"Attempting to calculate IV for {missing_iv_filter.sum()} options where IB IV is missing...")
        chain_df.loc[missing_iv_filter, 'impliedVolatility'] = chain_df[missing_iv_filter].apply(
            lambda row: implied_volatility(row['type'], row['midPrice'], S, row['strike'], T, r, model='bs'), # Use BS fallback
            axis=1
        )
        print(f"IV NaN count after fallback calculation: {chain_df['impliedVolatility'].isna().sum()}")

        return chain_df.dropna(subset=['impliedVolatility'])

    def _get_option_strikes_for_expiration(self, stock_contract, expiration):
        params_list = self.option_chain_params.get(list(self.option_chain_params.keys())[-1], []) # Get last reqId params - fragile!
        if not params_list: return [] # No params received

        target_chain_params = None
        for params in params_list:
            if expiration in params['expirations']:
                target_chain_params = params
                break
        if not target_chain_params: return [] # Expiration not found

        return sorted(list(target_chain_params['strikes']))

    def make_reqId_for_contract(self, contract):
        # Dummy reqId creation - replace with actual reqId management if needed
        return hash(str(contract)) % 10000 # Simple hash for demonstration - NOT robust for real reqId tracking



def get_ib_stock_contract(ib: ibapi, symbol: str):
    """Qualifies and returns an IB Stock contract using ibapi."""
    contract = IBContract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    what_to_show = 'TRADES'
    return ib.get_contract_details(contract)
    
def get_ib_current_price_historical(ib: ibapi, contract: IBContract):
        """Gets the last historical closing price for a contract as a fallback."""
        print(f"Fetching historical closing price for {contract.symbol}...")
        try:
            hist_df = get_ib_historical_data(ib, contract, duration='1 D', bar_size='1 day') # Request last day's daily bar
            if not hist_df.empty:
                last_close = hist_df['Close'].iloc[-1] # Get the last closing price
                print(f"Historical closing price for {contract.symbol}: {last_close:.2f}")
                return last_close
            else:
                print(f"Warning: No historical data returned for {contract.symbol} to get closing price.")
                return None # Or raise an exception if you prefer
        except Exception as e:
            print(f"Error fetching historical closing price for {contract.symbol}: {e}")
        return None # Or raise exception

def get_ib_current_price(ib: ibapi, contract: IBContract):
    """Gets the last traded price for a contract using ibapi."""
    market_data = ib.get_market_data(contract)
    last_price = market_data.get('Last', market_data.get('Close')) # Prefer Last if available
    if pd.isna(last_price):
        raise ValueError(f"Could not fetch current price for {contract.symbol}")
    return last_price

def get_ib_historical_data(ib: ibapi, contract: IBContract, duration: str = '4 M', bar_size: str = '1 day', ):
    """Gets historical OHLCV data and returns as pandas DataFrame using ibapi."""
    print(f"Fetching historical data ({duration} {bar_size})...")
    #app.reqHistoricalData(app.nextId(), mycontract, "20240523 16:00:00 US/Eastern", "1 D", "1 hour", "TRADES", 1, 1, False, [])
    df = ib.get_historical_data(6, contract, "", duration=duration, bar_size=bar_size, "TRADES", True, 6, False, [])
    return df


def get_ib_option_expirations(ib: ibapi, stock_contract: IBContract):
    """Gets available option expiration dates using ibapi."""
    print("Fetching option parameters (expirations/strikes)...")
    return ib.get_option_expirations(stock_contract)


def get_ib_option_chain_data(ib: ibapi, stock_contract: IBContract, expiration: str, S):
    """Fetches Calls/Puts market data for a specific expiration using ibapi."""
    print(f"Fetching option chain data for {expiration}...")
    chain_df = ib.get_option_chain_data(stock_contract, expiration, S)
    return chain_df


def compute_recommendation(ib: ibapi, ticker: str): # Pass IB API client
    """Computes recommendation based on refined metrics using IBKR data."""
    try:
        ticker = ticker.strip().upper()
        if not ticker: raise ValueError("No stock symbol provided.")
        print(f"\n--- Processing {ticker} ---")

        # --- Get IB Contract ---
        stock_contract = get_ib_stock_contract(ib, ticker)
        S = None #get_ib_current_price(ib, stock_contract) # Get current price via IB

        try:
            #S = get_ib_current_price(ib, stock_contract) # Try live price first
            print("Attempting to get historical closing price as fallback...")
            S_historical = get_ib_current_price_historical(ib, stock_contract)
            #print(f"Live current price fetched: S={S:.2f}")
        except Exception as live_price_error:
            print(f"Warning: Error fetching price: {live_price_error}")
            
            # Fallback to historical
            if S_historical is not None:
                S = S_historical
                print(f"Using historical closing price as fallback: S={S:.2f}")
            else:
                raise ValueError("Could not fetch current or historical stock price.") from live_price_error # Re-raise original error if historical also fails

        if S is None: # Should not happen if the above logic is correct, but as a safety check
            raise ValueError("Failed to determine stock price (live or historical).")

        q = 0.0 # Default dividend yield
        r = get_risk_free_rate()
        print(f"S={S:.2f}, q={q:.4%}, r={r:.4%}")

        # --- Get ~30d Expiration ---
        try:
            available_expirations = get_ib_option_expirations(ib, stock_contract)
            if not available_expirations: raise ValueError("No options found.")
            today = date.today(); exp_30d_info = None
            for exp_str in sorted(available_expirations):
                 exp_date = datetime.datetime.strptime(exp_str, '%Y%m%d').date() # IB format YYYYMMDD
                 days_out = (exp_date - today).days
                 if days_out >= 20: exp_30d_info = {'date_str': exp_str, 'days_out': days_out}; break
            if exp_30d_info is None: raise ValueError("No expiration >= 20 days out.")
            exp_date_30d = exp_30d_info['date_str']; T_30d = exp_30d_info['days_out'] / 365.25
            print(f"Selected ~30d expiration: {exp_date_30d} (T={T_30d:.4f})")
        except Exception as e: raise ValueError(f"Error processing dates: {e}")

        # --- Fetch and Process ~30d Chain using IBKR ---
        try:
            # Use the new IBKR function
            chain_df_30d = get_ib_option_chain_data(ib, stock_contract, exp_date_30d, S)
            if chain_df_30d.empty: raise ValueError(f"No valid options after cleaning for {exp_date_30d}.")

            # Calculate Greeks (still using BS for speed/simplicity)
            chain_df_30d = calculate_chain_iv_greeks(chain_df_30d, S, r, T_30d, q, pricing_model='bs')
            if chain_df_30d['marketIV'].isna().all(): raise ValueError("All IV calculations failed.")
            F_30d = chain_df_30d['F'].iloc[0]
        except Exception as e: raise ValueError(f"Error fetching/processing chain for {exp_date_30d}: {e}")

        # --- Fit SVI Model ---
        svi_params_30d = fit_svi_model_single_exp(chain_df_30d, S, r, T_30d, F_30d, SVI_FIT_THRESHOLDS)
        if svi_params_30d is None: raise ValueError("SVI model fit failed for ~30d expiration.")

        # --- Calculate Metrics ---
        iv30_svi = np.nan; rv30_yz = np.nan; ts_slope_svi = np.nan; avg_volume = 0; expected_move_pct = "N/A"
        try:
            w_atm_30d = svi_total_variance(0.0, svi_params_30d)
            sigma_atm_fitted = np.sqrt(np.maximum(w_atm_30d, 1e-9) / T_30d); iv30_svi = sigma_atm_fitted
            if pd.isna(iv30_svi): raise ValueError("NaN SVI IV30.")
            print(f"SVI Estimated IV30: {iv30_svi:.4f}")
        except Exception as e: print(f"Warning: Error calculating SVI IV30: {e}")
        try:
            # Use IBKR historical data
            price_history = get_ib_historical_data(ib, stock_contract, duration='4 M')
            rv30_yz = yang_zhang(price_history, window=30, return_last_only=True)
            print(f"Yang-Zhang RV30: {rv30_yz:.4f}")
            # Calculate Avg Volume from same history
            avg_volume = price_history["Volume"].rolling(window=30).mean().dropna().iloc[-1]
            print(f"Avg Stock Volume (30d): {avg_volume:.0f}")
        except Exception as e: print(f"Warning: Error calculating RV/AvgVol: {e}")
        try:
            skew_metrics = analyze_svi_skew_slope(svi_params_30d, F_30d, T_30d)
            ts_slope_svi = skew_metrics.get('atm_slope', np.nan)
            if pd.isna(ts_slope_svi): raise ValueError("NaN SVI slope.")
            print(f"SVI ATM Slope (~30d): {ts_slope_svi:.4f}")
        except Exception as e: print(f"Warning: Error calculating SVI slope: {e}")
        try:
            # Calculate Expected Move from fetched chain data
            atm_strike_idx = (np.abs(chain_df_30d['strike'] - S)).idxmin()
            atm_strike = chain_df_30d.loc[atm_strike_idx, 'strike']
            atm_call_row = chain_df_30d[(chain_df_30d['strike'] == atm_strike) & (chain_df_30d['type'] == 'call')]
            atm_put_row = chain_df_30d[(chain_df_30d['strike'] == atm_strike) & (chain_df_30d['type'] == 'put')]
            if not atm_call_row.empty and not atm_put_row.empty:
                 call_mid = atm_call_row.iloc[0]['midPrice']; put_mid = atm_put_row.iloc[0]['midPrice']
                 if pd.notna(call_mid) and pd.notna(put_mid):
                      straddle_price = call_mid + put_mid
                      expected_move_pct = f"{straddle_price / S * 100:.2f}%" if S > 0 else "N/A"
            print(f"Approx Expected Move ({exp_date_30d}): {expected_move_pct}")
        except Exception: print("Warning: Could not calculate expected move.")

        # --- Apply Thresholds ---
        iv30_rv30_ratio = iv30_svi / rv30_yz if pd.notna(iv30_svi) and pd.notna(rv30_yz) and rv30_yz > 1e-6 else np.inf
        print(f"IV30/RV30 Ratio: {iv30_rv30_ratio:.2f}")
        avg_volume_bool = avg_volume >= AVG_VOLUME_THRESHOLD
        iv30_rv30_bool = iv30_rv30_ratio >= IV_RV_THRESHOLD
        ts_slope_bool = pd.notna(ts_slope_svi) and ts_slope_svi <= MODIFIED_SLOPE_THRESHOLD
        print(f"Threshold Check: Volume Pass={avg_volume_bool}, IV/RV Pass={iv30_rv30_bool}, Slope Pass={ts_slope_bool} (Threshold={MODIFIED_SLOPE_THRESHOLD:.3f})")

        return {"avg_volume": avg_volume_bool, "iv30_rv30": iv30_rv30_bool, "ts_slope_0_45": ts_slope_bool, "expected_move": expected_move_pct, "raw_avg_volume": avg_volume, "iv30_svi": iv30_svi, "rv30_yz": rv30_yz, "ts_slope_svi": ts_slope_svi}

    except ValueError as ve: print(f"ValueError in compute_recommendation: {ve}"); raise ve
    except Exception as e: print(f"Unexpected Error: {e}"); traceback.print_exc(); raise ValueError(f"Error processing {ticker}: {str(e)}")


# --- GUI Functions ---
def main_gui():
    # --- IB Connection ---
    ib = IBapi()
    ib.connect(IB_HOST, IB_PORT, IB_CLIENT_ID)
    thread = threading.Thread(target=ib.run, daemon=True) # Start IB API client in thread
    thread.start()
    time.sleep(1) # Give time to connect

    if not ib.isConnected():
        sg.popup_error(f"Could not connect to IBKR TWS/Gateway on {IB_HOST}:{IB_PORT}\n Please ensure TWS/Gateway is running and API is enabled.", title="Connection Error")
        ib.disconnect()
        return # Exit GUI if connection fails
    print("IBKR Connection Successful.")


    # --- GUI Layout and Loop (Mostly unchanged) ---
    main_layout = [[sg.Text("Enter Stock Symbol:"), sg.Input(key="stock", size=(20, 1), focus=True)], [sg.Button("Submit", bind_return_key=True), sg.Button("Exit")], [sg.Text("", key="recommendation", size=(60, 1))]] # Increased size
    window = sg.Window("Earnings Position Checker v2 (SVI Metrics - IBKR)", main_layout)

    while True:
        event, values = window.read();
        if event in (sg.WINDOW_CLOSED, "Exit"): break
        if event == "Submit":
            window["recommendation"].update("Processing...")
            stock_ticker_input = values.get("stock", "")
            loading_layout = [[sg.Text(f"Loading data for {stock_ticker_input}...", key="loading", justification="center")]]; loading_window = sg.Window("Loading", loading_layout, modal=True, finalize=True, size=(300, 100)) # Adjusted size
            result_holder = {}

            def worker(ib_conn, ticker_sym): # Pass IB connection to worker
                try: result_holder["result"] = compute_recommendation(ib_conn, ticker_sym) # Pass ib to function
                except Exception as e: result_holder["error"] = str(e)

            # Pass the established ib connection to the thread
            thread = threading.Thread(target=worker, args=(ib, stock_ticker_input), daemon=True); thread.start()

            while thread.is_alive(): event_load, _ = loading_window.read(timeout=100);
            if event_load == sg.WINDOW_CLOSED: break # Allow closing loading window
            thread.join(timeout=0.1)
            loading_window.refresh() # Keep GUI responsive

            loading_window.close()

            if "error" in result_holder: window["recommendation"].update(f"Error: {result_holder['error']}")
            elif "result" in result_holder:
                window["recommendation"].update("")
                result = result_holder["result"]
                # ... (Access results and determine title/color - SAME AS BEFORE) ...
                avg_volume_bool = result["avg_volume"]; iv30_rv30_bool = result["iv30_rv30"]; ts_slope_bool = result["ts_slope_0_45"]; expected_move = result["expected_move"]
                avg_volume_val = result.get("raw_avg_volume", 0); iv30_val = result.get("iv30_svi", np.nan); rv30_val = result.get("rv30_yz", np.nan); slope_val = result.get("ts_slope_svi", np.nan)
                iv_rv_ratio = iv30_val / rv30_val if pd.notna(iv30_val) and pd.notna(rv30_val) and rv30_val > 1e-6 else np.nan
                if avg_volume_bool and iv30_rv30_bool and ts_slope_bool: title = "Recommended"; title_color = "#006600"
                elif ts_slope_bool and ((avg_volume_bool and not iv30_rv30_bool) or (iv30_rv30_bool and not avg_volume_bool)): title = "Consider"; title_color = "#ff9900"
                else: title = "Avoid"; title_color = "#800000"

                # ... (Build result layout - SAME AS BEFORE, using correct variables) ...
                result_layout = [
                    [sg.Text(title, text_color=title_color, font=("Helvetica", 16))],
                    [sg.Text(f"Avg Volume (>={AVG_VOLUME_THRESHOLD/1e6:.1f}M): {'PASS' if avg_volume_bool else 'FAIL'} ({avg_volume_val:,.0f})", text_color="#006600" if avg_volume_bool else "#800000")],
                    [sg.Text(f"IV30/RV30 (>={IV_RV_THRESHOLD:.2f}): {'PASS' if iv30_rv30_bool else 'FAIL'} ({iv_rv_ratio:.2f})", text_color="#006600" if iv30_rv30_bool else "#800000")],
                    [sg.Text(f"SVI Slope (<={MODIFIED_SLOPE_THRESHOLD:.3f}): {'PASS' if ts_slope_bool else 'FAIL'} ({slope_val:.4f})", text_color="#006600" if ts_slope_bool else "#800000")],
                    [sg.Text(f"Approx Expected Move: {expected_move}", text_color="blue")],
                    [sg.Text(f"(IV30={iv30_val:.2%}, RV30={rv30_val:.2%})", font=("Helvetica", 8))],
                    [sg.Button("OK")]
                ]
                result_window = sg.Window("Recommendation", result_layout, modal=True, finalize=True, size=(350, 220));
                while True: event_result, _ = result_window.read(timeout=100);
                if event_result in (sg.WINDOW_CLOSED, "OK"): break
                result_window.close()
            else: window["recommendation"].update("Processing finished, but no result returned.")
    # --- Disconnect IB ---
    print("Disconnecting from IBKR...")
    ib.disconnect()
    window.close()


def gui():
    main_gui()

if __name__ == "__main__":
    gui()