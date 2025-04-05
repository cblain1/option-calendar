import datetime
import time
import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def get_risk_free_rate():
    return 0.043 #todo placedolder

# --- Helper: Date Formatting ---
def format_ib_date(date_obj):
    """Formats python date/datetime to YYYYMMDD string for IB."""
    if isinstance(date_obj, (datetime.datetime, datetime.date)):
        return date_obj.strftime('%Y%m%d')
    return str(date_obj) # Assume already string if not date/datetime

def format_output_date(date_str_ib):
     """Formats YYYYMMDD string from IB to YYYY-MM-DD."""
     try:
         return datetime.datetime.strptime(str(date_str_ib), '%Y%m%d').strftime('%Y-%m-%d')
     except ValueError:
         warnings.warn(f"Could not parse IB date string: {date_str_ib}. Returning original.")
         return date_str_ib


# --- Core API Data Fetching Functions ---

def fetch_option_chains_list(app, ticker: str):
    """
    Fetches the list of option chain parameters (exchanges, expirations, strikes) for a ticker.

    Args:
        app: Connected IB API client instance.
        ticker (str): The underlying stock ticker symbol.

    Returns:
        list: A list of dictionaries, where each dict represents one exchange
              and contains 'exchange', 'expirations' (list, YYYYMMDD),
              and 'strikes' (list, float). Returns empty list on failure.
    """
    if not app.isConnected():
        print("Error: IB is not connected.")
        return []

    stock_contract = Stock(ticker, 'SMART', 'USD')
    try:
        # Attempt to qualify the contract first to get the primary exchange if needed,
        # although reqSecDefOptParams usually works with SMART.
        details = app.reqContractDetails(stock_contract)
        if not details:
             print(f"Error: Could not get contract details for {ticker}")
             return []
        underlying_conId = details[0].contract.conId
        # print(f"DEBUG: Found conId {underlying_conId} for {ticker}")

        # Fetch option parameters
        chains_params = app.reqSecDefOptParams(ticker, '', 'STK', underlying_conId)
        if not chains_params:
             print(f"Error: No option parameters found for {ticker} (conId: {underlying_conId})")
             return []

        # Convert to the desired format
        output_list = []
        for params in chains_params:
            output_list.append({
                'exchange': params.exchange,
                'underlyingConId': params.underlyingConId,
                'tradingClass': params.tradingClass,
                'multiplier': params.multiplier,
                'expirations': sorted(list(params.expirations)), # Ensure sorted list
                'strikes': sorted(list(params.strikes))        # Ensure sorted list
            })
        return output_list

    except Exception as e:
        print(f"Error fetching option chain list for {ticker}: {e}")
        return []

def get_current_underlying_price(app, ticker: str):
    """
    Gets the current market price using Ticker.marketPrice() for the underlying stock.

    Args:
        app: Connected IB API client instance.
        ticker (str): The underlying stock ticker symbol.

    Returns:
        float or None: The current price, or None if fetching fails.
    """
    if not app.isConnected():
        print("Error: IB is not connected.")
        return None

    stock_contract = Stock(ticker, 'SMART', 'USD')
    try:
        # Qualify is good practice even if not strictly needed for price sometimes
        qualified_contracts = app.qualifyContracts(stock_contract)
        if not qualified_contracts:
             print(f"Error: Cannot qualify contract for stock {ticker}")
             return None
        qualified_stock = qualified_contracts[0] # Use the qualified contract

        # Request ticker data
        app.reqMktData(qualified_stock, '', False, False) # Request streaming update, snapshot=False
        app.sleep(1.0) # Allow time for data to arrive, might need adjustment/better handling
        ticker_data: Ticker = app.ticker(qualified_stock) # Retrieve the ticker object
        app.cancelMktData(qualified_stock) # Clean up the request

        if ticker_data is None:
             print(f"Warning: Failed to retrieve ticker object for {ticker} after reqMktData.")
             return None

        # --- Use marketPrice() as preferred method ---
        price = ticker_data.marketPrice() # This handles NaN internally apparently

        # Additional checks for validity (e.g., positive price)
        if not isinstance(price, (float, int)) or price <= 0:
             print(f"Warning: Invalid marketPrice() ({price}) received for {ticker}. Trying last/close.")
             # Fallback logic (optional, based on marketPrice reliability)
             price = ticker_data.last
             if not isinstance(price, (float, int)) or np.isnan(price) or price <= 0:
                 price = ticker_data.close
             if not isinstance(price, (float, int)) or np.isnan(price) or price <= 0:
                 print(f"Warning: Could not get valid marketPrice, Last, or Close price for {ticker}. Ticker state: {ticker_data}")
                 return None

        return float(price)
    except Exception as e:
        print(f"Error fetching underlying price for {ticker}: {e}")
        return None

def get_nearby_expirations(expirations_list, days_out=60):
    """
    Filters a list of expiration date strings (YYYYMMDD) to find those
    within a specified number of days from today. Returns YYYY-MM-DD format.
    (Using 60 days as requested)
    """
    filtered_expirations = []
    if not expirations_list:
        print("DEBUG: Input expirations_list is empty.")
        return filtered_expirations

    today = datetime.date.today()
    end_date = today + datetime.timedelta(days=days_out)
    input_format = '%Y%m%d'
    output_format = '%Y-%m-%d'

    # print(f"DEBUG: Filtering dates between {today.strftime(output_format)} and {end_date.strftime(output_format)}")

    for exp_str in expirations_list:
        try:
            exp_date = datetime.datetime.strptime(str(exp_str), input_format).date()
            # Filter: expiry must be today or later, AND no later than end_date
            if today <= exp_date <= end_date:
                formatted_date = exp_date.strftime(output_format)
                filtered_expirations.append(formatted_date)
        except (ValueError, TypeError): # Catch parsing errors and non-strings
            print(f"Warning: Could not parse/process date string: {exp_str}")
            continue

    return sorted(filtered_expirations) # Return sorted list


def fetch_detailed_option_chain(app, ticker: str, expiration_date: str, underlying_price: float, risk_free_rate: float, exchange: str = 'SMART'):
    """
    Fetches detailed option data (strike, bid, ask) and calculates Implied Volatility
    using the provided 'implied_volatility' function for a specific expiration date.

    Args:
        app: Connected IB API client instance.
        ticker (str): The underlying stock ticker symbol.
        expiration_date (str): The target expiration date in 'YYYY-MM-DD' format.
        underlying_price (float): Current price of the underlying stock (S).
        risk_free_rate (float): Annualized risk-free interest rate (r).
        exchange (str, optional): The exchange to use. Defaults to 'SMART'.

    Returns:
        dict or None: A dictionary containing two pandas DataFrames: 'calls' and 'puts'.
                      Returns None if fetching fails or no data is found.
                      DataFrames contain columns: 'strike', 'impliedVolatility' (calculated),
                      'bid' (fetched), 'ask' (fetched).
    """
    if not app.isConnected():
        print("Error: IB is not connected.")
        return None

    try:
        # 1. Calculate Time to Expiration (T) in years
        today = datetime.date.today()
        try:
            exp_date_obj = datetime.datetime.strptime(expiration_date, "%Y-%m-%d").date()
            # Add small epsilon to avoid T=0 on expiration day if needed by IV calc
            time_to_expiry_days = (exp_date_obj - today).days
            if time_to_expiry_days < 0:
                 print(f"Warning: Expiration date {expiration_date} is in the past. Skipping.")
                 return {'calls': pd.DataFrame(columns=['strike', 'impliedVolatility', 'bid', 'ask']),
                         'puts': pd.DataFrame(columns=['strike', 'impliedVolatility', 'bid', 'ask'])}
            # Use 365.25 for slightly more accuracy including leap years
            T = max(time_to_expiry_days / 365.25, 1e-6) # Time in years, ensure > 0
        except ValueError:
             print(f"Error: Could not parse expiration date: {expiration_date}")
             return None

        # 2. Get chain parameters (strikes, trading class)
        # (Slightly inefficient to call this every time, could be cached/passed)
        chain_list = fetch_option_chains_list(app, ticker) # Assumes this function exists
        target_chain = None
        for chain in chain_list:
            tc = chain.get('tradingClass', ticker)
            if chain.get('exchange') == exchange:
                 if tc == ticker: target_chain = chain; break
                 elif target_chain is None: target_chain = chain
        if not target_chain:
             print(f"Error: Cannot find chain parameters for {ticker} on {exchange}")
             return None
        strikes = target_chain.get('strikes', [])
        if not strikes:
             print(f"Error: No strikes found for {ticker} on {exchange}")
             return None
        trading_class = target_chain.get('tradingClass', ticker)
        ib_expiration_str = expiration_date.replace('-', '')
        if ib_expiration_str not in target_chain.get('expirations', []):
             print(f"Error: Expiration {expiration_date} not found in parameters for {ticker} on {exchange}/{trading_class}.")
             return None

        # 3. Create Option contract objects
        contracts_to_fetch = []
        for strike in strikes:
            for right in ['C', 'P']:
                contracts_to_fetch.append(
                    Option(ticker, ib_expiration_str, float(strike), right, exchange, tradingClass=trading_class)
                )
        if not contracts_to_fetch: return None

        # 4. Qualify contracts
        qualified_contracts = app.qualifyContracts(*contracts_to_fetch)
        print(f"DEBUG: Attempting to fetch market data for {len(qualified_contracts)} qualified contracts for {expiration_date}...")
        if not qualified_contracts:
             return {'calls': pd.DataFrame(columns=['strike', 'impliedVolatility', 'bid', 'ask']),
                     'puts': pd.DataFrame(columns=['strike', 'impliedVolatility', 'bid', 'ask'])}

        # 5. Fetch Ticker data (Bid, Ask, Last/Close)
        tickers_data = []
        batch_size = 100
        for i in range(0, len(qualified_contracts), batch_size):
             batch = qualified_contracts[i:i+batch_size]
             print(f"DEBUG: Requesting tickers batch {i//batch_size + 1} (size {len(batch)})")
             # *** Ensure your reqTickers or equivalent gets Bid, Ask, Last/Close ***
             tickers_data.extend(app.reqTickers(*batch))
             app.sleep(0.5)
        app.sleep(1.5) # Adjust wait time

        # 6. Process the ticker data
        call_data = []
        put_data = []
        processed_conIds = set()
        S = underlying_price # Use passed underlying price
        r = risk_free_rate # Use passed risk-free rate

        for t in tickers_data:
            if not t or not t.contract or t.contract.conId in processed_conIds: continue
            processed_conIds.add(t.contract.conId)

            # --- Get required data from Ticker ---
            K = float(t.contract.strike)
            right_char = t.contract.right # 'C' or 'P'

            bid = getattr(t, 'bid', None)
            ask = getattr(t, 'ask', None)
            # Prioritize 'last' price, fallback to 'close' for IV calculation
            market_price = getattr(t, 'last', None)
            if not isinstance(market_price, (float, int)) or np.isnan(market_price) or market_price < 0: # Check validity of last
                 market_price = getattr(t, 'close', None) # Fallback to close

            # --- Clean fetched values ---
            bid = float(bid) if isinstance(bid, (float, int)) and not np.isnan(bid) and bid >= 0 else np.nan
            ask = float(ask) if isinstance(ask, (float, int)) and not np.isnan(ask) and ask >= 0 else np.nan
            market_price = float(market_price) if isinstance(market_price, (float, int)) and not np.isnan(market_price) and market_price >= 0 else np.nan

            # --- Calculate Implied Volatility ---
            calculated_iv = np.nan # Default to NaN
            if not np.isnan(market_price) and market_price > 1e-6 : # Check if market price is valid and non-zero
                try:
                     # Call the provided IV function
                     calculated_iv = implied_volatility(
                         option_type=right_char,
                         market_price=market_price,
                         S=S, K=K, T=T, r=r
                     )
                     # Ensure calculated IV is reasonable (optional, depends on IV function)
                     if not isinstance(calculated_iv, (float, int)) or np.isnan(calculated_iv) or calculated_iv < 0.001 or calculated_iv > 5.0: # e.g., 0.1% to 500%
                          # print(f"Warning: Unreasonable calculated IV ({calculated_iv}) for {t.contract.localSymbol}. Setting to NaN.")
                          calculated_iv = np.nan
                except Exception as iv_e:
                    print(f"Warning: Error calculating IV for {t.contract.localSymbol}: {iv_e}. MarketPrice={market_price}, S={S}, K={K}, T={T}, r={r}")
                    calculated_iv = np.nan
            # else:
                # print(f"DEBUG: Skipping IV calculation for {t.contract.localSymbol} due to invalid market price: {market_price}")


            option_info = {
                'strike': K,
                'impliedVolatility': calculated_iv, # Use calculated IV
                'bid': bid,                   # Use fetched Bid
                'ask': ask                    # Use fetched Ask
            }

            if right_char == 'C':
                call_data.append(option_info)
            elif right_char == 'P':
                put_data.append(option_info)

        # 7. Create DataFrames
        calls_df = pd.DataFrame(call_data).sort_values(by='strike').reset_index(drop=True) if call_data else pd.DataFrame(columns=['strike', 'impliedVolatility', 'bid', 'ask'])
        puts_df = pd.DataFrame(put_data).sort_values(by='strike').reset_index(drop=True) if put_data else pd.DataFrame(columns=['strike', 'impliedVolatility', 'bid', 'ask'])

        print(f"DEBUG: Found {len(calls_df)} calls and {len(puts_df)} puts for {expiration_date}.")
        if calls_df.empty and puts_df.empty:
             print(f"Warning: No valid call or put data processed for {ticker} {expiration_date}")

        return {'calls': calls_df, 'puts': puts_df}

    except Exception as e:
        print(f"CRITICAL Error fetching detailed option chain for {ticker} {expiration_date}: {e}")
        import traceback
        traceback.print_exc()
        return None

    

def fetch_historical_price_data(app, ticker: str, period: str = '3mo'):
    """
    Fetches historical OHLCV data for the underlying stock.

    Args:
        app: Connected IB API client instance.
        ticker (str): The underlying stock ticker symbol.
        period (str, optional): The duration string (e.g., '3mo', '1y'). Defaults to '3mo'.
                                Needs mapping to IBKR 'durationStr'.

    Returns:
        pd.DataFrame or None: DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume'
                              columns and datetime index. Returns None on failure.
    """
    if not app.isConnected():
        print("Error: IB is not connected.")
        return None

    stock_contract = Stock(ticker, 'SMART', 'USD')
    try:
        # Qualify contract
        cds = app.reqContractDetails(stock_contract)
        if not cds:
             print(f"Error: Cannot find contract details for stock {ticker}")
             return None
        qualified_stock = cds[0].contract

        # Map period to IBKR durationStr and barSizeSetting (basic mapping)
        # More sophisticated mapping might be needed for different periods
        duration_map = {
            '1mo': ('1 M', '1 day'),
            '3mo': ('3 M', '1 day'),
            '6mo': ('6 M', '1 day'),
            '1y': ('1 Y', '1 day'),
            '2y': ('2 Y', '1 day'),
             # Add more mappings as needed
        }
        if period not in duration_map:
             print(f"Error: Unsupported period '{period}'. Supported: {list(duration_map.keys())}")
             return None
        durationStr, barSizeSetting = duration_map[period]

        # Request historical data
        bars = app.reqHistoricalData(
            qualified_stock,
            endDateTime='', # Empty means current time
            durationStr=durationStr,
            barSizeSetting=barSizeSetting,
            whatToShow='TRADES',
            useRTH=True, # Regular Trading Hours
            formatDate=1 # Return as datetime objects
        )

        if not bars:
            print(f"Warning: No historical bars returned for {ticker} period {period}")
            return None

        # Convert to DataFrame
        df = util.df(bars)
        if df is None or df.empty:
             print(f"Warning: Historical data conversion to DataFrame failed for {ticker}")
             return None

        # Rename columns to match expected format (Open, High, Low, Close, Volume)
        # IB columns are typically: date, open, high, low, close, volume, average, barCount
        df = df.rename(columns={
            'date': 'Date', # Will set as index later
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })

        # Select and ensure correct columns exist
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
             print(f"Error: Historical data DataFrame missing required columns after rename. Columns found: {df.columns.tolist()}")
             return None

        df = df[required_cols]
        df = df.set_index('Date') # Set datetime index

        return df

    except Exception as e:
        print(f"Error fetching historical price data for {ticker}: {e}")
        return None