import datetime

# Assume 'chains' is the list of OptionChain objects provided by the user
# chains = [OptionChain(exchange='NASDAQOM', ...), OptionChain(exchange='AMEX', ...), ...]

def get_nearby_expirations(expirations_list, days_out=45):
    """
    Filters a list of expiration date strings (YYYYMMDD) to find those
    within a specified number of days from today.

    Args:
        expirations_list (list): A list of strings in 'YYYYMMDD' format.
        days_out (int): The maximum number of days from today to include.

    Returns:
        list: A list of expiration date strings in 'YYYY-MM-DD' format
              that fall within the specified range [today, today + days_out].
    """
    filtered_expirations = []
    if not expirations_list:
        print("DEBUG: Input expirations_list is empty.")
        return filtered_expirations

    today = datetime.date.today()
    end_date = today + datetime.timedelta(days=days_out)
    input_format = '%Y%m%d'
    output_format = '%Y-%m-%d'

    print(f"DEBUG: Filtering dates between {today.strftime(output_format)} and {end_date.strftime(output_format)}")

    for exp_str in expirations_list:
        try:
            exp_date = datetime.datetime.strptime(exp_str, input_format).date()
            if today <= exp_date <= end_date:
                formatted_date = exp_date.strftime(output_format)
                filtered_expirations.append(formatted_date)
        except ValueError:
            print(f"Warning: Could not parse date string: {exp_str}")
            continue
        except TypeError:
             print(f"Warning: Encountered non-string expiration data: {exp_str}")
             continue

    return filtered_expirations

# --- Usage ---

# Make sure 'chains' holds the list of OptionChain objects/dicts from your API response

smart_chain_expirations = [] # Initialize to empty list

# 1. Find the 'SMART' exchange entry and extract its expirations
if chains and isinstance(chains, list):
    found_smart = False
    for chain_item in chains:
        # Accessing attributes - assuming they are objects like OptionChain(...)
        try:
            # Check if the item has an 'exchange' attribute before accessing
            if hasattr(chain_item, 'exchange') and chain_item.exchange == 'SMART':
                print(f"DEBUG: Found chain item for exchange: {chain_item.exchange}")
                # Check if the item has an 'expirations' attribute
                if hasattr(chain_item, 'expirations'):
                    smart_chain_expirations = chain_item.expirations
                    found_smart = True
                    print(f"DEBUG: Extracted {len(smart_chain_expirations)} expirations for SMART.")
                    break # Stop searching once 'SMART' is found
                else:
                    print(f"Warning: Found SMART exchange item, but it has no 'expirations' attribute.")
                    found_smart = True # Mark as found but problematic
                    break

        except Exception as e:
            # Handle potential issues if items are not objects or lack attributes
            print(f"Warning: Could not process chain item: {e}. Item: {chain_item}")
            continue # Skip to the next item

    if not found_smart:
        print("Warning: No chain item found with exchange='SMART' in the provided list.")

else:
    print("Error: 'chains' variable does not appear to be a valid list.")

# 2. Call the filtering function *only* if SMART expirations were found
if smart_chain_expirations:
    nearby_dates = get_nearby_expirations(smart_chain_expirations, days_out=45) # Using 45 days

    # 3. Print the final result
    print("-" * 20)
    print(f"FINAL OUTPUT (Expirations for SMART within {45} days):")
    if nearby_dates:
        for date_str in nearby_dates:
            print(f"- {date_str}")
    else:
        # This will be printed if today is far from the SMART expiration dates (e.g., 2025)
        print(f"No SMART expirations found between {datetime.date.today().strftime('%Y-%m-%d')} and {(datetime.date.today() + datetime.timedelta(days=45)).strftime('%Y-%m-%d')}.")
    print("-" * 20)

else:
    # This message is printed if 'SMART' wasn't found or had no expirations
    print("Could not find SMART exchange data or its expirations list was empty.")