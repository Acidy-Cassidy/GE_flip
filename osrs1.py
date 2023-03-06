import json
import os
import requests
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm

# Define the criteria for selecting items to buy and sell
trade_volume_threshold = 1000
price_difference_threshold = 10
price_trend_threshold = 0.1
days_to_check = 7

# Define the file path for the saved data
filename = os.path.expanduser("~/Documents/grand_exchange.json")

# Check if the saved data file exists and is recent
if os.path.exists(filename):
    file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(filename))
    if file_age < timedelta(hours=1):
        # Load the data from the file
        with open(filename, "r") as f:
            data = json.load(f)
            data = data['data']
    else:
        # Retrieve new data from the Grand Exchange API
        url = "https://prices.runescape.wiki/api/v1/osrs/latest"
        response = requests.get(url)
        data = json.loads(response.text)
        # Save the data to the file
        with open(filename, "w") as f:
            json.dump(data, f)
            print(f"New data saved to {filename}")
else:
    # Retrieve data from the Grand Exchange API
    url = "https://prices.runescape.wiki/api/v1/osrs/latest"
    response = requests.get(url)
    data = json.loads(response.text)
    # Save the data to the file
    with open(filename, "w") as f:
        json.dump(data, f)
        print(f"New data saved to {filename}")

# Check if the item_names.json file exists and is recent
if os.path.exists("item_names.json"):
    file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime("item_names.json"))
    if file_age >= timedelta(days=7):
        print("item_names.json is more than 7 days old, updating...")
        os.remove("item_names.json")
if not os.path.exists("item_names.json"):
    # Retrieve the list of all items from the OSRS Grand Exchange API
    url = "https://prices.runescape.wiki/api/v1/osrs/mapping"
    response = requests.get(url)
    data = json.loads(response.text)

    # Create a dictionary mapping item IDs to names
    item_names = {item['id']: item['name'] for item in data}

    # Save the dictionary to a file
    with open('item_names.json', 'w') as f:
        json.dump(item_names, f)
        print("item_names.json created.")
else:
    # Load the dictionary of item IDs to names from the file
    with open('item_names.json', 'r') as f:
        item_names = json.load(f)
        print("item_names.json loaded.")

# Convert JSON data to a Pandas DataFrame
df = pd.DataFrame.from_dict(data, orient='index')

# Filter the DataFrame to include only items that meet the specified criteria
df_filtered = df[(df['high'] - df['low'] >= price_difference_threshold) &
                 (df['high'] >= trade_volume_threshold) &
                 (df['lowTime'] >= (datetime.now() - timedelta(days=days_to_check)).timestamp()) &
                 (df['highTime'] >= (datetime.now() - timedelta(days=days_to_check)).timestamp())]

# Add a column for item names
df_filtered['name'] = df_filtered.index.map(item_names.get)
# Sort the DataFrame by potential profit (high price minus low price)
df_sorted = df_filtered.sort_values(by=['high', 'low'], ascending=[False, True])

# Check if the user wants to filter by their available gold amount
user_gold = input("Enter the amount of gold you want to spend (or press Enter to skip filtering): ")
print("After input prompt") # Add this line
if user_gold:
    try:
        user_gold = int(user_gold)
    except ValueError:
        print("Invalid input. Gold amount should be a whole number.")
        exit()

    # Filter the DataFrame to include only items that are within the user's price range
    df_sorted = df_sorted[df_sorted['low'] <= user_gold]

# Print a message to indicate that filtering is starting
print("Starting to filter based on input....")

# Print the top 5 items to buy and their recommended sell prices
print("Top 5 items to buy and their recommended sell prices:")
for i in range(min(5, len(df_sorted))):
    row = df_sorted.iloc[i]
    item_name = row['name']
    buy_price = row['low']
    sell_price = row['high'] * (1 - price_trend_threshold)
    print(f"{item_name}: Buy at {buy_price}, sell at {sell_price:.2f}")

print("Done.")  # Print message to indicate completion