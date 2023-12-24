"""
Scrape and Analyze Stock Prices - Save Files
"""
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# Scraping US stock prices from Yahoo! finance
def scrape_stock(ticker, start_date, end_date):
    df_stock = yf.download(ticker, 
                           start=start_date, 
                           end=end_date
                           )
    df_stock.reset_index(inplace=True)
    return df_stock

df_sp500 = scrape_stock('%5EGSPC', 
                        '2000-01-01', 
                        '2023-12-08'
                        )
df_dow = scrape_stock('%5EDJI?p=%5EDJI', 
                      '2000-01-01', 
                      '2023-12-08'
                      )
df_nasdaq = scrape_stock('^IXIC', 
                         '2000-01-01', 
                         '2023-12-08'
                         )

# Adjust timezone
def timezone_adjust(df):
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize("UTC")
    return df

df_sp500 = timezone_adjust(df_sp500)
df_dow = timezone_adjust(df_dow)
df_nasdaq = timezone_adjust(df_nasdaq)



# Filter Year and Create a List of Dataframes
# S&P 500
df_sp500_list = {}
# Create different DataFrames based on the condition
for year in range(2000, 2024):
    df_sp500_list[year] = df_sp500[df_sp500['Date'].dt.year >= year]

# Dow Jones
df_dow_list = {}
for year in range(2000, 2024):
    df_dow_list[year] = df_dow[df_dow['Date'].dt.year >= year]

# NASDAQ
df_nasdaq_list = {}
for year in range(2000, 2024):
    df_nasdaq_list[year] = df_nasdaq[df_nasdaq['Date'].dt.year >= year]


# Calculate N-Day Average Prices
def average_calculate(df, window_sizes=[10, 20, 60]):
    # Sort the DataFrame by date
    df = df.sort_values(by='Date')

    for window_size in window_sizes:
        # Calculate the rolling average
        column_name = f'{window_size}-Day Average'
        df[column_name] = df['Close'].rolling(window=window_size, min_periods=1).mean()

        # Calculate the corresponding average signal
        signal_column_name = f'{window_size}-Day Average Signal'
        df[signal_column_name] = np.where(df[column_name] > df[column_name].shift(1), 1, 0)

        # Compare with the close price
        now_vs_column_name = f'Now_vs_{window_size} Day Avg'
        df[now_vs_column_name] = np.where(df[column_name] < df['Close'], 1, 0)

    return df

# S&P 500
df_sp500_calculated = {}
for year, df_year in df_sp500_list.items():
    df_calculated = average_calculate(df_year.copy())
    df_calculated['Year'] = year
    df_calculated['Stock Name'] = 'S&P 500'
    df_sp500_calculated[year] = df_calculated

# Dow Jones
df_dow_calculated = {}
for year, df_year in df_dow_list.items():
    df_calculated = average_calculate(df_year.copy())
    df_calculated['Year'] = year
    df_calculated['Stock Name'] = 'DOW JONES'
    df_dow_calculated[year] = df_calculated

# NASDAQ
df_nasdaq_calculated = {}
for year, df_year in df_nasdaq_list.items():
    df_calculated = average_calculate(df_year.copy())
    df_calculated['Year'] = year
    df_calculated['Stock Name'] = 'NASDAQ'
    df_nasdaq_calculated[year] = df_calculated

#####
sp500_dfs = pd.concat(df_sp500_calculated, ignore_index=True)
dow_dfs = pd.concat(df_dow_calculated, ignore_index=True)
nasdaq_dfs = pd.concat(df_nasdaq_calculated, ignore_index=True)

stock_dfs = pd.concat([sp500_dfs, dow_dfs, nasdaq_dfs], ignore_index=True)
stock_dfs['Date'] = stock_dfs['Date'].dt.tz_localize(None)
stock_dfs.to_excel('stock_dfs.xlsx', index=False)
#####



#INVESTMENT STRATEGY    
# Stratey 1: If current price is larger than monthly average price, BUY!!!, If current price is less than the monthly average price, SELL!!!
def strategy1_profit_statistics(df_dict, current_avg_columns, stock_name):
    result_profit_dfs = []
    result_stat_dfs = []

    for year, df in df_dict.items():
        # Find the index where the signal changes from 0 to 1
        buy_signals = [] 
        buy_signals.extend(df.index[(df[current_avg_columns].diff() == 1) & (df[current_avg_columns] == 1)].tolist())

        # Find the index where the signal changes from 1 to 0
        sell_signals = []            
        sell_signals.extend(df.index[(df[current_avg_columns].diff() == -1) & (df[current_avg_columns] == 0)].tolist())

        nearest_pairs = []

        # Loop through all buy signals
        for buy_signal in buy_signals:
            min_time_difference = pd.Timedelta.max  # Set initial value to maximum possible Timedelta
            buy_row = df.loc[buy_signal]

            # Loop through all sell signals
            for sell_signal in sell_signals:
                # Extract the rows for the current pair
                sell_row = df.loc[sell_signal]

                # Check if the current pair is positive
                if sell_row['Date'] > buy_row['Date']:
                    # Calculate the time difference
                    time_difference = sell_row['Date'] - buy_row['Date']

                    # Check if the current pair has a smaller time difference
                    if time_difference < min_time_difference:
                        min_time_difference = time_difference
                        nearest_sell_signal = sell_row

            if nearest_sell_signal is not None:
                nearest_pairs.append((buy_row, nearest_sell_signal))

        # Create a DataFrame from the list of nearest pairs
        strategy_profit_df = pd.DataFrame([(year, stock_name, 'Strategy 1', pair[0]['Date'], pair[1]['Date'], pair[1]['Close'] - pair[0]['Close'], (pair[1]['Close'] - pair[0]['Close']) / pair[0]['Close'], 
                                            f'{current_avg_columns}') for pair in nearest_pairs], columns=['Year', 'Stock Name', 'Strategy', 'BUY Date', 'SELL Date', 'Profit', 'Return Rate (%)', 'N-Day Average'])

        # Calculate and return mean and standard deviation of profit
        mean_profit = strategy_profit_df["Profit"].mean()
        std_profit = strategy_profit_df["Profit"].std()
        mean_return = strategy_profit_df["Return Rate (%)"].mean()
        std_return = strategy_profit_df["Return Rate (%)"].std()
        strategy_stat_dict = {
            'Year': [year],
            'Stock' : [stock_name],
            'Strategy': ['Strategy 1'],
            'N-Day Average Price': [f'{current_avg_columns}'],
            'Mean Profit': [mean_profit],
            'Std Dev Profit': [std_profit],
            'Mean Return (%)': [mean_return],
            'Std Dev Return (%)': [std_return]
        }

        # Create a DataFrame from the dictionary
        strategy_stat_df = pd.DataFrame.from_dict(strategy_stat_dict)

        # Append the result DataFrame to the list
        result_profit_dfs.append(strategy_profit_df)
        result_stat_dfs.append(strategy_stat_df)

    return result_profit_dfs, result_stat_dfs

# List of column names for the rolling averages (now_vs_column_names)
now_vs_columns = ['Now_vs_10 Day Avg', 'Now_vs_20 Day Avg', 'Now_vs_60 Day Avg']

# S&P 500
sp500_profit_dfs = []
sp500_stat_dfs = []
for column_name in now_vs_columns:
    df_sp500_profit, df_sp500_stat = strategy1_profit_statistics(df_sp500_calculated, column_name, 'S&P 500')
    sp500_profit_dfs.append(df_sp500_profit)
    sp500_stat_dfs.append(df_sp500_stat)

flat_sp500_profit_dfs = [item for sublist in sp500_profit_dfs for item in sublist]
flat_sp500_stat_dfs = [item for sublist in sp500_stat_dfs for item in sublist]

sp500_profit_dfs = pd.concat(flat_sp500_profit_dfs, ignore_index=True)
sp500_stat_dfs = pd.concat(flat_sp500_stat_dfs, ignore_index=True)


# Dow Jones
dow_profit_dfs = []
dow_stat_dfs = []
for column_name in now_vs_columns:
    df_dow_profit, df_dow_stat = strategy1_profit_statistics(df_dow_calculated, column_name, 'DOW JONES')
    dow_profit_dfs.append(df_dow_profit)
    dow_stat_dfs.append(df_dow_stat)

flat_dow_profit_dfs = [item for sublist in dow_profit_dfs for item in sublist]
flat_dow_stat_dfs = [item for sublist in dow_stat_dfs for item in sublist]

dow_profit_dfs = pd.concat(flat_dow_profit_dfs, ignore_index=True)
dow_stat_dfs = pd.concat(flat_dow_stat_dfs, ignore_index=True)

# NASDAQ
nasdaq_profit_dfs = []
nasdaq_stat_dfs = []
for column_name in now_vs_columns:
    df_nasdaq_profit, df_nasdaq_stat = strategy1_profit_statistics(df_nasdaq_calculated, column_name, 'NASDAQ')
    nasdaq_profit_dfs.append(df_nasdaq_profit)
    nasdaq_stat_dfs.append(df_nasdaq_stat)

flat_nasdaq_profit_dfs = [item for sublist in nasdaq_profit_dfs for item in sublist]
flat_nasdaq_stat_dfs = [item for sublist in nasdaq_stat_dfs for item in sublist]

nasdaq_profit_dfs = pd.concat(flat_nasdaq_profit_dfs, ignore_index=True)
nasdaq_stat_dfs = pd.concat(flat_nasdaq_stat_dfs, ignore_index=True)


# Combine all profit DataFrames
strategy1_profit_dfs = pd.concat([sp500_profit_dfs, dow_profit_dfs, nasdaq_profit_dfs], ignore_index=True)
strategy1_profit_dfs['BUY Date'] = strategy1_profit_dfs['BUY Date'].dt.tz_localize(None)
strategy1_profit_dfs['SELL Date'] = strategy1_profit_dfs['SELL Date'].dt.tz_localize(None)
strategy1_profit_dfs.to_excel('strategy1_profit_dfs.xlsx', index=False)
# Combine all stat DataFrames
strategy1_stat_dfs = pd.concat([sp500_stat_dfs, dow_stat_dfs, nasdaq_stat_dfs], ignore_index=True)
strategy1_stat_dfs.to_excel('strategy1_stat_dfs.xlsx', index=False)






# Strategy II: If monthly average price changes from 0 to 1, BUY!!! If current price is less than the monthly average price, SELL!!!
def strategy2_profit_statistics(df_dict, avg_change_column, current_avg_column, stock_name):
    result_profit_dfs = []
    result_stat_dfs = []

    for year, df in df_dict.items():
        # Find the index where the signal changes from 0 to 1 for any of the average change columns
        buy_signals = []
        buy_signals.extend(df.index[(df[avg_change_column].diff() == 1) & (df[avg_change_column] == 1)].tolist())

        # Find the index where the signal changes from 1 to 0 for any of the current average columns
        sell_signals = []
        sell_signals.extend(df.index[(df[current_avg_column].diff() == -1) & (df[current_avg_column] == 0)].tolist())

        nearest_pairs = []

        # Loop through all buy signals
        for buy_signal in buy_signals:
            min_time_difference = pd.Timedelta.max  # Set initial value to maximum possible Timedelta
            buy_row = df.loc[buy_signal]

            # Loop through all sell signals for the current average column
            for sell_signal in sell_signals:
                # Extract the rows for the current pair
                sell_row = df.loc[sell_signal]

                # Check if the current pair is positive
                if sell_row['Date'] > buy_row['Date']:
                    # Calculate the time difference
                    time_difference = sell_row['Date'] - buy_row['Date']

                    # Check if the current pair has a smaller time difference
                    if time_difference < min_time_difference:
                        min_time_difference = time_difference
                        nearest_sell_signal = sell_row

            # Check if a valid nearest pair was found before appending
            if min_time_difference != pd.Timedelta.max:
                # Append the nearest pair for the current buy signal
                nearest_pairs.append((buy_row, nearest_sell_signal))

        # Create a DataFrame from the list of nearest pairs
        strategy2_profit_df = pd.DataFrame([(year, stock_name, 'Strategy 2', pair[0]['Date'], pair[1]['Date'], pair[1]['Close'] - pair[0]['Close'], (pair[1]['Close'] - pair[0]['Close']) / pair[0]['Close'], f'{current_avg_column}') for pair in nearest_pairs], columns=['Year', 'Stock Name', 'Strategy', 'BUY Date', 'SELL Date', 'Profit', 'Return Rate (%)', 'N-Day Average'])

        # Calculate and return mean and standard deviation of profit
        mean_profit = strategy2_profit_df["Profit"].mean()
        std_profit = strategy2_profit_df["Profit"].std()
        mean_return = strategy2_profit_df["Return Rate (%)"].mean()
        std_return = strategy2_profit_df["Return Rate (%)"].std()
        strategy2_stat_dict = {
            'Year': [year],
            'Stock': [stock_name],
            'Strategy': ['Strategy 2'],
            'N-Day Average Price': [f'{current_avg_column}'],
            'Mean Profit': [mean_profit],
            'Std Dev Profit': [std_profit],
            'Mean Return (%)': [mean_return],
            'Std Dev Return (%)': [std_return]
        }

        # Create the strategy2_stat_df DataFrame from the dictionary
        strategy2_stat_df = pd.DataFrame.from_dict(strategy2_stat_dict)

        # Append the result DataFrame to the list
        result_profit_dfs.append(strategy2_profit_df)
        result_stat_dfs.append(strategy2_stat_df)

    return result_profit_dfs, result_stat_dfs



# List of column names for the rolling averages (now_vs_column_names)
now_vs_columns = ['Now_vs_10 Day Avg', 'Now_vs_20 Day Avg', 'Now_vs_60 Day Avg']
avg_change_columns = ['10-Day Average Signal', '20-Day Average Signal', '60-Day Average Signal']

# S&P 500
sp500_profit2_dfs = []
sp500_stat2_dfs = []

for now_vs_column, avg_change_column in zip(now_vs_columns, avg_change_columns): 
    df_sp500_profit2, df_sp500_stat2 = strategy2_profit_statistics(df_sp500_calculated, avg_change_column, now_vs_column, 'S&P 500')
    sp500_profit2_dfs.append(df_sp500_profit2)
    sp500_stat2_dfs.append(df_sp500_stat2)

flat_sp500_profit2_dfs = [item for sublist in sp500_profit2_dfs for item in sublist]
flat_sp500_stat2_dfs = [item for sublist in sp500_stat2_dfs for item in sublist]

sp500_profit2_dfs = pd.concat(flat_sp500_profit2_dfs, ignore_index=True)
sp500_stat2_dfs = pd.concat(flat_sp500_stat2_dfs, ignore_index=True)


# DOW JONES
dow_profit2_dfs = []
dow_stat2_dfs = []

for now_vs_column, avg_change_column in zip(now_vs_columns, avg_change_columns): 
    df_dow_profit2, df_dow_stat2 = strategy2_profit_statistics(df_dow_calculated, avg_change_column, now_vs_column, 'DOW JONES')
    dow_profit2_dfs.append(df_dow_profit2)
    dow_stat2_dfs.append(df_dow_stat2)

flat_dow_profit2_dfs = [item for sublist in dow_profit2_dfs for item in sublist]
flat_dow_stat2_dfs = [item for sublist in dow_stat2_dfs for item in sublist]

dow_profit2_dfs = pd.concat(flat_dow_profit2_dfs, ignore_index=True)
dow_stat2_dfs = pd.concat(flat_dow_stat2_dfs, ignore_index=True)


# NASDAQ
nasdaq_profit2_dfs = []
nasdaq_stat2_dfs = []

for now_vs_column, avg_change_column in zip(now_vs_columns, avg_change_columns): 
    df_nasdaq_profit2, df_nasdaq_stat2 = strategy2_profit_statistics(df_nasdaq_calculated, avg_change_column, now_vs_column, 'NASDAQ')
    nasdaq_profit2_dfs.append(df_nasdaq_profit2)
    nasdaq_stat2_dfs.append(df_nasdaq_stat2)

flat_nasdaq_profit2_dfs = [item for sublist in nasdaq_profit2_dfs for item in sublist]
flat_nasdaq_stat2_dfs = [item for sublist in nasdaq_stat2_dfs for item in sublist]

nasdaq_profit2_dfs = pd.concat(flat_nasdaq_profit2_dfs, ignore_index=True)
nasdaq_stat2_dfs = pd.concat(flat_nasdaq_stat2_dfs, ignore_index=True)


# Combine all profit DataFrames
strategy2_profit_dfs = pd.concat([sp500_profit2_dfs, dow_profit2_dfs, nasdaq_profit2_dfs], ignore_index=True)
strategy2_profit_dfs['BUY Date'] = strategy2_profit_dfs['BUY Date'].dt.tz_localize(None)
strategy2_profit_dfs['SELL Date'] = strategy2_profit_dfs['SELL Date'].dt.tz_localize(None)
strategy2_profit_dfs.to_excel('strategy2_profit_dfs.xlsx', index=False)
# Combine all stat DataFrames
strategy2_stat_dfs = pd.concat([sp500_stat2_dfs, dow_stat2_dfs, nasdaq_stat2_dfs], ignore_index=True)
strategy2_stat_dfs.to_excel('strategy2_stat_dfs.xlsx', index=False)










# Strtegy III: Combine Strategy I and Strategy II
def strategy3_profit_statistics(df_dict, avg_change_columns, current_avg_column, stock_name):
    result_profit_dfs = []
    result_stat_dfs = []

    for year, df in df_dict.items():
        # Find the index where the signal changes from 0 to 1 for any of the average change columns
        buy_signals = df.index[(df[avg_change_columns].diff() == 1) & (df[avg_change_columns] == 1) & (df[current_avg_column].diff() == 1) & (df[current_avg_column] == 1)].tolist()

        # Find the index where the signal changes from 1 to 0 for any of the current average columns
        sell_signals = df.index[(df[current_avg_column].diff() == -1) & (df[current_avg_column] == 0)].tolist()

        nearest_pairs = []

        # Loop through all 0_to_1 transitions
        for buy_signal in buy_signals:
            min_time_difference = pd.Timedelta.max
            buy_row = df.loc[buy_signal]

            # Initialize variables for the nearest pair
            nearest_sell_signal = None

            # Loop through all 1_to_0 transitions
            for sell_signal in sell_signals:
                # Extract the rows for the current pair
                sell_row = df.loc[sell_signal]

                # Check if the current pair is positive
                if sell_row["Date"] > buy_row["Date"]:
                    # Calculate the time difference
                    time_difference = sell_row["Date"] - buy_row["Date"]

                    # Check if the current pair has a smaller time difference
                    if time_difference < min_time_difference:
                        min_time_difference = time_difference
                        nearest_sell_signal = sell_row

            # Check if a valid nearest pair was found before appending
            if nearest_sell_signal is not None:
                # Append the nearest pair for the current 0_to_1 transition
                nearest_pairs.append((buy_row, nearest_sell_signal))

        # Create a DataFrame from the list of nearest pairs
        strategy3_profit_df = pd.DataFrame([(year, stock_name, 'Strategy 3', pair[0]['Date'], pair[1]['Date'], pair[1]['Close'] - pair[0]['Close'], (pair[1]['Close'] - pair[0]['Close']) / pair[0]['Close'], f'{current_avg_column}') for pair in nearest_pairs], columns=['Year', 'Stock Name', 'Strategy', 'BUY Date', 'SELL Date', 'Profit', 'Return Rate (%)', 'N-Day Average'])

        # Calculate and return mean and standard deviation of profit
        mean_profit = strategy3_profit_df["Profit"].mean()
        std_profit = strategy3_profit_df["Profit"].std()
        mean_return = strategy3_profit_df["Return Rate (%)"].mean()
        std_return = strategy3_profit_df["Return Rate (%)"].std()
        strategy3_stat_dict = {
            'Year': [year],
            'Stock': [stock_name],
            'Strategy': ['Strategy 3'],
            'N-Day Average Price': [f'{current_avg_column}'],
            'Mean Profit': [mean_profit],
            'Std Dev Profit': [std_profit],
            'Mean Return (%)': [mean_return],
            'Std Dev Return (%)': [std_return]
        }

        # Create the strategy3_stat_df DataFrame from the dictionary
        strategy3_stat_df = pd.DataFrame.from_dict(strategy3_stat_dict)

        # Append the result DataFrame to the list
        result_profit_dfs.append(strategy3_profit_df)
        result_stat_dfs.append(strategy3_stat_df)

    return result_profit_dfs, result_stat_dfs

# List of column names for the rolling averages (now_vs_column_names)
now_vs_columns = ['Now_vs_10 Day Avg', 'Now_vs_20 Day Avg', 'Now_vs_60 Day Avg']
avg_change_columns = ['10-Day Average Signal', '20-Day Average Signal', '60-Day Average Signal']

# S&P 500
sp500_profit3_dfs = []
sp500_stat3_dfs = []

for now_vs_column, avg_change_column in zip(now_vs_columns, avg_change_columns): 
    df_sp500_profit3, df_sp500_stat3 = strategy3_profit_statistics(df_sp500_calculated, avg_change_column, now_vs_column, 'S&P 500')
    sp500_profit3_dfs.append(df_sp500_profit3)
    sp500_stat3_dfs.append(df_sp500_stat3)

flat_sp500_profit3_dfs = [item for sublist in sp500_profit3_dfs for item in sublist]
flat_sp500_stat3_dfs = [item for sublist in sp500_stat3_dfs for item in sublist]

sp500_profit3_dfs = pd.concat(flat_sp500_profit3_dfs, ignore_index=True)
sp500_stat3_dfs = pd.concat(flat_sp500_stat3_dfs, ignore_index=True)


# DOW JONES
dow_profit3_dfs = []
dow_stat3_dfs = []

for now_vs_column, avg_change_column in zip(now_vs_columns, avg_change_columns): 
    df_dow_profit3, df_dow_stat3 = strategy3_profit_statistics(df_dow_calculated, avg_change_column, now_vs_column, 'DOW JONES')
    dow_profit3_dfs.append(df_dow_profit3)
    dow_stat3_dfs.append(df_dow_stat3)

flat_dow_profit3_dfs = [item for sublist in dow_profit3_dfs for item in sublist]
flat_dow_stat3_dfs = [item for sublist in dow_stat3_dfs for item in sublist]

dow_profit3_dfs = pd.concat(flat_dow_profit3_dfs, ignore_index=True)
dow_stat3_dfs = pd.concat(flat_dow_stat3_dfs, ignore_index=True)


# NASDAQ
nasdaq_profit3_dfs = []
nasdaq_stat3_dfs = []

for now_vs_column, avg_change_column in zip(now_vs_columns, avg_change_columns): 
    df_nasdaq_profit3, df_nasdaq_stat3 = strategy3_profit_statistics(df_nasdaq_calculated, avg_change_column, now_vs_column, 'NASDAQ')
    nasdaq_profit3_dfs.append(df_nasdaq_profit3)
    nasdaq_stat3_dfs.append(df_nasdaq_stat3)

flat_nasdaq_profit3_dfs = [item for sublist in nasdaq_profit3_dfs for item in sublist]
flat_nasdaq_stat3_dfs = [item for sublist in nasdaq_stat3_dfs for item in sublist]

nasdaq_profit3_dfs = pd.concat(flat_nasdaq_profit3_dfs, ignore_index=True)
nasdaq_stat3_dfs = pd.concat(flat_nasdaq_stat3_dfs, ignore_index=True)


# Combine all profit DataFrames
strategy3_profit_dfs = pd.concat([sp500_profit3_dfs, dow_profit3_dfs, nasdaq_profit3_dfs], ignore_index=True)
strategy3_profit_dfs['BUY Date'] = strategy3_profit_dfs['BUY Date'].dt.tz_localize(None)
strategy3_profit_dfs['SELL Date'] = strategy3_profit_dfs['SELL Date'].dt.tz_localize(None)
strategy3_profit_dfs.to_excel('strategy3_profit_dfs.xlsx', index=False)
# Combine all stat DataFrames
strategy3_stat_dfs = pd.concat([sp500_stat3_dfs, dow_stat3_dfs, nasdaq_stat3_dfs], ignore_index=True)
strategy3_stat_dfs.to_excel('strategy3_stat_dfs.xlsx', index=False)



# Final Step: Combine ALL!!!
profit_dfs = pd.concat([strategy1_profit_dfs, strategy2_profit_dfs, strategy3_profit_dfs], ignore_index=True)
profit_dfs.to_excel('profit_dfs.xlsx', index=False)
stat_dfs = pd.concat([strategy1_stat_dfs, strategy2_stat_dfs, strategy3_stat_dfs], ignore_index=True)
stat_dfs.to_excel('stat_dfs.xlsx', index=False)


