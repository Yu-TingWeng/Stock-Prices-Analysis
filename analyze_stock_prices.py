"""
Scrape and Analyze Stock Prices
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




#Plotting 
def stock_plotter(df, value, title):
    plt.plot(df['Date'], 
             df[value], 
             color="blue"
             )
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(title)
    plt.show()
    
stock_plotter(df_sp500, "Close", "S&P500 Historical Close Price from 2000 to now")
stock_plotter(df_dow, "Close", "Dow Jones Historical Close Prices from 2000 to now")
stock_plotter(df_nasdaq, "Close", "NASDAQ Hisotrical Close Prices from 2000 to now")





# Create an monthly average line
def average_calculate(df, day):
    
    day = int(day)
    # Sort the DataFrame by date
    df = df.sort_values(by='Date')

    # Calculate the 20-day rolling average
    df['Average'] = df['Close'].rolling(window=day, min_periods=1).mean()
    
    # If the average monthly average is larger than the previous day, give a value 1, otherwise, 0.
    df['Average Signal'] = np.where(df['Average'] > df['Average'].shift(1), 1, 0)

    # If the average monthly average is larger than that day's close price, give a value 1, otherwise, 0.
    df['now_avg_compare'] = np.where(df['Average'] < df['Close'], 1, 0)

    return df

# df_sp500 = average_calculate(df_sp500, "20")
# df_dow = average_calculate(df_dow, "20")

# df_sp500['Monthly Average Difference'] = df_sp500['Monthly Average'].diff()
# df_sp500_2013=df_sp500[df_sp500["Date"]>"2013-01-01"]






#INVESTMENT STRATEGY    
# Stratey 1: If current price is larger than monthly average price, BUY!!!, If current price is less than the monthly average price, SELL!!!
def strategy1_profit_statistics(df, current_avg_column):
    # Find the index where the signal changes from 0 to 1
    from_0_to_1 = df.index[(df[current_avg_column].diff() == 1) & (df[current_avg_column] == 1)].tolist()
    
    # Find the index where the signal changes from 1 to 0
    from_1_to_0 = df.index[(df[current_avg_column].diff() == -1) & (df[current_avg_column] == 0)].tolist()

    nearest_pairs = []

    # Loop through all 0_to_1 transitions
    for zero_to_one in from_0_to_1:
        min_time_difference = pd.Timedelta.max  # Set initial value to maximum possible Timedelta
        pair_0_to_1 = df.loc[zero_to_one]

        # Loop through all 1_to_0 transitions
        for one_to_zero in from_1_to_0:
            # Extract the rows for the current pair
            pair_1_to_0 = df.loc[one_to_zero]

            # Check if the current pair is positive
            if pair_1_to_0['Date'] > pair_0_to_1['Date']:
                # Calculate the time difference
                time_difference = pair_1_to_0['Date'] - pair_0_to_1['Date']

                # Check if the current pair has a smaller time difference
                if time_difference < min_time_difference:
                    min_time_difference = time_difference
                    nearest_1_to_0 = pair_1_to_0

        if nearest_1_to_0 is not None:
            nearest_pairs.append((pair_0_to_1, nearest_1_to_0))

    # Create a DataFrame from the list of nearest pairs
    strategy1_profit_df = pd.DataFrame([(pair[0]['Date'], pair[1]['Date'], pair[1]['Close'] - pair[0]['Close'], (pair[1]['Close'] - pair[0]['Close']) / pair[0]['Close']) for pair in nearest_pairs], columns=['BUY Date', 'SELL Date', 'Profit', 'Return Rate (%)'])

    
    # Calculate and return mean and standard deviation of profit
    mean_profit = strategy1_profit_df["Profit"].mean()
    std_profit = strategy1_profit_df["Profit"].std()
    mean_return = strategy1_profit_df["Return Rate (%)"].mean()
    std_return = strategy1_profit_df["Return Rate (%)"].std()
    strategy1_stat_dict = {
        'Strategy': ['Strategy 1'],
        'Mean Profit': [mean_profit],
        'Std Dev Profit': [std_profit],
        'Mean Return (%)': [mean_return],
        'Std Dev Return (%)': [std_return]
        }

    # Create the strategy1_stat_df DataFrame from the dictionary
    strategy1_stat_df = pd.DataFrame.from_dict(strategy1_stat_dict)

    return strategy1_profit_df, strategy1_stat_df

# strategy1_profit_sp500, strategy1_stat_sp500 = strategy1_profit_statistics(df_sp500, 'now_avg_compare')
# strategy1_profit_dow, strategy1_stat_dow = strategy1_profit_statistics(df_dow, 'now_avg_compare')








# Strategy II: If monthly average price changes from 0 to 1, BUY!!! If current price is less than the monthly average price, SELL!!!
def strategy2_profit_statistics(df, avg_change_column, current_avg_column):
    # Find the index where the signal changes from 0 to 1
    from_0_to_1 = df.index[(df[avg_change_column].diff() == 1) & (df[avg_change_column] == 1)].tolist()

    # Find the index where the signal changes from 1 to 0
    from_1_to_0 = df.index[(df[current_avg_column].diff() == -1) & (df[current_avg_column] == 0)].tolist()

    nearest_pairs = []

    # Loop through all 0_to_1 transitions
    for zero_to_one in from_0_to_1:
        min_time_difference = pd.Timedelta.max  # Set initial value to the maximum possible Timedelta
        pair_0_to_1 = df.loc[zero_to_one]

        # Initialize variables for the nearest pair
        nearest_1_to_0 = None

        # Loop through all 1_to_0 transitions
        for one_to_zero in from_1_to_0:
            # Extract the rows for the current pair
            pair_1_to_0 = df.loc[one_to_zero]

            # Check if the current pair is positive
            if pair_1_to_0["Date"] > pair_0_to_1["Date"]:
                # Calculate the time difference
                time_difference = pair_1_to_0["Date"] - pair_0_to_1["Date"]

                # Check if the current pair has a smaller time difference
                if time_difference < min_time_difference:
                    min_time_difference = time_difference
                    nearest_1_to_0 = pair_1_to_0

        # Check if a valid nearest pair was found before appending
        if nearest_1_to_0 is not None:
            # Append the nearest pair for the current 0_to_1 transition
            nearest_pairs.append((pair_0_to_1, nearest_1_to_0))

    # Create a DataFrame from the list of nearest pairs
    strategy2_profit_df = pd.DataFrame([(pair[0]['Date'], pair[1]['Date'], pair[1]['Close'] - pair[0]['Close'], (pair[1]['Close'] - pair[0]['Close']) / pair[0]['Close']) for pair in nearest_pairs], columns=['BUY Date', 'SELL Date', 'Profit', 'Return Rate (%)'])

    # Calculate and return mean and standard deviation of profit
    mean_profit = strategy2_profit_df["Profit"].mean()
    std_profit = strategy2_profit_df["Profit"].std()
    mean_return = strategy2_profit_df["Return Rate (%)"].mean()
    std_return = strategy2_profit_df["Return Rate (%)"].std()
    strategy2_stat_dict = {
        'Strategy': ['Strategy 2'],
        'Mean Profit': [mean_profit],
        'Std Dev Profit': [std_profit],
        'Mean Return (%)': [mean_return],
        'Std Dev Return (%)': [std_return]
        }

    # Create the strategy1_stat_df DataFrame from the dictionary
    strategy2_stat_df = pd.DataFrame.from_dict(strategy2_stat_dict)

    return strategy2_profit_df, strategy2_stat_df

# strategy2_profit_sp500, strategy2_stat_sp500 = strategy2_profit_statistics(df_sp500, 'Average Signal', 'now_avg_compare')
# strategy2_profit_dow, strategy2_stat_dow = strategy2_profit_statistics(df_dow, 'Average Signal', 'now_avg_compare')








# Strtegy III: Combine Strategy I and Strategy II
def strategy3_profit_statistics(df, avg_change_column, current_avg_column):
    # Find the index where the signal changes from 0 to 1
    from_0_to_1 = df.index[(df[current_avg_column].diff() == 1) & (df[current_avg_column] == 1) & (df[avg_change_column].diff() == 1) & (df[avg_change_column] == 1)].tolist()

    # Find the index where the signal changes from 1 to 0
    from_1_to_0 = df.index[(df[current_avg_column].diff() == -1) & (df[current_avg_column] == 0)].tolist()

    nearest_pairs = []

    # Loop through all 0_to_1 transitions
    for zero_to_one in from_0_to_1:
        min_time_difference = pd.Timedelta.max  
        pair_0_to_1 = df.loc[zero_to_one]

        # Initialize variables for the nearest pair
        nearest_1_to_0 = None

        # Loop through all 1_to_0 transitions
        for one_to_zero in from_1_to_0:
            # Extract the rows for the current pair
            pair_1_to_0 = df.loc[one_to_zero]

            # Check if the current pair is positive
            if pair_1_to_0["Date"] > pair_0_to_1["Date"]:
                # Calculate the time difference
                time_difference = pair_1_to_0["Date"] - pair_0_to_1["Date"]

                # Check if the current pair has a smaller time difference
                if time_difference < min_time_difference:
                    min_time_difference = time_difference
                    nearest_1_to_0 = pair_1_to_0

        # Check if a valid nearest pair was found before appending
        if nearest_1_to_0 is not None:
            # Append the nearest pair for the current 0_to_1 transition
            nearest_pairs.append((pair_0_to_1, nearest_1_to_0))

    # Create a DataFrame from the list of nearest pairs
    strategy3_profit_df = pd.DataFrame([(pair[0]['Date'], pair[1]['Date'], pair[1]['Close'] - pair[0]['Close'], (pair[1]['Close'] - pair[0]['Close']) / pair[0]['Close']) for pair in nearest_pairs], columns=['BUY Date', 'SELL Date', 'Profit', 'Return Rate (%)'])

    # Calculate and return mean and standard deviation of profit
    mean_profit = strategy3_profit_df["Profit"].mean()
    std_profit = strategy3_profit_df["Profit"].std()
    mean_return = strategy3_profit_df["Return Rate (%)"].mean()
    std_return = strategy3_profit_df["Return Rate (%)"].std()
    strategy3_stat_dict = {
        'Strategy': ['Strategy 3'],
        'Mean Profit': [mean_profit],
        'Std Dev Profit': [std_profit],
        'Mean Return (%)': [mean_return],
        'Std Dev Return (%)': [std_return]
        }

    # Create the strategy1_stat_df DataFrame from the dictionary
    strategy3_stat_df = pd.DataFrame.from_dict(strategy3_stat_dict)

    return strategy3_profit_df, strategy3_stat_df

# strategy3_profit_sp500, strategy3_stat_sp500= strategy3_profit_statistics(df_sp500, 'Average Signal', 'now_avg_compare')
# strategy3_profit_dow, strategy3_stat_dow= strategy3_profit_statistics(df_dow, 'Average Signal', 'now_avg_compare')



