import codecs
import datetime
import json
import math
import numpy as np
import pandas as pd
import ta
import yfinance as yf
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from backend_api.api import sector
from backend_api.util import settings
from typing import Tuple


def get_best_portfolios(df, model_name: str) -> list:
    if model_name == 'Markowitz':
        optional_portfolios = [build_return_markowitz_portfolios_dic(df[0]),
                               build_return_markowitz_portfolios_dic(df[1]),
                               build_return_markowitz_portfolios_dic(df[2])]
    else:
        optional_portfolios = [build_return_gini_portfolios_dic(df[0]),
                               build_return_gini_portfolios_dic(df[1]),
                               build_return_gini_portfolios_dic(df[2])]
    return [optional_portfolios[0]['Safest Portfolio'], optional_portfolios[1]['Sharpe Portfolio'],
            optional_portfolios[2]['Max Risk Portfolio']]


def get_best_weights_column(stocks_symbols, sectors_list, optional_portfolios, pct_change_table) -> list:
    pct_change_table.dropna(inplace=True)
    stock_sectors = set_stock_sectors(stocks_symbols, sectors_list)
    high = np.dot(optional_portfolios[2].iloc[0][3:], pct_change_table.T)
    medium = np.dot(optional_portfolios[1].iloc[0][3:], pct_change_table.T)
    pct_change_table_low = pct_change_table.copy()
    for i in range(len(stock_sectors)):
        if stock_sectors[i] == "US commodity":
            pct_change_table_low = pct_change_table_low.drop(stocks_symbols[i], axis=1)
    low = np.dot(optional_portfolios[0].iloc[0][3:], pct_change_table_low.T)
    return [low, medium, high]


def get_three_best_weights(optional_portfolios):
    weighted_low = optional_portfolios[0].iloc[0][3:]
    weighted_medium = optional_portfolios[1].iloc[0][3:]
    weighted_high = optional_portfolios[2].iloc[0][3:]

    return [weighted_low, weighted_medium, weighted_high]


def choose_portfolio_by_risk_score(optional_portfolios_list, risk_score):
    if 0 < risk_score <= 4:
        return optional_portfolios_list[0]
    elif 5 < risk_score <= 7:
        return optional_portfolios_list[1]
    elif risk_score > 7:
        return optional_portfolios_list[2]
    else:
        raise ValueError


def build_return_gini_portfolios_dic(df):
    return_dic = {'Max Risk Portfolio': {}, 'Safest Portfolio': {}, 'Sharpe Portfolio': {}}
    min_gini = df['Gini'].min()
    max_sharpe = df['Sharpe Ratio'].max()
    max_portfolio_annual = df['Portfolio_annual'].max()

    # use the min, max values to locate and create the two special portfolios
    sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
    safe_portfolio = df.loc[df['Gini'] == min_gini]
    max_portfolio = df.loc[df['Portfolio_annual'] == max_portfolio_annual]

    return_dic['Max Risk Portfolio'] = max_portfolio
    return_dic['Safest Portfolio'] = safe_portfolio
    return_dic['Sharpe Portfolio'] = sharpe_portfolio

    return return_dic


def build_return_markowitz_portfolios_dic(df):
    return_dic = {'Max Risk Portfolio': {}, 'Safest Portfolio': {}, 'Sharpe Portfolio': {}}
    min_markowitz = df['Volatility'].min()
    max_sharpe = df['Sharpe Ratio'].max()
    max_portfolio_annual = df['Returns'].max()

    # use the min, max values to locate and create the two special portfolios
    sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
    safe_portfolio = df.loc[df['Volatility'] == min_markowitz]
    max_portfolio = df.loc[df['Returns'] == max_portfolio_annual]

    return_dic['Max Risk Portfolio'] = max_portfolio
    return_dic['Safest Portfolio'] = safe_portfolio
    return_dic['Sharpe Portfolio'] = sharpe_portfolio

    return return_dic


def get_json_data(name: str):
    import os
    print(os.getcwd())
    with codecs.open(name + ".json", "r", encoding="utf-8") as file:
        json_data = json.load(file)
    return json_data


def get_sectors_data_from_file(mode: str):
    if mode == 'regular':
        sectors_data = get_json_data(settings.SECTORS_LOCATION)
    else:
        sectors_data = get_json_data('../../' + settings.SECTORS_LOCATION)
    return sectors_data['sectorsList']['result']


def set_sectors(stocks_symbols, mode: str) -> list:
    sectors_data = get_sectors_data_from_file(mode)
    sectors: list = []

    if (len(sectors_data)) > 0:
        for i in range(len(sectors_data)):
            curr_sector = sector.Sector(sectors_data[i]['sectorName'])
            for j in range(len(stocks_symbols)):
                if stocks_symbols[j] in sectors_data[i]['stocks']:
                    curr_sector.add_stock(stocks_symbols[j])
            if len(curr_sector.get_stocks()) > 0:
                sectors.append(curr_sector)

    return sectors


def set_stock_sectors(stocks_symbols, sectors: list) -> list:
    stock_sectors = []  # TODO - FIX ORDER
    for symbol in stocks_symbols:
        found_sector = False
        for curr_sector in sectors:
            if symbol in curr_sector.get_stocks():
                stock_sectors.append(curr_sector.get_name())
                found_sector = True
                break
        if not found_sector:
            stock_sectors.append(None)

    return stock_sectors


def return_sectors_weights_according_to_stocks_weights(sectors: list, stock_symbols, stocks_weights) -> list:
    sectors_weights = [0.0] * len(sectors)
    for i in range(len(sectors)):
        sectors_weights[i] = 0
        # get from stocks_weights each symbol name without weight

        for j in range(len(stocks_weights.index)):

            first_component = stocks_weights.index[j].split()[0]

            # Check if the first component can be converted to an integer
            try:
                first_component_int = int(first_component)
                # The first component is a valid integer, use it for integer comparison
                if first_component_int in sectors[i].get_stocks():
                    sectors_weights[i] += stocks_weights[j]
            except ValueError:
                # The first component is not a valid integer, use it for string comparison
                if first_component in sectors[i].get_stocks():
                    sectors_weights[i] += stocks_weights[j]

    return sectors_weights


def drop_stocks_from_us_commodity_sector(stocks_symbols, stock_sectors):
    new_stocks_symbols = []
    for i in range(len(stock_sectors)):
        if stock_sectors[i] != "US commodity":
            new_stocks_symbols.append(stocks_symbols[i])

    return new_stocks_symbols


def get_three_best_sectors_weights(sectors_list, stocks_symbols, three_best_stocks_weights) -> list:
    sectors_weights_list = []
    for i in range(len(three_best_stocks_weights)):
        sectors_weights_list.append(return_sectors_weights_according_to_stocks_weights(sectors_list, stocks_symbols,
                                                                                       three_best_stocks_weights[i]))

    return sectors_weights_list


def get_from_and_to_dates(num_of_years) -> Tuple[str, str]:
    today = datetime.datetime.now()
    start_year = today.year - num_of_years
    start_month = today.month
    start_day = today.day
    end_year = today.year
    end_month = today.month
    end_day = today.day
    from_date = str(start_year) + "-" + str(start_month) + "-" + str(start_day)
    to_date = str(end_year) + "-" + str(end_month) + "-" + str(end_day)
    return from_date, to_date


def price_forecast(df: pd.DataFrame, record_percentage_to_predict, is_data_from_tase: int):
    if is_data_from_tase == 1:
        # Fill NaN values in 'indexOpeningPrice' with corresponding 'base index price'
        df["indexOpeningPrice"].fillna(df["baseIndexPrice"], inplace=True)
        df["high"].fillna(df["closingIndexPrice"], inplace=True)
        df["low"].fillna(df["closingIndexPrice"], inplace=True)

        df["HL_PCT"] = (df["high"] - df["low"]) / df["low"] * 100.0
        df["PCT_change"] = (
            (df["closingIndexPrice"] - df["indexOpeningPrice"])
            / df["indexOpeningPrice"]
            * 100.0
        )

        df = df[["closingIndexPrice", "HL_PCT", "PCT_change"]]

        forecast_col = "closingIndexPrice"

    else:
        df["HL_PCT"] = (df["High"] - df["Low"]) / df["Low"] * 100.0
        df["PCT_change"] = (df["Close"] - df["Open"]) / df["Open"] * 100.0

        df = df[["Adj Close", "HL_PCT", "PCT_change", "Volume"]]

        forecast_col = "Adj Close"

    df.fillna(value=-99999, inplace=True)
    forecast_out = int(math.ceil(record_percentage_to_predict * len(df)))
    df["label"] = df[forecast_col].shift(-forecast_out)
    print(df.head())

    X = np.array(df.drop(columns=["label"]))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    df.dropna(inplace=True)

    y = np.array(df["label"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=record_percentage_to_predict)
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(confidence)
    forecast_set = clf.predict(X_lately)
    df["Forecast"] = np.nan

    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        # df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
        # Create a list of NaN values with the appropriate length (excluding the last column)
        nan_values = [np.nan] * (len(df.columns) - 1)
        # Append the value of i to the list of NaN values
        nan_values.append(i)
        # Assign the list of values to the next_date row using loc
        df.loc[next_date] = nan_values

    col = df["Forecast"]
    col = col.dropna()
    return df, col


def implement_bb_strategy(data, lower_bb, upper_bb):
    buy_price = []
    sell_price = []
    bb_signal = []
    signal = 0

    for i in range(len(data)):
        if data[i - 1] > lower_bb[i - 1] and data[i] < lower_bb[i]:
            if signal != 1:
                buy_price.append(data[i])
                sell_price.append(np.nan)
                signal = 1
                bb_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_signal.append(0)
        elif data[i - 1] < upper_bb[i - 1] and data[i] > upper_bb[i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(data[i])
                signal = -1
                bb_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            bb_signal.append(0)

    return buy_price, sell_price, bb_signal


def makes_yield_column(_yield, weighted_sum_column):
    _yield.iloc[0] = 1
    for i in range(1, weighted_sum_column.size):
        change = weighted_sum_column.item(i) + 1
        last_value = _yield.iloc[i - 1]
        new_value = last_value * change
        _yield.iloc[i] = new_value
    return _yield


def scan_good_stocks():
    # Define the parameters for the scanner
    min_avg_volume = 1000000  # minimum average daily volume
    min_rsi = 50  # minimum RSI value
    min_price = 0  # minimum price (in dollars)
    max_price = 1000  # maximum price (in dollars)

    # Download the list of all tickers from Yahoo Finance
    yf.Tickers("")
    # Fetch the list of top 2000 stocks listed on NASDAQ
    nasdaq_2000 = pd.read_csv('https://www.nasdaq.com/api/v1/screener?page=1&pageSize=2000')
    # Get the ticker symbols for the top 2000 stocks
    tickers = nasdaq_2000['symbol'].to_list()

    # Create an empty DataFrame to store the results
    results = pd.DataFrame(columns=["Ticker", "Price", "50-Day MA", "200-Day MA", "52-Week High", "RSI", "Avg Volume"])

    # Loop through the tickers and scan for the best stocks
    for ticker in tickers.tickers:
        try:
            # Download the historical prices for the stock
            history = ticker.history(period="max")

            # Compute the 50-day and 200-day moving averages
            ma50 = ta.trend.SMAIndicator(history["Close"], window=50).sma_indicator()
            ma200 = ta.trend.SMAIndicator(history["Close"], window=200).sma_indicator()

            # Check if the stock is in an uptrend
            if ma50.iloc[-1] > ma200.iloc[-1]:
                # Compute the 52-week high
                high52 = history["High"].rolling(window=252).max().iloc[-1]

                # Check if the stock has broken out to a new high
                if history["Close"].iloc[-1] > high52:
                    # Compute the RSI
                    rsi = ta.momentum.RSIIndicator(history["Close"]).rsi()

                    # Check if the RSI is above the minimum value
                    if rsi.iloc[-1] > min_rsi:
                        # Compute the average daily volume
                        avg_volume = history["Volume"].rolling(window=20).mean().iloc[-1]

                        # Check if the average daily volume is above the minimum value
                        if avg_volume > min_avg_volume:
                            # Check if the price is within the specified range
                            price = history["Close"].iloc[-1]
                            if min_price <= price <= max_price:
                                # Add the result to the DataFrame
                                results = results.append({"Ticker": ticker.ticker, "Price": price,
                                                          "50-Day MA": ma50.iloc[-1], "200-Day MA": ma200.iloc[-1],
                                                          "52-Week High": high52, "RSI": rsi.iloc[-1],
                                                          "Avg Volume": avg_volume}, ignore_index=True)
        finally:
            pass

    # Sort the results by RSI in descending order
    results = results.sort_values(by="RSI", ascending=False)
    return results.head(10)


def find_best_stocks():
    """
    # Fetch the list of top 2000 stocks listed on NASDAQ
    nasdaq_2000 = pd.read_csv('https://www.nasdaq.com/api/v1/screener?page=1&pageSize=2000')

    # Get the ticker symbols for the top 2000 stocks
    tickers = nasdaq_2000['symbol'].to_list()

    # Fetch the historical data for each stock
    data = yf.download(tickers, start="2022-05-15", end="2023-05-15")

    # Calculate the annual returns and volatility for each stock
    returns = data['Adj Close'].pct_change().groupby([pd.Grouper(freq='Y'), data.index.get_level_values(1)]).apply(
        lambda x: (1 + x).prod() - 1).reset_index(level=1, drop=True)
    volatility = data['Adj Close'].pct_change().groupby(
        [pd.Grouper(freq='Y'), data.index.get_level_values(1)]).std().reset_index(level=1, drop=True)

    # Calculate the Sharpe ratio for each stock
    sharpe = returns / volatility

    # Sort the stocks based on their Sharpe ratio
    sharpe_sorted = sharpe.sort_values(ascending=False)

    # Select the top 5 stocks with the highest Sharpe ratio
    top_5_stocks = sharpe_sorted.head(5)

    # Print the top 5 stocks with the highest Sharpe ratio
    print(top_5_stocks)"""

    """ Filter the stocks based on a set of criteria
     filtered_stocks = nasdaq_100[
         (nasdaq_100['lastsale'] >= 10) &  # Stocks with a last sale price of at least $10
         (nasdaq_100['marketCap'] >= 1e9) &  # Stocks with a market capitalization of at least $1 billion
         (nasdaq_100['ipoyear'] <= 2020)  # Stocks that went public before or in 2020
         ]

     # Get the ticker symbols for the filtered stocks
     tickers = filtered_stocks['symbol'].to_list()"""
    # Fetch the list of NASDAQ tickers from Yahoo Finance
    # nasdaq_tickers = yf.Tickers('^IXIC').tickers
    # Convert the dictionary of Ticker objects to a list of Ticker objects
    # ticker_list = list(nasdaq_tickers.values())
    # tickers = [ticker.ticker for ticker in ticker_list]
    tickers = ["TA35.TA", "TA90.TA", 'SPY', 'QQQ', 'rut', 'IEI', 'LQD', 'Gsg', 'GLD', 'OIL']

    # Fetch the historical data for the filtered stocks
    data = yf.download(tickers, start="2022-05-15", end="2023-05-15")
    data = data.set_index(pd.to_datetime(data.index))
    # Calculate the annual returns and volatility for each stock
    returns = data['Adj Close'].pct_change().groupby(pd.Grouper(freq='Y')).apply(
        lambda x: (1 + x).prod() - 1).reset_index(level=0, drop=True)
    volatility = data['Adj Close'].pct_change().groupby(pd.Grouper(freq='Y')).std().reset_index(level=0, drop=True)

    # Calculate the Sharpe ratio for each stock
    sharpe = returns / volatility

    # Sort the stocks based on their Sharpe ratio
    sharpe_sorted = sharpe.sort_values()

    # Select the top 5 stocks with the highest Sharpe ratio
    top_5_stocks = sharpe_sorted.head(5)

    return top_5_stocks
