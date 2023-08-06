import codecs
import datetime
import json
import math
import os
import re

import numpy as np
import pandas as pd
import ta
import yfinance as yf
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from typing import Tuple
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from sklearn.ensemble import GradientBoostingRegressor
from prophet import Prophet

from ..api.sector import Sector
from . import settings, tase_util


def get_best_portfolios(df, model_name: str):
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


def get_three_best_weights(optional_portfolios) -> list:

    weighted_low = optional_portfolios[0].iloc[0][3:]
    weighted_medium = optional_portfolios[1].iloc[0][3:]
    weighted_high = optional_portfolios[2].iloc[0][3:]

    return [weighted_low, weighted_medium, weighted_high]


def get_three_best_sectors_weights(sectors_list, three_best_stocks_weights) -> list:
    sectors_weights_list = []
    for i in range(len(three_best_stocks_weights)):
        sectors_weights_list.append(return_sectors_weights_according_to_stocks_weights(sectors_list,
                                                                                       three_best_stocks_weights[i]))

    return sectors_weights_list


def build_return_gini_portfolios_dic(df: pd.DataFrame):
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


def build_return_markowitz_portfolios_dic(df: pd.DataFrame):
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


def return_sectors_weights_according_to_stocks_weights(sectors: list, stocks_weights) -> list:
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
                if first_component_int in sectors[i].stocks:
                    sectors_weights[i] += stocks_weights[j]
            except ValueError:
                # The first component is not a valid integer, use it for string comparison
                if first_component in sectors[i].stocks:
                    sectors_weights[i] += stocks_weights[j]

    return sectors_weights


def analyze_with_machine_learning_linear_regression(returns_stock, table_index, closing_prices_mode=False):
    df_final = pd.DataFrame({})
    forecast_col = 'col'
    df_final[forecast_col] = returns_stock
    df_final.fillna(value=-0, inplace=True)
    forecast_out = int(math.ceil(settings.RECORD_PERCENT_TO_PREDICT * len(df_final)))
    df_final['label'] = df_final[forecast_col].shift(-forecast_out)

    # Added date
    df = df_final
    df['Date'] = table_index
    # print(df)
    X = np.array(df.drop(labels=['label', 'Date'], axis=1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    df.dropna(inplace=True)
    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=settings.TEST_SIZE_MACHINE_LEARNING)
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    # print(confidence)
    forecast_set = clf.predict(X_lately)
    df['Forecast'] = np.nan

    last_date = df.iloc[-1]['Date']
    last_date_datetime = pd.to_datetime(last_date)
    last_unix = last_date_datetime.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

    col = df["Forecast"]
    col = col.dropna()
    df["label"] = df['col']
    df["label"].fillna(df["Forecast"], inplace=True)


    forecast_returns_annual = (((1 + df['label'].mean()) ** 254) - 1) * 100
    excepted_returns = (((1 + df['Forecast'].mean()) ** 254) - 1) * 100
    if closing_prices_mode:
        forecast_returns_annual = (((1 + df['label'].pct_change().mean()) ** 254) - 1) * 100
        excepted_returns = (((1 + df['Forecast'].pct_change().mean()) ** 254) - 1) * 100
    return df, forecast_returns_annual, excepted_returns



def analyze_with_machine_learning_arima(returns_stock, table_index, closing_prices_mode=False):
    df_final = pd.DataFrame({})
    forecast_col = 'col'
    df_final[forecast_col] = returns_stock
    df_final.fillna(value=-0, inplace=True)

    forecast_out = int(math.ceil(settings.RECORD_PERCENT_TO_PREDICT * len(df_final)))
    df_final['label'] = df_final[forecast_col].shift(-forecast_out)

    # ARIMA requires datetime index for time series data
    df_final.index = pd.to_datetime(table_index)

    # Perform ARIMA forecasting
    model = pm.auto_arima(df_final[forecast_col], seasonal=False, suppress_warnings=True)
    forecast, conf_int = model.predict(n_periods=forecast_out, return_conf_int=True)

    df_final['Forecast'] = np.nan
    df_final.loc[df_final.index[-forecast_out]:, 'Forecast'] = forecast

    # forecast_returns_annual = (forecast.iloc[-1] / df_final[forecast_col].iloc[-forecast_out - 1]) ** 254 - 1

    # add dates
    last_date = df_final.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    for i in forecast:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df_final.loc[next_date] = [np.nan for _ in range(len(df_final.columns) - 1)] + [i]
    col = df_final["Forecast"]
    col = col.dropna()
    df_final["label"] = df_final["col"]
    df_final["label"].fillna(df_final["Forecast"], inplace=True)

    forecast_returns_annual = (((1 + df_final['label'].mean()) ** 254) - 1) * 100
    excepted_returns = (((1 + df_final['Forecast'].mean()) ** 254) - 1) * 100
    if closing_prices_mode:
        forecast_returns_annual = (((1 + df_final['label'].pct_change().mean()) ** 254) - 1) * 100
        excepted_returns = (((1 + df_final['Forecast'].pct_change().mean()) ** 254) - 1) * 100

    # Calculate annual return based on the forecast
    return df_final, forecast_returns_annual, excepted_returns


def analyze_with_machine_learning_gbm(returns_stock, table_index, closing_prices_mode=False):
    df_final = pd.DataFrame({})
    forecast_col = 'col'
    df_final[forecast_col] = returns_stock
    df_final.fillna(value=-0, inplace=True)

    forecast_out = int(math.ceil(settings.RECORD_PERCENT_TO_PREDICT * len(df_final)))
    df_final['label'] = df_final[forecast_col].shift(-forecast_out)

    df_final.index = pd.to_datetime(table_index)

    # Perform GBM forecasting
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    X = np.arange(len(df_final))[:, None]  # Use a simple sequence as features for demonstration
    y = df_final[forecast_col].values
    model.fit(X, y)
    forecast = model.predict(np.arange(len(df_final), len(df_final) + forecast_out)[:, None])
    df_final['Forecast'] = np.nan
    df_final.loc[df_final.index[-forecast_out]:, 'Forecast'] = forecast



    # add dates
    last_date = df_final.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    for i in forecast:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df_final.loc[next_date] = [np.nan for _ in range(len(df_final.columns) - 1)] + [i]
    df_final["Forecast"] = df_final["Forecast"].shift(forecast_out)
    col = df_final["Forecast"]
    col = col.dropna()
    df_final["label"] = df_final["col"]
    df_final["label"].fillna(df_final["Forecast"], inplace=True)

    # forecast_returns_annual = (forecast[-1] / df_final[forecast_col].iloc[-forecast_out - 1]) ** 254 - 1
    forecast_returns_annual = (((1 + df_final['label'].mean()) ** 254) - 1) * 100
    excepted_returns = (((1 + df_final['Forecast'].mean()) ** 254) - 1) * 100
    if closing_prices_mode:
        forecast_returns_annual = (((1 + df_final['label'].pct_change().mean()) ** 254) - 1) * 100
        excepted_returns = (((1 + df_final['Forecast'].pct_change().mean()) ** 254) - 1) * 100


    # Calculate annual return based on the forecast
    return df_final, forecast_returns_annual, excepted_returns



def analyze_with_machine_learning_prophet(returns_stock, table_index, closing_prices_mode=False):
    df_final = pd.DataFrame({})
    forecast_col = 'col'
    df_final[forecast_col] = returns_stock
    df_final.fillna(value=-0, inplace=True)

    forecast_out = int(math.ceil(settings.RECORD_PERCENT_TO_PREDICT * len(df_final)))
    df_final['label'] = df_final[forecast_col].shift(-forecast_out)

    # Prepare the data for Prophet
    df_prophet = pd.DataFrame({'ds': table_index, 'y': df_final[forecast_col]})

    # Create and fit the Prophet model
    model = Prophet()
    model.fit(df_prophet)

    # Generate future dates for forecasting
    future = model.make_future_dataframe(periods=forecast_out, freq='D')

    # Perform the forecast
    forecast = model.predict(future)

    # Extract the forecasted values for future dates
    forecast_for_future = forecast[forecast['ds'].isin(table_index[-forecast_out:])]['yhat']

    # Assign the forecasted values to the 'Forecast' column for future dates
    df_final.loc[table_index[-forecast_out:], 'Forecast'] = forecast_for_future.values

    # add dates
    last_date = df_final.iloc[-1].name
    try:
        last_unix = last_date.timestamp()
    except Exception:  # TODO: write relevant and specific Exception here
        # convert str to datetime
        last_date = datetime.datetime.strptime(last_date, '%Y-%m-%d')
        last_unix = last_date.timestamp()

    one_day = 86400
    next_unix = last_unix + one_day

    for i in forecast_for_future:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df_final.loc[next_date] = [np.nan for _ in range(len(df_final.columns) - 1)] + [i]
    df_final["label"] = forecast['yhat'].values
    df_final["label"][len(df_final)-forecast_out:] = forecast_for_future.values
    # df_final["label"].fillna(df_final["Forecast"], inplace=True)

    forecast_returns_annual = (((1 + forecast['yhat'].mean()) ** 254) - 1) * 100
    excepted_returns = (((1 + forecast_for_future.mean()) ** 254) - 1) * 100
    if closing_prices_mode:
        # Plot the forecast
        model.plot(forecast, xlabel='Date', ylabel='Stock Price', figsize=(12, 6))
        plt.title('Stock Price Forecast using Prophet')
        # plt.show()
        forecast_returns_annual = (((1 + forecast['yhat'].pct_change().mean()) ** 254) - 1) * 100
        excepted_returns = (((1 + forecast_for_future.pct_change().mean()) ** 254) - 1) * 100

    return df_final, forecast_returns_annual, excepted_returns, plt


def update_daily_change_with_machine_learning(returns_stock, table_index, closing_prices_mode=False):
    num_of_rows = len(table_index)
    prefix_row = int(math.ceil(settings.RECORD_PERCENT_TO_PREDICT * num_of_rows))
    for i, stock in enumerate(returns_stock.columns):
        df = None
        stock_name = str(stock)
        if settings.SELECTED_ML_MODEL_FOR_BUILD == settings.MACHINE_LEARNING_MODEL[0]:
            df, annual_return, excepted_returns = analyze_with_machine_learning_linear_regression(
                returns_stock[stock_name], table_index, closing_prices_mode
            )
        elif settings.SELECTED_ML_MODEL_FOR_BUILD == settings.MACHINE_LEARNING_MODEL[1]:
            df, annual_return, excepted_returns = analyze_with_machine_learning_arima(
                returns_stock[stock_name], table_index, closing_prices_mode
            )

        elif settings.SELECTED_ML_MODEL_FOR_BUILD == settings.MACHINE_LEARNING_MODEL[2]:
            df, annual_return, excepted_returns = analyze_with_machine_learning_gbm(
                returns_stock[stock_name], table_index, closing_prices_mode
            )


        elif settings.SELECTED_ML_MODEL_FOR_BUILD == settings.MACHINE_LEARNING_MODEL[3]:
            df, annual_return, excepted_returns, plt = analyze_with_machine_learning_prophet(
                returns_stock[stock_name], table_index, closing_prices_mode
            )
        else:
            raise ValueError('Invalid machine model')
        returns_stock[stock_name] = df['label'][prefix_row:].values
    # TODO: (annual_return, excepted_returns) not initialized
    return returns_stock, annual_return, excepted_returns


def convert_data_to_tables(location_saving, file_name, stocksNames, numOfYearsHistory, saveToCsv):
    frame = {}
    yf.pdr_override()
    start_date, end_date = get_from_and_to_dates(numOfYearsHistory)
    file_url = location_saving + file_name + ".csv"

    for i, stock in enumerate(stocksNames):

        if type(stock) == int:
            IsraeliStockData = get_israeli_indexes_data("get_past_10_years_history",
                                                        start_date, end_date, stock)
            df = pd.DataFrame(IsraeliStockData["indexEndOfDay"]["result"])
            df["tradeDate"] = pd.to_datetime(df["tradeDate"])
            df.set_index("tradeDate", inplace=True)
            price = df[["closingIndexPrice"]]
            frame[stocksNames[i]] = price
        else:
            df = yf.download(stock, start=start_date, end=end_date)
            price = df[["Adj Close"]]
            frame[stock] = price

    closingPricesTable = pd.concat(frame.values(), axis=1, keys=frame.keys())

    if saveToCsv:
        # convert to csv
        closingPricesTable.to_csv(file_url, index=True, header=True)

    return closingPricesTable


def choose_portfolio_by_risk_score(optional_portfolios_list, risk_score):
    if 0 < risk_score <= 4:
        return optional_portfolios_list[0]
    elif 5 < risk_score <= 7:
        return optional_portfolios_list[1]
    elif risk_score > 7:
        return optional_portfolios_list[2]
    else:
        raise ValueError


def get_json_data(name: str):
    import os
    print(os.getcwd())
    with codecs.open(name + ".json", "r", encoding="utf-8") as file:
        json_data = json.load(file)
    return json_data


def get_sectors_data_from_file(mode: str):
    if mode == 'regular':
        sectors_data = get_json_data(settings.SECTORS_JSON_NAME)
    else:
        sectors_data = get_json_data('../../' + settings.SECTORS_JSON_NAME)
    return sectors_data['sectorsList']['result']


def set_sectors(stocks_symbols: list, mode: str) -> list:
    """
    For each stock symbol, it checks for which sector does it belong.
    :return: It returns a list of sectors with the relevant stocks within each sector. Subset of the stock symbol
    """
    sectors_data = get_sectors_data_from_file(mode)
    sectors: list = []

    if (len(sectors_data)) > 0:
        for i in range(len(sectors_data)):
            curr_sector = Sector(sectors_data[i]['sectorName'])
            for j in range(len(stocks_symbols)):
                if stocks_symbols[j] in sectors_data[i]['stocks']:
                    curr_sector.add_stock(stocks_symbols[j])
            if len(curr_sector.stocks) > 0:
                sectors.append(curr_sector)

    return sectors


def set_stock_sectors(stocks_symbols, sectors: list) -> list:
    stock_sectors = []  # TODO - FIX ORDER
    for symbol in stocks_symbols:
        found_sector = False
        for curr_sector in sectors:
            if symbol in curr_sector.stocks:
                stock_sectors.append(curr_sector.name)
                found_sector = True
                break
        if not found_sector:
            stock_sectors.append(None)

    return stock_sectors


def drop_stocks_from_us_commodity_sector(stocks_symbols, stock_sectors):  # TODO - MAKE DYNAMIC
    new_stocks_symbols = []
    for i in range(len(stock_sectors)):
        if stock_sectors[i] != "US commodity":
            new_stocks_symbols.append(stocks_symbols[i])

    return new_stocks_symbols


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


def setStockSectors(stocksSymbols, sectorList) -> list:
    stock_sectors = []  # TODO - FIX ORDER
    for symbol in stocksSymbols:
        found_sector = False
        for sector in sectorList:
            if symbol in sector.stocks:
                stock_sectors.append(sector.name())
                found_sector = True
                break
        if not found_sector:
            stock_sectors.append(None)

    return stock_sectors


def makes_yield_column(_yield, weighted_sum_column):
    _yield.iloc[0] = 1
    for i in range(1, weighted_sum_column.size):
        change = weighted_sum_column.item(i) + 1
        last_value = _yield.iloc[i - 1]
        new_value = last_value * change
        _yield.iloc[i] = new_value
    return _yield


# functions for research

# plot the price of the stock with lower and upper bollinger band
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


def scan_good_stocks_with_filters():  # todo - make it work
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


def find_best_stocks(stocks_data):
    # Calculate the annual returns and volatility for each stock
    """returns = data['Adj Close'].pct_change().groupby(pd.Grouper(freq='Y')).apply(
        lambda x: (1 + x).prod() - 1).reset_index(level=0, drop=True)
    volatility = data['Adj Close'].pct_change().groupby(pd.Grouper(freq='Y')).std().reset_index(level=0, drop=True)

    # Calculate the Sharpe ratio for each stock
    sharpe = returns / volatility

    # Sort the stocks based on their Sharpe ratio
    sharpe_sorted = sharpe.sort_values()

    # Select the top 5 stocks with the highest Sharpe ratio
    top_5_stocks = sharpe_sorted.head(5)

    return top_5_stocks"""


# yfinance and israel tase api:
def get_israeli_indexes_data(command, start_date, end_date, israeliIndexes):  # get israeli indexes data from tase
    return tase_util.get_israeli_indexes_data(command, start_date, end_date, israeliIndexes)


def convert_usa_Index_to_full_name(UsaIndexes):
    UsaIndexesNames = {}
    for i, index in enumerate(UsaIndexes):
        ticker = yf.Ticker(index)
        UsaIndexesNames[i] = ticker.info["longName"]

    return UsaIndexesNames.values()


def convertIsraeliIndexToName(IsraliIndexes):
    hebrew_pattern = r"[\u0590-\u05FF\s]+"
    stocksNames = {}
    for (i, index) in enumerate(IsraliIndexes):
        text = getNameByIndexNumber(index)
        hebrew_parts = re.findall(hebrew_pattern, text)

        for hebrew_part in hebrew_parts:  # find all the Hebrew parts in the text
            hebrew_part_reversed = ''.join(reversed(hebrew_part))
            text = text.replace(hebrew_part, hebrew_part_reversed)

        stocksNames[i] = text
    return stocksNames.values()


def getNameByIndexNumber(indexNumber):

    # Load the JSON data from the file

    jsonData = get_json_data(settings.INDICES_LIST_JSON_NAME)
    result = [item['indexName'] for item in jsonData['indicesList']['result'] if item['indexId'] == indexNumber]

    # makes it from right to left

    name = result[0]
    return name
