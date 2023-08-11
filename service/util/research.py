import numpy as np
import pandas as pd
import ta
import yfinance as yf

from .data_management import read_csv_file
from . import plot_functions, helpers
from ..config import settings
from .helpers import get_sectors_data_from_file, get_israeli_indexes_list


def save_user_specific_stock(stock: str, operation: str, plt_instance) -> None:
    # Creating directories
    curr_user_directory = settings.RESEARCH_RESULTS_LOCATION
    # Saving files
    plot_functions.save_graphs(plt_instance, file_name=curr_user_directory + stock  + operation)


def forecast_specific_stock(stock: str, machine_learning_model, models_data, num_of_years_history: int):
    plt = None
    file_name = str(stock) + '.csv'
    table = helpers.convert_data_to_tables(settings.RESEARCH_LOCATION, file_name,
                                           [stock], num_of_years_history, saveToCsv=False)
    RECORD_PERCENT_TO_PREDICT = models_data["RECORD_PERCENT_TO_PREDICT"]
    TEST_SIZE_MACHINE_LEARNING = models_data["TEST_SIZE_MACHINE_LEARNING"]
    if machine_learning_model == settings.MACHINE_LEARNING_MODEL[0]:
        df, annual_return, excepted_returns = helpers.analyze_with_machine_learning_linear_regression(table,
                                                                                                       table.index,
                                                                                                      RECORD_PERCENT_TO_PREDICT,
                                                                                                      TEST_SIZE_MACHINE_LEARNING,
                                                                                                       closing_prices_mode=True)
    elif machine_learning_model == settings.MACHINE_LEARNING_MODEL[1]:
        df, annual_return, excepted_returns = helpers.analyze_with_machine_learning_arima(table, table.index,
                                                                                          RECORD_PERCENT_TO_PREDICT,
                                                                                           closing_prices_mode=True)
    elif machine_learning_model == settings.MACHINE_LEARNING_MODEL[2]:
        df, annual_return, excepted_returns = helpers.analyze_with_machine_learning_gbm(table, table.index,
                                                                                        RECORD_PERCENT_TO_PREDICT,
                                                                                         closing_prices_mode=True)
    elif machine_learning_model == settings.MACHINE_LEARNING_MODEL[3]:
        df, annual_return, excepted_returns, plt = helpers.analyze_with_machine_learning_prophet(table, table.index,
                                                                                                 RECORD_PERCENT_TO_PREDICT,
                                                                                                  closing_prices_mode=True)

    plt_instance = plot_functions.plot_price_forecast(stock, df, annual_return, plt)
    return plt_instance


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


def plotbb_strategy_stock(stock_name: str, start="2009-01-01", end="2023-01-01"):
    if type(stock_name) == int or stock_name.isnumeric():
        num_of_digits = len(str(stock_name))
        if num_of_digits > 3:
            is_index_type = False
        else:
            is_index_type = True
        stock_prices = helpers.get_israeli_symbol_data("get_past_10_years_history",
                                     start, end, stock_name, is_index_type)
        # list to dateframe
        stock_prices = pd.DataFrame(stock_prices)
        stock_prices["tradeDate"] = pd.to_datetime(stock_prices["tradeDate"])
        stock_prices.set_index("tradeDate", inplace=True)
        if is_index_type:
            stock_prices['Adj Close'] = stock_prices[["closingIndexPrice"]]
        else:
            stock_prices['Adj Close'] = stock_prices[["closingPrice"]]
    else:
        stock_prices = yf.download(stock_name, start, end)


    stock_prices['MA50'] = stock_prices['Adj Close'].rolling(window=50).mean()
    stock_prices['50dSTD'] = stock_prices['Adj Close'].rolling(window=50).std()
    stock_prices['Upper'] = stock_prices['MA50'] + (stock_prices['50dSTD'] * 2)
    stock_prices['Lower'] = stock_prices['MA50'] - (stock_prices['50dSTD'] * 2)

    stock_prices = stock_prices.dropna()
    stock_prices = stock_prices.iloc[51:]

    buy_price, sell_price, bb_signal = implement_bb_strategy(stock_prices['Adj Close'],
                                                                      stock_prices['Lower'], stock_prices['Upper'])
    plt_instance = plot_functions.plotbb_strategy_stock(stock_prices, buy_price, sell_price)
    return plt_instance


def plotbb_strategy_portfolio(pct_change_table, new_portfolio):  # TODO
    plt_instance = plot_functions.plotbb_strategy_portfolio(pct_change_table, new_portfolio)

    return plt_instance

def download_data_for_research(num_of_years_history: int) -> None: # TODO
    stocks_symbols = []

    sectors_data = get_sectors_data_from_file()
    israel_indexes_list = get_israeli_indexes_list()  # get from here data
    usa_indexes_list = sectors_data[3:5]["stocks"]
    israel_stocks_list = sectors_data[6]["stocks"]
    usa_stocks_list = sectors_data[7]["stocks"]

    helpers.convert_data_to_tables(settings.RESEARCH_LOCATION,
                                   'usa_stocks_closing_prices', usa_stocks_list, num_of_years_history, saveToCsv=True)

    helpers.convert_data_to_tables(settings.RESEARCH_LOCATION,
                                   'israel_indexes_closing_prices', israel_indexes_list, num_of_years_history,
                                   saveToCsv=True)

    stocks_symbols.extend(usa_stocks_list)
    stocks_symbols.extend(israel_indexes_list)
    helpers.convert_data_to_tables(settings.RESEARCH_LOCATION,
                                   'all_closing_prices', stocks_symbols, num_of_years_history, saveToCsv=True)


def find_good_stocks(group_of_stocks="usa_stocks", filter_option=False):  # TODO - fix
    max_returns_stocks_list = None
    min_volatility_stocks_list = None
    max_sharpest_stocks_list = None

    tickers_table = get_stocks_data_for_research_by_group(group_of_stocks)
    tickers_df = pd.DataFrame(tickers_table)
    tickers_df.iloc[2:3] = np.nan
    tickers_df.dropna(inplace=True)
    tickers_df.index.name = "Date"
    data = tickers_df.set_index(0)
    data.columns = data.iloc[0]
    data = data.iloc[1:]
    data = pd.DataFrame(data, columns=data.columns)
    data = data.rename_axis('Date')
    data = data.apply(pd.to_numeric, errors='coerce')

    data_pct_change = data.pct_change()
    data_pct_change.fillna(value=-0.0, inplace=True)

    # Convert the index to datetime
    data_pct_change.index = pd.to_datetime(data_pct_change.index)
    annual_returns = ((data_pct_change + 1).resample('Y').prod() - 1) * 100
    total_profit_return = ((data_pct_change + 1).prod() - 1) * 100
    total_volatility = data_pct_change.std() * np.sqrt(254)
    annual_volatility = data_pct_change.groupby(pd.Grouper(freq='Y')).std()
    sharpe = annual_returns / annual_volatility

    # forcast from total:
    returns_annual_forcecast = (((1 + data_pct_change.mean()) ** 254) - 1) * 100
    cov_daily = data_pct_change.cov()
    cov_annual = cov_daily * 254
    volatility_forecast = (data_pct_change.std() * np.sqrt(254)) * 100
    sharpe_forecast = returns_annual_forcecast / volatility_forecast

    total_sharpe = total_profit_return / total_volatility

    # sort total
    total_profit_return_sorted = total_profit_return.sort_values(ascending=False).head(10)
    total_volatility_sorted = total_volatility.sort_values(ascending=True).head(10)
    total_sharpe_sorted = total_sharpe.sort_values(ascending=False).head(10)

    print("total_profit_return_sorted")
    print(total_profit_return_sorted)
    print("total_volatility_sorted")
    print(total_volatility_sorted)
    print("total_sharpe_sorted")
    print(total_sharpe_sorted)

    # sort last year
    annual_sharpe_sorted = sharpe[-1].sort_values().head(10)
    annual_returns_sorted = annual_returns[-1].sort_values().head(10)
    annual_volatility_sorted = annual_volatility[-1].sort_values().head(10)
    annual_sharpe_sorted = annual_sharpe_sorted[-1].sort_values().head(10)
    print("annual_returns_sorted")
    print(annual_returns_sorted)
    print("annual_volatility_sorted")
    print(annual_volatility_sorted)
    print("annual_sharpe_sorted")
    print(annual_sharpe_sorted)

    # sort forecast
    returns_annual_forcecast_sorted = returns_annual_forcecast.sort_values().head(10)
    volatility_forecast_sorted = volatility_forecast.sort_values().head(10)
    sharpe_forecast_sorted = sharpe_forecast.sort_values().head(10)

    """max_sharpest_stocks_list = sharpe_sorted.head(10)
    max_returns_stocks_list = returns_sorted.head(10)
    min_volatility_stocks_list = min_volatility_sorted.tail(10)"""

    # plot_functions.plot_top_stocks(helpers.scan_good_stocks())
    # plot_functions.plot(plt_instance)  # TODO plot at site
    # plot_functions.plot(plt_instance)  # TODO plot at site

    return max_returns_stocks_list, min_volatility_stocks_list, max_sharpest_stocks_list


def get_stocks_data_for_research_by_group(group_of_stocks: str):
    if group_of_stocks == "nasdaq":  # TODO -fix
        nasdaq_tickers = yf.Tickers('^IXIC').tickers
        # Convert the dictionary of Ticker objects to a list of Ticker objects
        ticker_list = list(nasdaq_tickers.values())
        tickers = [ticker.ticker for ticker in ticker_list]
        filtered_stocks = tickers[
            (tickers['marketCap'] >= 1e9)  # &  # Stocks with a market capitalization of at least $1 billion
            # (tickers['ipoyear'] <= 2020)  # Stocks that went public before or in 2020
        ]
        # Get the ticker symbols for the filtered stocks
        tickers = filtered_stocks['symbol'].to_list()
    elif group_of_stocks == "sp500":  # TODO -fix
        sp500_tickers = yf.Tickers('^GSPC').tickers
        # Convert the dictionary of Ticker objects to a list of Ticker objects
        ticker_list = list(sp500_tickers.values())
        tickers = [ticker.ticker for ticker in ticker_list]
        filtered_stocks = tickers[
            (tickers['marketCap'] >= 1e9)  # &  # Stocks with a market capitalization of at least $1 billion
        ]
        # Get the ticker symbols for the filtered stocks
        tickers = filtered_stocks['symbol'].to_list()
    elif group_of_stocks == "dowjones":  # TODO -fix
        dowjones_tickers = yf.Tickers('^DJI').tickers
        # Convert the dictionary of Ticker objects to a list of Ticker objects
        ticker_list = list(dowjones_tickers.values())
        tickers = [ticker.ticker for ticker in ticker_list]
        filtered_stocks = tickers[
            (tickers['marketCap'] >= 1e9)  # &  # Stocks with a market capitalization of at least $1 billion
        ]
        # Get the ticker symbols for the filtered stocks
        tickers = filtered_stocks['symbol'].to_list()
    elif group_of_stocks == "TA35":  # TODO -fix
        TA35_tickers = yf.Tickers('^TA35.TA').tickers
        # Convert the dictionary of Ticker objects to a list of Ticker objects
        ticker_list = list(TA35_tickers.values())
        tickers = [ticker.ticker for ticker in ticker_list]
        """filtered_stocks = tickers[
            (tickers['marketCap'] >= 1e9) # &  # Stocks with a market capitalization of at least $1 billion
        ]"""
        # Get the ticker symbols for the filtered stocks
        tickers = tickers['symbol'].to_list()
    elif group_of_stocks == "TA90":  # TODO -fix
        TA90_tickers = yf.Tickers('^TA90.TA').tickers
        # Convert the dictionary of Ticker objects to a list of Ticker objects
        ticker_list = list(TA90_tickers.values())
        tickers = [ticker.ticker for ticker in ticker_list]
        """filtered_stocks = tickers[
            (tickers['marketCap'] >= 1e9) # &  # Stocks with a market capitalization of at least $1 billion
        ]"""
        # Get the ticker symbols for the filtered stocks
        tickers = tickers['symbol'].to_list()
    elif group_of_stocks == "TA125":  # TODO -fix
        TA125_tickers = yf.Tickers('^TA125.TA').tickers
        # Convert the dictionary of Ticker objects to a list of Ticker objects
        ticker_list = list(TA125_tickers.values())
        tickers = [ticker.ticker for ticker in ticker_list]
        """filtered_stocks = tickers[
            (tickers['marketCap'] >= 1e9) # &  # Stocks with a market capitalization of at least $1 billion
        ]"""
        # Get the ticker symbols for the filtered stocks
        tickers = tickers['symbol'].to_list()

    elif group_of_stocks == "usa_bonds":
        tickers = read_csv_file(settings.RESEARCH_LOCATION + 'usa_stocks_closing_prices.csv')

    elif group_of_stocks == "usa_stocks":
        tickers = read_csv_file(settings.RESEARCH_LOCATION + 'usa_stocks_closing_prices.csv')

    elif group_of_stocks == "israel_indexes":
        tickers = read_csv_file(settings.RESEARCH_LOCATION + 'israel_indexes_closing_prices.csv')

    elif group_of_stocks == "all":
        tickers = read_csv_file(settings.RESEARCH_LOCATION + 'all_closing_prices.csv')


    else:
        pass

    return tickers




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