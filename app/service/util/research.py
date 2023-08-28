import csv
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from watchlist.models import TopStock
from service.config import settings
from service.util import helpers, data_management
from service.util.helpers import Analyze
from service.util.graph import image_methods as graph_image_methods
from service.util.graph import plot_methods as graph_plot_methods

labels = [
    'Total Return Percentage',
    'Total Volatility Percentage',
    'Total Sharpe',
    'Annual Return Percentage',
    'Annual Volatility Percentage',
    'Annual Sharpe',
    'Monthly Return Percentage',
    'Monthly Volatility Percentage',
    'Monthly Sharpe',
    'Forecast Annual Return Percentage',
    'Forecast Annual Volatility Percentage',
    'Forecast Annual Sharpe'
]
ascending_list = [False, True, False, False, True, False, False, True, False, False, True, False]
row_selected = [0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0]


def save_user_specific_stock(stock: str, operation: str, plt_instance: plt) -> None:
    curr_user_directory = settings.RESEARCH_IMAGES
    # Saving files
    graph_image_methods.save_graph(plt_instance, file_name=curr_user_directory + stock + operation)
    plt.clf()
    plt.cla()
    plt.close()


def forecast_specific_stock(stock: str, machine_learning_model: str, models_data: dict, num_of_years_history: int,
                            start_date: str = None, end_date: str = None):
    plt = None
    file_name: str = f'{str(stock)}.csv'
    description = helpers.get_stocks_descriptions([stock])[1]
    if start_date is None or end_date is None:
        table = helpers.convert_data_to_tables(settings.RESEARCH_LOCATION, file_name,
                                               [stock], num_of_years_history, save_to_csv=False)
    else:
        table = helpers.convert_data_to_tables(
            location_saving=settings.RESEARCH_LOCATION,
            file_name=file_name,
            stocks_names=[stock],
            num_of_years_history=None,
            save_to_csv=False,
            start_date=start_date,
            end_date=end_date
        )
    record_percent_to_predict: float = models_data["record_percent_to_predict"]
    test_size_machine_learning: float = models_data["test_size_machine_learning"]
    analyze: Analyze = Analyze(
        returns_stock=table,
        table_index=table.index,
        record_percent_to_predict=float(record_percent_to_predict),
        is_closing_prices_mode=True
    )
    if machine_learning_model == settings.MACHINE_LEARNING_MODEL[0]:  # Linear Regression
        df, annual_return_with_forecast, excepted_returns = analyze.linear_regression_model(test_size_machine_learning)
    elif machine_learning_model == settings.MACHINE_LEARNING_MODEL[1]:  # Arima
        df, annual_return_with_forecast, excepted_returns = analyze.arima_model()
    elif machine_learning_model == settings.MACHINE_LEARNING_MODEL[2]:  # Gradient Boosting Regressor
        df, annual_return_with_forecast, excepted_returns = analyze.gbm_model()
    elif machine_learning_model == settings.MACHINE_LEARNING_MODEL[3]:  # Prophet
        df, annual_return_with_forecast, excepted_returns, plt = analyze.prophet_model()
    else:
        raise ValueError(f'Invalid machine learning model given: {machine_learning_model}!\n'
                         f'Pick one of:\n'
                         f'{settings.MACHINE_LEARNING_MODEL[0]}\n'
                         f'{settings.MACHINE_LEARNING_MODEL[1]}\n'
                         f'{settings.MACHINE_LEARNING_MODEL[2]}\n'
                         f'{settings.MACHINE_LEARNING_MODEL[3]}\n')

    plt_instance = graph_plot_methods.price_forecast(description, df, annual_return_with_forecast, excepted_returns,
                                                     plt)

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


def plot_bb_strategy_stock(stock_name: str, start="2009-01-01", end="2023-01-01") -> plt:
    """
    stock_name: str
        A name of a stock that is being plotted
    start: date
        A starting date for the graph_plot_methods
    end: date
        An ending date for the graph_plot_methods
    Return value:
        Returns the graph_plot_methods instance
    """

    if type(stock_name) == int or stock_name.isnumeric():  # israeli stock

        today = datetime.datetime.now()
        min_start_year = today.year - 10
        min_start_month = today.month
        min_start_day = today.day
        min_date = str(min_start_year) + "-" + str(min_start_month) + "-" + str(min_start_day)
        if start < min_date:
            start = min_date

        num_of_digits = len(str(stock_name))
        if num_of_digits > 3:
            is_index_type = False
        else:
            is_index_type = True
        stock_prices = helpers.get_israeli_symbol_data("get_past_10_years_history",
                                                       start, end, stock_name, is_index_type)
        # list to DataFrame
        stock_prices = pd.DataFrame(stock_prices)
        stock_prices["tradeDate"] = pd.to_datetime(stock_prices["tradeDate"])
        stock_prices.rename(columns={'tradeDate': 'Trade Date'}, inplace=True)
        stock_prices.set_index("Trade Date", inplace=True)

        if is_index_type:
            stock_prices['Adj Close'] = stock_prices[["closingIndexPrice"]]
        else:
            stock_prices['Adj Close'] = stock_prices[["closingPrice"]]

    else:  # US stock
        stock_prices = yf.download(stock_name, start, end)

    stock_prices['MA50'] = stock_prices['Adj Close'].rolling(window=50).mean()
    stock_prices['50dSTD'] = stock_prices['Adj Close'].rolling(window=50).std()
    stock_prices['Upper'] = stock_prices['MA50'] + (stock_prices['50dSTD'] * 2)
    stock_prices['Lower'] = stock_prices['MA50'] - (stock_prices['50dSTD'] * 2)

    stock_prices = stock_prices.iloc[51:]

    buy_price, sell_price, bb_signal = implement_bb_strategy(stock_prices['Adj Close'],
                                                             stock_prices['Lower'], stock_prices['Upper'])
    plt_instance = graph_plot_methods.bb_strategy_stock(stock_prices, buy_price, sell_price)

    return plt_instance


def download_data_for_research(num_of_years_history: int) -> None:
    """
    Downloading thousands of stocks to csv files - 10 years back
    """
    sectors_names_list = helpers.get_sectors_names_list()[7:]
    closing_price_all_sectors_table = pd.DataFrame()
    for i, sector_name in enumerate(sectors_names_list):
        stocks_symbols = helpers.get_stocks_symbols_list_by_sector(sector_name)
        converted_name = sector_name.replace(" ", "_") + "_closing_price"
        closing_price_table = helpers.convert_data_to_tables(settings.RESEARCH_LOCATION,
                                                             converted_name, stocks_symbols, num_of_years_history,
                                                             save_to_csv=True)
        print(f"table of sector::{sector_name} saved ")
        closing_price_all_sectors_table = pd.concat([closing_price_all_sectors_table, closing_price_table], axis=1)

    closing_price_all_sectors_table.to_csv(settings.RESEARCH_LOCATION + "all_sectors_closing_price.csv")
    print(f"table of all sectors: {sector_name} saved ")

    # creates


def get_stocks_stats(sector="US stocks indexes") -> list:
    """
    sector: str
        A single sector name
    """
    # get the relevant data
    data_pct_change = get_stocks_data_for_research_by_group(sector)
    interval_list = ["TOTAL", "Y", "M", "Y"]
    is_forecast_mode_list = [False, False, False, True]
    count = 0
    all_data_lists = []
    for i in range(len(interval_list)):
        stats_tuple = calculate_stats_of_stocks(data_pct_change, is_forecast_mode_list[i], interval_list[i])
        for j, stat_list in enumerate(stats_tuple):
            if row_selected[count] != 0:
                stat_list = stat_list.iloc[row_selected[count]]
            all_data_lists.append(stat_list)
            count += 1
    intercection = make_intersection(all_data_lists)

    return all_data_lists


def save_stocks_intersection_to_csv():
    sectors_list = helpers.get_sectors_names_list()
    unified_intersection_data = pd.DataFrame()
    for sector_name in sectors_list:
        all_stats_data_lists = get_stocks_stats(sector_name)
        intersection = make_intersection(all_stats_data_lists)
        # Replace spaces in sector_name with underscores for the file name
        sector_name_for_filename = sector_name.replace(" ", "_")
        intersection.to_csv(settings.RESEARCH_LOCATION + f'{sector_name_for_filename}' + "_intersection.csv")
        # Concatenate the current DataFrame with the unified DataFrame along columns
        unified_intersection_data = pd.concat([unified_intersection_data, intersection])

    # Save the unified DataFrame to a CSV file
    unified_intersection_data.to_csv(settings.RESEARCH_LOCATION + 'unified_intersection.csv')


def save_top_stocks_img_to_db(top_stocks: list, intersection_data, sector_name: str):
    data_management.plot_research_graphs(top_stocks, intersection_data, sector_name, labels)
    # Correct way to change value of a stock and its instance
    top_stock: TopStock = TopStock.objects.filter(sector_name=sector_name).first()  # Gets a stock from a certain sector
    prefix_str = "Top Stocks"
    # top_stock.img_src = f'{settings.RESEARCH_TOP_STOCKS_IMAGES}{prefix_str} {sector_name}.png'
    top_stock.img_src = f'{settings.RESEARCH_TOP_STOCKS_IMAGES}{prefix_str} {sector_name} intersection.png'
    top_stock.save()


def get_sorted_list_by_parameters(data_frame, ascending=False, filters=None,
                                  top_stocks_numbers=5000) -> list:
    """
    Filtering and organizing a subset of selected stocks
    """
    if filters is not None:
        min_value = filters[0]
        max_value = filters[1]
        data_frame = data_frame[(data_frame >= min_value) & (data_frame <= max_value)]

    return data_frame.sort_values(ascending=ascending).head(top_stocks_numbers)


def make_intersection(all_data):
    intersection = pd.concat(all_data, axis=1, join='inner')
    # Rename columns using the labels list
    intersection.columns = labels

    return intersection


def sort_good_stocks(all_data_lists, filters=None):
    """
    Filtering and organizing the selected stocks
    """
    if filters is not None:
        (minCap, maxCap, minAnnualReturns, maxAnnualVolatility, minAnnualSharpe, top_stocks_numbers,
         min_list_occurrences_intersections) = filters
        minFiltersList = [0, 0, 1,
                          minAnnualReturns, 0, minAnnualSharpe,
                          minAnnualReturns / 12, 0, minAnnualSharpe / 12,
                          minAnnualReturns, 0, minAnnualSharpe]

        maxFiltersList = [12000, 50, 500,
                          2000, maxAnnualVolatility, 500,
                          200, maxAnnualVolatility / 12, 500,
                          2000, maxAnnualVolatility, 500]
    else:
        min_list_occurrences_intersections = 0.0
        top_stocks_numbers = 5000

    # sort values and filters
    all_data_sorted = []
    for i in range(len(all_data_lists)):
        if filters is not None:
            filters = [minFiltersList[i], maxFiltersList[i]]
        all_data_sorted.append(get_sorted_list_by_parameters(all_data_lists[i],
                                                             ascending=ascending_list[i],
                                                             filters=filters,
                                                             top_stocks_numbers=top_stocks_numbers))

    intersection_without_filters = make_intersection(all_data_lists)
    intersection_with_filters = make_intersection(all_data_sorted)
    if intersection_with_filters.empty:
        # Flatten all the lists into a single list of indexes
        all_indexes = list(set(item for sublist in all_data_sorted for item in sublist.index))
        # Calculate the minimum number of lists that an index should appear in
        min_list_occurrences = int(min_list_occurrences_intersections * len(all_data_sorted))
        # Filter out indexes that appear in at least min_occurrences lists
        # Create a new DataFrame using the desired indexes
        num_of_lists = len(all_data_sorted)
        while num_of_lists >= min_list_occurrences:
            relevant_indexes = [index for index in all_indexes if
                                sum(index in sublist.index for sublist in all_data_sorted) >= num_of_lists]
            intersection_with_filters = intersection_without_filters.loc[relevant_indexes]
            if not intersection_with_filters.empty:
                break
            num_of_lists -= 1

    return all_data_sorted, intersection_with_filters, intersection_without_filters


def calculate_stats_of_stocks(data_pct_change, is_forecast_mode=False, interval="Y"):
    """
    Statistical calculation for each subgroup of shares (each of the 12 groups)
    """
    if is_forecast_mode:
        returns_annual_forecast = (((1 + data_pct_change.mean()) ** 254) - 1) * 100
        volatility_annual_forecast = (data_pct_change.std() * np.sqrt(254)) * 100
        sharpe_annual_forecast = returns_annual_forecast / volatility_annual_forecast

        return returns_annual_forecast, volatility_annual_forecast, sharpe_annual_forecast
    else:

        if interval == "TOTAL":
            profit_return = ((data_pct_change + 1).prod() - 1) * 100
            volatility = data_pct_change.std() * np.sqrt(254) * 100
            sharpe = profit_return / volatility
        else:
            profit_return = ((data_pct_change + 1).resample(interval).prod() - 1) * 100
            volatility = data_pct_change.groupby(pd.Grouper(freq=interval)).std() * 100
            sharpe = profit_return / volatility

    return profit_return, volatility, sharpe


def get_all_best_stocks(filters):
    """
    Input: parameters to filter
    Returns:
    - for each sector:
        12 tables with the best stocks, each of the 12 different tables defines a theme,
    - The stocks that overlap in all 12 tables (they are the best)

    """
    sectors_list = helpers.get_sectors_names_list()
    all_stats_data_list_of_lists = []
    unified_intersection_data = pd.DataFrame()
    for sector_name in sectors_list:
        all_data_lists = get_stocks_stats(sector_name)
        sorted_data_tuple, intersection_with_filters, intersection_without_filters = sort_good_stocks(
            all_data_lists, filters)
        save_top_stocks_img_to_db(sorted_data_tuple, intersection_with_filters, sector_name)
        all_stats_data_list_of_lists.append(all_data_lists)
        # Concatenate the current DataFrame with the unified DataFrame along columns
        unified_intersection_data = pd.concat([unified_intersection_data, intersection_with_filters])

    save_top_stocks_img_to_db(all_stats_data_list_of_lists, unified_intersection_data, "All")

    return all_stats_data_list_of_lists, unified_intersection_data


def get_stocks_data_for_research_by_group(sector_name: str) -> pd.DataFrame:
    """
    Input: sector name
    Returns: daily change table of all stocks in the sector
    """
    tickers_table = get_closing_price_table_by_sector(sector_name)
    tickers_df = pd.DataFrame(tickers_table)
    tickers_df.iloc[2:3] = np.nan
    # drop columns with almost all naN values
    tickers_df.dropna(axis=1, thresh=0.9 * len(tickers_df), inplace=True)
    tickers_df.dropna(inplace=True)
    # Set the first column as the index
    tickers_df.set_index(tickers_df.columns[0], inplace=True)
    data = tickers_df
    data = data.apply(pd.to_numeric, errors='coerce')
    data_pct_change = data.pct_change()
    data_pct_change.fillna(value=-0.0, inplace=True)
    # Convert the index to datetime
    data_pct_change.index = pd.to_datetime(data_pct_change.index)

    return data_pct_change


def get_closing_price_table_by_sector(sector_name: str):
    converted_name = sector_name.replace(" ", "_") + "_closing_price"
    return pd.read_csv(settings.RESEARCH_LOCATION + converted_name + ".csv")


def get_intersection_table_by_sector(sector_name: str):
    converted_name = sector_name.replace(" ", "_") + "_intersection"
    return pd.read_csv(settings.RESEARCH_LOCATION + converted_name + ".csv")


def get_top_stocks_by_label_and_sector(sector_name: str, label: str, ascending: bool = False, filters=None):
    pass


def update_collections_file():
    return None
