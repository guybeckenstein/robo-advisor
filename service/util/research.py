import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from service.config import settings
from service.util import helpers, data_management
from service.util.graph import plot_methods as graph_plot_methods

# Global static variables
LABELS: list[str] = [
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
groups: list = ["all", "annual_return", "volatility", "sharpe"]


def save_user_specific_stock(stock: str, operation: str, plt_instance: plt) -> None:
    from service.util.graph import image_methods as graph_image_methods
    # Saving files
    graph_image_methods.save_graph_locally(plt_instance, file_name=f'{settings.RESEARCH_IMAGES}{stock}{operation}')
    plt.clf()
    plt.cla()
    plt.close()


def forecast_specific_stock(stock: str, machine_learning_model: str, models_data: dict, num_of_years_history: int,
                            start_date: str = None, end_date: str = None):
    from service.util.helpers import Analyze

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
    elif machine_learning_model == settings.MACHINE_LEARNING_MODEL[1]:  # ARIMA
        df, annual_return_with_forecast, excepted_returns = analyze.arima_model()
    elif machine_learning_model == settings.MACHINE_LEARNING_MODEL[2]:  # Gradient Boosting Regressor
        df, annual_return_with_forecast, excepted_returns = analyze.lstm_model()
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
        A starting date for the graph
    end: date
        An ending date for the graph
    Return value:
        Returns the graph instance
    """

    if isinstance(stock_name, int) or stock_name.isnumeric():  # israeli stock

        today = datetime.datetime.now()
        min_start_year = today.year - 10
        min_start_month = today.month
        min_start_day = today.day
        min_date: str = f"{min_start_year}-{min_start_month}-{min_start_day}"
        if start < min_date:
            start = min_date

        num_of_digits = len(str(stock_name))
        if num_of_digits > 3:
            is_index_type: bool = False
        else:
            is_index_type: bool = True
        stock_prices = helpers.get_israeli_symbol_data(
            "get_past_10_years_history", start, end, stock_name, is_index_type
        )
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
    sectors_names_list = helpers.get_sectors_names_list()[:]
    closing_price_all_sectors_table = pd.DataFrame()
    sector_name: str = None
    for i, sector_name in enumerate(sectors_names_list):
        stocks_symbols = helpers.get_stocks_symbols_list_by_sector(sector_name)
        converted_name: str = f'{sector_name.replace(" ", "_")}_closing_price'
        closing_price_table = helpers.convert_data_to_tables(
            settings.RESEARCH_LOCATION, converted_name, stocks_symbols, num_of_years_history, save_to_csv=True
        )
        print(f"table of sector::{sector_name} saved ")
        closing_price_all_sectors_table = pd.concat([closing_price_all_sectors_table, closing_price_table], axis=1)

    closing_price_all_sectors_table.to_csv(f"{settings.RESEARCH_LOCATION}all_sectors_closing_price.csv")
    print(f"table of all sectors: {sector_name} saved ")  # TODO: unknown variable

    # creates


def get_stocks_stats(sector: str = "US Stocks Indexes") -> list:
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

    return all_data_lists


def save_stocks_intersection_to_csv():
    sectors_list = helpers.get_sectors_names_list()
    unified_intersection_data = pd.DataFrame()
    for sector_name in sectors_list:
        all_stats_data_lists = get_stocks_stats(sector_name)
        intersection = create_intersection(all_stats_data_lists)
        # Replace spaces in sector_name with underscores for the file name
        intersection.to_csv(f'{settings.RESEARCH_LOCATION}{sector_name.replace(__old=" ", __new="_")}_intersection.csv')
        # Concatenate the current DataFrame with the unified DataFrame along columns
        unified_intersection_data = pd.concat([unified_intersection_data, intersection])

    # Save the unified DataFrame to a CSV file
    unified_intersection_data.to_csv(f'{settings.RESEARCH_LOCATION}unified_intersection.csv')


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


def create_intersection(data, df_columns: list[str] = None):
    intersection = pd.concat(data, axis=1, join='inner')
    # Rename columns using the labels list
    if df_columns:
        intersection.columns = df_columns
    else:
        intersection.columns = LABELS

    return intersection


def make_intersection_by_group(all_data_sorted, group_name, min_list_occurrences_intersections,
                               intersection_without_filters):
    if group_name == groups[0]:
        data_sorted = all_data_sorted
        intersection_with_filters = create_intersection(all_data_sorted)
        df_columns = LABELS
    elif group_name == groups[1]:  # annual
        data_sorted = [all_data_sorted[0], all_data_sorted[3], all_data_sorted[6], all_data_sorted[9]]
        df_columns = [LABELS[0], LABELS[3], LABELS[6], LABELS[9]]
        intersection_with_filters = create_intersection(data_sorted, df_columns)
    elif group_name == groups[2]:  # volatility
        data_sorted = [all_data_sorted[1], all_data_sorted[4], all_data_sorted[7], all_data_sorted[10]]
        df_columns = [LABELS[1], LABELS[4], LABELS[7], LABELS[10]]
        intersection_with_filters = create_intersection(data_sorted, df_columns)
    elif group_name == groups[3]:  # sharpe
        data_sorted = [all_data_sorted[2], all_data_sorted[5], all_data_sorted[8], all_data_sorted[11]]
        df_columns = [LABELS[2], LABELS[5], LABELS[8], LABELS[11]]
        intersection_with_filters = create_intersection(data_sorted, df_columns)
    else:
        raise AttributeError
    intersection_without_filters = intersection_without_filters[df_columns]

    if intersection_with_filters.empty:
        # Flatten all the lists into a single list of indexes
        all_indexes = list(set(item for sublist in data_sorted for item in sublist.index))
        # Calculate the minimum number of lists that an index should appear in
        min_list_occurrences = int(min_list_occurrences_intersections * len(data_sorted))
        # Filter out indexes that appear in at least min_occurrences lists
        # Create a new DataFrame using the desired indexes
        num_of_lists = len(data_sorted)
        while num_of_lists >= min_list_occurrences:
            relevant_indexes = [index for index in all_indexes if
                                sum(index in sublist.index for sublist in data_sorted) >= num_of_lists]
            intersection_with_filters = intersection_without_filters.loc[relevant_indexes]
            if not intersection_with_filters.empty:
                break
            num_of_lists -= 1

    return intersection_with_filters


def sort_good_stocks(all_data_lists, filters=None) -> tuple:
    """
    Filtering and organizing the selected stocks
    """
    if filters is not None:
        (minCap, maxCap, minAnnualReturns, maxAnnualVolatility, minAnnualSharpe, top_stocks_numbers,
         min_list_occurrences_intersections) = filters
        min_filters_list: list = [
            0, 0, 0.5, minAnnualReturns, -10, minAnnualSharpe, minAnnualReturns / 12, -10, minAnnualSharpe,
            minAnnualReturns, -10, minAnnualSharpe
        ]
        max_filters_list: list = [
            2000, 50, 100, 200, maxAnnualVolatility, 30, 200, maxAnnualVolatility / 12, 30, 200,
            maxAnnualVolatility, 30
        ]
    else:
        min_filters_list: list = None
        max_filters_list: list = None
        min_list_occurrences_intersections = 0.0
        top_stocks_numbers = 5000

    # sort values and filters
    all_data_sorted = []
    table_groups_list = []
    for i in range(len(all_data_lists)):
        if filters is not None:
            filters = [min_filters_list[i], max_filters_list[i]]
        all_data_sorted.append(get_sorted_list_by_parameters(
            all_data_lists[i], ascending=ascending_list[i], filters=filters, top_stocks_numbers=top_stocks_numbers
        ))

    table_without_filters = create_intersection(all_data_lists)
    # makes intersections per groups
    for group_name in groups:
        table_groups_list.append(make_intersection_by_group(
            all_data_sorted, group_name, min_list_occurrences_intersections, table_without_filters
        ))

    return table_groups_list[1:], table_groups_list[0], table_without_filters


def make_union_of_table_groups(data_tuple_list):
    annual_returns_intersection_data = data_tuple_list[0].sort_values(by=LABELS[6], ascending=ascending_list[6]).head(3)
    volatility_intersection_data = data_tuple_list[1].sort_values(by=LABELS[7], ascending=ascending_list[7]).head(3)
    sharpe_intersection_data = data_tuple_list[2].sort_values(by=LABELS[8], ascending=ascending_list[8]).head(3)
    resulting_dataframe = pd.concat(
        [annual_returns_intersection_data, volatility_intersection_data, sharpe_intersection_data])

    return resulting_dataframe


def calculate_stats_of_stocks(data_pct_change, is_forecast_mode=False, interval="Y"):
    """
    Statistical calculation for each subgroup of shares (each of the 12 groups)
    """
    if is_forecast_mode:
        returns_annual_forecast = (((1 + data_pct_change.mean()) ** 254) - 1) * 100
        volatility_annual_forecast = data_pct_change.std() * np.sqrt(254) * 100
        sharpe_annual_forecast = returns_annual_forecast / volatility_annual_forecast

        return returns_annual_forecast, volatility_annual_forecast, sharpe_annual_forecast
    else:

        if interval == "TOTAL":
            profit_return = ((data_pct_change + 1).prod() - 1) * 100
            volatility = data_pct_change.std() * 100
            sharpe = profit_return / volatility
        else:
            profit_return = ((data_pct_change + 1).resample(interval).prod() - 1) * 100
            volatility = data_pct_change.groupby(pd.Grouper(freq=interval)).std() * 100
            sharpe = profit_return / volatility

    return profit_return, volatility, sharpe


def get_all_best_stocks() -> tuple[list[list[pd.Series]], pd.DataFrame, tuple[pd.DataFrame]]:
    """
    Input: parameters to filter
    Returns:
    - for each sector:
        12 tables with the best stocks, each of the 12 different tables defines a theme,
    - The stocks that overlap in all 12 tables (they are the best)

    """
    sectors_list = helpers.get_sectors_names_list()
    all_stats_data_list_of_lists = []
    unified_table_data: pd.DataFrame = pd.DataFrame()
    unified_table_data_tuple: pd.DataFrame = pd.DataFrame()
    annual_returns_table_data: pd.DataFrame = pd.DataFrame()
    volatility_table_data: pd.DataFrame = pd.DataFrame()
    sharpe_table_data: pd.DataFrame = pd.DataFrame()

    for sector_name in sectors_list:
        all_data_lists = get_stocks_stats(sector_name)
        all_stats_data_list_of_lists.append(all_data_lists)

        sorted_data_tuple, filtered_table, unfiltered_table = sort_good_stocks(
            all_data_lists, settings.RESEARCH_FILTERS
        )

        # Concatenate the current DataFrame with the unified DataFrame along columns
        annual_returns_table_data = pd.concat([annual_returns_table_data, sorted_data_tuple[0]])
        volatility_table_data = pd.concat([volatility_table_data, sorted_data_tuple[1]])
        sharpe_table_data = pd.concat([sharpe_table_data, sorted_data_tuple[2]])

        sorted_data_tuple = make_union_of_table_groups(sorted_data_tuple)
        data_management.plot_research_graphs(filtered_table, sector_name, LABELS)

        unified_table_data = pd.concat([unified_table_data, filtered_table])
        unified_table_data_tuple = pd.concat([unified_table_data_tuple, sorted_data_tuple])

    sorted_data_tuple = make_union_of_table_groups(
        [annual_returns_table_data, volatility_table_data, sharpe_table_data]
    )

    data_management.plot_research_graphs(unified_table_data, "All", LABELS)
    unified_table_data_list: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] = [
        annual_returns_table_data, volatility_table_data, sharpe_table_data
    ]
    return all_stats_data_list_of_lists, unified_table_data, unified_table_data_list


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


def get_closing_price_table_by_sector(sector_name: str) -> str:
    return pd.read_csv(f'{settings.RESEARCH_LOCATION}{sector_name.replace(" ", "_")}_closing_price.csv')


def get_intersection_table_by_sector(sector_name: str = "unified"):
    return pd.read_csv(f'{settings.RESEARCH_LOCATION}{sector_name.replace(" ", "_")}_intersection.csv')


def get_top_stocks_by_label_and_sector(sector_name: str = "unified", label: str = LABELS[0], ascending: bool = False,
                                       top_stocks_numbers: int = 10):
    intersection_table = get_intersection_table_by_sector(sector_name)
    return intersection_table.sort_values(by=label, ascending=ascending).head(top_stocks_numbers)


def update_collections_file(all_stats_data_list_of_lists: list[list[pd.Series]],
                            unified_table_data: pd.DataFrame) -> None:
    """
    updates stocks.json file with the new stocks according to the research
    :return: None
    """
    # get best of bests
    best_stocks_intersections = []
    for i in range(len(LABELS)):
        best_stocks_intersections.append(list(
            unified_table_data.sort_values(by=LABELS[i], ascending=ascending_list[i]).head(10).index)
        )
    # get intersection stocks
    # TODO: unused
    # stocks_tables: list[pd.DataFrame, pd.DataFrame, pd.DataFrame] = [
    #     unified_table_data_tuple[idx1].sort_values(by=LABELS[idx2], ascending=ascending_list[idx2]).head(10).index
    #     for idx1, idx2 in [(0, 9), (1, 10), (2, 11)]
    # ]
    # best_forecast_return_table: pd.DataFrame = stocks_tables[0]
    # safest_stocks_table: pd.DataFrame = stocks_tables[1]
    # sharpe_stocks_table: pd.DataFrame = stocks_tables[2]

    # get best of specific value
    top_indexes_annual_return: list = []
    top_indexes_annual_sharpe: list = []
    top_indexes_monthly_return: list = []
    top_stocks_annual_return: list = []
    top_stocks_annual_sharpe: list = []
    top_stocks_volatility: list = []
    top_stocks_monthly_return: list = []
    for i, sub_list in enumerate(all_stats_data_list_of_lists[0:6]):
        top_indexes_annual_return += list(sub_list[3].sort_values(ascending=ascending_list[4]).head(2).index)
        top_indexes_annual_sharpe += list(sub_list[5].sort_values(ascending=ascending_list[5]).head(2).index)
        top_indexes_monthly_return += list(sub_list[6].sort_values(ascending=ascending_list[6]).head(2).index)

    for i, sub_list in enumerate(all_stats_data_list_of_lists[6:8]):
        top_stocks_annual_return += list(sub_list[3].sort_values(ascending=ascending_list[3]).head(30).index)
        top_stocks_annual_sharpe += list(sub_list[5].sort_values(ascending=ascending_list[5]).head(30).index)
        top_stocks_volatility += list(sub_list[4].sort_values(ascending=ascending_list[4]).head(30).index)
        top_stocks_monthly_return += list(sub_list[6].sort_values(ascending=ascending_list[6]).index)

    # save to collections file
    # Find the intersection of the four lists
    collections_file = helpers.get_json_data(f'{settings.DATASET_LOCATION}stocks')

    collections_file['collections']['2'][0]["stocksSymbols"] = list(
        set(top_indexes_annual_return + top_indexes_annual_sharpe)
    )

    # save collection file
    helpers.save_json_data(f"{settings.DATASET_LOCATION}stocks", collections_file)

    # upload to google drive
    data_management.upload_file_to_google_drive(file_path=f"{settings.DATASET_LOCATION}stocks.json", num_of_elements=2)


def update_stocks_names_tables():  # update stocks tables with the new stocks according to the research
    sectors_names = helpers.get_sectors_names_list()  # sectors.json.json.json
    unified_intersection_data = get_intersection_table_by_sector("unified")  # updated data
    symbols_list_unified_tables = unified_intersection_data[unified_intersection_data.columns[0]].to_list()
    all_basics_data = helpers.get_all_stocks_table()
    # update sectors.json.json.json file
    sectors_json_file = helpers.get_json_data(settings.SECTORS_JSON_NAME)
    for i, sector_name in enumerate(sectors_names):
        stocks_list = sectors_json_file['sectorsList']['result'][i]['stocks']
        for stock in stocks_list:
            if str(stock) not in symbols_list_unified_tables:
                print("stock not in unified_intersection_data: ", stock)
                stocks_list.remove(stock)
                # Remove rows from all_basics_data
                all_basics_data = all_basics_data[all_basics_data['Symbol'] != str(stock)]
        sectors_json_file['sectorsList']['result'][i]['stocks'] = stocks_list
    # Save the updated sectors.json.json.json file
    helpers.save_json_data(settings.SECTORS_JSON_NAME, sectors_json_file)
    # Save the updated all_basics_data DataFrame back to CSV
    all_basics_data.to_csv(f"{settings.CONFIG_RESOURCE_LOCATION}all_stocks_basic_data.csv", index=False)


def upload_top_stocks_to_google_drive() -> None:
    # update top stocks images
    sectors_names = helpers.get_sectors_names_list()
    sectors_names.append("All")
    prefix_str = 'Top Stocks - '
    for sector_name in sectors_names:
        fully_qualified_image_name_base: str = f'{settings.RESEARCH_IMAGES}{prefix_str}{sector_name}'
        for file_path_suffix in ['Graphs', 'Table']:
            data_management.upload_file_to_google_drive(
                file_path=f'{fully_qualified_image_name_base} ({file_path_suffix}).png', num_of_elements=2
            )
