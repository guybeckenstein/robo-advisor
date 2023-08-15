import matplotlib

from service.impl.sector import Sector
import scipy.stats as stats
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image

from typing import List

# Global Const Variables
FIG_SIZE1: tuple[int, int] = (10, 8)
FIG_SIZE2: tuple[int, int] = (10, 4)
FIG_SIZE3: tuple[int, int] = (8, 4)
FIG_SIZE4: tuple[int, int] = (10, 6)
STYLE = 'oblique'
HA: str = 'center'
VA: str = 'center'
FONT_NAME: str = 'Arial'
LINE_STYLE = 'dashed'
FONT_WEIGHT = 'bold'
WRAP: bool = True
GRID: bool = True
BOTTOM: float = 0.4
ALPHA: float = 0.5


def plot_markowitz_graph(sectors: List[Sector], three_best_sectors_weights, min_variance_portfolio, sharpe_portfolio,
                         max_returns, max_vols_portfolio, df: pd.DataFrame) -> plt:
    # plot frontier, max sharpe & min Volatility values with a scatterplot
    create_scatter_plot(
        df=df,
        x='Volatility',
        y='Returns',
        sharpe_portfolio=sharpe_portfolio,
        min_variance_portfolio=min_variance_portfolio,
        max_vols_portfolio=max_vols_portfolio,
    )

    # ------------------ Printing 3 optimal Portfolios -----------------------
    # Setting max_X, max_Y to act as relative border for window size
    stocks: list[str] = ['', '', '']

    for i in range(len(sectors)):
        for j in range(3):
            weight = three_best_sectors_weights[2 - j][i] * 100
            stocks[j] += sectors[i].name + "(" + str("{:.2f}".format(weight)) + "%),\n"

    portfolios: list = [max_returns.iloc[0], sharpe_portfolio.iloc[0], min_variance_portfolio.iloc[0]]
    colors: list[str] = ['red', 'green', 'yellow']

    with pd.option_context("display.float_format", "%{:,.2f}".format):
        for i in range(len(portfolios)):
            x: float = 0.2 + 0.25 * i
            plt.figtext(
                x=x,
                y=0.15,
                s=f"Max Returns Portfolio:\n"
                  f"Annual Returns: {str(round(portfolios[i][0], 2))}%\n"
                  f"Annual Volatility: {str(round(portfolios[i][1], 2))}%\n"
                  f"Annual Max Loss: {str(round(portfolios[i][0] - 1.65 * portfolios[i][1], 2))}%\n"
                  f"Sharpe Ratio: {str(round(portfolios[i][2], 2))}\n"
                  f"{stocks[i]}",
                bbox=dict(facecolor=colors[i], alpha=ALPHA), fontsize=11, style=STYLE, ha=HA, va=VA, fontname=FONT_NAME,
                wrap=WRAP,
            )

    return plt


def create_scatter_plot(
        df, x: str, y: str, sharpe_portfolio, min_variance_portfolio, max_vols_portfolio, max_ginis=None
):
    plt.style.use("seaborn-dark")
    df.plot.scatter(
        x=x, y=y, c="Sharpe Ratio", cmap="RdYlGn", edgecolors="black", figsize=FIG_SIZE1, grid=GRID,
    )
    portfolios: list = [sharpe_portfolio, min_variance_portfolio, max_vols_portfolio]
    colors: list[str] = ['green', 'orange', 'red']
    for i in range(3):
        plt.scatter(x=portfolios[i][x], y=portfolios[i][y], c=colors[i], marker="D", s=200)
    plt.style.use("seaborn-dark")
    plt.xlabel(f"{x} (Std. Deviation) Percentage %")
    plt.ylabel(f"Expected {y} Percentage %")
    plt.title("Efficient Frontier")
    plt.subplots_adjust(bottom=BOTTOM)


def plot_gini_graph(sectors: list[Sector], three_best_sectors_weights, min_variance_portfolio, sharpe_portfolio,
                    max_portfolios_annual, max_ginis, df: pd.DataFrame) -> plt:
    # plot frontier, max sharpe & min Gini values with a scatterplot
    create_scatter_plot(
        df=df,
        x='Gini',
        y='Portfolio Annual',
        sharpe_portfolio=sharpe_portfolio,
        min_variance_portfolio=min_variance_portfolio,
        max_vols_portfolio=max_portfolios_annual,
        max_ginis=max_ginis,
    )

    # ------------------ Printing 3 optimal Portfolios -----------------------
    # Setting max_X, max_Y to act as relative border for window size
    stocks: list[str] = ['', '', '']
    for i in range(len(sectors)):
        for j in range(3):
            weight = three_best_sectors_weights[2 - j][i] * 100
            stocks[j] += sectors[i].name + "(" + str("{:.2f}".format(weight)) + "%),\n"

    portfolios: list = [max_portfolios_annual.iloc[0], sharpe_portfolio.iloc[0], min_variance_portfolio.iloc[0]]
    colors: list[str] = ['red', 'green', 'yellow']

    with pd.option_context("display.float_format", "%{:,.2f}".format):
        for i in range(len(colors)):
            x = 0.2 + 0.25 * i
            plt.figtext(
                x=x,
                y=0.15,
                s=f"Max Returns Portfolio:\n"
                  f"Annual Returns: {str(round(portfolios[i][0], 2))}%\n"
                  f"Annual Gini: {str(round(portfolios[i][1], 2))}%\n"
                  f"Annual Max Loss: {str(round(portfolios[i][0] - 1.65 * portfolios[i][1], 2))}%\n"
                  f"Sharpe Ratio: {str(round(portfolios[i][2], 2))}\n"
                  f"{stocks[i]}",
                bbox=dict(facecolor=colors[i], alpha=ALPHA), fontsize=11, style=STYLE, ha=HA, va=VA, fontname=FONT_NAME,
                wrap=WRAP,
            )

    return plt


def plot_bb_strategy_stock(stock_prices, buy_price, sell_price) -> plt:
    stock_prices[['Adj Close', 'Lower', 'Upper']].plot(figsize=FIG_SIZE2)
    plt.scatter(stock_prices.index, buy_price, marker='^', color='green', label='BUY', s=200)
    plt.scatter(stock_prices.index, sell_price, marker='v', color='red', label='SELL', s=200)

    print("Number of green:")
    print(np.count_nonzero(~np.isnan(buy_price)))
    print("Number of red:")
    print(np.count_nonzero(~np.isnan(sell_price)))

    return plt


def plot_bb_strategy_portfolio(stock_prices, buy_price, sell_price, new_portfolio) -> plt:
    plt.figure()  # Create a new plot instance
    stock_prices[['Adj Close', 'Lower', 'Upper']].plot(figsize=FIG_SIZE2)
    s: int = 200
    plt.scatter(stock_prices.index, buy_price, marker='^', color='green', label='BUY', s=s)
    plt.scatter(stock_prices.index, sell_price, marker='v', color='red', label='SELL', s=s)

    sectors: List[Sector] = new_portfolio.sectors()

    stocks_str: str = get_stocks_as_str(sectors)

    with pd.option_context("display.float_format", "%{:,.2f}".format):
        plt.figtext(
            x=0.45,
            y=0.15,
            s=f"Your Portfolio:\n"
              f"Returns: {str(round(new_portfolio.annual_returns(), 2))}%\n"
              f"Volatility: {str(round(new_portfolio.annual_volatility(), 2))}%\n"
              f"Max Loss: {str(round(new_portfolio.get_max_loss(), 2))}%\n"
              f"Sharpe Ratio: {str(round(new_portfolio.annual_sharpe(), 2))}\n"
              f"{stocks_str}",
            bbox=dict(facecolor="green", alpha=ALPHA), fontsize=11, style=STYLE, ha=HA, va=VA, fontname=FONT_NAME,
            wrap=WRAP,
        )

    plt.subplots_adjust(bottom=BOTTOM)

    print("Number of green:")
    print(np.count_nonzero(~np.isnan(buy_price)))
    print("Number of red:")
    print(np.count_nonzero(~np.isnan(sell_price)))

    return plt


def plot_three_portfolios_graph(min_variance_portfolio, sharpe_portfolio, max_returns, three_best_sectors_weights,
                                sectors: list[Sector], pct_change_table):
    plt.figure()  # Create a new plot instance
    # plot frontier, max sharpe & min Volatility values with a scatterplot
    plt.style.use("seaborn-dark")
    plt.xlabel("Date")
    plt.ylabel("Returns %")
    plt.title("Three Best Portfolios")

    for i in range(1, 3 + 1):
        pct_change_table[f'yield_{i}_percent'] = (pct_change_table[f'yield_{i}'] - 1) * 100

    labels: list[str] = ['Max Returns', 'Sharpe', 'Safest']
    colors = ['red', 'green', 'yellow']

    for i in range(1, 3 + 1):
        j = i - 1
        pct_change_table[f'yield_{str(4 - i)}_percent'].plot(
            figsize=FIG_SIZE1, grid=GRID, color=colors[j], linewidth=2, label=labels[j], legend=True,
            linestyle=LINE_STYLE
        )

    plt.subplots_adjust(bottom=BOTTOM)

    # ------------------ Printing 3 optimal Portfolios -----------------------
    # Setting max_X, max_Y to act as relative border for window size
    stocks_y: list[str] = ['', '', '']

    for i in range(len(sectors)):
        for j in range(3):
            weight = three_best_sectors_weights[2 - j][i] * 100
            stocks_y[j] += sectors[i].name + "(" + str("{:.2f}".format(weight)) + "%),\n "

    with pd.option_context("display.float_format", "%{:,.2f}".format):
        fig_text_data: dict = {
            'name': labels,
            'portfolio': [max_returns.iloc[0], sharpe_portfolio.iloc[0], min_variance_portfolio.iloc[0]],
            'stocks': stocks_y,
            'facecolor': colors
        }
        for i in range(3):
            x = 0.2 + 0.3 * i
            portfolio = fig_text_data['portfolio'][i]
            s: str = (
                f"Annual Returns: {str(round(portfolio[0], 2))}%\n "
                f"Annual Volatility: {str(round(portfolio[1], 2))}%\n"
                f"Annual Max Loss: {str(round(portfolio[0] - 1.65 * portfolio[1], 2))}%\n"
                f"Annual Sharpe Ratio: {str(round(portfolio[2], 2))}\n"
                f"{fig_text_data['stocks'][i][:-2]}"
            )
            bbox: dict = {'facecolor': fig_text_data['facecolor'][i], 'alpha': 0.5}
            plt.figtext(
                x=0.2 + 0.3 * i, y=0.15, s=s, bbox=bbox, fontsize=10, style=STYLE, ha=HA, va=VA,
                fontname=FONT_NAME, wrap=WRAP,
            )
            plt.figtext(
                x=0.2 + 0.3 * i, y=0.27, s=f"{fig_text_data['name'][i]} Portfolio:", fontsize=12,
                fontweight=FONT_WEIGHT, ha=HA, va=VA, fontname=FONT_NAME, wrap=WRAP,
            )
    return plt


def plot_distribution_of_portfolio(yields) -> plt:
    plt.figure()  # Create a new plot instance
    labels: list[str] = ['Low Risk', 'Medium Risk', 'High Risk']
    plt.subplots(figsize=FIG_SIZE1)
    plt.subplots_adjust(bottom=BOTTOM)

    monthly_yields: list[pd.core.series.Series] = [None] * len(yields)  # monthly yield change
    monthly_changes: list[pd.core.series.Series] = [None] * len(yields)  # yield changes
    df_describes: list[pd.core.series.Series] = [None] * len(yields)  # describe of yield changes

    for i in range(len(yields)):
        # Convert the index to datetime if it's not already in the datetime format
        curr_yield = yields[i]
        if not pd.api.types.is_datetime64_any_dtype(curr_yield.index):
            yields[i].index = pd.to_datetime(curr_yield.index)

        monthly_yields[i]: pd.core.series.Series = curr_yield.resample('M').first()
        monthly_changes[i]: pd.core.series.Series = monthly_yields[i].pct_change().dropna() * 100
        df_describes[i]: pd.core.series.Series = monthly_changes[i].describe().drop(["count"], axis=0)
        df_describes[i]: pd.core.series.Series = df_describes[i].rename(index={'std': 'Std. Deviation'})
        df_describes[i].index = df_describes[i].index.str.capitalize()
        sns.distplot(
            a=monthly_changes[i], kde=True, hist_kws={'alpha': 0.2}, norm_hist=False, rug=False, label=labels[i],
        )
        # Using sns.histplot(...):
        # sns.histplot(
        #     data=monthly_changes[i], kde=True, label=labels[i],
        # )

    plt.xlabel('Monthly Return %', fontsize=12)
    plt.ylabel('Distribution', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.title("Distribution of Portfolios - By Monthly Returns")

    # Creates figtext under the main graph
    with pd.option_context("display.float_format", "%{:,.2f}".format):
        y_header: float = 0.23
        y_content: float = 0.15
        alpha: float = 0.5
        s_params: list[str] = ['Low', 'Medium', 'High']
        colors: list[str] = ['blue', 'pink', 'green']
        header_fontsize: int = 14
        content_fontsize: int = 10
        for i in range(3):
            x = 0.2 + 0.3 * i
            s_content = df_describes[i].to_string()
            bbox: dict = {'facecolor': colors[i], 'alpha': alpha}

            # Content figtext
            plt.figtext(
                x=x, y=y_content, s=s_content, bbox=bbox, fontsize=content_fontsize, style=STYLE, ha=HA, va=VA,
                fontname=FONT_NAME, wrap=WRAP,
            )
            # Header figtext
            plt.figtext(
                x=x, y=y_header, s=f'{s_params[i]} Risk', fontsize=header_fontsize, fontweight=FONT_WEIGHT,
                ha=HA, fontname=FONT_NAME, wrap=WRAP,
            )

    return plt


def plot_investment_portfolio_yield(user_name, table, stats_details_tuple: tuple, sectors: list[Sector]):
    plt.figure()
    annual_returns, volatility, sharpe, max_loss, total_change = stats_details_tuple
    plt.style.use("seaborn-dark")
    plt.xlabel("Date")
    plt.ylabel("Returns %")
    plt.title(f"Hello {user_name}! This is your yield portfolio")

    tables: list[str] = ['selected_percent', 'selected_percent_forecast']
    colors: list[str] = ['green', 'blue']
    labels: list[str] = ['Returns', 'Forecast']

    for i in range(len(tables)):
        table[f'yield__{tables[i]}'].plot(
            figsize=FIG_SIZE1, grid=GRID, color=colors[i], linewidth=2, label=labels[i], legend=True,
            linestyle=LINE_STYLE
        )
    plt.subplots_adjust(bottom=BOTTOM)
    stocks_str: str = get_stocks_as_str(sectors)

    with pd.option_context("display.float_format", "%{:,.2f}".format):
        plt.figtext(
            x=0.45,
            y=0.15,
            s=f"Your Portfolio:\n"
              f"Total Change: {str(round(total_change, 2))}%\n"
              f"Annual Returns: {str(round(annual_returns, 2))}%\n"
              f"Annual Volatility: {str(round(volatility, 2))}%\n"
              f"Max Loss: {str(round(max_loss, 2))}%\n"
              f"Annual Sharpe Ratio: {str(round(sharpe, 2))}\n"
              f"{stocks_str}",
            bbox=dict(facecolor="green", alpha=ALPHA), fontsize=11, style=STYLE, ha=HA, va=VA, fontname=FONT_NAME,
            wrap=WRAP,
        )
    return plt


def get_stocks_as_str(sectors: list[Sector]) -> str:
    stocks_str: str = ""
    for i in range(len(sectors)):
        name: str = sectors[i].name
        weight: float = sectors[i].weight * 100
        stocks_str += name + "(" + str("{:.2f}".format(weight)) + "%),\n "
    return stocks_str


def plot_sectors_component(user_name: str, weights: List[float], names: List[str]):
    # TODO: some texts are cut from the image, fix this. Removing `plt.figure(1)` makes the image 50% empty
    plt.figure(1)
    plt.title(f"{user_name}'s Portfolio\n")
    plt.pie(
        x=weights,
        labels=names,
        autopct="%1.1f%%",
        startangle=140,
    )
    plt.axis("equal")
    return plt


def plot_portfolio_component_stocks(user_name: str, stocks_weights: List[float], stocks_symbols: list[int],
                                    descriptions: list[str]):
    if len(stocks_weights) != len(stocks_symbols) or len(stocks_weights) != len(descriptions):
        raise ValueError("Input lists must have the same length.")
    plt.figure(figsize=FIG_SIZE4)
    plt.title(f"{user_name}'s Portfolio", fontsize=20, pad=10)
    data = [["Stock", "Weight", "Description"]]
    for symbol, weight, description in zip(stocks_symbols, stocks_weights, descriptions):
        data.append([symbol, f"{weight:.1%}", description])

    table = plt.table(cellText=data, colLabels=None, cellLoc='left', loc='center',
                      cellColours=[['#D5DBDB', '#D5DBDB', '#D5DBDB']] * len(data),
                      colWidths=[0.1, 0.1, 0.8])

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)  # Adjust scaling to fit the plot better

    plt.axis('off')  # Turn off the axis

    # Automatically adjust the layout to prevent overlapping
    plt.tight_layout()
    plt_instance = plt.gcf()  # Get the current figure

    return plt_instance


def plot_price_forecast(stocks_symbols, description, df: pd.DataFrame, annual_returns, plt_instance=None) -> plt:
    if plt_instance is not None:
        return plt_instance
    df[df.columns[0]].plot()
    forecast_short_time = (((df['label'][-500:].pct_change().mean() +1) ** 252) - 1) * 100
    df['Forecast'].plot()
    plt.title(f"{description} Stock Price Forecast", pad=10)
    # add text box with annual returns value
    plt.figtext(
        x=0.15,
        y=0.75,
        s=f"average annual Return: {str(round(annual_returns, 2))}%\n"
        f"forecast Annual return: {str(round(forecast_short_time, 2))}%\n"
    )

    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    return plt


def plot_distribution_of_stocks(stock_names, pct_change_table) -> plt:
    plt.figure()  # Create a new plot instance
    plt.subplots(figsize=FIG_SIZE1)
    plt.legend()
    plt.xlabel('Return', fontsize=12)
    plt.ylabel('Distribution', fontsize=12)
    for i in range(len(stock_names)):
        sns.distplot(
            a=pct_change_table[stock_names[i]][::30] * 100, kde=True, hist=False, rug=False, label=stock_names[i],
        )
    plt.grid(True)
    plt.legend()
    return plt


def plot_research_graphs(data_stats_tuples):
    # Define metric names for the x-axis
    labels = [
        'top Total Return',
        'Total min Volatility',
        'Total Sharpe',
        'Annual Return',
        'Annual top min Volatility',
        'Annual Sharpe',
        'Monthly Return',
        'Monthly Volatility',
        'Monthly Sharpe',
        'Forecast Return',
        'Forecast Volatility',
        'Forecast Sharpe'
    ]
    top_returns_df = pd.DataFrame()
    top_returns_df['top Total Return'] = pd.DataFrame(data_stats_tuples[0] )
    # Extract stock symbols from the first series
    stock_symbols = data_stats_tuples[0].index.tolist()

    # Convert data_stats_tuples to a NumPy array for easier manipulation
    # values = np.array([series[1].values for series in data_stats_tuples])

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a horizontal bar plot for each metric
   # for i, metric in enumerate(metrics):
    #    ax.barh(stock_symbols, values[:, i], label=metric)

    # Set labels, title, and legend
   # ax.set_xlabel('Values')
    #ax.set_ylabel('Stock Symbols')

    plt.figure()  # Create a new plot instance
    plt.subplots(figsize=FIG_SIZE1)
    plt.legend()
    top_returns_df.plot()

    plt.grid(True)
    plt.legend()
    plt.show()





    ax.set_title('Values for Different Metrics by Stock Symbols')
    ax.legend()
    top_returns_df.plot()
    plt.tight_layout()
    plt.show()


def save_graphs(plt_instance, file_name) -> None:
    # Adjust font size of the table cells
    plt_instance.savefig(f'{file_name}.png', format='png', dpi=300)
    plt_instance.clf()  # Clear the figure after saving


def plot_image(file_name):
    image = Image.open(file_name)
    image.show()
