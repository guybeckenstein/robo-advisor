import matplotlib

from service.impl.sector import Sector
from service.util.graph import helpers

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
SEABORN_STYLE: str = 'seaborn-v0_8-dark'


def markowitz_graph(sectors: list[Sector], three_best_sectors_weights, min_variance_portfolio, sharpe_portfolio,
                    max_returns_portfolio, df: pd.DataFrame) -> plt:
    # plot frontier, max sharpe & min Volatility values with a scatterplot
    helpers.MarkowitzAndGini.create_scatter_plot(
        df=df,
        x='Volatility',
        y='Returns',
        min_variance_portfolio=min_variance_portfolio,
        sharpe_portfolio=sharpe_portfolio,
        max_returns_portfolio=max_returns_portfolio,
    )

    # ------------------ Printing 3 optimal Portfolios -----------------------
    # Setting max_X, max_Y to act as relative border for window size
    helpers.MarkowitzAndGini.plot_markowitz_or_gini_graph(
        portfolios=[min_variance_portfolio.iloc[0], sharpe_portfolio.iloc[0], max_returns_portfolio.iloc[0]],
        stocks=helpers.MarkowitzAndGini.get_stocks_str(sectors, three_best_sectors_weights),
        text1='Annual Returns',
        text2='Annual Volatility'
    )

    return plt


def gini_graph(sectors: list[Sector], three_best_sectors_weights, min_variance_portfolio, sharpe_portfolio,
               max_portfolios_annual_portfolio, df: pd.DataFrame) -> plt:
    # plot frontier, max sharpe & min Gini values with a scatterplot
    helpers.MarkowitzAndGini.create_scatter_plot(
        df=df,
        x='Gini',
        y='Portfolio Annual',
        min_variance_portfolio=min_variance_portfolio,
        sharpe_portfolio=sharpe_portfolio,
        max_returns_portfolio=max_portfolios_annual_portfolio
    )

    # ------------------ Printing 3 optimal Portfolios -----------------------
    # Setting max_X, max_Y to act as relative border for window size
    helpers.MarkowitzAndGini.plot_markowitz_or_gini_graph(
        portfolios=[
            min_variance_portfolio.iloc[0], sharpe_portfolio.iloc[0], max_portfolios_annual_portfolio.iloc[0]
        ],
        stocks=helpers.MarkowitzAndGini.get_stocks_str(sectors, three_best_sectors_weights),
        text1='Annual Returns',
        text2='Annual Gini'
    )

    return plt


def bb_strategy_stock(stock_prices, buy_price, sell_price) -> plt:
    import numpy as np

    stock_prices[['Adj Close', 'Lower', 'Upper']].plot(figsize=FIG_SIZE2)
    plt.scatter(stock_prices.index, buy_price, marker='^', color='green', label='BUY', s=200)
    plt.scatter(stock_prices.index, sell_price, marker='v', color='red', label='SELL', s=200)
    plt.title("Bollinger Bands Squeeze Strategy", pad=10)
    # Count non-NaN values in buy and sell price arrays
    num_buy_signals = np.count_nonzero(~np.isnan(buy_price))
    num_sell_signals = np.count_nonzero(~np.isnan(sell_price))

    # Add text annotations for the counts of buy and sell signals
    attributes: list[tuple] = [('Buy', num_buy_signals, 0.93, 'green'), ('Sell', num_sell_signals, 0.87, 'red')]
    for signal_str, signal_var, y, color in attributes:
        plt.annotate(f'{signal_str} Signals: {signal_var}',
                     xy=(0.25, y),
                     xycoords='axes fraction',
                     fontsize=12,
                     color=color)

    # Show legend
    plt.legend()

    return plt


def three_portfolios_graph(max_returns_portfolio, sharpe_portfolio, min_variance_portfolio, three_best_sectors_weights,
                           sectors: list[Sector], pct_change_table: pd.DataFrame) -> plt:  # second graph
    plt.figure()  # Create a new plot instance
    # plot frontier, max sharpe & min Volatility values with a scatterplot
    plt.style.use(SEABORN_STYLE)
    labels, colors = helpers.ThreePortfolios.main_plot(pct_change_table)

    # ------------------ Printing 3 optimal Portfolios -----------------------
    # Setting max_X, max_Y to act as relative border for window size
    helpers.ThreePortfolios.sub_plots(
        colors, labels, max_returns_portfolio, sharpe_portfolio, min_variance_portfolio, sectors,
        three_best_sectors_weights
    )
    return plt


def portfolio_distribution(yields) -> plt:  # first graph
    plt.figure()  # Creates a new plot instance
    plt.subplots(figsize=FIG_SIZE1)
    plt.subplots_adjust(bottom=BOTTOM)

    monthly_yields: list[pd.Series] = [None] * len(yields)  # monthly yield change
    monthly_changes: list[pd.Series] = [None] * len(yields)  # yield changes
    df_describes: list[pd.Series] = [None] * len(yields)  # describe of yield changes

    labels: list[str, str, str] = ['Low Risk', 'Medium Risk', 'High Risk']
    colors: list[str] = ['orange', 'green', 'red']
    for i in range(len(yields)):
        # Convert the index to datetime if it's not already in the datetime format
        curr_yield = yields[i]
        if not pd.api.types.is_datetime64_any_dtype(curr_yield.index):
            yields[i].index = pd.to_datetime(curr_yield.index)

        monthly_yields[i]: pd.Series = curr_yield.resample('M').first()
        monthly_changes[i]: pd.Series = monthly_yields[i].pct_change().dropna() * 100
        df_describes[i]: pd.Series = monthly_changes[i].describe().drop(["count"], axis=0)
        df_describes[i]: pd.Series = df_describes[i].rename(index={'mean': 'Mean Yield', 'std': 'Standard Deviation',
                                                                   '50%': '50%(Median)', '25%': '25%(Q1)',
                                                                   '75%': '75%(Q3)', 'max': 'Max', 'min': 'Min'})
        df_describes[i].index = df_describes[i].index.str.capitalize()
        ax = sns.distplot(
            a=monthly_changes[i], kde=True, hist_kws={'alpha': 0.2}, norm_hist=False, rug=False, color=colors[i],
            label=labels[i],
        )
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0%}".format(x)))

    plt.legend(frameon=True, facecolor='white')  # Adjust legend background color
    plt.xlabel('Monthly Yield Percentage', fontsize=12)
    plt.ylabel('Distribution Percentage', fontsize=12)
    plt.grid(True)
    plt.title("Distribution of Portfolios - By Monthly Returns", fontsize=20, fontweight="bold")

    # Creates subplots under the main graph
    with pd.option_context("display.float_format", "%{:,.2f}".format):
        y_header: float = 0.25
        y_content: float = 0.15
        alpha: float = 0.5
        s_params: list[str] = labels
        colors: list[str] = colors
        header_fontsize: int = 14
        content_fontsize: int = 12
        for i in range(3):
            x = 0.2 + 0.3 * i
            s_content = df_describes[i].to_string()
            bbox: dict = {'facecolor': colors[i], 'alpha': alpha}

            # Content figtext
            plt.figtext(
                x=x, y=y_content, s=s_content, bbox=bbox, fontsize=content_fontsize, style=STYLE, ha=HA,
                multialignment='left', va=VA, fontname=FONT_NAME, wrap=WRAP,
            )
            # Header figtext
            plt.figtext(
                x=x, y=y_header, s=f'{s_params[i]}', fontsize=header_fontsize, fontweight=FONT_WEIGHT,
                ha=HA, fontname=FONT_NAME, wrap=WRAP,
            )

    return plt


def investment_portfolio_estimated_yield(df: pd.DataFrame, annual_returns: float, volatility: float, sharpe: float,
                                         max_loss: float, total_change: float, sectors: list[Sector],
                                         excepted_returns: float) -> plt:
    plt.figure()
    plt.style.use(SEABORN_STYLE)
    plt.xlabel("Date")
    plt.ylabel("Returns Percentage")

    tables: list[str] = ['selected_percent', 'selected_percent_forecast']
    colors: list[str] = ['green', 'blue']
    labels: list[str] = ['History', 'Forecast']

    for i in range(len(tables)):
        df[f'yield__{tables[i]}'].plot(
            figsize=FIG_SIZE1, grid=GRID, color=colors[i], linewidth=2, label=labels[i], legend=True,
            linestyle=LINE_STYLE
        )
    plt.legend(frameon=True, facecolor='white')  # Adjust legend background color
    plt.subplots_adjust(bottom=BOTTOM)
    stocks_str: str = helpers.EstimatedYield.get_stocks_as_str(sectors)

    with pd.option_context("display.float_format", "%{:,.2f}".format):
        plt.figtext(
            x=0.45,
            y=0.15,
            s=f"Total Change: {str(round(total_change, 2))}%\n"
              f"Annual Return: {str(round(annual_returns, 2))}%\n"
              f"Excepted Annual Return: {str(round(excepted_returns, 2))}%\n"
              f"Annual Volatility: {str(round(volatility, 2))}%\n"
              f"Max Loss: {str(round(max_loss, 2))}%\n"
              f"Annual Sharpe Ratio: {str(round(sharpe, 2))}\n"
              "## Weights: ##\n"
              f"{stocks_str.strip()}",
            bbox=dict(facecolor="0.8", alpha=ALPHA), fontsize=11, style=STYLE, ha=HA, multialignment='left', va=VA,
            fontname=FONT_NAME, wrap=WRAP,
        )
    return plt


def sectors_component(weights: list[float], names: list[str]) -> plt:
    plt.figure(figsize=FIG_SIZE4)  # Adjust width and height as needed
    plt.pie(
        x=weights,
        labels=names,
        autopct="%1.1f%%",
        startangle=140,

    )
    plt.axis("equal")
    return plt


def price_forecast(description, df: pd.DataFrame, annual_return_with_forecast, excepted_returns,
                   plt_instance=None) -> plt:
    history_annual_return = (((df[df.columns[0]].pct_change().mean() + 1) ** 254) - 1) * 100
    x: float = 0.15
    y: float = 0.75
    if plt_instance is not None:
        plt_instance = helpers.PriceForecast.plt_figtext(
            plt_instance=plt, x=x, y=y, history_annual_return=history_annual_return,
            average_annual_return=annual_return_with_forecast, forecast_short_time=excepted_returns
        )
        return plt_instance
    else:
        df[df.columns[0]].plot(label='History')
        df['Forecast'].plot()
        plt.title(f"{description} Stock Price Forecast", pad=10)
        helpers.PriceForecast.plt_figtext(
            plt_instance=plt, x=x, y=y, history_annual_return=history_annual_return,
            average_annual_return=annual_return_with_forecast, forecast_short_time=excepted_returns
        )

        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Price')
        return plt


def distribution_of_stocks(stock_names, pct_change_table) -> plt:
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
