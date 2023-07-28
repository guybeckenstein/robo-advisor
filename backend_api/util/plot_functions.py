import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_markowitz_graph(sectors: list, three_best_sectors_weights, min_variance_port, sharpe_portfolio,
                         max_returns, max_vols, df) -> plt:
    # plot frontier, max sharpe & min Volatility values with a scatterplot
    fig_size_X = 10
    fig_size_Y = 8
    fig_size = (fig_size_X, fig_size_Y)
    plt.style.use("seaborn-dark")
    df.plot.scatter(
        x="Volatility", y="Returns", c="Sharpe Ratio", cmap="RdYlGn", edgecolors="black", figsize=fig_size, grid=True,
    )
    plt.scatter(
        x=sharpe_portfolio["Volatility"], y=sharpe_portfolio["Returns"], c="green", marker="D", s=200,
    )
    plt.scatter(
        x=min_variance_port["Volatility"], y=min_variance_port["Returns"], c="orange", marker="D", s=200,
    )
    plt.scatter(
        x=max_vols["Volatility"], y=max_returns["Returns"], c="red", marker="D", s=200
    )
    plt.style.use("seaborn-dark")

    plt.xlabel("Volatility (Std. Deviation) Percentage %")
    plt.ylabel("Expected Returns Percentage %")
    plt.title("Efficient Frontier")
    plt.subplots_adjust(bottom=0.4)

    # ------------------ Printing 3 optimal Portfolios -----------------------
    # Setting max_X, max_Y to act as relative border for window size
    stocks_str_high = ""
    stocks_str_medium = ""
    stocks_str_low = ""

    # stocks_str_high
    for i in range(len(sectors)):
        weight = three_best_sectors_weights[2][i] * 100
        stocks_str_high += sectors[i].get_name() + "(" + str("{:.2f}".format(weight)) + "%),\n "
    # stocks_str_medium
    for i in range(len(sectors)):
        weight = three_best_sectors_weights[1][i] * 100
        stocks_str_medium += sectors[i].get_name() + "(" + str("{:.2f}".format(weight)) + "%),\n "
    # stocks_str_low
    for i in range(len(sectors)):
        weight = three_best_sectors_weights[0][i] * 100
        stocks_str_low += sectors[i].get_name() + "(" + str("{:.2f}".format(weight)) + "%),\n "

    with pd.option_context("display.float_format", "%{:,.2f}".format):
        plt.figtext(
            0.2,
            0.15,
            "Max returns Portfolio: \n"
            + "Annual returns: " + str(round(max_returns.iloc[0][0], 2)) + "%\n"
            + "Annual volatility: " + str(round(max_returns.iloc[0][1], 2)) + "%\n"
            + "Annual max loss: " + str(round(max_returns.iloc[0][0] - 1.65 * max_returns.iloc[0][1], 2)) + "%\n"
            + "Sharpe Ratio: " + str(round(max_returns.iloc[0][2], 2)) + "\n"
            + stocks_str_high,
            bbox=dict(facecolor="red", alpha=0.5),
            fontsize=11,
            style="oblique",
            ha="center",
            va="center",
            fontname="Arial",
            wrap=True,
        )
        plt.figtext(
            0.45,
            0.15,
            "Safest Portfolio: \n"
            + "Annual returns: " + str(round(min_variance_port.iloc[0][0], 2)) + "%\n"
            + "Annual volatility: " + str(round(min_variance_port.iloc[0][1], 2)) + "%\n"
            + "Annual max loss: " + str(round(min_variance_port.iloc[0][0] - 1.65 * min_variance_port.iloc[0][1], 2))
            + "%\n"
            + "Sharpe Ratio: " + str(round(min_variance_port.iloc[0][2], 2)) + "\n"
            + stocks_str_low,
            bbox=dict(facecolor="yellow", alpha=0.5),
            fontsize=11,
            style="oblique",
            ha="center",
            va="center",
            fontname="Arial",
            wrap=True,
        )
        plt.figtext(
            0.7,
            0.15,
            "Sharpe  Portfolio: \n"
            + "Annual returns: " + str(round(sharpe_portfolio.iloc[0][0], 2)) + "%\n"
            + "Annual volatility: " + str(round(sharpe_portfolio.iloc[0][1], 2)) + "%\n"
            + "Annual max loss: " + str(round(sharpe_portfolio.iloc[0][0] - 1.65 * sharpe_portfolio.iloc[0][1], 2))
            + "%\n"
            + "Sharpe Ratio: " + str(round(sharpe_portfolio.iloc[0][2], 2)) + "\n"
            + stocks_str_medium,
            bbox=dict(facecolor="green", alpha=0.5),
            fontsize=11,
            style="oblique",
            ha="center",
            va="center",
            fontname="Arial",
            wrap=True,
        )

    return plt


def plot_gini_graph(sectors, three_best_sectors_weights, min_variance_port, sharpe_portfolio, max_portfolios_annual,
                    max_ginis, df) -> plt:
    # plot frontier, max sharpe & min Gini values with a scatterplot
    plt.style.use('seaborn-dark')
    df.plot.scatter(x='Gini', y='Portfolio_annual', c='Sharpe Ratio',
                    cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
    plt.scatter(x=sharpe_portfolio['Gini'], y=sharpe_portfolio['Portfolio_annual'], c='green', marker='D', s=200)
    plt.scatter(x=min_variance_port['Gini'], y=min_variance_port['Portfolio_annual'], c='orange', marker='D', s=200)
    plt.scatter(x=max_ginis['Gini'], y=max_portfolios_annual['Portfolio_annual'], c='red', marker='D', s=200)
    plt.style.use('seaborn-dark')

    plt.xlabel('Gini (Std. Deviation) Percentage %')
    plt.ylabel('Expected portfolio annual Percentage %')
    plt.title('Efficient Frontier')
    plt.subplots_adjust(bottom=0.4)

    # ------------------ Printing 3 optimal Portfolios -----------------------
    # Setting max_X, max_Y to act as relative border for window size
    stocks_str_high = ""
    stocks_str_medium = ""
    stocks_str_low = ""

    # stocks_str_high
    for i in range(len(sectors)):
        weight = three_best_sectors_weights[2][i] * 100
        stocks_str_high += sectors[i].get_name() + "(" + str("{:.2f}".format(weight)) + "%),\n "
    # stocks_str_medium
    for i in range(len(sectors)):
        weight = three_best_sectors_weights[1][i] * 100
        stocks_str_medium += sectors[i].get_name() + "(" + str("{:.2f}".format(weight)) + "%),\n "
    # stocks_str_low
    for i in range(len(sectors)):
        weight = three_best_sectors_weights[0][i] * 100
        stocks_str_low += sectors[i].get_name() + "(" + str("{:.2f}".format(weight)) + "%),\n "

    with pd.option_context("display.float_format", "%{:,.2f}".format):
        plt.figtext(
            0.2,
            0.15,
            "Max returns Portfolio: \n"
            + "Annual returns: " + str(round(max_portfolios_annual.iloc[0][0], 2)) + "%\n"
            + "Annual gini: " + str(round(max_portfolios_annual.iloc[0][1], 2)) + "%\n"
            + "Annual max loss: " + str(round(max_portfolios_annual.iloc[0][0]
                                              - 1.65 * max_portfolios_annual.iloc[0][1], 2))
            + "%\n"
            + "Sharpe Ratio: " + str(round(max_portfolios_annual.iloc[0][2], 2)) + "\n"
            + stocks_str_high,
            bbox=dict(facecolor="red", alpha=0.5),
            fontsize=11,
            style="oblique",
            ha="center",
            va="center",
            fontname="Arial",
            wrap=True,
        )
        plt.figtext(
            0.45,
            0.15,
            "Safest Portfolio: \n"
            + "Annual returns: " + str(round(min_variance_port.iloc[0][0], 2)) + "%\n"
            + "Annual gini: " + str(round(min_variance_port.iloc[0][1], 2)) + "%\n"
            + "Annual Max loss: " + str(round(min_variance_port.iloc[0][0] - 1.65 * min_variance_port.iloc[0][1], 2))
            + "%\n"
            + "Sharpe Ratio: " + str(round(min_variance_port.iloc[0][2], 2)) + "\n"
            + stocks_str_low,
            bbox=dict(facecolor="yellow", alpha=0.5),
            fontsize=11,
            style="oblique",
            ha="center",
            va="center",
            fontname="Arial",
            wrap=True,
        )
        plt.figtext(
            0.7,
            0.15,
            "Sharpe  Portfolio: \n"
            + "Annual returns: " + str(round(sharpe_portfolio.iloc[0][0], 2)) + "%\n"
            + "Annual gini: " + str(round(sharpe_portfolio.iloc[0][1], 2)) + "%\n"
            + "Annual Max loss: " + str(round(sharpe_portfolio.iloc[0][0] - 1.65 * sharpe_portfolio.iloc[0][1], 2))
            + "%\n"
            + "Sharpe Ratio: " + str(round(sharpe_portfolio.iloc[0][2], 2)) + "\n"
            + stocks_str_medium,
            bbox=dict(facecolor="green", alpha=0.5),
            fontsize=11,
            style="oblique",
            ha="center",
            va="center",
            fontname="Arial",
            wrap=True,
        )

    return plt


def plotbb_strategy_stock(stock_prices, buy_price, sell_price) -> plt:
    stock_prices[['Adj Close', 'Lower', 'Upper']].plot(figsize=(10, 4))
    plt.scatter(stock_prices.index, buy_price, marker='^', color='green', label='BUY', s=200)
    plt.scatter(stock_prices.index, sell_price, marker='v', color='red', label='SELL', s=200)
    plt.show()

    print("number of green :")
    print(np.count_nonzero(~np.isnan(buy_price)))
    print("number of red :")
    print(np.count_nonzero(~np.isnan(sell_price)))


def plotbb_strategy_portfolio(stock_prices, buy_price, sell_price, new_portfolio) -> plt:
    stock_prices[['Adj Close', 'Lower', 'Upper']].plot(figsize=(10, 4))
    plt.scatter(stock_prices.index, buy_price, marker='^', color='green', label='BUY', s=200)
    plt.scatter(stock_prices.index, sell_price, marker='v', color='red', label='SELL', s=200)

    sectors = new_portfolio.get_sectors()

    stocks_str = ""
    for i in range(len(sectors)):
        name = sectors[i].get_name()
        weight = sectors[i].get_weight() * 100
        stocks_str += name + "(" + str("{:.2f}".format(weight)) + "%),\n "

    with pd.option_context("display.float_format", "%{:,.2f}".format):
        plt.figtext(
            0.45,
            0.15,
            "your Portfolio: \n"
            + "Returns: " + str(round(new_portfolio.get_annual_returns(), 2)) + "%\n"
            + "Volatility: " + str(round(new_portfolio.get_volatility(), 2)) + "%\n"
            + "max loss: " + str(round(new_portfolio.get_max_loss(), 2)) + "%\n"
            + "Sharpe Ratio: " + str(round(new_portfolio.get_sharpe(), 2)) + "\n"
            + stocks_str,
            bbox=dict(facecolor="green", alpha=0.5),
            fontsize=11,
            style="oblique",
            ha="center",
            va="center",
            fontname="Arial",
            wrap=True,
        )

    plt.subplots_adjust(bottom=0.4)

    print("number of green :")
    print(np.count_nonzero(~np.isnan(buy_price)))
    print("number of red :")
    print(np.count_nonzero(~np.isnan(sell_price)))

    return plt


def plot_three_portfolios_graph(min_variance_port, sharpe_portfolio, max_returns, three_best_sectors_weights, sectors,
                                pct_change_table) -> plt:
    # plot frontier, max sharpe & min Volatility values with a scatterplot
    fig_size_X = 10
    fig_size_Y = 8
    fig_size = (fig_size_X, fig_size_Y)
    plt.style.use("seaborn-dark")
    plt.xlabel("Date")
    plt.ylabel("Returns %")
    plt.title("3 best portfolios")

    pct_change_table['yield_1_percent'] = (pct_change_table['yield_1'] - 1) * 100
    pct_change_table['yield_2_percent'] = (pct_change_table['yield_2'] - 1) * 100
    pct_change_table['yield_3_percent'] = (pct_change_table['yield_3'] - 1) * 100

    pct_change_table['yield_1_percent'].plot(figsize=fig_size, grid=True, color="yellow", linewidth=2, label="safest",
                                             legend=True, linestyle="dashed")
    pct_change_table['yield_2_percent'].plot(figsize=fig_size, grid=True, color="green", linewidth=2, label="sharpe",
                                             legend=True, linestyle="dashed")
    pct_change_table['yield_3_percent'].plot(figsize=fig_size, grid=True, color="red", linewidth=2, label="max return",
                                             legend=True, linestyle="dashed")

    plt.subplots_adjust(bottom=0.4)

    # ------------------ Printing 3 optimal Portfolios -----------------------
    # Setting max_X, max_Y to act as relative border for window size
    stocks_str_high = ""
    stocks_str_medium = ""
    stocks_str_low = ""

    # stocks_str_high
    for i in range(len(sectors)):
        weight = three_best_sectors_weights[2][i] * 100
        stocks_str_high += sectors[i].get_name() + "(" + str("{:.2f}".format(weight)) + "%),\n "
    # stocks_str_medium
    for i in range(len(sectors)):
        weight = three_best_sectors_weights[1][i] * 100
        stocks_str_medium += sectors[i].get_name() + "(" + str("{:.2f}".format(weight)) + "%),\n "
    # stocks_str_low
    for i in range(len(sectors)):
        weight = three_best_sectors_weights[0][i] * 100
        stocks_str_low += sectors[i].get_name() + "(" + str("{:.2f}".format(weight)) + "%),\n "

    with pd.option_context("display.float_format", "%{:,.2f}".format):
        plt.figtext(
            0.2,
            0.15,
            "Max returns Portfolio: \n"
            + "Annual returns: " + str(round(max_returns.iloc[0][0], 2)) + "%\n"
            + "Annual volatility: " + str(round(max_returns.iloc[0][1], 2)) + "%\n"
            + "Annual max loss: " + str(round(max_returns.iloc[0][0] - 1.65 * max_returns.iloc[0][1], 2)) + "%\n"
            + "Annual sharpe Ratio: " + str(round(max_returns.iloc[0][2], 2)) + "\n"
            + stocks_str_high,
            bbox=dict(facecolor="red", alpha=0.5),
            fontsize=11,
            style="oblique",
            ha="center",
            va="center",
            fontname="Arial",
            wrap=True,
        )
        plt.figtext(
            0.7,
            0.15,
            "Sharpe  Portfolio: \n"
            + "Annual returns: " + str(round(sharpe_portfolio.iloc[0][0], 2)) + "%\n"
            + "Annual Volatility: " + str(round(sharpe_portfolio.iloc[0][1], 2)) + "%\n"
            + "Annual max loss: " + str(round(sharpe_portfolio.iloc[0][0] - 1.65 * sharpe_portfolio.iloc[0][1], 2))
            + "%\n"
            + "Annual sharpe Ratio: " + str(round(sharpe_portfolio.iloc[0][2], 2)) + "\n"
            + stocks_str_medium,
            bbox=dict(facecolor="green", alpha=0.5),
            fontsize=11,
            style="oblique",
            ha="center",
            va="center",
            fontname="Arial",
            wrap=True,
        )
        plt.figtext(
            0.45,
            0.15,
            "Safest Portfolio: \n"
            + "Annual returns: " + str(round(min_variance_port.iloc[0][0], 2)) + "%\n"
            + "Annual volatility: " + str(round(min_variance_port.iloc[0][1], 2)) + "%\n"
            + "Annual max loss: " + str(round(min_variance_port.iloc[0][0] - 1.65 * min_variance_port.iloc[0][1], 2))
            + "%\n"
            + "Annual sharpe Ratio: " + str(round(min_variance_port.iloc[0][2], 2)) + "\n"
            + stocks_str_low,
            bbox=dict(facecolor="yellow", alpha=0.5),
            fontsize=11,
            style="oblique",
            ha="center",
            va="center",
            fontname="Arial",
            wrap=True,
        )
    return plt


def plot_price_forecast(stock_symbol, df, is_data_got_from_tase) -> plt:
    if is_data_got_from_tase:
        df['closingIndexPrice'].plot()
    else:
        df['Adj Close'].plot()
    df['Forecast'].plot()
    plt.title(stock_symbol + " Stock Price Forecast")
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    return plt


def plot_distribution_of_stocks(stock_names, pct_change_table) -> plt:
    plt.subplots(figsize=(8, 8))
    plt.legend()
    plt.xlabel('Return', fontsize=12)
    plt.ylabel('Distribution', fontsize=12)
    for i in range(len(stock_names)):
        sns.distplot(pct_change_table[stock_names[i]][::30] * 100, kde=True, hist=False, rug=False,
                     label=stock_names[i])
    plt.grid(True)
    plt.legend()
    return plt


def plot_distribution_of_portfolio(yields: list) -> plt:
    labels = ['low risk', 'medium risk', 'high risk']
    plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.4)

    monthly_changes = [None] * len(yields)  # yield changes
    monthly_yields = [None] * len(yields)  # monthly yield change
    df_describes = [None] * len(yields)  # describe of yield changes
    # monthlyCompoundedReturns = [None] * len(yields) # total change in percent from beginning
    # monthlyCompoundedReturns[i] = (1 + monthly_changes[i]).cumprod() - 1

    for i in range(len(yields)):
        # Convert the index to datetime if it's not already in the datetime format
        if not pd.api.types.is_datetime64_any_dtype(yields[i].index):
            yields[i].index = pd.to_datetime(yields[i].index)

        monthly_yields[i] = yields[i].resample('M').first()
        monthly_changes[i] = monthly_yields[i].pct_change().dropna() * 100
        df_describes[i] = monthly_changes[i].describe().drop(["count"], axis=0)
        sns.distplot(pd.Series(monthly_changes[i]), kde=True, hist_kws={'alpha': 0.2}, norm_hist=False,
                     rug=False, label=labels[i])

    plt.xlabel('Monthly Return %', fontsize=12)
    plt.ylabel('Distribution', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.title("Distribution of Portfolios - by monthly returns")

    with pd.option_context("display.float_format", "%{:,.2f}".format):
        plt.figtext(0.2, 0.15, "low risk\n" + str(df_describes[0]), bbox=dict(facecolor="blue", alpha=0.5),
                    fontsize=11, style="oblique", ha="center", va="center", fontname="Arial", wrap=True)
        plt.figtext(0.45, 0.15, "medium risk\n" + str(df_describes[1]), bbox=dict(facecolor="pink", alpha=0.5),
                    fontsize=11, style="oblique", ha="center", va="center", fontname="Arial", wrap=True)
        plt.figtext(0.7, 0.15, "high risk\n" + str(df_describes[2]), bbox=dict(facecolor="green", alpha=0.5),
                    fontsize=11, style="oblique", ha="center", va="center", fontname="Arial", wrap=True)

    return plt


def plot_top_stocks(top_stocks) -> None:
    print(top_stocks)


def plot(plt_instance) -> None:
    plt_instance.show()


def save_graphs(plt_instance, file_name) -> None:
    plt_instance.savefig(file_name, format='png')
