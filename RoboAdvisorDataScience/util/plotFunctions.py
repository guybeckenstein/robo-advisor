import http.client
import json
import codecs
import re
import requests
import base64
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf
import seaborn as sns
from util import setting
from util import taseUtil
from util import consoleHandler
from api import Portfolio as Portfolio
from api import User as User
import ta
import requests_cache
from api import StatsModels as StatsModels
from api import Sector as Sector


def plotMarkowitzGraph(sectorsList, threeBestSectorsWeights, min_variance_port,
                                sharpe_portfolio, max_returns, max_vols, df):

    # plot frontier, max sharpe & min Volatility values with a scatterplot
    figSize_X = 10
    figSize_Y = 8
    figSize = (figSize_X, figSize_Y)
    plt.style.use("seaborn-dark")
    df.plot.scatter(
        x="Volatility",
        y="Returns",
        c="Sharpe Ratio",
        cmap="RdYlGn",
        edgecolors="black",
        figsize=figSize,
        grid=True,
    )
    plt.scatter(
        x=sharpe_portfolio["Volatility"],
        y=sharpe_portfolio["Returns"],
        c="green",
        marker="D",
        s=200,
    )
    plt.scatter(
        x=min_variance_port["Volatility"],
        y=min_variance_port["Returns"],
        c="orange",
        marker="D",
        s=200,
    )
    plt.scatter(
        x=max_vols["Volatility"], y=max_returns["Returns"], c="red", marker="D", s=200
    )
    plt.style.use("seaborn-dark")

    plt.xlabel("Volatility (Std. Deviation) Percentage %")
    plt.ylabel("Expected Returns Percentage %")
    plt.title("Efficient Frontier")
    plt.subplots_adjust(bottom=0.4)

    # ------------------ Pritning 3 optimal Protfolios -----------------------
    # Setting max_X, max_Y to act as relative border for window size
    stocksStrHigh = ""
    stocksStrLow = ""
    stocksStrMedium = ""

    # stocksStrHigh
    for i in range(len(sectorsList)):
        weight = threeBestSectorsWeights[2][i] * 100
        stocksStrHigh += sectorsList[i].getName() + "(" + str("{:.2f}".format(weight)) + "%),\n "
    # stocksStrMedium
    for i in range(len(sectorsList)):
        weight = threeBestSectorsWeights[1][i] * 100
        stocksStrMedium += sectorsList[i].getName() + "(" + str("{:.2f}".format(weight)) + "%),\n "
    # stocksStrLow
    for i in range(len(sectorsList)):
        weight = threeBestSectorsWeights[0][i] * 100
        stocksStrLow += sectorsList[i].getName() + "(" + str("{:.2f}".format(weight)) + "%),\n "

    with pd.option_context("display.float_format", "%{:,.2f}".format):
        plt.figtext(
            0.2,
            0.15,
            "Max returns Porfolio: \n"
            + "Annual returns: " + str(round(max_returns.iloc[0][0], 2)) + "%\n"
            + "Annual volatility: " + str(round(max_returns.iloc[0][1], 2)) + "%\n"
            + "Annual max loss: " + str(round(max_returns.iloc[0][0] - 1.65*max_returns.iloc[0][1], 2)) + "%\n"
            + "Sharpe Ratio: " + str(round(max_returns.iloc[0][2], 2)) + "\n"
            + stocksStrHigh,
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
            + "Annual max loss: " + str(round(min_variance_port.iloc[0][0] - 1.65 * min_variance_port.iloc[0][1], 2)) + "%\n"
            + "Sharpe Ratio: " + str(round(min_variance_port.iloc[0][2], 2)) + "\n"
            + stocksStrLow,
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
            + "Annual max loss: " + str(round(sharpe_portfolio.iloc[0][0] - 1.65 * sharpe_portfolio.iloc[0][1], 2)) + "%\n"
            + "Sharpe Ratio: " + str(round(sharpe_portfolio.iloc[0][2], 2)) + "\n"
            + stocksStrMedium,
            bbox=dict(facecolor="green", alpha=0.5),
            fontsize=11,
            style="oblique",
            ha="center",
            va="center",
            fontname="Arial",
            wrap=True,
        )
    plt.show()


def plotGiniGraph(sectorsList,threeBestSectorsWeights, min_variance_port, sharpe_portfolio, max_profolios_annual, max_ginis, df):
    # plot frontier, max sharpe & min Gini values with a scatterplot
    plt.style.use('seaborn-dark')
    df.plot.scatter(x='Gini', y='Profolio_annual', c='Sharpe Ratio',
                    cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
    plt.scatter(x=sharpe_portfolio['Gini'], y=sharpe_portfolio['Profolio_annual'], c='green', marker='D', s=200)
    plt.scatter(x=min_variance_port['Gini'], y=min_variance_port['Profolio_annual'], c='orange', marker='D', s=200)
    plt.scatter(x=max_ginis['Gini'], y=max_profolios_annual['Profolio_annual'], c='red', marker='D', s=200)
    plt.style.use('seaborn-dark')

    plt.xlabel('Gini (Std. Deviation) Percentage %')
    plt.ylabel('Expected profolio annual Percentage %')
    plt.title('Efficient Frontier')
    plt.subplots_adjust(bottom=0.4)

    # ------------------ Pritning 3 optimal Protfolios -----------------------
    # Setting max_X, max_Y to act as relative border for window size
    stocksStrHigh = ""
    stocksStrLow = ""
    stocksStrMedium = ""

    # stocksStrHigh
    for i in range(len(sectorsList)):
        weight = threeBestSectorsWeights[2][i] * 100
        stocksStrHigh += sectorsList[i].getName() + "(" + str("{:.2f}".format(weight)) + "%),\n "
    # stocksStrMedium
    for i in range(len(sectorsList)):
        weight = threeBestSectorsWeights[1][i] * 100
        stocksStrMedium += sectorsList[i].getName() + "(" + str("{:.2f}".format(weight)) + "%),\n "
    # stocksStrLow
    for i in range(len(sectorsList)):
        weight = threeBestSectorsWeights[0][i] * 100
        stocksStrLow += sectorsList[i].getName() + "(" + str("{:.2f}".format(weight)) + "%),\n "

    with pd.option_context("display.float_format", "%{:,.2f}".format):
        plt.figtext(
            0.2,
            0.15,
            "Max returns Porfolio: \n"
            + "Annual returns: " + str(round(max_profolios_annual.iloc[0][0], 2)) + "%\n"
            + "Annual gini: " + str(round(max_profolios_annual.iloc[0][1], 2)) + "%\n"
            + "Annual max loss: " + str(round(max_profolios_annual.iloc[0][0] - 1.65*max_profolios_annual.iloc[0][1], 2)) + "%\n"
            + "Sharpe Ratio: " + str(round(max_profolios_annual.iloc[0][2], 2)) + "\n"
            + stocksStrHigh,
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
            + "Annual Max loss: " + str(round(min_variance_port.iloc[0][0] - 1.65 * min_variance_port.iloc[0][1], 2)) + "%\n"
            + "Sharpe Ratio: " + str(round(min_variance_port.iloc[0][2], 2)) + "\n"
            + stocksStrLow,
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
            + "Annual Max loss: " + str(round(sharpe_portfolio.iloc[0][0] - 1.65 * sharpe_portfolio.iloc[0][1], 2)) + "%\n"
            + "Sharpe Ratio: " + str(round(sharpe_portfolio.iloc[0][2], 2)) + "\n"
            + stocksStrMedium,
            bbox=dict(facecolor="green", alpha=0.5),
            fontsize=11,
            style="oblique",
            ha="center",
            va="center",
            fontname="Arial",
            wrap=True,
        )
    plt.show()


def plotbb_strategy_stock(stockprices, buy_price, sell_price):
    stockprices[['Adj Close', 'Lower', 'Upper']].plot(figsize=(10, 4))
    plt.scatter(stockprices.index, buy_price, marker='^', color='green', label='BUY', s=200)
    plt.scatter(stockprices.index, sell_price, marker='v', color='red', label='SELL', s=200)
    plt.show()

    print("number of green :")
    print(np.count_nonzero(~np.isnan(buy_price)))
    print("number of red :")
    print(np.count_nonzero(~np.isnan(sell_price)))


def plotbb_strategy_Portfolio(stockprices, buy_price, sell_price , newPortfolio): #TODO - IF NEED
    stockprices[['Adj Close', 'Lower', 'Upper']].plot(figsize=(10, 4))
    plt.scatter(stockprices.index, buy_price, marker='^', color='green', label='BUY', s=200)
    plt.scatter(stockprices.index, sell_price, marker='v', color='red', label='SELL', s=200)

    sectors = newPortfolio.getSectors()

    stocksStr = ""
    for i in range(len(sectors)):
        name = sectors[i].getName()
        weight = sectors[i].getWeight() * 100
        stocksStr += name + "(" + str("{:.2f}".format(weight)) + "%),\n "

    with pd.option_context("display.float_format", "%{:,.2f}".format):
        plt.figtext(
            0.45,
            0.15,
            "your Portfolio: \n"
            + "Returns: " + str(round(newPortfolio.getAnnualReturns(), 2)) + "%\n"
            + "Volatility: " + str(round(newPortfolio.getVolatility(), 2)) + "%\n"
            + "max loss: " + str(round(newPortfolio.getMaxLoss(), 2)) + "%\n"
            + "Sharpe Ratio: " + str(round(newPortfolio.getSharpe(), 2)) + "\n"
            + stocksStr,
            bbox=dict(facecolor="green", alpha=0.5),
            fontsize=11,
            style="oblique",
            ha="center",
            va="center",
            fontname="Arial",
            wrap=True,
        )

    plt.subplots_adjust(bottom=0.4)

    plt.show()

    print("number of green :")
    print(np.count_nonzero(~np.isnan(buy_price)))
    print("number of red :")
    print(np.count_nonzero(~np.isnan(sell_price)))


def plotThreePortfoliosGraph(min_variance_port,sharpe_portfolio,max_returns, threeBestSectorsWeights, sectorsList, pctChangeTable):
    # plot frontier, max sharpe & min Volatility values with a scatterplot
    figSize_X = 10
    figSize_Y = 8
    figSize = (figSize_X, figSize_Y)
    plt.style.use("seaborn-dark")
    plt.xlabel("Datte")
    plt.ylabel("Expected Returns")
    plt.title("3 best portfolios")

    pctChangeTable['yield_1'].plot(figsize=figSize, grid=True, color="yellow", linewidth=2, label="saftest", legend=True,
                        linestyle="dashed")
    pctChangeTable['yield_2'].plot(figsize=figSize, grid=True, color="green", linewidth=2, label="sharpe", legend=True,
                            linestyle="dashed")
    pctChangeTable['yield_3'].plot(figsize=figSize, grid=True, color="red", linewidth=2, label="max return", legend=True,
                            linestyle="dashed")

    plt.subplots_adjust(bottom=0.4)


    # ------------------ Pritning 3 optimal Protfolios -----------------------
    # Setting max_X, max_Y to act as relative border for window size
    stocksStrHigh = ""
    stocksStrLow = ""
    stocksStrMedium = ""

    # stocksStrHigh
    for i in range(len(sectorsList)):
        weight = threeBestSectorsWeights[2][i] * 100
        stocksStrHigh += sectorsList[i].getName() + "(" + str("{:.2f}".format(weight)) + "%),\n "
    # stocksStrMedium
    for i in range(len(sectorsList)):
        weight = threeBestSectorsWeights[1][i] * 100
        stocksStrMedium += sectorsList[i].getName() + "(" + str("{:.2f}".format(weight)) + "%),\n "
    # stocksStrLow
    for i in range(len(sectorsList)):
        weight = threeBestSectorsWeights[0][i] * 100
        stocksStrLow += sectorsList[i].getName() + "(" + str("{:.2f}".format(weight)) + "%),\n "

    with pd.option_context("display.float_format", "%{:,.2f}".format):
        plt.figtext(
            0.2,
            0.15,
            "Max returns Porfolio: \n"
            + "Annual returns: " + str(round(max_returns.iloc[0][0], 2)) + "%\n"
            + "Annual volatility: " + str(round(max_returns.iloc[0][1], 2)) + "%\n"
            + "Annual max loss: " + str(round(max_returns.iloc[0][0] - 1.65 * max_returns.iloc[0][1], 2)) + "%\n"
            + "Annual sharpe Ratio: " + str(round(max_returns.iloc[0][2], 2)) + "\n"
            + stocksStrHigh,
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
            + "Annual max loss: " + str(round(min_variance_port.iloc[0][0] - 1.65 * min_variance_port.iloc[0][1], 2)) + "%\n"
            + "Annual sharpe Ratio: " + str(round(min_variance_port.iloc[0][2], 2)) + "\n"
            + stocksStrLow,
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
            + "Annual Volatility: " + str(round(sharpe_portfolio.iloc[0][1], 2)) + "%\n"
            + "Annual max loss: " + str(round(sharpe_portfolio.iloc[0][0] - 1.65 * sharpe_portfolio.iloc[0][1], 2)) + "%\n"
            + "Annual sharpe Ratio: " + str(round(sharpe_portfolio.iloc[0][2], 2)) + "\n"
            + stocksStrMedium,
            bbox=dict(facecolor="green", alpha=0.5),
            fontsize=11,
            style="oblique",
            ha="center",
            va="center",
            fontname="Arial",
            wrap=True,
        )
    plt.show()

def plotpriceForcast(stockSymbol, df, isDataGotFromTase):
    if isDataGotFromTase:
        df['closingIndexPrice'].plot()
    else:
        df['Adj Close'].plot()
    df['Forecast'].plot()
    plt.title(stockSymbol + " Stock Price Forecast")
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


def plotDistributionOfStocks(stockNames, pctChangeTable):
    plt.subplots(figsize=(8, 8))
    plt.legend()
    plt.xlabel('Return', fontsize=12)
    plt.ylabel('Distribution', fontsize=12)
    for i in range(len(stockNames)):
        sns.distplot(pctChangeTable[stockNames[i]][::30]*100, kde=True, hist=False, rug=False, label=stockNames[i])
    plt.grid(True)
    plt.legend()
    plt.show()


def plotDistributionOfPortfolio(yieldsList):

    labels = ['low risk', 'medium risk', 'high risk']
    plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.4)

    monthlyChanges = [None] * len(yieldsList)# yield changes
    monthlyYields = [None] * len(yieldsList)# monthly yield change
    df_describes = [None] * len(yieldsList)# describe of yield changes
    #monthlyCompoundedReturns = [None] * len(yieldsList) # total change in percent from begining
    #monthlyCompoundedReturns[i] = (1 + monthlyChanges[i]).cumprod() - 1

    for i in range(len(yieldsList)):
        monthlyYields[i] = yieldsList[i].resample('M').first()
        monthlyChanges[i] = monthlyYields[i].pct_change().dropna() * 100
        df_describes[i] = monthlyChanges[i].describe().drop(["count"], axis=0)
        sns.distplot(pd.Series(monthlyChanges[i]), kde=True, hist_kws={'alpha': 0.2}, norm_hist=False, rug=False, label=labels[i])

    plt.xlabel('Monthly Return %', fontsize=12)
    plt.ylabel('Distribution', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.title("Distribution of Portfolios - by monthly returns")

    with pd.option_context("display.float_format", "%{:,.2f}".format):
        plt.figtext(0.2, 0.15,
            "low risk\n"
            + str(df_describes[0]),
            bbox=dict(facecolor="blue", alpha=0.5), fontsize=11, style="oblique", ha="center", va="center", fontname="Arial", wrap=True)
        plt.figtext(0.45, 0.15,
            "medium risk\n"
            + str(df_describes[1]),
            bbox=dict(facecolor="pink", alpha=0.5), fontsize=11, style="oblique", ha="center", va="center", fontname="Arial",  wrap=True )
        plt.figtext(0.7, 0.15,
            "high risk\n"
            + str(df_describes[2]),
            bbox=dict(facecolor="green", alpha=0.5), fontsize=11, style="oblique", ha="center",  va="center", fontname="Arial", wrap=True )

    plt.show()


def plotTopStocks(topStocks):
    print(topStocks)

def plot(plt):
    plt.show()