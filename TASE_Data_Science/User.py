import pandas as pd
from matplotlib import pyplot as plt
import Portfolio as Portfolio


class User:
    __name = ""
    __myPortfolio = Portfolio.Portfolio(1, 0, [], [])

    def __init__(self, name, portfolio):
        self.__name = name
        self.__myPortfolio = portfolio

    def getName(self):
        return self.__name

    def getPortfolio(self):
        return self.__myPortfolio

    def updatePortfolio(self, portfolio):
        self.__myPortfolio.setPortfolio(portfolio.getLevelOfRisk(), portfolio.getInvestmentAmount(),
                                        portfolio.getIsraeliIndexes(), portfolio.getUsaIndexes())

    def plotInvestmentPortfolioYield(self):
        portfolio = self.getPortfolio()
        table = portfolio.getPctChangeTable()
        annualReturns, volatility, sharpe = portfolio.getPortfolioStats()
        sectors = portfolio.getSectors()

        figSize_X = 10
        figSize_Y = 8
        figSize = (figSize_X, figSize_Y)
        plt.style.use("seaborn-dark")

        plt.title("Hello, " + self.getName() + "! This is your yield portfolio")
        plt.ylabel("Expected Profit")

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
                + "Returns: " + str(round(annualReturns, 2)) + "%\n"
                + "Volatility: " + str(round(volatility, 2)) + "%\n"
                + "Sharpe Ratio: " + str(round(sharpe, 2)) + "\n"
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
        table["yield"].plot(figsize=figSize, grid=True, color="green", linewidth=2, label="yield", legend=True,
                            linestyle="dashed")

        plt.show()

    def plotPortfolioComponent(self):
        portfolio = self.getPortfolio()
        sectorsWeights = portfolio.getSectorsWeights()
        sectorsNames = portfolio.getSectorsNames()
        plt.title("Hello, " + self.getName() + "! This is your portfolio")
        plt.pie(
            sectorsWeights,
            labels=sectorsNames,
            autopct="%1.1f%%",
            shadow=True,
            startangle=140,
        )
        plt.axis("equal")
        plt.show()
