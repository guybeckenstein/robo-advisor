import codecs
import json
import pandas as pd
from matplotlib import pyplot as plt
from api import Portfolio as Portfolio


class User:

    __name = ""
    __myPortfolio = Portfolio.Portfolio(1, 0, [], [], 1, 0)

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
        annualReturns, volatility, sharpe, maxLoss = portfolio.getPortfolioStats()
        totalChange = portfolio.getTotalChange()
        sectors = portfolio.getSectors()

        figSize_X = 10
        figSize_Y = 8
        figSize = (figSize_X, figSize_Y)
        plt.style.use("seaborn-dark")

        plt.title("Hello, " + self.getName() + "! This is your yield portfolio")
        plt.ylabel("Returns %")

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
                + "Total change: " + str(round(totalChange, 2)) + "%\n"
                + "Annual returns: " + str(round(annualReturns, 2)) + "%\n"
                + "Annual volatility: " + str(round(volatility, 2)) + "%\n"
                + "max loss: " + str(round(maxLoss, 2)) + "%\n"
                + "Annual sharpe Ratio: " + str(round(sharpe, 2)) + "\n"
                + stocksStr,
                bbox=dict(facecolor="green", alpha=0.5),
                fontsize=11,
                style="oblique",
                ha="center",
                va="center",
                fontname="Arial",
                wrap=True,
            )
        table['yield__selected_percent'] = (table["yield_selected"] - 1) * 100
        table['yield__selected_percent'].plot(figsize=figSize, grid=True, color="green", linewidth=2, label="yield",
                                     legend=True, linestyle="dashed")

        plt.subplots_adjust(bottom=0.4)

        return plt

    def plotPortfolioComponent(self):
        portfolio = self.getPortfolio()
        sectorsWeights = portfolio.getSectorsWeights()
        labels = portfolio.getSectorsNames()
        plt.title("Hello, " + self.getName() + "! This is your portfolio")
        plt.pie(
            sectorsWeights,
            labels=labels,
            autopct="%1.1f%%",
            shadow=True,
            startangle=140,
        )
        plt.axis("equal")
        return plt

    def plotPortfolioComponentStocks(self):
        portfolio = self.getPortfolio()
        stocksWeights = portfolio.getStocksWeights()
        labels = portfolio.getStocksSymbols()
        plt.title("Hello, " + self.getName() + "! This is your portfolio")
        plt.pie(
            stocksWeights,
            labels=labels,
            autopct="%1.1f%%",
            shadow=True,
            startangle=140,
        )
        plt.axis("equal")
        return plt



    def getJsonData(self, name):
        with codecs.open(name + ".json", "r", encoding="utf-8") as file:
            json_data = json.load(file)
        return json_data

    def updateJsonFile(self, jsonName):
        jsonData = self.getJsonData(jsonName)

        (levelOfRisk, startingInvestmentAmount, stocksSymbols, sectorsNames, sectorsWeights, stocksWeights,
         annualReturns, annualMaxLoss, annualVolatility, annualSharpe, totalChange, monthlyChange,
         dailyChange, selectedModel, machineLearningOpt) = self.__myPortfolio.getPortfolioData()

        # Create a new dictionary
        # TODO- change startingInvestmentAmount with something better(yarden)
        new_user_data = {
            "levelOfRisk": levelOfRisk,
            "startingInvestmentAmount": startingInvestmentAmount,
            "stocksSymbols": stocksSymbols,
            "sectorsNames": sectorsNames,
            "sectorsWeights": sectorsWeights,
            "stocksWeights": stocksWeights,
            "annualReturns": annualReturns,
            "annualMaxLoss": annualMaxLoss,
            "annualVolatility": annualVolatility,
            "annualSharpe": annualSharpe,
            "totalChange": totalChange,
            "monthlyChange": monthlyChange,
            "dailyChange": dailyChange,
            "selectedModel": selectedModel,
            "machineLearningOpt": machineLearningOpt
        }
        jsonData['usersList'][self.__name] = [new_user_data]

        with open(jsonName + ".json", 'w') as f:
            json.dump(jsonData, f, indent=4)
