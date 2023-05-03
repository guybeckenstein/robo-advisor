#!/usr/bin/python
# -*- coding: utf-8 -*-
import Sector as Sector


class Portfolio:

    # private members
    # list of Sector

    __sectors = []
    __levelOfRisk = 1
    __investmentAmount = 0
    __stocksSymbols = []
    __stocksWeights = []
    __closingPricesTable = []
    __pctChangeTable = []
    __annualReturns = None
    __volatility = None
    __sharpe = None

    def __init__(self, levelOfRisk, investmentAmount, stocksSymbols, sectorsNames):
        self.__levelOfRisk = levelOfRisk
        self.__investmentAmount = investmentAmount
        self.__stocksSymbols = stocksSymbols
        self.setSectors(sectorsNames)

    # GETTERS

    def getLevelOfRisk(self):
        return self.__levelOfRisk

    def getInvestmentAmount(self):
        return self.__investmentAmount

    def getStocksSymbols(self):
        return self.__stocksSymbols

    def getSectors(self):
        return self.__sectors

    def getIsraeliStocksIndexes(self):
        israeliStcoksIndexes = []
        for i in range(3):
            israeliStcoksIndexes.extend(self.__sectors[i].getStocks())
        return israeliStcoksIndexes

    def getUsaStocksIndexes(self):
        usaStocksIndexes = []
        for i in range(3, 6):
            usaStocksIndexes.extend(self.__sectors[i].getStocks())
        return usaStocksIndexes

    """def getStocksSymbols(self):
        return self.__stocksSymbols"""

    def getclosingPricesTable(self):
        return self.__closingPricesTable

    def getPctChangeTable(self):
        return self.__pctChangeTable

    def getPortfolioStats(self):
        return self.__annualReturns, self.__volatility, self.__sharpe

    def getStocksWeights(self):
        return self.__stocksWeights

    def getSector(self, sectorName):
        for sector in self.__sectors:
            if sector.getName() == sectorName:
                return sector
        return None

    def getSectorByIndex(self, index):
        return self.__sectors[index]

    def getSectorStocks(self, sectorName):
        for sector in self.__sectors:
            if sector.getName() == sectorName:
                return sector.getStocks()
        return None

    def getSectorStocksByIndex(self, index):
        return self.__sectors[index].getStocks()

    def getSectorWeight(self, sectorName):
        for sector in self.__sectors:
            if sector.getName() == sectorName:
                return sector.getWeight()
        return None

    def getSectorWeightByIndex(self, index):
        return self.__sectors[index].getWeight()

    def getSectorsWeights(self):
        weights = []
        for sector in self.__sectors:
            weights.append(sector.getWeight())
        return weights

    def getSectorsNames(self):
        names = []
        for sector in self.__sectors:
            names.append(sector.getName())
        return names

    # SETTERS and UPDATERS

    def updateStocksData(self, closingPricesTable, pctChangeTable, stock_weights, annualReturns, volatility, sharpe):
        self.setTables(closingPricesTable, pctChangeTable)
        self.setStocksWeights(stock_weights)
        self.__annualReturns = annualReturns
        self.__volatility = volatility
        self.__sharpe = sharpe

        for i in range(len(self.__stocksSymbols)):
            for j in range(len(self.__sectors)):
                if self.__stocksSymbols[i] in self.__sectors[j].getStocks():
                    self.__sectors[j].addWeight(self.__stocksWeights[i])

    def setSectors(self, sectors):
        for i in range(len(sectors)):
            self.setSector(sectors[i])
        self.arrangeStocksToSectors()

    def updateLevelOfRisk(self, levelOfRisk):
        self.__levelOfRisk = levelOfRisk

    def updateInvestmentAmount(self, investmentAmount):
        self.__investmentAmount = investmentAmount

    def setStocksWeights(self, stocksWeights):
        self.__stocksWeights = stocksWeights

    def setSector(self, name):
        sector = Sector.Sector(name)
        self.__sectors.append(sector)

    def setSectorWeight(self, name, weight):
        for i in range(len(self.__sectors)):
            if self.__sectors[i].getName() == name:
                self.__sectors[i].setWeight(weight)

    def setSectorWeightByIndex(self, index, weight):
        self.__sectors[index].setWeight(weight)

    def setSectorsWeights(self, SectorsWeights):
        for i in range(len(SectorsWeights)):
            self.setSectorWeightByIndex(i, SectorsWeights[i])

    def setSectorStocks(self, name, stocks):
        for i in range(len(self.__sectors)):
            if self.__sectors[i].getName() == name:
                self.__sectors[i].setStocks(stocks)

    def setSectorStockByIndex(self, index, stock):
        self.__sectors[index].addStock(stock)

    def setTables(self, closingPricesTable, pctChangeTable):
        self.__closingPricesTable = closingPricesTable
        self.__pctChangeTable = pctChangeTable

    def arrangeStock(self, stock):
        if type(stock) == int:
            if stock < 600:
                self.setSectorStockByIndex(0, stock)
            elif stock < 700:
                self.setSectorStockByIndex(1, stock)
            else:
                self.setSectorStockByIndex(2, stock)
        else:
            if stock == 'SPY':  # TODO DEFINE LIST OF USA INDEXES
                self.setSectorStockByIndex(3, stock)
            elif stock == 'Gsg':
                self.setSectorStockByIndex(5, stock)
            else:
                self.setSectorStockByIndex(4, stock)

    def arrangeStocksToSectors(self):
        if len(self.__stocksSymbols) > 0:
            for stock in self.__stocksSymbols:
                self.arrangeStock(stock)

    def addStockSymbol(self, stockSymbol):
        self.arrangeStock(stockSymbol)

    def removeStockSymbol(self, stockSymbol):
        for i in range(len(self.__sectors)):
            if stockSymbol in self.__sectors[i].getStocks():
                self.__sectors[i].getStocks().remove(stockSymbol)

    def returnSectorsWeightsAccordingToStocksWeights(self, stocksWeights):
        sectorsWeights = [0.0] * len(self.__sectors)
        for i in range(len(self.__sectors)):
            sectorsWeights[i] = 0
            for j in range(len(self.__stocksSymbols)):
                if self.__stocksSymbols[j] in self.__sectors[i].getStocks():
                    sectorsWeights[i] += stocksWeights[j]
        return sectorsWeights
