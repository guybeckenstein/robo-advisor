#!/usr/bin/python
# -*- coding: utf-8 -*-
from RoboAdvisorDataScience.api import Sector as Sector
import datetime


class Portfolio:

    __sectors = []
    __levelOfRisk = 1
    __startingInvestmentAmount = 0
    __lastDateOfInvestment = None
    __stocksSymbols = []
    __stocksWeights = []
    __closingPricesTable = []
    __pctChangeTable = []
    __annualReturns = None
    __annualVolatility = None
    __annualSharpe = None
    __selectedModel = None
    __machineLearningOpt = None

    def __init__(self, levelOfRisk, startingInvestmentAmount, stocksSymbols, sectorsData,
                 selectedModel, machineLearningOpt):

        self.__levelOfRisk = levelOfRisk
        self.__startingInvestmentAmount = startingInvestmentAmount
        self.__stocksSymbols = stocksSymbols
        self.setSectors(sectorsData)
        self.__selectedModel = selectedModel
        self.__machineLearningOpt = machineLearningOpt
        self.__lastDateOfInvestment = datetime.datetime.now().date()

    # GETTERS

    def getLevelOfRisk(self):
        return self.__levelOfRisk

    def getInvestmentAmount(self):
        return self.__startingInvestmentAmount

    def getSelectedModel(self):
        return self.__selectedModel

    def getMachineLearningOpt(self):
        return self.__machineLearningOpt

    def getStocksSymbols(self):
        return self.__stocksSymbols

    def getStocksWeights(self):
        return self.__stocksWeights

    def getClosingPricesTable(self):
        return self.__closingPricesTable

    def getPctChangeTable(self):
        return self.__pctChangeTable

    def getAnnualReturns(self):
        return self.__annualReturns

    def getMaxLoss(self):
        return self.__annualReturns - 1.65 * self.__annualVolatility

    def getVolatility(self):
        return self.__annualVolatility

    def getSharpe(self):
        return self.__annualSharpe

    def getSectors(self):
        return self.__sectors

    def getIsraeliStocksIndexes(self):
        israeliStcoksIndexes = []

        for i in range(3):
            israeliStcoksIndexes.extend(self.__sectors[i].getStocks())

        return israeliStcoksIndexes

    def getIsraeliBondsStocksIndexes(self):
        israeliBondsStcoksIndexes = []

        for i in range(1, 2):
            israeliBondsStcoksIndexes.extend(self.__sectors[i].getStocks())

        return israeliBondsStcoksIndexes

    def getUsaStocksIndexes(self):
        usaStocksIndexes = []

        for i in range(3, 6):
            usaStocksIndexes.extend(self.__sectors[i].getStocks())

        return usaStocksIndexes

    def getPortfolioStats(self):
        return self.__annualReturns, self.__annualVolatility, self.__annualSharpe, self.getMaxLoss()

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

    def getPortfolioData(self):
        return self.__levelOfRisk, self.__startingInvestmentAmount, self.__stocksSymbols, self.getSectorsNames(),\
               self.getSectorsWeights(), self.__stocksWeights, self.__annualReturns, self.getMaxLoss(),\
               self.__annualVolatility, self.__annualSharpe, self.getTotalChange(), self.getMonthlyChange(),\
               self.getDailyChange(), self.getSelectedModel(), self.getMachineLearningOpt()

    # get changes
    def getTotalChange(self):
        totalChange = (self.getTotalValueChange()+self.__startingInvestmentAmount)/self.__startingInvestmentAmount
        return totalChange

    def getYearlyChange(self):
        yearlyYields = self.__pctChangeTable['yield_selected'].resample('M').first()
        yearlyChanges = yearlyYields.pct_change().dropna() * 100
        return yearlyChanges[-1]

    def getMonthlyChange(self):
        monthlyYields = self.__pctChangeTable['yield_selected'].resample('M').first()
        monthlyChanges = monthlyYields.pct_change().dropna() * 100
        return monthlyChanges[-1]

    def getDailyChange(self):
        return self.__pctChangeTable['weighted_sum_selected'].iloc[-1]

    def getTotalValueChange(self):
        return self.__pctChangeTable['yield_selected'].iloc[-1] - self.__pctChangeTable['yield_selected'].iloc[0]

    def getYearlyValueChange(self):
        return self.getYearlyPctChangeTable()["yield_selected"].iloc[-1] - self.getYearlyPctChangeTable()
        ["yield_selected"].iloc[-2]

    def getMonthlyValueChange(self):
        return self.getMonthlyPctChangeTable()["yield_selected"].iloc[-1] - self.getMonthlyPctChangeTable()
        ["yield_selected"].iloc[-2]

    def getDailyValueChange(self):
        return self.__pctChangeTable['yield_selected'].iloc[-1] - self.__pctChangeTable['yield_selected'].iloc[-2]

    # get tables
    def getclosingPricesTable(self):
        return self.__closingPricesTable

    def getMonthlyPctChangeTable(self):
        table = self.__pctChangeTable.resample('M').apply(lambda x: (x[-1] / x[0] - 1) * 100)
        return table

    def getYearlyPctChangeTable(self):
        return self.__pctChangeTable.resample('Y').apply(lambda x: (x[-1] / x[0] - 1) * 100)

    # SETTERS and UPDATERS
    def updateStocksData(self, closingPricesTable, pctChangeTable, stockWeights, annualReturns,
                         annualVolatility, annualSharpe):
        self.setTables(closingPricesTable, pctChangeTable)
        self.setStocksWeights(stockWeights)
        self.__annualReturns = annualReturns
        self.__annualVolatility = annualVolatility
        self.__annualSharpe = annualSharpe

        pctChangeTable["weighted_sum_selected"] = pctChangeTable["weighted_sum_" + str(self.__levelOfRisk)]
        pctChangeTable["yield_selected"] = pctChangeTable["yield_" + str(self.__levelOfRisk)]

        for i in range(len(self.__stocksSymbols)):
            for j in range(len(self.__sectors)):
                if self.__stocksSymbols[i] in self.__sectors[j].getStocks():
                    self.__sectors[j].addWeight(self.__stocksWeights[i])

    def setSectors(self, sectorsData):
        if (len(sectorsData)) > 0:
            sectorsData = sectorsData['sectorsList']['result']
            for i in range(len(sectorsData)):
                self.setSector(sectorsData[i]['sectorName'])
                for j in range(len(self.__stocksSymbols)):
                    if self.__stocksSymbols[j] in sectorsData[i]['stocks']:
                        self.setSectorStockByIndex(i, self.__stocksSymbols[j])

    def updateLevelOfRisk(self, levelOfRisk):
        self.__levelOfRisk = levelOfRisk

    def updateInvestmentAmount(self, investmentAmount):
        self.__startingInvestmentAmount = investmentAmount

    def setStocksWeights(self, stocksWeights):
        if type(stocksWeights) == list:
            self.__stocksWeights = stocksWeights
        else:
            ndArray = stocksWeights.values
            newList = []*len(ndArray)
            for i in range(len(ndArray)):
                newList.append(ndArray[i])
            self.__stocksWeights = newList

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

    def addStockSymbol(self, stockSymbol):

        # TODO VALIDATION

        for i in range(len(self.__sectors)):
            if stockSymbol in self.__sectors[i].getStocks():
                self.__sectors[i].getStocks().append(stockSymbol)
        self.__stocksSymbols.append(stockSymbol)

    def removeStockSymbol(self, stockSymbol):
        for i in range(len(self.__sectors)):
            if stockSymbol in self.__sectors[i].getStocks():
                self.__sectors[i].getStocks().remove(stockSymbol)
        self.__stocksSymbols.remove(stockSymbol)

    def returnSectorsWeightsAccordingToStocksWeights(self, stocksWeights):
        sectorsWeights = [0.0] * len(self.__sectors)
        for i in range(len(self.__sectors)):
            sectorsWeights[i] = 0
            for j in range(len(self.__stocksSymbols)):
                if self.__stocksSymbols[j] in self.__sectors[i].getStocks():
                    sectorsWeights[i] += stocksWeights[j]
        return sectorsWeights
