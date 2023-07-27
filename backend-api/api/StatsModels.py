import numpy as np
import pandas as pd
from util import apiUtil


class StatsModels:

    _df = None
    _threeBestPortfolios = None
    _threeBestStocksWeights = None
    _threeBestSectorsWeights = None
    _closingPricesTable = None
    _bestStocksWeightsColumn = None
    _stock_sectors = None
    _sectorsList = None
    _limitPercenLowRiskCommodity = None
    _limitPercentMediumRiskCommodity = None
    _limitPercenLowRiskStocks = None
    _limitPercentMediumRiskStocks = None
    _modelName = None


    def __init__(self, stocksSymbols, sectorsList, closingPricesTable, Num_porSimulation, minNumporSimulation,
                 maxPercentCommodity,
                 maxPercentStocks, modelName):
        self._stock_sectors = apiUtil.setStockSectors(stocksSymbols, sectorsList)
        self._sectorsList = sectorsList
        self._modelName = modelName
        self._closingPricesTable = closingPricesTable
        if modelName == "Markowitz":
            self.GetOptimalPortfolioByMarkowitz(Num_porSimulation, minNumporSimulation, self._closingPricesTable,
                                                           stocksSymbols, maxPercentCommodity, maxPercentStocks)
        else:
            self.GetOptimalPortfolioByGini(Num_porSimulation, minNumporSimulation, self._closingPricesTable,
                                                      stocksSymbols, maxPercentCommodity, maxPercentStocks)

    def GetOptimalPortfolioByMarkowitz(self, Num_porSimulation, minNumporSimulation, closingPricesTable, stocksSymbols,
                                       maxPercentCommodity, maxPercentStocks):

        stocksNames = []
        for symbol in stocksSymbols:
            if type(symbol) == int:
                stocksNames.append(str(symbol))
            else:
                stocksNames.append(symbol)

        returns_daily = closingPricesTable.pct_change()
        returns_annual = ((1 + returns_daily.mean()) ** 254) - 1
        cov_daily = returns_daily.cov()
        cov_annual = cov_daily * 254

        # empty lists to store returns, volatility and weights of imiginary portfolios
        port_returns = []
        port_volatility = []
        sharpe_ratio = []
        stock_weights = []

        # set the number of combinations for imaginary portfolios
        num_assets = len(stocksSymbols)
        num_portfolios = Num_porSimulation

        # set random seed for reproduction's sake
        np.random.seed(101)

        # populate the empty lists with each portfolios returns,risk and weights
        single_portfolio = 0
        while single_portfolio < num_portfolios:
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            if(single_portfolio >= (num_portfolios - 1) and len(stock_weights) < minNumporSimulation):
                num_portfolios *= 2
            # Calculate the percentage of stocks in the "Commodity" sector
            commodity_percentage = np.sum([weights[i] for i in range(num_assets) if self._stock_sectors[i] == "US commodity"])
            israeli_stocks_percentage = np.sum([weights[i] for i in range(num_assets) if self._stock_sectors[i] == "Israel stocks"])
            us_stocks_percentage = np.sum([weights[i] for i in range(num_assets) if self._stock_sectors[i] == "US stocks"])

            if commodity_percentage > maxPercentCommodity:
                single_portfolio += 1
                continue  # Skip this portfolio and generate a new one
            if (israeli_stocks_percentage + us_stocks_percentage) > maxPercentStocks:
                single_portfolio += 1
                continue # # Skip this portfolio and generate a new one

            returns = np.dot(weights, returns_annual)
            volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
            sharpe = returns / volatility
            sharpe_ratio.append(sharpe)
            port_returns.append(returns * 100)
            port_volatility.append(volatility * 100)
            stock_weights.append(weights)

            single_portfolio += 1

        # a dictionary for Returns and Risk values of each portfolio
        portfolio = {
            "Returns": port_returns,
            "Volatility": port_volatility,
            "Sharpe Ratio": sharpe_ratio,
        }

        # extend original dictionary to accomodate each ticker and weight in the portfolio
        for counter, symbol in enumerate(stocksNames):
            portfolio[symbol + " Weight"] = [Weight[counter] for Weight in stock_weights]

        # make a nice dataframe of the extended dictionary
        df = pd.DataFrame(portfolio)

        # get better labels for desired arrangement of columns
        column_order = ["Returns", "Volatility", "Sharpe Ratio"] + [
            stock + " Weight" for stock in stocksNames
        ]
        # reorder dataframe columns
        self._df = df[column_order]


    def GetOptimalPortfolioByGini(self, Num_porSimulation, minNumporSimulation, table, stocksSymbols,
                                  maxPercentCommodity, maxPercentStocks):
        stocksNames = []
        for symbol in stocksSymbols:
            if type(symbol) == int:
                stocksNames.append(str(symbol))
            else:
                stocksNames.append(symbol)
        vValue = 4
        returns_daily = table.pct_change()
        port_profolio_annual = []
        port_gini_annual = []
        sharpe_ratio = []
        stock_weights = []

        # set the number of combinations for imaginary portfolios
        num_assets = len(stocksNames)
        num_portfolios = Num_porSimulation

        # set random seed for reproduction's sake
        np.random.seed(101)

        # Mathematical calculations, creation of 5000 portfolios,
        for stock in returns_daily.keys():
            # populate the empty lists with each portfolios returns,risk and weights
            single_portfolio = 0
            while single_portfolio < num_portfolios:
                weights = np.random.random(num_assets)
                weights /= np.sum(weights)
                if (single_portfolio >= (num_portfolios - 1) and len(stock_weights) < minNumporSimulation):
                    num_portfolios *= 2
                # Calculate the percentage of stocks in the "Commodity" sector
                commodity_percentage = np.sum(
                    [weights[i] for i in range(num_assets) if self._stock_sectors[i] == "US commodity"])
                israeli_stocks_percentage = np.sum(
                    [weights[i] for i in range(num_assets) if self._stock_sectors[i] == "Israel stocks"])
                us_stocks_percentage = np.sum(
                    [weights[i] for i in range(num_assets) if self._stock_sectors[i] == "US stocks"])

                if commodity_percentage > maxPercentCommodity:
                    single_portfolio += 1
                    continue  # Skip this portfolio and generate a new one
                if (israeli_stocks_percentage + us_stocks_percentage) > maxPercentStocks:
                    single_portfolio += 1
                    continue  # # Skip this portfolio and generate a new one

                profolio = np.dot(returns_daily, weights)
                profolio_return = pd.DataFrame(profolio)
                rank = profolio_return.rank()
                rank_divided_N = rank / len(rank)  # Rank/N
                one_sub_rank_divided_N = 1 - rank_divided_N  # 1-Rank/N
                one_sub_rank_divided_N_power_v_sub_one = one_sub_rank_divided_N ** (vValue - 1)  # (1-Rank/N)^(V-1)
                mue = profolio_return.mean().tolist()[0]
                x_avg = one_sub_rank_divided_N_power_v_sub_one.mean().tolist()[0]
                profolio_mue = profolio_return - mue
                rank_sub_x_avg = one_sub_rank_divided_N_power_v_sub_one - x_avg
                profolio_mue_mult_rank_x_avg = profolio_mue * rank_sub_x_avg
                summary = profolio_mue_mult_rank_x_avg.sum().tolist()[0] / (len(rank) - 1)
                gini_daily = summary * (-vValue)
                gini_annual = gini_daily * (254 ** 0.5)
                profolio_annual = ((1 + mue) ** 254) - 1
                sharpe = profolio_annual / gini_annual
                sharpe_ratio.append(sharpe)
                port_profolio_annual.append(profolio_annual * 100)
                port_gini_annual.append(gini_annual * 100)
                stock_weights.append(weights)

                single_portfolio += 1

            # a dictionary for Returns and Risk values of each portfolio
            portfolio = {'Profolio_annual': port_profolio_annual,
                         'Gini': port_gini_annual,
                         'Sharpe Ratio': sharpe_ratio}

            # extend original dictionary to accomodate each ticker and weight in the portfolio
            for counter, symbol in enumerate(stocksNames):
                portfolio[symbol + ' Weight'] = [Weight[counter] for Weight in stock_weights]

            # make a nice dataframe of the extended dictionary
            df = pd.DataFrame(portfolio)

            # get better labels for desired arrangement of columns
            column_order = ['Profolio_annual', 'Gini', 'Sharpe Ratio'] + [stock + ' Weight' for stock in stocksNames]

            # reorder dataframe columns
            self._df = df[column_order]

    def getDf(self):
        return self._df

    def getThreeBestPortfolios(self):
        return self._threeBestPortfolios

    def getThreeBestWeights(self):
        return self._threeBestStocksWeights

    def getThreeBestSectorsWeights(self):
        return self._threeBestSectorsWeights

    def getBestStocksWeightsColumn(self):
        return self._bestStocksWeightsColumn

    def getFinalPortfolio(self, riskScore):
        return apiUtil.choosePortfolioByRiskScore(self._threeBestPortfolios, riskScore)

    def getClosingPricesTable(self):
        return self._closingPricesTable

    def getPctChangeTable(self):
        table = self._closingPricesTable
        return table.pct_change()

    def getMaxVols(self):
        if self._modelName == "Markowitz":
            max_vol = self._df["Volatility"].max()
            max_vol_portfolio = self._df.loc[self._df['Volatility'] == max_vol]
        else:
            max_vol = self._df["Gini"].max()
            max_vol_portfolio = self._df.loc[self._df['Gini'] == max_vol]
        return max_vol_portfolio

    # def get(self):
        # pass
        # response = ResponseApi("Markowitz", final_invest_portfolio, amountToInvest, datetime.datetime.now())
        # return jsonify(response.__str__())
