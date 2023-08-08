import numpy as np
import pandas as pd
from ..util import helpers


class StatsModels:

    def __init__(
            self,
            stocks_symbols,
            pct_change_table,
            num_por_simulation,
            min_num_por_simulation,
            max_percent_commodity,
            max_percent_stocks,
            closing_prices_table = None,
            sectors = None,
            model_name = None,
            is_machine_learning: int = 0
    ):
        self._df = None
        self._three_best_portfolios = None
        self._three_best_stocks_weights = None
        self._three_best_sectors_weights = None
        self._best_stocks_weights_column = None
        self._stock_sectors = helpers.set_stock_sectors(stocks_symbols, sectors)
        self._sectors_list = sectors
        self._model_name = model_name
        self._closing_prices_table = closing_prices_table
        if model_name == "Markowitz":
            self.get_optimal_portfolio_by_markowitz(
                num_por_simulation, min_num_por_simulation, closing_prices_table, pct_change_table,  stocks_symbols,
                max_percent_commodity, max_percent_stocks, is_machine_learning
            )
        else:
            self.get_optimal_portfolio_by_gini(
                num_por_simulation, min_num_por_simulation, pct_change_table,  stocks_symbols,
                max_percent_commodity, max_percent_stocks, is_machine_learning
            )

    def get_optimal_portfolio_by_markowitz(self, num_por_simulation, min_num_por_simulation, closing_prices_table,
                                           pct_change_table, stocks_symbols, max_percent_commodity, max_percent_stocks,
                                           is_machine_learning: int):

        stocks_names: list = []
        for symbol in stocks_symbols:
            if type(symbol) == int:
                stocks_names.append(str(symbol))
            else:
                stocks_names.append(symbol)


        returns_daily = pct_change_table
        returns_annual = ((1 + returns_daily.mean()) ** 254) - 1
        cov_daily = returns_daily.cov()
        cov_annual = cov_daily * 254

        # empty lists to store returns, volatility and weights of imaginary portfolios
        port_returns: list = []
        port_volatility: list = []
        sharpe_ratio: list = []
        stock_weights: list = []

        # set the number of combinations for imaginary portfolios
        num_assets: int = len(stocks_symbols)
        num_portfolios = num_por_simulation

        # set random seed for reproduction's sake
        np.random.seed(101)

        # populate the empty lists with each portfolios returns,risk and weights
        single_portfolio = 0
        while single_portfolio < num_portfolios:
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            if single_portfolio >= (num_portfolios - 1) and len(stock_weights) < min_num_por_simulation:
                num_portfolios *= 2
            # Calculate the percentage of stocks in the "Commodity" sector
            commodity_percentage = np.sum(
                [weights[i] for i in range(num_assets) if self._stock_sectors[i] == "US commodity"]
            )
            israeli_stocks_percentage = np.sum(
                [weights[i] for i in range(num_assets) if self._stock_sectors[i] == "Israel stocks"]
            )
            us_stocks_percentage = np.sum(
                [weights[i] for i in range(num_assets) if self._stock_sectors[i] == "US stocks"]
            )

            if commodity_percentage > max_percent_commodity:
                single_portfolio += 1
                continue  # Skip this portfolio and generate a new one
            if (israeli_stocks_percentage + us_stocks_percentage) > max_percent_stocks:
                single_portfolio += 1
                continue  # Skip this portfolio and generate a new one

            returns = np.dot(weights, returns_annual)
            volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
            sharpe = returns / volatility
            sharpe_ratio.append(sharpe)
            port_returns.append(returns * 100)
            port_volatility.append(volatility * 100)
            stock_weights.append(weights)

            single_portfolio += 1

        # a dictionary for Returns and Risk values of each portfolio
        portfolio: dict = {
            "Returns": port_returns,
            "Volatility": port_volatility,
            "Sharpe Ratio": sharpe_ratio,
        }

        # extend original dictionary to accommodate each ticker and weight in the portfolio
        for counter, symbol in enumerate(stocks_names):
            portfolio[symbol + " Weight"] = [Weight[counter] for Weight in stock_weights]

        # make a nice dataframe of the extended dictionary
        df = pd.DataFrame(portfolio)

        # get better labels for desired arrangement of columns
        column_order = ["Returns", "Volatility", "Sharpe Ratio"] + [stock + " Weight" for stock in stocks_names]
        # reorder dataframe columns
        self._df = df[column_order]

    def get_optimal_portfolio_by_gini(self, num_por_simulation, min_num_por_simulation, pct_change_table, stocks_symbols,
                                      max_percent_commodity, max_percent_stocks, is_machine_learning: int):
        stocks_names: list = []
        for symbol in stocks_symbols:
            if type(symbol) == int:
                stocks_names.append(str(symbol))
            else:
                stocks_names.append(symbol)
        v_value = helpers.settings.GINI_V_VALUE  # TODO: not recognizing method
        returns_daily = pct_change_table
        port_portfolio_annual = []
        port_gini_annual = []
        sharpe_ratio = []
        stock_weights = []

        # set the number of combinations for imaginary portfolios
        num_assets = len(stocks_names)
        num_portfolios = num_por_simulation

        # set random seed for reproduction's sake
        np.random.seed(101)

        # Mathematical calculations, creation of 5000 portfolios,
        for _ in returns_daily.keys():
            # populate the empty lists with each portfolios returns,risk and weights
            single_portfolio = 0
            while single_portfolio < num_portfolios:
                weights = np.random.random(num_assets)
                weights /= np.sum(weights)
                if (single_portfolio >= (num_portfolios - 1)) and (len(stock_weights) < min_num_por_simulation):
                    num_portfolios *= 2
                # Calculate the percentage of stocks in the "Commodity" sector
                commodity_percentage = np.sum(
                    [weights[i] for i in range(num_assets) if self._stock_sectors[i] == "US commodity"])
                israeli_stocks_percentage = np.sum(
                    [weights[i] for i in range(num_assets) if self._stock_sectors[i] == "Israel stocks"])
                us_stocks_percentage = np.sum(
                    [weights[i] for i in range(num_assets) if self._stock_sectors[i] == "US stocks"])

                if commodity_percentage > max_percent_commodity:
                    single_portfolio += 1
                    continue  # Skip this portfolio and generate a new one
                if (israeli_stocks_percentage + us_stocks_percentage) > max_percent_stocks:
                    single_portfolio += 1
                    continue  # # Skip this portfolio and generate a new one

                portfolio = np.dot(returns_daily, weights)
                portfolio_return = pd.DataFrame(portfolio)
                rank = portfolio_return.rank()
                rank_divided_n = rank / len(rank)  # Rank/N
                one_sub_rank_divided_n = 1 - rank_divided_n  # 1-Rank/N
                one_sub_rank_divided_n_power_v_sub_one = one_sub_rank_divided_n ** (v_value - 1)  # (1-Rank/N)^(V-1)
                mue = portfolio_return.mean().tolist()[0]
                x_avg = one_sub_rank_divided_n_power_v_sub_one.mean().tolist()[0]
                portfolio_mue = portfolio_return - mue
                rank_sub_x_avg = one_sub_rank_divided_n_power_v_sub_one - x_avg
                portfolio_mue_mult_rank_x_avg = portfolio_mue * rank_sub_x_avg
                summary = portfolio_mue_mult_rank_x_avg.sum().tolist()[0] / (len(rank) - 1)
                gini_daily = summary * (-v_value)
                gini_annual = gini_daily * (254 ** 0.5)
                portfolio_annual = ((1 + mue) ** 254) - 1
                sharpe = portfolio_annual / gini_annual
                sharpe_ratio.append(sharpe)
                port_portfolio_annual.append(portfolio_annual * 100)
                port_gini_annual.append(gini_annual * 100)
                stock_weights.append(weights)

                single_portfolio += 1

            # a dictionary for Returns and Risk values of each portfolio
            portfolio = {'Portfolio_annual': port_portfolio_annual,
                         'Gini': port_gini_annual,
                         'Sharpe Ratio': sharpe_ratio}

            # extend original dictionary to accommodate each ticker and weight in the portfolio
            for counter, symbol in enumerate(stocks_names):
                portfolio[symbol + ' Weight'] = [Weight[counter] for Weight in stock_weights]

            # make a nice dataframe of the extended dictionary
            df = pd.DataFrame(portfolio)

            # get better labels for desired arrangement of columns
            column_order = ['Portfolio_annual', 'Gini', 'Sharpe Ratio'] + [stock + ' Weight' for stock in stocks_names]

            # reorder dataframe columns
            self._df = df[column_order]



    def get_df(self):
        return self._df

    def get_three_best_portfolios(self):
        return self._three_best_portfolios

    def get_three_best_weights(self):
        return self._three_best_stocks_weights

    def get_three_best_sectors_weights(self):
        return self._three_best_sectors_weights

    def get_best_stocks_weights_column(self):
        return self._best_stocks_weights_column

    def get_final_portfolio(self, risk_score):
        return helpers.choose_portfolio_by_risk_score(self._three_best_portfolios, risk_score)

    def get_closing_prices_table(self):
        return self._closing_prices_table

    def get_pct_change_table(self):
        table = self._closing_prices_table
        return table.pct_change()

    def get_max_vols(self):
        if self._model_name == "Markowitz":
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
