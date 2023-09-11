from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from service.impl.sector import Sector
from service.util import helpers


@dataclass(init=True, order=False, frozen=False)
class StatsModels:
    _model_name: str
    _gini_v_value: float
    # Default values
    _stocks_symbols: list[int | str] = field(default_factory=list)
    _sectors: list[Sector] = field(default_factory=list)
    _df: pd.DataFrame = field(default=None)

    def __post_init__(self):
        self._stock_sectors = helpers.set_stock_sectors(self._stocks_symbols, self._sectors)

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def check_model_name_and_get_optimal_portfolio_as_dataframe(
            self, num_por_simulation: int, min_num_por_simulation: int, pct_change_table: pd.DataFrame,
            max_percent_commodity: float, max_percent_stocks: list[float]
    ):
        if self._model_name == "Markowitz":
            self._get_markowitz_optimal_portfolio_as_dataframe(
                num_por_simulation, min_num_por_simulation, pct_change_table, self._stocks_symbols,
                max_percent_commodity, max_percent_stocks
            )
        else:
            self._get_gini_optimal_portfolio_as_dataframe(
                num_por_simulation, min_num_por_simulation, pct_change_table, self._stocks_symbols,
                max_percent_commodity, max_percent_stocks
            )

    def _get_markowitz_optimal_portfolio_as_dataframe(self, num_por_simulation: int, min_num_por_simulation: int,
                                                      pct_change_table: pd.DataFrame, stocks_symbols: list[str],
                                                      max_percent_commodity, max_percent_stocks) -> None:

        stocks_names: list[str] = []
        for symbol in stocks_symbols:
            if isinstance(symbol, int):
                stocks_names.append(str(symbol))
            else:
                stocks_names.append(symbol)

        returns_daily = pct_change_table
        returns_annual = ((1 + returns_daily.mean()) ** 254) - 1
        cov_daily = returns_daily.cov()
        cov_annual = cov_daily * 254

        # empty lists to store returns, volatility and weights of imaginary portfolios
        port_returns: list[np.ndarray] = []
        port_volatility: list[np.ndarray] = []
        sharpe_ratio: list[np.ndarray] = []
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
                [weights[i] for i in range(num_assets) if self._stock_sectors[i] == "US Commodity Indexes"]
            )
            israeli_stocks_percentage = np.sum(
                [weights[i] for i in range(num_assets) if self._stock_sectors[i] == "Israel Stocks Indexes"]
            ) + np.sum(
                [weights[i] for i in range(num_assets) if self._stock_sectors[i] == "Israel Stocks"]
            )
            us_stocks_percentage = np.sum(
                [weights[i] for i in range(num_assets) if self._stock_sectors[i] == "US Stocks Indexes"]
            ) + np.sum(
                [weights[i] for i in range(num_assets) if self._stock_sectors[i] == "US Stocks"]
            )

            if commodity_percentage > max_percent_commodity:
                single_portfolio += 1
                continue  # Skip this portfolio and generate a new one
            if (israeli_stocks_percentage + us_stocks_percentage) > max_percent_stocks:
                single_portfolio += 1
                continue  # Skip this portfolio and generate a new one

            returns: np.ndarray = np.dot(weights, returns_annual)
            volatility: np.ndarray = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
            sharpe: np.ndarray = returns / volatility
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
            portfolio[f"{symbol} Weight"] = [Weight[counter] for Weight in stock_weights]

        # make a nice dataframe of the extended dictionary
        df = pd.DataFrame(portfolio)

        # get better labels for desired arrangement of columns
        column_order = ["Returns", "Volatility", "Sharpe Ratio"] + [f"{stock} Weight" for stock in stocks_names]
        # reorder dataframe columns
        self._df = df[column_order]

    def _get_gini_optimal_portfolio_as_dataframe(self, num_por_simulation, min_num_por_simulation, pct_change_table,
                                                 stocks_symbols, max_percent_commodity, max_percent_stocks) -> None:
        stocks_names: list[str] = []
        for symbol in stocks_symbols:
            if isinstance(symbol, int):
                stocks_names.append(str(symbol))
            else:
                stocks_names.append(symbol)
        returns_daily = pct_change_table
        port_portfolio_annual: list = []
        portfolio_gini_annual: list = []
        sharpe_ratio: list = []
        stock_weights: list = []

        # set the number of combinations for imaginary portfolios
        num_assets = len(stocks_names)
        num_portfolios = num_por_simulation

        # set random seed for reproduction's sake
        np.random.seed(101)

        # Mathematical calculations ,
        # for _ in returns_daily.keys():
        # populate the empty lists with each portfolios returns,risk and weights
        single_portfolio = 0
        while single_portfolio < num_portfolios:
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            if (single_portfolio >= (num_portfolios - 1)) and (len(stock_weights) < min_num_por_simulation):
                num_portfolios *= 2
            # Calculate the percentage of stocks in the "Commodity" sector
            commodity_percentage = np.sum(
                [weights[i] for i in range(num_assets) if self._stock_sectors[i] == "US Commodity Indexes"])
            israeli_stocks_percentage = np.sum(
                [weights[i] for i in range(num_assets) if self._stock_sectors[i] == "Israel Stocks Indexes"]
            ) + np.sum(
                [weights[i] for i in range(num_assets) if self._stock_sectors[i] == "Israel Stocks"]
            )
            us_stocks_percentage = np.sum(
                [weights[i] for i in range(num_assets) if self._stock_sectors[i] == "US Stocks Indexes"]
            ) + np.sum(
                [weights[i] for i in range(num_assets) if self._stock_sectors[i] == "US Stocks"]
            )

            if commodity_percentage > max_percent_commodity:
                single_portfolio += 1
                continue  # Skip this portfolio and generate a new one
            if (israeli_stocks_percentage + us_stocks_percentage) > max_percent_stocks:
                single_portfolio += 1
                continue  # # Skip this portfolio and generate a new one

            portfolio = np.dot(returns_daily, weights)
            portfolio_return = pd.DataFrame(portfolio)
            rank = portfolio_return.rank()
            # Rank/N
            rank_divided_n = rank / len(rank)
            # 1-Rank/N
            one_sub_rank_divided_n = 1 - rank_divided_n
            # (1-Rank/N)^(V-1)
            one_sub_rank_divided_n_power_v_sub_one = one_sub_rank_divided_n ** (self._gini_v_value - 1)
            mue = portfolio_return.mean().tolist()[0]
            x_avg = one_sub_rank_divided_n_power_v_sub_one.mean().tolist()[0]
            portfolio_mue = portfolio_return - mue
            rank_sub_x_avg = one_sub_rank_divided_n_power_v_sub_one - x_avg
            portfolio_mue_mult_rank_x_avg = portfolio_mue * rank_sub_x_avg
            summary = portfolio_mue_mult_rank_x_avg.sum().tolist()[0] / (len(rank) - 1)
            gini_daily = summary * (-self._gini_v_value)
            gini_annual = gini_daily * (254 ** 0.5)
            portfolio_annual = ((1 + mue) ** 254) - 1
            sharpe = portfolio_annual / gini_annual
            sharpe_ratio.append(sharpe)
            port_portfolio_annual.append(portfolio_annual * 100)
            portfolio_gini_annual.append(gini_annual * 100)
            stock_weights.append(weights)

            single_portfolio += 1

            # a dictionary for Returns and Risk values of each portfolio
            portfolio = {'Portfolio Annual': port_portfolio_annual,
                         'Gini': portfolio_gini_annual,
                         'Sharpe Ratio': sharpe_ratio}

            # extend original dictionary to accommodate each ticker and weight in the portfolio
            for counter, symbol in enumerate(stocks_names):
                portfolio[f'{symbol} Weight'] = [Weight[counter] for Weight in stock_weights]

            # make a nice dataframe of the extended dictionary
            df = pd.DataFrame(portfolio)

            # get better labels for desired arrangement of columns
            column_order = ['Portfolio Annual', 'Gini', 'Sharpe Ratio'] + [f'{stock} Weight' for stock in stocks_names]

            # reorder dataframe columns
            self._df = df[column_order]

    def get_max_vols(self):
        if self._model_name == "Markowitz":
            max_vol = self._df["Volatility"].max()
            max_vol_portfolio = self._df.loc[self._df['Volatility'] == max_vol]
        else:
            max_vol = self._df["Gini"].max()
            max_vol_portfolio = self._df.loc[self._df['Gini'] == max_vol]
        return max_vol_portfolio
