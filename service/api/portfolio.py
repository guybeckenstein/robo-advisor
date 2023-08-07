from typing import List, Tuple

import pandas as pd

from .sector import Sector
import datetime


class Portfolio:

    def __init__(
            self,
            stocks_symbols: List,
            sectors: List[Sector],
            risk_level: int = 1,
            starting_investment_amount: int = 0,
            selected_model=None,
            is_machine_learning=None
    ):

        self._sectors: List[Sector] = sectors
        self._risk_level: int = risk_level
        self._starting_investment_amount: int = starting_investment_amount
        self._last_date_of_investment: datetime.date = datetime.datetime.now().date()
        self._stocks_symbols: List = stocks_symbols
        self._stocks_weights = []
        self._closing_prices_table = []
        self._pct_change_table: pd.DataFrame = pd.DataFrame
        self._selected_model = selected_model
        self._is_machine_learning = is_machine_learning
        self._annual_returns = 0.0
        self._annual_volatility = 0.0
        self._annual_sharpe = 0.0

    # Getters and setters
    @property
    def risk_level(self):
        return self._risk_level

    @risk_level.setter
    def risk_level(self, value) -> None:
        self._risk_level = value

    @property
    def investment_amount(self):
        return self._starting_investment_amount

    @investment_amount.setter
    def investment_amount(self, value) -> None:
        self._starting_investment_amount = value

    @property
    def selected_model(self):
        return self._selected_model

    @property
    def machine_learning_opt(self):
        return self._is_machine_learning

    @property
    def stocks_symbols(self):
        return self._stocks_symbols

    @property
    def stocks_weights(self):
        return self._stocks_weights

    @property
    def closing_prices_table(self):
        return self._closing_prices_table

    @property
    def pct_change_table(self):
        return self._pct_change_table

    @property
    def annual_returns(self) -> float:
        return self._annual_returns

    @property
    def annual_volatility(self) -> float:
        return self._annual_volatility

    @property
    def annual_sharpe(self) -> object:
        return self._annual_sharpe

    @property
    def sectors(self):
        return self._sectors

    # More methods

    def get_max_loss(self) -> float:
        return self._annual_returns - 1.65 * self._annual_volatility

    def get_israeli_stocks_indexes(self):
        israeli_stocks_indexes = []

        for i in range(3):
            israeli_stocks_indexes.extend(self._sectors[i].stocks)

        return israeli_stocks_indexes

    def get_israeli_bonds_stocks_indexes(self):
        israeli_bonds_stocks_indexes = []

        for i in range(1, 2):
            israeli_bonds_stocks_indexes.extend(self._sectors[i].stocks)

        return israeli_bonds_stocks_indexes

    def get_usa_stocks_indexes(self):
        usa_stocks_indexes = []

        for i in range(3, 6):
            usa_stocks_indexes.extend(self._sectors[i].stocks)

        return usa_stocks_indexes

    def get_portfolio_stats(self) -> Tuple[float, float, float, float]:
        return self._annual_returns, self._annual_volatility, self._annual_sharpe, self.get_max_loss()

    def get_sector(self, sector_name: str):
        for curr_sector in self._sectors:
            if curr_sector.name == sector_name:
                return curr_sector
        return None

    def get_sector_by_index(self, index):
        return self._sectors[index]

    def get_sector_stocks(self, sector_name: str):
        for curr_sector in self._sectors:
            if curr_sector.name == sector_name:
                return curr_sector.stocks
        return None

    def get_sector_stocks_by_index(self, index: int):
        return self._sectors[index].stocks

    def get_sector_weight(self, sector_name: str):
        for curr_sector in self._sectors:
            if curr_sector.name == sector_name:
                return curr_sector.weight
        return None

    def get_sector_weight_by_index(self, index: int):
        return self._sectors[index].weight

    def get_sectors_weights(self) -> List[float]:
        weights = []

        for curr_sector in self._sectors:
            weights.append(curr_sector.weight)

        return weights

    def get_sectors_names(self):
        names = []

        for sector in self._sectors:
            names.append(sector.name)

        return names

    def get_portfolio_data(self):
        return self._risk_level, self._starting_investment_amount, self._stocks_symbols, self.get_sectors_names(),\
               self.get_sectors_weights(), self._stocks_weights, self._annual_returns, self.get_max_loss(),\
               self._annual_volatility, self._annual_sharpe, self.get_total_change(), self.get_monthly_change(),\
               self.get_daily_change(), self.selected_model, self.machine_learning_opt

    # get changes
    def get_total_change(self):
        total_change = self.get_total_value_change()  # + self.__startingInvestmentAmount)/self.__startingInvestmentAmount
        total_change = total_change * 100 - 100
        return total_change

    def get_yearly_change(self):  # TODO FIX
        self._pct_change_table['yield_selected'].index = pd.to_datetime(self._pct_change_table['yield_selected'].index)
        yearly_yields = self._pct_change_table['yield_selected'].resample('Y').first()
        yearly_changes = yearly_yields.pct_change().dropna() * 100
        return yearly_changes[-1]

    def get_monthly_change(self):  # TODO FIX
        self._pct_change_table['yield_selected'].index = pd.to_datetime(self._pct_change_table['yield_selected'].index)
        monthly_yields = self._pct_change_table['yield_selected'].resample('M').first()
        monthly_changes = monthly_yields.pct_change().dropna() * 100
        return monthly_changes[-1]

    def get_daily_change(self):
        return self._pct_change_table['weighted_sum_selected'].iloc[-1]

    def get_total_value_change(self):
        return self._pct_change_table['yield_selected'].iloc[-1]

    def get_yearly_value_change(self):
        data = self.get_yearly_pct_change_table()["yield_selected"]
        return data.iloc[-1] - data.iloc[-2]

    def get_monthly_value_change(self):
        data = self.get_monthly_pct_change_table()["yield_selected"]
        return data.iloc[-1] - data.iloc[-2]

    def get_daily_value_change(self):
        data = self._pct_change_table()["yield_selected"]
        return data.iloc[-1] - data.iloc[-2]

    # Get tables
    def get_monthly_pct_change_table(self):
        table = self._pct_change_table.resample('M').apply(lambda x: (x[-1] / x[0] - 1) * 100)
        return table

    def get_yearly_pct_change_table(self):
        return self._pct_change_table.resample('Y').apply(lambda x: (x[-1] / x[0] - 1) * 100)

    # Setters and updaters
    def update_stocks_data(self, closing_prices_table, pct_change_table: pd.DataFrame, stocks_weights, annual_returns,
                           annual_volatility, annual_sharpe):
        self.set_tables(closing_prices_table, pct_change_table)
        self.set_stocks_weights(stocks_weights)
        self._annual_returns = annual_returns
        self._annual_volatility = annual_volatility
        self._annual_sharpe = annual_sharpe
        pct_change_table["weighted_sum_selected"] = pct_change_table["weighted_sum_" + str(self._risk_level)]
        pct_change_table["yield_selected"] = pct_change_table["yield_" + str(self._risk_level)]

        for i in range(len(self._stocks_symbols)):
            for j in range(len(self._sectors)):
                if self._stocks_symbols[i] in self._sectors[j].stocks:
                    self._sectors[j].add_weight(self._stocks_weights[i])

    def set_sectors(self, sectors_data) -> None:
        if (len(sectors_data)) > 0:
            sectors_data = sectors_data['sectorsList']['result']
            for i in range(len(sectors_data)):
                self.add_user_sector_to_portfolio(sectors_data[i]['sectorName'])
                for j in range(len(self._stocks_symbols)):
                    if self._stocks_symbols[j] in sectors_data[i]['stocks']:
                        self.set_sector_stock_by_index(i, self._stocks_symbols[j])

    def set_stocks_weights(self, stocks_weights) -> None:
        if type(stocks_weights) == List:
            self._stocks_weights = stocks_weights
        else:
            nd_array = stocks_weights.values
            new_list = []*len(nd_array)
            for i in range(len(nd_array)):
                new_list.append(nd_array[i])
            self._stocks_weights = new_list

    def add_user_sector_to_portfolio(self, name: str) -> None:
        curr_sector = Sector(name)
        self._sectors.append(curr_sector)

    def set_sector_weight(self, name: str, weight) -> None:
        for i in range(len(self._sectors)):
            if self._sectors[i].name == name:
                self._sectors[i].weight = weight

    def set_sector_weight_by_index(self, index: int, weight) -> None:
        self._sectors[index].weight = weight

    def set_sectors_weights(self, sectors_weights) -> None:
        for i in range(len(sectors_weights)):
            self.set_sector_weight_by_index(i, sectors_weights[i])

    def set_sector_stocks(self, name: str, stocks) -> None:
        for i in range(len(self._sectors)):
            if self._sectors[i].name == name:
                self._sectors[i].stocks = stocks

    def set_sector_stock_by_index(self, index: int, stock) -> None:
        self._sectors[index].add_stock(stock)

    def set_tables(self, closing_prices_table, pct_change_table) -> None:
        self._closing_prices_table = closing_prices_table
        self._pct_change_table = pct_change_table

    def add_stock_symbol(self, stocks_symbols) -> None:
        for i in range(len(self._sectors)):
            if stocks_symbols in self._sectors[i].stocks:
                self._sectors[i].stocks.append(stocks_symbols)
        self._stocks_symbols.append(stocks_symbols)

    def remove_stock_symbol(self, stocks_symbols) -> None:
        for i in range(len(self._sectors)):
            if stocks_symbols in self._sectors[i].stocks:
                self._sectors[i].stocks.remove(stocks_symbols)
        self._stocks_symbols.remove(stocks_symbols)

    def return_sectors_weights_according_to_stocks_weights(self, stocks_weights):
        sectors_weights = [0.0] * len(self._sectors)
        for i in range(len(self._sectors)):
            sectors_weights[i] = 0
            for j in range(len(self._stocks_symbols)):
                if self._stocks_symbols[j] in self._sectors[i].stocks:
                    sectors_weights[i] += stocks_weights[j]
        return sectors_weights
