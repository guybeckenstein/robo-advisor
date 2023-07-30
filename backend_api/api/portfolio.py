#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd

from backend_api.api import sector
import datetime


class Portfolio:

    __sectors = []
    __level_of_risk = 1
    __starting_investment_amount = 0
    __last_date_of_investment = None
    __stocks_symbols = []
    __stocks_weights = []
    __closing_prices_table = []
    __pct_change_table = []
    __annual_returns = None
    __annual_volatility = None
    __annual_sharpe = None
    __selected_model = None
    __machine_learning_opt = None

    def __init__(self, level_of_risk, starting_investment_amount, stocks_symbols, sectors_data,
                 selected_model, machine_learning_opt):

        self.__level_of_risk = level_of_risk
        self.__starting_investment_amount = starting_investment_amount
        self.__stocks_symbols = stocks_symbols
        self.set_sectors(sectors_data)
        self.__selected_model = selected_model
        self.__machine_learning_opt = machine_learning_opt
        self.__last_date_of_investment = datetime.datetime.now().date()

    # GETTERS

    def get_level_of_risk(self):
        return self.__level_of_risk

    def get_investment_amount(self):
        return self.__starting_investment_amount

    def get_selected_model(self):
        return self.__selected_model

    def get_machine_learning_opt(self):
        return self.__machine_learning_opt

    def get_stocks_symbols(self):
        return self.__stocks_symbols

    def get_stocks_weights(self):
        return self.__stocks_weights

    def get_closing_prices_table(self):
        return self.__closing_prices_table

    def get_pct_change_table(self):
        return self.__pct_change_table

    def get_annual_returns(self):
        return self.__annual_returns

    def get_max_loss(self):
        return self.__annual_returns - 1.65 * self.__annual_volatility

    def get_volatility(self):
        return self.__annual_volatility

    def get_sharpe(self):
        return self.__annual_sharpe

    def get_sectors(self):
        return self.__sectors

    def get_israeli_stocks_indexes(self):
        israeli_stocks_indexes = []

        for i in range(3):
            israeli_stocks_indexes.extend(self.__sectors[i].get_stocks())

        return israeli_stocks_indexes

    def get_israeli_bonds_stocks_indexes(self):
        israeli_bonds_stocks_indexes = []

        for i in range(1, 2):
            israeli_bonds_stocks_indexes.extend(self.__sectors[i].get_stocks())

        return israeli_bonds_stocks_indexes

    def get_usa_stocks_indexes(self):
        usa_stocks_indexes = []

        for i in range(3, 6):
            usa_stocks_indexes.extend(self.__sectors[i].get_stocks())

        return usa_stocks_indexes

    def get_portfolio_stats(self):
        return self.__annual_returns, self.__annual_volatility, self.__annual_sharpe, self.get_max_loss()

    def get_sector(self, sector_name: str):
        for curr_sector in self.__sectors:
            if curr_sector.get_name() == sector_name:
                return curr_sector
        return None

    def get_sector_by_index(self, index):
        return self.__sectors[index]

    def get_sector_stocks(self, sector_name: str):
        for curr_sector in self.__sectors:
            if curr_sector.get_name() == sector_name:
                return curr_sector.get_stocks()
        return None

    def get_sector_stocks_by_index(self, index: int):
        return self.__sectors[index].get_stocks()

    def get_sector_weight(self, sector_name: str):
        for curr_sector in self.__sectors:
            if curr_sector.get_name() == sector_name:
                return curr_sector.get_weight()
        return None

    def get_sector_weight_by_index(self, index: int):
        return self.__sectors[index].get_weight()

    def get_sectors_weights(self):
        weights = []

        for curr_sector in self.__sectors:
            weights.append(curr_sector.get_weight())

        return weights

    def get_sectors_names(self):
        names = []

        for curr_sector in self.__sectors:
            names.append(curr_sector.get_name())

        return names

    def get_portfolio_data(self):
        return self.__level_of_risk, self.__starting_investment_amount, self.__stocks_symbols, self.get_sectors_names(),\
               self.get_sectors_weights(), self.__stocks_weights, self.__annual_returns, self.get_max_loss(),\
               self.__annual_volatility, self.__annual_sharpe, self.get_total_change(), self.get_monthly_change(),\
               self.get_daily_change(), self.get_selected_model(), self.get_machine_learning_opt()

    # get changes
    def get_total_change(self):
        total_change = self.get_total_value_change()  # +self.__startingInvestmentAmount)/self.__startingInvestmentAmount
        total_change = total_change * 100 - 100
        return total_change

    def get_yearly_change(self):  # TODO FIX
        self.__pct_change_table['yield_selected'].index = pd.to_datetime(self.__pct_change_table['yield_selected'].index)
        yearly_yields = self.__pct_change_table['yield_selected'].resample('Y').first()
        yearly_changes = yearly_yields.pct_change().dropna() * 100
        return yearly_changes[-1]

    def get_monthly_change(self):  # TODO FIX
        self.__pct_change_table['yield_selected'].index = pd.to_datetime(self.__pct_change_table['yield_selected'].index)
        monthly_yields = self.__pct_change_table['yield_selected'].resample('M').first()
        monthly_changes = monthly_yields.pct_change().dropna() * 100
        return monthly_changes[-1]

    def get_daily_change(self):
        return self.__pct_change_table['weighted_sum_selected'].iloc[-1]

    def get_total_value_change(self):
        return self.__pct_change_table['yield_selected'].iloc[-1]

    def get_yearly_value_change(self):
        data = self.get_yearly_pct_change_table()["yield_selected"]
        return data.iloc[-1] - data.iloc[-2]

    def get_monthly_value_change(self):
        data = self.get_monthly_pct_change_table()["yield_selected"]
        return data.iloc[-1] - data.iloc[-2]

    def get_daily_value_change(self):
        data = self.__pct_change_table()["yield_selected"]
        return data.iloc[-1] - data.iloc[-2]

    # get tables
    def get_monthly_pct_change_table(self):
        table = self.__pct_change_table.resample('M').apply(lambda x: (x[-1] / x[0] - 1) * 100)
        return table

    def get_yearly_pct_change_table(self):
        return self.__pct_change_table.resample('Y').apply(lambda x: (x[-1] / x[0] - 1) * 100)

    # SETTERS and UPDATERS
    def update_stocks_data(self, closing_prices_table, pct_change_table: pd.DataFrame, stock_weights, annual_returns,
                           annual_volatility, annual_sharpe):
        self.set_tables(closing_prices_table, pct_change_table)
        self.set_stocks_weights(stock_weights)
        self.__annual_returns = annual_returns
        self.__annual_volatility = annual_volatility
        self.__annual_sharpe = annual_sharpe
        pct_change_table["weighted_sum_selected"] = pct_change_table["weighted_sum_" + str(self.__level_of_risk)]
        pct_change_table["yield_selected"] = pct_change_table["yield_" + str(self.__level_of_risk)]

        for i in range(len(self.__stocks_symbols)):
            for j in range(len(self.__sectors)):
                if self.__stocks_symbols[i] in self.__sectors[j].get_stocks():
                    self.__sectors[j].add_weight(self.__stocks_weights[i])

    def set_sectors(self, sectors_data) -> None:
        if (len(sectors_data)) > 0:
            sectors_data = sectors_data['sectorsList']['result']
            for i in range(len(sectors_data)):
                self.set_sector(sectors_data[i]['sectorName'])
                for j in range(len(self.__stocks_symbols)):
                    if self.__stocks_symbols[j] in sectors_data[i]['stocks']:
                        self.set_sector_stock_by_index(i, self.__stocks_symbols[j])

    def update_level_of_risk(self, level_of_risk) -> None:
        self.__level_of_risk = level_of_risk

    def update_investment_amount(self, investment_amount) -> None:
        self.__starting_investment_amount = investment_amount

    def set_stocks_weights(self, stocks_weights) -> None:
        if type(stocks_weights) == list:
            self.__stocks_weights = stocks_weights
        else:
            nd_array = stocks_weights.values
            new_list = []*len(nd_array)
            for i in range(len(nd_array)):
                new_list.append(nd_array[i])
            self.__stocks_weights = new_list

    def set_sector(self, name: str) -> None:
        curr_sector = sector.Sector(name)
        self.__sectors.append(curr_sector)

    def set_sector_weight(self, name: str, weight) -> None:
        for i in range(len(self.__sectors)):
            if self.__sectors[i].get_name() == name:
                self.__sectors[i].set_weight(weight)

    def set_sector_weight_by_index(self, index: int, weight) -> None:
        self.__sectors[index].set_weight(weight)

    def set_sectors_weights(self, sectors_weights) -> None:
        for i in range(len(sectors_weights)):
            self.set_sector_weight_by_index(i, sectors_weights[i])

    def set_sector_stocks(self, name: str, stocks) -> None:
        for i in range(len(self.__sectors)):
            if self.__sectors[i].get_name() == name:
                self.__sectors[i].set_stocks(stocks)

    def set_sector_stock_by_index(self, index: int, stock) -> None:
        self.__sectors[index].add_stock(stock)

    def set_tables(self, closing_prices_table, pct_change_table) -> None:
        self.__closing_prices_table = closing_prices_table
        self.__pct_change_table = pct_change_table

    def add_stock_symbol(self, stock_symbol) -> None:
        for i in range(len(self.__sectors)):
            if stock_symbol in self.__sectors[i].get_stocks():
                self.__sectors[i].get_stocks().append(stock_symbol)
        self.__stocks_symbols.append(stock_symbol)

    def remove_stock_symbol(self, stock_symbol) -> None:
        for i in range(len(self.__sectors)):
            if stock_symbol in self.__sectors[i].get_stocks():
                self.__sectors[i].get_stocks().remove(stock_symbol)
        self.__stocks_symbols.remove(stock_symbol)

    def return_sectors_weights_according_to_stocks_weights(self, stocks_weights):
        sectors_weights = [0.0] * len(self.__sectors)
        for i in range(len(self.__sectors)):
            sectors_weights[i] = 0
            for j in range(len(self.__stocks_symbols)):
                if self.__stocks_symbols[j] in self.__sectors[i].get_stocks():
                    sectors_weights[i] += stocks_weights[j]
        return sectors_weights
