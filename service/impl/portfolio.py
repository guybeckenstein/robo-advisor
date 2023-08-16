import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from django.db.models import QuerySet
# django imports
import django
from django.conf import settings as django_settings
# Set up Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "robo_advisor_project.settings")
django.setup()
from django.db import models
from investment.models import Investment
from .sector import Sector
import datetime


class Portfolio:
    def __init__(
            self,
            stocks_symbols: List,
            sectors: List[Sector],
            risk_level: int = 1,
            total_investment_amount: int = 0,
            selected_model=None,
            is_machine_learning=None
    ):

        self._sectors: List[Sector] = sectors
        self._risk_level: int = risk_level
        self._total_investment_amount: int = total_investment_amount
        self._last_date_of_investment: datetime.date = datetime.datetime.now().date()
        self._stocks_symbols: List = stocks_symbols
        self._stocks_weights = []
        self._closing_prices_table = []
        self._pct_change_table: pd.DataFrame = pd.DataFrame
        self._selected_model: int = selected_model
        self._is_machine_learning: int = is_machine_learning
        self._annual_returns: float = 0.0
        self._annual_volatility: float = 0.0
        self._annual_sharpe: float = 0.0

    # Getters and setters
    @property
    def risk_level(self) -> int:
        return self._risk_level

    @risk_level.setter
    def risk_level(self, value) -> None:
        self._risk_level = value

    @property
    def investment_amount(self) -> int:
        return self._total_investment_amount

    @investment_amount.setter
    def investment_amount(self, value) -> None:
        self._total_investment_amount = value

    @property
    def selected_model(self) -> int:
        return self._selected_model

    @property
    def machine_learning_opt(self) -> int:
        return self._is_machine_learning

    @property
    def stocks_symbols(self) -> list:
        return self._stocks_symbols

    @property
    def stocks_weights(self) -> list:
        return self._stocks_weights

    @property
    def closing_prices_table(self) -> list:
        return self._closing_prices_table

    @property
    def pct_change_table(self) -> pd.DataFrame:
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
    def sectors(self)-> list[Sector]:
        return self._sectors

    # More methods

    def get_max_loss(self) -> float:
        return self._annual_returns - 1.65 * self._annual_volatility

    def get_portfolio_stats(self) -> Tuple[float, float, float, float, float]:
        return (self._annual_returns, self._annual_volatility, self._annual_sharpe, self.get_max_loss(),
                self.get_total_change())

    def get_sectors_weights(self) -> List[float]:
        weights = []

        for curr_sector in self._sectors:
            weights.append(curr_sector.weight)

        return weights

    def get_sectors_names(self) -> list[str]:
        names: list[str] = []

        for sector in self._sectors:
            names.append(sector.name)

        return names

    def get_portfolio_data(self) -> tuple[int, int, list[str], list[str], list[float], list[float], float, float, float,
    float, object, object, object, int, int]:
        return self._risk_level, self._total_investment_amount, self._stocks_symbols, self.get_sectors_names(), \
            self.get_sectors_weights(), self._stocks_weights, self._annual_returns, self.get_max_loss(), \
            self._annual_volatility, self._annual_sharpe, self.get_total_change(), self.get_monthly_change(), \
            self.get_daily_change(), self.selected_model, self.machine_learning_opt

    def calculate_total_profit_according_to_dates_dates(self, investments) -> float:
        profit: float = 0.0

        for i, investment in enumerate(investments):
            if type(investment) == dict:
                amount = investment["amount"]
                purchase_date = investment["date"]
                is_it_active = investment["status"]
                automatic_investment = investment["automatic_investment"]
            elif type(investment) == Investment:
                amount = investment.amount
                purchase_date = investment.formatted_date()
                is_it_active = investment.status  # status is `ACTIVE` or `INACTIVE`
                automatic_investment = investment.mode  # mode is `USER` or `ROBOT`
            else:
                raise ValueError('Invalid value for `investments`')
            if is_it_active:
                profit += amount * self.get_total_value_change(from_date=purchase_date)
            else:  # TODO: check this for the console application version
                break

        return profit

    # get changes
    def get_total_change(self):  # total  change in % (10 years)
        total_change = self.get_total_value_change()
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

    def get_total_value_change(self, from_date: datetime.date = None):
        if from_date is None:
            result = self._pct_change_table['yield_selected'].iloc[-1]
        else:
            # Convert your dates to datetime objects (assuming your from_date is a string)
            from_date = pd.to_datetime(from_date)
            pct_change_dates = pd.to_datetime(self._pct_change_table.index)
            # Calculate the absolute differences between from_date and all pct_change_dates
            date_differences = np.abs(pct_change_dates - from_date)
            # Find the index of the closest date
            closest_date_index = date_differences.argmin()
            # Use the index to get the closest date
            closest_date = pct_change_dates[closest_date_index]
            # Now you can use this closest_date to get the corresponding value
            result = (self._pct_change_table['yield_selected'].iloc[-1]
                      - self._pct_change_table['yield_selected'].iloc[closest_date_index])

        return result

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

        # self.set_sectors(self._sectors)
        for i in range(len(self._stocks_symbols)):
            for j in range(len(self._sectors)):
                if self._stocks_symbols[i] in self._sectors[j].stocks:
                    self._sectors[j].add_weight(self._stocks_weights[i])

    def set_stocks_weights(self, stocks_weights) -> None:
        if type(stocks_weights) == list:
            self._stocks_weights = stocks_weights
        else:
            nd_array = stocks_weights.values
            new_list = [] * len(nd_array)
            for i in range(len(nd_array)):
                new_list.append(nd_array[i])
            self._stocks_weights = new_list

    def set_tables(self, closing_prices_table, pct_change_table) -> None:
        self._closing_prices_table = closing_prices_table
        self._pct_change_table = pct_change_table
