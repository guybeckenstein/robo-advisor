import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import datetime
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "robo_advisor_project.settings")
django.setup()
from investment.models import Investment
from service.impl.sector import Sector


@dataclass(init=True, order=False, frozen=False)
class Portfolio:
    _stocks_symbols: list[str | int] = field(default_factory=list)
    _sectors: list[Sector] = field(default_factory=list)
    _risk_level: int = field(default=1)
    _total_investment_amount: int = field(default=0)
    _stat_model_name: str = field(default='1')
    _is_machine_learning: int = field(default=0)
    _last_date_of_investment: datetime.date = field(default=datetime.datetime.now().date())
    _stocks_weights: list = field(default_factory=list)
    _closing_prices_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    _pct_change_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    _annual_returns: np.float64 = field(default=0.0)
    _annual_volatility: np.float64 = field(default=0.0)
    _annual_sharpe: np.float64 = field(default=0.0)

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
    def stat_model_name(self) -> int:
        return self._stat_model_name

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
    def sectors(self) -> list[Sector]:
        return self._sectors

    # More methods
    def get_max_loss(self) -> float:
        return self._annual_returns - 1.65 * self._annual_volatility

    def get_portfolio_stats(self) -> tuple[float, float, float, float, float]:
        return (self._annual_returns, self._annual_volatility, self._annual_sharpe, self.get_max_loss(),
                self.get_total_change())

    def get_sectors_weights(self) -> list[float]:
        weights = []

        for curr_sector in self._sectors:
            weights.append(curr_sector.weight)

        return weights

    def get_sectors_names(self) -> list[str]:
        names: list[str] = []

        for sector in self._sectors:
            names.append(sector.name)

        return names

    def get_portfolio_data(self) -> tuple[
        int, int, list[str], list[str], list[float], list[float], float, float, float, float, object, object, object,
        int, int
    ]:
        return self._risk_level, self._total_investment_amount, self._stocks_symbols, self.get_sectors_names(), \
            self.get_sectors_weights(), self._stocks_weights, self._annual_returns, self.get_max_loss(), \
            self._annual_volatility, self._annual_sharpe, self.get_total_change(), self.get_monthly_change(), \
            self.get_daily_change(), self._stat_model_name, self._is_machine_learning

    def calculate_total_profit_according_to_dates_dates(self, investments) -> float:
        profit: float = 0.0

        for i, investment in enumerate(investments):
            if type(investment) == dict:
                amount = investment["amount"]
                purchase_date = investment["date"]
                is_it_active = investment["status"]
                # automatic_investment = investment["automatic_investment"]
            elif type(investment) == Investment:
                amount = investment.amount
                purchase_date = investment.formatted_date()
                is_it_active = investment.status  # status is `ACTIVE` or `INACTIVE`
                # automatic_investment = investment.mode  # mode is `USER` or `ROBOT`
            else:
                raise ValueError('Invalid value for `investments`')
            if is_it_active:
                profit += amount * self.get_total_value_change(from_date=purchase_date)
            else:
                break

        return profit

    # get changes
    def get_total_change(self):  # total  change in % (10 years)
        total_change = self.get_total_value_change()
        total_change = total_change * 100 - 100
        return total_change

    def get_monthly_change(self):
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
            # closest_date = pct_change_dates[closest_date_index]
            # Now you can use this closest_date to get the corresponding value
            val1 = self._pct_change_table['yield_selected'].iloc[-1]
            val2 = self._pct_change_table['yield_selected'].iloc[closest_date_index]
            result = val1 - val2

        return result

    # Get tables
    # Setters and updaters
    def update_stocks_data(self, closing_prices_table: pd.DataFrame, pct_change_table: pd.DataFrame,
                           stocks_weights: list[np.float64], annual_returns: np.float64,
                           annual_volatility: np.float64, annual_sharpe: np.float64):
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
