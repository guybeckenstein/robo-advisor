import codecs
import json
import pandas as pd
from matplotlib import pyplot as plt
from .portfolio import Portfolio
from .sector import Sector


class User:

    def __init__(
            self,
            name: str = "",
            portfolio: Portfolio = Portfolio(
                stocks_symbols=[],
                sectors=[],
                risk_level=1,
                starting_investment_amount=0,
                selected_model=1,
                is_machine_learning=0
            )
    ):
        self._name: str = name
        self._myPortfolio: Portfolio = portfolio

    @property
    def name(self) -> str:
        return self._name

    @property
    def portfolio(self) -> Portfolio:
        return self._myPortfolio

    def update_portfolio(self, portfolio: Portfolio) -> None:
        # TODO: decide what to do with this method
        self._myPortfolio.set_portfolio(
            portfolio.risk_level, portfolio.investment_amount,
            portfolio.get_israeli_stocks_indexes(), portfolio.get_usa_stocks_indexes()
        )

    def plot_investment_portfolio_yield(self):
        from ..util import api_util

        portfolio = self.portfolio
        table = portfolio.pct_change_table
        annual_returns, volatility, sharpe, max_loss = portfolio.get_portfolio_stats()
        total_change = portfolio.get_total_change()
        sectors: list[Sector] = portfolio.sectors

        fig_size_x = 10
        fig_size_y = 8
        fig_size = (fig_size_x, fig_size_y)
        plt.style.use("seaborn-dark")

        plt.figure()
        plt.title("Hello, " + self.name + "! This is your yield portfolio")
        plt.ylabel("Returns %")

        stocks_str = ""
        for i in range(len(sectors)):
            name = sectors[i].name
            weight = sectors[i].weight * 100
            stocks_str += name + "(" + str("{:.2f}".format(weight)) + "%),\n "

        with pd.option_context("display.float_format", "%{:,.2f}".format):
            plt.figtext(
                0.45,
                0.15,
                "your Portfolio: \n"
                + "Total change: " + str(round(total_change, 2)) + "%\n"
                + "Annual returns: " + str(round(annual_returns, 2)) + "%\n"
                + "Annual volatility: " + str(round(volatility, 2)) + "%\n"
                + "max loss: " + str(round(max_loss, 2)) + "%\n"
                + "Annual sharpe Ratio: " + str(round(sharpe, 2)) + "\n"
                + stocks_str,
                bbox=dict(facecolor="green", alpha=0.5),
                fontsize=11,
                style="oblique",
                ha="center",
                va="center",
                fontname="Arial",
                wrap=True,
            )
        table['yield__selected_percent'] = (table["yield_selected"] - 1) * 100

        table['yield__selected_percent'].plot(figsize=fig_size, grid=True, color="green", linewidth=2, label="yield",
                                              legend=True, linestyle="dashed")
        """__, forecast_returns, __ = api_util.analyze_with_machine_learning_arima(table['yield__selected_percent'],
                                                                                table.index, closing_prices_mode=False)
        forecast_returns.plot(figsize=fig_size, grid=True, color="red", linewidth=2,
                                                         label="forecast", legend=True, linestyle="dashed")"""

        plt.subplots_adjust(bottom=0.4)
        fig1, ax1 = plt.subplots()


        return plt

    def plot_portfolio_component(self):
        portfolio = self.portfolio
        sectors_weights: list[float] = portfolio.get_sectors_weights()
        sectors_names: list[str] = portfolio.get_sectors_names()
        plt.figure()
        plt.title(f"{self.name}'s portfolio")
        plt.pie(
            x=sectors_weights,
            labels=sectors_names,
            autopct="%1.1f%%",
            shadow=True,
            startangle=140,
        )
        plt.axis("equal")
        return plt

    def plot_portfolio_component_stocks(self):
        portfolio: Portfolio = self.portfolio
        stocks_weights: list[float] = portfolio.stocks_weights
        stocks_symbols: list[str] = portfolio.stocks_symbols
        if len(stocks_weights) != len(stocks_symbols):
            raise ValueError(portfolio.stocks_symbols)
        plt.figure()
        plt.title(f"{self.name}'s portfolio")
        plt.pie(
            x=stocks_weights,
            labels=stocks_symbols,
            autopct="%1.1f%%",
            shadow=True,
            startangle=140,
        )
        plt.axis("equal")
        return plt

    @staticmethod
    def get_json_data(name):
        with codecs.open(name + ".json", "r", encoding="utf-8") as file:
            json_data = json.load(file)
        return json_data

    def update_json_file(self, json_name: str):
        json_data = self.get_json_data(json_name)

        (level_of_risk, starting_investment_amount, stocks_symbols, sectors_names, sectors_weights, stocks_weights,
         annual_returns, annual_max_loss, annual_volatility, annual_sharpe, total_change, monthly_change,
         daily_change, selected_model, machine_learning_opt) = self._myPortfolio.get_portfolio_data()

        # Create a new dictionary
        # TODO - change starting_investment_amount with something better (yarden)
        new_user_data = {
            "levelOfRisk": level_of_risk,
            "startingInvestmentAmount": starting_investment_amount,
            "stocksSymbols": stocks_symbols,
            "sectorsNames": sectors_names,
            "sectorsWeights": sectors_weights,
            "stocksWeights": stocks_weights,
            "annualReturns": annual_returns,
            "annualMaxLoss": annual_max_loss,
            "annualVolatility": annual_volatility,
            "annualSharpe": annual_sharpe,
            "totalChange": total_change,
            "monthlyChange": monthly_change,
            "dailyChange": daily_change,
            "selectedModel": selected_model,
            "machineLearningOpt": machine_learning_opt
        }
        json_data['usersList'][self._name] = [new_user_data]

        with open(json_name + ".json", 'w') as f:
            json.dump(json_data, f, indent=4)
