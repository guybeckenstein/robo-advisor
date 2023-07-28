import codecs
import json
import pandas as pd
from matplotlib import pyplot as plt
from backend_api.api import portfolio


class User:

    __name = ""
    __myPortfolio = portfolio.Portfolio(1, 0, [], [], 1, 0)

    def __init__(self, name, curr_portfolio):
        self.__name = name
        self.__myPortfolio = curr_portfolio

    def get_name(self) -> str:
        return self.__name

    def get_portfolio(self) -> portfolio.Portfolio:
        return self.__myPortfolio

    def update_portfolio(self, curr_portfolio: portfolio.Portfolio) -> None:
        self.__myPortfolio.setPortfolio(
            curr_portfolio.get_level_of_risk(), curr_portfolio.get_investment_amount(),
            curr_portfolio.get_israeli_stocks_indexes(), curr_portfolio.get_usa_stocks_indexes()
        )

    def plot_investment_portfolio_yield(self):

        curr_portfolio = self.get_portfolio()
        table = curr_portfolio.get_pct_change_table()
        annual_returns, volatility, sharpe, max_loss = curr_portfolio.get_portfolio_stats()
        total_change = curr_portfolio.get_total_change()
        sectors = curr_portfolio.get_sectors()

        fig_size_x = 10
        fig_size_y = 8
        fig_size = (fig_size_x, fig_size_y)
        plt.style.use("seaborn-dark")

        plt.title("Hello, " + self.get_name() + "! This is your yield portfolio")
        plt.ylabel("Returns %")

        stocks_str = ""
        for i in range(len(sectors)):
            name = sectors[i].get_name()
            weight = sectors[i].get_weight() * 100
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

        plt.subplots_adjust(bottom=0.4)

        return plt

    def plot_portfolio_component(self):
        curr_portfolio = self.get_portfolio()
        sectors_weights = curr_portfolio.get_sectors_weights()
        labels = curr_portfolio.get_sectors_names()
        plt.title("Hello, " + self.get_name() + "! This is your portfolio")
        plt.pie(
            sectors_weights,
            labels=labels,
            autopct="%1.1f%%",
            shadow=True,
            startangle=140,
        )
        plt.axis("equal")
        return plt

    def plot_portfolio_component_stocks(self):
        curr_portfolio = self.get_portfolio()
        stocks_weights = curr_portfolio.get_stocks_weights()
        labels = curr_portfolio.get_stocks_symbols()
        plt.title("Hello, " + self.get_name() + "! This is your portfolio")
        plt.pie(
            stocks_weights,
            labels=labels,
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
         daily_change, selected_model, machine_learning_opt) = self.__myPortfolio.get_portfolio_data()

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
        json_data['usersList'][self.__name] = [new_user_data]

        with open(json_name + ".json", 'w') as f:
            json.dump(json_data, f, indent=4)
