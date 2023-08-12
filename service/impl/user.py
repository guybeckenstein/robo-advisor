import codecs
import json

from service.impl.portfolio import Portfolio


class User:

    def __init__(
            self,
            user_id: int = "",
            name: str = "",
            portfolio: Portfolio = Portfolio(
                stocks_symbols=[],
                sectors=[],
                risk_level=1,
                starting_investment_amount=0,
                selected_model=1,
                is_machine_learning=0
            ),
            stocks_collection_number = "1"
    ):
        self._id: int = user_id
        self._name: str = name
        self._myPortfolio: Portfolio = portfolio
        self.stocks_collection_number = stocks_collection_number

    @property
    def id(self) -> str:
        return str(self._id)

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
            "machineLearningOpt": machine_learning_opt,
            "stocksCollectionNumber": self.stocks_collection_number
        }
        json_data['usersList'][self._name] = [new_user_data]

        with open(json_name + ".json", 'w') as f:
            json.dump(json_data, f, indent=4)
