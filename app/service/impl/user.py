import codecs
import json
from dataclasses import dataclass, field

from service.impl.portfolio import Portfolio


@dataclass(init=True, order=False, frozen=True)
class User:
    _id: int = field(default=-1)
    _name: str = field(default="")
    _portfolio: Portfolio = field(default_factory=Portfolio())
    _stocks_collection_number: str = field(default="1")

    @property
    def id(self) -> str:
        return str(self._id)

    @property
    def name(self) -> str:
        return self._name

    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio

    @staticmethod
    def get_json_data(name):
        with codecs.open(name + ".json", mode="r", encoding="utf-8") as file:
            json_data = json.load(file)
        return json_data

    def update_json_file(self, json_name: str):
        json_data = self.get_json_data(json_name)

        (level_of_risk, total_investment_amount, stocks_symbols, sectors_names, sectors_weights, stocks_weights,
         annual_returns, annual_max_loss, annual_volatility, annual_sharpe, total_change, monthly_change,
         daily_change, stat_model_name, machine_learning_opt) = self._portfolio.get_portfolio_data()

        user_entry = json_data['usersList'].get(self._name)  # Try to get existing user entry
        user_entry_dict: dict = {
            "levelOfRisk": level_of_risk,
            "startingInvestmentAmount": total_investment_amount,
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
            "statModelName": stat_model_name,
            "machineLearningOpt": machine_learning_opt,
            "stocksCollectionNumber": self._stocks_collection_number,
            "id": self._id
        }
        if user_entry is None:
            # Create a new user entry if the user doesn't exist
            user_entry = [user_entry_dict]
            json_data['usersList'][self._name] = user_entry
        else:
            # Update existing user entry
            user_entry[0] = user_entry_dict

        with open(json_name + ".json", 'w') as f:
            json.dump(json_data, f, indent=4)
