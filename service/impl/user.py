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
                total_investment_amount=0,
                stat_model_name=1,
                is_machine_learning=0
            ),
            stocks_collection_number: str = "1",

    ):
        self._id: int = user_id
        self._name: str = name
        self._portfolio: Portfolio = portfolio
        self.stocks_collection_number: str = stocks_collection_number

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
        with codecs.open(name + ".json", "r", encoding="utf-8") as file:
            json_data = json.load(file)
        return json_data

    def update_json_file(self, json_name: str):
        json_data = self.get_json_data(json_name)

        (level_of_risk, total_investment_amount, stocks_symbols, sectors_names, sectors_weights, stocks_weights,
         annual_returns, annual_max_loss, annual_volatility, annual_sharpe, total_change, monthly_change,
         daily_change, stat_model_name, machine_learning_opt) = self._portfolio.get_portfolio_data()

        user_entry = json_data['usersList'].get(self._name)  # Try to get existing user entry
        if user_entry is None:
            # Create a new user entry if the user doesn't exist
            user_entry = [{
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
                "stocksCollectionNumber": self.stocks_collection_number,
                "id": self._id
            }]
            json_data['usersList'][self._name] = user_entry
        else:
            # Update existing user entry
            user_entry[0]["levelOfRisk"] = level_of_risk
            user_entry[0]["startingInvestmentAmount"] = total_investment_amount
            user_entry[0]["stocksSymbols"] = stocks_symbols
            user_entry[0]["sectorsNames"] = sectors_names
            user_entry[0]["sectorsWeights"] = sectors_weights
            user_entry[0]["stocksWeights"] = stocks_weights
            user_entry[0]["annualReturns"] = annual_returns
            user_entry[0]["annualMaxLoss"] = annual_max_loss
            user_entry[0]["annualVolatility"] = annual_volatility
            user_entry[0]["annualSharpe"] = annual_sharpe
            user_entry[0]["totalChange"] = total_change
            user_entry[0]["monthlyChange"] = monthly_change
            user_entry[0]["dailyChange"] = daily_change
            user_entry[0]["statModelName"] = stat_model_name
            user_entry[0]["machineLearningOpt"] = machine_learning_opt
            user_entry[0]["stocksCollectionNumber"] = self.stocks_collection_number
            user_entry[0]["id"] = self._id

        with open(json_name + ".json", 'w') as f:
            json.dump(json_data, f, indent=4)
