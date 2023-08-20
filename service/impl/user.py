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
                selected_model=1,
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
         daily_change, selected_model, machine_learning_opt) = self._portfolio.get_portfolio_data()

        # Create a new dictionary
        json_data['usersList'][self._name][0]["levelOfRisk"] = level_of_risk
        json_data['usersList'][self._name][0]["startingInvestmentAmount"] = total_investment_amount
        json_data['usersList'][self._name][0]["stocksSymbols"] = stocks_symbols
        json_data['usersList'][self._name][0]["sectorsNames"] = sectors_names
        json_data['usersList'][self._name][0]["sectorsWeights"] = sectors_weights
        json_data['usersList'][self._name][0]["stocksWeights"] = stocks_weights
        json_data['usersList'][self._name][0]["annualReturns"] = annual_returns
        json_data['usersList'][self._name][0]["annualMaxLoss"] = annual_max_loss
        json_data['usersList'][self._name][0]["annualVolatility"] = annual_volatility
        json_data['usersList'][self._name][0]["annualSharpe"] = annual_sharpe
        json_data['usersList'][self._name][0]["totalChange"] = total_change
        json_data['usersList'][self._name][0]["monthlyChange"] = monthly_change
        json_data['usersList'][self._name][0]["dailyChange"] = daily_change
        json_data['usersList'][self._name][0]["selectedModel"] = selected_model
        json_data['usersList'][self._name][0]["machineLearningOpt"] = machine_learning_opt
        json_data['usersList'][self._name][0]["stocksCollectionNumber"] = self.stocks_collection_number
        with open(json_name + ".json", 'w') as f:
            json.dump(json_data, f, indent=4)

