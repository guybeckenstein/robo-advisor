from typing import List

import numpy as np
import pandas as pd
from django.shortcuts import get_object_or_404

from service.impl.user import User
from service.impl.portfolio import Portfolio
from service.util import data_management
from service.config import settings
from service.util import helpers
from service.util.data_management import get_closing_prices_table
from core.models import QuestionnaireA
from accounts.models import InvestorUser


def save_three_user_graphs_as_png(request) -> None:
    # Receive instances from two models - QuestionnaireA, InvestorUser
    questionnaire_a: QuestionnaireA = get_object_or_404(QuestionnaireA, user=request.user)
    investor_user: InvestorUser = get_object_or_404(InvestorUser, user=request.user)
    # Create three plots - starting with metadata
    is_machine_learning: int = questionnaire_a.ml_answer
    selected_model: int = questionnaire_a.model_answer
    starting_investment_amount: int = investor_user.starting_investment_amount
    risk_level: int = investor_user.risk_level
    if type(investor_user.stocks_symbols) is list:
        stocks_symbols: List[str] = investor_user.stocks_symbols
        for idx, symbol in enumerate(stocks_symbols):
            if symbol.isnumeric():
                if type(idx) is not int:
                    raise ValueError("Invalid type for idx")
                stocks_symbols[idx] = int(stocks_symbols[idx])
        stocks_weights: List[str] = investor_user.stocks_weights
        stocks_weights: List[float] = [float(weight) for weight in stocks_weights]
    elif type(investor_user.stocks_symbols) is str:
        stocks_symbols: List[str] = investor_user.stocks_symbols[1:-1].split(',')
        for idx, symbol in enumerate(stocks_symbols):
            if symbol.isnumeric():
                stocks_symbols[idx] = int(stocks_symbols[idx])
        stocks_weights: List[str] = investor_user.stocks_weights[1:-1].split(',')
        stocks_weights: List[float] = [float(weight) for weight in stocks_weights]
    else:
        ValueError("Invalid type for stocks_symbols of investor_user")
    annual_returns = investor_user.annual_returns
    annual_volatility = investor_user.annual_volatility
    annual_sharpe = investor_user.annual_sharpe
    stocks_collection_number: str = '1' #investor_user.stocks_collection_number TODO
    closing_price_table_path = settings.BASIC_STOCK_COLLECTION_REPOSITORY_DIR + stocks_collection_number + '/'
    closing_prices_table: pd.DataFrame = get_closing_prices_table(closing_price_table_path, mode='regular')
    sectors = helpers.set_sectors(stocks_symbols=stocks_symbols, mode='regular')
    portfolio: Portfolio = Portfolio(
        stocks_symbols=stocks_symbols,
        sectors=sectors,
        risk_level=risk_level,
        starting_investment_amount=starting_investment_amount,
        selected_model=selected_model,
        is_machine_learning=is_machine_learning
    )
    pct_change_table: pd = closing_prices_table.pct_change()
    pct_change_table.dropna(inplace=True)
    weighted_sum: np.ndarray = np.dot(stocks_weights, pct_change_table.T)
    pct_change_table["weighted_sum_" + str(risk_level)] = weighted_sum
    yield_column: str = "yield_" + str(risk_level)
    pct_change_table[yield_column] = weighted_sum
    pct_change_table[yield_column] = helpers.makes_yield_column(pct_change_table[yield_column], weighted_sum)
    portfolio.update_stocks_data(
        closing_prices_table=closing_prices_table,
        pct_change_table=pct_change_table,
        stocks_weights=stocks_weights,
        annual_returns=annual_returns,
        annual_volatility=annual_volatility,
        annual_sharpe=annual_sharpe,
    )
    # Save plots
    data_management.save_user_portfolio(User(
        user_id=request.user.id,
        name=f'{request.user.first_name} {request.user.last_name}',
        portfolio=portfolio)
    )
