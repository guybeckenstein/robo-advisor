import numpy as np
import pandas as pd
from django.shortcuts import get_object_or_404

from service.api.user import User
from service.api.portfolio import Portfolio
from service.util import manage_data
from service.util import api_util
from service.util.manage_data import get_closing_prices_table
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
    stocks_symbols: list[str] = investor_user.stocks_symbols.split(';')
    for idx, symbol in enumerate(stocks_symbols):
        if symbol.isnumeric():
            stocks_symbols[idx] = int(stocks_symbols[idx])
    stocks_weights: list[str] = investor_user.stocks_weights.split(';')
    stocks_weights: list[float] = [float(weight) for weight in stocks_weights]
    annual_returns = investor_user.annual_returns
    annual_volatility = investor_user.annual_volatility
    annual_sharpe = investor_user.annual_sharpe
    closing_prices_table: pd.DataFrame = get_closing_prices_table(mode='regular')
    sectors = api_util.set_sectors(stocks_symbols=stocks_symbols, mode='regular')
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
    pct_change_table[yield_column] = api_util.makes_yield_column(pct_change_table[yield_column], weighted_sum)
    portfolio.update_stocks_data(
        closing_prices_table=closing_prices_table,
        pct_change_table=pct_change_table,
        stocks_weights=stocks_weights,
        annual_returns=annual_returns,
        annual_volatility=annual_volatility,
        annual_sharpe=annual_sharpe,
    )
    # Save plots
    manage_data.save_user_portfolio(User(str(investor_user.user.id), portfolio))
