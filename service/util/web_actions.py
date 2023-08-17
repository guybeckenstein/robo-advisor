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
from accounts.models import InvestorUser, CustomUser


def save_three_user_graphs_as_png(user: CustomUser) -> None:
    try:
        investor_user: InvestorUser = InvestorUser.objects.get(user=user)
    except InvestorUser.DoesNotExist:
        raise ValueError('Invalid behavior! Couldn\'t find investor_user within `web_actions`')
    (annual_returns, annual_sharpe, annual_volatility, is_machine_learning, portfolio, risk_level,
     stocks_weights) = create_portfolio_instance(user, investor_user)
    closing_prices_table: pd.DataFrame = get_closing_prices_table(
        path=f'{settings.BASIC_STOCK_COLLECTION_REPOSITORY_DIR}{investor_user.stocks_collection_number}/'
    )
    pct_change_table: pd.DataFrame = closing_prices_table.pct_change()
    pct_change_table.dropna(inplace=True)
    models_data = helpers.get_collection_json_data()
    weighted_sum: np.ndarray = np.dot(stocks_weights, pct_change_table.T)
    pct_change_table[f"weighted_sum_{str(risk_level)}"] = weighted_sum
    if is_machine_learning:
        weighted_sum: pd.DataFrame = helpers.update_daily_change_with_machine_learning(
            [weighted_sum], pct_change_table.index, models_data
        )[0][0]
    offset_row, record_percent_to_predict = helpers.get_daily_change_sub_table_offset(
        models_data, pct_change_table.index
    )
    # Update the new sub-table's length (should be at most equal to the old one), then update the table itself
    pct_change_table = pct_change_table[offset_row:]  # Update length
    yield_column: str = f"yield_{str(risk_level)}"
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
        user_id=user.id, name=f'{user.first_name} {user.last_name}', portfolio=portfolio)
    )


def create_portfolio_and_get_data(answers_sum: int, stocks_collection_number: str,
                                  questionnaire_a: QuestionnaireA) -> tuple:
    # Backend
    risk_level: int = data_management.get_level_of_risk_by_score(answers_sum)
    stocks_symbols = data_management.get_stocks_symbols_from_collection(stocks_collection_number)
    tables = data_management.get_extended_data_from_db(
        stocks_symbols=stocks_symbols,
        is_machine_learning=questionnaire_a.ml_answer,
        model_option=questionnaire_a.model_answer,
        stocks_collection_number=stocks_collection_number,
    )
    portfolio = data_management.create_new_user_portfolio(
        stocks_symbols=stocks_symbols,
        investment_amount=0,
        is_machine_learning=questionnaire_a.ml_answer,
        model_option=questionnaire_a.model_answer,
        risk_level=risk_level,
        extended_data_from_db=tables,
    )
    _, _, stocks_symbols, sectors_names, sectors_weights, stocks_weights, annual_returns, annual_max_loss, \
        annual_volatility, annual_sharpe, total_change, monthly_change, daily_change, selected_model, \
        machine_learning_opt = portfolio.get_portfolio_data()
    return (annual_max_loss, annual_returns, annual_sharpe, annual_volatility, daily_change, monthly_change, risk_level,
            sectors_names, sectors_weights, stocks_symbols, stocks_weights, total_change, portfolio)



def create_portfolio_instance(user: CustomUser, investor_user: InvestorUser):
    # Receive instances from two models - QuestionnaireA, InvestorUser
    questionnaire_a: QuestionnaireA = get_object_or_404(QuestionnaireA, user=user)
    # Create three plots - starting with metadata
    is_machine_learning: int = questionnaire_a.ml_answer
    selected_model: int = questionnaire_a.model_answer
    total_investment_amount: int = investor_user.total_investment_amount
    risk_level: int = investor_user.risk_level
    stocks_symbols: List[str] = None
    stocks_weights: List[float] = None
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
    sectors = helpers.set_sectors(stocks_symbols=stocks_symbols)
    portfolio: Portfolio = Portfolio(
        stocks_symbols=stocks_symbols,
        sectors=sectors,
        risk_level=risk_level,
        total_investment_amount=total_investment_amount,
        selected_model=selected_model,
        is_machine_learning=is_machine_learning
    )
    annual_returns = investor_user.annual_returns
    annual_volatility = investor_user.annual_volatility
    annual_sharpe = investor_user.annual_sharpe
    return annual_returns, annual_sharpe, annual_volatility, is_machine_learning, portfolio, risk_level, stocks_weights
