from service.impl.portfolio import Portfolio
from service.util import data_management
from service.config import settings
from core.models import QuestionnaireA
from accounts.models import InvestorUser, CustomUser


def save_three_user_graphs_as_png(user: CustomUser, portfolio: Portfolio = None) -> None:
    from service.impl.user import User
    import numpy as np
    import pandas as pd
    from service.util import helpers

    investor_user: InvestorUser = get_investor_user(user)

    # get portfolio data
    if portfolio is None:
        (annual_returns, annual_sharpe, annual_volatility, is_machine_learning, portfolio, risk_level,
         stocks_weights) = create_portfolio_instance(user)
    else:
        risk_level, __, __, __, __, stocks_weights, annual_returns, __, \
            annual_volatility, annual_sharpe, __, __, \
            __, __, is_machine_learning = portfolio.get_portfolio_data()

    # get the closing prices table
    closing_prices_table: pd.DataFrame = data_management.get_closing_prices_table(
        path=f'{settings.BASIC_STOCK_COLLECTION_REPOSITORY_DIR}{investor_user.stocks_collection_number}/'
    ).ffill()
    pct_change_table: pd.DataFrame = closing_prices_table.pct_change()
    pct_change_table.dropna(inplace=True)
    weighted_sum: np.ndarray = np.dot(stocks_weights, pct_change_table.T)
    pct_change_table[f"weighted_sum_{str(risk_level)}"] = weighted_sum

    # Update the new sub-table's length (should be at most equal to the old one), then update the table itself
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
    data_management.save_user_portfolio(
        User(_id=user.id, _name=f'{user.first_name} {user.last_name}', _portfolio=portfolio)
    )


def get_investor_user(user: CustomUser) -> InvestorUser:
    try:
        investor_user: InvestorUser = InvestorUser.objects.get(user=user)
    except InvestorUser.DoesNotExist:
        raise ValueError('Invalid behavior! Couldn\'t find investor_user within `web_actions`')
    return investor_user


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
        stat_model_name=settings.MODEL_NAME[questionnaire_a.model_answer],
        risk_level=risk_level,
        extended_data_from_db=tables,
    )
    risk_level, _, stocks_symbols, sectors_names, sectors_weights, stocks_weights, annual_returns, annual_max_loss, \
        annual_volatility, annual_sharpe, total_change, monthly_change, daily_change, stat_model_name, \
        machine_learning_opt = portfolio.get_portfolio_data()
    return (annual_max_loss, annual_returns, annual_sharpe, annual_volatility, daily_change, monthly_change, risk_level,
            sectors_names, sectors_weights, stocks_symbols, stocks_weights, total_change, portfolio)


def create_portfolio_instance(user: CustomUser):
    from django.shortcuts import get_object_or_404

    from core.models import QuestionnaireB

    try:
        investor_user: InvestorUser = InvestorUser.objects.get(user=user)
    except InvestorUser.DoesNotExist:
        raise ValueError('Invalid behavior! Couldn\'t find investor_user within `web_actions`')

    # Receive instances from two models - QuestionnaireA, InvestorUser
    questionnaire_a: QuestionnaireA = get_object_or_404(QuestionnaireA, user=user)
    # Receive instances from two models - QuestionnaireB, InvestorUser
    questionnaire_b: QuestionnaireB = get_object_or_404(QuestionnaireB, user=user)
    # Create three plots - starting with metadata
    is_machine_learning: int = questionnaire_a.ml_answer

    (__, annual_returns, annual_sharpe, annual_volatility, __, __, risk_level,
     __, __, __, stocks_weights, __, portfolio) \
        = create_portfolio_and_get_data(
        answers_sum=questionnaire_b.answers_sum,
        stocks_collection_number=investor_user.stocks_collection_number,
        questionnaire_a=questionnaire_a,
    )

    return annual_returns, annual_sharpe, annual_volatility, is_machine_learning, portfolio, risk_level, stocks_weights


def send_email(subject, message, recipient_list, attachment_path=None):
    from django.conf import settings as django_settings
    # email imports
    from django.core.mail import EmailMultiAlternatives
    from django.utils.html import strip_tags

    from_email = django_settings.DEFAULT_FROM_EMAIL
    # Send the email with attachment
    text_content = strip_tags(message)
    msg = EmailMultiAlternatives(subject, text_content, from_email, recipient_list)

    # Attach the image
    with open(attachment_path, 'rb') as image_file:
        msg.attach_file(image_file.name, 'image/png')

    msg.send()
