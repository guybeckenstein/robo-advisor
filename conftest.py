from typing import Callable

import pytest

from accounts.models import CustomUser, InvestorUser
from core.models import QuestionnaireA, QuestionnaireB
from investment.models import Investment


@pytest.fixture(scope='function')
def user_factory() -> Callable[[str, str, str, str, str], CustomUser]:
    def _create_user(**kwargs):
        user: CustomUser = CustomUser.objects.create(
            first_name=kwargs.get('first_name', 'test'),
            last_name=kwargs.get('last_name', 'user'),
            phone_number=kwargs.get('phone_number', '+97221234567'),
            email=kwargs.get('email', 'test@example.ac.il'),
            password=kwargs.get('password', 'django1234'),
        )

        return user

    return _create_user


@pytest.fixture(scope='function')
def superuser_factory() -> Callable[[str, str, str, str, str], CustomUser]:
    def _create_superuser(**kwargs):
        user: CustomUser = CustomUser.objects.create_superuser(
            email=kwargs.get('email', 'test@example.ac.il'),
            password=kwargs.get('password', 'django1234'),
        )
        user.first_name = kwargs.get('first_name', 'test')
        user.last_name = kwargs.get('last_name', 'user')
        user.phone_number = kwargs.get('phone_number', '+97221234567')
        user.save()

        return user

    return _create_superuser


@pytest.fixture(scope='function')
def questionnaire_a_factory() -> Callable[[CustomUser, int, int], QuestionnaireA]:
    def _create_questionnaire_a(**kwargs) -> QuestionnaireA:
        questionnaire_a: QuestionnaireA = QuestionnaireA.objects.create(
            user=kwargs.get('user', AttributeError),
            ml_answer=kwargs.get('ml_answer', 0),
            model_answer=kwargs.get('model_answer', 0),
        )
        return questionnaire_a

    return _create_questionnaire_a


@pytest.fixture(scope='function')
def questionnaire_b_factory() -> Callable[[CustomUser, int, int, int, int], QuestionnaireB]:
    def _create_questionnaire_b(**kwargs) -> QuestionnaireB:
        questionnaire_b: QuestionnaireB = QuestionnaireB.objects.create(
            user=kwargs.get('user', AttributeError),
            answer_1=kwargs.get('answer_1', 1),
            answer_2=kwargs.get('answer_2', 1),
            answer_3=kwargs.get('answer_3', 1),
            answers_sum=kwargs.get('answers_sum', 3),
        )
        return questionnaire_b

    return _create_questionnaire_b


@pytest.fixture(scope='function')
def investor_user_factory() -> Callable[[CustomUser, int, int, int, str, list[str], list[float], list[str], list[float],
                                         float, float, float, float, float, float, float], InvestorUser]:
    def _create_investor_user(**kwargs) -> InvestorUser:
        investor_user: InvestorUser = InvestorUser.objects.create(
            user=kwargs.get('user', AttributeError),
            risk_level=kwargs.get('risk_level', 1),
            total_investment_amount=kwargs.get('total_investment_amount', 0),
            total_profit=kwargs.get('total_profit', 0),
            stocks_collection_number=kwargs.get('stocks_collection_number', '1'),
            stocks_symbols=kwargs.get('stocks_symbols', []),
            stocks_weights=kwargs.get('stocks_weights', []),
            sectors_names=kwargs.get('sectors_names', []),
            sectors_weights=kwargs.get('sectors_weights', []),
            annual_returns=kwargs.get('annual_returns', 0.0),
            annual_max_loss=kwargs.get('annual_max_loss', 0.0),
            annual_volatility=kwargs.get('annual_volatility', 0.0),
            annual_sharpe=kwargs.get('annual_sharpe', 0.0),
            total_change=kwargs.get('total_change', 0.0),
            monthly_change=kwargs.get('monthly_change', 0.0),
            daily_change=kwargs.get('daily_change', 0.0),
        )
        return investor_user

    return _create_investor_user


@pytest.fixture(scope='function')
def investment_factory() -> Callable[[InvestorUser, int], Investment]:
    def _create_investment(**kwargs) -> Investment:
        investment: Investment = Investment.objects.create(
            investor_user=kwargs.get('investor_user', AttributeError),
            amount=kwargs.get('amount', 1),
        )
        return investment

    return _create_investment
