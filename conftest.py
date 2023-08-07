from typing import Callable

import pytest

from accounts.models import CustomUser
from core.models import QuestionnaireA


@pytest.fixture(scope='function')
def user_factory() -> Callable[[str, str, str, str, str], CustomUser]:
    def _create_user(**kwargs):
        user = CustomUser.objects.create(
            first_name=kwargs.get('first_name', 'test'),
            last_name=kwargs.get('last_name', 'user'),
            phone_number=kwargs.get('phone_number', '+97221234567'),
            email=kwargs.get('email', 'test@example.ac.il'),
            password=kwargs.get('password', 'django1234')
        )

        return user

    return _create_user


@pytest.fixture(scope='function')
def questionnaire_a_factory() -> Callable[[CustomUser, int, int], QuestionnaireA]:
    def _create_questionnaire_a(**kwargs) -> QuestionnaireA:
        user_preferences = QuestionnaireA.objects.create(
            user=kwargs.get('user', AttributeError),
            ml_answer=kwargs.get('ml_answer', 0),
            model_answer=kwargs.get('model_answer', 0),
        )
        return user_preferences

    return _create_questionnaire_a
