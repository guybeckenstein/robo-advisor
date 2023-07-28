from typing import Callable

import pytest

from django.contrib.auth.models import User

from user.models import UserPreferencesA


@pytest.fixture(scope='function')
def create_user_default():
    return User.objects.create_user(
        username='testuser',
        email='test@email.com',
        password='testPassword123!',
        first_name='Test',
        last_name='User',
    )


@pytest.fixture(scope='function')
def create_user_non_default() -> Callable[[str, str, str, User, str], User]:
    def _user_factory(username: str, email: str, password: str, user: User, last_name: str) -> User:
        user = User.objects.create(
            username=username,
            email=email,
            password=password,
            first_name=user,
            last_name=last_name,
        )
        return user

    return _user_factory


@pytest.fixture(scope='function')
def create_user_preferences_default(user: User) -> UserPreferencesA:
    user_preferences = UserPreferencesA.objects.create(
        user=user,
        ml_answer=0,
        model_answer=0,
    )
    return user_preferences


@pytest.fixture(scope='function')
def create_user_preferences_non_default() -> Callable[[User, int, int], UserPreferencesA]:
    def _user_preferences_factory(user: User, ml_answer: int, model_answer: int) -> UserPreferencesA:
        user_preferences = UserPreferencesA.objects.create(
            user=user,
            ml_answer=ml_answer,
            model_answer=model_answer,
        )
        return user_preferences

    return _user_preferences_factory
