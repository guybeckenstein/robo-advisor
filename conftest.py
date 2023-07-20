from typing import Callable

import pytest

from django.contrib.auth.models import User


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
def create_user_non_default() -> User:
    user = User.objects.create_user(
        username='testuser',
        email='test@email.com',
        password='testPassword123!',
        first_name='Test',
        last_name='User',
    )
    return user


@pytest.fixture(scope='function')
def create_user_non_default() -> Callable[[str, str, str, str, str], User]:
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
