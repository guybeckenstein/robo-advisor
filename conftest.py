import pytest

from django.contrib.auth.models import User
from django.utils import timezone


@pytest.fixture(scope='class')
def create_user_default() -> User:
    return User.objects.create_user(
        username='testuser',
        email='test@email.com',
        password='testPassword123!',
        first_name='Test',
        last_name='User',
        date_of_birth=timezone.now()
    )