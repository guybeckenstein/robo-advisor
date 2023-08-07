from typing import Callable

import pytest
from django.urls import reverse

from accounts.models import CustomUser


@pytest.mark.django_db
class TestBase:
    def test_base_logged_user(self, client, user_factory: Callable):
        user: CustomUser = user_factory()
        client.force_login(user)
        response = client.get(reverse('homepage'))
        assert response.status_code == 200
        assert 'core/homepage.html' in response.templates[0].name
        # Navigation Bar
        assert 'Home' in response.content.decode()
        assert 'About' in response.content.decode()
        assert 'Profile' in response.content.decode()
        assert 'Investments' in response.content.decode()
        assert 'Capital Market Form' in response.content.decode()
        assert 'Logout' in response.content.decode()
        # Hello message
        assert 'Test User' in response.content.decode()

    def test_base_guest(self, client):
        response = client.get(reverse('homepage'))
        assert response.status_code == 200
        assert 'core/homepage.html' in response.templates[0].name
        # Navigation Bar
        assert 'Home' in response.content.decode()
        assert 'About' in response.content.decode()
        assert 'Sign Up' in response.content.decode()
        assert 'Login' in response.content.decode()
        # Hello message
        assert 'guest' in response.content.decode()
