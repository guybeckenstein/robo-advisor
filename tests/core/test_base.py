from typing import Callable

import pytest
from django.urls import reverse

from accounts.models import CustomUser


@pytest.mark.django_db
class TestBase:
    def test_base_get_request_as_logged_user(self, client, user_factory: Callable):
        user: CustomUser = user_factory()
        client.force_login(user)
        response = client.get(reverse('homepage'))
        assert response.status_code == 200
        assert 'core/homepage.html' in response.templates[0].name
        # Navigation Bar
        for link in ['Home', 'About', 'Profile', 'Investments', 'Capital Market Form', 'Logout']:
            assert link in response.content.decode()
        # Hello message
        for value in ['Hello', 'Test User']:
            assert value in response.content.decode()

    def test_base_get_request_as_admin(self, client, superuser_factory: Callable):
        user: CustomUser = superuser_factory()
        client.force_login(user)
        response = client.get(reverse('homepage'))
        assert response.status_code == 200
        assert 'core/homepage.html' in response.templates[0].name
        # Navigation Bar
        for link in [
            'Home', 'About', 'Profile', 'Investments', 'Capital Market Form', 'Admin', 'Administrative Tools', 'Logout'
        ]:
            assert link in response.content.decode()
        # Hello message
        for value in ['Hello', 'Test User']:
            assert value in response.content.decode()

    def test_base_get_request_as_guest(self, client):
        response = client.get(reverse('homepage'))
        assert response.status_code == 200
        assert 'core/homepage.html' in response.templates[0].name
        # Navigation Bar
        for link in ['Home', 'About', 'Sign Up', 'Login']:
            assert link in response.content.decode()
        # Hello message
        for value in ['Hello', 'guest']:
            assert value in response.content.decode()
