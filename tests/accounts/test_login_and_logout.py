from typing import Callable

import pytest
from django.urls import reverse

from accounts.models import CustomUser


@pytest.mark.django_db
class TestLoginAndLogout:
    def test_login_get_request_as_guest(self, client):
        response = client.get(reverse('account_login'))
        assert response.status_code == 200
        assert 'account/login.html' in response.templates[0].name
        for attribute in [
            'E-mail', 'Password', 'Remember Me', 'Login', 'Forgot Password?', "Don't have an account?", 'Sign up here'
        ]:
            assert attribute in response.content.decode()

    def test_login_get_request_as_logged_user(self, client, user_factory: Callable):
        user: CustomUser = user_factory()
        client.force_login(user)
        response = client.get(reverse('account_login'))
        assert response.status_code == 302

    def test_login_invalid_credentials(self, client, user_factory: Callable):
        user: CustomUser = user_factory()

        response = client.post(reverse('account_login'), data={
            'login': user.email,
            'password': 'wrongpassword',
        })

        assert response.status_code == 200
        assert '_auth_user_id' not in client.session

    def test_user_successful_login_and_logout(self, client, user_factory: Callable):
        # Create a test user
        user: CustomUser = user_factory()
        print(user.email)

        # Test user login
        data = {
            'login': user.email,
            'password': 'django1234',
        }
        response = client.post(reverse('account_login'), data)
        assert response.status_code == 200
        assert response.request['REQUEST_METHOD'] == 'POST'

        # Test user logout
        response = client.post(reverse('account_logout'))
        for attribute in ['Logout', 'You have been logged out.', 'Login again']:
            assert attribute in response.content.decode()

        assert response.status_code == 200
