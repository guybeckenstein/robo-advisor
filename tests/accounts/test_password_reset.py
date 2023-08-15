from typing import Callable

import pytest
from django.urls import reverse

from accounts.models import CustomUser


@pytest.mark.django_db
class TestUserPasswordReset:
    def test_get_request_as_guest(self, client):
        response = client.get(reverse('account_reset_password'))
        assert response.status_code == 200
        assert 'account/password_reset.html' in response.templates[0].name
        for attribute in [
            'Password Reset', 'Forgotten your password?', 'E-mail', 'E-mail address', 'Reset My Password', "contact us"
        ]:
            assert attribute in response.content.decode()

    def test_get_request_as_logged_user(self, client, user_factory: Callable):
        user = user_factory()
        client.force_login(user)
        response = client.get(reverse('account_reset_password'))
        assert response.status_code == 302
    def test_user_successful_password_reset(self, client, user_factory: Callable):
        user: CustomUser = user_factory()
        response = client.post(reverse('account_reset_password'), data={
            'email': user.email,
        })
        assert response.status_code == 302

        # Retrieve the reset URL from the response
        reset_url = response.url
        response = client.get(reset_url)

        assert response.status_code == 200