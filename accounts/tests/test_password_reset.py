from typing import Callable

import pytest
from django.urls import reverse

from accounts.models import CustomUser


@pytest.mark.django_db
class TestUserPasswordReset:
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