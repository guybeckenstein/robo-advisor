from typing import Callable
import traceback

import pytest
from django.urls import reverse

from accounts.models import CustomUser

# Global const variables
RESET_LINK_URL = 'http://localhost:8000/account/password/reset/key/1-111111/'


@pytest.mark.django_db
class TestPasswordReset:
    def test_reset_get_request_as_guest(self, client):
        response = client.get(reverse('account_reset_password'))
        assert response.status_code == 200
        assert 'account/password_reset.html' in response.templates[0].name
        for attribute in [
            'Password Reset', 'Forgotten your password?', 'E-mail', 'E-mail address', 'Reset My Password', "contact us"
        ]:
            assert attribute in response.content.decode()

    def test_reset_done_user_successful_password_reset(self, client, user_factory: Callable):
        """
        It is including `account_reset_password_done` URL
        """
        user: CustomUser = user_factory()
        try:
            response = client.post(reverse('account_reset_password'), data={'email': user.email})
            assert response.status_code == 302

            # Retrieve the reset URL from the response
            reset_url = response.url
            response = client.get(reset_url)
            assert response.status_code == 200
            for attribute in [
                'Password Reset',
                'We have emailed you.',
                'If you have not received it, please check your spam folder, or try refreshing your mailbox.',
                "Otherwise, if you won't receive the mail in the next few minutes,",
                'contact us.'
            ]:
                assert attribute in response.content.decode()
        except ValueError:
            print(traceback.print_exc())

    def test_reset_from_key_done_get_request_as_guest(self, client):
        response = client.get(reverse('account_reset_password_from_key_done'))
        assert response.status_code == 200
        assert 'account/password_reset_from_key_done.html' in response.templates[0].name
        for attribute in ['Password Reset', 'Your password has now been successfully changed.', 'Login']:
            assert attribute in response.content.decode()

    def test_all_reset_webpages_get_request_as_logged_user(self, client, user_factory: Callable):
        user: CustomUser = user_factory()
        client.force_login(user)
        for url_name in [
            'account_reset_password',
            'account_reset_password_done',
            'account_reset_password_from_key_done',
        ]:
            response = client.get(reverse(url_name))
            assert response.status_code == 200
            for attribute in ['Password Reset', 'Note:', f'you are already logged in as {user.email}.']:
                assert attribute in response.content.decode()

    def test_reset_password_link_get_request_as_logged_user(self, client, user_factory: Callable):
        user: CustomUser = user_factory()
        client.force_login(user)
        response = client.get(RESET_LINK_URL)
        assert response.status_code == 200
        for attribute in [
            'Bad Token',
            'The password reset link is invalid; possibly because it has already been used.',
            'Please try',
            'resetting password again.',
        ]:
            assert attribute in response.content.decode()

    def test_reset_password_link_get_request_as_guest(self, client):
        response = client.get(RESET_LINK_URL)
        assert response.status_code == 200
        for attribute in [
            'Bad Token',
            'The password reset link is invalid; possibly because it has already been used.',
            'Please try',
            'resetting password again.',
        ]:
            assert attribute in response.content.decode()
