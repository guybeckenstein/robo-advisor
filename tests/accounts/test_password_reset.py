from typing import Callable
import traceback
import os

import pytest
from django.template.response import TemplateResponse
from django.test import Client
from django.urls import reverse

from accounts.models import CustomUser
from tests import helper_methods

# Global const variables
RESET_LINK_URL = f'http://{os.environ.get("HOST_IP", "localhost")}:8000/accounts/password/reset/key/1-111111/'


@pytest.mark.django_db
class TestPasswordReset:
    def test_reset_successful_get_request_as_guest(self, client: Client):
        response: TemplateResponse = helper_methods.successful_get_request_as_guest(
            client, url_name='account_reset_password', template_src='account/password_reset.html'
        )
        for attribute in [
            'Password Reset', 'Forgotten your password?', 'E-mail', 'E-mail address', 'Reset My Password', "contact us"
        ]:
            assert attribute in response.content.decode()

    # This test requires running Redis container!
    def test_reset_done_user_successful_password_reset(self, client: Client, user_factory: Callable):
        """
        It is including `account_reset_password_done` URL
        """
        user: CustomUser = user_factory()
        try:
            data: dict = {'email': user.email}
            response: TemplateResponse = helper_methods.post_request(
                client, url_name='account_reset_password', data=data, status_code=302
            )

            # Retrieve the reset URL from the response
            reset_url = response.url
            response: TemplateResponse = helper_methods.successful_get_request_as_guest(
                client, url_name=reset_url, template_src=''
            )
            helper_methods.assert_attributes(response, attributes=[
                'Password Reset',
                'We have emailed you.',
                'If you have not received it, please check your spam folder, or try refreshing your mailbox.',
                "Otherwise, if you won't receive the mail in the next few minutes,",
                'contact us.'
            ])
        except ValueError:
            print(traceback.print_exc())

    def test_reset_from_key_done_successful_get_request_as_guest(self, client: Client):
        response: TemplateResponse = helper_methods.successful_get_request_as_guest(
            client,
            url_name='account_reset_password_from_key_done',
            template_src='account/password_reset_from_key_done.html'
        )
        helper_methods.assert_attributes(response, attributes=[
            'Password Reset', 'Your password has now been successfully changed.', 'Login'
        ])

    def test_all_reset_webpages_successful_get_request_logged_user(self, client: Client, user_factory: Callable):
        user: CustomUser = helper_methods.login_user(client, user_factory)
        for url_name in [
            'account_reset_password',
            'account_reset_password_done',
            'account_reset_password_from_key_done',
        ]:
            response = client.get(reverse(url_name))
            assert response.status_code == 200
            helper_methods.assert_attributes(response, attributes=[
                'Password Reset', 'Note:', f'you are already logged in as {user.email}.'
            ])

    def test_reset_password_link_successful_get_request_as_logged_user_and_guest(self, client: Client,
                                                                                 user_factory: Callable):
        url: str = RESET_LINK_URL
        template_src: str = 'account/password_reset_from_key.html'
        response1, _ = helper_methods.successful_get_request_as_logged_user(client, user_factory, url, template_src)
        response2: TemplateResponse = helper_methods.successful_get_request_as_guest(client, url, template_src)
        for response in response1, response2:
            helper_methods.assert_attributes(response, attributes=[
                'Bad Token',
                'The password reset link is invalid; possibly because it has already been used.',
                'Please try',
                'resetting password again.',
            ])
