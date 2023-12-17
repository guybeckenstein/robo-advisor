from typing import Callable

import pytest
from django.test import Client
from django.template.response import TemplateResponse

from accounts.models import CustomUser, UserSession
from tests import helper_methods


@pytest.mark.django_db
class TestLoginAndLogout:
    def test_login_redirection_get_request_as_logged_user(self, client: Client, user_factory: Callable):
        helper_methods.redirection_get_request_as_logged_user(
            client, user_factory, url_name='account_login',
        )

    def test_login_successful_get_request_as_guest(self, client: Client):
        response: TemplateResponse = helper_methods.successful_get_request_as_guest(
            client,
            url_name='account_login',
            template_src='account/guest/login.html',
        )
        helper_methods.assert_attributes(response, attributes=[
            'email', 'password', 'Login', 'Forgot Password?', "Don't have an account?", 'Sign up here'
        ])

    # This test requires running Redis container!
    def test_login_invalid_credentials(self, client: Client, user_factory: Callable):
        user: CustomUser = user_factory()

        data = {'login': user.email, 'password': 'wrongpassword'}
        helper_methods.post_request(client, url_name='account_login', data=data, status_code=200)
        assert '_auth_user_id' not in client.session

    # This test requires running Redis container!
    def test_user_successful_login_and_logout(self, client: Client, user_factory: Callable):
        # Create a test user
        user: CustomUser = user_factory()

        # Test user login
        data = {
            'login': user.email,
            'password': 'django1234',
        }
        session_key = user.id
        client.session['_auth_user_id'] = str(session_key)
        client.session.save()
        helper_methods.post_request(client, url_name='account_login', data=data, status_code=302)

        # Test post login session key
        assert len(UserSession.objects.all()) == 1

        # Test user logout
        response: TemplateResponse = helper_methods.successful_get_request_as_guest(
            client, url_name='account_logout', template_src='account/authenticated/logout.html'
        )
        helper_methods.assert_attributes(response, attributes=['Logout', 'You have been logged out.', 'Login again'])
        assert len(UserSession.objects.all()) == 0


    def test_logout_exception_get_request_as_guest(self, client: Client):
        # Test user logout
        with pytest.raises(UserSession.DoesNotExist):
            helper_methods.successful_get_request_as_guest(
                client, url_name='account_logout', template_src='account/authenticated/logout.html'
            )
