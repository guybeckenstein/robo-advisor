from typing import Callable

import pytest
from django.template.response import TemplateResponse
from django.test import Client
from django.urls import reverse

from accounts.models import CustomUser
from tests import helper_methods


@pytest.mark.django_db
class TestAdminForm:
    url_name: str = 'admin:index'
    template_src: str = 'admin/index.html'

    def test_successful_get_request_as_admin(self, client: Client, superuser_factory: Callable):
        response, user = helper_methods.successful_get_request_as_admin(
            client, superuser_factory, url_name=self.url_name, template_src=self.template_src
        )
        self.successful_get_request(response)


    def test_successful_get_request_as_logged_user(self, client: Client, user_factory: Callable):
        helper_methods.redirection_get_request_as_logged_user(
            client, user_factory, url_name=self.url_name
        )

    def test_successful_get_request_as_guest(self, client: Client):
        helper_methods.redirection_get_request_as_guest(
            client, url_name=self.url_name
        )

    def test_form_successful_post_request(self, client: Client, superuser_factory: Callable):
        user: CustomUser = helper_methods.login_user(client, user_factory=superuser_factory)
        data: dict[str, str] = {
            'username': user.email,
            'password': user.password,
        }
        helper_methods.post_request(
            client, url_name=self.url_name, data=data, status_code=302
        )

        response: TemplateResponse = client.get(reverse(self.url_name))
        self.successful_get_request(response)


    def test_form_invalid_input_post_request(self, client: Client, user_factory: Callable):
        user: CustomUser = helper_methods.login_user(client, user_factory=user_factory)
        data: dict[str, str] = {
            'username': user.email,
            'password': user.password,
        }
        helper_methods.post_request(
            client, url_name=self.url_name, data=data, status_code=200
        )

        response: TemplateResponse = client.get(reverse(self.url_name))
        helper_methods.assert_attributes(response, attributes=[
            'callout callout-danger',
            'Please enter the correct email and password for a staff account.',
            'Note that both fields may be case-sensitive.',
        ])

    @staticmethod
    def successful_get_request(response: TemplateResponse):
        helper_methods.assert_attributes_and_values(response, attributes_and_values=[
            ('Authentication and Authorization', 'Groups'),
            ('Investment', 'Investment'),
            ('Accounts', 'Email addresses'),
            ('Sites', 'Sites'),
            ('Accounts', 'Investor User'),
            ('Accounts', 'Users'),
            ('Social Accounts', 'Social accounts'),
            ('Social Accounts', 'Social application tokens'),
            ('Social Accounts', 'Social applications'),
            ('Core', 'Capital Market - Algorithm Preferences'),
            ('Core', 'Capital Market - Investment Preferences'),
            ('Watchlist', 'Top Stock'),
        ])

    @staticmethod
    def redirection_get_request(response: TemplateResponse):
        # Assert attributes
        helper_methods.assert_attributes_and_values(response, attributes_and_values=[
            ('RoboAdvisor', '/static/img/Logo-sm.png'),
            ('login-box-msg', 'Welcome to the RoboAdvisor admin form'),
            ('username', 'Email'),
            ('password', 'Password'),
            ('btn btn-primary btn-block', 'Log in')
        ])
