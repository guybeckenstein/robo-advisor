from typing import Callable

import pytest
from django.template.response import TemplateResponse
from django.test import Client
from django.urls import reverse

from tests import helper_methods


@pytest.mark.django_db
class TestAdminLoginAndIndex:
    index_url_name: str = 'admin:index'
    index_template_src: str = 'admin/index.html'
    index_redirection_url: str = '/admin/login/?next=/admin/'
    login_url_name: str = 'admin:login'

    # Successful
    def test_successful_get_request_as_admin(self, client: Client, superuser_factory: Callable):
        # Webpage 1
        response, user = helper_methods.successful_get_request_as_admin(
            client, superuser_factory, url_name=self.index_url_name, template_src=self.index_template_src
        )
        self.successful_get_request_admin(response)

    def test_redirection_get_request_as_admin(self, client: Client, superuser_factory: Callable):
        helper_methods.redirection_get_request_as_admin(
            client, superuser_factory, url_name=self.login_url_name, url='/admin/'
        )

    def test_successful_get_request_as_logged_user(self, client: Client, user_factory: Callable):
        helper_methods.login_user(client, user_factory)
        self.successful_get_request_non_admin(client)

    def test_successful_get_request_as_guest(self, client: Client):
        self.successful_get_request_non_admin(client)

    def test_redirection_get_request_as_logged_user(self, client: Client, user_factory: Callable):
        helper_methods.redirection_get_request_as_logged_user(
            client, user_factory, url_name=self.index_url_name, url=self.index_redirection_url
        )

    def test_redirection_get_request_as_guest(self, client: Client):
        helper_methods.redirection_get_request_as_guest(
            client, url_name=self.index_url_name, url=self.index_redirection_url
        )

    @staticmethod
    def successful_get_request_admin(response: TemplateResponse):
        # Assert attributes
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
    def successful_get_request_non_admin(client: Client):
        response = client.get(reverse('admin:login'))
        helper_methods.assert_successful_status_code_for_get_request(response, template_src='admin/login.html')
        # Assert attributes
        helper_methods.assert_attributes_and_values(response, attributes_and_values=[
            ('RoboAdvisor', '/static/img/Logo-sm.png'),
            ('login-box-msg', 'Welcome to the RoboAdvisor admin form'),
            ('username', 'Email'),
            ('password', 'Password'),
            ('btn btn-primary btn-block', 'Log in')
        ])
