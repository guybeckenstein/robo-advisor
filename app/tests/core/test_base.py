from typing import Callable

import pytest
from django.template.response import TemplateResponse
from django.test import Client

from tests import helper_methods


@pytest.mark.django_db
class TestBase:
    def test_successful_get_request_as_admin(self, client: Client, superuser_factory: Callable):
        response, _ = helper_methods.successful_get_request_as_admin(
            client, superuser_factory, url_name='homepage', template_src='core/homepage.html'
        )
        # Navigation Bar & Hello message
        helper_methods.assert_attributes(response, attributes=[
            'Home', 'About', 'Profile', 'Investments', 'Capital Market Form', 'Admin', 'Administrative Tools', 'Logout',
        ])

    def test_successful_get_request_as_logged_user(self, client: Client, user_factory: Callable):
        response, _ = helper_methods.successful_get_request_as_logged_user(
            client, user_factory, url_name='homepage', template_src='core/homepage.html'
        )
        # Navigation Bar & Hello message
        helper_methods.assert_attributes(response, attributes=[
            'Home', 'About', 'Profile', 'Investments', 'Capital Market Form', 'Logout',
        ])

    def test_successful_get_request_as_guest(self, client: Client):
        response: TemplateResponse = helper_methods.successful_get_request_as_guest(
            client, url_name='homepage', template_src='core/homepage.html'
        )
        # Navigation Bar & Hello message
        helper_methods.assert_attributes(response, attributes=['Home', 'About', 'Sign Up', 'Login', ])
