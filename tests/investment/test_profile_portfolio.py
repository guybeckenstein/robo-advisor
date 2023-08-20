from typing import Callable

import pytest
from django.test import Client
from django.urls import reverse

from accounts.models import CustomUser
from tests import helper_methods

# Global constant variables
DASHBOARD: str = "'s Dashboard"
ATTRIBUTES: list[str] = ['Account', 'Investor', 'Portfolio', 'Capital Market Preferences']


@pytest.mark.django_db
class TestProfilePortfolio:
    def test_successful_get_request_as_logged_user_without_investor_user(self, client: Client, user_factory: Callable):
        response, user = helper_methods.successful_get_request_as_logged_user(
            client,
            user_factory,
            url_name='profile_portfolio',
            template_src='investment/profile_portfolio.html'
        )
        self.generic_assertions(response, user, 'Portfolio', ['Please fill the form for more information'])

    def test_successful_get_request_as_logged_user_with_investor_user(self, client: Client, user_factory: Callable,
                                                                      investor_user_factory: Callable):
        user: CustomUser = helper_methods.login_user(client, user_factory)
        investor_user_factory(user=user)
        response = client.get(reverse('profile_portfolio'))
        helper_methods.assert_successful_status_code_for_get_request(
            response, template_src='investment/profile_portfolio.html'
        )
        self.generic_assertions(
            response, user, 'Portfolio', more_attributes=[
                'Graph', 'Image', 'Sectors Representation', 'Stocks Representation', 'Estimated Yield',
                f'/static/img/user/{user.id}/sectors_weights_graph.png',
                f'/static/img/user/{user.id}/stocks_weights_graph.png',
                f'/static/img/user/{user.id}/estimated_yield_graph.png',
            ]
        )


    def test_redirection_get_request_as_guest(self, client: Client):
        helper_methods.redirection_get_request_as_guest(client, url_name='profile_portfolio')

    @staticmethod
    def generic_assertions(response, user: CustomUser, webpage_title: str, more_attributes: list[str]):
        # Title, Sidebar & More attributes
        helper_methods.assert_attributes(
            response,
            attributes=more_attributes + [webpage_title, f"{user.first_name}{DASHBOARD}"] + ATTRIBUTES
        )
