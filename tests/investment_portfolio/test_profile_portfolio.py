from typing import Callable

import pytest
from django.urls import reverse

from accounts.models import CustomUser

# Global constant variables
DASHBOARD: str = "'s Dashboard"
ATTRIBUTES: list[str] = ['Account', 'Investor', 'Portfolio', 'Capital Market Preferences']


@pytest.mark.django_db
class TestProfilePortfolio:
    def test_get_request_as_logged_user_without_investor_user(self, client, user_factory: Callable):
        user: CustomUser = user_factory()
        client.force_login(user)
        response = client.get(reverse('profile_portfolio'))
        assert response.status_code == 200
        assert 'investment_portfolio/profile_portfolio.html' in response.templates[0].name
        self.generic_assertions(response, user, 'Portfolio')
        assert 'Please fill the form for more information' in response.content.decode()

    def test_get_request_as_logged_user_with_investor_user(self, client, user_factory: Callable,
                                                           investor_user_factory: Callable):
        user: CustomUser = user_factory()
        client.force_login(user)
        investor_user_factory(user=user)
        response = client.get(reverse('profile_portfolio'))
        assert response.status_code == 200
        assert 'investment_portfolio/profile_portfolio.html' in response.templates[0].name
        self.generic_assertions(response, user, 'Portfolio')
        for attribute in [
            'Graph', 'Image', 'Sectors Component', 'Stocks Component', 'Yield',
            f'/static/img/user/{user.id}/sectors_component.png',
            f'/static/img/user/{user.id}/stocks_component.png',
            f'/static/img/user/{user.id}/yield_graph.png',
        ]:
            assert attribute in response.content.decode()


    def test_get_request_as_guest(self, client):
        response = client.get(reverse('profile_portfolio'))
        assert response.status_code == 302

    @staticmethod
    def generic_assertions(response, user: CustomUser, webpage_title: str):
        # Title
        assert webpage_title in response.content.decode()
        # Sidebar
        assert f"{user.first_name}{DASHBOARD}" in response.content.decode()
        for attribute in ATTRIBUTES:
            assert attribute in response.content.decode()