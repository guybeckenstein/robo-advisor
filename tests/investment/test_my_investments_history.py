from typing import Callable

import pytest
from django.urls import reverse

from accounts.models import CustomUser

# Global constant variables
DASHBOARD: str = "'s Investments Page"
ATTRIBUTES: list[str] = ['Investments History', 'Discover Stocks', 'Top Stocks']

@pytest.mark.django_db
class TestInvestmentsMyInvestmentsHistory:
    def test_get_request_as_logged_user_without_investor_user(self, client, user_factory: Callable):
        user: CustomUser = user_factory()
        client.force_login(user)
        response = client.get(reverse('my_investments_history'))
        assert response.status_code == 200
        assert 'investment/my_investments_history.html' in response.templates[0].name
        generic_assertions(response, user, 'My Investments History')
        assert 'Please fill the form for more information' in response.content.decode()

    def test_get_request_as_logged_user_with_investor_user(self, client, user_factory: Callable,
                                                           investor_user_factory: Callable):
        user: CustomUser = user_factory()
        client.force_login(user)
        investor_user_factory(user=user)
        response = client.get(reverse('my_investments_history'))
        assert response.status_code == 200
        assert 'investment/my_investments_history.html' in response.templates[0].name
        generic_assertions(response, user, 'My Investments History')
        for value in ['Amount To Invest', 'Add Amount', 'Invest', 'No results!']:
            assert value in response.content.decode()

    def test_get_request_as_guest(self, client):
        response = client.get(reverse('my_investments_history'))
        assert response.status_code == 302

def generic_assertions(response, user: CustomUser, webpage_title: str):
    # Title
    assert webpage_title in response.content.decode()
    # Sidebar
    assert f"{user.first_name}{DASHBOARD}" in response.content.decode()
    for attribute in ATTRIBUTES:
        assert attribute in response.content.decode()