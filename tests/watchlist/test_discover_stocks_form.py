import datetime
from datetime import date
from typing import Callable

import pytest
from django.urls import reverse

from accounts.models import CustomUser

# Global constant variables
DASHBOARD: str = "'s Investments Page"
ATTRIBUTES: list[str] = ['Investments History', 'Discover Stocks', 'Top Stocks']


@pytest.mark.django_db
class TestDiscoverStocksForm:
    def test_get_request_as_logged_user(self, client, user_factory: Callable):
        user: CustomUser = user_factory()
        client.force_login(user)
        response = client.get(reverse('discover_stocks_form'))
        assert response.status_code == 200
        assert 'watchlist/discover_stocks.html' in response.templates[0].name
        generic_assertions(response, user, 'Discover stocks')
        for value in ['Ml model', 'Symbol', 'Start Date to End Date', 'submit-id-submit', 'Submit']:
            assert value in response.content.decode()

    def test_get_request_as_guest(self, client):
        response = client.get(reverse('discover_stocks_form'))
        assert response.status_code == 302

    def test_post_request(self, client, user_factory: Callable):
        # TODO fix test
        user: CustomUser = user_factory()
        client.force_login(user)
        # US Symbol & Israeli Symbol
        for symbol in ['GOOG', '1']:
            response = client.get(reverse('chosen_stock'), data={
                'ml_model': 1,
                'symbol': 'GOOG',
                'start_date': date.today(),
                'end_date': (date.today() - datetime.timedelta(days=10*365)),
            })
            assert response.status_code == 302


def generic_assertions(response, user: CustomUser, webpage_title: str):
    # Title
    assert webpage_title in response.content.decode()
    # Sidebar
    assert f"{user.first_name}{DASHBOARD}" in response.content.decode()
    for attribute in ATTRIBUTES:
        assert attribute in response.content.decode()
