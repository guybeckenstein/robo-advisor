import datetime
from datetime import date
from typing import Callable

import pytest
from django.test import Client
from django.urls import reverse

from accounts.models import CustomUser
from tests import helper_methods

# Global constant variables
DASHBOARD: str = "'s Investments Page"
ATTRIBUTES: list[str] = ['Investments History', 'Discover Stocks', 'Top Stocks']


@pytest.mark.django_db
class TestDiscoverStocksForm:
    def test_successful_get_request_as_logged_user(self, client: Client, user_factory: Callable):
        response, user = helper_methods.successful_get_request_as_logged_user(
            client, user_factory, url_name='discover_stocks_form', template_src='watchlist/discover_stocks.html'
        )
        helper_methods.assert_attributes(response, attributes=[
            'Discover stocks', 'Ml model', 'Symbol', 'Start Date to End Date', 'submit-id-submit', 'Submit',
            f"{user.first_name}{DASHBOARD}"
        ])

    def test_redirection_get_request_as_guest(self, client: Client):
        helper_methods.redirection_get_request_as_guest(client, 'discover_stocks_form')

    def test_post_request(self, client: Client, user_factory: Callable):
        # TODO fix test
        helper_methods.login_user(client, user_factory)
        # US Symbol & Israeli Symbol
        for symbol in ['GOOG', '1']:
            response = client.get(reverse('chosen_stock'), data={
                'ml_model': 1,
                'symbol': symbol,
                'start_date': date.today(),
                'end_date': (date.today() - datetime.timedelta(days=10*365)),
            })
            helper_methods.assert_redirection_status_code_for_get_request(response, url='')
