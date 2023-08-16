from typing import Callable

import pytest
from django.db.models import QuerySet
from django.test import Client

from tests import helper_methods

from watchlist.models import TopStock

# Global constant variables
DASHBOARD: str = "'s Investments Page"


@pytest.mark.django_db
class TestTopStocks:
    def test_successful_get_request_as_logged_user(self, client: Client, user_factory: Callable):
        response, user = helper_methods.successful_get_request_as_logged_user(
            client, user_factory, url_name='top_stocks', template_src='watchlist/top_stocks.html'
        )
        helper_methods.assert_attributes(response, attributes=[
            'Top stocks', 'Investments History', 'Discover Stocks', 'Top Stocks', f"{user.first_name}{DASHBOARD}"
        ])
        top_stocks: QuerySet[TopStock] = TopStock.objects.all()
        for top_stock in top_stocks:
            assert top_stock.sector_name in response.content.decode()
            assert top_stock.img_src in response.content.decode()

    def test_redirection_get_request_as_guest(self, client: Client):
        helper_methods.redirection_get_request_as_guest(client, url_name='top_stocks')
