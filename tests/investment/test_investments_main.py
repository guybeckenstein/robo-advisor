from typing import Callable

import pytest
from django.test import Client

from tests import helper_methods

# Global constant variables


@pytest.mark.django_db
class TestInvestmentsMain:
    dashboard: str = "'s Investments Page"
    attributes: list[str] = ['Investments History', 'Discover Stocks', 'Top Stocks']

    def test_successful_get_request_as_logged_user(self, client: Client, user_factory: Callable):
        response, user = helper_methods.successful_get_request_as_logged_user(
            client, user_factory, url_name='investments_main', template_src='investment/investments_main.html'
        )
        helper_methods.assert_attributes(response, attributes=[f'{user.first_name}{self.dashboard}'] + self.attributes)

    def test_redirection_get_request_as_guest(self, client: Client):
        helper_methods.redirection_get_request_as_guest(client, url_name='investments_main')
