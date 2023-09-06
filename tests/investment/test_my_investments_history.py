from typing import Callable

import pytest
import pytz
from django.template.response import TemplateResponse
from django.test import Client
from django.urls import reverse

from accounts.models import CustomUser, InvestorUser
from investment.models import Investment
from tests import helper_methods


@pytest.mark.django_db
class TestInvestmentsMyInvestmentsHistory:
    dashboard: str = "'s Investments Page"
    attributes: list[str] = ['Investments History', 'Discover Stocks', 'Top Stocks']

    def test_successful_get_request_as_logged_user_without_investor_user(self, client: Client, user_factory: Callable):
        response, user = helper_methods.successful_get_request_as_logged_user(
            client,
            user_factory,
            url_name='my_investments_history',
            template_src='investment/my_investments_history.html'
        )
        helper_methods.assert_attributes(response, attributes=self.attributes + [
            'My Investments History', f'{user.first_name}{self.dashboard}', 'Please fill the form for more information'
        ])

    def test_successful_get_request_as_logged_user_with_investor_user(
            self, client: Client, user_factory: Callable, investor_user_factory: Callable, investment_factory: Callable
    ):
        user: CustomUser = helper_methods.login_user(client, user_factory)
        investor_user: InvestorUser = investor_user_factory(user=user)
        response: TemplateResponse = client.get(reverse('my_investments_history'))
        helper_methods.assert_successful_status_code_for_get_request(
            response, template_src='investment/my_investments_history.html'
        )
        helper_methods.assert_attributes(response, attributes=self.attributes + [
            'My Investments History', f'{user.first_name}{self.dashboard}',
            'Amount To Invest', 'Add Amount', 'Invest', 'No results!'
        ])
        # Adding investment to database
        amount: int = 10
        investment: Investment = investment_factory(investor_user=investor_user, amount=amount)
        response: TemplateResponse = client.get(reverse('add_investment'))
        helper_methods.assert_successful_status_code_for_get_request(
            response, template_src='investment/add_investment.html'
        )
        helper_methods.assert_attributes(response, attributes=[
            amount,
            ' '.join(investment.date.astimezone(pytz.timezone('Asia/Jerusalem')).strftime("%B %#d, %Y").split(' 0')),
            'Active'.upper()
        ])

    def test_post_request(self, client: Client, user_factory: Callable, investor_user_factory: Callable):
        user: CustomUser = helper_methods.login_user(client, user_factory)
        investor_user: InvestorUser = investor_user_factory(user=user)
        data: dict = {'investor_user': investor_user, 'amount': 10}
        helper_methods.post_request(
            client, url_name='add_investment', data=data, status_code=302
        )

    def test_redirection_get_request_as_guest(self, client):
        helper_methods.redirection_get_request_as_guest(client, url_name='my_investments_history')
