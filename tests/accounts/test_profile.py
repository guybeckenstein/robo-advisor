import pytz
from typing import Callable

import pytest
from django.template.response import TemplateResponse
from django.test import Client
from django.urls import reverse

from accounts.models import CustomUser, InvestorUser
from accounts import views as accounts_views
from service.util import data_management
from tests import helper_methods

# Global constant variables
DASHBOARD: str = "'s Dashboard"
ATTRIBUTES: list[str] = ['Account', 'Investor', 'Portfolio', 'Capital Market Preferences']



@pytest.mark.django_db
class TestProfile:

    class TestProfileMain:
        def test_successful_get_request_as_logged_user(self, client: Client, user_factory: Callable):
            response, user = helper_methods.successful_get_request_as_logged_user(
                client, user_factory, url_name='profile_main', template_src='account/authenticated/profile_main.html',
            )
            TestProfile.generic_assertions(response, user, 'Account Details')
            phone_number: str = user.phone_number.raw_input
            helper_methods.assert_attributes_and_values(response, attributes_and_values=[
                ('Email', user.email),
                ('First name', user.first_name),
                ('Last name', user.last_name),
                ('Phone number', f'0{phone_number[4]}-{phone_number[5:8]}-{phone_number[8:]}'),
                ('Date joined', str(user.date_joined.astimezone(pytz.timezone('Asia/Jerusalem'))).split('.')[0])
            ])


        def test_redirection_get_request_as_guest(self, client: Client):
            helper_methods.redirection_get_request_as_guest(client, url_name='profile_main')

    class TestProfileAccount:
        def test_main_successful_get_request_as_logged_user(self, client: Client, user_factory: Callable):
            response, user = helper_methods.successful_get_request_as_logged_user(
                client, user_factory, url_name='profile_account',
                template_src='account/authenticated/profile_account.html',
            )
            TestProfile.generic_assertions(response, user, 'Account')
            helper_methods.assert_attributes(response, attributes=['Details', 'Password'])

        def test_main_redirection_get_request_as_guest(self, client: Client):
            helper_methods.redirection_get_request_as_guest(client, url_name='profile_account')

        def test_details_successful_get_request_as_logged_user(self, client: Client, user_factory: Callable):
            response, user = helper_methods.successful_get_request_as_logged_user(
                client,
                user_factory,
                url_name='profile_account_details',
                template_src='account/authenticated/profile_account_details.html',
            )
            TestProfile.generic_assertions(response, user, 'Account Details')
            helper_methods.assert_attributes(response, attributes=['<<', 'Password'])
            phone_number: str = user.phone_number.raw_input
            helper_methods.assert_attributes_and_values(response, attributes_and_values=[
                ('First name', user.first_name),
                ('Last name', user.last_name),
                ('Phone number', f'0{phone_number[4]}-{phone_number[5:8]}-{phone_number[8:]}'),
                ('submit-id-submit', 'Save')
            ])

        def test_details_redirection_get_request_as_guest(self, client: Client):
            helper_methods.redirection_get_request_as_guest(client, url_name='profile_account_details')

        def test_password_successful_get_request_as_logged_user(self, client: Client, user_factory: Callable):
            response, user = helper_methods.successful_get_request_as_logged_user(
                client,
                user_factory,
                url_name='profile_account_password',
                template_src='account/authenticated/profile_account_password.html',
            )
            TestProfile.generic_assertions(response, user, 'Account Password')
            helper_methods.assert_attributes(response, attributes=['Details', '<<', 'Old password', 'New password1',
                                                                   'New password2', 'Change Password'])

        def test_password_redirection_get_request_as_guest(self, client: Client):
            helper_methods.redirection_get_request_as_guest(client, url_name='profile_account_password')

    class TestProfileInvestor:
        def test_successful_get_request_as_logged_user_without_investor_user(self, client: Client,
                                                                             user_factory: Callable):
            response, user = helper_methods.successful_get_request_as_logged_user(
                client, user_factory, url_name='profile_investor',
                template_src='account/authenticated/profile_investor.html',
            )
            TestProfile.generic_assertions(response, user, 'Investor')
            assert 'Please fill the form for more information' in response.content.decode()

        def test_successful_get_request_as_logged_user_with_investor_user(self, client: Client, user_factory: Callable,
                                                                          investor_user_factory: Callable):
            user: CustomUser = user_factory()
            client.force_login(user)
            investor_user_factory(user=user)
            response = client.get(reverse('profile_investor'))
            assert response.status_code == 200
            assert 'account/authenticated/profile_investor.html' in response.templates[0].name
            TestProfile.generic_assertions(response, user, 'Investor')
            assert ('After updating the investor details, '
                    'you are required to fill the Capital Market Preferences Form, again.') in response.content.decode()
            helper_methods.assert_attributes_and_values(response, attributes_and_values=[
                ('Total investment amount', '0'),
                ('Total profit', '0'),
                ('Stocks collection number', '1'),
                ('submit-id-update', 'Update')
            ])
            assert 'Stocks symbols' in response.content.decode()
            for symbol in self.get_stocks_symbols():
                        assert symbol in response.content.decode()

        def test_post_request(self, client: Client, user_factory: Callable, investor_user_factory: Callable,
                              questionnaire_a_factory: Callable, questionnaire_b_factory: Callable):
            user: CustomUser = helper_methods.login_user(client, user_factory)
            investor_user: InvestorUser = investor_user_factory(user=user)
            questionnaire_a_factory(user=user)
            questionnaire_b_factory(user=user)

            assert investor_user.stocks_collection_number == '1'
            data: dict[str, InvestorUser, InvestorUser] = {
                'stocks_collection_number': '2',
                'investor_user_instance': investor_user,
                'instance': investor_user,
            }
            helper_methods.post_request(client, 'profile_investor', data=data, status_code=302)

        def test_redirection_get_request_as_guest(self, client: Client):
            helper_methods.redirection_get_request_as_guest(client, url_name='profile_investor')

        @staticmethod
        def get_stocks_symbols() -> list[str]:
            stocks_symbols_data: dict[list] = data_management.get_stocks_from_json_file()
            styled_stocks_symbols_data: dict[list] = dict()
            for key, value in stocks_symbols_data.items():
                styled_value: list[str] = list()
                for symbol in value[0]:
                    styled_value.append(str(symbol))
                styled_stocks_symbols_data[key] = styled_value
            return styled_stocks_symbols_data['1'][0]


    @staticmethod
    def generic_assertions(response: TemplateResponse, user: CustomUser, webpage_title: str):
        # Title
        assert webpage_title in response.content.decode()
        # Sidebar
        assert f"{user.first_name}{DASHBOARD}" in response.content.decode()
        for attribute in ATTRIBUTES:
            assert attribute in response.content.decode()
