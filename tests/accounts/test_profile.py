import pytz
from typing import Callable

import pytest
from django.urls import reverse

from accounts.models import CustomUser, InvestorUser
from accounts import views as accounts_views

# Global constant variables
DASHBOARD: str = "'s Dashboard"
ATTRIBUTES: list[str] = ['Account', 'Investor', 'Portfolio', 'Capital Market Preferences']


@pytest.mark.django_db
class TestProfileMain:
    def test_get_request_as_logged_user(self, client, user_factory: Callable):
        user: CustomUser = user_factory()
        client.force_login(user)
        response = client.get(reverse('profile_main'))
        assert response.status_code == 200
        assert 'account/profile_main.html' in response.templates[0].name
        generic_assertions(response, user, 'Account Details')
        phone_number: str = user.phone_number.raw_input
        for attribute, value in [
            ('Email', user.email),
            ('First name', user.first_name),
            ('Last name', user.last_name),
            ('Phone number', f'0{phone_number[4]}-{phone_number[5:8]}-{phone_number[8:]}'),
            ('Date joined', str(user.date_joined.astimezone(pytz.timezone('Asia/Jerusalem'))).split('.')[0])
        ]:
            assert attribute in response.content.decode()
            assert str(value) in response.content.decode()


    def test_get_request_as_guest(self, client):
        response = client.get(reverse('profile_main'))
        assert response.status_code == 302


@pytest.mark.django_db
class TestProfileAccount:
    def test_get_request_main_as_logged_user(self, client, user_factory: Callable):
        user: CustomUser = user_factory()
        client.force_login(user)
        response = client.get(reverse('profile_account'))
        assert response.status_code == 200
        assert 'account/profile_account.html' in response.templates[0].name
        generic_assertions(response, user, 'Account')
        assert 'Details' in response.content.decode()
        assert 'Password' in response.content.decode()

    def test_get_request_main_as_guest(self, client):
        response = client.get(reverse('profile_account'))
        assert response.status_code == 302

    def test_get_request_details_as_logged_user(self, client, user_factory: Callable):
        user: CustomUser = user_factory()
        client.force_login(user)
        response = client.get(reverse('profile_account_details'))
        assert response.status_code == 200
        assert 'account/profile_account_details.html' in response.templates[0].name
        generic_assertions(response, user, 'Account Details')
        assert '<<' in response.content.decode()
        assert 'Password' in response.content.decode()
        phone_number: str = user.phone_number.raw_input
        for attribute, value in [
            ('First name', user.first_name),
            ('Last name', user.last_name),
            ('Phone number', f'0{phone_number[4]}-{phone_number[5:8]}-{phone_number[8:]}'),
            ('submit-id-submit', 'Save'),
        ]:
            assert attribute in response.content.decode()
            assert value in response.content.decode()

    def test_get_request_details_as_guest(self, client):
        response = client.get(reverse('profile_account_details'))
        assert response.status_code == 302

    def test_get_request_password_as_logged_user(self, client, user_factory: Callable):
        user: CustomUser = user_factory()
        client.force_login(user)
        response = client.get(reverse('profile_account_password'))
        assert response.status_code == 200
        assert 'account/profile_account_password.html' in response.templates[0].name
        generic_assertions(response, user, 'Account Password')
        assert 'Details' in response.content.decode()
        assert '<<' in response.content.decode()
        for value in ['Old password', 'New password1', 'New password2', 'Change Password']:
            assert value in response.content.decode()

    def test_get_request_password_as_guest(self, client):
        response = client.get(reverse('profile_account_password'))
        assert response.status_code == 302



@pytest.mark.django_db
class TestProfileInvestor:
    def test_get_request_as_logged_user_without_investor_user(self, client, user_factory: Callable):
        user: CustomUser = user_factory()
        client.force_login(user)
        response = client.get(reverse('profile_investor'))
        assert response.status_code == 200
        assert 'account/profile_investor.html' in response.templates[0].name
        generic_assertions(response, user, 'Investor')
        assert 'Please fill the form for more information' in response.content.decode()

    def test_get_request_as_logged_user_with_investor_user(self, client, user_factory: Callable,
                                                           investor_user_factory: Callable):
        user: CustomUser = user_factory()
        client.force_login(user)
        investor_user_factory(user=user)
        response = client.get(reverse('profile_investor'))
        assert response.status_code == 200
        assert 'account/profile_investor.html' in response.templates[0].name
        generic_assertions(response, user, 'Investor')
        assert ('After updating the investor details, '
                'you are required to fill the Capital Market Preferences Form, again.') in response.content.decode()
        for attribute, value in [
            ('Total investment amount', '0'),
            ('Total profit', '0'),
            ('Stocks collection number', '1'),
            ('Stocks symbols', self.get_stocks_symbols()),
            ('submit-id-update', 'Update')
        ]:
            assert attribute in response.content.decode()
            if type(value) is str:
                assert value in response.content.decode()
            else:
                for symbol in value:
                    assert symbol in response.content.decode()

    def test_post_request(
            self, client, user_factory: Callable, investor_user_factory: Callable, questionnaire_a_factory: Callable,
            questionnaire_b_factory: Callable
    ):
        user: CustomUser = user_factory()
        client.force_login(user)
        investor_user: InvestorUser = investor_user_factory(user=user)
        questionnaire_a_factory(user=user)
        questionnaire_b_factory(user=user)

        assert investor_user.stocks_collection_number == '1'
        response = client.post(reverse('profile_investor'), data={
            'stocks_collection_number': '2',
            'investor_user_instance': investor_user,
            'instance': investor_user,
        })
        assert response.status_code == 302

    def test_get_request_as_guest(self, client):
        response = client.get(reverse('profile_investor'))
        assert response.status_code == 302

    @staticmethod
    def get_stocks_symbols() -> list[str]:
        stocks_symbols_data: dict[list] = accounts_views.get_stocks_from_json_file()
        styled_stocks_symbols_data: dict[list] = dict()
        for key, value in stocks_symbols_data.items():
            styled_value: list[str] = list()
            for symbol in value[0]:
                styled_value.append(str(symbol))
            styled_stocks_symbols_data[key] = styled_value
        return styled_stocks_symbols_data['1'][0]


def generic_assertions(response, user: CustomUser, webpage_title: str):
    # Title
    assert webpage_title in response.content.decode()
    # Sidebar
    assert f"{user.first_name}{DASHBOARD}" in response.content.decode()
    for attribute in ATTRIBUTES:
        assert attribute in response.content.decode()
