from typing import Callable

from django.template.response import TemplateResponse
from django.test import Client
import pytest

from accounts.models import CustomUser
from tests import helper_methods


@pytest.mark.django_db
class TestRegistration:
    def test_redirection_get_request_as_logged_user(self, client: Client, user_factory: Callable):
        helper_methods.redirection_get_request_as_logged_user(
            client, user_factory, url_name='signup',
        )

    def test_successful_get_request_as_guest(self, client: Client):
        response: TemplateResponse = helper_methods.successful_get_request_as_guest(
            client,
            url_name='signup',
            template_src='account/guest/registration.html',
        )
        helper_methods.assert_attributes(response, attributes=[
            'Join Today', 'Email', 'First name', 'Last name', 'Phone number', 'Password', 'Password confirmation',
            'Enter the same password as before, for verification.', 'Submit', 'Already Have An Account?', 'Login'
        ])

    def test_successful_post_request_as_guest(self, client: Client, user_factory: Callable):
        # Test user registration
        data = {
            'first_name': 'test',
            'last_name': 'user',
            'phone_number': '+97221234567',
            'email': 'test@example.ac.il',
            'password1': 'django1234',
            'password2': 'django1234',

        }
        user: CustomUser = user_factory(**data)
        assert user is not None
        assert user.first_name == 'test'
        assert user.last_name == 'user'
        assert len(CustomUser.objects.all()) > 0
        assert CustomUser.objects.filter(email='test@example.ac.il').exists()
        helper_methods.post_request(client, url_name='signup', data=data, status_code=200)

    def test_invalid_input_post_request(self, client: Client):
        data: list[dict[str]] = [
            {
                'first_name': 'test',
                'last_name': 'user',
                'phone_number': '12345',  # Invalid phone number
                'email': 'invalid_email@gmail.com',
                'password': 'pass',
            },
            {
                'first_name': 'test',
                'last_name': 'user',
                'phone_number': '+97221234567',
                'email': 'invalid_email@gmail.com',  # Invalid email number
                'password': 'pass',
            },
            {
                'first_name': 'test',
                'last_name': 'user',
                'phone_number': '+97221234567',
                'email': 'valid@gmail.ac.il',
                'password': 'pass',  # No password confirmation
            },
            {
                'first_name': 'test',
                'last_name': 'user',
                'phone_number': '+97221234567',
                'email': 'valid@gmail.ac.il',
                'password': 'pass',   # Password too short and common
                'password2': 'pass',  # Password too short and common
            }
        ]
        for data, error_value in [(data[0], 'Enter a valid phone number'), (data[1], 'Enter a '),
                                  (data[1], 'valid email address with the domain ending'),
                                  (data[2], 'This field is required.'), (data[3], 'This password is too common.'),
                                  (data[3], 'This password is too short. It must contain at least 8 characters.')]:
            response: TemplateResponse = helper_methods.post_request(
                client, url_name='signup', data=data, status_code=200
            )
            assert error_value in response.content.decode()
