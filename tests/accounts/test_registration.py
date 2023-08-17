from typing import Callable

from django.template.response import TemplateResponse
from django.test import Client
import pytest

from accounts.models import CustomUser
from tests import helper_methods


@pytest.mark.django_db
class TestRegistration:
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
        # Test user registration with invalid phone number format
        data1 = {
            'first_name': 'test',
            'last_name': 'user',
            'phone_number': '12345',  # Invalid phone number format
            'email': 'test@example.com',
            'password': 'testpassword',
        }
        data2 = {
            'first_name': 'test',
            'last_name': 'user',
            'phone_number': '+97221234567',
            'email': 'invalid_email@gmail.com',
            'password': 'testpassword',
        }
        for data, error_value in [(data1, 'Enter a valid phone number'), (data2, 'Enter a valid email address')]:
            response: TemplateResponse = helper_methods.post_request(
                client, url_name='signup', data=data, status_code=200
            )
            assert error_value in response.content.decode()