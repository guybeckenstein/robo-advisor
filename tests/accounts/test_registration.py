from typing import Callable

from django.urls import reverse
import pytest

from accounts.models import CustomUser


@pytest.mark.django_db
class TestRegistration:
    def test_user_successful_registration(self, client, user_factory: Callable):
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

        response = client.post(reverse('signup'), data)
        assert response.request['REQUEST_METHOD'] == 'POST'
        assert response.status_code == 200

        assert CustomUser.objects.filter(email='test@example.ac.il').exists()

    def test_user_registration_invalid_phone_number(self, client):
        # Test user registration with invalid phone number format
        data = {
            'first_name': 'test',
            'last_name': 'user',
            'phone_number': '12345',  # Invalid phone number format
            'email': 'test@example.com',
            'password': 'testpassword',
        }
        response = client.post(reverse('signup'), data)
        assert response.status_code == 200
        assert 'Enter a valid phone number' in response.content.decode()

    def test_user_registration_invalid_email(self, client):
        # Test user registration with invalid email format
        data = {
            'first_name': 'test',
            'last_name': 'user',
            'phone_number': '+97221234567',
            'email': 'invalid_email@gmail.com',
            'password': 'testpassword',
        }
        response = client.post(reverse('signup'), data)
        assert response.status_code == 200
        assert 'Enter a valid email address' in response.content.decode()