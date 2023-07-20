import pytest
from django.contrib.auth.models import User
from django.urls import reverse


@pytest.mark.django_db
class TestLogin:
    def test_login_logged_user(self, create_user_default: User, client):
        client.force_login(create_user_default)
        response = client.get(reverse('login'))
        assert response.status_code == 200
        assert 'user/login.html' in response.templates[0].name

    def test_login_guest(self, client):
        response = client.get(reverse('login'))
        assert response.status_code == 200
        assert 'user/login.html' in response.templates[0].name
