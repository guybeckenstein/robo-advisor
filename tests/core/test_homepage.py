from typing import Callable

import pytest
from django.urls import reverse

from accounts.models import CustomUser


@pytest.mark.django_db
class TestHomepage:
    def test_homepage_logged_user(self, client, user_factory: Callable):
        user: CustomUser = user_factory()
        client.force_login(user)
        response = client.get(reverse('homepage'))
        self.assert_attributes(response)

    def test_homepage_guest(self, client):
        response = client.get(reverse('homepage'))
        self.assert_attributes(response)

    @staticmethod
    def assert_attributes(response):
        assert response.status_code == 200
        assert 'core/homepage.html' in response.templates[0].name
        # Our names
        assert 'GuyBeckenstein' in response.content.decode()
        assert 'YardenAgami' in response.content.decode()
        assert 'YardenGazit' in response.content.decode()
        assert 'HagaiLevy' in response.content.decode()
        # Our GitHubs
        assert 'guybeckenstein' in response.content.decode()
        assert 'yardet' in response.content.decode()
        assert 'Yardengz' in response.content.decode()
        assert 'hagailevy' in response.content.decode()
