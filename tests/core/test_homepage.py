from typing import Callable

import pytest
from django.urls import reverse

from accounts.models import CustomUser


@pytest.mark.django_db
class TestHomepage:
    def test_get_request_as_logged_user(self, client, user_factory: Callable):
        user: CustomUser = user_factory()
        client.force_login(user)
        response = client.get(reverse('homepage'))
        self.assert_attributes(response)

    def test_get_request_as_guest(self, client):
        response = client.get(reverse('homepage'))
        self.assert_attributes(response)

    @staticmethod
    def assert_attributes(response):
        assert response.status_code == 200
        assert 'core/homepage.html' in response.templates[0].name
        # Our names & GitHubs
        for name, github in [
            ('GuyBeckenstein', 'guybeckenstein'),
            ('YardenAgami', 'yardet'),
            ('YardenGazit', 'Yardengz'),
            ('HagaiLevy', 'hagailevy'),
        ]:
            assert name in response.content.decode()
            assert github in response.content.decode()
