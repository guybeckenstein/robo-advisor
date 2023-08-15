from typing import Callable

import pytest
from django.urls import reverse

from accounts.models import CustomUser


@pytest.mark.django_db
class TestAbout:
    def test_get_request_as_logged_user(self, client, user_factory: Callable):
        user: CustomUser = user_factory()
        client.force_login(user)
        response = client.get(reverse('about'))
        self.assert_attributes(response)

    def test_get_request_as_guest(self, client):
        response = client.get(reverse('about'))
        self.assert_attributes(response)

    @staticmethod
    def assert_attributes(response):
        assert response.status_code == 200
        assert 'core/about.html' in response.templates[0].name
        for accordion_header in ['Our Idea?', 'Who Are We?']:
            assert accordion_header in response.content.decode()
