import pytest
from django.contrib.auth.models import User
from django.urls import reverse


@pytest.mark.django_db
class TestHomepage:
    def test_homepage_logged_user(self, create_user_default: User, client):
        client.force_login(create_user_default)
        response = client.get(reverse('homepage'))
        assert response.status_code == 200
        assert 'core/homepage.html' in response.templates[0].name
        # Our names
        assert 'GuyBeckenstein' in response.content.decode()
        assert 'YardenAgami' in response.content.decode()
        assert 'YardenGazit' in response.content.decode()
        assert 'HagaiLevy' in response.content.decode()

    def test_homepage_guest(self, client):
        response = client.get(reverse('homepage'))
        assert response.status_code == 200
        assert 'core/homepage.html' in response.templates[0].name
        # Our names
        assert 'GuyBeckenstein' in response.content.decode()
        assert 'YardenAgami' in response.content.decode()
        assert 'YardenGazit' in response.content.decode()
        assert 'HagaiLevy' in response.content.decode()
