from typing import Callable

import pytest
from django.template.response import TemplateResponse
from django.test import Client

from tests import helper_methods

# Global variables
ATTRIBUTES: list[str] = [
    'GuyBeckenstein', 'guybeckenstein', 'YardenAgami', 'yardet', 'YardenGazit', 'Yardengz', 'HagaiLevy', 'hagailevy'
]


@pytest.mark.django_db
class TestHomepage:
    def test_successful_get_request_as_logged_user(self, client: Client, user_factory: Callable):
        response, _ = helper_methods.successful_get_request_as_logged_user(
            client, user_factory, url_name='homepage', template_src='core/homepage.html'
        )
        helper_methods.assert_attributes(response, attributes=ATTRIBUTES)  # Our names & GitHubs

    def test_successful_get_request_as_guest(self, client: Client):
        response: TemplateResponse = helper_methods.successful_get_request_as_guest(
            client, url_name='homepage', template_src='core/homepage.html'
        )
        helper_methods.assert_attributes(response, attributes=ATTRIBUTES)  # Our names & GitHubs
