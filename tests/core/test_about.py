from typing import Callable

import pytest
from django.test import Client

from tests import helper_methods


@pytest.mark.django_db
class TestAbout:
    def test_get_request_as_logged_user(self, client: Client, user_factory: Callable):
        response, _ = helper_methods.successful_get_request_as_logged_user(
            client, user_factory, url_name='about', template_src='core/about.html',
        )
        helper_methods.assert_attributes(response, attributes=['Our Idea?', 'Who Are We?'])
        return

    def test_get_request_as_guest(self, client: Client):
        response = helper_methods.successful_get_request_as_guest(
            client, url_name='about', template_src='core/about.html',
        )
        helper_methods.assert_attributes(response, attributes=['Our Idea?', 'Who Are We?'])
