import os

import pytest
from django.test import LiveServerTestCase


@pytest.mark.django_db
class LoginFormTest(LiveServerTestCase):
    URL = f'http:{os.environ.get("HOST_IP", "localhost")}'
    PORT = '8000'
    pass
