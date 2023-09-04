import pytest
from django.test import LiveServerTestCase


@pytest.mark.django_db
class LoginFormTest(LiveServerTestCase):
    URL = 'http:localhost'
    PORT = '8000'
    pass
