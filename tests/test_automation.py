import pytest
from django.test import LiveServerTestCase

URL = 'http:localhost'
PORT = '8000'


@pytest.mark.django_db
class LoginFormTest(LiveServerTestCase):
    pass

