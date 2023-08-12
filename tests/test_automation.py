from time import sleep

from django.test import LiveServerTestCase
from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver

URL = 'http:localhost'
PORT = '8000'


class LoginFormTest(LiveServerTestCase):

    # EXAMPLE
    def test_homepage(self):
        driver: WebDriver = webdriver.Chrome()
        driver.get(f'{URL}:{PORT}')

    def test_registration_fail(self):  # לא להצליח להירשם
        # TODO
        pass

    def test_registration_success_then_login_fail(self):  # להצליח להירשם ולא להצליח להתחבר
        # TODO
        pass

    def test_registration_success_then_login_success(self):  # להצליח להירשם ולהצליח להתחבר
        # TODO
        pass

    def test_logout(self):
        pass

