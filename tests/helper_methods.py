from typing import Callable

from django.test import Client
from django.urls import reverse
from django.template.response import TemplateResponse

from accounts.models import CustomUser


# Public methods
def successful_get_request_as_admin(client: Client, superuser_factory: Callable, url_name: str,
                                    template_src: str) -> tuple[TemplateResponse, CustomUser]:
    user: CustomUser = superuser_factory()
    client.force_login(user)
    response: TemplateResponse = client.get(reverse(url_name))
    _assert_successful_status_code_for_get_request(response, template_src)
    return response, user


def successful_get_request_as_logged_user(client: Client, user_factory: Callable, url_name: str,
                                          template_src: str) -> tuple[TemplateResponse, CustomUser]:
    user: CustomUser = login_user(client, user_factory)
    if 'http://' not in url_name:
        url_name = reverse(url_name)
    response: TemplateResponse = client.get(url_name)
    _assert_successful_status_code_for_get_request(response, template_src)
    return response, user


def successful_get_request_as_guest(client: Client, url_name: str, template_src: str) -> TemplateResponse:
    if 'http://' not in url_name:
        url_name = reverse(url_name)
    response: TemplateResponse = client.get(url_name)
    _assert_successful_status_code_for_get_request(response, template_src)
    return response


def redirection_get_request_as_admin(client: Client, superuser_factory: Callable, url_name: str,
                                     url: str = None) -> None:
    reversed_url: str = reverse(url_name)
    login_user(client, superuser_factory)
    response = client.get(reversed_url)
    if not url:
        url = '/'
    _assert_redirection_status_code_for_get_request(response, url=url)


def redirection_get_request_as_logged_user(client: Client, user_factory: Callable, url_name: str,
                                           url: str = None) -> None:
    reversed_url: str = reverse(url_name)
    login_user(client, user_factory)
    response = client.get(reversed_url)
    if not url:
        url = '/'
    _assert_redirection_status_code_for_get_request(response, url=url)


def redirection_get_request_as_guest(client: Client, url_name: str, url: str = None) -> None:
    reversed_url: str = reverse(url_name)
    response = client.get(reversed_url)
    if url:
        _assert_redirection_status_code_for_get_request(response, url=url)
    else:
        _assert_redirection_status_code_for_get_request(response, url=f"{reverse('account_login')}?next={reversed_url}")


def page_not_found_get_request_as_logged_user(client: Client, user_factory: Callable, url_name: str) -> None:
    login_user(client, user_factory)
    response = client.get(reverse(url_name))
    assert response.status_code == 404


def post_request(client: Client, url_name: str, data: dict, status_code: int) -> TemplateResponse:
    response: TemplateResponse = client.post(reverse(url_name), data=data)
    assert response.status_code == status_code
    assert response.request['REQUEST_METHOD'] == 'POST'
    return response


def login_user(client: Client, user_factory: Callable) -> CustomUser:
    user: CustomUser = user_factory()
    client.force_login(user)
    return user


def assert_attributes(response: TemplateResponse, attributes: list[int | str]) -> None:
    content: str = response.content.decode()
    for attribute in attributes:
        assert str(attribute) in content


def assert_attributes_and_values(response: TemplateResponse, attributes_and_values: list[tuple[str, object]]) -> None:
    content: str = response.content.decode()
    for attribute, value in attributes_and_values:
        assert attribute in content
        assert str(value) in content


def assert_successful_status_code_for_get_request(response: TemplateResponse, template_src: str) -> None:
    _assert_successful_status_code_for_get_request(response, template_src)


def assert_redirection_status_code_for_get_request(response: TemplateResponse, url: str) -> None:
    _assert_redirection_status_code_for_get_request(response, url)


# Private methods
def _assert_successful_status_code_for_get_request(response: TemplateResponse, template_src: str) -> None:
    assert response.status_code == 200
    if template_src != '':
        assert template_src in response.templates[0].name


def _assert_redirection_status_code_for_get_request(response: TemplateResponse, url: str) -> None:
    assert response.status_code == 302
    assert response.url == url
    assert len(response.templates) == 0
