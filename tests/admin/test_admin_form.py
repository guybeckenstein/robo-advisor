from numbers import Number
from typing import Callable

import pytest
from django.test import Client

from service.util import data_management
from tests import helper_methods


@pytest.mark.django_db
class TestAdminForm:

    def test_successful_get_request_as_admin(self, client: Client, superuser_factory: Callable):
        response, _ = helper_methods.successful_get_request_as_admin(
            client, superuser_factory, url_name='admin:index',
            template_src='admin/index.html'
        )
        models_data: dict[Number] = data_management.get_models_data_from_collections_file()
        # Assert attributes
        helper_methods.assert_attributes(response, attributes=[
            "Update Models Data",
            'Administration Tools enable modification of some of the metadata, which affects models directly',
            'Num por simulation', models_data['num_por_simulation'],
            'Min num por simulation', models_data['min_num_por_simulation'],
            'Record percent to predict', models_data['min_num_por_simulation'],
            'Test size machine learning', models_data['test_size_machine_learning'],
            'Selected ml model for build', models_data['selected_ml_model_for_build'],
            'Gini v value', models_data['gini_v_value'],
            'Submit',
        ])

    def test_page_not_found_get_request_as_logged_user(self, client: Client, user_factory: Callable):
        helper_methods.redirection_get_request_as_logged_user(client, user_factory, url_name='admin:index')

    def test_redirection_get_request_as_guest(self, client: Client):
        helper_methods.redirection_get_request_as_guest(client, url_name='admin:index')

    def test_form_successful_post_request(self, client: Client, superuser_factory: Callable):
        helper_methods.login_user(client, user_factory=superuser_factory)
        models_data: dict[Number] = data_management.get_models_data_from_collections_file()
        helper_methods.post_request(
            client, url_name='administrative_tools_form', data=models_data, status_code=302
        )
        # helper_methods.assert_attributes(response, attributes=["Successfully updated models' data."])
