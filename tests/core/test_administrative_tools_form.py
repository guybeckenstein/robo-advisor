from numbers import Number
from typing import Callable

import pytest
from django.urls import reverse

from accounts.models import CustomUser
from service.util import data_management


@pytest.mark.django_db
class TestAdministrativeToolsForm:
    def test_get_request_as_logged_user(self, client, user_factory: Callable):
        user: CustomUser = user_factory()
        client.force_login(user)
        response = client.get(reverse('administrative_tools_form'))
        assert response.status_code == 404

    def test_get_request_as_admin(self, client, superuser_factory: Callable):
        user: CustomUser = superuser_factory()
        client.force_login(user)
        response = client.get(reverse('administrative_tools_form'))
        assert response.status_code == 200
        assert 'core/form.html' in response.templates[0].name
        models_data: dict[Number] = data_management.get_models_data_from_collections_file()
        # Assert attributes
        for attribute in [
            "Update Models' Data",
            'Administration Tools enable modification of some of the metadata, which affects models directly',
            'Num por simulation', models_data['num_por_simulation'],
            'Min num por simulation', models_data['min_num_por_simulation'],
            'Record percent to predict', models_data['min_num_por_simulation'],
            'Test size machine learning', models_data['test_size_machine_learning'],
            'Selected ml model for build', models_data['selected_ml_model_for_build'],
            'Gini v value', models_data['gini_v_value'],
            'Submit',
        ]:
            assert attribute in response.content.decode()

    def test_get_request_as_guest(self, client):
        administrative_tools_form_url: str = reverse('administrative_tools_form')
        response = client.get(administrative_tools_form_url)
        assert response.status_code == 302
        assert response.url == f"{reverse('account_login')}?next={administrative_tools_form_url}"
        assert len(response.templates) == 0
