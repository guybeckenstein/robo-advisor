import pytest
from django.contrib.auth.models import User
from django.urls import reverse


@pytest.mark.django_db
class TestCapitalMarketForm:
    def test_template_success(self, create_user_default: User, client):
        client.force_login(create_user_default)
        response = client.get(reverse('capital_market_form'))
        assert response.status_code == 200
        assert 'core/capital_market_form_create.html' in response.templates[0].name

    def test_template_failure(self, client):
        response = client.get(reverse('capital_market_form'))
        assert response.status_code == 302
        assert response.url == f"{reverse('login')}?next={reverse('capital_market_form')}"
        assert len(response.templates) == 0
