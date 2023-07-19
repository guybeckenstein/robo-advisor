import pytest
from django.urls import reverse


@pytest.mark.django_db
class TestCapitalMarketForm:
    def test_template_success(self, create_user_default, client):
        client.force_login(create_user_default)
        response = client.get(reverse('capital_market_form'))

