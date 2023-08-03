from typing import Callable

import pytest
from django.contrib.auth.models import User
from django.core.exceptions import ObjectDoesNotExist
from django.template import Template, Context
from django.urls import reverse

from core.models import QuestionnaireA, QuestionnaireB


@pytest.mark.django_db
class TestCapitalMarketForm:
    def test_template_success(self, client, create_user_default: User, questionnaire_a_factory: Callable):
        self.sign_in(client, create_user_default, questionnaire_a_factory)
        response = client.get(reverse('capital_market_investment_preferences_form'))
        assert response.status_code == 200
        assert b'/static/img/graphs/distribution_graph.png' in response.content
        assert b'/static/img/graphs/three_portfolios.png' in response.content
        assert 'core/capital_market_form_create.html' in response.templates[0].name

    def test_template_guest_failure(self, client):
        response = client.get(reverse('capital_market_investment_preferences_form'))
        assert response.status_code == 302
        assert response.url == f"{reverse('account_login')}?next={reverse('capital_market_investment_preferences_form')}"
        assert len(response.templates) == 0

    def test_template_user_failure(self, client, create_user_default: User, questionnaire_a_factory: Callable):
        self.sign_in(client, create_user_default, questionnaire_a_factory)
        extra_kwargs = {
            'mode': 'test'
        }
        response = client.get(reverse('capital_market_investment_preferences_form'), mode="test")
        assert response.status_code == 404

    def test_form_post_success(self, client, create_user_default: User, questionnaire_a_factory: Callable):
        user = create_user_default
        self.sign_in(client, user, questionnaire_a_factory)
        # Testing the user form is not within the DB
        with pytest.raises(ObjectDoesNotExist):
            QuestionnaireB.objects.get(user=user)
        data = {
            'answer_1': '3',
            'answer_2': '3',
            'answer_3': '3',
        }
        response = client.post(reverse('capital_market_investment_preferences_form'), data=data)
        # Testing we are redirected and the new user form is within the DB
        assert response.status_code == 302
        assert QuestionnaireB.objects.get(user=user) is not None

    def test_form_post_fail(self, client, create_user_default: User, questionnaire_a_factory: Callable):
        user = create_user_default
        self.sign_in(client, user, questionnaire_a_factory)
        # Testing the user form is not within the DB
        with pytest.raises(ObjectDoesNotExist):
            QuestionnaireB.objects.get(user=user)
        response = client.post(reverse('capital_market_investment_preferences_form'), data={})
        # Checking we are in the same template
        assert response.status_code == 200
        template = Template(response.content.decode('utf-8'))
        context = Context(response.context)
        rendered_template = template.render(context)

        # Assert that the template contains the form elements
        for i in range(1, 4):
            assert f'div_id_answer_{i}' in rendered_template

    @staticmethod
    def sign_in(client, create_user_default: User, questionnaire_a_factory: Callable) -> None:
        user: User = create_user_default
        client.force_login(user)
        user_preferences: QuestionnaireA = questionnaire_a_factory(user, 0, 0)
