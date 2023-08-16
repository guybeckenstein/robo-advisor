from typing import Callable

import pytest
from django.core.exceptions import ObjectDoesNotExist
from django.template import Template, Context
from django.urls import reverse

from accounts.models import CustomUser
from core.models import QuestionnaireA, QuestionnaireB


@pytest.mark.django_db
class TestCapitalMarketAlgorithmPreferencesForm:
    def test_get_request_as_logged_user_without_questionnaire_a(self, client, user_factory: Callable):
        user: CustomUser = user_factory()
        client.force_login(user)
        response = client.get(reverse('capital_market_algorithm_preferences_form'))
        # Assert attributes
        for attribute in ['Completing the survey is', 'essential', 'for using our website and AI algorithm', 'Submit']:
            assert attribute in response.content.decode()
        self.make_assertions(response)

    def test_get_request_as_logged_user_with_questionnaire_a(self, client, user_factory: Callable,
                                                             questionnaire_a_factory: Callable):
        sign_in(client, user_factory, questionnaire_a_factory)
        response = client.get(reverse('capital_market_algorithm_preferences_form'))
        # Assert attributes
        for attribute in ['Update your capital market algorithm preferences form', 'Update']:
            assert attribute in response.content.decode()
        self.make_assertions(response)

    @staticmethod
    def make_assertions(response) -> None:
        assert response.status_code == 200
        assert 'core/form.html' in response.templates[0].name
        for attribute in [
            'Capital Market Preferences Form - Algorithms',
            'Question #1: Would you like to use machine learning algorithms for stock market investments?',
            'Question #2: Which statistic model would you like to use for stock market investments?',
            'div_id_ml_answer', 'div_id_model_answer',
        ]:
            assert attribute in response.content.decode()

    def test_get_request_as_guest(self, client):
        capital_market_preferences_form_url: str = reverse('capital_market_algorithm_preferences_form')
        response = client.get(capital_market_preferences_form_url)
        assert response.status_code == 302
        assert response.url == f"{reverse('account_login')}?next={capital_market_preferences_form_url}"
        assert len(response.templates) == 0

    def test_form_post_request_success(self, client, user_factory: Callable, questionnaire_a_factory: Callable):
        user: CustomUser = user_factory()
        client.force_login(user)
        # Testing the user form is not within the DB
        with pytest.raises(ObjectDoesNotExist):
            QuestionnaireA.objects.get(user=user)
        data = {
            'ml_answer': '1',
            'model_answer': '1',
        }
        response = client.post(reverse('capital_market_algorithm_preferences_form'), data=data)
        # Testing we are redirected and the new user form is within the DB
        assert response.status_code == 302
        assert QuestionnaireA.objects.get(user=user) is not None
        # GET request to Investment Preferences form
        response = client.get(reverse('capital_market_investment_preferences_form'))
        assert response.status_code == 200
        assert 'core/form.html' in response.templates[0].name
        questionnaire_a: QuestionnaireA = QuestionnaireA.objects.get(user=user)
        ml_answer: int = questionnaire_a.ml_answer
        model_answer: int = questionnaire_a.model_answer
        graph_img_prefix: str = f'/static/img/graphs/1/{ml_answer}{model_answer}'
        for suffix in ['distribution_graph.png', 'three_portfolios.png']:
            assert f'{graph_img_prefix}/{suffix}' in response.content.decode()


    def test_form_post_request_failure(self, client, user_factory: Callable, questionnaire_a_factory: Callable):
        user: CustomUser = user_factory()
        client.force_login(user)
        # Testing the user form is not within the DB
        with pytest.raises(ObjectDoesNotExist):
            QuestionnaireA.objects.get(user=user)
        response = client.post(reverse('capital_market_algorithm_preferences_form'), data={})
        # Checking we are in the same template
        assert response.status_code == 200
        template = Template(response.content.decode('utf-8'))
        context = Context(response.context)
        rendered_template = template.render(context)

        # Assert that the template contains the form elements
        for answer in ['ml_answer', 'model_answer']:
            assert f'div_id_{answer}' in rendered_template


@pytest.mark.django_db
class TestCapitalMarketInvestmentPreferencesForm:
    def test_get_request_as_logged_user_without_questionnaire_b(self, client, user_factory: Callable,
                                                                questionnaire_a_factory: Callable):
        user_n_questionnaire: tuple[CustomUser, QuestionnaireA] = sign_in(client, user_factory, questionnaire_a_factory)
        _, questionnaire_a = user_n_questionnaire
        ml_answer: int = questionnaire_a.ml_answer
        model_answer: int = questionnaire_a.model_answer
        response = client.get(reverse('capital_market_investment_preferences_form'))
        # Assert attributes
        for attribute in ['Completing the survey is', 'essential', 'for using our website and AI algorithm', 'Submit']:
            assert attribute in response.content.decode()
        self.make_assertions(response, graph_img_prefix=f'/static/img/graphs/1/{ml_answer}{model_answer}')

    def test_get_request_as_logged_user_with_questionnaire_b(self, client, user_factory: Callable,
                                                             questionnaire_a_factory: Callable,
                                                             questionnaire_b_factory: Callable):
        user_n_questionnaire: tuple[CustomUser, QuestionnaireA] = sign_in(client, user_factory, questionnaire_a_factory)
        user, questionnaire_a = user_n_questionnaire
        questionnaire_b_factory(user=user)
        ml_answer: int = questionnaire_a.ml_answer
        model_answer: int = questionnaire_a.model_answer
        response = client.get(reverse('capital_market_investment_preferences_form'))
        # Assert attributes
        for attribute in ['Update your capital market investment preferences form', 'Update']:
            assert attribute in response.content.decode()
        self.make_assertions(response, graph_img_prefix=f'/static/img/graphs/1/{ml_answer}{model_answer}')

    @staticmethod
    def make_assertions(response, graph_img_prefix: str) -> None:
        assert response.status_code == 200
        assert 'core/form.html' in response.templates[0].name
        for attribute in [
            'Capital Market Preferences Form - Investments',            # Attribute
            'Question #1: For how many years do you want to invest?',   # Attribute
            'Question #2: Which distribution do you prefer?',           # Attribute
            'Question #3: What is your preferable graph?',              # Attribute
            'div_id_answer_1', 'div_id_answer_2', 'div_id_answer_3',    # Attributes
            f'{graph_img_prefix}/distribution_graph.png',               # Image
            f'{graph_img_prefix}/three_portfolios.png',                 # Image
        ]:
            assert attribute in response.content.decode()

    def test_get_request_as_guest(self, client):
        capital_market_preferences_form_url: str = reverse('capital_market_investment_preferences_form')
        response = client.get(capital_market_preferences_form_url)
        assert response.status_code == 302
        assert response.url == f"{reverse('account_login')}?next={capital_market_preferences_form_url}"
        assert len(response.templates) == 0

    def test_template_user_failure(self, client, user_factory: Callable):
        """
        We expect to get 4xx response code, because there is no instance of `QuestionnaireA`
        """
        user: CustomUser = user_factory()
        client.force_login(user)
        response = client.get(reverse('capital_market_investment_preferences_form'))
        assert response.status_code == 404

    def test_form_post_request_success(self, client, user_factory: Callable, questionnaire_a_factory: Callable):
        user_n_questionnaire: tuple[CustomUser, QuestionnaireA] = sign_in(client, user_factory, questionnaire_a_factory)
        user, _ = user_n_questionnaire
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

    def test_form_post_request_failure(self, client, user_factory: Callable, questionnaire_a_factory: Callable):
        user_n_questionnaire: tuple[CustomUser, QuestionnaireA] = sign_in(client, user_factory, questionnaire_a_factory)
        user, _ = user_n_questionnaire
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


def sign_in(client, user_factory: Callable, questionnaire_a_factory: Callable) -> tuple[CustomUser, QuestionnaireA]:
    user: CustomUser = user_factory()
    client.force_login(user)
    questionnaire_a: QuestionnaireA = questionnaire_a_factory(user=user)
    return user, questionnaire_a

