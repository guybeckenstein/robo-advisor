from crispy_forms.bootstrap import InlineRadios
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit, Layout, Div, HTML
from django import forms
from django.utils.html import format_html

from backend_api.util import manage_data, settings
from core.models import QuestionnaireA, QuestionnaireB
from django.urls import reverse_lazy

from robo_advisor_project import settings


class AlgorithmPreferencesForm(forms.ModelForm):
    ml_answer = forms.ChoiceField(
        choices=((0, 'No'), (1, 'Yes')),
        widget=forms.RadioSelect(),
    )
    model_answer = forms.ChoiceField(
        choices=((0, 'Markowitz'), (1, 'Gini')),
        widget=forms.RadioSelect(),
    )

    def __init__(self, *args, **kwargs):
        form_type = kwargs.pop('form_type', 'create')
        super().__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.form_id = 'preferences-form'
        self.helper.attrs = {
            'hx-post': reverse_lazy('capital_market_algorithm_preferences_form'),
            'hx-target': '#preferences-form',
            'hx-swap': 'outerHTML'
        }
        # TODO: make dynamic code that updates CSV files from `/backend_api/DB/...`
        self.fields['ml_answer'].label = format_html('<span class="capital-market-form-label">Question #1: Would you like to use machine learning algorithms for stock market investments?</span>')
        self.fields['model_answer'].label = format_html('<span class="capital-market-form-label">Question #2: Which statistic model would you like to use for stock market investments?')

        if form_type == 'create':
            self.helper.layout = Layout(
                HTML('<h1 style="font-weight: 900;">Preferences Form</h1>'),
                HTML('<h5 style="color: rgb(150, 150, 150);">Completing the survey is <u>essential</u>'
                     ' for using our website and AI algorithm</h5>'),
                Div(InlineRadios('ml_answer', css_class='capital-market-form-radio')),
                Div(InlineRadios('model_answer', css_class='capital-market-form-radio')),
            )
            self.helper.add_input(Submit('submit', 'Submit', css_class='btn-dark'))
        elif form_type == 'update':
            self.helper.layout = Layout(
                HTML('<h1 style="font-weight: 900;">Preferences Form</h1>'),
                HTML('<h5 style="color: rgb(150, 150, 150);">Update your capital market investments preferences form</h5>'),
                Div(InlineRadios('ml_answer', css_class='capital-market-form-radio')),
                Div(InlineRadios('model_answer', css_class='capital-market-form-radio')),
            )
            self.helper.add_input(Submit('submit', 'Update', css_class='btn-dark'))

    class Meta:
        model = QuestionnaireA
        fields = ['ml_answer', 'model_answer']


class InvestmentPreferencesForm(forms.ModelForm):
    answer_1 = forms.ChoiceField(
        choices=((1, '0-1'), (2, '2-3'), (3, '4-100')),
        widget=forms.RadioSelect(),
    )
    answer_2 = forms.ChoiceField(
        choices=((1, 'Low risk'), (2, 'Medium risk'), (3, 'High risk')),
        widget=forms.RadioSelect(),
    )
    answer_3 = forms.ChoiceField(
        choices=((1, 'Safest'), (2, 'Sharpest'), (3, 'Maximum return')),
        widget=forms.RadioSelect(),
    )

    def __init__(self, *args, **kwargs):
        user_preferences_instance = kwargs.pop('user_preferences_instance', None)
        # self.get_questionnaire_graphs(kwargs)

        # Form
        form_type = kwargs.pop('form_type', 'create')
        super().__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.form_id = 'capital-market-form'
        self.helper.attrs = {
            'hx-post': reverse_lazy('capital_market_investment_preferences_form'),
            'hx-target': '#capital-market-form',
            'hx-swap': 'outerHTML'
        }
        # TODO: make dynamic code that updates CSV files from `/backend_api/DB/...`
        self.fields['answer_1'].label = format_html('<span class="capital-market-form-label">'
                                                    'Question #1: For how many years do you want to invest?'
                                                    '</span>')
        first_graph = f"{settings.STATIC_URL}img/graphs/distribution_graph.png"
        self.fields['answer_2'].label = format_html('<span class="capital-market-form-label">'
                                                    'Question #2: Which distribution do you prefer?'
                                                    '</span>'
                                                    f'<div class="capital-market-form-label capital-market-form-img">'
                                                    f'<img src="{first_graph}">'
                                                    f'</div>')
        second_graph = f"{settings.STATIC_URL}img/graphs/three_portfolios.png"
        self.fields['answer_3'].label = format_html('<span class="capital-market-form-label">'
                                                    'Question #3: What is your preferable graph?'
                                                    '</span>'
                                                    f'<div class="capital-market-form-label capital-market-form-img">'
                                                    f'<img src="{second_graph}">'
                                                    f'</div>')

        if form_type == 'create':
            self.helper.layout = Layout(
                HTML('<h1 style="font-weight: 900;">Capital Market Form</h1>'),
                HTML('<h5 style="color: rgb(150, 150, 150);">Completing the survey is <u>essential</u>'
                     ' for using our website and AI algorithm</h5>'),
                Div(InlineRadios('answer_1', css_class='capital-market-form-radio')),
                Div(InlineRadios('answer_2', css_class='capital-market-form-radio')),
                Div(InlineRadios('answer_3', css_class='capital-market-form-radio')),
            )
            self.helper.add_input(Submit('submit', 'Submit', css_class='btn-dark'))
        elif form_type == 'update':
            self.helper.layout = Layout(
                HTML('<h1 style="font-weight: 900;">Capital Market Form</h1>'),
                HTML('<h5 style="color: rgb(150, 150, 150);">Update the previous survey you completed</h5>'),
                Div(InlineRadios('answer_1', css_class='capital-market-form-radio')),
                Div(InlineRadios('answer_2', css_class='capital-market-form-radio')),
                Div(InlineRadios('answer_3', css_class='capital-market-form-radio')),
            )
            self.helper.add_input(Submit('submit', 'Update', css_class='btn-dark'))

    @staticmethod
    def get_questionnaire_graphs(user_preferences_instance, kwargs):
        # User preferences
        ml_answer = user_preferences_instance.ml_answer
        model_answer = user_preferences_instance.model_answer
        db_tuple = manage_data.get_extended_data_from_db(settings.stocks_symbols, ml_answer, model_answer)
        sectors_data, sectors, closing_prices_table, three_best_portfolios, three_best_sectors_weights, \
            pct_change_table, yield_list = db_tuple
        # Saves two graphs
        manage_data.plot_distribution_of_portfolio(yield_list)
        manage_data.plot_three_portfolios_graph(
            three_best_portfolios, three_best_sectors_weights, sectors, pct_change_table
        )

    class Meta:
        model = QuestionnaireB
        fields = ['answer_1', 'answer_2', 'answer_3']
