from crispy_forms.bootstrap import InlineRadios
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit, Layout, Div, HTML, Field
from django import forms
from django.utils.html import format_html

from accounts.models import InvestorUser
from service.util import data_management
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
        self.fields['ml_answer'].label = format_html(
            '<span class="capital-market-form-label">'
            'Question #1: Would you like to use machine learning algorithms for stock market investments?'
            '</span>'
        )
        self.fields['model_answer'].label = format_html(
            '<span class="capital-market-form-label">'
            'Question #2: Which statistic model would you like to use for stock market investments?'
            '</span>'
        )

        main_header = '<h1 style="font-weight: 900;">Capital Market Preferences Form - Algorithms</h1>'
        if form_type == 'create':
            self.helper.layout = Layout(
                HTML(main_header),
                HTML('<h5 class="text-info">'
                     'Completing the survey is <u>essential</u> for using our website and AI algorithm'
                     '</h5>'),
                Div(InlineRadios('ml_answer', css_class='capital-market-form-radio')),
                Div(InlineRadios('model_answer', css_class='capital-market-form-radio')),
            )
            self.helper.add_input(Submit('submit', 'Submit', css_class='btn-dark'))
        elif form_type == 'update':
            self.helper.layout = Layout(
                HTML(main_header),
                HTML('<h5 class="text-info">'
                     'Update your capital market algorithm preferences form'
                     '</h5>'),
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
        questionnaire_a: QuestionnaireA = kwargs.pop('user_preferences_instance', None)
        self.get_questionnaire_graphs(questionnaire_a)

        # Form
        form_type = kwargs.pop('form_type', 'create')
        stocks_collections_number: str = kwargs.pop('collections_number', None)
        super().__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.form_id = 'capital-market-form'
        self.fields['answer_1'].label = format_html('<span class="capital-market-form-label">'
                                                    'Question #1: For how many years do you want to invest?'
                                                    '</span>')
        ml_answer: int = questionnaire_a.ml_answer
        model_answer: int = questionnaire_a.model_answer
        sub_folder = f'{stocks_collections_number}/{str(ml_answer)}{str(model_answer)}'  # Sub folder for current user to fetch its relevant graphs
        first_graph = f"{settings.STATIC_URL}img/graphs/{sub_folder}/distribution_graph.png"
        self.fields['answer_2'].label = format_html('<span class="capital-market-form-label">'
                                                    'Question #2: Which distribution do you prefer?'
                                                    '</span>'
                                                    f'<div class="capital-market-form-label capital-market-form-img">'
                                                    f'<img src="{first_graph}">'
                                                    f'</div>')
        second_graph = f"{settings.STATIC_URL}img/graphs/{sub_folder}/three_portfolios.png"
        self.fields['answer_3'].label = format_html('<span class="capital-market-form-label">'
                                                    'Question #3: What is your preferable graph?'
                                                    '</span>'
                                                    f'<div class="capital-market-form-label capital-market-form-img">'
                                                    f'<img class="capital-market-form-img" src="{second_graph}">'
                                                    f'</div>')

        main_header = '<h1 style="font-weight: 900;">Capital Market Preferences Form - Investments</h1>'
        if form_type == 'create':
            self.helper.layout = Layout(
                HTML(main_header),
                HTML('<h5 class="text-info">'
                     'Completing the survey is <u>essential</u> for using our website and AI algorithm'
                     '</h5>'),
                Div(InlineRadios('answer_1', css_class='capital-market-form-radio')),
                Div(InlineRadios('answer_2', css_class='capital-market-form-radio')),
                Div(InlineRadios('answer_3', css_class='capital-market-form-radio')),
            )
            self.helper.add_input(Submit('submit', 'Submit', css_class='btn-dark'))
        elif form_type == 'update':
            self.helper.layout = Layout(
                HTML(main_header),
                HTML('<h5 class="text-info">'
                     'Update your capital market investment preferences form'
                     '</h5>'),
                Div(InlineRadios('answer_1', css_class='capital-market-form-radio')),
                Div(InlineRadios('answer_2', css_class='capital-market-form-radio')),
                Div(InlineRadios('answer_3', css_class='capital-market-form-radio')),
            )
            self.helper.add_input(Submit('submit', 'Update', css_class='btn-dark'))

    @staticmethod
    def get_questionnaire_graphs(questionnaire_a: QuestionnaireA):
        # User preferences
        ml_answer = questionnaire_a.ml_answer
        model_answer = questionnaire_a.model_answer
        try:
            investor_user: InvestorUser = InvestorUser.objects.get(user=questionnaire_a.user)
            stocks_collection_number: str = investor_user.stocks_collection_number
        except InvestorUser.DoesNotExist:
            stocks_collection_number: str = '1'
        stocks_symbols = data_management.get_stocks_symbols_from_collection(stocks_collection_number)
        db_tuple = data_management.get_extended_data_from_db(
            stocks_symbols, ml_answer, model_answer, stocks_collection_number
        )
        sectors_data, sectors, closing_prices_table, three_best_portfolios, three_best_sectors_weights, \
            pct_change_table, yield_list = db_tuple
        # Saves two graphs
        sub_folder = f'{str(stocks_collection_number)}/{str(ml_answer)}{str(model_answer)}/'
        data_management.plot_distribution_of_portfolio(yield_list, sub_folder=sub_folder)
        data_management.plot_three_portfolios_graph(
            three_best_portfolios, three_best_sectors_weights, sectors, pct_change_table, sub_folder=sub_folder
        )

    class Meta:
        model = QuestionnaireB
        fields = ['answer_1', 'answer_2', 'answer_3']


class CustomNumberInput(forms.widgets.NumberInput):
    template_name = 'widgets/custom_number_input.html'


class AdministrativeToolsForm(forms.Form):
    num_por_simulation = forms.CharField(widget=forms.TextInput(attrs={'type': 'number'}))
    min_num_por_simulation = forms.CharField(widget=forms.TextInput(attrs={'type': 'number'}))
    record_percent_to_predict = forms.CharField(widget=CustomNumberInput(attrs={'type': 'number', 'step': '0.01'}))
    test_size_machine_learning = forms.CharField(widget=CustomNumberInput(attrs={'type': 'number', 'step': '0.01'}))
    selected_ml_model_for_build = forms.CharField(widget=forms.TextInput(attrs={'type': 'number'}))
    gini_v_value = forms.CharField(widget=forms.TextInput(attrs={'type': 'number'}))

    def __init__(self, *args, **kwargs):
        # Form
        super().__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.form_id = 'administrative-tools-form'
        self.helper.attrs = {
            'hx-post': reverse_lazy('administrative_tools_form'),
            'hx-target': '#administrative-tools-form',
            'hx-swap': 'outerHTML'
        }
        self.helper.layout = Layout(
            HTML('<h1 style="font-weight: 900;">Update Models\' Data</h1>'),
            HTML('<h5 class="text-info">'
                 'Administration Tools enable modification of some of the metadata, which affects models directly'
                 '</h5>'),
            Field('num_por_simulation'),
            Field('min_num_por_simulation'),
            Field('record_percent_to_predict'),
            Field('test_size_machine_learning'),
            Field('selected_ml_model_for_build'),
            Field('gini_v_value'),
        )
        self.helper.add_input(Submit('submit', 'Submit', css_class='btn-dark'))
