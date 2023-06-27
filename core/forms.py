from crispy_forms.bootstrap import InlineRadios
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit, Layout, Div, HTML
from django import forms
from django.utils.html import format_html

from core.models import Questionnaire
from django.urls import reverse_lazy


class CapitalMarketForm(forms.ModelForm):
    answer_1 = forms.ChoiceField(
        choices=((1, '1'), (2, '2'), (3, '3')),
        widget=forms.RadioSelect(),
        label='Question #1: ...',
    )
    answer_2 = forms.ChoiceField(
        choices=((1, '1'), (2, '2'), (3, '3')),
        widget=forms.RadioSelect(),
        label='Question #2: ...',
    )
    answer_3 = forms.ChoiceField(
        choices=((1, '1'), (2, '2'), (3, '3')),
        widget=forms.RadioSelect(),
        label='Question #3: ...',
    )

    def __init__(self, *args, **kwargs):
        form_type = kwargs.pop('form_type', 'create')
        super().__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.form_id = 'capital-market-form'
        self.helper.attrs = {
            'hx-post': reverse_lazy('capital_market_form'),
            'hx-target': '#capital-market-form',
            'hx-swap': 'outerHTML'
        }
        self.fields['answer_1'].label = format_html('<span class="capital-market-form-label">Question #1: ...</span>')
        self.fields['answer_2'].label = format_html('<span class="capital-market-form-label">Question #2: ...</span>')
        self.fields['answer_3'].label = format_html('<span class="capital-market-form-label">Question #3: ...</span>')

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

    class Meta:
        model = Questionnaire
        fields = ['answer_1', 'answer_2', 'answer_3']
