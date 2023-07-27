from crispy_forms.bootstrap import InlineRadios
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, HTML, Div, Submit
from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse_lazy
from django.utils.html import format_html

from .models import UserPreferences

# TODO: make a User model that its unique field is `email` and not `username`; i.e. we don't need username at all


class UserRegisterForm(UserCreationForm):
    """
    User fields are:
    1) Email
    2) Phone number
    3) First name
    4) Last name
    5) Password
    Once a user is created, it cannot change its email (academic email is immutable)
    """
    # TODO: Email must include `@ac` - thus be academic email!
    email = forms.EmailField(label="Email address")
    # TODO: add phone number field with validation
    # phone = PhoneNumberField()
    first_name = forms.CharField(label="First name", max_length=20)
    last_name = forms.CharField(label="Last name", max_length=20)

    class Meta:
        model = User
        # TODO: add phone number field with validation
        # fields = ['email', 'phone', 'first_name', 'last_name', 'password']
        fields = ['email', 'first_name', 'last_name']


class UserUpdateForm(forms.ModelForm):
    first_name = forms.CharField(label="First name", max_length=20)
    last_name = forms.CharField(label="Last name", max_length=20)
    # TODO: add phone number field with validation
    # phone = PhoneNumberField()

    class Meta:
        model = User
        # TODO: make password update possible to user (it is not currently), including validation
        # TODO: add phone number field with validation
        # fields = ['phone', 'first_name', 'last_name']
        fields = ['first_name', 'last_name']


class UserPreferencesForm(forms.ModelForm):
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
            'hx-post': reverse_lazy('preferences_form'),
            'hx-target': '#preferences-form',
            'hx-swap': 'outerHTML'
        }
        # TODO: make dynamic code that updates CSV files from `/backend-api/DB/...`
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
        model = UserPreferences
        fields = ['ml_answer', 'model_answer']
