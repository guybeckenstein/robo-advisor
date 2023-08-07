from crispy_forms.bootstrap import FormActions
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit, Layout, HTML
from django import forms
from django.contrib.auth.forms import UserCreationForm, PasswordChangeForm
from django.core.exceptions import ValidationError
from django.urls import reverse_lazy

from .models import InvestorUser, CustomUser


def ac_il_email_validator(value):
    if not value.endswith('.ac.il'):
        raise ValidationError('Enter a valid email address with the domain ending ".ac.il".')


class UserRegisterForm(UserCreationForm):
    first_name = forms.CharField(label="First name", max_length=20)
    last_name = forms.CharField(label="Last name", max_length=20)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper(self)

        self.helper.form_id = 'signup-form'
        self.helper.add_input(Submit('submit', 'Submit'))

    def clean_email(self):
        email = self.cleaned_data.get('email')
        try:
            ac_il_email_validator(email)
        except ValidationError as e:
            raise forms.ValidationError(str(e))
        return email

    class Meta:
        model = CustomUser
        fields = ('email', 'first_name', 'last_name', 'phone_number',)

        widgets = {
            'password': forms.PasswordInput(),

            'email': forms.TextInput(attrs={
                'hx-post': reverse_lazy('check_email'),
                'hx-target': '#div_id_email',
                'hx-trigger': 'keyup changed delay:1s'
            }),
            'phone_number': forms.TextInput(attrs={
                'hx-post': reverse_lazy('check_phone_number'),
                'hx-target': '#div_id_phone_number',
                'hx-trigger': 'keyup changed delay:1s'
            })
        }


class AccountMetadataForm(forms.ModelForm):
    def __init__(self, *args, disabled_project=True, **kwargs):
        super(AccountMetadataForm, self).__init__(*args, **kwargs)
        self.fields['email'].disabled = disabled_project
        self.fields['first_name'].disabled = disabled_project
        self.fields['last_name'].disabled = disabled_project
        self.fields['phone_number'].disabled = disabled_project
        self.fields['date_joined'].disabled = disabled_project

    class Meta:
        model = CustomUser
        fields = ('email', 'first_name', 'last_name', 'phone_number', 'date_joined')


class UpdateUserNameAndPhoneNumberForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super(UpdateUserNameAndPhoneNumberForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.layout = Layout(
            Layout(
                'first_name',
                'last_name',
                'phone_number',
            ),
            HTML('<hr>'),
            FormActions(Submit('submit', 'Save', css_class='btn-dark')),
        )

    class Meta:
        model = CustomUser
        fields = ('first_name', 'last_name', 'phone_number',)

    @property
    def clean_fields(self):
        cleaned_data = super().clean()
        # Test first name
        first_name: str = cleaned_data.get("first_name")
        error_message: str = self.check_name(first_name, True)
        if error_message != "":
            raise forms.ValidationError(error_message)
        # Test last name
        last_name: str = cleaned_data.get("last_name")
        error_message: str = self.check_name(last_name, False)
        if error_message != "":
            raise forms.ValidationError(error_message)
        return cleaned_data

    @staticmethod
    def check_name(name: str, is_first_name: bool) -> str:
        error_message = ""
        if name == "":
            if is_first_name:
                error_message = "Please add a first name."
            else:
                error_message = "Please add a last name."
        for letter in name:
            if not (
                    (ord('a') <= ord(letter) <= ord('z')) or
                    (ord('A') <= ord(letter) <= ord('Z')) or
                    (ord('א') <= ord(letter) <= ord('ת'))
            ):
                if is_first_name:
                    error_message = "First name can only contains English/Hebrew letters."
                else:
                    error_message = "Last name can only contains English/Hebrew letters."
                break
        print('Valid name')
        return error_message


class PasswordChangingForm(PasswordChangeForm):
    old_password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control', 'type': 'password'}))
    new_password1 = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control', 'type': 'password'}))
    new_password2 = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control', 'type': 'password'}))

    class Meta:
        model = CustomUser
        fields = ('old_password', 'new_password1', 'new_password2',)


class UpdateInvestorUserForm(forms.ModelForm):
    starting_investment_amount = forms.CharField()
    stocks_symbols = forms.MultipleChoiceField(widget=forms.SelectMultiple)

    def __init__(self, *args, disabled_project=True, **kwargs):
        super(UpdateInvestorUserForm, self).__init__(*args, **kwargs)
        self.fields['starting_investment_amount'].disabled = disabled_project
        self.fields['stocks_symbols'].disabled = disabled_project
        if self.instance.stocks_symbols:
            symbols_list = self.instance.stocks_symbols[1:-1].split(',')
            self.fields['stocks_symbols'].choices = [(symbol.strip(), symbol.strip()) for symbol in symbols_list]
        else:
            self.fields['stocks_symbols'].choices = []  # or provide default choices if needed

    class Meta:
        model = InvestorUser
        fields = ('starting_investment_amount', 'stocks_symbols',)
