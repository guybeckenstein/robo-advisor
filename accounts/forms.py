from allauth.account.forms import LoginForm, ResetPasswordForm
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit, Layout, HTML
from django import forms
from django.contrib.auth.forms import UserCreationForm, PasswordChangeForm
from django.core.exceptions import ValidationError
from django.urls import reverse_lazy

from service.util import data_management
from service.util.data_management import get_stocks_symbols_from_json_file
from accounts.models import InvestorUser, CustomUser


def ac_il_email_validator(value):
    if not value.endswith('.ac.il'):
        raise ValidationError('Enter a valid email address with the domain ending ".ac.il".')


class UserRegisterForm(UserCreationForm):
    first_name = forms.CharField(label="First name", max_length=20)
    last_name = forms.CharField(label="Last name", max_length=20)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper(self)

        self.fields['first_name'].widget.attrs.update({
            'hx-get': reverse_lazy('check_first_name'),
            'hx-target': '#div_id_first_name',
            'hx-trigger': 'keyup[target.value.length > 6]'

        })
        self.fields['last_name'].widget.attrs.update({
            'hx-get': reverse_lazy('check_last_name'),
            'hx-target': '#div_id_last_name',
            'hx-trigger': 'keyup[target.value.length > 4]'

        })

        self.helper.form_id = 'signup-form'
        self.helper.add_input(Submit('submit', 'Submit'))

    class Meta:
        model = CustomUser
        fields = ('email', 'first_name', 'last_name', 'phone_number', 'password1', 'password2')
        widgets = {
            'email': forms.TextInput(attrs={
                'hx-get': reverse_lazy('check_email'),
                'hx-target': '#div_id_email',
                'hx-trigger': 'keyup[target.value.length > 8]'
            }),
            'phone_number': forms.TextInput(attrs={
                'hx-get': reverse_lazy('check_phone_number'),
                'hx-target': '#div_id_phone_number',
                'hx-trigger': 'keyup changed delay:2s'
            }),
        }

    def clean_email(self):
        email = self.cleaned_data.get('email')
        try:
            ac_il_email_validator(email)
        except ValidationError as e:
            raise forms.ValidationError(str(e))
        if CustomUser.objects.filter(email=email).exists():
            raise forms.ValidationError("User with this email address is already registered.")
        return email

    # Name check (must be English letters)
    def clean_first_name(self):
        first_name = self.cleaned_data['first_name']
        # Add your validation logic here
        if not first_name.replace(" ", "").isalpha():
            raise forms.ValidationError("Only alphabetic characters and spaces are allowed.")
        return first_name

    def clean_last_name(self):
        last_name = self.cleaned_data['last_name']
        # Add your validation logic here
        if not last_name.replace(" ", "").isalpha():
            raise forms.ValidationError("Only alphabetic characters and spaces are allowed.")
        return last_name


class CustomLoginForm(LoginForm):
    # fields = ['email', 'password']
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper(self)

        self.fields['login'].widget.attrs.update({
            'hx-post': reverse_lazy('check_login_email_view'),
            'hx-target': '#email-validation',
            'hx-trigger': 'keyup changed delay:2s'
        })


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


class UpdateAccountDetailsForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super(UpdateAccountDetailsForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.layout = Layout(
            Layout(
                'first_name',
                'last_name',
                'phone_number',
            ),
            HTML('<hr>'),
        )

    class Meta:
        model = CustomUser
        fields = ('first_name', 'last_name', 'phone_number',)
        widgets = {
            'phone_number': forms.TextInput(attrs={
                'hx-get': reverse_lazy('check_phone_number'),
                'hx-target': '#div_id_phone_number',
                'hx-trigger': 'keyup changed delay:1s'
            }),
            'first_name': forms.TextInput(attrs={
                'hx-get': reverse_lazy('check_first_name'),
                'hx-target': '#div_id_first_name',
                'hx-trigger': 'keyup[target.value.length > 6]'

            }),
            'last_name': forms.TextInput(attrs={

                'hx-get': reverse_lazy('check_last_name'),
                'hx-target': '#div_id_last_name',
                'hx-trigger': 'keyup[target.value.length > 4]'

            })
        }

    def clean_last_name(self):
        last_name = self.cleaned_data['last_name']
        # Add your validation logic here
        if not last_name.replace(" ", "").isalpha():
            raise forms.ValidationError("Only alphabetic characters and spaces are allowed.")
        return last_name

    def clean_first_name(self):
        first_name = self.cleaned_data['first_name']
        # Add your validation logic here
        if not first_name.replace(" ", "").isalpha():
            raise forms.ValidationError("Only alphabetic characters and spaces are allowed.")
        return first_name

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
            english_small_letter: bool = ord('a') <= ord(letter) <= ord('z')
            english_capital_letter: bool = ord('A') <= ord(letter) <= ord('Z')
            hebrew_letter: bool = ord('א') <= ord(letter) <= ord('ת')
            if not (english_small_letter or english_capital_letter or hebrew_letter):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.form_id = 'profile-account-password'
        self.helper.attrs = {
            'hx-post': reverse_lazy('profile_account_password'),
            'hx-target': 'body',
            'hx-swap': 'outerHTML',
            # 'hx-trigger': 'change delay:1s',
        }
        self.helper.add_input(Submit('submit', 'Change Password', css_class='btn-dark margin-bottom'))

    class Meta:
        model = CustomUser
        fields = ('old_password', 'new_password1', 'new_password2',)

    def clean_old_password(self):
        old_password = self.cleaned_data['old_password']
        if len(old_password) == 0:
            raise forms.ValidationError("Old password is empty")
        return old_password


def get_indexes_tuple(size) -> list[tuple]:
    res: list = []
    for i in range(1, size + 1):
        str_i = str(i)
        curr_tuple = (str_i, str_i)
        res.append(curr_tuple)
    return res


class UpdateInvestorUserForm(forms.ModelForm):
    total_investment_amount = forms.CharField(required=False)
    total_profit = forms.CharField(required=False)
    list_of_indexes_tuple: list[tuple] = get_indexes_tuple(size=4)
    stocks_collection_number = forms.ChoiceField(
        widget=forms.RadioSelect(attrs={'class': 'horizontal-radio'}), choices=list_of_indexes_tuple
    )
    stocks_symbols = forms.MultipleChoiceField(widget=forms.SelectMultiple, required=False)

    def __init__(self, *args, disabled_project=True, **kwargs):
        investor_user_instance: InvestorUser = kwargs.pop('investor_user_instance', None)
        collection_number: str = str(investor_user_instance.stocks_collection_number)
        stocks_symbols_data: dict[list] = data_management.get_stocks_from_json_file()
        symbols_list: list[str] = sorted(
            data_management.get_styled_stocks_symbols_data(stocks_symbols_data)[collection_number]
        )

        # Form
        super(UpdateInvestorUserForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.form_id = 'investor-form'

        self.fields['total_investment_amount'].disabled = disabled_project
        self.fields['total_profit'].disabled = disabled_project
        self.fields['stocks_symbols'].choices = [(symbol.strip(), symbol.strip()) for symbol in symbols_list]
        self.helper.add_input(Submit('update', 'Update', css_class='btn-dark'))

    def save(self, commit=True):
        instance = super().save(commit=False)

        collection_number: int = self.cleaned_data.get('stocks_collection_number', -1)
        print(f"collection number in save button {collection_number}")
        stocks_symbols: list[str] = get_stocks_symbols_from_json_file(collection_number=collection_number)

        # Convert the list to a formatted string
        instance.stocks_symbols = stocks_symbols

        if commit:
            instance.save()

        return instance

    class Meta:
        model = InvestorUser
        fields = ('total_investment_amount', 'total_profit', 'stocks_collection_number', 'stocks_symbols',)


class CustomResetPasswordForm(ResetPasswordForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add any additional fields you want
        self.fields['email'].widget.attrs.update({
            'hx-post': reverse_lazy('check_login_email_reset'),
            'hx-target': '#email-validation',
            'hx-trigger': 'keyup changed delay:2s'
        })
