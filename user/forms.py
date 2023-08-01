from crispy_forms.bootstrap import FormActions
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit, Layout, HTML
from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, PasswordChangeForm

from user.models import InvestorUser


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


class AccountMetadataForm(forms.ModelForm):
    def __init__(self, *args, disabled_project=True, **kwargs):
        super(AccountMetadataForm, self).__init__(*args, **kwargs)
        self.fields['email'].disabled = disabled_project
        self.fields['first_name'].disabled = disabled_project
        self.fields['last_name'].disabled = disabled_project
        self.fields['date_joined'].disabled = disabled_project

    class Meta:
        model = User
        fields = ('email', 'first_name', 'last_name', 'date_joined')


class UpdateUserNameForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super(UpdateUserNameForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.layout = Layout(
            Layout(
                'first_name',
                'last_name',
            ),
            HTML('<hr>'),
            FormActions(Submit('submit', 'Save', css_class='btn-dark')),
        )

    class Meta:
        model = User
        fields = ('first_name', 'last_name',)

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
        model = User
        fields = ('old_password', 'new_password1', 'new_password2',)


class UpdateInvestorUserForm(forms.ModelForm):
    def __init__(self, *args, disabled_project=True, **kwargs):
        super(UpdateInvestorUserForm, self).__init__(*args, **kwargs)
        self.fields['starting_investment_amount'].disabled = disabled_project
        self.fields['stocks_symbols'].disabled = disabled_project
        self.fields['stocks_weights'].disabled = disabled_project
        self.fields['sectors_names'].disabled = disabled_project
        self.fields['stocks_weights'].disabled = disabled_project
        self.fields['sectors_weights'].disabled = disabled_project
        self.fields['annual_returns'].disabled = disabled_project
        self.fields['annual_max_loss'].disabled = disabled_project
        self.fields['annual_volatility'].disabled = disabled_project
        self.fields['annual_sharpe'].disabled = disabled_project
        self.fields['total_change'].disabled = disabled_project
        self.fields['monthly_change'].disabled = disabled_project

    class Meta:
        model = InvestorUser
        fields = (
            'starting_investment_amount',
            'stocks_symbols',
            'stocks_weights',
            'sectors_names',
            'sectors_weights',
            'annual_returns',
            'annual_max_loss',
            'annual_volatility',
            'annual_sharpe',
            'total_change',
            'monthly_change',
            'stocks_weights',
        )
