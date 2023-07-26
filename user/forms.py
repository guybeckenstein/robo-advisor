from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm

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
