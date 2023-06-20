from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from .models import Profile


class UserRegisterForm(UserCreationForm):
    first_name = forms.CharField(label="First name", max_length=20)
    last_name = forms.CharField(label="Last name", max_length=20)
    email = forms.EmailField()
    date_of_birth = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))
    # phone = PhoneNumberField()
    # birth_date = forms.DateField(label='What is your birth date?', widget=forms.SelectDateWidget)
    # TODO: add security question number & answer

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'date_of_birth', 'password1', 'password2']


class UserUpdateForm(forms.ModelForm):
    first_name = forms.CharField(label="First name", max_length=20)
    last_name = forms.CharField(label="Last name", max_length=20)
    email = forms.EmailField()

    # phone = PhoneNumberField()
    # birth_date = forms.DateField(label='What is your birth date?', widget=forms.SelectDateWidget)
    # TODO: add security question number & answer

    class Meta:
        model = User
        # fields = ['username', 'first_name', 'last_name', 'email', 'phone', 'birth_date']
        fields = ['username', 'first_name', 'last_name', 'email']


class ProfileUpdateForm(forms.ModelForm):

    class Meta:
        model = Profile
        fields = ['image']
