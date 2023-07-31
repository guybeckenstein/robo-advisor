from django import forms
from django.contrib.auth.views import PasswordChangeView
from django.core.exceptions import BadRequest
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.urls import reverse_lazy

from .forms import UserRegisterForm, AccountMetadataForm, UpdateUserNameForm, UpdateInvestorUserForm, \
    PasswordChangingForm


def register(request):
    if request.user.is_authenticated:
        return redirect('homepage')
    elif request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account " {username} " has been created! You are now able to log in')
            return redirect('login')
    else:
        form = UserRegisterForm()

    context = {
        'form': form,
        'title': "Register"
    }
    return render(request, 'user/register.html', context=context)


def login(request):
    context = {
        'title': "Login"
    }
    if request.user.is_authenticated:
        return redirect('homepage')
    else:
        return render(request, 'user/login.html', context=context)


def logout(request):
    # TODO: make this function prevent guests from entering 'logout' page
    if request.user.is_authenticated:
        return redirect('logout')
    else:
        return redirect('homepage')


@login_required
def profile_main(request):
    if request.method == 'GET':
        form: forms.ModelForm = AccountMetadataForm(instance=request.user, disabled_project=True)
    else:
        raise BadRequest
    context = {
        'form': form,
        'user': request.user,
        'title': f"{request.user.first_name}'s profile"
    }
    return render(request, 'user/profile_main.html', context=context)


@login_required
def profile_account(request):
    context = {
        'user': request.user,
        'title': "Account page"
    }
    return render(request, 'user/profile_account.html', context=context)


@login_required
def profile_account_details(request):
    if request.method == 'GET':
        form: forms.ModelForm = UpdateUserNameForm(instance=request.user)
    elif request.method == 'POST':
        form = UpdateUserNameForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            # messages.success(request, 'Your account details have been updated successfully.')
            return redirect('profile_account')
    else:
        raise BadRequest
    context = {
        'form': form,
        'user': request.user,
        'title': "Update details"
    }
    return render(request, 'user/profile_account_name.html', context=context)


class MyPasswordChangeForm(PasswordChangeView):
    form_class = PasswordChangingForm
    template_name = 'user/profile_account_password.html'
    success_url = reverse_lazy('profile_account')


@login_required
def profile_investor(request):
    if request.method == 'GET':
        form: forms.ModelForm = UpdateInvestorUserForm(instance=request.user, disabled_project=True)
    elif request.method == 'POST':
        pass
    else:
        raise BadRequest
    context = {
        'form': form,
        'user': request.user,
        'title': "Update investments details"
    }
    return render(request, 'user/profile_investor.html', context=context)
