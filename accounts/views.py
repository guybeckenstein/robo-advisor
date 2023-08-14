import json

from allauth.account.views import SignupView, LoginView
from crispy_forms.templatetags.crispy_forms_filters import as_crispy_field
from django import forms
from django.contrib.auth import logout
from django.contrib.auth.views import PasswordChangeView
from django.core.exceptions import BadRequest
from django.db.models import QuerySet
from django.http import Http404
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.urls import reverse_lazy

from core.models import QuestionnaireA, QuestionnaireB
from investment.models import Investment
from service.util import web_actions, data_management
from service.util.data_management import get_stocks_from_json_file
from .forms import UserRegisterForm, AccountMetadataForm, UpdateUserNameAndPhoneNumberForm, UpdateInvestorUserForm, \
    PasswordChangingForm
from .models import InvestorUser


class SignUpView(SignupView):
    """
    Creates new employee
    """
    template_name = 'account/registration.html'
    form_class = UserRegisterForm

    def get(self, request, *args, **kwargs):
        # Use RequestContext instead of render_to_response from 3.0
        context = {
            'form': self.form_class,
            'title': "Sign Up",
        }
        return render(request, self.template_name, context=context)

    def post(self, request, *args, **kwargs):
        form: forms.ModelForm = self.form_class(request.POST)
        if form.is_valid():
            user = form.save()
            # complete_signup(request, user, app_settings.EMAIL_VERIFICATION, "/")
            email = form.cleaned_data.get('email')
            messages.success(request, f"Successfully created your account - '{email}'.")
            return redirect('account_login')

        context = {
            'form': form,
            'title': "Sign Up"
        }
        return render(request, self.template_name, context=context)


class HtmxLoginView(LoginView):
    template_name = 'account/login.html'
    htmx = True

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Login'
        return context


def logout_view(request):
    logout(request)
    context = {
        'title': "You Have Been Logged Out"
    }
    return render(request, 'account/logout.html', context=context)


@login_required
def profile_main(request):
    if request.method == 'GET':
        form: forms.ModelForm = AccountMetadataForm(instance=request.user, disabled_project=True)
    else:
        raise BadRequest
    context = {
        'form': form,
        'title': f"{request.user.first_name}'s Profile"
    }
    return render(request, 'account/profile_main.html', context=context)


@login_required
def profile_account(request):
    context = {
        'title': "Account Page"
    }
    return render(request, 'account/profile_account.html', context=context)


@login_required
def profile_account_details(request):
    if request.method == 'GET':
        form: forms.ModelForm = UpdateUserNameAndPhoneNumberForm(instance=request.user)
    elif request.method == 'POST':
        form: forms.ModelForm = UpdateUserNameAndPhoneNumberForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, 'Your account details have been updated successfully.')
            return redirect('profile_account')
    else:
        raise BadRequest
    context = {
        'form': form,
        'title': "Update Details",
    }
    return render(request, 'account/profile_account_details.html', context=context)


class MyPasswordChangeForm(PasswordChangeView):
    form_class = PasswordChangingForm
    template_name = 'account/profile_account_password.html'
    success_url = reverse_lazy('profile_account')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Forgot Password'  # Add your desired value for the 'title' key here
        return context


def get_styled_stocks_symbols_data(stocks_symbols_data) -> dict[list]:
    styled_stocks_symbols_data: dict[list] = dict()
    for key, value in stocks_symbols_data.items():
        styled_value: list[str] = list()
        for symbol, description in zip(value[0], value[1]):
            styled_value.append(f'{symbol} -> {description}')
        styled_stocks_symbols_data[key] = styled_value
    return styled_stocks_symbols_data


@login_required
def profile_investor(request):
    stocks_symbols_data: dict[list] = get_stocks_from_json_file()
    styled_stocks_symbols_data: dict[list] = get_styled_stocks_symbols_data(stocks_symbols_data=stocks_symbols_data)
    is_form_filled = True

    try:
        investor_user: InvestorUser = get_object_or_404(InvestorUser, user=request.user)
        if request.method == 'GET':
            form: forms.ModelForm = UpdateInvestorUserForm(
                instance=investor_user,
                investor_user_instance=investor_user,
                disabled_project=True,
            )
        elif request.method == 'POST':
            form: forms.ModelForm = UpdateInvestorUserForm(
                request.POST,
                investor_user_instance=investor_user,
                instance=investor_user,
            )
            if form.is_valid():
                investments: QuerySet[Investment] = Investment.objects.filter(investor_user=investor_user)
                questionnaire_a: QuestionnaireA = get_object_or_404(QuestionnaireA, user=request.user)
                questionnaire_b: QuestionnaireB = get_object_or_404(QuestionnaireB, user=request.user)
                (annual_max_loss, annual_returns, annual_sharpe, annual_volatility, daily_change, monthly_change,
                 risk_level, sectors_names, sectors_weights, stocks_symbols, stocks_weights,
                 total_change, portfolio) = web_actions.create_portfolio_and_get_data(
                    answers_sum=questionnaire_b.answers_sum,
                    stocks_collection_number=investor_user.stocks_collection_number,
                    questionnaire_a=questionnaire_a,
                )
                data_management.changing_portfolio_investments_treatment_web(investor_user, portfolio, investments)
                for investment in investments:
                    if investment.make_investment_inactive() is False:
                        break  # In this case we should not continue iterating over the new-to-old-sorted investments
                    investment.save()
                form.save()
                messages.success(request, 'Your account details have been updated successfully.')
                return redirect('capital_market_algorithm_preferences_form')
        else:
            raise BadRequest
    except Http404:
        is_form_filled = False
        form = None
    context = {
        'form': form,
        'is_form_filled': is_form_filled,
        'stocks_symbols_data': json.dumps(styled_stocks_symbols_data),
        'title': "Update Collections of Stocks Details",
    }
    return render(request, 'account/profile_investor.html', context=context)


# Checks
def check_email(request):
    if request.method == 'POST':
        form: forms.ModelForm = UserRegisterForm(request.POST)
        print(form)
        context = {
            'field': as_crispy_field(form['email']),
            'valid': not form['email'].errors
        }
        return render(request, 'partials/field.html', context)
    else:
        # If it's a GET request, return an empty form
        form: forms.ModelForm = UserRegisterForm()
        context = {
            'field': as_crispy_field(form['email']),
            'valid': True
        }
        return render(request, 'partials/field.html', context)


def check_phone_number(request):
    if request.method == 'POST':
        form: forms.ModelForm = UserRegisterForm(request.POST)
        context = {
            'field': as_crispy_field(form['phone_number']),
            'valid': not form['phone_number'].errors
        }
        return render(request, 'partials/field.html', context)
    else:

        form: forms.ModelForm = UserRegisterForm()
        context = {
            'field': as_crispy_field(form['phone_number']),
            'valid': True
        }
        return render(request, 'partials/field.html', context)
