import datetime
import json

import pytz

from allauth.account.views import SignupView, LoginView
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
from accounts import forms as account_forms
from .models import InvestorUser, CustomUser


class SignUpView(SignupView):
    """
    Creates new employee
    """
    template_name = 'account/guest/registration.html'
    form_class = account_forms.UserRegisterForm
    htmx = True

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
            form.save()
            email = form.cleaned_data.get('email')
            messages.success(request, f"Successfully created your account - '{email}'.")
            return redirect('account_login')

        context = {
            'form': form,
            'title': "Sign Up"
        }
        return render(request, self.template_name, context=context)


    def non_field_errors(self) -> list[str]:
        errors = super().non_field_errors()
        if 'email' in self.errors and 'already exists' in self.errors['email'][0]:
            errors = ["A user is already assigned with this email. Please use a different email address."]
            return errors


class HtmxLoginView(LoginView):
    template_name = 'account/guest/login.html'
    htmx = True

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Login'
        return context

    def form_valid(self, form):
        email: str = form.cleaned_data['login']
        try:
            user: CustomUser = CustomUser.objects.get(email=email)
        except CustomUser.DoesNotExist:
            raise ValueError(f'No user with email - `{email}`')
        try:
            InvestorUser.objects.get(user=user)
            # Can only proceed if there is an InvestorUser instance
            if user.last_login is not None:
                last_login = user.last_login.astimezone(pytz.timezone('Asia/Jerusalem'))
                current: datetime.datetime = datetime.datetime.now(tz=pytz.timezone('Asia/Jerusalem'))
                if (current - last_login).days > 0:
                    web_actions.save_three_user_graphs_as_png(user=user)
            else:
                raise AttributeError('Invalid logic - InvestorUser exists before the user has logged in!')
        except InvestorUser.DoesNotExist:
            pass
        return super().form_valid(form)


def logout_view(request):
    logout(request)
    context = {
        'title': "You Have Been Logged Out"
    }
    return render(request, 'account/authenticated/logout.html', context=context)


@login_required
def profile_main(request):
    if request.method == 'GET':
        form: forms.ModelForm = account_forms.AccountMetadataForm(instance=request.user, disabled_project=True)
    else:
        raise BadRequest
    context = {
        'form': form,
        'title': f"{request.user.first_name}'s Profile"
    }
    return render(request, 'account/authenticated/profile_main.html', context=context)


@login_required
def profile_account(request):
    context = {
        'title': "Account Page"
    }
    return render(request, 'account/authenticated/profile_account.html', context=context)


@login_required
def profile_account_details(request):
    if request.method == 'GET':
        form: forms.ModelForm = account_forms.UpdateUserNameAndPhoneNumberForm(instance=request.user)
    elif request.method == 'POST':
        form: forms.ModelForm = account_forms.UpdateUserNameAndPhoneNumberForm(request.POST, instance=request.user)
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
    return render(request, 'account/authenticated/profile_account_details.html', context=context)


class MyPasswordChangeForm(PasswordChangeView):
    form_class = account_forms.PasswordChangingForm
    template_name = 'account/authenticated/profile_account_password.html'
    success_url = reverse_lazy('profile_account')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Forgot Password'  # Add your desired value for the 'title' key here
        return context


@login_required
def profile_investor(request):
    stocks_symbols_data: dict[list] = data_management.get_stocks_from_json_file()
    styled_stocks_symbols_data: dict[list] = data_management.get_styled_stocks_symbols_data(
        stocks_symbols_data=stocks_symbols_data
    )
    is_form_filled = True

    try:
        investor_user: InvestorUser = get_object_or_404(InvestorUser, user=request.user)
        if request.method == 'GET':
            form: forms.ModelForm = account_forms.UpdateInvestorUserForm(
                instance=investor_user,
                investor_user_instance=investor_user,
                disabled_project=True,
            )
        elif request.method == 'POST':
            form: forms.ModelForm = account_forms.UpdateInvestorUserForm(
                request.POST,
                investor_user_instance=investor_user,
                instance=investor_user,
            )
            if form.is_valid():
                investments: QuerySet[Investment] = Investment.objects.filter(investor_user=investor_user)
                questionnaire_a: QuestionnaireA = get_object_or_404(QuestionnaireA, user=request.user)
                questionnaire_b: QuestionnaireB = get_object_or_404(QuestionnaireB, user=request.user)
                (annual_max_loss, annual_returns, annual_sharpe, annual_volatility, daily_change, monthly_change,
                 risk_level, sectors_names, sectors_weights, stocks_symbols, stocks_weights, total_change, portfolio) \
                    = web_actions.create_portfolio_and_get_data(
                    answers_sum=questionnaire_b.answers_sum,
                    stocks_collection_number=investor_user.stocks_collection_number,
                    questionnaire_a=questionnaire_a,
                )
                data_management.changing_portfolio_investments_treatment_web(investor_user, portfolio, investments)
                # Update Investments' Data
                for investment in investments:
                    if investment.make_investment_inactive() is False:
                        break  # In this case we should not continue iterating over the new-to-old-sorted investments
                    investment.save()
                # Update Form Data
                form.save()
                # Update InvestorUser data
                investor_user.stocks_weights = stocks_weights
                investor_user.stocks_collection_number = form.cleaned_data['stocks_collection_number']
                investor_user.annual_returns = annual_returns
                investor_user.annual_max_loss = annual_max_loss
                investor_user.annual_volatility = annual_volatility
                investor_user.annual_sharpe = annual_sharpe
                investor_user.save()
                # Continue
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
    return render(request, 'account/authenticated/profile_investor.html', context=context)
