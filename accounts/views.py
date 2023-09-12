import datetime
import json

import pytz

from allauth.account.views import SignupView, LoginView
from crispy_forms.templatetags.crispy_forms_filters import as_crispy_field
from django import forms
from django.contrib.auth import logout, login
from django.contrib.auth.views import PasswordChangeView
from django.core.exceptions import BadRequest
from django.core.files.storage import FileSystemStorage
from django.http import Http404
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.urls import reverse_lazy

from core.models import QuestionnaireA, QuestionnaireB
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_control
from django.views.decorators.http import require_http_methods

from core.views import make_investments_inactive
from investment.models import Investment
from service.util import web_actions, data_management
from service.config import settings as service_settings
from accounts import forms as account_forms
from core import views as core_views

from .forms import CustomLoginForm
from .models import InvestorUser, CustomUser


@method_decorator(cache_control(no_cache=True, must_revalidate=True, no_store=True), name='dispatch')
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
        if 'email' in self.errors and 'already exists' in self.errors['email'][0]:
            errors = ["A user is already assigned with this email. Please use a different email address."]
            return errors


def check_email(request):
    form = account_forms.UserRegisterForm(request.GET)
    context = {
        'field': as_crispy_field(form['email']),
        'valid': not form['email'].errors
    }
    return render(request, 'partials/field.html', context)


def check_first_name(request):
    form = account_forms.UserRegisterForm(request.GET)
    context = {
        'field': as_crispy_field(form['first_name']),
        'valid': not form['first_name'].errors
    }
    return render(request, 'partials/field.html', context)


def check_last_name(request):
    form = account_forms.UserRegisterForm(request.GET)
    context = {
        'field': as_crispy_field(form['last_name']),
        'valid': not form['last_name'].errors
    }
    return render(request, 'partials/field.html', context)


def check_phone_number(request):
    form = account_forms.UserRegisterForm(request.GET)
    print(form)
    context = {
        'field': as_crispy_field(form['phone_number']),
        'valid': not form['phone_number'].errors
    }
    return render(request, 'partials/field.html', context)


def check_password_confirmation(request):
    form = account_forms.UserRegisterForm(request.GET)
    context = {
        'field': as_crispy_field(form['password2']),
        'valid': not form['password2'].errors
    }
    return render(request, 'partials/field.html', context)


# @method_decorator(cache_control(no_cache=True, must_revalidate=True, no_store=True), name='dispatch')
class HtmxLoginView(LoginView):
    template_name = 'account/guest/login.html'
    form_class = CustomLoginForm
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
            if core_views.check_is_user_last_login_was_up_to_yesterday(user=user):
                # it will be displayed if the last date for the change is different from today's date
                web_actions.save_three_user_graphs_as_png(user=user)
                if service_settings.GOOGLE_DRIVE_DAILY_DOWNLOAD:
                    data_management.update_files_from_google_drive()
                """
                Dataset and static images are updated daily, only when the date of the last update is different
                from today
                """
        except InvestorUser.DoesNotExist:
            pass
        return super().form_valid(form)


def check_login_email(request):
    form = CustomLoginForm(request.POST)

    User = CustomUser
    email = request.POST.get('login')
    print(email)
    # print(form.cleaned_data.get['login'])

    try:
        User.objects.get(email=email)
        valid = True
        print('valid true')
    except User.DoesNotExist:
        valid = False
    context = {
        'field': form['login'],

        'valid': valid,
    }
    return render(request, 'partials/email_validation.html', context)

    # return JsonResponse({'valid': valid})


def check_login_email_reset(request):
    form = CustomLoginForm(request.POST)

    User = CustomUser
    email = request.POST.get('email')
    print(email)
    # print(form.cleaned_data.get['login'])

    try:
        User.objects.get(email=email)
        valid = True
        print('valid true')
    except User.DoesNotExist:
        valid = False
    context = {
        'field': form['login'],

        'valid': valid,
    }
    return render(request, 'partials/email_validation.html', context)


def custom_login_view(request):
    if request.method == 'POST':
        form = CustomLoginForm(request, data=request.POST)
        if form.is_valid():
            # Authenticate user
            user = form.get_user()
            login(request, user)
            # Handle any additional logic
            return redirect('home')  # Redirect to a different page after login
    else:
        form = CustomLoginForm()
    return render(request, 'account/guest/login.html', {'form': form})


@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def logout_view(request):
    logout(request)
    context = {
        'title': "You Have Been Logged Out"
    }
    return render(request, 'account/authenticated/logout.html', context=context)


@cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required
@require_http_methods(["GET"])
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


@cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required
def profile_account(request):
    context = {
        'title': "Account Page"
    }
    return render(request, 'account/authenticated/profile_account.html', context=context)


@cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required
@require_http_methods(["GET", "POST"])
def profile_account_details(request):
    if request.method == 'GET':
        form: forms.ModelForm = account_forms.UpdateAccountDetailsForm(instance=request.user)
    elif request.method == 'POST':
        form: forms.ModelForm = account_forms.UpdateAccountDetailsForm(request.POST, instance=request.user)
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


@method_decorator(cache_control(no_cache=True, must_revalidate=True, no_store=True), name='dispatch')
class MyPasswordChangeForm(PasswordChangeView):
    form_class = account_forms.PasswordChangingForm
    template_name = 'account/authenticated/profile_account_password.html'
    success_url = reverse_lazy('profile_account')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Forgot Password'  # Add your desired value for the 'title' key here
        return context


@cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required
@require_http_methods(["GET", "POST"])
def profile_investor(request):
    stocks_symbols_data: dict[list] = data_management.get_stocks_from_json_file()
    styled_stocks_symbols_data: dict[list] = data_management.get_styled_stocks_symbols_data(
        stocks_symbols_data=stocks_symbols_data
    )
    is_form_filled = True

    try:
        investor_user: InvestorUser = get_object_or_404(InvestorUser, user=request.user)
        old_collection_number = investor_user.stocks_collection_number
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
                investments: Investment = Investment.objects.filter(investor_user=investor_user)
                if len(investments) > 0:
                    if int(old_collection_number) != int(
                            form.cleaned_data['stocks_collection_number']):
                        messege = update_data(form, investor_user, request, investments)
                        messages.warning(
                            request,
                            messege
                        )
                        return redirect('capital_market_algorithm_preferences_form')
                    else:
                        messages.warning(
                            request,
                            "You chose the same stocks' collection number you already had before.\n"
                        )
                        context = {
                            'form': form,
                            'is_form_filled': is_form_filled,
                            'stocks_symbols_data': json.dumps(styled_stocks_symbols_data),
                            'title': "Update Collections of Stocks' Details",
                        }
                        return render(request, 'account/authenticated/profile_investor.html', context=context)
                else:  # there are no investments
                    messege = update_data(form, investor_user, request, investments)
                    messages.warning(
                        request,
                        messege
                    )
                    return redirect('capital_market_algorithm_preferences_form')
            else:
                messages.info(
                    request,
                    'Your account details have been updated successfully.\n'
                    "No investments with your previous stocks' collection found, thus no stocks are affected."
                )
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
        'title': "Update Collections of Stocks' Details",
    }
    return render(request, 'account/authenticated/profile_investor.html', context=context)


# Currently irrelevant
@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def image_upload(request):
    if request.method == "POST" and request.FILES["image_file"]:
        image_file = request.FILES["image_file"]
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        image_url = fs.url(filename)
        print(image_url)
        return render(request, "upload.html", {
            "image_url": image_url
        })
    return render(request, "upload.html")


def update_data(form, investor_user, request, investments):
    questionnaire_a: QuestionnaireA = get_object_or_404(QuestionnaireA, user=request.user)
    questionnaire_b: QuestionnaireB = get_object_or_404(QuestionnaireB, user=request.user)
    (annual_max_loss, annual_returns, annual_sharpe, annual_volatility, daily_change, monthly_change,
     risk_level, sectors_names, sectors_weights, stocks_symbols, stocks_weights, total_change,
     portfolio) = web_actions.create_portfolio_and_get_data(
        answers_sum=questionnaire_b.answers_sum,
        stocks_collection_number=investor_user.stocks_collection_number,
        questionnaire_a=questionnaire_a,
    )
    affected_investments: int = 0
    if len(investments) > 0:
        # add "robot" investment as one investment with amount of total investments + profit
        data_management.changing_portfolio_investments_treatment_web(investor_user, portfolio, investments)
        # All investments.STATUS are changed to INACTIVE
        make_investments_inactive(investments=investments)

        if affected_investments == 0:
            message = 'Your account details have been updated successfully. \n' \
                      'You must complete the two forms so changes will be made over your' \
                      "investments and stocks' collection number. Otherwise, it won't change data."
        elif affected_investments == 1:
            message = 'Your account details have been updated successfully. \n' \
                      'A single investment is affected by this, and became inactive.\n' \
                      'You must complete the two forms so changes will be made over your' \
                      "investments and stocks' collection number. Otherwise, it won't change data."
        else:
            message = 'Your account details have been updated successfully. \n' \
                      f'{affected_investments} investments are affected by this, and became inactive.' \
                      'You must complete the two forms so changes will be made over your' \
                      "investments and stocks' collection number. Otherwise, it won't change data."

    else:
        message = 'Your account details have been updated successfully. \n' \
                  "No investments with your previous stocks' collection found, thus no stocks are affected."
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

    return message
