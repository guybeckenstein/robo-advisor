from numbers import Number

from crispy_forms.utils import render_crispy_form
from django.contrib import messages
from django.db import transaction
from django.http import Http404, HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.template.context_processors import csrf

from service.util.web_actions import save_three_user_graphs_as_png
from service.util import data_management
from service.util.data_management import create_new_user_portfolio
from core.forms import AlgorithmPreferencesForm, InvestmentPreferencesForm, AdministrativeToolsForm
from core.models import TeamMember, QuestionnaireA, QuestionnaireB
from accounts.models import InvestorUser


def homepage(request):
    context = {'team_members': TeamMember.objects.all()}
    return render(request, 'core/homepage.html', context=context)


def about(request):
    return render(request, 'core/about.html', context={'title': 'About Us'})


@login_required
def administrative_tools_form(request):
    if request.user.is_superuser is False:
        raise Http404
    if request.method == 'GET':
        models_data: dict[Number] = data_management.get_models_data_from_collections_file()
        context = {
            'title': 'Administrative Tools',
            'form': AdministrativeToolsForm(initial=models_data),
        }
        return render(request, 'core/form.html', context=context)
    elif request.method == 'POST':
        form = AdministrativeToolsForm(request.POST)
        if form.is_valid():  # CREATE and UPDATE
            # Access cleaned data from the form
            with transaction.atomic():
                num_por_simulation = form.cleaned_data['num_por_simulation']
                min_num_por_simulation = form.cleaned_data['min_num_por_simulation']
                record_percent_to_predict = form.cleaned_data['record_percent_to_predict']
                test_size_machine_learning = form.cleaned_data['test_size_machine_learning']
                selected_ml_model_for_build = form.cleaned_data['selected_ml_model_for_build']
                gini_v_value = form.cleaned_data['gini_v_value']
                data_management.update_models_data_settings(
                    num_por_simulation=num_por_simulation,
                    min_num_por_simulation=min_num_por_simulation,
                    record_percent_to_predict=record_percent_to_predict,
                    test_size_machine_learning=test_size_machine_learning,
                    selected_ml_model_for_build=selected_ml_model_for_build,
                    gini_v_value=gini_v_value,
                )
            messages.success(request, message="Successfully updated models' data.")
            return redirect('administrative_tools_form')
        else:  # CREATE and UPDATE
            context = {
                'title': 'Administrative Tools',
                'form': form,
            }
            ctx = {}
            ctx.update(csrf(request))
            form_html = render_crispy_form(form=context['form'], context=ctx)
            return HttpResponse(form_html)
    else:
        raise Http404


@login_required
def capital_market_algorithm_preferences_form(request):
    try:
        preferences = QuestionnaireA.objects.get(user=request.user)
    except QuestionnaireA.DoesNotExist:
        preferences = None
    if request.method == 'GET':
        if preferences is None:  # CREATE
            context = {
                'title': 'Fill Form',
                'form': AlgorithmPreferencesForm(form_type='create')
            }
            return render(request, 'core/form.html', context=context)
        else:  # UPDATE
            context = {
                'title': 'Update Filled Form',
                'form': AlgorithmPreferencesForm(form_type='update', instance=preferences)
            }
            return render(request, 'core/form.html', context=context)
    elif request.method == 'POST':
        if preferences is None:  # CREATE
            form = AlgorithmPreferencesForm(request.POST)
        else:  # UPDATE
            form = AlgorithmPreferencesForm(request.POST, instance=preferences)

        if form.is_valid():  # CREATE and UPDATE
            form.instance.user = request.user
            form.save()
            return redirect('capital_market_investment_preferences_form')
        else:  # CREATE and UPDATE
            context = {
                'title': 'Update Filled Form',
                'form': form,
            }
            ctx = {}
            ctx.update(csrf(request))
            form_html = render_crispy_form(form=context['form'], context=ctx)
            return HttpResponse(form_html)
    else:
        raise Http404


@login_required
def capital_market_investment_preferences_form(request):
    try:
        user_preferences_instance = get_object_or_404(QuestionnaireA, user=request.user)
    except Http404:
        return HttpResponse("You must have an instance of QuestionnaireA to fill this form.", status=404)

    # Each user fills this form, and it gets a rating from 3 to 9
    try:
        questionnaire = QuestionnaireB.objects.get(user=request.user)
    except QuestionnaireB.DoesNotExist:
        questionnaire = None
        # Retrieve the UserPreferencesA instance for the current user
    if request.method == 'GET':
        try:
            investor_user: InvestorUser = InvestorUser.objects.get(user=request.user)
            collections_number: str = investor_user.stocks_collection_number
        except InvestorUser.DoesNotExist:
            collections_number: str = '1'
        if questionnaire is None:  # CREATE
            context = {
                'title': 'Fill Form',
                'form': InvestmentPreferencesForm(
                    form_type='create',
                    user_preferences_instance=user_preferences_instance,
                    collections_number=collections_number,
                )
            }
            return render(request, 'core/form.html', context=context)
        else:  # UPDATE
            context = {
                'title': 'Update Filled Form',
                'form': InvestmentPreferencesForm(
                    form_type='update',
                    instance=questionnaire,
                    user_preferences_instance=user_preferences_instance,
                    collections_number=collections_number,
                )
            }
            return render(request, 'core/form.html', context=context)

    elif request.method == 'POST':
        if questionnaire is None:  # CREATE
            form = InvestmentPreferencesForm(
                request.POST,
                user_preferences_instance=user_preferences_instance
            )
        else:  # UPDATE
            form = InvestmentPreferencesForm(
                request.POST,
                user_preferences_instance=user_preferences_instance,
                instance=questionnaire
            )
        if form.is_valid():  # CREATE and UPDATE
            # DEBUGGING, without this the code won't work
            print("Form errors:", form.errors)
            # Sum answers' values
            answer_1_value = int(form.cleaned_data['answer_1'])
            answer_2_value = int(form.cleaned_data['answer_2'])
            answer_3_value = int(form.cleaned_data['answer_3'])
            answers_sum = answer_1_value + answer_2_value + answer_3_value
            # Form instance
            form.instance.user = request.user
            form.instance.answers_sum = answers_sum
            form.save()

            # Backend
            stocks_collection_number: str = "1"
            risk_level = data_management.get_level_of_risk_by_score(answers_sum)
            stocks_symbols = data_management.get_stocks_symbols_from_collection(stocks_collection_number)
            tables = data_management.get_extended_data_from_db(
                stocks_symbols=stocks_symbols,
                is_machine_learning=user_preferences_instance.ml_answer,
                model_option=user_preferences_instance.model_answer,
                stocks_collection_number=stocks_collection_number,
                mode='regular'
            )
            investment_amount = 0  # TODO: THINK ABOUT THIS (YARDEN AND GUY)
            portfolio = create_new_user_portfolio(
                stocks_symbols=stocks_symbols,
                investment_amount=investment_amount,
                is_machine_learning=user_preferences_instance.ml_answer,
                model_option=user_preferences_instance.model_answer,
                risk_level=risk_level,
                extended_data_from_db=tables,
            )

            _, _, stocks_symbols, sectors_names, sectors_weights, stocks_weights, annual_returns, annual_max_loss, \
                annual_volatility, annual_sharpe, total_change, monthly_change, daily_change, selected_model, \
                machine_learning_opt = portfolio.get_portfolio_data()
            try:
                investor_user = InvestorUser.objects.get(user=request.user)
                # If we get here, it means that the user is on UPDATE form (there is InvestorUser instance)
                investor_user.risk_level = risk_level
                investor_user.stocks_symbols = stocks_symbols
                investor_user.stocks_weights = stocks_weights
                investor_user.sectors_names = sectors_names
                investor_user.sectors_weights = sectors_weights
                investor_user.annual_returns = annual_returns
                investor_user.annual_max_loss = annual_max_loss
                investor_user.annual_volatility = annual_volatility
                investor_user.annual_sharpe = annual_sharpe
                investor_user.total_change = total_change
                investor_user.monthly_change = monthly_change
                investor_user.daily_change = daily_change
                # TODO - maybe add more fields later
            except InvestorUser.DoesNotExist:
                # If we get here, it means that the user is on CREATE form (no InvestorUser instance)
                InvestorUser.objects.create(
                    user=request.user,
                    risk_level=risk_level,
                    starting_investment_amount=0,
                    stocks_symbols=stocks_symbols,
                    stocks_weights=stocks_weights,
                    sectors_names=sectors_names,
                    sectors_weights=sectors_weights,
                    annual_returns=annual_returns,
                    annual_max_loss=annual_max_loss,
                    annual_volatility=annual_volatility,
                    annual_sharpe=annual_sharpe,
                    total_change=total_change,
                    monthly_change=monthly_change,
                    daily_change=daily_change,
                    # TODO - maybe add more fields later
                )
            # Frontend
            save_three_user_graphs_as_png(request)
            return redirect('homepage')

        else:  # CREATE and UPDATE
            context = {
                'form': form,
            }
            ctx = {}
            ctx.update(csrf(request))
            form_html = render_crispy_form(form=context['form'], context=ctx)
            return HttpResponse(form_html)
    else:
        raise Http404