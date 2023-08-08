from crispy_forms.utils import render_crispy_form
from django.http import Http404, HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.template.context_processors import csrf

from service.util.web_actions import save_three_user_graphs_as_png
from service.util import data_management, settings
from service.util.data_management import create_new_user_portfolio
from core.forms import AlgorithmPreferencesForm, InvestmentPreferencesForm
from core.models import TeamMember, QuestionnaireA, QuestionnaireB
from accounts.models import InvestorUser


def homepage(request):
    context = {'team_members': TeamMember.objects.all()}
    return render(request, 'core/homepage.html', context=context)


def about(request):
    return render(request, 'core/about.html', context={'title': 'About Us'})


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
            return render(request, 'core/capital_market_preferences_form.html', context=context)
        else:  # UPDATE
            context = {
                'title': 'Update Filled Form',
                'form': AlgorithmPreferencesForm(form_type='update', instance=preferences)
            }
            return render(request, 'core/capital_market_preferences_form.html', context=context)
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
                'form': form,
            }
            ctx = {}
            ctx.update(csrf(request))
            form_html = render_crispy_form(form=context['form'], context=ctx)
            return HttpResponse(form_html)
    else:
        raise Http404


@login_required
def capital_market_investment_preferences_form(request, **kwargs):
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
        if questionnaire is None:  # CREATE
            context = {
                'title': 'Fill Form',
                'form': InvestmentPreferencesForm(
                    form_type='create', user_preferences_instance=user_preferences_instance
                )
            }
            return render(request, 'core/capital_market_preferences_form.html', context=context)
        else:  # UPDATE
            context = {
                'title': 'Update Filled Form',
                'form': InvestmentPreferencesForm(
                    form_type='update',
                    instance=questionnaire,
                    user_preferences_instance=user_preferences_instance
                )
            }
            return render(request, 'core/capital_market_preferences_form.html', context=context)

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
            risk_level = data_management.get_level_of_risk_by_score(answers_sum)
            tables = data_management.get_extended_data_from_db(
                stocks_symbols=settings.STOCKS_SYMBOLS,
                is_machine_learning=user_preferences_instance.ml_answer,
                model_option=user_preferences_instance.model_answer,
                mode='regular'
            )
            investment_amount = 0  # TODO: THINK ABOUT THIS (YARDEN AND GUY)
            portfolio = create_new_user_portfolio(
                stocks_symbols=settings.STOCKS_SYMBOLS,
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


def convert_type_list_to_str_list(input_list: list):
    # Convert all values within settings.STOCKS_SYMBOLS to `str`. Some values are `int`
    str_list = []
    for value in input_list:
        str_list.append(str(value))
    return str_list
