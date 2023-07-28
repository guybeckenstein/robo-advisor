from crispy_forms.utils import render_crispy_form
from django.http import Http404, HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.template.context_processors import csrf

from backend_api.util import manage_data
from core.forms import AlgorithmPreferencesForm, InvestmentPreferencesForm
from core.models import TeamMember, QuestionnaireA, QuestionnaireB


def homepage(request):
    context = {'team_members': TeamMember.objects.all()}
    return render(request, 'core/homepage.html', context=context)


def about(request):
    return render(request, 'core/about.html', context={'title': 'About Us'})


@login_required
def services(request):
    return render(request, 'core/services.html', context={'title': 'Services'})


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
            return render(request, 'core/preferences_form_create.html', context=context)
        else:  # UPDATE
            context = {
                'title': 'Update Filled Form',
                'form': AlgorithmPreferencesForm(form_type='update', instance=preferences)
            }
            return render(request, 'core/preferences_form_update.html', context=context)
    elif request.method == 'POST':
        if preferences is None:  # CREATE
            form = AlgorithmPreferencesForm(request.POST)
            # TODO: connect to relevant part in the logic Backend
        else:  # UPDATE
            form = AlgorithmPreferencesForm(request.POST, instance=preferences)
            # TODO: connect to relevant part in the logic Backend

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
        if questionnaire is None:  # CREATE
            context = {
                'title': 'Fill Form',
                'form': InvestmentPreferencesForm(form_type='create', user_preferences_instance=user_preferences_instance)
            }
            return render(request, 'core/capital_market_form_create.html', context=context)
        else:  # UPDATE
            context = {
                'title': 'Update Filled Form',
                'form': InvestmentPreferencesForm(
                    form_type='update',
                    instance=questionnaire,
                    user_preferences_instance=user_preferences_instance
                )
            }
            return render(request, 'core/capital_market_form_update.html', context=context)
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
            level_of_risk = manage_data.get_level_of_risk_by_score(answers_sum)
            # TODO: create new user instance in the database, with the following (lines 70-74) parameters
            # new_user = manage_data.create_new_user(
            #     login_name, settings.stocks_symbols, investment_amount, machine_learning_opt, model_option,
            #     level_of_risk, sectors_data, sectors, closing_prices_table, three_best_portfolios, pct_change_table
            # )
            # new_user.update_json_file("backend_api/DB/users")
            # Frontend
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
