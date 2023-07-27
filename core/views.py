from crispy_forms.utils import render_crispy_form
from django.http import Http404, HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.template.context_processors import csrf

from backend_api.util import manageData
from core.forms import CapitalMarketForm
from core.models import TeamMember, Questionnaire
from user.models import UserPreferences


def homepage(request):
    context = {'team_members': TeamMember.objects.all()}
    return render(request, 'core/homepage.html', context=context)


def about(request):
    return render(request, 'core/about.html', context={'title': 'About Us'})


@login_required
def services(request):
    return render(request, 'core/services.html', context={'title': 'Services'})


@login_required
def capital_market_form(request):
    try:
        user_preferences_instance = UserPreferences.objects.get(user=request.user)
    except UserPreferences.DoesNotExist:
        raise Http404

    # Each user fills this form, and it gets a rating from 3 to 9
    try:
        questionnaire = Questionnaire.objects.get(user=request.user)
    except Questionnaire.DoesNotExist:
        questionnaire = None
        # Retrieve the UserPreferences instance for the current user
    if request.method == 'GET':
        if questionnaire is None:
            context = {'title': 'Fill Form', 'form': CapitalMarketForm(form_type='create', user_preferences_instance=user_preferences_instance)}
            return render(request, 'core/capital_market_form_create.html', context=context)
        else:
            context = {
                'title': 'Update Filled Form',
                'form': CapitalMarketForm(form_type='update', instance=questionnaire, user_preferences_instance=user_preferences_instance)
            }
            return render(request, 'core/capital_market_form_update.html', context=context)
    elif request.method == 'POST':
        if questionnaire is None:  # CREATE
            form = CapitalMarketForm(request.POST)
        else:  # UPDATE
            form = CapitalMarketForm(request.POST, instance=questionnaire)
        # DEBUGGING, without this the code won't work
        print("Form errors:", form.errors)
        # Sum answers' values
        answer_1_value = int(form.cleaned_data['answer_1'])
        answer_2_value = int(form.cleaned_data['answer_2'])
        answer_3_value = int(form.cleaned_data['answer_3'])
        answers_sum = answer_1_value + answer_2_value + answer_3_value
        if form.is_valid():  # CREATE and UPDATE
            # Form instance
            form.instance.user = request.user
            form.instance.answers_sum = answers_sum
            form.save()
            # Backend
            levelOfRisk = manageData.getLevelOfRiskByScore(answers_sum)
            # TODO: create new user instance in the database, with the following (lines 70-74) parameters
            # newUser = manageData.createsNewUser(loginName, setting.stocksSymbols,
            #                                     investmentAmount, machineLearningOpt, modelOption, levelOfRisk,
            #                                     sectorsData, sectorsList,
            #                                     closingPricesTable, threeBestPortfolios, pctChangeTable)
            # newUser.updateJsonFile("backend_api/DB/users")
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
