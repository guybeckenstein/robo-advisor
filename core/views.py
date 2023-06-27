from crispy_forms.utils import render_crispy_form
from django.http import Http404, HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.template.context_processors import csrf

from core.forms import CapitalMarketForm
from core.models import TeamMember, Questionnaire


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
        questionnaire = Questionnaire.objects.get(user=request.user)
    except Questionnaire.DoesNotExist:
        questionnaire = None
    if request.method == 'GET':
        if questionnaire is None:
            context = {'title': 'Fill Form', 'form': CapitalMarketForm(form_type='create')}
            return render(request, 'core/capital_market_form_create.html', context=context)
        else:
            context = {'title': 'Update Filled Form', 'form': CapitalMarketForm(form_type='update', instance=questionnaire)}
            return render(request, 'core/capital_market_form_update.html', context=context)
    elif request.method == 'POST':
        if questionnaire is None:  # CREATE
            form = CapitalMarketForm(request.POST)
        else:  # UPDATE
            form = CapitalMarketForm(request.POST, instance=questionnaire)

        if form.is_valid():  # CREATE and UPDATE
            form.instance.user = request.user
            form.save()
            return redirect('homepage')
        else:  # CREATE and UPDATE
            context = {'form': form}
            ctx = {}
            ctx.update(csrf(request))
            form_html = render_crispy_form(form=context['form'], context=ctx)
            return HttpResponse(form_html)
    else:
        raise Http404
