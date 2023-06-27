from django.shortcuts import render
from django.contrib.auth.decorators import login_required

from core.models import TeamMember


def homepage(request):
    context = {'team_members': TeamMember.objects.all()}
    return render(request, 'core/homepage.html', context=context)


def about(request):
    return render(request, 'core/about.html', context={'title': 'About Us'})


@login_required
def services(request):
    return render(request, 'core/services.html', context={'title': 'Services'})


@login_required
def form(request):
    return render(request, 'core/capital_market_form.html', context={'title': 'Capital Market Form'})
