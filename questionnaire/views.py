from django.shortcuts import render
from django.contrib.auth.decorators import login_required


def homepage(request):
    return render(request, 'questionnaire/index.html', {})


def about(request):
    return render(request, 'questionnaire/about.html', {'title': 'About Us'})


@login_required
def services(request):
    return render(request, 'questionnaire/services.html', {'title': 'Services'})


@login_required
def form(request):
    return render(request, 'questionnaire/form.html', {'title': 'Form'})
