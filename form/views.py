from django.shortcuts import render
from django.contrib.auth.decorators import login_required


def homepage(request):
    return render(request, 'html/index.html', {})


def about(request):
    return render(request, 'html/about.html', {'title': 'About Us'})


def contact(request):
    return render(request, 'html/contact.html', {'title': 'Contact Us'})


@login_required
def services(request):
    return render(request, 'html/services.html', {'title': 'Services'})


#@login_required
def form(request):
    return render(request, 'html/form.html', {'title': 'Form'})
