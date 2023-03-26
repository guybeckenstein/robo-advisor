from django.shortcuts import render
# from django.shortcuts import redirect
# from .models import Message
# from .forms import MessageForm


def homepage(request):
    return render(request, 'html/index.html', {})


def about(request):
    return render(request, 'html/about.html')


def info(request):
    return render(request, 'html/info.html')


def services(request):
    return render(request, 'html/services.html')


def form(request):
    return render(request, 'html/form.html')


def contact(request):
    return render(request, 'html/contact.html')
