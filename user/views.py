from crispy_forms.utils import render_crispy_form
from django.http import HttpResponseNotFound, Http404, HttpResponse
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.template.context_processors import csrf

from .forms import UserRegisterForm, UserUpdateForm, UserPreferencesForm
from .models import UserPreferences


def register(request):
    if request.user.is_authenticated:
        return redirect('homepage')
    elif request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account " {username} " has been created! You are now able to log in')
            return redirect('login')
    else:
        form = UserRegisterForm()

    context = {
        'form': form,
    }
    return render(request, 'user/register.html', context=context)


def login(request):
    # TODO: make this function prevent logged-in user from entering 'login' page
    if request.user.is_authenticated:
        return redirect('homepage')
    else:
        return redirect('login')


def logout(request):
    # TODO: make this function prevent guests from entering 'logout' page
    if request.user.is_authenticated:
        return redirect('logout')
    else:
        return redirect('homepage')


@login_required
def profile(request):
    # Each user fills this form, and it gets a rating from 3 to 9
    try:
        preferences = UserPreferences.objects.get(user=request.user)
    except UserPreferences.DoesNotExist:
        preferences = None
    if request.method == 'GET':
        if preferences is None:
            context = {'title': 'Fill Form', 'preferences_form': UserPreferencesForm(form_type='create')}
            return render(request, 'core/capital_market_form_create.html', context=context)
        else:
            context = {
                'title': 'Update Filled Form',
                'preferences_form': UserPreferencesForm(form_type='update', instance=preferences)
            }
            return render(request, 'core/capital_market_form_update.html', context=context)
    elif request.method == 'POST':
        if preferences is None:  # CREATE
            preferences_form = UserPreferencesForm(request.POST)
            # TODO: connect to relevant part in the logic Backend
        else:  # UPDATE
            preferences_form = UserPreferencesForm(request.POST, instance=preferences)
            # TODO: connect to relevant part in the logic Backend

        if preferences_form.is_valid():  # CREATE and UPDATE
            preferences_form.instance.user = request.user
            preferences_form.save()
            return redirect('homepage')
        else:  # CREATE and UPDATE
            context = {
                'preferences_form': preferences_form,
            }
            ctx = {}
            ctx.update(csrf(request))
            form_html = render_crispy_form(form=context['preferences_form'], context=ctx)
            return HttpResponse(form_html)
    else:
        raise Http404
