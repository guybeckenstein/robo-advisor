from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required

from .forms import UserRegisterForm


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
def profile_main(request):
    context = {
        'user': request.user
    }
    return render(request, 'user/profile_main.html', context=context)


@login_required
def profile_user(request):
    context = {
        'user': request.user
    }
    return render(request, 'user/profile_user.html', context=context)


@login_required
def profile_user(request):
    context = {
        'user': request.user
    }
    return render(request, 'user/profile_user.html', context=context)


@login_required
def profile_investor(request):
    context = {
        'user': request.user
    }
    return render(request, 'user/profile_investor.html', context=context)


@login_required
def profile_portfolio(request):
    context = {
        'user': request.user
    }
    return render(request, 'user/profile_portfolio.html', context=context)
