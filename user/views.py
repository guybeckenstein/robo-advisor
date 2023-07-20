from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import UserRegisterForm, UserUpdateForm


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
    if request.method == 'POST':
        u_form = UserUpdateForm(request.POST, instance=request.user)
        if u_form.is_valid():
            u_form.save()
            messages.success(request, 'Your account has been updated!')
            return redirect('profile')
    else:
        u_form = UserUpdateForm(instance=request.user)

    context = {
        'u_form': u_form,
    }

    return render(request, 'user/profile.html', context)
