from django.contrib.auth.decorators import login_required
from django.http import Http404
from django.shortcuts import render, get_object_or_404

from accounts.models import InvestorUser


def check_if_preferences_form_is_filled(request):
    is_form_filled = True
    try:
        get_object_or_404(InvestorUser, user=request.user)
    except Http404:
        is_form_filled = False
    return is_form_filled


@login_required
def profile_portfolio(request):
    is_form_filled: bool = check_if_preferences_form_is_filled(request)
    context = {
        'user': request.user,
        'is_form_filled': is_form_filled,
        'title': 'Profile Portfolio',
    }
    return render(request, 'investment_portfolio/profile_portfolio.html', context=context)


@login_required
def investment_main(request):
    context = {
        'title': 'Investments',
    }
    return render(request, 'investment_portfolio/investments_main.html', context=context)


@login_required
def discover_stocks(request):
    context = {
        'title': 'Discover Stocks',
    }
    return render(request, 'investment_portfolio/discover_stocks.html', context=context)
