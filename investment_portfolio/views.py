from django.contrib.auth.decorators import login_required
from django.http import Http404
from django.shortcuts import render, get_object_or_404

from user.models import InvestorUser


@login_required
def profile_portfolio(request):
    is_form_filled = True
    try:
        get_object_or_404(InvestorUser, user=request.user)
    except Http404:
        is_form_filled = False
    context = {
        'user': request.user,
        'is_form_filled': is_form_filled,
    }
    return render(request, 'investment_portfolio/profile_portfolio.html', context=context)


@login_required
def investment_main(request):
    context = {
        'title': 'Investments'
    }
    return render(request, 'investment_portfolio/investments_main.html', context=context)


@login_required
def my_investments_history(request):
    context = {
        'title': "My Investments' History"
    }
    return render(request, 'investment_portfolio/my_investments_history.html', context=context)


@login_required
def discover_stocks(request):
    context = {
        'title': 'Discover Stocks'
    }
    return render(request, 'investment_portfolio/discover_stocks.html', context=context)


@login_required
def sell_stocks(request):
    context = {
        'title': 'Sell Stocks'
    }
    return render(request, 'investment_portfolio/sell_stocks.html', context=context)
