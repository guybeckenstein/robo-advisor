from django.contrib.auth.decorators import login_required
from django.core.exceptions import BadRequest
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.db.models import QuerySet
from django.http import Http404
from django.shortcuts import render, redirect, get_object_or_404

from accounts.models import InvestorUser
from investment.models import Investment
from investment_portfolio.views import check_if_preferences_form_is_filled


@login_required
def investments_list_view(request):
    is_form_filled: bool = check_if_preferences_form_is_filled(request)
    if is_form_filled:
        investments: QuerySet[Investment] = _investments_list_view(request)
        context: dict = {
            'investments': investments,
            'is_form_filled': is_form_filled,
            'title': 'Investments',
            'is_investments_view': False,
        }
    else:
        context: dict = {
            'is_form_filled': is_form_filled,
            'title': 'Investments',
        }
    return render(request, 'investment/my_investments_history.html', context=context)

@login_required
def add_investment_view(request):
    is_form_filled: bool = check_if_preferences_form_is_filled(request)
    if is_form_filled is False:
        raise Http404
    investments: QuerySet[Investment] = _investments_list_view(request)
    context: dict = {
        'investments': investments,
        'title': 'Investments',
        'is_investments_view': True,
    }
    if request.method == 'GET':
        return render(request, 'investment/add_investment.html', context=context)
    elif request.method == 'POST':
        amount: int = int(request.POST.get('amount', -1))
        if amount > 0:
            investor_user: InvestorUser = get_object_or_404(InvestorUser, user=request.user)
            Investment.objects.create(investor_user=investor_user, amount=amount)
            investor_user.total_investment_amount += amount
            investor_user.save()
            return redirect('my_investments_history')
        else:
            raise ValueError('Invalid amount value')
    else:
        raise BadRequest

def _investments_list_view(request) -> QuerySet[Investment]:
    page = request.GET.get("page", None)
    investments = Investment.objects.filter(mode=Investment.Mode.USER)

    if request.method == 'GET':
        paginator = Paginator(investments, per_page=3)
        try:
            investments = paginator.page(page)
        except PageNotAnInteger:
            investments = paginator.page(1)
        except EmptyPage:
            investments = paginator.page(paginator.num_pages)
    return investments
