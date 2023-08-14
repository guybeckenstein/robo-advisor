from django.contrib.auth.decorators import login_required
from django.core.exceptions import BadRequest
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.db.models import QuerySet
from django.shortcuts import render, redirect, get_object_or_404

from accounts.models import InvestorUser
from investment.models import Investment


@login_required
def investments_list_view(request):
    investments: QuerySet[Investment] = _investments_list_view(request)
    context: dict = {
        'investments': investments,
        'title': 'Investments',
        'is_investments_view': False,
    }
    return render(request, 'investment/my_investments_history.html', context=context)

@login_required
def add_investment_view(request):
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
            investor_user = get_object_or_404(InvestorUser, user=request.user)
            Investment.objects.create(investor_user=investor_user, amount=amount)
            return redirect('my_investments_history')
        else:
            raise ValueError('Invalid amount value')
    else:
        raise BadRequest

def _investments_list_view(request) -> QuerySet[Investment]:
    page = request.GET.get("page", None)
    investments = Investment.objects.all()

    if request.method == 'GET':
        paginator = Paginator(investments, per_page=3)
        try:
            investments = paginator.page(page)
        except PageNotAnInteger:
            investments = paginator.page(1)
        except EmptyPage:
            investments = paginator.page(paginator.num_pages)
    return investments
