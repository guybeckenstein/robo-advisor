from crispy_forms.templatetags.crispy_forms_filters import as_crispy_field
from django.contrib.auth.decorators import login_required
from django.core.exceptions import BadRequest
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.db.models import QuerySet
from django.http import Http404
from django.shortcuts import render, redirect, get_object_or_404

from accounts.models import InvestorUser
from django.views.decorators.http import require_http_methods

from investment.forms import InvestmentsHistoryForm
from investment.models import Investment

from service.config import settings
from service.util import data_management, web_actions


# Investment
@login_required
@require_http_methods(["GET"])
def investments_list_view(request):
    is_form_filled: bool = _check_if_preferences_form_is_filled(request)
    if is_form_filled:
        investments: QuerySet[Investment] = _investments_list_view(request)
        context: dict = {
            'investments': investments,
            'is_form_filled': is_form_filled,
            'title': 'Investments',
            'is_investments_view': False,
            'form': InvestmentsHistoryForm,
        }
    else:
        context = {
            'is_form_filled': is_form_filled,
            'title': 'Investments',
            'form': InvestmentsHistoryForm,
        }
    return render(request, 'investment/my_investments_history.html', context=context)


@login_required
@require_http_methods(["GET", "POST"])
def add_investment_view(request):
    is_form_filled: bool = _check_if_preferences_form_is_filled(request)
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
            Investment.objects.create(investor_user=investor_user, amount=amount,
                                      stocks_collection_number=investor_user.stocks_collection_number)
            investor_user.total_investment_amount += amount
            investor_user.save()
            # save report according to a new investment
            stocks_weights = investor_user.stocks_weights
            stocks_symbols = investor_user.stocks_symbols

            # Save image
            data_management.view_investment_report(str(request.user.id), amount, stocks_weights, stocks_symbols)

            # send report to user in email
            subject: str = 'Investment Report From RoboAdvisor'
            message: str = (
                'Here is your investment report.\n'
                f'You have invested {amount} dollars.\n\n\n'
                'Sincerely yours,\n'
                'RoboAdvisor team'
            )
            recipient_list: list[str] = [request.user.email]
            attachment_path: str = f'{settings.USER_IMAGES}{str(request.user.id)}/Investment Report.png'
            web_actions.send_email(
                subject=subject, message=message, recipient_list=recipient_list, attachment_path=attachment_path
            )

            return redirect('my_investments_history')
        else:
            raise ValueError('Invalid amount value')
    else:
        raise BadRequest


@require_http_methods(["GET"])
def _investments_list_view(request) -> QuerySet[Investment]:
    page = request.GET.get("page", None)
    # Assuming you want to retrieve the single InvestorUser object for the current user
    investor_user = get_object_or_404(InvestorUser, user=request.user)

    # Now, you can use the retrieved investor_user to filter the investments
    investments = Investment.objects.filter(investor_user=investor_user, mode=Investment.Mode.USER)

    if request.method == 'GET':
        paginator = Paginator(investments, per_page=3)
        try:
            investments = paginator.page(page)
        except PageNotAnInteger:
            investments = paginator.page(1)
        except EmptyPage:
            investments = paginator.page(paginator.num_pages)
    return investments


# Investment Portfolio
@login_required
def investment_main(request):
    context = {
        'title': 'Investments',
    }
    return render(request, 'investment/investments_main.html', context=context)


@login_required
def profile_portfolio(request):
    is_form_filled: bool = _check_if_preferences_form_is_filled(request)
    context = {
        'user': request.user,
        'is_form_filled': is_form_filled,
        'title': 'Profile Portfolio',

    }
    return render(request, 'investment/profile_portfolio.html', context=context)


@login_required
def _check_if_preferences_form_is_filled(request):
    is_form_filled = True
    try:
        get_object_or_404(InvestorUser, user=request.user)
    except Http404:
        is_form_filled = False
    return is_form_filled


def check_positive_number(request):
    form = InvestmentsHistoryForm(request.GET)
    context = {
        'field': as_crispy_field(form['amount']),
        'valid': not form['amount'].errors
    }
    return render(request, 'partials/field.html', context)
