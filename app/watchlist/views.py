import pandas as pd
from django import forms
from django.contrib.auth.decorators import login_required
from django.core.exceptions import BadRequest
from django.db.models import QuerySet
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from matplotlib import pyplot as plt

from service.config import settings
from service.util import data_management, research, helpers
from watchlist.forms import DiscoverStocksForm
from watchlist.models import TopStock


@login_required
@require_http_methods(["GET"])
def discover_stocks_form(request):
    if request.method == 'GET':
        form: forms.ModelForm = DiscoverStocksForm()
    else:
        raise BadRequest
    context = {
        'form': form,
        'is_chosen_stock_template': False,
        'title': 'Discover Stocks',
    }
    return render(request, 'watchlist/discover_stocks.html', context=context)


@login_required
@require_http_methods(["GET", "POST"])
def chosen_stock(request):
    if request.method == 'GET':
        form: forms.ModelForm = DiscoverStocksForm()
        # Form data
        data = request.GET
        ml_model = data.get('ml_model', None)
        if type(ml_model) == int or ml_model.isnumeric():
            ml_model = int(data.get('ml_model', None)) - 1
            ml_model = settings.MACHINE_LEARNING_MODEL[ml_model]
        description = data.get('symbol', None)
        symbol: str = helpers.get_symbol_by_description(description)
        start_date = data.get('start_date', None)
        end_date = data.get('end_date', None)
        # Run backend-service methods - it creates two different images (using Matplotlib)
        models_data: dict = data_management.get_models_data_from_collections_file()
        forecast_plt = research.forecast_specific_stock(
            stock=symbol,
            machine_learning_model=str(ml_model),
            models_data=models_data,
            num_of_years_history=None,
            start_date=start_date,
            end_date=end_date,
        )
        research.save_user_specific_stock(stock=symbol, operation='_forecast', plt_instance=forecast_plt)
        plt.clf()
        plt.cla()
        plt.close()
        bb_strategy_plt = research.plot_bb_strategy_stock(stock_name=symbol, start=start_date, end=end_date)
        research.save_user_specific_stock(stock=symbol, operation='_bb_strategy', plt_instance=bb_strategy_plt)
        plt.clf()
        plt.cla()
        plt.close()

        overview:str = f"https://finance.yahoo.com/quote/{symbol}/?p = {symbol}"
        conversation_link:str = f"https://finance.yahoo.com/quote/{symbol}/community?p = {symbol}"
        more_statistics:str = f"https://finance.yahoo.com/quote/{symbol}/key-statistics?p = {symbol}"

        # israeli stock
        if type(symbol) == int or symbol.isnumeric():
            num_of_digits = len(str(symbol))
            conversation_link = f"https://www.sponser.co.il/Tag.aspx?id={symbol}"
            if num_of_digits > 3:
                overview = f"https://market.tase.co.il/he/market_data/security/{symbol}/major_data"
                more_statistics = f"https://market.tase.co.il/he/market_data/security/{symbol}/statistics"
            else:
                overview = f"https://market.tase.co.il/he/market_data/index/{symbol}/major_data"
                more_statistics = f"https://market.tase.co.il/he/market_data/index/{symbol}/statistics"


    else:
        raise BadRequest
    context = {
        'form': form,
        'is_chosen_stock_template': True,
        'title': 'Discover Stocks',
        'symbol': symbol,
        'bb_strategy_img_name': f'{settings.RESEARCH_IMAGES}{symbol}_bb_strategy.png',
        'forecast_stock_img_name': f'{settings.RESEARCH_IMAGES}{symbol}_forecast.png',
        'more_statistics': more_statistics,
        'overview': overview,
        'conversation': conversation_link,

    }
    return render(request, 'watchlist/chosen_stock.html', context=context)


@login_required
def top_stocks(request):
    top_stocks: QuerySet[TopStock] = TopStock.objects.all()
    context = {
        'top_stocks': top_stocks,
        'title': 'Top Stocks',
    }
    return render(request, 'watchlist/top_stocks.html', context=context)
