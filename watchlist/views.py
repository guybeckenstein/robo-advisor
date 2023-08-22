import pandas as pd
from django import forms
from django.contrib.auth.decorators import login_required
from django.core.exceptions import BadRequest
from django.db.models import QuerySet
from django.shortcuts import render
from matplotlib import pyplot as plt

from service.config import settings
from service.util import data_management, research, helpers
from watchlist.forms import DiscoverStocksForm
from watchlist.models import TopStock


@login_required
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
        plt.close()
        bb_strategy_plt = research.plot_bb_strategy_stock(stock_name=symbol, start=start_date, end=end_date)
        research.save_user_specific_stock(stock=symbol, operation='_bb_strategy', plt_instance=bb_strategy_plt)
        plt.close()
    else:
        raise BadRequest
    context = {
        'form': form,
        'is_chosen_stock_template': True,
        'title': 'Discover Stocks',
        'symbol': symbol,
        'bb_strategy_img_name': f'{settings.RESEARCH_IMAGES}{symbol}_bb_strategy.png',
        'forecast_stock_img_name': f'{settings.RESEARCH_IMAGES}{symbol}_forecast.png',
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