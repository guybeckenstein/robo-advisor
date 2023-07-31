from django.urls import path
from . import views

urlpatterns = [
    path('profile/portfolio/', views.profile_portfolio, name='profile_portfolio'),
    path('investment/', views.investment_main, name='investments_main'),
    path('investment/my-investments-history/', views.my_investments_history, name='my_investments_history'),
    path('investment/discover-stocks/', views.discover_stocks, name='discover_stocks'),
    path('investment/sell-stocks/', views.sell_stocks, name='sell_stocks'),
]
