from django.urls import path
from watchlist import views

urlpatterns = [
    path('investment/discover-stocks/', views.discover_stocks_form, name='discover_stocks_form'),
    path('investment/chosen-stock/', views.chosen_stock, name='chosen_stock'),
    path('investment/top-stocks/', views.top_stocks, name='top_stocks'),
]
