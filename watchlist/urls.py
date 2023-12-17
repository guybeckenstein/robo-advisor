from django.urls import path, include

from watchlist import views

urlpatterns = [
    path('discover-stocks/', views.discover_stocks_form, name='discover_stocks_form'),
    path('chosen-stock/', views.chosen_stock, name='chosen_stock'),
    path('top-stocks/', views.top_stocks, name='top_stocks'),
    # Django REST Framework APIs
    path('api/', include('watchlist.api.urls', 'watchlist_api')),
]
