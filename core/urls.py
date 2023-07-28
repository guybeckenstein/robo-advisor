from django.urls import path
from . import views

urlpatterns = [
    path('', views.homepage, name='homepage'),
    path('about/', views.about, name='about'),
    path('services/', views.services, name='services'),
    path('form/1/', views.capital_market_algorithm_preferences_form, name='capital_market_algorithm_preferences_form'),
    path('form/2/', views.capital_market_investment_preferences_form, name='capital_market_investment_preferences_form'),
]
