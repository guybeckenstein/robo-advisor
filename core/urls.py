from django.urls import path
from . import views

urlpatterns = [
    path('', views.homepage, name='homepage'),
    path('about/', views.about, name='about'),
    path('services/', views.services, name='services'),
    path('form/', views.capital_market_form, name='capital_market_form'),
]
