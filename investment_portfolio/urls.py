from django.urls import path
from . import views

urlpatterns = [
    path('profile/portfolio/', views.profile_portfolio, name='profile_portfolio'),
    path('investment/', views.investment_main, name='investments_main'),
]
