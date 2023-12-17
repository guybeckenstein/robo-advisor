from django.urls import path, include
from investment import views

urlpatterns = [
    # Investment
    path('my-history/', views.investments_list_view, name='my_investments_history'),
    path('add/', views.add_investment_view, name='add_investment'),
    # Investment Portfolio
    path('profile/portfolio/', views.profile_portfolio, name='profile_portfolio'),
    path('', views.investment_main, name='investments_main'),
    path('check_positive_amount/', views.check_positive_number, name='check_amount'),
    # Django REST Framework APIs
    path('api/', include('investment.api.urls', 'investment_api')),
]
