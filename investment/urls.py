from django.urls import path

from investment import views

urlpatterns = [
    path('investment/history/', views.investments_list_view, name='my_investments_history'),
    path('investment/add/', views.add_investment_view, name='add_investment'),
    # path('investment/history/', views.investments_list_view, name='load_more_investments'),
]
