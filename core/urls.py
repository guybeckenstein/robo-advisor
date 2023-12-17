from django.urls import path, include
from core import views

urlpatterns = [
    path('', views.homepage, name='homepage'),
    path('about/', views.about, name='about'),
    path('administrative-tools/', views.administrative_tools_form, name='administrative_tools_form'),
    path('form/', include([
        path('1/', views.capital_market_algorithm_preferences_form, name='capital_market_algorithm_preferences_form'),
        path('2/', views.capital_market_investment_preferences_form, name='capital_market_investment_preferences_form'),
    ])),
    # Django REST Framework APIs
    path('api/', include('core.api.urls', 'core_api')),
]
