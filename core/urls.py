from django.urls import path
from . import views

urlpatterns = [
    # Basic app & form
    path('', views.homepage, name='homepage'),
    path('about/', views.about, name='about'),
    path('services/', views.services, name='services'),
    path('form/', views.form, name='form'),
]
