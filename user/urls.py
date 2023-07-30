from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register, name='register'),
    path('profile/', views.profile_main, name='profile_main'),                      # TODO
    path('profile/user/', views.profile_user, name='profile_user'),                 # TODO
    path('profile/investor/', views.profile_investor, name='profile_investor'),     # TODO
    path('profile/portfolio/', views.profile_portfolio, name='profile_portfolio'),  # TODO
]
