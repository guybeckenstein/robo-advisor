from django.urls import path
from . import views

urlpatterns = [
    # Basic user
    path('sign-up/', views.SignUpView.as_view(), name='signup'),
    path("check_email/", views.check_email, name='check_email'),                        # Validation for `Sign Up` form
    path("check_phone_number/", views.check_phone_number, name='check_phone_number'),   # Validation for `Sign Up` form
    path('login/', views.HtmxLoginView.as_view(), name='account_login'),
    path('logout/', views.logout_view, name='account_logout'),
    # Profile API
    path('profile/', views.profile_main, name='profile_main'),
    path('profile/accounts/', views.profile_account, name='profile_account'),
    path('profile/accounts/change/details/', views.profile_account_details, name='profile_account_details'),
    path('profile/accounts/change/password/', views.MyPasswordChangeForm.as_view(), name='profile_account_password'),
    path('profile/investor/', views.profile_investor, name='profile_investor'),
]
