from django.urls import path
from . import views

urlpatterns = [
    # Basic user API
    path('signup/', views.SignUpView.as_view(), name='signup'),
    path("check/email/", views.check_email, name='check_email'),
    path("check/phone_number/", views.check_phone_number, name='check_phone_number'),
    path("check/first_name/", views.check_first_name, name='check_first_name'),
    path("check/last_name/", views.check_last_name, name='check_last_name'),
    path("validate/password/", views.check_password_confirmation, name='check_password_confirmation'),
    path('login/', views.HtmxLoginView.as_view(), name='account_login'),
    path('login/email/check/', views.check_login_email, name='check_login_email_view'),
    path('reset/email/check/', views.check_login_email_reset, name='check_login_email_reset'),
    path('logout/', views.logout_view, name='account_logout'),
    # Profile API
    path('profile/', views.profile_main, name='profile_main'),
    path('profile/accounts/', views.profile_account, name='profile_account'),
    path('profile/accounts/change/details/', views.profile_account_details, name='profile_account_details'),
    path('profile/accounts/change/password/', views.MyPasswordChangeForm.as_view(), name='profile_account_password'),
    path('profile/investor/', views.profile_investor, name='profile_investor'),
]
