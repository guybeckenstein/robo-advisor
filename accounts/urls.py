from django.urls import path, include
from accounts import views

urlpatterns = [
    # Basic user API
    path('signup/', views.SignUpView.as_view(), name='signup'),
    path('check/', include([
        path('email/', views.check_email, name='check_email'),
        path('phone_number/', views.check_phone_number, name='check_phone_number'),
        path('first_name/', views.check_first_name, name='check_first_name'),
        path('last_name/', views.check_last_name, name='check_last_name'),
    ])),
    path('validate/password/', views.check_password_confirmation, name='check_password_confirmation'),
    path('login/', views.HtmxLoginView.as_view(), name='account_login'),
    path('login/email/check/', views.check_login_email, name='check_login_email_view'),
    path('reset/email/check/', views.check_login_email_reset, name='check_login_email_reset'),
    path('logout/', views.LogoutViewV2.as_view(), name='account_logout'),
    # Profile API
    path('profile/', include([
        path('', views.profile_main, name='profile_main'),
        path('accounts/', include([
            path('', views.profile_account, name='profile_account'),
            path('change/details/', views.profile_account_details, name='profile_account_details'),
            path('change/password/', views.MyPasswordChangeForm.as_view(), name='profile_account_password'),
        ])),
        path('investor/', views.profile_investor, name='profile_investor'),
    ])),
    # Django REST Framework APIs
    path('api/', include('accounts.api.urls', 'accounts_api')),
]
