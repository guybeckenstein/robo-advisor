from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    # User authentication
    path('login/', auth_views.LoginView.as_view(template_name='user/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(template_name='user/logout.html'), name='logout'),
    path('password-reset/',
         auth_views.PasswordResetView.as_view(template_name='user/password_reset.html'),
         name='password_reset'),
    path('password-reset/done/',
         auth_views.PasswordResetDoneView.as_view(template_name='user/password_reset_done.html'),
         name='password_reset_done'),
    path('password-reset-confirm/<uidb64>/<token>/',
         auth_views.PasswordResetConfirmView.as_view(template_name='user/password_reset_confirm.html'),
         name='password_reset_confirm'),
    path('password-reset-complete/',
         auth_views.PasswordResetCompleteView.as_view(template_name='user/password_reset_complete.html'),
         name='password_reset_complete'),
    # Other user functionality
    path('register/', views.register, name='register'),
    path('profile/', views.profile_main, name='profile_main'),                      # TODO
    path('profile/user/', views.profile_user, name='profile_user'),                 # TODO
    path('profile/investor/', views.profile_investor, name='profile_investor'),     # TODO
    path('profile/portfolio/', views.profile_portfolio, name='profile_portfolio'),  # TODO
]
