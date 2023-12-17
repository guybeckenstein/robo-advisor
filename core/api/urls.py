from django.urls import path
from core.api import views

app_name = 'core'

urlpatterns = [
    path('', views.TeamMemberListView.as_view(), name='homepage'),
]
