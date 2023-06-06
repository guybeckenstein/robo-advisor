from django.urls import path
from . import views

urlpatterns = [
    path('feedback/', views.FeedbackListView.as_view(), name='feedback'),
    path('user/<str:username>', views.UserFeedbackListView.as_view(), name='user-feedbacks'),
    path('feedback/<int:pk>/', views.FeedbackDetailView.as_view(), name='feedback-detail'),
    path('feedback/new/', views.FeedbackCreateView.as_view(), name='feedback-create'),
    path('feedback/<int:pk>/update/', views.FeedbackUpdateView.as_view(), name='feedback-update'),
    path('feedback/<int:pk>/delete/', views.FeedbackDeleteView.as_view(), name='feedback-delete'),
]
