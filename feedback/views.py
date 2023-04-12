from django.shortcuts import render, get_object_or_404
from django.contrib.auth.models import User
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from .models import Feedback


def feedback(request):
    context = {
        'feedbacks': Feedback.objects.all()
    }
    return render(request, 'feedback/feedback.html', {'title': 'Feedbacks', **context})


class FeedbackListView(ListView):
    model = Feedback
    template_name = 'feedback/feedback.html'  # <app>/<model>_<viewtype>.html
    context_object_name = 'feedbacks'
    ordering = ['-date_posted']
    paginate_by = 10


class UserFeedbackListView(ListView):
    model = Feedback
    template_name = 'feedback/user-posts.html'  # <app>/<model>_<viewtype>.html
    context_object_name = 'feedbacks'
    paginate_by = 10

    def get_queryset(self):
        user = get_object_or_404(User, username=self.kwargs.get('username'))
        return Feedback.objects.filter(author=user).order_by('-date_posted')


class FeedbackDetailView(DetailView):
    model = Feedback


class FeedbackCreateView(LoginRequiredMixin, CreateView):
    model = Feedback
    fields = ['content']

    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)


class FeedbackUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Feedback
    fields = ['content']

    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)


    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        else:
            return False


class FeedbackDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Feedback
    success_url = '/'

    def test_func(self):
        post = self.get_object()

        if self.request.user == post.author:
            return True
        else:
            return False
