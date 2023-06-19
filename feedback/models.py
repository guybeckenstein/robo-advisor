from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from django.urls import reverse


class Feedback(models.Model):
    class FeedbackType(models.TextChoices):
        # TODO: update date range
        POSITIVE = ('POS', 'POSITIVE')
        NEGATIVE = ('NEG', 'NEGATIVE')
        NEUTRAL = ('NEU', 'NEUTRAL')
    id = models.BigAutoField(primary_key=True, verbose_name="ID")
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    feedback_type = models.CharField(max_length=3, choices=FeedbackType.choices, default=FeedbackType.NEUTRAL)
    rating = models.IntegerField(null=True, validators=[MinValueValidator(1), MaxValueValidator(5)])
    title = models.CharField(max_length=50)
    content = models.TextField()
    date_posted = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = 'Feedback'

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse('feedback-detail', kwargs={'pk': self.pk})
