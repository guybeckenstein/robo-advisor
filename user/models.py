from django.core.validators import MinValueValidator, MaxValueValidator
from django.contrib.auth.models import User
from django.db import models

MIN_ANSWER = 0
MAX_ANSWER = 1


class UserPreferences(models.Model):
    id = models.BigAutoField(primary_key=True, verbose_name="ID")
    user = models.OneToOneField(User, on_delete=models.RESTRICT)
    ml_answer = models.IntegerField(validators=[MinValueValidator(MIN_ANSWER), MaxValueValidator(MAX_ANSWER)])
    model_answer = models.IntegerField(validators=[MinValueValidator(MIN_ANSWER), MaxValueValidator(MAX_ANSWER)])
    date = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'UserPreferences'
