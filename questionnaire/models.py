from django.core.validators import MinValueValidator, MaxValueValidator
from django.contrib.auth.models import User
from django.db import models

MIN_ANSWER = 1
MAX_ANSWER = 3


class Questionnaire(models.Model):
    id = models.BigAutoField(primary_key=True, verbose_name="ID")
    user = models.OneToOneField(User, on_delete=models.RESTRICT)
    answer_1 = models.IntegerField(validators=[MinValueValidator(MIN_ANSWER), MaxValueValidator(MAX_ANSWER)])
    answer_2 = models.IntegerField(validators=[MinValueValidator(MIN_ANSWER), MaxValueValidator(MAX_ANSWER)])
    answer_3 = models.IntegerField(validators=[MinValueValidator(MIN_ANSWER), MaxValueValidator(MAX_ANSWER)])
    date = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'Questionnaire'
