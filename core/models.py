from django.core.validators import MinValueValidator, MaxValueValidator
from django.contrib.auth.models import User
from django.db import models

PREFERENCES_MIN = 0
PREFERENCES_MAX = 1
MIN_ANSWER = 1
MAX_ANSWER = 3


class QuestionnaireA(models.Model):
    id = models.BigAutoField(primary_key=True, verbose_name="ID")
    user = models.OneToOneField(User, on_delete=models.RESTRICT)
    ml_answer = models.IntegerField(
        validators=[MinValueValidator(PREFERENCES_MIN), MaxValueValidator(PREFERENCES_MAX)]
    )
    model_answer = models.IntegerField(
        validators=[MinValueValidator(PREFERENCES_MIN), MaxValueValidator(PREFERENCES_MAX)]
    )
    date = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'CapitalMarketAlgorithmPreferences'


class QuestionnaireB(models.Model):
    id = models.BigAutoField(primary_key=True, verbose_name="ID")
    user = models.OneToOneField(User, on_delete=models.RESTRICT)
    answer_1 = models.IntegerField(validators=[MinValueValidator(MIN_ANSWER), MaxValueValidator(MAX_ANSWER)])
    answer_2 = models.IntegerField(validators=[MinValueValidator(MIN_ANSWER), MaxValueValidator(MAX_ANSWER)])
    answer_3 = models.IntegerField(validators=[MinValueValidator(MIN_ANSWER), MaxValueValidator(MAX_ANSWER)])
    answers_sum = models.IntegerField(validators=[MinValueValidator(MIN_ANSWER * 3), MaxValueValidator(MAX_ANSWER * 3)])
    date = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'CapitalMarketInvestmentPreferences'


class TeamMember(models.Model):
    id = models.BigAutoField(primary_key=True, verbose_name="ID")
    alt = models.CharField(max_length=20, unique=True)
    full_name = models.CharField(max_length=20, unique=True)
    img = models.CharField(max_length=30, unique=True)

    class Meta:
        db_table = 'TeamMember'
