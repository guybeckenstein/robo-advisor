from django.contrib.postgres.fields import ArrayField
from django.core.validators import MinValueValidator, MaxValueValidator
from django.contrib.auth.models import User
from django.db import models

PREFERENCES_MIN = 0
PREFERENCES_MAX = 1
RISK_MIN = 1
RISK_MAX = 3


class UserPreferencesA(models.Model):
    id = models.BigAutoField(primary_key=True, verbose_name="ID")
    user = models.OneToOneField(User, on_delete=models.RESTRICT)
    ml_answer = models.IntegerField(
        default=PREFERENCES_MIN,
        validators=[MinValueValidator(PREFERENCES_MIN), MaxValueValidator(PREFERENCES_MAX)]
    )
    model_answer = models.IntegerField(
        default=PREFERENCES_MIN,
        validators=[MinValueValidator(PREFERENCES_MIN), MaxValueValidator(PREFERENCES_MAX)]
    )
    date = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'UserPreferencesA'


class UserPreferencesB(models.Model):
    id = models.BigAutoField(primary_key=True, verbose_name="ID")
    user = models.OneToOneField(User, on_delete=models.RESTRICT)
    risk_level = models.IntegerField(validators=[MinValueValidator(RISK_MIN), MaxValueValidator(RISK_MAX)])
    starting_investment_amount = models.IntegerField()
    stocks_symbols = ArrayField(models.CharField(max_length=100))
    stocks_weights = ArrayField(models.FloatField())
    sectors_names = ArrayField(models.CharField(max_length=100))
    sectors_weights = ArrayField(models.FloatField())
    annual_returns = models.FloatField()
    annual_max_loss = models.FloatField()
    annual_volatility = models.FloatField()
    annual_sharpe = models.FloatField()
    total_change = models.FloatField()
    monthly_change = models.FloatField()
    daily_change = models.FloatField()

    class Meta:
        db_table = 'UserPreferencesB'
