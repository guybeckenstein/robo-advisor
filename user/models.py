from django.core.validators import MinValueValidator, MaxValueValidator
from django.contrib.auth.models import User
from django.db import models

RISK_MIN = 1
RISK_MAX = 3


class InvestorUser(models.Model):
    id = models.BigAutoField(primary_key=True, verbose_name="ID")
    user = models.OneToOneField(User, on_delete=models.RESTRICT)
    risk_level = models.IntegerField(validators=[MinValueValidator(RISK_MIN), MaxValueValidator(RISK_MAX)])
    starting_investment_amount = models.IntegerField()
    stocks_symbols = models.CharField(max_length=500)  # Symbols are divided by `;`
    stocks_weights = models.CharField(max_length=500)  # Symbols are divided by `;`
    sectors_names = models.CharField(max_length=500)  # Symbols are divided by `;`
    sectors_weights = models.CharField(max_length=500)  # Symbols are divided by `;`
    annual_returns = models.FloatField()
    annual_max_loss = models.FloatField()
    annual_volatility = models.FloatField()
    annual_sharpe = models.FloatField()
    total_change = models.FloatField()
    monthly_change = models.FloatField()
    daily_change = models.FloatField()

    class Meta:
        db_table = 'InvestorUser'
