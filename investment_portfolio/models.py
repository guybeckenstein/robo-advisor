from django.core.validators import MinValueValidator
from django.contrib.auth.models import User
from django.db import models


class InvestmentPortfolio(models.Model):
    class Strategy(models.TextChoices):
        # TODO: update strategies
        STRATEGY1 = ('STRATEGY1', 'STRATEGY1')
        STRATEGY2 = ('STRATEGY2', 'STRATEGY2')
        STRATEGY3 = ('STRATEGY3', 'STRATEGY3')

    id = models.BigAutoField(primary_key=True, verbose_name="ID")
    user = models.ForeignKey(User, on_delete=models.RESTRICT)
    name = models.CharField(max_length=30)
    total_investment_amount = models.IntegerField(null=True, blank=True, validators=[MinValueValidator(0)])
    current_value = models.IntegerField(null=True, blank=True, validators=[MinValueValidator(0)])
    return_on_investment = models.IntegerField(null=True, blank=True, validators=[MinValueValidator(0)])
    investment_strategy = models.CharField(max_length=20, choices=Strategy.choices)

    class Meta:
        db_table = 'InvestmentPortfolio'
        unique_together = ['user', 'name']
