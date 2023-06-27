from django.core.validators import MinValueValidator
from django.db import models

from investment_portfolio.models import InvestmentPortfolio


class Investment(models.Model):
    class StockSymbol(models.TextChoices):
        # TODO: update stock symbols
        SYMBOL1 = ('SYMBOL1', 'SYMBOL1')
        SYMBOL2 = ('SYMBOL2', 'SYMBOL2')
        SYMBOL3 = ('SYMBOL3', 'SYMBOL3')

    id = models.BigAutoField(primary_key=True, verbose_name="ID")
    portfolio = models.OneToOneField(InvestmentPortfolio, on_delete=models.RESTRICT)
    name = models.CharField(max_length=30)
    company = models.CharField(max_length=30)
    stock_symbol = models.CharField(max_length=10, choices=StockSymbol.choices)
    purchase_price = models.FloatField()
    quantity = models.IntegerField(validators=[MinValueValidator(0)])
    date = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'Investment'
