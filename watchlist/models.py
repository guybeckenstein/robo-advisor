from django.db import models

from accounts.models import CustomUser
from investment.models import Investment


class Watchlist(models.Model):
    id = models.BigAutoField(primary_key=True, verbose_name="ID")
    user = models.OneToOneField(CustomUser, on_delete=models.RESTRICT)
    investment = models.OneToOneField(Investment, on_delete=models.RESTRICT)
    date = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'Watchlist'
        verbose_name = 'Watchlist'
        verbose_name_plural = 'Watchlist'
