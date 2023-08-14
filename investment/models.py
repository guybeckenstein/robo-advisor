from django.core.validators import MinValueValidator
from django.db import models

from accounts.models import InvestorUser


class Investment(models.Model):
    class Status(models.TextChoices):
        ACTIVE = ('ACTIVE', 'ACTIVE')
        INACTIVE = ('INACTIVE', 'INACTIVE')
    id = models.BigAutoField(primary_key=True, verbose_name="ID")
    investor_user = models.ForeignKey(InvestorUser, on_delete=models.RESTRICT)
    amount = models.IntegerField(validators=[MinValueValidator(1)])
    date = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.ACTIVE)

    class Meta:
        db_table = 'Investment'
        ordering = ['-id']
        verbose_name = 'Investment'
        verbose_name_plural = 'Investment'

    def make_investment_inactive(self) -> bool:
        """
        Returns true if investment is active, false if inactive
        """
        if self.status == self.Status.ACTIVE:
            self.status = self.Status.INACTIVE
            return True
        else:
            return False
