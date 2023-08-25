from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models

from accounts.models import InvestorUser
from service.util import helpers

COLLECTION_MIN: int = 1
COLLECTION_MAX: int = len(helpers.get_collection_json_data().keys()) - 1  # Dynamic code


class Investment(models.Model):
    class Status(models.TextChoices):
        ACTIVE = ('ACTIVE', 'ACTIVE')
        INACTIVE = ('INACTIVE', 'INACTIVE')

    class Mode(models.TextChoices):
        ROBOT = ('ROBOT', 'ROBOT')
        USER = ('USER', 'USER')

    id = models.BigAutoField(primary_key=True, verbose_name="ID")
    investor_user = models.ForeignKey(InvestorUser, on_delete=models.RESTRICT)
    amount = models.IntegerField(validators=[MinValueValidator(1)])
    date = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.ACTIVE)
    mode = models.CharField(max_length=10, choices=Mode.choices, default=Mode.USER)
    stocks_collection_number = models.IntegerField(
        validators=[MinValueValidator(COLLECTION_MIN), MaxValueValidator(COLLECTION_MAX)], default=COLLECTION_MIN
    )

    class Meta:
        app_label = 'investment'
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

    def formatted_date(self):
        return self.date.strftime("%Y-%m-%d")
