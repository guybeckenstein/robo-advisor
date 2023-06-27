from django.contrib.auth.models import User
from django.db import models

from investment.models import Investment


class Analytics(models.Model):
    class DateRange(models.TextChoices):
        # TODO: update date range
        RANGE1 = ('RANGE1', 'RANGE1')
        RANGE2 = ('RANGE2', 'RANGE2')
        RANGE3 = ('RANGE3', 'RANGE3')
    id = models.BigAutoField(primary_key=True, verbose_name="ID")
    user = models.OneToOneField(User, on_delete=models.RESTRICT)
    investment = models.OneToOneField(Investment, on_delete=models.RESTRICT)
    date_range = models.CharField(max_length=10, choices=DateRange.choices)
    # performance_metrics = ...
    # benchmark_indices = ...
    # comparison_data = ...

    class Meta:
        db_table = 'Analytics'
