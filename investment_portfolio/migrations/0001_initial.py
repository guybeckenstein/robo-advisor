# Generated by Django 4.2.1 on 2023-06-06 18:11

from django.conf import settings
import django.core.validators
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='InvestmentPortfolio',
            fields=[
                (
                    'id', models.BigAutoField(primary_key=True, serialize=False, verbose_name='ID')
                ),
                (
                    'name', models.CharField(max_length=30)
                ),
                (
                    'total_investment_amount', models.IntegerField(
                        blank=True,
                        null=True,
                        validators=[django.core.validators.MinValueValidator(0)]
                    )
                ),
                (
                    'current_value', models.IntegerField(
                        blank=True,
                        null=True,
                        validators=[django.core.validators.MinValueValidator(0)]
                    )
                ),
                (
                    'return_on_investment', models.IntegerField(
                        blank=True,
                        null=True,
                        validators=[django.core.validators.MinValueValidator(0)]
                    )
                ),
                (
                    'investment_strategy', models.CharField(choices=[
                        ('STRATEGY1', 'STRATEGY1'),
                        ('STRATEGY2', 'STRATEGY2'),
                        ('STRATEGY3', 'STRATEGY3')
                    ], max_length=20)
                ),
                (
                    'user', models.ForeignKey(
                        on_delete=django.db.models.deletion.RESTRICT,
                        to=settings.AUTH_USER_MODEL
                    )
                ),
            ],
            options={
                'db_table': 'InvestmentPortfolio',
            },
        ),
    ]
