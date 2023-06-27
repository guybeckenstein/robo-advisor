# Generated by Django 4.2.1 on 2023-06-06 18:11

import django.core.validators
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('investment_portfolio', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Investment',
            fields=[
                (
                    'id',
                    models.BigAutoField(primary_key=True, serialize=False, verbose_name='ID')
                ),
                (
                    'name',
                    models.CharField(max_length=30)
                ),
                (
                    'company',
                    models.CharField(max_length=30)
                ),
                (
                    'stock_symbol',
                    models.CharField(choices=[
                        ('SYMBOL1', 'SYMBOL1'),
                        ('SYMBOL2', 'SYMBOL2'),
                        ('SYMBOL3', 'SYMBOL3')
                    ], max_length=10)
                ),
                (
                    'purchase_price',
                    models.FloatField()),
                (
                    'quantity',
                    models.IntegerField(validators=[django.core.validators.MinValueValidator(0)])
                ),
                (
                    'date',
                    models.DateTimeField(auto_now_add=True)),
                (
                    'portfolio_id',
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.RESTRICT,
                        to='investment_portfolio.investmentportfolio'
                    )
                ),
            ],
            options={
                'db_table': 'Investment',
            },
        ),
    ]