# Generated by Django 4.2.1 on 2023-06-06 18:11

from django.conf import settings
import django.core.validators
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    """initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('form', '0002_test_data'),
    ]

    operations = [
        migrations.CreateModel(
            name='Questionnaire',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False, verbose_name='ID')),
                ('answer_1', models.IntegerField(validators=[django.core.validators.MinValueValidator(1),
                                                             django.core.validators.MaxValueValidator(3)])),
                ('answer_2', models.IntegerField(validators=[django.core.validators.MinValueValidator(1),
                                                             django.core.validators.MaxValueValidator(3)])),
                ('answer_3', models.IntegerField(validators=[django.core.validators.MinValueValidator(1),
                                                             django.core.validators.MaxValueValidator(3)])),
                ('date', models.DateTimeField(auto_now_add=True)),
                ('user',
                 models.OneToOneField(on_delete=django.db.models.deletion.RESTRICT, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'db_table': 'Questionnaire',
            },
        ),
    ]"""
