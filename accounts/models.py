from django.contrib.auth.base_user import BaseUserManager
from django.contrib.postgres.fields import ArrayField
from django.core.validators import MinValueValidator, MaxValueValidator
from django.contrib.auth.models import AbstractUser
from django.db import models
from phonenumber_field.modelfields import PhoneNumberField



RISK_MIN = 1
RISK_MAX = 3


class UserManager(BaseUserManager):
    def create_user(self, email, password=None):
        if not email:
            raise ValueError("email is required")
        email = self.normalize_email(email)
        user = self.model(
            email=self.normalize_email(email),
        )

        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password):
        user = self.create_user(email, password)
        user.is_superuser = True
        user.is_staff = True
        user.save(using=self._db)
        return user


class CustomUser(AbstractUser):
    id = models.BigAutoField(primary_key=True, verbose_name="ID")
    username = models.CharField(max_length=150, unique=True, blank=True)
    email = models.EmailField(unique=True)
    first_name = models.CharField(max_length=255)
    last_name = models.CharField(max_length=255)
    phone_number = PhoneNumberField()

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = []

    objects = UserManager()

    def save(self, *args, **kwargs):
        self.username = self.email
        super().save(*args, **kwargs)

    def __str__(self):
        return self.email

    class Meta:
        db_table = 'CustomUser'
        verbose_name = 'Custom User'
        verbose_name_plural = 'Custom User'


class InvestorUser(models.Model):
    id = models.BigAutoField(primary_key=True, verbose_name="ID")
    user = models.OneToOneField(CustomUser, on_delete=models.RESTRICT)
    risk_level = models.IntegerField(validators=[MinValueValidator(RISK_MIN), MaxValueValidator(RISK_MAX)])
    total_investment_amount = models.IntegerField()
    total_profit = models.IntegerField(default=0)
    stocks_collection_number = models.CharField(max_length=1, default='1')
    stocks_symbols = ArrayField(models.CharField(max_length=50))
    stocks_weights = ArrayField(models.FloatField())
    sectors_names = ArrayField(models.CharField(max_length=50))
    sectors_weights = ArrayField(models.FloatField())
    annual_returns = models.FloatField()
    annual_max_loss = models.FloatField()
    annual_volatility = models.FloatField()
    annual_sharpe = models.FloatField()
    total_change = models.FloatField()
    monthly_change = models.FloatField()
    daily_change = models.FloatField()

    class Meta:
        db_table = 'InvestorUser'
        verbose_name = 'Investor User'
        verbose_name_plural = 'Investor User'