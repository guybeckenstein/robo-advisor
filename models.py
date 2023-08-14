# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class Capitalmarketalgorithmpreferences(models.Model):
    id = models.BigAutoField(primary_key=True)
    ml_answer = models.IntegerField()
    model_answer = models.IntegerField()
    date = models.DateTimeField()
    user = models.OneToOneField('Customuser', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'CapitalMarketAlgorithmPreferences'


class Capitalmarketinvestmentpreferences(models.Model):
    id = models.BigAutoField(primary_key=True)
    answer_1 = models.IntegerField()
    answer_2 = models.IntegerField()
    answer_3 = models.IntegerField()
    answers_sum = models.IntegerField()
    date = models.DateTimeField()
    user = models.OneToOneField('Customuser', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'CapitalMarketInvestmentPreferences'


class Customuser(models.Model):
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.BooleanField()
    is_staff = models.BooleanField()
    is_active = models.BooleanField()
    date_joined = models.DateTimeField()
    id = models.BigAutoField(primary_key=True)
    username = models.CharField(unique=True, max_length=150)
    email = models.CharField(unique=True, max_length=254)
    first_name = models.CharField(max_length=255)
    last_name = models.CharField(max_length=255)
    phone_number = models.CharField(max_length=128)

    class Meta:
        managed = False
        db_table = 'CustomUser'


class CustomuserGroups(models.Model):
    id = models.BigAutoField(primary_key=True)
    customuser = models.ForeignKey(Customuser, models.DO_NOTHING)
    group = models.ForeignKey('AuthGroup', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'CustomUser_groups'
        unique_together = (('customuser', 'group'),)


class CustomuserUserPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    customuser = models.ForeignKey(Customuser, models.DO_NOTHING)
    permission = models.ForeignKey('AuthPermission', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'CustomUser_user_permissions'
        unique_together = (('customuser', 'permission'),)


class Investmentportfolio(models.Model):
    id = models.BigAutoField(primary_key=True)
    name = models.CharField(max_length=30)
    total_investment_amount = models.IntegerField(blank=True, null=True)
    current_value = models.IntegerField(blank=True, null=True)
    return_on_investment = models.IntegerField(blank=True, null=True)
    investment_strategy = models.CharField(max_length=20)
    user = models.ForeignKey(Customuser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'InvestmentPortfolio'
        unique_together = (('user', 'name'),)


class Investoruser(models.Model):
    id = models.BigAutoField(primary_key=True)
    risk_level = models.IntegerField()
    starting_investment_amount = models.IntegerField()
    stocks_symbols = models.CharField(max_length=500)
    stocks_weights = models.CharField(max_length=1000)
    sectors_names = models.CharField(max_length=500)
    sectors_weights = models.CharField(max_length=1000)
    annual_returns = models.FloatField()
    annual_max_loss = models.FloatField()
    annual_volatility = models.FloatField()
    annual_sharpe = models.FloatField()
    total_change = models.FloatField()
    monthly_change = models.FloatField()
    daily_change = models.FloatField()
    user = models.OneToOneField(Customuser, models.DO_NOTHING)
    stocks_collection_number = models.TextField()  # This field type is a guess.

    class Meta:
        managed = False
        db_table = 'InvestorUser'


class Teammember(models.Model):
    id = models.BigAutoField(primary_key=True)
    alt = models.CharField(unique=True, max_length=20)
    full_name = models.CharField(unique=True, max_length=20)
    github_username = models.CharField(unique=True, max_length=30)
    img = models.CharField(unique=True, max_length=30)

    class Meta:
        managed = False
        db_table = 'TeamMember'


class AccountEmailaddress(models.Model):
    email = models.CharField(unique=True, max_length=254)
    verified = models.BooleanField()
    primary = models.BooleanField()
    user = models.ForeignKey(Customuser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'account_emailaddress'


class AccountEmailconfirmation(models.Model):
    created = models.DateTimeField()
    sent = models.DateTimeField(blank=True, null=True)
    key = models.CharField(unique=True, max_length=64)
    email_address = models.ForeignKey(AccountEmailaddress, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'account_emailconfirmation'


class AuthGroup(models.Model):
    name = models.CharField(unique=True, max_length=150)

    class Meta:
        managed = False
        db_table = 'auth_group'


class AuthGroupPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)
    permission = models.ForeignKey('AuthPermission', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_group_permissions'
        unique_together = (('group', 'permission'),)


class AuthPermission(models.Model):
    name = models.CharField(max_length=255)
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING)
    codename = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'auth_permission'
        unique_together = (('content_type', 'codename'),)


class DjangoAdminLog(models.Model):
    action_time = models.DateTimeField()
    object_id = models.TextField(blank=True, null=True)
    object_repr = models.CharField(max_length=200)
    action_flag = models.SmallIntegerField()
    change_message = models.TextField()
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING, blank=True, null=True)
    user = models.ForeignKey(Customuser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'django_admin_log'


class DjangoContentType(models.Model):
    app_label = models.CharField(max_length=100)
    model = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'django_content_type'
        unique_together = (('app_label', 'model'),)


class DjangoSession(models.Model):
    session_key = models.CharField(primary_key=True, max_length=40)
    session_data = models.TextField()
    expire_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_session'


class SocialaccountSocialaccount(models.Model):
    provider = models.CharField(max_length=30)
    uid = models.CharField(max_length=191)
    last_login = models.DateTimeField()
    date_joined = models.DateTimeField()
    extra_data = models.TextField()
    user = models.ForeignKey(Customuser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'socialaccount_socialaccount'
        unique_together = (('provider', 'uid'),)


class SocialaccountSocialapp(models.Model):
    provider = models.CharField(max_length=30)
    name = models.CharField(max_length=40)
    client_id = models.CharField(max_length=191)
    secret = models.CharField(max_length=191)
    key = models.CharField(max_length=191)

    class Meta:
        managed = False
        db_table = 'socialaccount_socialapp'


class SocialaccountSocialtoken(models.Model):
    token = models.TextField()
    token_secret = models.TextField()
    expires_at = models.DateTimeField(blank=True, null=True)
    account = models.ForeignKey(SocialaccountSocialaccount, models.DO_NOTHING)
    app = models.ForeignKey(SocialaccountSocialapp, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'socialaccount_socialtoken'
        unique_together = (('app', 'account'),)
