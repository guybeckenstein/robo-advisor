from django.contrib import admin
from .models import CustomUser, InvestorUser

admin.site.register(CustomUser)
admin.site.register(InvestorUser)
