from accounts.models import CustomUser, InvestorUser
from rest_framework import serializers


class CustomUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = '__all__'


class InvestorUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = InvestorUser
        fields = '__all__'
