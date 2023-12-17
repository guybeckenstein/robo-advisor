from investment.models import Investment
from rest_framework import serializers


class InvestmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Investment
        fields = ['investor_user', 'amount', 'date', 'status']
