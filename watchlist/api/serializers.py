from watchlist.models import TopStock
from rest_framework import serializers


class TopStockSerializer(serializers.ModelSerializer):
    class Meta:
        model = TopStock
        fields = '__all__'
