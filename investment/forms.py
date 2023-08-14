from django import forms

from investment.models import Investment


class InvestmentForm(forms.ModelForm):
    class Meta:
        model = Investment
        fields = ('amount',)