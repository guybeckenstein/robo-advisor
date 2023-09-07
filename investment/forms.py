from django import forms
from django.core.validators import MinValueValidator
from django.urls import reverse_lazy

from investment.models import Investment


class InvestmentsHistoryForm(forms.ModelForm):
    amount = forms.IntegerField(
        validators=[MinValueValidator(1)],
        label='Amount To Invest',
        widget=forms.TextInput(
            attrs={
                'placeholder': 'Add Amount',
                'style': 'border-radius: 5px 0px 0px 5px; width: 830px;'
            }
        )
    )

    class Meta:
        model = Investment
        fields = ('amount',)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['amount'].widget.attrs.update({
            'hx-get': reverse_lazy('check_amount'),
            'hx-target': '#div_id_amount',
            'hx-trigger': 'keyup changed delay:2s'
        })
