from crispy_forms.bootstrap import PrependedText
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout
from django import forms
from django.core.validators import MinValueValidator
from django.urls import reverse_lazy

from investment.models import Investment


class InvestmentForm(forms.ModelForm):
    amount = forms.IntegerField(
        validators=[MinValueValidator(1)], label='Amount To Invest'
    )

    class Meta:
        model = Investment
        fields = ('amount',)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.helper = FormHelper()
        # self.helper.form_class = 'form-horizontal'

        # self.helper = FormHelper()
        # self.helper.form_show_labels = True
        # for field in MyFor.Meta.unlabelled_fields:
        #     self.fields[field].label = False

        # Override widget attributes for the 'amount' field
        self.fields['amount'].widget.attrs.update({
            'hx-get': reverse_lazy('check_amount'),
            'hx-target': '#div_id_amount',
            'hx-trigger': 'keyup changed delay:2s'
            # Optional delay to reduce frequent requests
        })

