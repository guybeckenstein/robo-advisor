from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, HTML, Field, Submit
from django import forms
from django.urls import reverse_lazy

from service.config import settings
from service.util import helpers


def get_ml_options_tuple(size: int) -> list[tuple]:
    ml_models_str = settings.MACHINE_LEARNING_MODEL
    res: list = []
    for i in range(1, size + 1):
        str_i = ml_models_str[i - 1]
        curr_tuple = (str_i, str_i)
        res.append(curr_tuple)
    return res


def get_symbols_tuple(symbols: list[str]) -> list[tuple[str, str]]:
    res: list = []
    for value in symbols:
        res.append((value, value))
    return res


class DiscoverStocksForm(forms.Form):
    list_of_ml_options_tuple: list[tuple] = get_ml_options_tuple(size=4)
    ml_model = forms.ChoiceField(
        widget=forms.RadioSelect(attrs={'class': 'horizontal-radio'}), choices=list_of_ml_options_tuple
    )
    symbols_descriptions: list[tuple[str, str]] = get_symbols_tuple(symbols=helpers.get_descriptions_list())
    symbol = forms.ChoiceField(choices=symbols_descriptions)
    start_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))
    end_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))

    def __init__(self, *args, **kwargs):
        # Form
        super().__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.form_id = 'discover-stocks-form'
        self.helper.attrs = {
            'hx-get': reverse_lazy('chosen_stock'),
            'hx-target': '.watchlist-chosen-stock',
            'hx-swap': 'outerHTML'
        }
        self.helper.layout = Layout(
            Field('ml_model'),
            Field('symbol'),
            HTML(
                '<label for="id_end_date" class="form-label requiredField" style="display: block;">'
                'Start Date to End Date'
                '</label>'
                '<div id="div_id_start_date" class="mb-3" style="width: 432px; display: inline-block;">'
                '<input type="date" name="start_date" class="dateinput form-control" required="" id="id_start_date">'
                '</div>'
                '<div style="width: 38px; height: 38px; background-color: rgb(200, 200, 200); color: rgb(72,72,72); '
                'display: inline-block; text-align: center; padding-top: 7px;">'
                'to'
                '</div>'
                '<div id="div_id_end_date" class="mb-3" style="width: 432px; display: inline-block;">'
                '<input type="date" name="end_date" class="dateinput form-control" required="" id="id_end_date">'
                '</div>'
            ),
        )
        # TODO : add static/img/spiner1 img for wating
        self.helper.add_input(Submit('submit', 'Submit', css_class='btn-dark'))
