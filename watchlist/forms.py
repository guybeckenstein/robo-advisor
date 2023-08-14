from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, HTML, Field, Submit
from django import forms
from django.urls import reverse_lazy

from service.util import helpers


def get_indexes_tuple(size: int) -> list[tuple]:
    res: list = []
    for i in range(1, size + 1):
        str_i = str(i)
        curr_tuple = (str_i, str_i)
        res.append(curr_tuple)
    return res


def get_symbols_tuple(symbols: list[str]) -> list[tuple[str, str]]:
    res: list = []
    for value in symbols:
        res.append((value, value))
    return res


class DiscoverStocksForm(forms.Form):
    list_of_indexes_tuple: list[tuple] = get_indexes_tuple(size=4)
    ml_model = forms.ChoiceField(
        widget=forms.RadioSelect(attrs={'class': 'horizontal-radio'}), choices=list_of_indexes_tuple
    )
    symbols_names: list[tuple[str, str]] = get_symbols_tuple(symbols=helpers.get_symbols_names_list())
    symbol = forms.ChoiceField(choices=symbols_names)
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
                '<div id="div_id_start_date" class="mb-3" style="width: 423px; display: inline-block;">'
                    '<input type="date" name="start_date" class="dateinput form-control" required="" id="id_start_date">'
                '</div>'
                '<div style="width: 38px; height: 38px; background-color: rgb(200, 200, 200); color: rgb(72,72,72); display: inline-block; text-align: center; padding-top: 7px;">'
                    'to'
                '</div>'
                '<div id="div_id_end_date" class="mb-3" style="width: 423px; display: inline-block;">'
                    '<input type="date" name="end_date" class="dateinput form-control" required="" id="id_end_date">'
                '</div>'
            ),
        )
        self.helper.add_input(Submit('submit', 'Submit', css_class='btn-dark'))