const radioButtons = document.querySelectorAll('input[name="stocks_collection_number"]');
const stocksSymbolsSelect = document.getElementById('id_stocks_symbols');

document.addEventListener('DOMContentLoaded', function() {
    const radioButtons = document.querySelectorAll('[name="stocks_collection_number"]');
    const stocksSymbolsSelect = document.getElementById('id_stocks_symbols');
    const stocksSymbolsOptions = stocksSymbolsSelect.options;

    // Disable the stocks_symbols field on page load
    stocksSymbolsSelect.disabled = true;

    radioButtons.forEach(function(radioButton) {
        radioButton.addEventListener('change', function() {
            const selectedValue = this.value;
            const stocksSymbols = stocksSymbolsDataHtml[selectedValue.toString()].sort();
            stocksSymbolsSelect.disabled = true;
            stocksSymbolsSelect.innerHTML = '';
            stocksSymbols.forEach(symbol => {
                const option = document.createElement('option');
                option.value = symbol;
                option.textContent = symbol;
                stocksSymbolsSelect.appendChild(option);
            });
        });
    });
});