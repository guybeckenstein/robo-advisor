import pandas as pd
from matplotlib import pyplot as plt

from service.impl.sector import Sector

# Global Const Variables
FIG_SIZE: tuple[int, int] = (10, 8)
STYLE = 'oblique'
HA: str = 'center'
VA: str = 'center'
FONT_NAME: str = 'Arial'
LINE_STYLE = 'dashed'
FONT_WEIGHT = 'bold'
WRAP: bool = True
GRID: bool = True
BOTTOM: float = 0.4
ALPHA: float = 0.5
SEABORN_STYLE: str = 'seaborn-v0_8-dark'

plt.rcParams['font.family'] = 'Arial'


class MarkowitzAndGini:
    @staticmethod
    def create_scatter_plot(df, x: str, y: str, min_variance_portfolio, sharpe_portfolio, max_returns_portfolio):
        plt.style.use(SEABORN_STYLE)
        df.plot.scatter(
            x=x, y=y, c="Sharpe Ratio", cmap="RdYlGn_r", edgecolors="black", figsize=FIG_SIZE, grid=GRID,
        )
        portfolios: list = [min_variance_portfolio, sharpe_portfolio, max_returns_portfolio]
        colors: list[str, str, str] = ['orange', 'green', 'red']
        for i in range(3):
            plt.scatter(x=portfolios[i][x], y=portfolios[i][y], c=colors[i], marker="D", s=200)

        plt.xlabel(f"{x} (Std. Deviation) Percentage")
        plt.ylabel(f"Expected {y} Percentage")
        plt.title("Efficient Frontier", fontsize=20, fontweight="bold")
        plt.suptitle("Each point expresses a different distribution of weights of the investment portfolio.\n"
                     " where 'sharpe ratio' is the ratio between the annual profit and the standard deviation.",
                     fontsize=12, fontweight="bold")
        plt.subplots_adjust(bottom=BOTTOM)

    @staticmethod
    def get_stocks_str(sectors, three_best_sectors_weights) -> list[str, str, str]:
        stocks: list[str, str, str] = ['', '', '']
        for i in range(len(sectors)):
            for j in range(3):
                weight = three_best_sectors_weights[2 - j][i] * 100
                stocks[j] += f"{sectors[i].name}({'{:.2f}'.format(weight)}%,\n"
        return stocks

    @staticmethod
    def plot_markowitz_or_gini_graph(portfolios: list[pd.Series], stocks, text1: str, text2: str) -> None:
        colors: list[str, str, str] = ['orange', 'green', 'red']
        portfolios_names: list[str, str, str] = ['Safest Portfolio', 'Sharpest Portfolio', 'Max Returns Portfolio']
        with pd.option_context("display.float_format", "%{:,.2f}".format):
            for i in range(len(portfolios)):
                x: float = 0.2 + 0.3 * i
                plt.figtext(
                    x=x,
                    y=0.15,
                    s=f"{text1}: {str(round(portfolios[i].iloc[0], 2))}%\n"
                      f"{text2}: {str(round(portfolios[i].iloc[1], 2))}%\n"
                      f"Annual Max Loss: {str(round(portfolios[i].iloc[0] - 1.65 * portfolios[i].iloc[1], 2))}%\n"
                      f"Sharpe Ratio: {str(round(portfolios[i].iloc[2], 2))}\n"
                      "## Weights: ##\n"
                      f"{stocks[i]}",
                    bbox=dict(facecolor=colors[i], alpha=ALPHA), fontsize=10, style=STYLE, ha=HA, multialignment='left',
                    va=VA, fontname=FONT_NAME, wrap=WRAP,
                )

                plt.figtext(
                    x=x, y=0.28, s=f"{portfolios_names[i]}\n", fontsize=14,
                    fontweight=FONT_WEIGHT, ha=HA, va=VA, fontname=FONT_NAME, wrap=WRAP,
                )


class ThreePortfolios:
    @staticmethod
    def main_plot(pct_change_table: pd.DataFrame) -> tuple[list[str, str, str], list[str, str, str]]:
        plt.xlabel("Date")
        plt.ylabel("Returns Percentage")
        plt.title("Three Best Portfolios (Backtesting)", fontsize=20, fontweight="bold")
        labels: list[str, str, str] = ['Safest', 'Sharpe', 'Max Returns']
        colors: list[str, str, str] = ['orange', 'green', 'red']
        labels_len: int = len(labels)
        # Creates yield values for each portfolio
        for i in range(labels_len):
            pct_change_table[f'yield_{str(i + 1)}_percent'] = (pct_change_table[f'yield_{str(i + 1)}'] - 1) * 100
        # Plot yield for each portfolio
        for i in range(labels_len):
            pct_change_table[f'yield_{str(i + 1)}_percent'].plot(
                figsize=FIG_SIZE, grid=GRID, color=colors[i], linewidth=2, label=labels[i], legend=True,
                linestyle=LINE_STYLE
            )
        plt.legend(frameon=True, facecolor='white')  # Adjust legend background color
        return labels, colors

    @staticmethod
    def sub_plots(colors, labels, max_returns_portfolio, sharpe_portfolio, min_variance_portfolio, sectors,
                  three_best_sectors_weights):
        plt.subplots_adjust(bottom=BOTTOM)
        stocks_y = ['', '', '']
        for i in range(len(sectors)):
            for j in range(3):
                weight = three_best_sectors_weights[j][i] * 100
                stocks_y[j] += f"{sectors[i].name} ({weight:.2f}%),\n"
        stocks_y = [stock_y[:-2] for stock_y in stocks_y]
        with pd.option_context("display.float_format", "%{:,.2f}".format):
            fig_text_data: dict = {
                'name': labels,
                'portfolio': [min_variance_portfolio.iloc[0], sharpe_portfolio.iloc[0], max_returns_portfolio.iloc[0]],
                'stocks': stocks_y,
                'facecolor': colors
            }
            plt.rcParams['font.family'] = 'monospace'  # Set a monospaced font
            for i in range(3):
                x: float = 0.2 + 0.3 * i
                portfolio = fig_text_data['portfolio'][i]
                s = (
                    f"Annual Returns: {portfolio[0]:.2f}%\n"
                    f"Annual Volatility: {portfolio[1]:.2f}%\n"
                    f"Annual Max Loss: {portfolio[0] - 1.65 * portfolio[1]:.2f}%\n"
                    f"Annual Sharpe Ratio: {portfolio[2]:.2f}\n"
                    "## Weights: ##\n"
                    f"{fig_text_data['stocks'][i]}"
                )
                s = Spaces.add_spaces_to_each_line(s)
                bbox: dict = {'facecolor': fig_text_data['facecolor'][i], 'alpha': 0.5}
                plt.figtext(
                    x=x, y=0.15, s=s, bbox=bbox, fontsize=10.5, style=STYLE, ha=HA, multialignment='left', va=VA,
                    fontname=FONT_NAME, wrap=WRAP,
                )
                plt.figtext(
                    x=x, y=0.29, s=f"{fig_text_data['name'][i]} Portfolio:", fontsize=14,
                    fontweight=FONT_WEIGHT, ha=HA, va=VA, fontname=FONT_NAME, wrap=WRAP,
                )


class Spaces:
    @staticmethod
    def add_spaces_to_each_line(s):
        lines = s.split('\n')

        # Find the length of the longest line
        max_line_length = max(len(line) for line in lines)

        # Create a formatted version of each line with added spaces
        formatted_lines = []
        for line in lines:
            # Calculate the number of spaces needed to match the length of the longest line
            spaces_needed = max_line_length - len(line)

            # Add spaces to the line
            formatted_line = f"{line}{' ' * spaces_needed}"

            formatted_lines.append(formatted_line)

            # Join the formatted lines back into a single string
        formatted_s = '\n'.join(formatted_lines)
        return formatted_s


class EstimatedYield:
    @staticmethod
    def get_stocks_as_str(sectors: list[Sector]) -> str:
        stocks_str: str = ""
        for i in range(len(sectors)):
            name: str = sectors[i].name
            weight: float = sectors[i].weight * 100
            stocks_str += f"{name}({'{:.2f}'.format(weight)}%,\n"
        return stocks_str


class PriceForecast:
    @staticmethod
    def plt_figtext(plt_instance, x, y, history_annual_return, average_annual_return, forecast_short_time):
        """
        Add text box with annual returns value
        """
        plt_instance.figtext(
            x=x,
            y=y,
            s=f"Average Annual Return: {str(round(history_annual_return, 2))}%\n"
              f"Forecast Annual Return: {str(round(forecast_short_time, 2))}%\n"
              f"Average Annual Return with forecast: {str(round(average_annual_return, 2))}%\n"

        )

        return plt_instance
