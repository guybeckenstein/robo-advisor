import pandas as pd


class Broker:
    def __init__(self, cash_balance=0):
        self.cash_balance = cash_balance
        self.transactions = pd.DataFrame(columns=['Date', 'Action', 'Stock', 'Quantity', 'Price', 'Cumulative Profit'])

    def add_cash(self, amount):
        if amount > 0:
            self.cash_balance += amount
            print(f"Added ${amount} to cash balance. Total cash balance: ${self.cash_balance}")
        else:
            print("Amount must be greater than zero.")

    def buy_stock(self, stock_name, quantity, stock_price, date):
        total_cost = quantity * stock_price
        if total_cost <= self.cash_balance:
            self.cash_balance -= total_cost
            print(f"Bought {quantity} shares of {stock_name} at ${stock_price} per share.")
            print(f"Remaining cash balance: ${self.cash_balance}")

            if stock_name in self.transactions['Stock'].values:
                prev_cumulative_profit = self.transactions.loc[self.transactions['Stock'] == stock_name, 'Cumulative Profit'].iloc[-1]
            else:
                prev_cumulative_profit = 0

            cumulative_profit = prev_cumulative_profit - total_cost
            self.transactions = self.transactions.append({
                'Date': date,
                'Action': 'Buy',
                'Stock': stock_name,
                'Quantity': quantity,
                'Price': stock_price,
                'Cumulative Profit': cumulative_profit
            }, ignore_index=True)
        else:
            print("Insufficient funds to buy the stock.")

    def calculate_total_profit(self):
        current_prices = {'AAPL': 200, 'GOOGL': 300}  # Replace with real-time data or use an API to fetch current stock prices

        # Calculate the total profit for each share
        self.transactions['Current Price'] = self.transactions['Stock'].map(current_prices)
        self.transactions['Current Value'] = self.transactions['Current Price'] * self.transactions['Quantity']
        self.transactions['Total Profit'] = self.transactions['Current Value'] - self.transactions['Price'] * self.transactions['Quantity']

    def display_portfolio(self):
        print("Current Portfolio:")
        print(self.transactions)
        print(f"Total cash balance: ${self.cash_balance}")


# Example usage:
if __name__ == "__main__":
    broker = Broker(cash_balance=1000)

    broker.add_cash(500)  # Add $500 to the account
    broker.buy_stock("AAPL", 5, 150, date='2023-08-01')  # Buy 5 shares of AAPL at $150 per share on 2023-08-01
    broker.buy_stock("AAPL", 3, 180, date='2023-08-02')  # Buy 3 more shares of AAPL at $180 per share on 2023-08-02
    broker.calculate_total_profit()
    broker.display_portfolio()