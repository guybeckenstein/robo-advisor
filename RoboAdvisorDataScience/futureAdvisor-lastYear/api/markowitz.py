from flask import jsonify
from flask_apispec import MethodResource, marshal_with, use_kwargs
from flask_restful import Resource
from app.api.myResponses import InputSchema
from app.util.apiUtil import choosePortfolioByRiskScore,buildReturnMarkowitzPortfoliosDic
from app.dto.responseApi import ResponseApi
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
from pandas_datareader import data as pdr


class Markowitz(MethodResource, Resource):

    def get_optimal_portfolio(self, selected, start_year, end_year):
        Num_porSimulation = 500
        # Building the dataframe
        yf.pdr_override()
        frame = {}
        for stock in selected:
            data_var = pdr.get_data_yahoo(stock, start_year, end_year)['Adj Close']
            # data_var = yf.download(stock, start_year, end_year)['Adj Close']
            data_var.to_frame()
            frame.update({stock: data_var})
        table = pd.DataFrame(frame)

        # set the number of combinations for imaginary portfolios
        num_assets = len(selected)
        num_portfolios = Num_porSimulation

        # set random seed for reproduction's sake
        np.random.seed(101)
        selected_prices_value = table.dropna()
        years = len(selected_prices_value) / 253
        starting_value = selected_prices_value.iloc[0, :]
        ending_value = selected_prices_value.iloc[len(selected_prices_value) - 1, :]
        total_period_return = ending_value / starting_value
        annual_returns = (total_period_return ** (1 / years)) - 1
        annual_covariance = selected_prices_value.pct_change().cov() * 253
        port_returns = []
        port_volatility = []
        sharpe_ratio = []
        stock_weights = []

        for single_portfolio in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            returns = np.dot(weights, annual_returns)
            volatility = np.sqrt(np.dot(weights.T, np.dot(annual_covariance, weights)))
            sharpe = returns / volatility
            sharpe_ratio.append(sharpe)
            port_returns.append(returns * 100)
            port_volatility.append(volatility * 100)
            stock_weights.append(weights)
        portfolio = {'Returns': port_returns,
                     'Volatility': port_volatility,
                     'Sharpe Ratio': sharpe_ratio}
        for counter, symbol in enumerate(selected):
            portfolio[symbol] = [Weight[counter] for Weight in stock_weights]
        df = pd.DataFrame(portfolio)
        column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock for stock in selected]
        df = df[column_order]

        # calc the optimal portfolio
        # best_sharpe_portfolio = df.loc[df['Sharpe Ratio'] == df['Sharpe Ratio'].max()]
        # sharpe_portfolio = pd.DataFrame(columns=['Ticker', 'Weight'])
        # for i in range(len(selected)):
        #     ticker = selected[i]
        #     weight = best_sharpe_portfolio.loc[:, ticker].iloc[0]
        #     sharpe_portfolio = sharpe_portfolio.append({'Ticker': ticker, 'Weight': weight}, ignore_index=True)
        # sharpe_portfolio = sharpe_portfolio.set_index('Ticker')
        return df

    @marshal_with(InputSchema)  # marshalling with marshmallow library
    @use_kwargs(InputSchema, location=('query'))
    def get(self, amountToInvest, riskScore):
        stocks = ["BTC-USD", "ETH-USD", "ADA-USD"]
        start_date = '2020-01-01'
        end_date = '2022-08-10'
        markowitz_df = self.get_optimal_portfolio(stocks, start_date, end_date)
        PortfoliosDic = buildReturnMarkowitzPortfoliosDic(amountToInvest, stocks, markowitz_df)
        final_invest_portfolio = choosePortfolioByRiskScore(PortfoliosDic, riskScore)
        response = ResponseApi("Markowitz", final_invest_portfolio, amountToInvest, datetime.datetime.now())
        return jsonify(response.__str__())
