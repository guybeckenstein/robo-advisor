import datetime

from flask import jsonify
from flask_apispec import MethodResource, marshal_with, use_kwargs
from flask_restful import Resource

from app.api.myResponses import InputSchema

from app.util.apiUtil import choosePortfolioByRiskScore, buildReturnGiniPortfoliosDic
from app.dto.responseApi import ResponseApi

import yfinance as yf
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr


class Gini(MethodResource, Resource):

    def gini(self, selected, start_date, end_date):
        # Select stocks, start year and end year, stock number has no known limit
        selected = selected
        start_year = start_date
        end_year = end_date
        Num_porSimulation = 500
        V = 1.5

        # Building the dataframe
        yf.pdr_override()
        frame = {}
        for stock in selected:
            data_var = pdr.get_data_yahoo(stock, start_year, end_year)['Adj Close']
            data_var.to_frame()
            frame.update({stock: data_var})

        # Mathematical calculations, creation of 5000 portfolios,
        table = pd.DataFrame(frame)
        # pd.DataFrame(frame).to_csv('Out.csv')
        returns_daily = table.pct_change()
        port_profolio_annual = []
        port_gini_annual = []
        sharpe_ratio = []
        stock_weights = []

        # set the number of combinations for imaginary portfolios
        num_assets = len(selected)
        num_portfolios = Num_porSimulation

        # set random seed for reproduction's sake
        np.random.seed(101)

        # Mathematical calculations, creation of 5000 portfolios,
        table = pd.DataFrame(frame)
        # pd.DataFrame(frame).to_csv('Out.csv')
        returns_daily = table.pct_change()
        for stock in returns_daily.keys():
            table[stock + '_change'] = returns_daily[stock]

        # populate the empty lists with each portfolios returns,risk and weights
        for single_portfolio in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            profolio = np.dot(returns_daily, weights)
            profolio_return = pd.DataFrame(profolio)
            rank = profolio_return.rank()
            rank_divided_N = rank / len(rank)  # Rank/N
            one_sub_rank_divided_N = 1 - rank_divided_N  # 1-Rank/N
            one_sub_rank_divided_N_power_v_sub_one = one_sub_rank_divided_N ** (V - 1)  # (1-Rank/N)^(V-1)
            mue = profolio_return.mean().tolist()[0]
            x_avg = one_sub_rank_divided_N_power_v_sub_one.mean().tolist()[0]
            profolio_mue = profolio_return - mue
            rank_sub_x_avg = one_sub_rank_divided_N_power_v_sub_one - x_avg
            profolio_mue_mult_rank_x_avg = profolio_mue * rank_sub_x_avg
            summary = profolio_mue_mult_rank_x_avg.sum().tolist()[0] / (len(rank) - 1)
            gini_daily = summary * (-V)
            gini_annual = gini_daily * (254 ** 0.5)
            profolio_annual = ((1 + mue) ** 254) - 1
            # A call to the function we wrote
            sharpe = profolio_annual / gini_annual * 100
            sharpe_ratio.append(sharpe)
            port_profolio_annual.append(profolio_annual)
            port_gini_annual.append(gini_annual * 100)
            stock_weights.append(weights)
        # a dictionary for Returns and Risk values of each portfolio
        portfolio = {'Profolio_annual': profolio_annual,
                     'Gini': port_gini_annual,
                     'Sharpe Ratio': sharpe_ratio}

        # extend original dictionary to accomodate each ticker and weight in the portfolio
        for counter, symbol in enumerate(selected):
            portfolio[symbol + ' Weight'] = [Weight[counter] for Weight in stock_weights]

        # make a nice dataframe of the extended dictionary
        df = pd.DataFrame(portfolio)

        # get better labels for desired arrangement of columns
        column_order = ['Profolio_annual', 'Gini', 'Sharpe Ratio'] + [stock + ' Weight' for stock in selected]

        # reorder dataframe columns
        df = df[column_order]
        return df

    @marshal_with(InputSchema)  # marshalling with marshmallow library
    @use_kwargs(InputSchema, location=('query'))
    def get(self, amountToInvest, riskScore):
        stocks = ["BTC-USD", "ETH-USD", "ADA-USD"]
        start_date = '2020-01-01'
        end_date = '2022-08-10'

        gini_df = self.gini(stocks, start_date, end_date)
        PortfoliosDic = buildReturnGiniPortfoliosDic(amountToInvest, stocks, gini_df)
        final_invest_portfolio = choosePortfolioByRiskScore(PortfoliosDic, riskScore)
        response = ResponseApi("Gini", final_invest_portfolio, amountToInvest, datetime.datetime.now())
        return jsonify(response.__str__())
