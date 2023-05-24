from flask import jsonify
from flask_apispec import MethodResource, marshal_with, use_kwargs
from flask_restful import Resource
from app.api.myResponses import InputSchema

from app.util.apiUtil import choosePortfolioByRiskScore, buildReturnGiniPortfoliosDic
from app.dto.responseApi import ResponseApi

import yfinance as yf
import math
import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pandas_datareader import data as pdr



class GiniWithML(MethodResource, Resource):

    def gini(self, selected, start_year, end_year):
        # Select stocks, start year and end year, stock number has no known limit
        Num_porSimulation = 500
        V = 1.5

        # Building the dataframe
        yf.pdr_override()
        frame = {}
        for stock in selected:
            data_var = pdr.get_data_yahoo(stock, start_year, end_year)['Adj Close']
            #data_var = yf.download(stock, start_year, end_year)['Adj Close']
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
            profolio_annual_new = self.analyzeMechainLearningFunc(profolio_return, table.index)
            sharpe = profolio_annual_new / gini_annual * 100
            sharpe_ratio.append(sharpe)
            port_profolio_annual.append(profolio_annual_new)
            port_gini_annual.append(gini_annual * 100)
            stock_weights.append(weights)

        # a dictionary for Returns and Risk values of each portfolio
        portfolio = {'Profolio_annual': port_profolio_annual,
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


    def analyzeMechainLearningFunc(self, profolio_return, table_index):
        df_final = pd.DataFrame({})
        forecast_col = 'col'
        df_final[forecast_col] = profolio_return
        # forecast_col= 'ADJ_PCT_change_SPY'
        df_final.fillna(value=-0, inplace=True)
        forecast_out = int(math.ceil(0.01 * len(df_final)))
        df_final['label'] = df_final[forecast_col].shift(-forecast_out)
        # print(df_final.head())
        # df_final.to_csv('Out.csv')

        # Added date
        df = df_final
        df['Date'] = table_index
        # print(df)
        X = np.array(df.drop(['label', 'Date'], 1))
        X = preprocessing.scale(X)
        X_lately = X[-forecast_out:]
        X = X[:-forecast_out]
        df.dropna(inplace=True)

        y = np.array(df['label'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        clf = LinearRegression()
        clf.fit(X_train, y_train)
        confidence = clf.score(X_test, y_test)
        print(confidence)
        forecast_set = clf.predict(X_lately)
        df['Forecast'] = np.nan

        last_date = df.iloc[-1]['Date']
        last_unix = last_date.timestamp()
        one_day = 86400
        next_unix = last_unix + one_day

        for i in forecast_set:
            next_date = datetime.datetime.fromtimestamp(next_unix)
            next_unix += 86400
            df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

        df['Forecast'].plot()
        # df['Forecast'].to_csv('Out-f.csv')

        ans = (((1 + df['Forecast'].mean()) ** 254) - 1) * 100
        print(ans)
        return ans




    @marshal_with(InputSchema)  # marshalling with marshmallow library
    @use_kwargs(InputSchema, location=('query'))
    def get(self, amountToInvest, riskScore):
        stocks = ["BTC-USD","ETH-USD", "ADA-USD"]
        start_date = '2020-01-01'
        end_date = '2022-08-10'
        gini_df = self.gini(stocks, start_date, end_date)
        PortfoliosDic = buildReturnGiniPortfoliosDic(amountToInvest, stocks, gini_df)
        final_invest_portfolio = choosePortfolioByRiskScore(PortfoliosDic, riskScore)
        response = ResponseApi("GiniWithML",final_invest_portfolio, amountToInvest, datetime.datetime.now())
        return jsonify(response.__str__())
