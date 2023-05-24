
def choosePortfolioByRiskScore(optionalPortfolios, riskScore):
        if 0 < riskScore < 31:
            return optionalPortfolios['Safest Portfolio']
        if 30 < riskScore < 71:
            return optionalPortfolios['Sharpe Portfolio']
        if riskScore > 70:
            return optionalPortfolios['Max Risk Porfolio']


def buildReturnGiniPortfoliosDic(amountToInvest, selected, df):
        returnDic = {'Max Risk Porfolio': {}, 'Safest Portfolio': {}, 'Sharpe Portfolio': {}}
        min_gini = df['Gini'].min()
        max_sharpe = df['Sharpe Ratio'].max()
        max_profolio_annual = df['Profolio_annual'].max()

        # use the min, max values to locate and create the two special portfolios
        sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
        safe_portfolio = df.loc[df['Gini'] == min_gini]
        max_portfolio = df.loc[df['Profolio_annual'] == max_profolio_annual]

        # returnDic['Max Risk Porfolio']['Profolio_annual'] = max_portfolio['Profolio_annual'].values[0]
        # returnDic['Safest Portfolio']['Profolio_annual'] = safe_portfolio['Profolio_annual'].values[0]
        # returnDic['Sharpe Portfolio']['Profolio_annual'] = sharpe_portfolio['Profolio_annual'].values[0]
        for stock in selected:
            returnDic['Max Risk Porfolio'][stock] = max_portfolio[stock + ' Weight'].values[0] * amountToInvest
            returnDic['Safest Portfolio'][stock] = safe_portfolio[stock + ' Weight'].values[0] * amountToInvest
            returnDic['Sharpe Portfolio'][stock] = sharpe_portfolio[stock + ' Weight'].values[0] * amountToInvest
        return returnDic


def buildReturnMarkowitzPortfoliosDic(amountToInvest, selected, df):
    returnDic = {'Max Risk Porfolio': {}, 'Safest Portfolio': {}, 'Sharpe Portfolio': {}}
    min_Markowiz = df['Returns'].min()
    max_sharpe = df['Sharpe Ratio'].max()
    max_profolio_annual = df['Volatility'].max()

    # use the min, max values to locate and create the two special portfolios
    sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
    safe_portfolio = df.loc[df['Returns'] == min_Markowiz]
    max_portfolio = df.loc[df['Volatility'] == max_profolio_annual]

    # returnDic['Max Risk Porfolio']['Profolio_annual'] = max_portfolio['Profolio_annual'].values[0]
    # returnDic['Safest Portfolio']['Profolio_annual'] = safe_portfolio['Profolio_annual'].values[0]
    # returnDic['Sharpe Portfolio']['Profolio_annual'] = sharpe_portfolio['Profolio_annual'].values[0]
    for stock in selected:
        returnDic['Max Risk Porfolio'][stock] = max_portfolio[stock].values[0] * amountToInvest
        returnDic['Safest Portfolio'][stock] = safe_portfolio[stock].values[0] * amountToInvest
        returnDic['Sharpe Portfolio'][stock] = sharpe_portfolio[stock].values[0] * amountToInvest
    return returnDic