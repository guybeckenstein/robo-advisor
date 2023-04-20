"""
import sys
sys.path.append('/path/TASE-API/TASE')
from TASE import manageData
#from TASE import User"""

import manageData
import User
import setting
 
#GET INDEXES SYMBOLS
################################################################
#TODO- FIND BEST STOCKS AND INDEXES-FEATURES

#SET INDEXES MANUUALLY-alternative - only for now
IsraliIndexes=[142,601,602,715]
UsaIndexes=["Spy","Lqd","Gsg"]
################################################################

#WEIGHTS DEFINED IN SETTING, TODO- SET FROM MARKOVIZ

#INIT
################################################################
def init(name,level,inverstment):
    #user = User(name,level,sybmolIndex)
    #choose command- TODO
    #command="get_past_10_years_history"
    #User.setPortfolioData(command)

    #building portfolio
    if(level==1):
        sLossR=1.4
    elif(level==2):
        sLossR=1.5
    elif(level==3):
        sLossR=1.65
    
    #table,sLoss,returns_annual=manageData.buildingPortfolio(IsraliIndexes,UsaIndexes,setting.weights[level-1],sLossR,level)
    #print('retrun annual', '{:.2%}'.format(returns_annual), 'Max Loss annual' , '{:.2%}'.format(sLoss))
    #table.to_csv('table.csv')
    #manageData.plotPortfolioComponent(sybmolIndexs,setting.weights[level-1],name)#DIAGRAM
    #manageData.plotPctChange(table['weighted_sum'],sLoss,returns_annual,name)
    #manageData.plotYieldChange(table,name,inverstment)
    #TODO- SHOW DISTRIBUTION OF PORTFOLIO

########################################################################################################################################################
#OPERATIONS

#TODO- GET NAME FROM FORM AND DEFINES IN USER CLASS
name="Yarden"

#Markviz and prediction
record_percentage_to_predict = 0.3
manageData.markovich( 2000, IsraliIndexes,UsaIndexes, record_percentage_to_predict,name)

#TODO- GET LEVEL OF RISK FROM FORM

#GET LEVEL OF RISK FROM USER
#level=manageData.getLevelFromTerminal()
#inverstment=manageData.getInverstmentFromTerminal()
#init(name,3,1000)










