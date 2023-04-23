import manageData

# TODO -BUILD USER CLASS


class User:
    level = 1
    name = ""
    sybmolIndex = []
    portfolio = []

    def __init__(self, name="", level=1, sybmolIndex=[]):
        User.name = name
        User.level = level
        User.sybmolIndex = sybmolIndex

    def getName(self):
        return User.name

    def getLevel(self):
        return User.level

    def getSybmolIndex(self):
        return User.sybmolIndex

    def getPortfolioData(self):
        return User.portfolio

    def setPortfolioData(self, command):
        User.portfolio = manageData.setPortfolioData(command, User.sybmolIndex)
