class Sector:
    __name = ""
    __weight = 0.0
    __stocks = []

    def __init__(self, name):
        self.__name = name
        self.__weight = 0.0
        self.__stocks = []

    def getName(self):
        return self.__name

    def getWeight(self):
        return self.__weight

    def getStocks(self):
        return self.__stocks

    def setWeight(self, weight):
        self.__weight = weight

    def addWeight(self, weight):
        self.__weight = self.__weight + weight

    def setStocks(self, stocks):
        self.__stocks = stocks

    def addStock(self, stock):
        self.__stocks.append(stock)
