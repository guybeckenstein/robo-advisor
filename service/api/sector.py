class Sector:
    __name: str = ""
    __weight: float = 0.0
    __stocks: list = []

    def __init__(self, name: str):
        self.__name = name
        self.__weight: float = 0.0
        self.__stocks: list = []

    def get_name(self) -> str:
        return self.__name

    def get_weight(self) -> float:
        return self.__weight

    def get_stocks(self) -> list:
        return self.__stocks

    def set_weight(self, weight: float) -> None:
        self.__weight = weight

    def add_weight(self, weight: float) -> None:
        self.__weight = self.__weight + weight

    def set_stocks(self, stocks: list) -> None:
        self.__stocks = stocks

    def add_stock(self, stock) -> None:
        self.__stocks.append(stock)
