class Sector:
    def __init__(self, name: str = ""):
        self._name: str = name
        self._weight: float = 0.0
        self._stocks: list = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def weight(self) -> float:
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        self._weight = value

    @property
    def stocks(self) -> list:
        return self._stocks

    @stocks.setter
    def stocks(self, value: list) -> None:
        self._stocks = value

    def add_weight(self, value: float) -> None:
        self._weight = self.weight + value

    def add_stock(self, value) -> None:
        self._stocks.append(value)
