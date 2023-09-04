from dataclasses import dataclass, field


@dataclass(init=True, order=False, frozen=False)
class Sector:
    _name: str
    _weight: float = field(default=0.0)
    _stocks: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return self._name

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def stocks(self) -> list:
        return self._stocks

    def add_weight(self, value: float) -> None:
        self._weight = self.weight + value

    def add_stock(self, value) -> None:
        self._stocks.append(value)
