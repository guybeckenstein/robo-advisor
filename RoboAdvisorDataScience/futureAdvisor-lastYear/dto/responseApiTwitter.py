
class ResponseApiTwitter:
    def __str__(self, **options):
        return {
            'stock': self.stock,
            'Buy': self.buy,
            'Sell': self.sell,
            'Hold': self.hold,
        }

    def __init__(self, stock, buy=float('inf'),sell=float('inf') ,hold=float('inf')):
        self.stock = stock
        self.buy = buy *10
        self.sell = sell*10
        self.hold = hold*10
