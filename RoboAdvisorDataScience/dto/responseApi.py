
class ResponseApi:
    def __str__(self, **options):
        return {
            'algorithm': self.algorithm,
            'Profolios': self.Profolios,
            'totalInvestment': self.totalInvestment,
            'date': self.date
        }

    def __init__(self, algorithm, Profolios=None, totalInvestment=float('inf'), date=None):
        self.algorithm = algorithm
        self.Profolios = Profolios
        self.totalInvestment = totalInvestment
        self.date = date
