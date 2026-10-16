import numpy as np
from .ials import IALS

class TimeIALS(IALS):
    def __init__(self, time, T=10 * 24 * 3600, **kwargs):
        super().__init__(**kwargs)
        self.time = time
        self.T = T

    def fit(self, matrix):
        matrix, last_time = matrix
        target = last_time.astype(float).copy()
        target.data = np.exp(-(self.time - target.data) / self.T)
        return super().fit(matrix, target)
        
