import numpy as np
from .utils.classes import BaseModel
from scipy.linalg import solve

class EASE(BaseModel):
    def __init__(self, lambda_ : float = 10):
        self.lambda_ = lambda_

    def fit(self, matrix):
        self.matrix = matrix.tocsr().astype(float)
        self.matrix.eliminate_zeros()
        self.matrix.data[:] = 1

        eye = np.eye(self.matrix.shape[1])
        P = solve((self.matrix.T @ self.matrix).toarray() + eye * self.lambda_, 
                  eye, assume_a="pos", check_finite=False)
        self.B = np.ascontiguousarray(-P / np.diag(P)[None, :])
        np.fill_diagonal(self.B, 0)
        return self


    def predict_score(self, user):
        return np.asarray(self.matrix[user] @ self.B).ravel()
        
