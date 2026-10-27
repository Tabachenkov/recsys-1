import numpy as np
from scipy import sparse
from .utils.classes import BaseModel


class CF(BaseModel):
    def __init__(self, k=100, lambda_=10, 
                 similarity: str = "cosine",
                 batch_size: int = 128):
        self.k = k
        self.lambda_ = lambda_
        if similarity not in ["cosine", "jaccard"]:
            raise ValueError("similarity parameter must be 'cosine' or 'jaccard'")
        self.similarity = similarity
        self.batch_size = batch_size

    def fit(self, matrix):
        self.matrix = matrix.tocsr().astype(float)
        self.matrix.eliminate_zeros()
        self.matrix.data[:] = 1
        counts = np.asarray(self.matrix.sum(axis=0)).ravel()
        norms = np.sqrt(counts)
        rows, cols, values = [], [], []
        for start in range(0, self.matrix.shape[1], self.batch_size):
            overlap = (self.matrix[:, start:start + self.batch_size].T @ self.matrix).tocsr()
            for row in range(overlap.shape[0]):
                i = start + row
                left, right = overlap.indptr[row:row + 2]
                j, common = overlap.indices[left:right], overlap.data[left:right]
                keep = j != i
                j, common = j[keep], common[keep]
                denominator = (norms[i] * norms[j] if self.similarity == "cosine"
                               else counts[i] + counts[j] - common)
                scores = common / denominator
                scores *= common / (common + self.lambda_)
                top = np.argsort(-scores, kind="stable")[:self.k]
                rows.extend([i] * len(top))
                cols.extend(j[top])
                values.extend(scores[top])
        self.sim_matrix = sparse.csr_matrix(
            (values, (rows, cols)), shape=(matrix.shape[1], matrix.shape[1]),
        )
        return self

    def predict_score(self, user):
        return (self.matrix[user] @ self.sim_matrix).toarray().ravel()
        
