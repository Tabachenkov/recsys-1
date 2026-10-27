import numpy as np
from scipy import sparse
from .utils.classes import BaseModel
from tqdm.auto import tqdm
from scipy.linalg import solve

class IALS(BaseModel):
    def __init__(self, alpha: float = 10, pow: float = 1, lambda_: float = 1,
                 k=128, num_iters=10, tol=1e-6, seed=42):
        self.alpha = alpha
        self.pow = pow
        self.lambda_ = lambda_
        self.k = k
        self.num_iters = num_iters
        self.tol = tol
        self.seed = seed

    def fit(self, matrix, target=None, short=None, short_weight=0.5):
        matrix = matrix.tocsr().astype(float)
        matrix.eliminate_zeros()
        matrix.data[:] = 1
        target = matrix.copy() if target is None else target.tocsr().astype(float).multiply(matrix).tocsr()
        if short is not None:
            short = short.tocsr().astype(float)
            short.eliminate_zeros()
            short.data[:] = 1
            short = short - short.multiply(matrix)
            matrix = matrix + short_weight * short
            matrix.eliminate_zeros()
        
        rng = np.random.default_rng(self.seed)
        self.user_embeddings = rng.normal(0, 0.01, (matrix.shape[0], self.k))
        self.item_embeddings = rng.normal(0, 0.01, (matrix.shape[1], self.k))

        matrix_T = matrix.T.tocsr()
        target_T = target.T.tocsr()

        def init(matrix: sparse.csr_matrix):
            c = 1 + self.alpha * np.abs(matrix.data)
            C_ = np.asarray(sparse.csr_matrix((c, matrix.indices, matrix.indptr),
                            shape=matrix.shape).sum(1)).ravel() ** self.pow
            C_ = C_ / C_.mean()
            return c, C_

        user_inited, item_inited = init(matrix), init(matrix_T)
        I = np.eye(self.k)
        def step(c, C_, matrix, target, x, y, name: str):
            y_t_y = y.T @ y
            for i in tqdm(range(x.shape[0]), leave=False, desc=f"{name} step"):
                begin, end = matrix.indptr[i], matrix.indptr[i + 1]
                if begin == end:
                    x[i] = 0
                    continue
                ids = matrix.indices[begin:end]
                p = target[i, ids].toarray().ravel()
                cur_c = c[begin:end]
                cur_y = y[matrix.indices[begin:end]]
                x[i] = solve(y_t_y + self.lambda_ * C_[i] * I + cur_y.T @ ((cur_c - 1)[:, None] * cur_y),
                                cur_y.T @ (cur_c * p), assume_a="pos", check_finite=False)
        n_u = self.user_embeddings.size
        n_i = self.item_embeddings.size
        for _ in (pbar := tqdm(range(self.num_iters), leave=False)):
            user_embeddings_old, item_embeddings_old = self.user_embeddings.copy(), self.item_embeddings.copy()
            step(user_inited[0], user_inited[1], matrix, target, self.user_embeddings, self.item_embeddings, name='user')
            step(item_inited[0], item_inited[1], matrix_T, target_T, self.item_embeddings, self.user_embeddings, name='item')
            changes = (
                np.linalg.norm(self.user_embeddings - user_embeddings_old) / np.sqrt(n_u),
                np.linalg.norm(self.item_embeddings - item_embeddings_old) / np.sqrt(n_i)
            )
            pbar.set_description(f"user: {changes[0]:.3e}, item: {changes[1]:.3e}")
            if max(changes) < self.tol:
                break
        return self


    def predict_score(self, user):
        return self.item_embeddings @ self.user_embeddings[user]
