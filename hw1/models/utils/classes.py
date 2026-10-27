from scipy import sparse

class BaseModel:
    def fit(self, matrix):
        raise NotImplementedError(
            f"{type(self).__name__} must implement 'fit'"
        )

    def predict_score(self, user: int):
        raise NotImplementedError(
            f"{type(self).__name__} must implement 'predict_score'"
        )
