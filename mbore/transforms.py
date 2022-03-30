import numpy as np


def _ensure_2d(method):
    def _ensure_2d_wrapper(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = np.reshape(X, (1, -1))
        return method(self, X)

    return _ensure_2d_wrapper


class TransformBase:
    @_ensure_2d
    def __init__(self, Xtrain: np.ndarray):
        self.Xtrain = Xtrain

    def transform(self, X: np.ndarray) -> np.ndarray:
        """map from original space to transformed space"""
        raise NotImplementedError

    def untransform(self, X: np.ndarray) -> np.ndarray:
        """map from transformed space back to the to original space"""
        raise NotImplementedError


class ZeroOneTransform(TransformBase):
    def __init__(self, Xtrain: np.ndarray, lb: np.ndarray, ub: np.ndarray):
        super(ZeroOneTransform, self).__init__(Xtrain)

        self.lb = lb.reshape(1, -1)
        self.ub = ub.reshape(1, -1)
        self.diff = self.ub - self.lb

    @_ensure_2d
    def transform(self, X: np.ndarray) -> np.ndarray:
        # map to [0, 1]
        return (X - self.lb) / self.diff

    @_ensure_2d
    def untransform(self, X: np.ndarray) -> np.ndarray:
        # map from [0, 1] to [lb, ub]
        return X * self.diff + self.lb


class StandardizeTransform(TransformBase):
    def __init__(self, Xtrain: np.ndarray):
        super(StandardizeTransform, self).__init__(Xtrain)

        self.mu = np.mean(self.Xtrain, axis=0, keepdims=True)
        self.sigma = np.std(self.Xtrain, ddof=1, axis=0, keepdims=True)

        # a fudge to deal with cases where all X values are approx equal
        if self.sigma < 1e-6:
            self.sigma = 1e-6

    @_ensure_2d
    def transform(self, X: np.ndarray) -> np.ndarray:
        # map to have zero mean with unit standard deviation
        return (X - self.mu) / self.sigma

    @_ensure_2d
    def untransform(self, X: np.ndarray) -> np.ndarray:
        # map to have zero mean with unit standard deviation
        return X * self.sigma + self.mu
