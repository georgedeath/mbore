from typing import Any, Dict, Optional
from . import problem_sets
import numpy as np


def ensure_2d(X):
    """Ensures the input (X) is two-dimensional"""
    nd = X.ndim

    if nd == 1:
        X = np.reshape(X, (1, -1))
    elif nd > 2:
        raise ValueError(
            "Input data must be two-dimensional. Given dimensions: " f"{nd:d} {X.shape}"
        )
    return X


class ClassifierBase:
    def __init__(
        self, Xtr: np.ndarray, class_labels: np.ndarray, weights: Optional[np.ndarray]
    ):
        assert Xtr.shape[0] == class_labels.shape[0]

        self.Xtr = Xtr
        self.labels = class_labels
        self.weights = weights

    def train(self) -> None:
        """train the classifier on the data given during initialisation"""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        take in an array/vector of locations X, shape (n, d) or (d, )
        and return the class-conditional probabilities of class 1.
        """
        raise NotImplementedError


class XGBoostClassifier(ClassifierBase):
    def __init__(
        self,
        Xtr: np.ndarray,
        class_labels: np.ndarray,
        weights: Optional[np.ndarray],
        xgb_settings: Dict[str, Any] = {},
    ) -> None:
        super(XGBoostClassifier, self).__init__(Xtr, class_labels, weights)

        # only import xgboost if we're using it
        from xgboost import XGBClassifier

        # default xgboost settings. taken from the BORE paper supplement (J.2)
        # http://proceedings.mlr.press/v139/tiao21a/tiao21a-supp.pdf
        self.xgb_settings = {
            "n_estimators": 100,  # boosting rounds
            "learning_rate": 0.3,  # eta
            "min_child_weight": 1,
            "max_depth": 6,
            # "min_split_loss": 1,
            # the commands below are to avoid warning messages
            "use_label_encoder": False,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            # xgboost seems to be faster without parallelisation, although this
            # may will likely not be true for all use-cases
            "n_jobs": 1,
        }

        # update the settings with any that are user-specified.
        self.xgb_settings.update(xgb_settings)

        # instantiate the classifier
        self.clf = XGBClassifier(**self.xgb_settings)

    def train(self):
        self.clf.fit(self.Xtr, self.labels, sample_weight=self.weights)

    def predict(self, X):
        X = ensure_2d(X)
        _, pi_x = self.clf.predict_proba(X).T  # class 0, class 1
        return pi_x


class FCNetClassifier(ClassifierBase):
    def __init__(
        self,
        Xtr: np.ndarray,
        class_labels: np.ndarray,
        weights: Optional[np.ndarray],
        activation: str = "elu",  # BORE paper activation
        layer_size: int = 32,  # BORE paper layer size
        n_layers: int = 2,  # BORE paper number of layers
        batch_size: int = 64,  # B in BORE paper
        gradient_steps: int = 100,  # S in BORE paper
        use_batchnorm: bool = False,
    ):
        super(FCNetClassifier, self).__init__(Xtr, class_labels, weights)

        # only import from bore and tensorflow if we're using the model
        from bore.models import MaximizableSequential
        from tensorflow.keras.layers import Dense, BatchNormalization, Input
        from tensorflow.keras.losses import BinaryCrossentropy

        # create the model
        self.model = MaximizableSequential()
        self.model.add(Input(shape=(Xtr.shape[1],)))

        for _ in range(n_layers):
            self.model.add(Dense(layer_size, activation=activation))

            if use_batchnorm:
                self.model.add(BatchNormalization())

        # no sigmoid layer as we calculate bce from logits
        self.model.add(Dense(1))
        self.model.compile(
            optimizer="adam", loss=BinaryCrossentropy(from_logits=True),
        )

        # store the training settings
        self.gradient_steps = gradient_steps
        self.batch_size = min(self.Xtr.shape[0], batch_size)

    def train(self, return_history: bool = False):
        # calculate the number of epochs to run. this comes from
        # appendix J.3. of the BORE paper.
        total_evals, dim = self.Xtr.shape
        timestep = total_evals - problem_sets.get_number_of_samples(dim)

        B = self.batch_size
        S = self.gradient_steps  # steps per iter
        N = timestep + 1  # current BO iter
        M = np.ceil(N / B)
        E = np.floor(S / M).astype("int")  # number of epochs

        history = self.model.fit(
            self.Xtr,
            self.labels,
            epochs=E,
            batch_size=B,
            verbose=0,
            shuffle=True,
            sample_weight=self.weights,
        )

        if return_history:
            return history

    def predict(self, X):
        X = ensure_2d(X)
        pi_x = self.model.predict(X)
        # n by 1 predictions need flattening
        pi_x = pi_x.ravel()  # type: ignore
        return pi_x
