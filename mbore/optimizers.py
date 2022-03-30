import cma
import numpy as np
import warnings
from .classifiers import ClassifierBase, FCNetClassifier
from scipy.stats.qmc import Sobol, scale

from typing import Any, Dict
from scipy.optimize import Bounds


class OptimizerBase:
    def __init__(
        self,
        model: ClassifierBase,
        lb: np.ndarray,
        ub: np.ndarray,
        budget: int = 2000,
    ) -> None:
        self.model = model
        self.lb = lb
        self.ub = ub
        self.budget = budget

    def get_xnext(self) -> np.ndarray:
        raise NotImplementedError


class RandomSearchOptimizer(OptimizerBase):
    def __init__(
        self,
        model: ClassifierBase,
        lb: np.ndarray,
        ub: np.ndarray,
        budget: int = 2048,
        boltzman_sample: bool = True,
    ):
        super(RandomSearchOptimizer, self).__init__(model, lb, ub, budget)
        self.boltzman_sample = boltzman_sample
        self.sobol = Sobol(d=lb.size, scramble=True)

    def get_xnext(self):
        # sample from a scrambled sobol sequence in [0, 1]^d and
        # ignore warnings about not sampling a power of 2 samples
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xtest = self.sobol.random(n=self.budget)

        # rescale to original bounds
        Xtest = scale(Xtest, self.lb, self.ub)

        # evaluate with the model
        p = self.model.predict(Xtest)

        if self.boltzman_sample:
            eta = 1.0

            # standardize p first -- avoiding any tiny standard deviation
            pmean, pstd = np.mean(p), np.std(p)
            pstd = pstd if pstd > 1e-6 else 1.0
            p = (p - pmean) / pstd

            # also clip into the range we can exponentiate without failure.
            # note that mostly this won't do anything
            p = np.clip(p, -eta * 88.72, eta * 88.72)

            # temper the values and rescale so they add up to one
            p = np.exp(eta * p)
            p /= np.sum(p)

            # sample a location proportional to its tempered value
            xnext_idx = np.random.choice(self.budget, p=p, replace=False)

        # simply argmax
        else:
            # get the index/indices that are equal to the maximum
            argmax_locs = np.where(p == np.max(p))[0]

            # if there's more than one, choose randomly
            if len(argmax_locs) > 1:
                xnext_idx = np.random.choice(argmax_locs, replace=False)
            else:
                xnext_idx = argmax_locs[0]

        return Xtest[xnext_idx, :]


class CMAESOptimizer(OptimizerBase):
    def __init__(
        self,
        model: ClassifierBase,
        lb: np.ndarray,
        ub: np.ndarray,
        budget: int = 2048,
        cma_options: Dict[str, Any] = {},
    ):
        super(CMAESOptimizer, self).__init__(model, lb, ub, budget)

        self.cma_options = {
            "bounds": [list(self.lb), list(self.ub)],
            "tolfun": 1e-7,
            "maxfevals": self.budget,
            "verb_log": 0,
            "verb_disp": 0,
            "verbose": -3,  # suppress any warnings (from flat fitness)
            "CMA_stds": np.abs(self.ub - self.lb),
        }

        self.cma_options.update(cma_options)

        self.cma_sigma = 0.25
        self.cma_centroid = lambda: np.random.uniform(lb, ub)

    def get_xnext(self):
        res = cma.fmin(
            None,  # no function as specifying a parallel one
            self.cma_centroid,  # random evaluation within bounds
            self.cma_sigma,
            options=self.cma_options,
            parallel_objective=self._cma_objective,
            args=(self.model,),
            bipop=True,
            restarts=10,
        )

        xnext = res[0]
        return xnext

    @staticmethod
    def _cma_objective(X, model):
        # turn the list of decision vectors into a numpy array
        X = np.stack(X)

        # evaluate the model
        fx = model.predict(X)

        # negate because CMA-ES minimises
        fx = -fx

        # convert to a list as this is what CMA-ES expects back
        fx = fx.tolist()

        return fx


class FCNetOptimizer(OptimizerBase):
    def __init__(
        self,
        model: ClassifierBase,
        lb: np.ndarray,
        ub: np.ndarray,
        budget: int = 2048,
        n_restarts: int = 10,
    ):
        super(FCNetOptimizer, self).__init__(model, lb, ub, budget)

        # we need to check that we're only using this optimiser with a NN
        assert isinstance(model, FCNetClassifier)

        self.bounds = Bounds(lb, ub)
        self.n_restarts = n_restarts

    def get_xnext(self):
        ret = self.model.model.argmax(  # type: ignore
            bounds=self.bounds,
            num_starts=self.n_restarts,
            num_samples=self.budget,
            # this just disables the debug printing
            print_fn=lambda x: None,
        )

        xnext = ret.x
        return xnext
