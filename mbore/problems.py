import os
import numpy as np

from pymoo.factory import get_problem
from typing import Tuple
from . import _rw_problems


class REF_POINTS_DTLZ:
    def __init__(self):
        DTLZ1 = {2: 120, 5: 450, 10: 1000, 25: 1800, 50: 3500, 100: 6500}
        DTLZ245 = {2: 2, 5: 2, 10: 4, 25: 7, 50: 13, 100: 25}
        DTLZ3 = {2: 250, 5: 1000, 10: 2000, 25: 4000, 50: 10000, 100: 21000}
        DTLZ6 = {2: 2.5, 5: 5, 10: 10, 25: 25, 50: 50, 100: 100}
        DTLZ7 = {
            2: [1.5, 23],
            5: [1.5, 60],
            10: [1.5, 110],
            25: [1.5, 250],
            50: [1.5, 600],
            100: [1.5, 1200],
        }
        self.prob_dict = {
            1: DTLZ1,
            2: DTLZ245,
            3: DTLZ3,
            4: DTLZ245,
            5: DTLZ245,
            6: DTLZ6,
            7: DTLZ7,
        }

    def get_ref_point(self, problem_id: int, dim: int, fdim: int):
        prob_d = self.prob_dict[problem_id]
        bound = prob_d[dim]

        if isinstance(bound, list):
            ref_point = ([bound[0]] * (fdim - 1)) + [bound[1]]
        else:
            ref_point = [bound] * fdim

        return np.array(ref_point, dtype="float")


class IDEAL_POINTS_DTZ:
    def get_ideal_point(self, problem_id: int, dim: int, fdim: int):
        ideal_pt = np.zeros(fdim, dtype="float")

        if problem_id == 7:
            ideal_pt[-1] = {2: 2.307, 3: 2.614, 5: 3.228, 10: 4.763}[fdim]

        return ideal_pt


class REF_POINTS_WFG:
    def get_ref_point(self, problem_id: int, dim: int, fdim: int):
        ref_point = 2 * np.arange(1, fdim + 1) + 1 + 1e-6
        return ref_point


class IDEAL_POINTS_WFG:
    def get_ideal_point(self, problem_id: int, dim: int, fdim: int):
        return np.zeros(fdim, dtype="float")


class REF_POINTS_RW:
    def __init__(self):
        self.prob_dict = {
            21: [2995, 0.051],
            24: [6005, 45],
            31: [817, 8250000, 19360000],
            32: [334, 17600, 425100000],
            # 33: [8.5, 9.1, 13200000000],
            34: [1705, 11.8, 0.27],
            37: [1.01, 1.25, 1.1],
            41: [43, 4.5, 13.1, 14.2],
            42: [0, 20100, 31100, 15.4],
            61: [83100, 1351, 2854000, 16028000, 358000, 99800],
            91: [42.7, 1.6, 350, 1.1, 1.57, 1.75, 1.25, 1.3, 1.06],
        }

    def get_ref_point(self, problem_id: int, dim: int, fdim: int):
        return np.array(self.prob_dict[problem_id], dtype="float")


class IDEAL_POINTS_RW:
    def __init__(self):
        self.prob_dict = {
            21: [1237, 0.002],
            24: [60.5, 0],
            31: [0, 0.3, 0],
            32: [0.01, 0.0004, 0],
            # 33: [0.01, 0, 0],
            34: [-0.73, 1.13, 0],
            37: [0, 0, -0.44],
            41: [15.5, 3.5, 10.6, 0],
            42: [-2757, 3962, 1947, 0],
            61: [63840, 30, 285346, 183749, 7.2, 0],
            91: [15.5, 0, 0, 0.09, 0.36, 0.52, 0.73, 0.61, 0.66],
        }

    def get_ideal_point(self, problem_id: int, dim: int, fdim: int):
        return np.array(self.prob_dict[problem_id], dtype="float")


class Problem:
    def __init__(
        self, problem_name: str, problem_id: int, dim: int, fdim: int
    ):
        self.problem_name = problem_name.upper()
        self.problem_id = problem_id
        self.dim = dim
        self.fdim = fdim

        self.name = f"{self.problem_name:s}{self.problem_id:d}"
        self.name += f" (d={self.dim:d}, obj={self.fdim:d})"

    def get_reference_points(self) -> Tuple[np.ndarray, np.ndarray]:
        return get_points(
            self.problem_name, self.problem_id, self.dim, self.fdim
        )

    def get_ideal_point(self):
        pass

    def __repr__(self):
        return self.name

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def get_pareto_front(self) -> np.ndarray:
        return get_pareto_front(
            self.problem_name,
            self.problem_id,
            self.fdim,
            pareto_dir="pareto_fronts",
        )

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RW(Problem):
    def __init__(self, problem_id: int, dim: int, fdim: int):
        super(RW, self).__init__("RW", problem_id, dim, fdim)

        # get the problem and check it matches the input dim/fdim
        prob_class = getattr(_rw_problems, f"RE{problem_id:d}")
        self.problem = prob_class()
        assert dim == self.problem.n_variables
        assert fdim == self.problem.n_objectives
        assert self.problem.n_constraints == 0

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.problem.lbound, self.problem.ubound

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = np.reshape(X, (1, -1))

        ret = np.zeros((X.shape[0], self.fdim))

        for i, x in enumerate(X):
            ret[i] = self.problem.evaluate(x)

        return ret


class PyMooProblem(Problem):
    def __init__(self, name: str, problem_id: int, dim: int, fdim: int):
        super(PyMooProblem, self).__init__(name, problem_id, dim, fdim)

        self.problem = get_problem(
            name=f"{self.problem_name.lower():s}{self.problem_id:d}",
            n_var=self.dim,
            n_obj=self.fdim,
        )

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.problem.bounds()

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return self.problem.evaluate(X)  # type: ignore


class DTLZ(PyMooProblem):
    def __init__(self, problem_id: int, dim: int, fdim: int):
        super(DTLZ, self).__init__("DTLZ", problem_id, dim, fdim)


class WFG(PyMooProblem):
    def __init__(self, problem_id: int, dim: int, fdim: int):
        super(WFG, self).__init__("WFG", problem_id, dim, fdim)


def get_points(
    problem_name, problem_id, dim, fdim
) -> Tuple[np.ndarray, np.ndarray]:
    problem_name = problem_name.upper()
    avaliable_problems = ["DTLZ", "WFG"]

    if problem_name == "DTLZ":
        rp_class = REF_POINTS_DTLZ
        id_class = IDEAL_POINTS_DTZ
    elif problem_name == "WFG":
        rp_class = REF_POINTS_WFG
        id_class = IDEAL_POINTS_WFG
    elif problem_name == "RW":
        rp_class = REF_POINTS_RW
        id_class = IDEAL_POINTS_RW

    else:
        raise ValueError(
            f"Problem type not {problem_name:s} not avaliable."
            f" Choose from: {avaliable_problems}"
        )

    ref_point = rp_class().get_ref_point(problem_id, dim, fdim)
    ideal_point = id_class().get_ideal_point(problem_id, dim, fdim)

    return ref_point, ideal_point


def get_pareto_front(problem_name, problem_id, fdim, pareto_dir):
    filename = f"{problem_name.upper():s}{problem_id}_obj_{fdim}.csv"
    filepath = os.path.join(pareto_dir, filename)
    if not os.path.exists(filepath):
        raise ValueError(f"No Pareto front values were found at: {filepath:s}")

    return np.loadtxt(filepath, dtype="float", delimiter=",")
