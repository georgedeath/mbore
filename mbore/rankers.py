import numpy as np
import pygmo as pg
from pymoo.factory import get_reference_directions
from typing import Tuple, Union


class BaseRanker:
    """
    Functionality we want:
        instantiation(F):
            - F: n by n_obj numpy ndarray of objective values

        get_ranks():
            - return the ranking of the objective values in indices form
              ideally this should cache the result unless the result is
              stochastic

    """

    def __init__(self, reference_point: np.ndarray):
        self.ref = np.ravel(reference_point)

    def _scalarize(self, F: np.ndarray) -> np.ndarray:
        """returns the scalarised version of F"""
        raise NotImplementedError

    def get_ranks(
        self, F: np.ndarray, return_scalers: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        scalars = self._scalarize(F)

        # rank the values, lower rank = larger value
        ranks = np.argsort(scalars)[::-1]

        if return_scalers:
            return ranks, scalars

        return ranks


class HypIRanker(BaseRanker):
    def _scalarize(self, F: np.ndarray) -> np.ndarray:
        # get the list of shell indices (ndf)
        ndf, _, _, _ = pg.core.fast_non_dominated_sorting(F)

        hypi = np.zeros((F.shape[0],), dtype="float")

        zero_arr = np.array([0], dtype=ndf[0].dtype)

        for shell, next_shell in zip(ndf[:-1], ndf[1:]):
            # preallocate the indices of the locations
            combined_front_inds = np.concatenate([next_shell, zero_arr])

            for location_idx in shell:
                # add each location from shell n to shell n + 1
                combined_front_inds[-1] = location_idx

                # calculate the hypervolume of the combined front
                hypi[location_idx] = pg.core.hypervolume(
                    F[combined_front_inds]
                ).compute(self.ref)

        # last shell: hypervolume of each individual shell member
        last_shell = ndf[-1]

        # calculate the volume of the hyper-rectangle, with edges spanning
        # from each element of the shell to the reference points, by simply
        # taking the product of its edge lengths.
        hypi[last_shell] = np.prod(
            self.ref[np.newaxis, :] - F[last_shell], axis=1
        )

        return hypi


class DomRankRanker(BaseRanker):
    def _scalarize(self, F: np.ndarray) -> np.ndarray:
        # get non-domination ranks of F
        _, _, dom_count, _ = pg.core.fast_non_dominated_sorting(F)

        domrank = 1 - (dom_count / (F.shape[0] - 1))

        return domrank


class HypervolumeContributionRanker(BaseRanker):
    def _scalarize(self, F: np.ndarray) -> np.ndarray:
        # in order for the scalarisation to make sense and be pareto-compliant,
        # we first calculate the exclusive hypervolume contribution of each
        # pareto. we then add the largest contribution from each shell to its
        # predecessor, i.e. hv[shell[i]] += np.max(hv[shell[i+1]). this ensures
        # shell i's values are always larger than shell i+1's.

        # get the list of shell indices (ndf)
        ndf, _, _, _ = pg.core.fast_non_dominated_sorting(F)
        hvc = np.zeros((F.shape[0],), dtype="float")

        for shell_idx in range(len(ndf) - 1, -1, -1):
            shell = ndf[shell_idx]

            # first, calculate the exclusive hypervolume for each shell
            # (in reverse order), and add it to the current hypervolume value
            hvc[shell] += pg.core.hypervolume(F[shell]).contributions(self.ref)

            # then, if we're not dealing with the last shell, add the maximum
            # of this shell's contributions (plus and prev added contributions)
            # to the previous shell's values
            if shell_idx > 0:
                hvc[ndf[shell_idx - 1]] += np.max(hvc[shell])

        return hvc


class ParEGORanker(BaseRanker):
    def __init__(self, reference_point: np.ndarray):
        super(ParEGORanker, self).__init__(reference_point)

        # number of reference vectors for a given number of objectives
        self._number_of_ref_vectors = {
            2: 100,
            3: 105,
            4: 120,
            5: 126,
            6: 132,
            7: 112,
            8: 156,
            9: 90,
            10: 275,
        }

        n_obj = self.ref.size
        if n_obj not in self._number_of_ref_vectors:
            raise ValueError(
                "ParEgoRanker is only configured to work with"
                f" {list(self._number_of_ref_vectors.keys())}"
                f" number of objectives, it was given: {n_obj:d}"
            )

        self.n_ref = self._number_of_ref_vectors[n_obj]

        # generate the reference vectors once
        self.ref_vectors = get_reference_directions(
            "energy", n_obj, self.n_ref
        )

        # Tchebycheff coefficient -- from ParEGO paper
        self.rho = 0.05

    def _scalarize(self, F: np.ndarray, ref_vec_idx: int = None) -> np.ndarray:
        # if we're not given a reference vector index to use (the default!)
        # then select random reference vector
        if ref_vec_idx is None:
            ref_vec_idx = np.random.choice(self.n_ref)

        lambdas = self.ref_vectors[ref_vec_idx, :]

        # rescale the fitness values each reside in [0, 1]
        fmins = np.min(F, axis=0, keepdims=True)
        fmaxes = np.max(F, axis=0, keepdims=True)
        F = (F - fmins) / (fmaxes - fmins)

        # scalarize via the Tchebycheff function (from ParEGO paper, eqn. 2):
        # max_i(lambda_i * f_i(x)) + rho * sum_i (lambda_i * f_i(x))
        lF = lambdas[np.newaxis, :] * F
        scalarized = np.max(lF, axis=1) + self.rho * np.sum(lF, axis=1)

        # flip so that largest = best and worst = 0
        return -scalarized

    # we have to override the base class signature because we need to be
    # able to specify the reference vector index on occasion
    def get_ranks(
        self,
        F: np.ndarray,
        return_scalers: bool = False,
        ref_vec_idx: int = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        scalars = self._scalarize(F, ref_vec_idx)

        # rank the values, lower rank = larger value
        ranks = np.argsort(scalars)[::-1]

        if return_scalers:
            return ranks, scalars

        return ranks
