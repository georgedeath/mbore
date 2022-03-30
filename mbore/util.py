import os
import warnings
import tqdm.auto
import numpy as np
from . import (
    rankers,
    classifiers,
    optimizers,
    problems,
    problem_sets,
    schedulers,
    transforms,
)
from typing import Callable, Dict, List, Tuple, Union
from pymoo.factory import get_performance_indicator
from pygmo.core import hypervolume
from joblib import Parallel, delayed
from scipy.stats import median_abs_deviation, wilcoxon
from statsmodels.stats.multitest import multipletests


def generate_data_filename(
    problem_name: str,
    problem_id: int,
    dim: int,
    fdim: int,
    run_no: int,
    data_dir: str,
):

    fname_components = [
        f"{problem_name.upper():s}{problem_id:d}",
        f"_d={dim:d}",
        f"_o={fdim:d}",
        f"_{run_no:d}",
        ".npz",
    ]

    fname = "".join(fname_components)

    return os.path.join(data_dir, fname)


def generate_save_filename(
    problem_name: str,
    problem_id: int,
    dim: int,
    fdim: int,
    run_no: Union[int, None],
    scalarizer: str,  # HypI or DomRank
    model: Union[str, None],  # XGB or FCNet,
    model_opt_method: Union[str, None],  # Sobol, CMAES, Grad (for NN)
    gamma_start: Union[float, None],
    gamma_end: Union[float, None],
    gamma_schedule: Union[str, None],
    save_dir: str,
):
    fname_components = [
        f"{problem_name.upper():s}{problem_id:d}",
        f"_d={dim:d}",
        f"_o={fdim:d}",
        f"_{run_no:d}" if run_no is not None else "",
        f"_{scalarizer:s}",
        f"_{model:s}",
        f"_{model_opt_method:s}",
        f"_g={gamma_start:0.2f}" if gamma_start is not None else "",
        f"-{gamma_end:0.2f}({gamma_schedule:s})"
        if gamma_schedule is not None
        else "",
        ".npz",
    ]
    fname = "".join(fname_components)

    return os.path.join(save_dir, fname)


def get_ranker(name: str):
    names_to_classes = {
        "HypI": rankers.HypIRanker,
        "DomRank": rankers.DomRankRanker,
        "HypCont": rankers.HypervolumeContributionRanker,
        "ParEGO": rankers.ParEGORanker,
    }

    if name not in names_to_classes:
        raise ValueError(
            f"Invalid ranker given: {name:s}. Valid options are:"
            f" {names_to_classes.keys()}"
        )

    return names_to_classes[name]


def get_classifier(name: str):
    names_to_classes = {
        "XGB": classifiers.XGBoostClassifier,
        "FCNet": classifiers.FCNetClassifier,
    }
    if name not in names_to_classes:
        raise ValueError(
            f"Invalid classifier given: {name:s}. Valid options are:"
            f" {names_to_classes.keys()}"
        )

    return names_to_classes[name]


def get_optimizer(name: str):
    names_to_classes = {
        "Sobol": optimizers.RandomSearchOptimizer,
        "CMAES": optimizers.CMAESOptimizer,
        "Grad": optimizers.FCNetOptimizer,
    }
    if name not in names_to_classes:
        raise ValueError(
            f"Invalid optimizer given: {name:s}. Valid options are:"
            f" {names_to_classes.keys()}"
        )

    return names_to_classes[name]


def get_scheduler(name: str):
    names_to_classes = {
        "Fixed": schedulers.FixedScheduler,
        "Linear": schedulers.LinearScheduler,
    }
    if name not in names_to_classes:
        raise ValueError(
            f"Invalid scheduler given: {name:s}. Valid options are:"
            f" {names_to_classes.keys()}"
        )

    return names_to_classes[name]


def func_to_par(
    i,
    results_path,
    budget,
    n_expected,
    ref_point,
    ideal_point,
    scaled_indicator_igd,
):
    if os.path.exists(results_path):
        with np.load(results_path, allow_pickle=True) as fd:
            Xtr = fd["Xtr"]
            Ytr = fd["Ytr"]
            timing = fd["timing"]

        n = Ytr.shape[0]

        if n != n_expected:
            print(f"Run not finished: {results_path:s} ({n}/{n_expected})")

        else:
            # rescale to reside in [0, 1]
            sYtr = (Ytr - ideal_point) / (ref_point - ideal_point)

            hv = np.zeros(budget + 1)
            igdplus = np.zeros(budget + 1)

            sref_point = np.ones_like(ref_point)
            for idx, j in enumerate(
                range(n_expected - budget, n_expected + 1)
            ):
                sYtrj = sYtr[:j]

                hv[idx] = hypervolume(sYtrj).compute(sref_point)
                igdplus[idx] = scaled_indicator_igd.do(sYtrj)

            return (
                i,
                Xtr,
                sYtr,
                timing,
                hv,
                igdplus,
            )

    else:
        print(f"File not found: {results_path:s}")

    return [i] + [None] * 5


def gather_results(
    problem_name: str,
    prob_dict: Dict[int, List[Tuple[int, List[int]]]],
    models_and_optimizers: List[Tuple[str, str]],
    scalarizers: List[str],
    gamma_start: float,
    gamma_end: Union[float, None] = None,
    gamma_schedule: Union[str, None] = None,
    start_run: int = 1,
    end_run: int = 21,
    budget: int = 300,
    results_dir: str = "results",
    processed_results_dir: str = "processed_results",
    num_jobs: int = -1,
):
    runs = range(start_run, end_run + 1)
    n_runs = len(runs)

    total = (
        len(models_and_optimizers)
        * len(scalarizers)
        * problem_sets.get_number_of_problems(prob_dict)
    )

    with tqdm.auto.tqdm(total=total) as pbar, Parallel(
        n_jobs=num_jobs
    ) as parallel:
        for scalarizer in scalarizers:
            for model, model_opt_method in models_and_optimizers:
                for problem_id in prob_dict:
                    for dim, fdims in prob_dict[problem_id]:
                        for fdim in fdims:

                            # get the problem and related info
                            problem_class = getattr(
                                problems, problem_name.upper()
                            )
                            problem = problem_class(problem_id, dim, fdim)
                            (
                                ref_point,
                                ideal_point,
                            ) = problem.get_reference_points()
                            pf = problem.get_pareto_front()

                            # rescale the pareto front to [0, 1]
                            spf = (pf - ideal_point) / (
                                ref_point - ideal_point
                            )

                            # therefore, the indicator function should be also
                            # be defined on [0, 1]
                            scaled_indicator_igd = get_performance_indicator(
                                "igd+", spf
                            )

                            # storage arrays
                            n_expected = (
                                budget
                                + problem_sets.get_number_of_samples(dim)
                            )
                            Xtrs = np.zeros((n_runs, n_expected, dim))
                            Ytrs = np.zeros((n_runs, n_expected, fdim))
                            timing = np.zeros((n_runs, budget))
                            hvs = np.zeros((n_runs, budget + 1))
                            igds = np.zeros((n_runs, budget + 1))

                            if model == "GP":
                                meth_gamma_start = None
                                meth_gamma_end = None
                                meth_gamma_schedule = None
                            else:
                                meth_gamma_start = gamma_start
                                meth_gamma_end = gamma_end
                                meth_gamma_schedule = gamma_schedule

                            # generate the paths of each run
                            paths = [
                                (
                                    i,
                                    generate_save_filename(
                                        problem_name=problem_name,
                                        problem_id=problem_id,
                                        dim=dim,
                                        fdim=fdim,
                                        run_no=run_no,
                                        scalarizer=scalarizer,
                                        model=model,
                                        model_opt_method=model_opt_method,
                                        gamma_start=meth_gamma_start,
                                        gamma_end=meth_gamma_end,
                                        gamma_schedule=meth_gamma_schedule,
                                        save_dir=results_dir,
                                    ),
                                )
                                for (i, run_no) in enumerate(runs)
                            ]

                            # evaluate, in parallel, the igd and hv
                            res = parallel(
                                delayed(func_to_par)(
                                    i,
                                    result_path,
                                    budget,
                                    n_expected,
                                    ref_point,
                                    ideal_point,
                                    scaled_indicator_igd,
                                )
                                for (i, result_path) in paths
                            )
                            for i, Xtr, Ytr, eval_timing, hv, igdplus in res:  # type: ignore
                                if Xtr is None:
                                    continue

                                Xtrs[i] = Xtr
                                Ytrs[i] = Ytr
                                timing[i] = eval_timing
                                hvs[i] = hv
                                igds[i] = igdplus

                            save_path = generate_save_filename(
                                problem_name=problem_name,
                                problem_id=problem_id,
                                dim=dim,
                                fdim=fdim,
                                run_no=None,
                                scalarizer=scalarizer,
                                model=model,
                                model_opt_method=model_opt_method,
                                gamma_start=meth_gamma_start,
                                gamma_end=meth_gamma_end,
                                gamma_schedule=meth_gamma_schedule,
                                save_dir=processed_results_dir,
                            )

                            np.savez_compressed(
                                save_path,
                                Xtrs=Xtrs,
                                Ytrs=Ytrs,
                                timing=timing,
                                hvs=hvs,
                                igds=igds,
                            )

                            pbar.update()


def stats_test(
    best_seen_values: np.ndarray,
    argmethod: Callable,
    wilcoxon_side: str,
    alpha: float = 0.05,
):
    n_methods, _ = best_seen_values.shape

    best_mask = np.zeros(n_methods, dtype="bool")

    # calculate the median and MAD
    medians = np.median(best_seen_values, axis=1)
    mads = median_abs_deviation(best_seen_values, axis=1)

    # get the index of the best method
    best_method_idx = argmethod(medians)
    best_mask[best_method_idx] = True

    # compare the best method to the others, so first work out which are the
    # non-best method's indices
    non_best_indices = [i for i in range(n_methods) if i != best_method_idx]

    # perform a ONE-SIDED wilcoxon signed rank test between best and
    # all other methods
    p_values = []
    for i in non_best_indices:
        # Note: a ValueError will be thrown if they are all
        #       identical

        # ignore the warnings about ties
        # (in which case it switches to a normal approximation)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, p_value = wilcoxon(
                x=best_seen_values[best_method_idx, :],
                y=best_seen_values[i, :],
                alternative=wilcoxon_side,
            )
        p_values.append(p_value)

    # perform stats test on the p-values
    reject_hyp, _, _, _ = multipletests(p_values, alpha=alpha, method="holm")

    for i, reject in zip(non_best_indices, reject_hyp):
        # if we can't reject the hypothesis that a
        # technique is statistically equivalent to the
        # best method, then set it to being equal
        if not reject:
            best_mask[i] = True

    return medians, mads, best_mask


def get_class_labels_from_ranks(ranks, gamma):
    """labels the top gamma proportion of ranks class 1, the rest class 0."""
    n = ranks.shape[0]

    tau = int(n * gamma)

    labels = np.zeros((n,), dtype="int")
    labels[ranks[:tau]] = 1

    return labels


def standardize_vector(v):
    v = np.reshape(v, (-1, 1))
    scalertransformer = transforms.StandardizeTransform(v)
    sv = scalertransformer.transform(v)

    return sv
