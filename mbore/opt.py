import os
import time
import numpy as np
import torch
from . import gp, model_training, problems, transforms, util
from gpytorch.settings import cholesky_jitter
from typing import Optional, Union


def optimize(
    problem_name: str,
    problem_id: int,
    dim: int,
    fdim: int,
    run_no: int,
    budget: int,
    scalarizer: str,  # HypI or DomRank
    model: str,  # XGB or FCNet,
    model_opt_method: str,  # Sobol, SobolBoltzmann, CMAES, Grad (for NN)
    acq_func: Optional[str] = None,
    gamma_start: float = 1 / 3,
    gamma_end: Union[float, None] = None,
    gamma_schedule: Union[str, None] = None,
    data_dir: str = "data",
    save_dir: str = "results",
    save_every: int = 10,
    verbose: bool = False,
):
    # get the problem, its bounds, reference point and pareto front
    problem_class = getattr(problems, problem_name.upper())
    problem = problem_class(problem_id, dim, fdim)
    lb, ub = problem.get_bounds()
    ref_point, _ = problem.get_reference_points()

    # set up the ranker
    ranker_class = util.get_ranker(scalarizer)
    ranker = ranker_class(ref_point)

    # load the training data
    data_path = util.generate_data_filename(
        problem_name, problem_id, dim, fdim, run_no, data_dir
    )

    with np.load(data_path) as fd:
        Xtr = fd["Xtr"]
        Ytr = fd["Ytr"]
        start_amount = Xtr.shape[0]

    # generate the location at which to save the data
    save_path = util.generate_save_filename(
        problem_name=problem_name,
        problem_id=problem_id,
        dim=dim,
        fdim=fdim,
        run_no=run_no,
        scalarizer=scalarizer,  # HypI or DomRank
        model=model,  # XGB or FCNet,
        model_opt_method=model_opt_method,  # Sobol, SobolBoltzmann, CMAES, Grad (for NN)
        gamma_start=gamma_start,
        gamma_end=gamma_end,
        gamma_schedule=gamma_schedule,
        save_dir=save_dir,
        acq_func=acq_func,
    )

    # get the gamma scheduler
    gs_class = util.get_scheduler("Fixed" if gamma_schedule is None else gamma_schedule)
    gamma_scheduler = gs_class(gamma_start, gamma_end, budget)

    # timing storage
    timing = []

    # resume the run if it has already been started
    if os.path.exists(save_path):
        with np.load(save_path, allow_pickle=True) as fd:
            Xtr = fd["Xtr"]
            Ytr = fd["Ytr"]
            timing = list(fd["timing"])  # convert the array back to a list

        if verbose:
            print(
                "Data already exists, loading and resuming run:",
                f"{save_path:s} ({Xtr.shape[0]:d})",
            )

    # starting value for the iteration counter, if starting a new run this
    # will be zero, else it'll be the number of already-evaluated solutions,
    # excluding the initial training data
    starting_t_value = Xtr.shape[0] - start_amount

    for t in range(starting_t_value + 1, budget + 1):
        gamma = gamma_scheduler(t)

        _, scalers = ranker.get_ranks(Ytr, return_scalers=True)

        # rescale decision vectors to [0, 1] and the associated bounds
        xtransformer = transforms.ZeroOneTransform(Xtr, lb, ub)
        sXtr = xtransformer.transform(Xtr)
        slb, sub = np.zeros(dim), np.ones(dim)

        opt_budget = 1024 * dim

        # start the method timing
        start_time = time.time()

        # take optimisation step -- note that the returned value is scaled
        if model == "GP":
            # rescale the scalers to have unit variance and zero mean
            scalers = np.reshape(scalers, (-1, 1))
            scalertransformer = transforms.StandardizeTransform(scalers)
            sscalers = scalertransformer.transform(scalers)

            sXnext = opt_step_GP(
                train_x=sXtr,
                train_y=sscalers,
                lb=slb,
                ub=sub,
                opt_budget=opt_budget,
                train_restarts=10,
                opt_restarts=10,
                verbose=verbose,
            )

        else:
            sXnext = opt_step_clf(
                Xtr=sXtr,
                y=scalers,
                lb=slb,
                ub=sub,
                model=model,
                model_opt_method=model_opt_method,
                gamma=gamma,
                opt_budget=opt_budget,
                weight_type=acq_func,
            )

        # method over -- get its runtime
        total_time = time.time() - start_time

        # un-scale the suggested value back to the original decision space
        Xnext = xtransformer.untransform(sXnext)

        # evaluate function
        Ynext = problem.evaluate(Xnext)

        # store the newly evaluate solution, its fitness and timing
        Xtr = np.concatenate((Xtr, Xnext.reshape(1, dim)))
        Ytr = np.concatenate((Ytr, Ynext.reshape(1, fdim)))
        timing.append(total_time)

        # save if we need to
        if t % save_every == 0:
            np.savez(
                save_path,
                Xtr=Xtr,
                Ytr=Ytr,
                timing=timing,
                scalarizer=scalarizer,
                model=model,
                model_opt_method=model_opt_method,
                gamma_start=gamma_start,
                gamma_end=gamma_end,
                gamma_schedule=gamma_schedule,
            )

        # print a summary if we're being verbose
        if verbose:
            print(
                f"Iter: {t:03d}/{budget:03d}", f"Time taken: {total_time:0.2f}s",
            )


def opt_step_clf(
    Xtr, y, lb, ub, model, model_opt_method, gamma, opt_budget, weight_type
):
    x, y, z, w, _ = util.load_classification_data(Xtr, y, gamma, weight_type)

    clf = model_training.train_classifier(Xtr=x, labels=z, method=model, weights=w)

    # optimise it
    opt_class = util.get_optimizer(model_opt_method)
    opt = opt_class(clf, lb, ub, opt_budget)
    Xnext = opt.get_xnext()

    return Xnext


def opt_step_GP(
    train_x,
    train_y,
    lb,
    ub,
    opt_budget,
    train_restarts=10,
    opt_restarts=10,
    chol_jitter=1e-3,
    verbose=False,
):
    # convert bounds to torch
    lb, ub = map(lambda x: torch.tensor(x, dtype=torch.float64), [lb, ub])
    problem_bounds = torch.stack((lb, ub))
    best_f = np.max(train_y)

    model = model_training.train_gp(
        train_x,
        train_y,
        train_restarts=train_restarts,
        chol_jitter=chol_jitter,
        verbose=verbose,
    )

    # create, train and optimize the gp, using cholesky jitter to avoid PSD
    with cholesky_jitter(float=chol_jitter, double=chol_jitter, half=chol_jitter):
        Xnext = gp.ei_optimize(model, best_f, problem_bounds, opt_budget, opt_restarts)

    return Xnext.numpy().ravel()


if __name__ == "__main__":
    problem_name = "DTLZ"
    problem_id = 1
    dim = 2
    fdim = 2
    run_no = 1
    budget = 200
    scalarizer = "HypI"  # HypI or DomRank
    model = "XGB"  # XGB or FCNet,
    model_opt_method = "Sobol"  # Sobol, CMAES, Grad (for NN)
    gamma_start = 1 / 3
    gamma_end = None
    gamma_schedule = None
    data_dir = "data"
    save_dir = "results"

    optimize(
        problem_name=problem_name,
        problem_id=problem_id,
        dim=dim,
        fdim=fdim,
        run_no=run_no,
        budget=budget,
        scalarizer=scalarizer,
        model=model,
        model_opt_method=model_opt_method,
        gamma_start=gamma_start,
        gamma_end=gamma_end,
        gamma_schedule=gamma_schedule,
        data_dir=data_dir,
        save_dir=save_dir,
        verbose=True,
    )
