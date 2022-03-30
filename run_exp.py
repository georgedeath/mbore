"""MBORE Experiment Runner
Usage:
    run_exp.py <problem_name> <problem_id> <problem_dim> <problem_fdim> <run_no>
               <scalarization> <classifier> <optimizer> <gamma_start>
               [--gamma_end=<gamma_end>] [--gamma_schedule=<gamma_schedule>]
               [--budget=<budget>] [--verbose] [--allcores]

Arguments:
    <problem_name>   Type of problem to be optimised, e.g. DTLZ.
    <problem_id>     Version of the problem, e.g. 1.
    <problem_dim>    Dimensionality of the problem (decision space).
    <problem_fdim>   Number of objectives.
    <run_no>         Run number, typically 1 to 51 -- training data must exist
                     for this.
    <scalarization>  Method used to convert objective values to scalar, e.g.
                     HypI or DomRank.
    <classifier>     Type of <classifier>, either XGB or FCNet.
    <optimizer>      Optimization method for the <classifier>, choose from:
                     Sobol, CMAES or Grad (Grad only avaliable for FCNet).
    <gamma_start>    Starting gamma value.

Options:
    --gamma_end=<ge>       Ending gamma value (only for use with schedule)
                           [default: Unused]
    --gamma_schedule=<gs>  Scheduler for changing gamma values over time.
                           [default Unused]
    --budget=<b>           Number of function values to optimise for
                           [default: 300]
    --verbose              Print the status of the optimisation.
    --allcores             Use all processor cores on the machine.
"""

import os
from docopt import docopt

if __name__ == "__main__":
    args = docopt(__doc__)  # type:ignore

    # parse all the args
    problem_name = args["<problem_name>"]
    problem_id = int(args["<problem_id>"])
    dim = int(args["<problem_dim>"])
    fdim = int(args["<problem_fdim>"])
    run_no = int(args["<run_no>"])
    scalarizer = args["<scalarization>"]
    model = args["<classifier>"]
    model_opt_method = args["<optimizer>"]
    gamma_start = float(args["<gamma_start>"])
    if args["--gamma_end"] == "Unused":
        gamma_end = None
    else:
        gamma_end = float(args["--gamma_end"])
    if args["--gamma_schedule"] == "Unused":
        gamma_schedule = None
    else:
        gamma_schedule = args["--gamma_schedule"]
    budget = int(args["--budget"])
    verbose = args["--verbose"]
    allcores = args["--allcores"]

    # if we're not using all cores, set everything to use 1
    if not allcores:
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

    # we need to import after (optionally) setting all needed cores
    from mbore import opt

    if model == "GP":
        gamma_start = None
        gamma_end = None
        gamma_schedule = None

    opt.optimize(
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
        data_dir="data",
        save_dir="results",
        save_every=10,
        verbose=verbose,
    )
