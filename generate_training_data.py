import os
import sys
import numpy as np
from pyDOE2 import lhs
from mbore import problem_sets, problems, util


def generate_training_data_LHS(
    problem_name: str,
    problem_id: int,
    dim: int,
    fdim: int,
    data_dir: str,
    n_samples: int,
    n_exp_start: int = 1,
    n_exp_end: int = 51,
):
    exp_nos = np.arange(n_exp_start, n_exp_end + 1)

    problem_class = getattr(problems, problem_name.upper())
    problem = problem_class(problem_id, dim, fdim)
    lb, ub = problem.get_bounds()
    ref_point, ideal_point = problem.get_reference_points()

    for run_no in exp_nos:
        save_path = util.generate_data_filename(problem_name, problem_id, dim, fdim, run_no, data_dir)

        if os.path.exists(save_path):
            print(f"File exists, skipping: {save_path:s}")
            continue

        # generate samples and rescale to problem domain
        Xtr = lhs(dim, n_samples, criterion="maximin")
        Xtr = Xtr * (ub - lb) + lb

        # evaluate
        Ytr = problem.evaluate(Xtr)

        # check all points are within the reference point
        valid_mask = np.all(Ytr <= ref_point, axis=1)

        if not np.all(valid_mask):
            with np.printoptions(precision=6):
                print("Reference point", ref_point)
                print(save_path)
                for i in np.where(~valid_mask)[0]:
                    print("Fitness values:", Ytr[i])
            raise ValueError()

        # save
        np.savez_compressed(
            save_path, Xtr=Xtr, Ytr=Ytr, problem_name=problem_name, problem_id=problem_id, run_no=run_no,
        )
        print(f"Saved: {save_path:s}")


if __name__ == "__main__":
    data_dir = "data"

    # make the dir if needed
    os.makedirs(data_dir, exist_ok=True)

    # get the problem name and associated information dict
    problem_name = sys.argv[1]
    problem_name, prob_dict = problem_sets.get_problem_dict(problem_name)

    # create each set of lhs samples
    for problem_id in prob_dict:
        for dim, fdims in prob_dict[problem_id]:
            # 10 lhs samples per dimension
            ns = problem_sets.get_number_of_samples(dim)

            for fdim in fdims:
                generate_training_data_LHS(
                    problem_name,
                    problem_id,
                    dim,
                    fdim,  # type:ignore
                    data_dir,
                    ns,
                )
