import sys
import mbore

if __name__ == "__main__":
    real_problem_name = sys.argv[1]

    if len(sys.argv) > 2:
        num_jobs = int(sys.argv[2])
    else:
        num_jobs = -1

    problem_name, prob_dict = mbore.problem_sets.get_problem_dict(real_problem_name)
    results_dir = f"results_{real_problem_name:s}"

    models_and_optimizers = [
        ("XGB", "Sobol"),
        ("XGB", "CMAES"),
        ("FCNet", "Grad"),
        ("GP", "EI"),
    ]

    scalarizers = [
        "HypI",
        "DomRank",
        "HypCont",
        "ParEGO",
    ]

    gamma_start = 0.33
    gamma_end = None
    gamma_schedule = None

    start_run = 1
    end_run = 21  # inclusive

    budget = 300

    mbore.util.gather_results(
        problem_name=problem_name,
        prob_dict=prob_dict,
        models_and_optimizers=models_and_optimizers,
        scalarizers=scalarizers,
        gamma_start=gamma_start,
        gamma_end=gamma_end,
        gamma_schedule=gamma_schedule,
        start_run=start_run,
        end_run=end_run,
        budget=budget,
        results_dir=results_dir,
        processed_results_dir="processed_results",
        num_jobs=num_jobs,
    )
