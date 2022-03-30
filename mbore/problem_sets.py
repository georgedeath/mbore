from typing import Dict, Tuple, List


def get_number_of_samples(dim: int) -> int:
    """
    Returns number of initial samples to use for a given dimensionality (d),
    with the rule that if d <= 10 then return 10 * d else return 5 * d.
    """
    multiplier = 10 if dim <= 10 else 5
    return multiplier * dim


def get_problem_dict(
    problem_name: str,
) -> Tuple[str, Dict[int, List[Tuple[int, List[int]]]]]:
    # creates a dictionary defining the problems as follows:
    # {problem_id: [(dim: [obj_dim_1, obj_dim_2, ...], dim2: [....], ...)]}

    if problem_name == "DTLZ_HD":
        # high-dimensional dtlz test problems
        # {1: [(25, [5]), (50, [5]), (100, [5])],
        # 2: [(25, [5]), (50, [5]), (100, [5])],
        # 3: [(25, [5]), (50, [5]), (100, [5])],
        # 4: [(25, [5]), (50, [5]), (100, [5])],
        # 5: [(25, [5]), (50, [5]), (100, [5])],
        # 6: [(25, [5]), (50, [5]), (100, [5])],
        # 7: [(25, [5]), (50, [5]), (100, [5])]}
        problem_name = "DTLZ"
        prob_dict = {
            problem_id: [(25, [5]), (50, [5]), (100, [5])]
            for problem_id in range(1, 8)
        }

    elif problem_name == "DTLZ":
        # dtlz test problems
        # {1: [(2, [2]), (5, [2, 3, 5]), (10, [2, 3, 5, 10])],
        #  2: [(2, [2]), (5, [2, 3, 5]), (10, [2, 3, 5, 10])],
        #  3: [(2, [2]), (5, [2, 3, 5]), (10, [2, 3, 5, 10])],
        #  4: [(2, [2]), (5, [2, 3, 5]), (10, [2, 3, 5, 10])],
        #  5: [(2, [2]), (5, [2, 3, 5]), (10, [2, 3, 5, 10])],
        #  6: [(2, [2]), (5, [2, 3, 5]), (10, [2, 3, 5, 10])],
        #  7: [(2, [2]), (5, [2, 3, 5]), (10, [2, 3, 5, 10])]}
        prob_dict = {
            problem_id: [(2, [2]), (5, [2, 3, 5]), (10, [2, 3, 5, 10])]
            for problem_id in range(1, 8)
        }

    elif problem_name == "WFG":
        # walking fish group
        # {1: [(6, [2, 3]), (8, [2, 3]), (10, [2, 3, 5])],
        #  2: [(6, [2, 3]), (8, [2, 3]), (10, [2, 3, 5])],
        #  3: [(6, [2, 3]), (8, [2, 3]), (10, [2, 3, 5])],
        #  4: [(6, [2, 3]), (8, [2, 3]), (10, [2, 3, 5])],
        #  5: [(6, [2, 3]), (8, [2, 3]), (10, [2, 3, 5])],
        #  6: [(6, [2, 3]), (8, [2, 3]), (10, [2, 3, 5])],
        #  7: [(6, [2, 3]), (8, [2, 3]), (10, [2, 3, 5])],
        #  8: [(6, [2, 3]), (8, [2, 3]), (10, [2, 3, 5])],
        #  9: [(6, [2, 3]), (8, [2, 3]), (10, [2, 3, 5])]}
        prob_dict = {
            problem_id: [(6, [2, 3]), (8, [2, 3]), (10, [2, 3, 5])]
            for problem_id in range(1, 10)
        }

    elif problem_name == "WFG_HD":
        # high-dimensional walking fish group test problems
        # {1: [(25, [10]), (50, [10]), (100, [10])],
        # 2: [(25, [10]), (50, [10]), (100, [10])],
        # 3: [(25, [10]), (50, [10]), (100, [10])],
        # 4: [(25, [10]), (50, [10]), (100, [10])],
        # 5: [(25, [10]), (50, [10]), (100, [10])],
        # 6: [(25, [10]), (50, [10]), (100, [10])],
        # 7: [(25, [10]), (50, [10]), (100, [10])],
        # 8: [(25, [10]), (50, [10]), (100, [10])],
        # 9: [(25, [10]), (50, [10]), (100, [10])]}
        problem_name = "WFG"
        prob_dict = {
            problem_id: [(20, [10]), (50, [10]), (100, [10])]
            for problem_id in range(1, 10)
        }

    elif problem_name == "RW":
        # real-world problems
        # {21: [(4, [2])],
        #  24: [(2, [2])],
        #  31: [(3, [3])],
        #  32: [(4, [3])],
        #  34: [(5, [3])],
        #  37: [(4, [3])],
        #  41: [(7, [4])],
        #  42: [(6, [4])],
        #  61: [(3, [6])],
        #  91: [(7, [9])]}
        prob_dict = {
            21: [(4, [2])],  # RE2-4-1 Four bar truss design [24]
            24: [(2, [2])],  # RE2-2-4 Hatch cover design [23]
            31: [(3, [3])],  # RE3-3-1 Two bar truss design [27]
            32: [(4, [3])],  # RE3-4-2 Welded beam design [28]
            # 33: [(4, [3])],  # RE3-4-3 Disc brake design [28] <-- do not use
            34: [(5, [3])],  # RE3-5-4 Vehicle crashworthiness design [29]
            37: [(4, [3])],  # RE3-4-7 Rocket injector design [32]
            41: [(7, [4])],  # RE4-7-1 Car side impact design [15]
            42: [(6, [4])],  # RE4-6-2 Conceptual marine design [33]
            61: [(3, [6])],  # RE6-3-1 Water resource planning [20]
            91: [(7, [9])],  # RE9-7-1 Car cab design [7]
        }

    else:
        raise ValueError(f"Invalid problem name given: {problem_name}")

    return problem_name, prob_dict


def get_number_of_problems(
    prob_dict: Dict[int, List[Tuple[int, List[int]]]]
) -> int:
    return sum(
        1
        for problem_id in prob_dict
        for _, fdims in prob_dict[problem_id]
        for _ in fdims
    )
