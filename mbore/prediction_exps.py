import os
import torch
import tqdm.auto
import numpy as np
from joblib import Parallel, delayed
from . import model_training, problems, rankers, transforms, util


def perform_prediction(
    test_idx: int,
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    ranker: rankers.BaseRanker,
    method: str,
    gamma: float = None,
):
    # get the mask of values to use for training
    train_mask = np.arange(Ytr.shape[0]) != test_idx

    # extract the training data
    Xtrain = Xtr[train_mask]
    Ytrain = Ytr[train_mask]

    # arguments for the ranker
    ranker_kwargs = {"return_scalers": True}

    # special case for ParEGO because we need to ensure that the same reference
    # vector is used for the test and training scalarisations
    if isinstance(ranker, rankers.ParEGORanker):
        # randomly choose a reference vector index for both rankings
        ranker_kwargs["ref_vec_idx"] = np.random.choice(ranker.n_ref)

    # get the ranking and scalarisation for the training data
    train_ranks, train_scalers = ranker.get_ranks(Ytrain, **ranker_kwargs)

    # also get the ranks/scalers with the addition of the test location
    train_plus_test_ranks, train_plus_test_scalers = ranker.get_ranks(
        Ytr, **ranker_kwargs
    )

    # rescale inputs
    xtransformer = transforms.ZeroOneTransform(Xtrain, lb, ub)
    sXtrain = xtransformer.transform(Xtrain)
    Xtest = xtransformer.transform(Xtr[test_idx])

    # classification
    if method == "XGB":
        train_labels = util.get_class_labels_from_ranks(train_ranks, gamma)
        test_labels = util.get_class_labels_from_ranks(
            train_plus_test_ranks, gamma
        )

        # get the test data
        Ytest = test_labels[test_idx]

        # train the model
        model = model_training.train_classifier(sXtrain, train_labels, method)

        # predict the test location -- here we use the most likely class
        # label instead of the raw class conditional probability
        Ypred = model.clf.predict(Xtest.reshape(1, -1)).item()

    # if GP is in the method (this could be either:
    #   "GP" for regression, or "GPclass" for a classifier extension
    elif "GP" in method:
        # standardise the scalers (unit std and zero mean)
        s_train_scalers = util.standardize_vector(train_scalers)
        s_train_plus_test_scalers = util.standardize_vector(
            train_plus_test_scalers
        )

        # train the GP
        model = model_training.train_gp(
            sXtrain, s_train_scalers, train_restarts=10
        )

        # set the model to evaluation mode
        model.eval()

        # get the test data inputs
        Xtest = torch.tensor(Xtest).reshape(1, -1)

        if method == "GP":
            # test target is just the regression target
            Ytest = s_train_plus_test_scalers[test_idx]

            # make the prediction
            with torch.no_grad():
                posterior = model(Xtest)
                Ypred = posterior.mean.numpy()

        # i.e. method == "GPclass"
        else:
            # test targets are the target class
            test_labels = util.get_class_labels_from_ranks(
                train_plus_test_ranks, gamma
            )
            Ytest = test_labels[test_idx]

            # select the threshold at which to calculate the likelihood of
            # being below the threshold (i.e. in class 1).
            n = int(s_train_scalers.shape[0] * gamma)  # type:ignore
            threshold = np.sort(s_train_scalers)[n]

            # make the prediction
            with torch.no_grad():
                posterior = model(Xtest)

            # get the mean (mu) and standard devialtion (sigma)
            # note that these are in torch format (not numpy)
            mu = posterior.mean
            sigma = posterior.variance.sqrt()

            # calculate the probability of having a predicted value larger
            # than the threshold -- this is just the normal cdf evaluated
            # with (prediction - threshold) / standard_deviation
            p = torch.distributions.Normal(1.0, 1.0).cdf(
                (mu - threshold) / sigma
            )
            p = p.numpy().item()

            # convert this to a class label:
            #   p >= 0.5 = class 1, else class 0
            Ypred = 1 if p >= 0.5 else 0  # or just int(p >= 0.5)

    else:
        raise ValueError(f"Invalid method: {method:s}")

    return Ytest, Ypred


def carry_out_exp(
    problem_name: str,
    problem_id: int,
    dim: int,
    fdim: int,
    scalarizer: str,
    method: str,
    gamma: float,
    base_dir: str,
    data_dir: str = "lhs_samples",
    save_dir: str = "prediction_results",
    n_jobs: int = 6,
):
    # sanity check input arguments -- we shouldn't be having a gamma value with
    # a GP model, and we need a gamma value with XGBoost.
    allowed_methods = ["GP", "GPclass", "XGB"]
    assert (
        method in allowed_methods
    ), f"Method must be one of: {allowed_methods}"

    # create the paths to the data and save file
    data_fname = f"{problem_name}{problem_id}_d={dim}_o={fdim}_LHS_samples.npz"
    data_path = os.path.join(base_dir, data_dir, data_fname)

    save_fname = (
        f"{problem_name}{problem_id}"
        f"_d={dim}_o={fdim}_{scalarizer}_{method}"
        ".npz"
    )
    save_path = os.path.join(base_dir, save_dir, save_fname)

    if os.path.exists(save_path):
        print(f"Results file already exists, skipping: {save_path:s}")
        return

    if not os.path.exists(data_path):
        print(f"LHS samples data does not exist, skipping: {data_path:s}")
        return

    with np.load(data_path) as fd:
        Xtrs = fd["lhs_samples"]
        Ytrs = fd["Ytrs"]
        assert problem_name == fd["problem_name"]
        assert problem_id == fd["problem_id"]
        assert dim == fd["dim"]
        assert fdim == fd["fdim"]

    # sanity check shapes
    M, N = Xtrs.shape[:2]
    assert (Ytrs.shape[0] == M) and (Ytrs.shape[1] == N)

    # get the problem info
    problem_class = getattr(problems, problem_name.upper())
    problem = problem_class(problem_id, dim, fdim)
    lb, ub = problem.get_bounds()
    ref_point, _ = problem.get_reference_points()

    # initialise the scalarizer
    ranker_class = util.get_ranker(scalarizer)
    ranker = ranker_class(ref_point)

    # storage arrays
    targets = np.zeros((M, N))
    predictions = np.zeros((M, N))

    with Parallel(n_jobs=n_jobs) as parallel:
        for m in tqdm.auto.trange(M, leave=False):

            # evaluate the predictions in parallel. returns a
            # list of [[Ytest_1, Ypred_1], ..., [Ytest_N, Ypred_N]]
            res = parallel(
                delayed(perform_prediction)(
                    test_idx=n,
                    Xtr=Xtrs[m],
                    Ytr=Ytrs[m],
                    lb=lb,
                    ub=ub,
                    ranker=ranker,
                    method=method,
                    gamma=gamma,
                )
                for n in range(N)
            )

            for n, (Ytest, Ypred) in enumerate(res):  # type:ignore
                targets[m, n] = Ytest
                predictions[m, n] = Ypred

    np.savez(
        save_path,
        problem_name=problem_name,
        problem_id=problem_id,
        dim=dim,
        fdim=fdim,
        method=method,
        gamma=gamma,
        targets=targets,
        predictions=predictions,
    )
