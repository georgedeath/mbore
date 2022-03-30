import torch
import copy
import numpy as np

from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.models.gp_regression import FixedNoiseGP
from botorch.optim import optimize_acqf

from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior, UniformPrior
from gpytorch.utils.errors import NotPSDError


def create_model_and_mll(train_x, train_y, priors="uniform"):
    d = train_x.shape[-1]
    dtype = train_x.dtype

    # set up the covariance module with prior on each length-scale
    if priors == "uniform":
        ls_prior = UniformPrior(
            torch.tensor([1e-4] * d, dtype=dtype),
            torch.tensor([np.sqrt(d)] * d, dtype=dtype),
        )
        os_prior = UniformPrior(1e-4, 10)
        ls_prior._validate_args = False
        os_prior._validate_args = False

    elif priors == "gamma":
        ls_prior = GammaPrior(
            torch.tensor([3.0] * d, dtype=dtype),
            torch.tensor([6.0] * d, dtype=dtype),
        )
        os_prior = GammaPrior(2.0, 0.15)
    else:
        raise ValueError("Invalid prior type given")

    covar_module = ScaleKernel(
        base_kernel=MaternKernel(
            nu=2.5, ard_num_dims=d, lengthscale_prior=ls_prior,
        ),
        outputscale_prior=os_prior,
    )
    train_Yvar = torch.full_like(train_y, 1e-6, dtype=dtype)

    model = FixedNoiseGP(train_x, train_y, train_Yvar, covar_module)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    return model, mll


def ei_optimize(model, best_f, problem_bounds, raw_samples, opt_restarts):
    acq_func = ExpectedImprovement(model, best_f, maximize=True)

    train_xnew, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=problem_bounds,
        q=1,
        num_restarts=opt_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": 500},
    )

    return train_xnew


def train_model_restarts(mll, n_restarts, max_failures=5, verbose=False):
    def model_loss(mll):
        mll.train()
        output = mll.model(*mll.model.train_inputs)
        loss = -mll(output, mll.model.train_targets)
        return loss.sum().item()

    noise_prior = "noise_prior" in mll.model.likelihood.noise_covar._priors

    # start by assuming the current parameters are the best
    best_params = copy.deepcopy(mll.state_dict())
    best_loss = model_loss(mll)

    failures = 0

    for i in range(n_restarts):
        if failures == max_failures:
            raise RuntimeError(
                "Maximum number of failures when trying to fit the GP"
                f" reached ({max_failures:d})."
            )
        # sample new hyperparameters from the kernel priors
        mll.model.covar_module.base_kernel.sample_from_prior(
            "lengthscale_prior"
        )
        mll.model.covar_module.sample_from_prior("outputscale_prior")

        #  if we have one, sample from noise prior
        if noise_prior:
            mll.model.likelihood.noise_covar.sample_from_prior("noise_prior")

        # try and fit the model using bfgs, starting at the sampled params.
        # note that sometimes BFGS-B takes giant steps and causes PSD errors
        # in the cholesky decomposition. catch these errors and try to continue
        try:
            fit_gpytorch_model(mll, method="L-BFGS-B")
        except NotPSDError:
            failures += 1
            if verbose:
                print(
                    "A failure occurred while trying to fit the model,"
                    f" {failures:d}/{max_failures:d} so far."
                )
            continue

        # calculate the loss
        curr_loss = model_loss(mll)

        # if we've ended up with better hyperparameters, save them to use
        if curr_loss < best_loss:
            best_params = copy.deepcopy(mll.state_dict())
            best_loss = curr_loss

    # load the best found parameters into the model
    mll.load_state_dict(best_params)

    if verbose:
        ls = mll.model.covar_module.base_kernel.lengthscale.detach().numpy()
        ops = mll.model.covar_module.outputscale.item()
        print("Best found hyperparameters:")
        print(f"\tLengthscale: {ls.ravel()}")
        print(f"\tOutputscale: {ops}")
