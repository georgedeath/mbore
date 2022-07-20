import torch
from gpytorch.settings import cholesky_jitter
from . import gp, util


def train_classifier(Xtr, labels, method, weights):
    # get and train the classifier
    clf_class = util.get_classifier(method)
    clf = clf_class(Xtr, labels, weights)
    clf.train()

    return clf


def train_gp(
    train_x, train_y, train_restarts=10, chol_jitter=1e-3, verbose=False,
):
    # convert training data to torch
    train_x, train_y = map(
        lambda x: torch.tensor(x, dtype=torch.float64), [train_x, train_y],
    )
    # ensure the output (scalers) of the GP is shape (n, 1)
    train_y = torch.reshape(train_y, (-1, 1))

    # create, train and optimize the gp, using cholesky jitter to avoid PSD
    with cholesky_jitter(float=chol_jitter, double=chol_jitter, half=chol_jitter):
        model, mll = gp.create_model_and_mll(train_x, train_y)
        gp.train_model_restarts(mll, train_restarts, verbose=verbose)

    return model
