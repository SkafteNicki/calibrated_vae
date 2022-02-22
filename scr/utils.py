import functools

import torch
import torch.distributions as D


def rmse(x, mean):
    return (x - mean).abs().pow(2.0).mean().sqrt().item()


def ll(x, mean, var):
    d = D.Normal(mean, var.sqrt())
    return -d.log_prob(x).mean().item()


def brierscore(probs, targets):
    n_class = probs.shape[1]
    return (
        (probs - torch.nn.functional.one_hot(targets, n_class))
        .pow(2.0)
        .sum(dim=-1)
        .mean()
    )


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))
