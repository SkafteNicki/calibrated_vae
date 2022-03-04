import functools

import torch
import torch.distributions as D
from scipy.spatial.distance import cosine


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


def cosine_sim(x, y):
    return 1 - cosine(x, y)


def disagreeement_score(m1, m2, x):
    pred1 = m1(x).argmax(dim=-1)
    pred2 = m2(x).argmax(dim=-1)
    return (pred1 != pred2).float().mean()


def disagreement_score_from_preds(p1, p2):
    return (p1.argmax(dim=-1) != p2.argmax(dim=-1)).float().mean()


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))
