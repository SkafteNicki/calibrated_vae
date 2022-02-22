import argparse
import pickle as pkl
from copy import deepcopy
from itertools import product

import numpy as np
import torch
from scipy.spatial.distance import cosine

from scr.classification_models import (
    DeepEnsembles,
    MixBlockEnsembles,
    MixLayerEnsembles,
)
from scr.layers import EnsampleLayer
from scr.utils import rgetattr, rsetattr


def cosine_sim(x, y):
    return 1 - cosine(x, y)


def disagreeement_score(m1, m2, x):
    pred1 = m1(x).argmax(dim=-1)
    pred2 = m2(x).argmax(dim=-1)
    return (pred1 != pred2).float().mean()


def get_all_combinations(n_ensemble_size, n_ensemble_layers):
    list1 = list(range(0, n_ensemble_size))
    list2 = list(range(0, n_ensemble_size))
    for i in range(n_ensemble_layers - 1):
        all_combinations = list(product(list1, list2))
        if i != 0:
            all_combinations = [
                (*element[0], element[1]) for element in all_combinations
            ]
        list1 = all_combinations
    return all_combinations


parser = argparse.ArgumentParser()
parser.add_argument("weight_file")
args = parser.parse_args()

with open(args.weight_file, "rb") as file:
    model = pkl.load(file)

# deep ensemble
if isinstance(model, list):
    n = len(model)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            weight1 = np.concatenate(
                [w.detach().flatten().numpy() for w in model[i].parameters()]
            )
            weight2 = np.concatenate(
                [w.detach().flatten().numpy() for w in model[j].parameters()]
            )
            sim[i, j] = cosine_sim(weight1, weight2)

# mix ensemble
else:
    n_ensemble_layers = len(
        [m for m in model.modules() if isinstance(m, EnsampleLayer)]
    )
    n_ensemble_size = len(
        [m for m in model.modules() if isinstance(m, EnsampleLayer)][0]
    )
    n = n_ensemble_size ** n_ensemble_layers
    sim = np.zeros((n, n))
    combinations = get_all_combinations(n_ensemble_size, n_ensemble_layers)
    for i, comb1 in enumerate(combinations):
        for j, comb2 in enumerate(combinations):
            print(i, j)
            if i == j:
                sim[i, j] = 1.0
                continue
            if sim[j, i] != 0.0:
                sim[i, j] = sim[j, i]
                continue

            temp1 = deepcopy(model)
            temp2 = deepcopy(model)
            idx = 0
            for m in temp1.named_modules():
                if isinstance(m[1], EnsampleLayer):
                    rsetattr(temp1, m[0], rgetattr(temp1, m[0])[comb1[idx]])
                    idx += 1
            idx = 0
            for m in temp2.named_modules():
                if isinstance(m[1], EnsampleLayer):
                    rsetattr(temp2, m[0], rgetattr(temp2, m[0])[comb2[idx]])
                    idx += 1

            weight1 = np.concatenate(
                [w.detach().flatten().numpy() for w in temp1.parameters()]
            )
            weight2 = np.concatenate(
                [w.detach().flatten().numpy() for w in temp2.parameters()]
            )
            sim[i, j] = cosine_sim(weight1, weight2)

            del temp1, temp2, weight1, weight2
