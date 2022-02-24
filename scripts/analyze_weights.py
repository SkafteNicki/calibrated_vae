import argparse
import pickle as pkl
from copy import deepcopy
from itertools import product
from torch import nn

import numpy as np

from scr.classification_models import (
    DeepEnsembles,
    MixBlockEnsembles,
    MixLayerEnsembles,
)
from scr.layers import EnsampleLayer
from scr.utils import rgetattr, rsetattr, cosine_sim, disagreeement_score
from scr.data import get_dataset


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weight_file")
    parser.add_argument("--test_data", default=None)
    args = parser.parse_args()

    with open(args.weight_file, "rb") as file:
        model = pkl.load(file)

    if args.test_data is not None:
        test_data = get_dataset(args.test_data)

    # deep ensemble
    if isinstance(model, list):
        n = len(model)
        sim = np.zeros((n, n))
        dis = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                weight1 = np.concatenate(
                    [w.detach().flatten().numpy() for w in model[i].parameters()]
                )
                weight2 = np.concatenate(
                    [w.detach().flatten().numpy() for w in model[j].parameters()]
                )
                sim[i, j] = cosine_sim(weight1, weight2)
                if test_data is not None:
                    dis[i, j] = disagreeement_score(model[i], model[j], test_data)


        if args.test_data is not None:
            embeddings = [ ]
            for m in model:
                for i, batch in enumerate(torch.utils.data.DataLoader(test_data, batch_size=10)):
                    if i == 10: break
                    temp = deepcopy(model[i])
                    temp.fc = nn.Identity()
                    embeddings.append(temp(batch[0]).reshape(-1))
            from sklearn.manifold import TSNE

            embeddings = torch.stack(embeddings).numpy()

            tsne = TSNE(n_components=2)
            tsne.fit(embeddings)



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
