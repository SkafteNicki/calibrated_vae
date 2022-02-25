import argparse
import pickle as pkl
from copy import deepcopy
from itertools import product

import numpy as np
import torch
from torch import nn

from scr.classification_models import (
    DeepEnsembles,
    MixBlockEnsembles,
    MixLayerEnsembles,
)
from scr.data import get_dataset
from scr.layers import EnsampleLayer
from scr.utils import cosine_sim, disagreeement_score, rgetattr, rsetattr


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
        _, _, test_data = get_dataset(args.test_data)
        test_data = torch.utils.data.Subset(
            test_data, list(np.random.permutation(1000))
        )
        test = torch.utils.data.DataLoader(test_data, batch_size=10)

    # deep ensemble
    if isinstance(model, list):
        n = len(model)
        sim = np.ones((n, n))
        dis = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                print(f"Comparing ensemble {i} and {j}")
                if i == j:
                    continue

                weight1 = np.concatenate(
                    [w.detach().flatten().numpy() for w in model[i].parameters()]
                )
                weight2 = np.concatenate(
                    [w.detach().flatten().numpy() for w in model[j].parameters()]
                )
                sim[i, j] = cosine_sim(weight1, weight2)
                if args.test_data is not None:
                    for batch in test:
                        with torch.no_grad():
                            dis[i, j] += disagreeement_score(
                                model[i], model[j], batch[0]
                            )
                    dis[i, j] /= len(test)

                sim[j, i] = sim[i, j]
                dis[j, i] = sim[i, j]

        if args.test_data is not None:
            n_batches = 50
            embeddings = []
            for m in model:
                temp = deepcopy(m)
                temp.base.fc = nn.Identity()
                with torch.no_grad():
                    for i, batch in enumerate(
                        torch.utils.data.DataLoader(test_data, batch_size=10)
                    ):
                        if i == n_batches:
                            break
                        embeddings.append(temp(batch[0]).reshape(-1))
            from sklearn.manifold import TSNE

            embeddings = torch.stack(embeddings).detach().numpy()

            tsne = TSNE(n_components=2)
            tsne_embeddings = tsne.fit_transform(embeddings)

            import matplotlib.pyplot as plt

            plt.figure()
            for i in range(len(model)):
                plt.plot(
                    tsne_embeddings[n_batches * i : n_batches * (i + 1), 0],
                    tsne_embeddings[n_batches * i : n_batches * (i + 1), 1],
                    ".",
                    label=f"model_{i}",
                )
            plt.legend()

    # mix ensemble
    else:
        n_ensemble_layers = len(
            [m for m in model.modules() if isinstance(m, EnsampleLayer)]
        )
        n_ensemble_size = len(
            [m for m in model.modules() if isinstance(m, EnsampleLayer)][0]
        )
        n = n_ensemble_size ** n_ensemble_layers
        sim = np.ones((n, n))
        dis = np.zeros((n, n))
        combinations = get_all_combinations(n_ensemble_size, n_ensemble_layers)
        for i, comb1 in enumerate(combinations):
            for j, comb2 in enumerate(combinations):
                print(f"Comparing ensemble {i} and {j}")
                if i == j:
                    sim[i, j] = 1.0
                    continue
                if sim[j, i] != 1.0:
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

                if args.test_data is not None:
                    for batch in test:
                        with torch.no_grad():
                            dis[i, j] += disagreeement_score(temp1, temp2, batch[0])
                    dis[i, j] /= len(test)

                del temp1, temp2, weight1, weight2
