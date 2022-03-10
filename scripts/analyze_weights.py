import argparse
from copy import deepcopy

import os
import pickle

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tqdm

from scr.data import get_dataset
from scr.classification_models import get_classification_model_from_file
from scr.layers import EnsampleLayer
from scr.utils import cosine_sim, disagreement_score_from_preds, rgetattr, rsetattr, get_all_combinations


def dis_where_wrong(pred1, pred2, target):
    p1 = pred1.argmax(dim=-1)
    p2 = pred2.argmax(dim=-1)
    mask = torch.logical_and(p1 != target, p2 != target)
    return (p1 != p2)[mask].float().mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weight_file")
    parser.add_argument("test_data")
    args = parser.parse_args()
    save_name = args.weight_file[:-3].split("/")[-1]

    model = get_classification_model_from_file(args.weight_file)

    n_samples = 1000
    batch_size = 20
    n_batches = int(n_samples / batch_size)
    # Downsample for speeding up things
    _, _, test_data = get_dataset(args.test_data)
    test_data = torch.utils.data.Subset(
        test_data, list(np.random.permutation(n_samples))
    )
    test = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    targets = torch.cat([batch[1] for batch in test])

    if isinstance(model, list):
        n = len(model)

        def get_model_instance(index):
            return deepcopy(model[index])

    else:
        n_ensemble_layers = len(
            [m for m in model.modules() if isinstance(m, EnsampleLayer)]
        )
        n_ensemble_size = len(
            [m for m in model.modules() if isinstance(m, EnsampleLayer)][0]
        )
        n = n_ensemble_size ** n_ensemble_layers
        combinations = get_all_combinations(n_ensemble_size, n_ensemble_layers)

        def get_model_instance(index):
            comb = combinations[index]
            temp = deepcopy(model)
            idx = 0
            for name, module in temp.named_modules():
                if isinstance(module, EnsampleLayer):
                    rsetattr(temp, name, rgetattr(temp, name)[comb[idx]])
                    idx += 1
            return temp

    sim = np.ones((n, n))
    dis = np.zeros((n, n))
    dis2 = np.zeros((n, n))

    preds = torch.zeros(n, n_samples, 10)
    embeddings = torch.zeros(n, n_batches, 512 * batch_size)
    for i in tqdm.tqdm(range(n), total=n, desc="Extracting predictions and embeddings"):
        m = get_model_instance(i)
        with torch.no_grad():
            for j, batch in enumerate(test):
                preds[i, batch_size * j : batch_size * (j + 1)] = m(batch[0])

        m.base.fc = nn.Identity()
        with torch.no_grad():
            for j, batch in enumerate(test):
                embeddings[i, j] = m(batch[0]).reshape(-1)

    os.makedirs("results/analyze_weight_space", exist_ok=True)
    with open(f"results/analyze_weight_space/{save_name}_preds.pkl", "wb") as file:
        pickle.dump(preds, file)
    with open(f"results/analyze_weight_space/{save_name}_embeddings.pkl", "wb") as file:
        pickle.dump(embeddings.reshape(n, n_batches, batch_size, 512).reshape(n, -1, 512), file)

    for i in range(n):
        for j in range(i, n):
            print(f"Comparing ensemble {i} and {j}")
            if i == j:
                continue

            m1 = get_model_instance(i)
            m2 = get_model_instance(j)

            weight1 = np.concatenate(
                [w.detach().flatten().numpy() for w in m1.parameters()]
            )
            weight2 = np.concatenate(
                [w.detach().flatten().numpy() for w in m2.parameters()]
            )

            sim[i, j] = cosine_sim(weight1, weight2)
            dis[i, j] = disagreement_score_from_preds(preds[i], preds[j])
            dis2[i, j] = dis_where_wrong(preds[i], preds[j], targets)

            sim[j, i] = sim[i, j]
            dis[j, i] = dis[i, j]
            dis2[j, i] = dis2[i, j]

    with open(f"results/analyze_weight_space/{save_name}_results.pkl", "wb") as file:
        pickle.dump([sim, dis, dis2], file)

    tsne = TSNE(n_components=2)
    tsne_embeddings = tsne.fit_transform(
        embeddings.reshape(-1, 512 * batch_size).numpy()
    )

    os.makedirs("figures/analyze_weight_space/", exist_ok=True)

    fig = plt.figure()
    plt.imshow(sim)
    plt.colorbar()
    fig.savefig(
        f"figures/analyze_weight_space/{save_name}_sim.png", bbox_inches="tight"
    )
    fig.savefig(
        f"figures/analyze_weight_space/{save_name}_sim.svg", bbox_inches="tight"
    )

    fig = plt.figure()
    plt.imshow(dis)
    plt.colorbar()
    fig.savefig(
        f"figures/analyze_weight_space/{save_name}_dis.png", bbox_inches="tight"
    )
    fig.savefig(
        f"figures/analyze_weight_space/{save_name}_dis.svg", bbox_inches="tight"
    )

    fig = plt.figure()
    plt.imshow(dis2)
    plt.colorbar()
    fig.savefig(
        f"figures/analyze_weight_space/{save_name}_dis2.png", bbox_inches="tight"
    )
    fig.savefig(
        f"figures/analyze_weight_space/{save_name}_dis2.svg", bbox_inches="tight"
    )


    fig = plt.figure()
    for i in range(n):
        plt.plot(
            tsne_embeddings[n_batches * i : n_batches * (i + 1), 0],
            tsne_embeddings[n_batches * i : n_batches * (i + 1), 1],
            ".",
            label=f"model_{i}",
        )
    plt.legend()
    fig.savefig(
        f"figures/analyze_weight_space/{save_name}_tsne.png", bbox_inches="tight"
    )
    fig.savefig(
        f"figures/analyze_weight_space/{save_name}_tsne.svg", bbox_inches="tight"
    )
