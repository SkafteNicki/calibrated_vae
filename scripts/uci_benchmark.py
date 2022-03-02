import argparse
import pickle
import time
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, tensor
from torch.utils.data import DataLoader, TensorDataset

from scr.regression_models import Ensamble, EnsambleNLL, MixEnsemble, MixEnsembleNLL
from scr.utils import ll, rmse

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", type=str)
    args = parser.parse_args()

    if args.dataset == "all":
        datasets = [
            "boston",
            "concrete",
            "naval",
            "power_plant",
            "protein_structure",
            "wine_red",
            "wine_white",
            "yacht_hydrodynamics",
        ]
    else:
        datasets = [args.dataset]

    os.makedirs("results/", exist_ok=True)
    with open("results/uci_benchmark_scores.txt", "w") as file:
        file.write("dataset, model_class, n_ensemble, train_time, rmse, ll \n")

    for dataset in datasets:
        with open(f"data/uci_datasets/{dataset}_uci_dataset.pkl", "rb") as file:
            data, target = pickle.load(file)
        target = target.reshape(-1, 1)
        input_size = data.shape[1]

        for model_class in [Ensamble, EnsambleNLL, MixEnsemble, MixEnsembleNLL]:
            model_name = model_class.__name__
            scores = {"rmse": [], "nll": [], "time": []}

            for rep in range(20):
                train_index, test_index = train_test_split(
                    np.arange(len(data)), test_size=0.1, random_state=(rep + 1) * SEED
                )
                train_data, train_target = data[train_index], target[train_index]
                test_data, test_target = data[test_index], target[test_index]

                data_scaler = StandardScaler()
                data_scaler.fit(train_data)

                train_data = tensor(
                    data_scaler.transform(train_data), dtype=torch.float32
                )
                test_data = tensor(
                    data_scaler.transform(test_data), dtype=torch.float32
                )

                target_scaler = StandardScaler()
                target_scaler.fit(train_target)
                train_target = tensor(
                    target_scaler.transform(train_target), dtype=torch.float32
                )

                train = DataLoader(
                    TensorDataset(train_data, train_target),
                    batch_size=100,
                )

                if "mix" in model_class.__name__.lower():
                    reps = 1
                    n_epochs = 500
                else:
                    reps = 5
                    n_epochs = 40

                start = time.time()
                models = []
                for _ in range(reps):
                    # TODO: refactor into model_class.fit
                    model = model_class(input_size, 100, nn.ReLU())
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

                    for epoch in range(n_epochs):
                        for batch in train:
                            optimizer.zero_grad()
                            loss = model.loss(*batch)
                            loss.backward()
                            optimizer.step()

                    models.append(model)
                end = time.time()

                y = tensor(test_target, dtype=torch.float32)
                mean, var = model_class.ensample_predict(
                    models, test_data, scaler=target_scaler
                )

                scores["rmse"].append(rmse(y, mean))
                scores["nll"].append(ll(y, mean, var))
                scores["time"].append(end - start)

            with open("results/uci_benchmark_scores.txt", "a") as file:
                file.write(
                    f"{dataset}, {model_name}, {5}, {scores['time']}, {scores['rmse']}, {scores['nll']} \n"
                )

            print(
                f"Model {model_class.__name__}. \n"
                f"RMSE: {np.mean(scores['rmse'])}+-{np.std(scores['rmse'])} \n"
                f"NLL : {np.mean(scores['nll'])}+-{np.std(scores['nll'])} \n"
                f"Time: {np.mean(scores['time'])}+-{np.std(scores['time'])} \n"
            )
