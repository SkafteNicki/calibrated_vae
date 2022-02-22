import os
import pickle as pkl
import time

import torch
import torchvision

from scr.classification_models import (
    DeepEnsembles,
    MixBlockEnsembles,
    MixLayerEnsembles,
)

if __name__ == "__main__":
    os.makedirs("results/", exist_ok=True)
    with open("results/classification_scores.txt", "w") as file:
        file.write(
            "dataset, model_class, n_ensemble, train_time, inference_time, acc, nll, brier \n"
        )

    for dataset_name in ["cifar10", "svhn"]:
        if dataset_name == "svhn":
            train = torchvision.datasets.SVHN(
                root=f"data/{dataset_name}/",
                download=True,
                split="train",
                transform=torchvision.transforms.ToTensor(),
            )
            test = torchvision.datasets.SVHN(
                root=f"data/{dataset_name}/",
                download=True,
                split="test",
                transform=torchvision.transforms.ToTensor(),
            )
        elif dataset_name == "cifar10":
            train = torchvision.datasets.CIFAR10(
                root=f"data/{dataset_name}/",
                download=True,
                train=True,
                transform=torchvision.transforms.ToTensor(),
            )
            test = torchvision.datasets.CIFAR10(
                root=f"data/{dataset_name}/",
                download=True,
                train=False,
                transform=torchvision.transforms.ToTensor(),
            )
        n_train = int(len(train) * 0.9)
        train, val = torch.utils.data.random_split(
            train, [n_train, len(train) - n_train]
        )

        train_dataloader = torch.utils.data.DataLoader(train, batch_size=64)
        val_dataloader = torch.utils.data.DataLoader(val, batch_size=64)
        test_dataloader = torch.utils.data.DataLoader(test, batch_size=64)

        for model_class in [DeepEnsembles, MixLayerEnsembles, MixBlockEnsembles]:
            model_name = model_class.__name__
            for n_ensemble in [1, 2, 3, 4, 5, 8, 10, 12, 15, 20, 25]:
                print(
                    "======================================================== \n"
                    f"Model={model_name}, n_ensemble={n_ensemble}   \n"
                    "======================================================== \n"
                )

                try:
                    train_start = time.time()
                    model = model_class.fit(
                        n_ensemble, train_dataloader, val_dataloader
                    )
                    train_end = time.time()

                    os.makedirs("models/classification_models/", exist_ok=True)
                    with open(
                        f"models/classification_models/{dataset_name}_{model_name}_{n_ensemble}.pkl",
                        "wb",
                    ) as file:
                        pkl.dump(model, file)

                    inference_start = time.time()
                    acc, nll, brier = model_class.ensample_predict(
                        model, test_dataloader
                    )
                    inference_end = time.time()

                    train_time = train_end - train_start
                    inference_time = inference_end - inference_start
                    with open("results/classification_scores.txt", "a") as file:
                        file.write(
                            f"{dataset_name}, {model_name}, {n_ensemble}, {train_time}, {inference_time}, {acc}, {nll}, {brier} \n"
                        )

                except Exception as e:
                    print(f"Exception happened: {e}")
                    continue
