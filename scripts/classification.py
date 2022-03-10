import argparse
import os
import time

import torch
import wandb
from pytorch_lightning.utilities.seed import seed_everything

from scr.classification_models import get_model
from scr.data import get_dataset
from scr.notify import post_message

if __name__ == "__main__":
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        default=[
            "DeepEnsembles",
            "MixLayerEnsembles",
            "MixBlockEnsembles",
            "MixConvEnsembles",
            "DeepMixLayerEnsembles",
            "DeepMixBlockEnsembles",
            "DeepMixConvEnsembles",
        ],
    )
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        default=["cifar10", "mnist", "fmnist", "svhn"],
    )
    parser.add_argument(
        "-e", "--ensemble", nargs="+", default=[1, 2, 3, 4, 5, 8, 10], type=int
    )
    args = parser.parse_args()

    print(
        "*************************************************** \n"
        "Running script with args: \n"
        f" - models: {args.models} \n"
        f" - datasets: {args.datasets} \n"
        f" - ensemble: {args.ensemble} \n"
        "*************************************************** \n"
    )

    os.makedirs("results/", exist_ok=True)
    with open("results/classification_scores.txt", "w") as file:
        file.write(
            "dataset, model_class, n_ensemble, train_time, inference_time, acc, nll, brier \n"
        )

    for dataset_name in args.datasets:
        train, val, test = get_dataset(dataset_name)
        train_dataloader = torch.utils.data.DataLoader(train, batch_size=64)
        val_dataloader = torch.utils.data.DataLoader(val, batch_size=64)
        test_dataloader = torch.utils.data.DataLoader(test, batch_size=64)

        for model_name in args.models:
            model_class = get_model(model_name)
            for n_ensemble in args.ensemble:
                print(
                    "==================================================================== \n"
                    f"Dataset={dataset_name}, Model={model_name}, n_ensemble={n_ensemble} \n"
                    "==================================================================== \n"
                )

                try:
                    train_start = time.time()
                    model = model_class.fit(
                        n_ensemble, train_dataloader, val_dataloader
                    )
                    train_end = time.time()
                    if "ENABLE_LOGGING" in os.environ:
                        wandb.config.update(
                            {
                                "dataset": dataset_name,
                                "model_class": model_name,
                                "n_ensemble": n_ensemble,
                            }
                        )
                        wandb.finish()

                    os.makedirs("models/classification_models/", exist_ok=True)
                    model_class.save_checkpoint(
                        model,
                        f"models/classification_models/{dataset_name}_{model_name}_{n_ensemble}.pt",
                    )

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
                    post_message(
                        "For combination: \n"
                        f"Dataset={dataset_name}, Model={model_name}, n_ensemble={n_ensemble} \n"
                        "the following exception happended: \n"
                        f"{e}"
                    )
                    continue
