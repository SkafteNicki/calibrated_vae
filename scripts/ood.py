from typing import Union
import argparse
from copy import deepcopy
import os

import torch
from sklearn.metrics import roc_auc_score
import pytorch_lightning as pl

from scr.data import get_dataset
from scr.generative_models import get_model_from_file


def calc_log_prob(
    model_class: pl.LightningModule,
    model_instance,
    refit_encoder: Union[bool, torch.utils.data.DataLoader],
    score_method: str,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
):
    if refit_encoder:
        model_class.refit_encoder(model_instance, refit_encoder)

    log_probs_train = model_class.calc_score(
        model_instance, score_method, train_dataloader
    )
    log_probs_test = model_class.calc_score(
        model_instance, score_method, test_dataloader
    )

    return log_probs_train, log_probs_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weight_file")
    parser.add_argument("score_method")
    args = parser.parse_args()

    model, model_class = get_model_from_file(args.weight_file)
    dataset = args.weight_file.split("_")[-1][:-3]
    refit_dataset = torch.utils.data.DataLoader(get_dataset(dataset)[0], batch_size=64)

    all_datasets = ["mnist", "fmnist", "kmnist", "omniglot"]
    res = {}
    for dataset_name in all_datasets:
        print(f"Evaluating dataset {dataset_name}")
        train, val, test = get_dataset(dataset_name)
        train_dataloader = torch.utils.data.DataLoader(train, batch_size=64)
        val_dataloader = torch.utils.data.DataLoader(val, batch_size=64)
        test_dataloader = torch.utils.data.DataLoader(test, batch_size=64)

        output = calc_log_prob(
            model_class=model_class,
            model_instance=deepcopy(model),
            refit_encoder=False,
            score_method=args.score_method,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
        )
        res[f"train_{dataset_name}"], res[f"test_{dataset_name}"] = output

        output = calc_log_prob(
            model_class=model_class,
            model_instance=deepcopy(model),
            refit_encoder=refit_dataset,
            score_method=args.score_method,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
        )
        res[f"train_{dataset_name}_refit"], res[f"test_{dataset_name}_refit"] = output

    # calculate auroc score
    if not os.path.isfile('results/ood_scores.txt'):
        os.makedirs("results/", exist_ok=True)
        with open("results/ood_scores.txt", "w") as file:
            file.write("model, score_method, primary, secondary, subset, refit, auroc \n")

    for subset in ["train", "test"]:
        for refit in [False, True]:
            primary = res[f"{subset}_{dataset}_refit"] if refit else res[f"{subset}_{dataset}"]
            for dataset_name in all_datasets:
                if not dataset_name == dataset:
                    secondary = (
                        res[f"{subset}_{dataset_name}_refit"]
                        if refit
                        else res[f"{subset}_{dataset_name}"]
                    )
                    labels = torch.cat(
                        [torch.zeros(len(secondary)), torch.ones(len(primary))], 
                        dim=0,
                    )
                    score = roc_auc_score(labels, torch.cat([secondary, primary], dim=0))

                    with open("results/ood_scores.txt", "a") as file:
                        file.write(
                            f"{model_class.__name__}, {args.score_method}, {dataset}, {dataset_name}, {subset}, {refit}, {score} \n"
                        )
