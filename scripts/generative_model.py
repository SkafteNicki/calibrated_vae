import argparse
import os
import time
import traceback

import matplotlib.pyplot as plt
import torch
from pytorch_lightning.utilities.seed import seed_everything

from scr.data import get_dataset
from scr.generative_models import get_model
from scr.notify import post_message

if __name__ == "__main__":
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--models", nargs="+", default=["MixVAE"])
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        default=["mnist"],
    )
    args = parser.parse_args()

    print(
        "*************************************************** \n"
        "Running script with args: \n"
        f" - models: {args.models} \n"
        f" - datasets: {args.datasets} \n"
        "*************************************************** \n"
    )
    n_ensemble = 5

    for dataset_name in args.datasets:
        train, val, test = get_dataset(dataset_name)
        train_dataloader = torch.utils.data.DataLoader(train, batch_size=64)
        val_dataloader = torch.utils.data.DataLoader(val, batch_size=64)
        test_dataloader = torch.utils.data.DataLoader(test, batch_size=64)

        for model_name in args.models:
            model_class = get_model(model_name)

            print(
                "==================================================================== \n"
                f"Dataset={dataset_name}, Model={model_name} \n"
                "==================================================================== \n"
            )

            try:
                train_start = time.time()
                model = model_class.fit(n_ensemble, train_dataloader, val_dataloader)
                train_end = time.time()

                os.makedirs("models/generative_models/", exist_ok=True)
                model_class.save_checkpoint(
                    model, f"models/generative_models/{model_name}_{dataset_name}.pt"
                )
            except Exception as e:
                print(f"Exception happened:")
                traceback.print_exc()
                post_message(
                    "the following exception happended: \n" f"{traceback.format_exc()}"
                )

    post_message("generative_model.py finished")
