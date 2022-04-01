import time

import argparse
from pytorch_lightning.utilities.seed import seed_everything

import torch

from scr.data import get_dataset


if __name__ == "__main__":
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="MixVAE")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="mnist"
    )
    args = parser.parse_args()

    train, val, test = get_dataset(args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=64)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=64)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=64)

    model_class = get_model(args.model)

    train_start = time.time()


    train_end = time.time()