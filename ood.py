from argparse import ArgumentParser
from itertools import chain

import numpy as np
import seaborn as sns
import torch
from pytorch_lightning import Trainer
from torch import distributions as D
import pandas as pd

from models import get_model
from data import get_data


def calc_log_probs(
        model, 
        train_dataloader,
        test_dataloader,
        refit_encoder: bool = False
    ):
    
    if refit_encoder:
        optimizer = torch.optim.Adam(chain(model.encoder, model.encoder_mu, model.encoder_std), lr=1e-3)
        for _ in range(10):
            for batch in train_dataloader:
                optimizer.zero_grad()
                loss = model.training_step(batch)
                loss.backward()
                optimizer.step()

    log_probs_train = model.calc_log_probs(train_dataloader)
    log_probs_test = model.calc_log_probs(test_dataloader)

    dataframe = pd.DataFrame(
        torch.cat([log_probs_train, log_probs_test]).numpy(), columns=['log_probs']
    )
    dataframe['split'] = len(log_probs_train) * ['train'] + len(log_probs_test) * ['test']
    dataframe['refit'] = refit_encoder

    return dataframe


if __name__ == "__main__":
    # Common arguments
    parser = ArgumentParser()
    parser.add_argument("model", type=str, default="")
    parser.add_argument("checkpoint", type=str, default="")
    parser.add_argument("dataset", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    model_class = get_model(args.model)
    model = model_class.load_from_checkpoint(args.checkpoint)
    model.eval()

    datamodule_class = get_data(args.dataset)
    datamodule = datamodule_class(batch_size=args.batch_size)
    datamodule.setup()

    dataframe = calc_log_probs(
        model, datamodule.train_dataloader(), datamodule.test_dataloader(),
    )
    dataframe_refit = calc_log_probs(
        model, datamodule.train_dataloader(), datamodule.test_dataloader(),
        refit_encoder=True
    )
    pd.cat(dataframe, dataframe_refit)


    sns.histplot(dataframe, x='log_probs', hue='split')

    #torch.save(scores1, "results1.pt")
    #torch.save(scores2, "results2.pt")

    #if args.plot_this:
        #combined = np.array([scores1, scores2])
#
        #sns.histplot(combined, x="log prob", y="Procentage samples")
