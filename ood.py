from argparse import ArgumentParser

import numpy as np
import seaborn as sns
import torch
from pytorch_lightning.core import datamodule
from torch import distributions as D
import pandas as pd

from models import get_model
from data import get_data


def calc_log_probs(model, dataloader):
    log_probs = []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            _, _, x_hat, _ = model.encode_decode(x)
            d = D.Independent(D.Bernoulli(probs=x_hat), 3)
            log_probs.append(d.log_prob(x))
    log_probs = torch.cat(log_probs, dim=0)
    return log_probs


if __name__ == "__main__":
    # Common arguments
    parser = ArgumentParser()
    parser.add_argument("model", type=str, default="")
    parser.add_argument("checkpoint", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    model_class = get_model(args.model)
    model = model_class.load_from_checkpoint(args.checkpoint)
    model.eval()

    datamodule_class = get_data('small_mnist')
    datamodule = datamodule_class(batch_size=args.batch_size)
    datamodule.setup()

    log_probs_train = calc_log_probs(model, datamodule.train_dataloader())
    log_probs_test = calc_log_probs(model, datamodule.test_dataloader())

    dataframe = pd.DataFrame(
        torch.cat([log_probs_train, log_probs_test]).numpy(), columns=['log_probs']
    )
    dataframe['split'] = len(log_probs_train) * ['train'] + len(log_probs_test) * ['test']

    sns.histplot(dataframe, x='log_probs', hue='split')

    #torch.save(scores1, "results1.pt")
    #torch.save(scores2, "results2.pt")

    #if args.plot_this:
        #combined = np.array([scores1, scores2])
#
        #sns.histplot(combined, x="log prob", y="Procentage samples")
