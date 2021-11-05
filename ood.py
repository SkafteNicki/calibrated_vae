from argparse import ArgumentParser
from itertools import chain

import seaborn as sns
import torch
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers.base import DummyLogger
from torch import distributions as D
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

from models import get_model
from data import get_data


def get_encodings(model: LightningModule, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
    with torch.no_grad():
        encodings, labels = [ ], [ ]
        for batch in dataloader:
            x, y = batch
            z_mu, z_std = model.encode(x)
            encodings.append(z_mu)
            labels.append(y)
        encodings = torch.cat(encodings, dim=0)
        labels = torch.cat(labels, dim=0)
        return encodings, labels


def calc_log_prob(
    model: LightningModule, 
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    refit_encoder: bool = False,
    plot_res: bool = False
) -> pd.DataFrame:

    if refit_encoder:
        init_state_dict = deepcopy(model.state_dict())

        if plot_res:
            encodings, labels = get_encodings(model, train_dataloader)

            _, ax = plt.subplots(ncols=2)
            for i in labels.unique():
                ax[0].scatter(encodings[labels==i, 0], encodings[labels==i, 1])

        model.configure_optimizers = lambda: torch.optim.Adam(chain(
            model.encoder.parameters(),
            model.encoder_mu.parameters(), 
            model.encoder_std.parameters())
            , lr=1e-3
        )
        trainer = Trainer(logger=DummyLogger(), max_epochs=10, gpus=1 if torch.cuda.is_available() else 0)
        trainer.fit(model, train_dataloader=train_dataloader)

        if plot_res:
            encodings, labels = get_encodings(model, train_dataloader)

            for i in labels.unique():
                ax[1].scatter(encodings[labels==i, 0], encodings[labels==i, 1])
            plt.show()

    log_probs_train = model.calc_log_prob(train_dataloader)
    log_probs_test = model.calc_log_prob(test_dataloader)

    dataframe = pd.DataFrame(
        torch.cat([log_probs_train, log_probs_test]).numpy(), columns=['log_probs']
    )
    dataframe['split'] = len(log_probs_train) * ['train'] + len(log_probs_test) * ['test']

    if refit_encoder:  # reset
        model.load_state_dict(init_state_dict)

    return dataframe


def calc_mixture_score(
    model: LightningModule,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    batch_size: int = 100,
    n: int = 1000,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        # sample images
        samples = [ ]
        n_batches = int(n / batch_size)
        for i in range(n_batches):
            z = model.prior.sample((batch_size,))
            samples.append(model(z))
        samples = torch.cat(samples, dim=0)

        # construct mixture
        components = D.Independent(D.Bernoulli(probs=samples), 3)
        weight = D.Categorical(torch.ones(n))
        mixture = D.MixtureSameFamily(weight, components)

        # calc score for train
        mixture_score_train = [ ]
        for batch in tqdm(train_dataloader, desc='Train mixture score'):
            x, _ = batch
            mixture_score_train.append(mixture.log_prob(x))
        mixture_score_train = torch.cat(mixture_score_train, dim=0)

        # calc score for test
        mixture_score_test = [ ]
        for batch in tqdm(test_dataloader, desc='Test mixture score'):
            x, _ = batch
            mixture_score_test.append(mixture.log_prob(x))
        mixture_score_test = torch.cat(mixture_score_test, dim=0)

    return torch.cat([mixture_score_train, mixture_score_test], dim=0)


if __name__ == "__main__":
    # Common arguments
    parser = ArgumentParser()
    parser.add_argument("model", type=str, default="")
    parser.add_argument("checkpoint", type=str, default="")
    parser.add_argument("dataset", type=str, default="")
    parser.add_argument("--filename", type=str, default='result')
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()

    model_class = get_model(args.model)
    model = model_class.load_from_checkpoint(args.checkpoint)
    model.eval()

    datamodule_class = get_data(args.dataset)
    datamodule = datamodule_class(batch_size=args.batch_size)
    datamodule.setup()
    
    # Standard log prob score
    dataframe = calc_log_prob(
        model, datamodule.train_dataloader(), datamodule.test_dataloader(),
    )

    # Log prob score with refitted encoder
    dataframe_refit = calc_log_prob(
        model, datamodule.train_dataloader(), datamodule.test_dataloader(),
        refit_encoder=True, plot_res=True
    )
    
    # Log prob score by mixture of samples
    mixture_score = calc_mixture_score(
        model, datamodule.train_dataloader(), datamodule.test_dataloader()
    )

    dataframe['log_probs_refit'] = dataframe_refit['log_probs']
    dataframe['mixture_score'] = mixture_score.tolist()

    dataframe.to_csv(f"{args.filename}.csv")
    