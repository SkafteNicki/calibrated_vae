from copy import deepcopy
from itertools import chain

import matplotlib.pyplot as plt
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch import Tensor
from torch import distributions as D
from torch import nn
from torch.utils.data import DataLoader

import wandb
from models.layers import EnsembleList, weight_reset
from models.vae import VAE


class MEVAE(VAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.encoder = EnsembleList(
            [deepcopy(self.encoder).apply(weight_reset) for _ in range(self.hparams.n_ensemble)]
        )
        self.encoder_mu = EnsembleList(
            [deepcopy(self.encoder_mu).apply(weight_reset) for _ in range(self.hparams.n_ensemble)]
        )
        self.encoder_std = EnsembleList(
            [deepcopy(self.encoder_std).apply(weight_reset) for _ in range(self.hparams.n_ensemble)]
        )
        self.decoder = EnsembleList(
            [deepcopy(self.decoder).apply(weight_reset) for _ in range(self.hparams.n_ensemble)]
        )

    def forward(self, z: Tensor, use_all: bool = False) -> Tensor:
        if use_all:
            return self.decoder(z)
        else:
            idx = torch.randint(self.hparams.n_ensemble, (1,))
            return self.decoder[idx](z)

    def encode(self, x: Tensor, use_all: bool = False) -> Tensor:
        if use_all:
            h = self.encoder(x)
            z_mu = self.encoder_mu(h)
            z_std = self.encoder_std(h)
        else:
            idx = torch.randint(self.hparams.n_ensemble, (1,))
            h = self.encoder[idx](x)
            z_mu = self.encoder_mu[idx](h)
            z_std = self.encoder_std[idx](h)
        return z_mu, z_std

    def training_step(self, batch, batch_idx=0):
        x, y = batch
        loss, z_mu, _, z, _ = self._step(x, "train")

        if batch_idx == 0:
            x_hat = self(z, use_all=True)
            xh_s = x_hat.shape
            images = wandb.Image(
                torch.cat([x[:8], x_hat[:, :8].reshape(xh_s[0] * 8, *xh_s[2:])], dim=0),
                caption="Top: Original, Bottom: Reconstruction",
            )
            self.logger.experiment.log({"recon": images}, commit=False)

            z_sample = self.prior.sample((8,))
            x_sample = self(z_sample, use_all=True)
            images = wandb.Image(x_sample.reshape(xh_s[0] * 8, *xh_s[2:]), caption="Samples")
            self.logger.experiment.log({"samples": images}, commit=False)

        return {"loss": loss, "latents": z_mu.detach(), "labels": y.detach()}

    def calc_log_prob(self, dataloader: DataLoader) -> Tensor:
        current_state = self.training
        self.eval()
        log_probs = []
        with torch.no_grad():
            for batch in dataloader:
                x_hats = []
                for _ in range(self.hparams.mc_samples):
                    x, _ = batch
                    _, _, x_hat, _ = self.encode_decode(x)
                    x_hats.append(x_hat)
                x_hats=torch.stack(x_hats)
                d = D.MixtureSameFamily(
                    D.Categorical(torch.ones(self.hparams.mc_samples)),
                    D.Independent(D.Bernoulli(probs=x_hat), 3)
                )
                log_probs.append(d.log_prob(x))
        log_probs = torch.cat(log_probs, dim=0)
        self.training = current_state
        return log_probs

    def training_epoch_end(self, outputs):
        self.training_epoch_end_plotter(outputs, mc_samples=self.hparams.mc_samples)
