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

    def forward(self, z: Tensor) -> Tensor:
        idx = torch.randint(self.hparams.n_ensemble, (1,))
        return self.decoder[idx](z)

    def encode(self, x: Tensor) -> Tensor:
        idx = torch.randint(self.hparams.n_ensemble, (1,))
        h = self.encoder[idx](x)
        z_mu = self.encoder_mu[idx](h)
        z_std = self.encoder_std[idx](h)
        return z_mu, z_std

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
        latents = torch.cat([o["latents"] for o in outputs], dim=0)
        labels = torch.cat([o["labels"] for o in outputs], dim=0)

        # plot latent encodings
        for i in labels.unique():
            plt.scatter(
                latents[i == labels, 0].cpu(),
                latents[i == labels, 1].cpu(),
                label=str(i.cpu()),
                zorder=10,
                alpha=0.5,
            )
        plt.axis([-5, 5, -5, 5])
        plt.legend()
        plt.grid(True)

        # plot latent variance
        n_points = 20
        x_var_mc, mc_samples = [], 50
        z_sample = (
            torch.stack(torch.meshgrid([torch.linspace(-5, 5, n_points) for _ in range(2)]))
            .reshape(2, -1)
            .T
        )
        for _ in range(mc_samples):
            x_out = self(z_sample.to(self.device))
            x_var = D.Bernoulli(probs=x_out).entropy().sum(dim=[1, 2, 3])
            x_var_mc.append(x_var)

        x_var = torch.stack(x_var_mc).mean(dim=0)
        x_var_std = torch.stack(x_var_mc).std(dim=0)
        plt.contourf(
            z_sample[:, 0].reshape(n_points, n_points),
            z_sample[:, 1].reshape(n_points, n_points),
            x_var.reshape(n_points, n_points).detach().cpu(),
            levels=50,
            zorder=0,
        )
        plt.colorbar()
        self.logger.experiment.log({"latent_entropy": wandb.Image(plt)}, commit=False)
        plt.clf()

        plt.contourf(
            z_sample[:, 0].reshape(n_points, n_points),
            z_sample[:, 1].reshape(n_points, n_points),
            x_var_std.reshape(n_points, n_points).detach().cpu(),
            levels=50,
            zorder=0,
        )
        plt.colorbar()
        self.logger.experiment.log({"latent_entropy_std": wandb.Image(plt)}, commit=False)
        plt.clf()
