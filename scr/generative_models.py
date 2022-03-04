from typing import Tuple
import os

from pytorch_lightning import LightningModule, loggers, callbacks
import torch
from torch import nn, distributions as D, Tensor

from scr.layers import Reshape, AddBound

class DeepEnsembles(LightningModule):
    @classmethod
    @property
    def trainer_config(cls):
        config = {
            "logger": loggers.WandbLogger()
            if "ENABLE_LOGGING" in os.environ
            else False,
            "accelerator": "auto",
            "num_sanity_val_steps": 0,
            "devices": 1,
            "callbacks": [
                callbacks.EarlyStopping(
                    monitor="val_loss", mode="min", patience=10, verbose=True
                ),
                callbacks.RichProgressBar(leave=True),
            ],
            "min_epochs": 20,
        }
        return config

    def __init__(self, **kwargs):
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(32, 64, 3, stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(64, 128, 3, stride=2),
                nn.LeakyReLU(),
                nn.Flatten(),
            )
        self.decoder = nn.Sequential(
            nn.Linear(self.hparams.latent_size, 128),
            nn.LeakyReLU(),
            Reshape(128, 1, 1),
            nn.ConvTranspose2d(128, 64, 3, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(8, 1, 4),
            nn.Sigmoid(),
        )

        self.encoder_mu = nn.Linear(512, self.hparams.latent_size)
        self.encoder_std = nn.Sequential(
            nn.Linear(512, self.hparams.latent_size),
            nn.Softplus(),
            AddBound(1e-6),
        )

        # warmup scaling of kl term
        self.warmup_steps = 0

        # recon loss
        self.recon = nn.BCELoss(reduction="none")

        # storing prior
        self._prior = None

    @property
    def prior(self):
        if self._prior is None or \
            (self._prior is not None and self._prior.base_dist.loc.device != self.device):
            ls = self.hparams.latent_size
            self._prior = D.Independent(
                D.Normal(
                    torch.zeros(1, ls, device=self.device),
                    torch.ones(1, ls, device=self.device),
                ),
                1,
            )
        return self._prior

    @property
    def beta(self):
        if self.training:
            self.warmup_steps += 1
        return min([float(self.warmup_steps / self.hparams.kl_warmup_steps), 1.0])

    def forward(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def encode(self, x: Tensor) -> Tensor:
        h = self.encoder(x)
        z_mu = self.encoder_mu(h)
        z_std = self.encoder_std(h)
        return z_mu, z_std

    def encode_decode(self, x: Tensor) -> Tuple[Tensor, Tensor, D.Distribution, Tensor]:
        z_mu, z_std = self.encode(x)
        q_dist = D.Independent(D.Normal(z_mu, z_std), 1)
        z = q_dist.rsample()
        x_hat = self(z)
        kl = D.kl_divergence(q_dist, self.prior).mean()
        return z_mu, z_std, z, x_hat, kl