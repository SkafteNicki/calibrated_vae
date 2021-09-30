import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch import distributions as D
from torch import nn

import wandb
from models.layers import AdditiveRegularizer, MCDropout, MCDropout2d, Reshape
from models.vae import VAE


class MCVAE(VAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        self.encoder_mu = nn.Linear(512, self.hparams.latent_size)
        self.encoder_std = nn.Sequential(
            nn.Linear(512, self.hparams.latent_size),
            nn.Softplus(),
            AdditiveRegularizer(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hparams.latent_size, 128),
            MCDropout(p=self.hparams.prob),
            nn.LeakyReLU(),
            Reshape(128, 1, 1),
            nn.ConvTranspose2d(128, 64, 3, stride=2),
            MCDropout2d(p=self.hparams.prob),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2),
            MCDropout2d(p=self.hparams.prob),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2),
            MCDropout2d(p=self.hparams.prob),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2),
            MCDropout2d(p=self.hparams.prob),
            nn.LeakyReLU(),
            nn.Conv2d(8, 1, 4),
            MCDropout2d(p=self.hparams.prob),
            nn.Sigmoid(),
        )

    def training_epoch_end(self, outputs):
        self.training_epoch_end_plotter(outputs, mmc_samples=50)
