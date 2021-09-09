import matplotlib.pyplot as plt
import torch
import wandb
from models.layers import AdditiveRegularizer, MCDropout, MCDropout2d, Reshape
from models.vae import VAE
from torch import Tensor
from torch import distributions as D
from torch import nn


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
            torch.stack(
                torch.meshgrid([torch.linspace(-5, 5, n_points) for _ in range(2)])
            )
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
        self.logger.experiment.log(
            {"latent_entropy_std": wandb.Image(plt)}, commit=False
        )
        plt.clf()
