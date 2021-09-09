from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import wandb
from models.layers import EnsembleList, weight_reset
from models.vae import VAE
from torch import distributions as D
from torch import nn


class EVAE(VAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not self.hparams.ensemble_only_decoder:
            self.encoder = EnsembleList(
                [
                    deepcopy(self.encoder).apply(weight_reset)
                    for _ in range(self.hparams.n_ensemble)
                ]
            )
            self.encoder_mu = EnsembleList(
                [
                    deepcopy(self.encoder_mu).apply(weight_reset)
                    for _ in range(self.hparams.n_ensemble)
                ]
            )
            self.encoder_std = EnsembleList(
                [
                    deepcopy(self.encoder_std).apply(weight_reset)
                    for _ in range(self.hparams.n_ensemble)
                ]
            )

        self.decoder = EnsembleList(
            [
                deepcopy(self.decoder).apply(weight_reset)
                for _ in range(self.hparams.n_ensemble)
            ]
        )

    def _step(self, x, state):
        z_mu, z_std = self.encode(x)
        q_dist = D.Independent(D.Normal(z_mu, z_std), 1)
        z = q_dist.rsample()
        x_hat = self(z)

        kl = D.kl_divergence(q_dist, self.prior).mean()
        x_ = x.repeat(self.hparams.n_ensemble, 1, 1, 1).reshape(
            self.hparams.n_ensemble, *x.shape
        )
        recon = self.recon(x_hat, x_).sum(dim=[2, 3, 4]).mean()

        beta = self.beta
        loss = recon + beta * kl

        self.log(f"{state}_kl", kl, prog_bar=True)
        self.log(f"{state}_recon", recon, prog_bar=True)
        self.log(f"{state}_loss", loss, prog_bar=True)
        self.log("kl_beta", beta)

        return loss, z_mu, z_std, x_hat

    def training_step(self, batch, batch_idx=0):
        x, y = batch
        loss, z_mu, _, x_hat = self._step(x, "train")
        if batch_idx == 0:

            # plot reconstructions
            xh_s = x_hat.shape
            images = wandb.Image(
                torch.cat([x[:8], x_hat[:, :8].reshape(xh_s[0] * 8, *xh_s[2:])], dim=0),
                caption="Top: Original, Bottom: Reconstruction",
            )
            self.logger.experiment.log({"recon_epoch": images}, commit=False)

            # plot samples
            z_sample = torch.randn(8, self.hparams.latent_size, device=self.device)
            x_sample = self(z_sample)
            images = wandb.Image(
                x_sample.reshape(xh_s[0] * 8, *xh_s[2:]), caption="Samples"
            )
            self.logger.experiment.log({"samples_epoch": images}, commit=False)

        return {"loss": loss, "latents": z_mu.detach(), "labels": y.detach()}

    def training_epoch_end(self, outputs):
        if self.hparams.ensemble_only_decoder:
            labels = torch.cat([o["labels"] for o in outputs], dim=0)
            latents = torch.cat([o["latents"] for o in outputs], dim=0)
            latents = latents.repeat(self.hparams.n_ensemble, 1).reshape(
                self.hparams.n_ensemble, *latents.shape
            )
        else:
            labels = torch.cat([o["labels"] for o in outputs], dim=0)
            latents = torch.cat([o["latents"] for o in outputs], dim=1)

        n_points = 20
        z_sample = (
            torch.stack(
                torch.meshgrid([torch.linspace(-5, 5, n_points) for _ in range(2)])
            )
            .reshape(2, -1)
            .T
        )
        x_out = self(z_sample.to(self.device))
        x_var = (
            D.Bernoulli(probs=x_out).entropy().sum(dim=[2, 3, 4])
        )  # [N_ensample, n_points*n_points]

        for n in range(self.hparams.n_ensemble):
            # plot latent encodings
            for i in labels.unique():
                plt.scatter(
                    latents[n, i == labels, 0].cpu(),
                    latents[n, i == labels, 1].cpu(),
                    label=str(i),
                    zorder=10,
                    alpha=0.5,
                )
            plt.axis([-5, 5, -5, 5])
            plt.legend()
            plt.grid(True)

            plt.contourf(
                z_sample[:, 0].reshape(n_points, n_points),
                z_sample[:, 1].reshape(n_points, n_points),
                x_var[n].reshape(n_points, n_points).detach().cpu(),
                levels=50,
                zorder=0,
            )
            plt.colorbar()
            self.logger.experiment.log(
                {f"latent_entropy_{n}": wandb.Image(plt)}, commit=False
            )
            plt.clf()

        x_var_std = x_var.std(dim=0)
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

        x_var_mean = x_var.mean(dim=0)
        plt.contourf(
            z_sample[:, 0].reshape(n_points, n_points),
            z_sample[:, 1].reshape(n_points, n_points),
            x_var_mean.reshape(n_points, n_points).detach().cpu(),
            levels=50,
            zorder=0,
        )
        plt.colorbar()
        self.logger.experiment.log(
            {"latent_entropy_mean": wandb.Image(plt)}, commit=False
        )
        plt.clf()
