from typing import Tuple

import matplotlib.pyplot as plt
import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch import distributions as D
from torch import nn
from torch.utils.data.dataloader import DataLoader

import wandb
from models.layers import AdditiveRegularizer, Reshape, get_activation_layer


class VAE(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        hidden_size = 512 if self.hparams.n_channels==1 else 1152
        self.encoder = nn.Sequential(
            nn.Conv2d(self.hparams.n_channels, 32, 3, stride=2),
            get_activation_layer(self.hparams.activation_fn),
            nn.Conv2d(32, 64, 3, stride=2),
            get_activation_layer(self.hparams.activation_fn),
            nn.Conv2d(64, 128, 3, stride=2),
            get_activation_layer(self.hparams.activation_fn),
            nn.Flatten(),
        )
        self.encoder_mu = nn.Linear(hidden_size, self.hparams.latent_size)
        self.encoder_std = nn.Sequential(
            nn.Linear(hidden_size, self.hparams.latent_size),
            nn.Softplus(),
            AdditiveRegularizer(),
        )

        final_decoder = [
            nn.ConvTranspose2d(8, 8, 3),
            get_activation_layer(self.hparams.activation_fn),
            nn.Conv2d(8, self.hparams.n_channels, 2)
        ] if self.hparams.n_channels==3 else [
            nn.Conv2d(8, self.hparams.n_channels, 4),
        ]
        # TODO: use pl_bolts resnet18_encoder/decoder for cifar10

        self.decoder = nn.Sequential(
            nn.Linear(self.hparams.latent_size, 128),
            get_activation_layer(self.hparams.activation_fn),
            Reshape(128, 1, 1),
            nn.ConvTranspose2d(128, 64, 3, stride=2),
            get_activation_layer(self.hparams.activation_fn),
            nn.ConvTranspose2d(64, 32, 3, stride=2),
            get_activation_layer(self.hparams.activation_fn),
            nn.ConvTranspose2d(32, 16, 3, stride=2),
            get_activation_layer(self.hparams.activation_fn),
            nn.ConvTranspose2d(16, 8, 3, stride=2),
            get_activation_layer(self.hparams.activation_fn),
            *final_decoder,
            nn.Sigmoid(),
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

    def log_prob(self, x: Tensor) -> Tensor:
        _, _, _, x_hat, _ = self.encode_decode()
        dist = D.Independent(D.Bernoulli(probs=x_hat), 3)
        return dist.log_prob(x)

    def calc_log_prob(self, dataloader: DataLoader) -> Tensor:
        current_state = self.training
        self.eval()
        log_probs = []
        with torch.no_grad():
            for batch in dataloader:
                x, _ = batch
                _, _, _, x_hat, _ = self.encode_decode(x.to(self.device))
                d = D.Independent(D.Bernoulli(probs=x_hat), 3)
                log_probs.append(d.log_prob(x.to(self.device)))
        log_probs = torch.cat(log_probs, dim=0)
        self.training = current_state
        return log_probs

    def _step(self, x, state):
        z_mu, z_std, z, x_hat, kl = self.encode_decode(x)

        recon = self.recon(x_hat, x).sum(dim=[1, 2, 3]).mean()

        beta = self.beta
        loss = recon + beta * kl

        self.log(f"{state}_kl", kl, prog_bar=True)
        self.log(f"{state}_recon", recon, prog_bar=True)
        self.log(f"{state}_loss", loss, prog_bar=True)
        self.log("kl_beta", beta)

        return loss, z_mu, z_std, z, x_hat

    def training_step(self, batch, batch_idx=0):
        x, y = batch
        loss, z_mu, _, _, x_hat = self._step(x, "train")
        
        if batch_idx == 0:
            # plot reconstructions
            images = wandb.Image(
                torch.cat([x[:8], x_hat[:8]], dim=0),
                caption="Top: Original, Bottom: Reconstruction",
            )
            self.logger.experiment.log({"recon": images})

            # plot samples
            z_sample = self.prior.sample((8,))
            x_sample = self(z_sample)
            images = wandb.Image(x_sample, caption="Samples")
            self.logger.experiment.log({"samples": images})

        return {"loss": loss, "latents": z_mu.detach(), "labels": y.detach()}

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        self._step(x, "val")

    def test_step(self, batch, batch_idx):
        x, _ = batch
        self._step(x, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=int(self.hparams.patience / 2), verbose=True
            ),
            "monitor": "val_loss",
            "strict": False,
        }
        return [optimizer], [scheduler]

    def training_epoch_end(self, outputs):
        self.training_epoch_end_plotter(outputs)

    def training_epoch_end_plotter(
        self, outputs, mc_samples: int = 1, plot_bound: float = 7.0, n_points: int = 30,
    ) -> None:
        with torch.no_grad():
            latents = torch.cat([o["latents"] for o in outputs], dim=0)
            labels = torch.cat([o["labels"] for o in outputs], dim=0)

            # plot latent encodings
            for i in labels.unique():
                plt.scatter(
                    latents[i == labels, 0].cpu(),
                    latents[i == labels, 1].cpu(),
                    label=str(i.item()),
                    zorder=10,
                    alpha=0.5,
                )
            plt.axis([-plot_bound, plot_bound, -plot_bound, plot_bound])
            plt.legend()
            plt.grid(True)

            # plot latent variance
            linspaces = [torch.linspace(-plot_bound, plot_bound, n_points) for _ in range(2)]
            meshgrid = torch.meshgrid(linspaces)
            z_sample = torch.stack(meshgrid).reshape(2, -1).T

            samples = []
            for _ in range(mc_samples):
                x_out = self(z_sample.to(self.device))
                x_var = D.Bernoulli(probs=x_out).entropy().sum(dim=[1, 2, 3])
                samples.append(x_var)

            x_var = torch.stack(samples).mean(dim=0)
            x_var_std = torch.stack(samples).std(dim=0)
            x_var_std[torch.isnan(x_var_std)] = 0.0

            plt.contourf(
                z_sample[:, 0].reshape(n_points, n_points),
                z_sample[:, 1].reshape(n_points, n_points),
                x_var.reshape(n_points, n_points).detach().cpu(),
                levels=50,
                zorder=0,
            )
            plt.colorbar()
            self.logger.experiment.log({"latent_entropy": wandb.Image(plt)})
            plt.clf()

            plt.contourf(
                z_sample[:, 0].reshape(n_points, n_points),
                z_sample[:, 1].reshape(n_points, n_points),
                x_var_std.reshape(n_points, n_points).detach().cpu(),
                levels=50,
                zorder=0,
            )
            plt.colorbar()
            self.logger.experiment.log({"latent_entropy_std": wandb.Image(plt)})
            plt.clf()
