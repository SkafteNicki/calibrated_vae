from torch import nn, Tensor
from models.layers import AdditiveRegularizer, Reshape
from pytorch_lightning import LightningModule
from torch import distributions as D
import torch
import wandb
import matplotlib.pyplot as plt


class VAE(LightningModule):
    def __init__(self, latent_size, learning_rate, kl_warmup_steps, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        self.encoder_mu = nn.Linear(512, latent_size)
        self.encoder_std = nn.Sequential(nn.Linear(512, latent_size), nn.Softplus(), AdditiveRegularizer())

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 128),
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

        # warmup scaling
        self.warmup_steps = 0

        # recon loss
        self.recon = nn.BCELoss(reduction="none")

    @property
    def prior(self):
        return D.Independent(
            D.Normal(
                torch.zeros(1, self.hparams.latent_size, device=self.device),
                torch.ones(1, self.hparams.latent_size, device=self.device),
            ),
            1,
        )

    @property
    def beta(self):
        if self.training:
            self.warmup_steps += 1
        return min([float(self.warmup_steps / self.hparams.kl_warmup_steps), 1.0])

    def forward(self, z: Tensor) -> Tensor:
        x = self.decoder(z)
        return x

    def encode(self, x: Tensor) -> Tensor:
        h = self.encoder(x)
        z_mu = self.encoder_mu(h)
        z_std = self.encoder_std(h)
        return z_mu, z_std

    def _step(self, x, state):
        z_mu, z_std = self.encode(x)
        q_dist = D.Independent(D.Normal(z_mu, z_std), 1)
        z = q_dist.rsample()
        x_hat = self(z)

        kl = D.kl_divergence(q_dist, self.prior).mean()
        recon = self.recon(x_hat, x).sum(dim=[1, 2, 3]).mean()

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
            images = wandb.Image(
                torch.cat([x[:8], x_hat[:8]], dim=0),
                caption="Top: Original, Bottom: Reconstruction",
            )
            self.logger.experiment.log({"recon_epoch": images}, commit=False)

            # plot samples
            z_sample = torch.randn(8, self.hparams.latent_size, device=self.device)
            x_sample = self(z_sample)
            images = wandb.Image(x_sample, caption="Samples")
            self.logger.experiment.log({"samples_epoch": images}, commit=False)

        return {"loss": loss, "latents": z_mu.detach(), "labels": y.detach()}

    def training_epoch_end(self, outputs):
        latents = torch.cat([o["latents"] for o in outputs], dim=0)
        labels = torch.cat([o["labels"] for o in outputs], dim=0)

        # plot latent encodings
        for i in labels.unique():
            plt.scatter(
                latents[i==labels, 0].cpu(), latents[i==labels, 1].cpu(), 
                label=str(i), zorder=10, alpha=0.5
            )
        plt.axis([-5, 5, -5, 5])
        plt.legend()
        plt.grid(True)
        self.logger.experiment.log({"latent_space": wandb.Image(plt)}, commit=False)

        # plot latent variance
        n_points = 20
        z_sample = torch.stack(torch.meshgrid([torch.linspace(-5, 5, n_points) for _ in range(2)])).reshape(2, -1).T
        x_out = self(z_sample.to(self.device))
        x_var = D.Bernoulli(probs=x_out).entropy().sum(dim=[1,2,3])
        plt.contourf(
            z_sample[:,0].reshape(n_points, n_points),
            z_sample[:,1].reshape(n_points, n_points),
            x_var.reshape(n_points, n_points).detach(),
            levels=50,
            zorder=0
        )
        plt.colorbar()
        self.logger.experiment.log({"latent_entropy": wandb.Image(plt)}, commit=False)
        plt.clf()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self._step(x, "val")

    def test_step(self, batch, batch_idx):
        x, y = batch
        self._step(x, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=2
            ),
            "monitor": "val_loss",
            "strict": False,
        }
        return [optimizer], [scheduler]