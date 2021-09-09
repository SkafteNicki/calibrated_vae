from torch import nn, Tensor
from models.layers import AdditiveRegularizer, weight_reset, NormalSigmoidResample
from models.vae import VAE
from models.mc_dropout import MCVAE
from copy import deepcopy
from torch import distributions as D
import torch
import wandb
import matplotlib.pyplot as plt


class NVAE(VAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # copy and remove sigmoid
        self.decoder_mu = deepcopy(self.decoder).apply(weight_reset)
        self.decoder_mu[-1] = nn.Identity()
        self.decoder_std = deepcopy(self.decoder).apply(weight_reset)
        self.decoder_std[-1] = nn.Sequential(
            nn.Softplus(), 
            AdditiveRegularizer()
        )
        
        self.decoder = NormalSigmoidResample(self.decoder_mu, self.decoder_std)

    def training_step(self, batch, batch_idx=0):
        output_dict = super().training_step(batch, batch_idx)
        std = self.decoder._std_out
        if std is not None:
            self.log("training_std", std.mean(), prog_bar=True)
            output_dict['loss'] += 100.0 / std.mean()
            self.log("adjusted_training_loss", output_dict['loss'], prog_bar=True)
            # plot std images
            if batch_idx == 0:
                self.logger.experiment.log({"std": wandb.Image(std[:8].detach())}, commit=False)

        return output_dict

    def training_epoch_end(self, outputs):
        latents = torch.cat([o["latents"] for o in outputs], dim=0)
        labels = torch.cat([o["labels"] for o in outputs], dim=0)

        # plot latent encodings
        for i in labels.unique():
            plt.scatter(
                latents[i==labels, 0].cpu(), latents[i==labels, 1].cpu(), 
                label=str(i.cpu()), zorder=10, alpha=0.5
            )
        plt.axis([-5, 5, -5, 5])
        plt.legend()
        plt.grid(True)

        # plot latent variance
        n_points = 20
        x_var_mc, mc_samples = [], 50
        z_sample = torch.stack(torch.meshgrid([torch.linspace(-5, 5, n_points) for _ in range(2)])).reshape(2, -1).T
        for _ in range(mc_samples):
            x_out = self(z_sample.to(self.device))
            x_var = D.Bernoulli(probs=x_out).entropy().sum(dim=[1,2,3])
            x_var_mc.append(x_var)

        x_var = torch.stack(x_var_mc).mean(dim=0)
        x_var_std = torch.stack(x_var_mc).std(dim=0)
        plt.contourf(
            z_sample[:,0].reshape(n_points, n_points),
            z_sample[:,1].reshape(n_points, n_points),
            x_var.reshape(n_points, n_points).detach().cpu(),
            levels=50,
            zorder=0
        )
        plt.colorbar()
        self.logger.experiment.log({"latent_entropy": wandb.Image(plt)}, commit=False)
        plt.clf()

        plt.contourf(
            z_sample[:,0].reshape(n_points, n_points),
            z_sample[:,1].reshape(n_points, n_points),
            x_var_std.reshape(n_points, n_points).detach().cpu(),
            levels=50,
            zorder=0
        )
        plt.colorbar()
        self.logger.experiment.log({"latent_entropy_std": wandb.Image(plt)}, commit=False)
        plt.clf()

        std = self.decoder._std_out
        if std is not None:
            plt.contourf(
                z_sample[:,0].reshape(n_points, n_points),
                z_sample[:,1].reshape(n_points, n_points),
                std.sum(dim=[1,2,3]).reshape(n_points, n_points).detach().cpu(),
                levels=50,
                zorder=0
            )
            plt.colorbar()
            self.logger.experiment.log({"latent_entropy_std_2": wandb.Image(plt)}, commit=False)
            plt.clf()
