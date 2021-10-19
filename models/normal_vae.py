from copy import deepcopy

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch import distributions as D
from torch import nn

import wandb
from models.layers import (AdditiveRegularizer, NormalSigmoidResample,
                           weight_reset)
from models.mc_dropout import MCVAE
from models.vae import VAE


class NVAE(VAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # copy and remove sigmoid
        self.decoder_mu = deepcopy(self.decoder).apply(weight_reset)
        self.decoder_mu[-1] = nn.Identity()
        self.decoder_std = deepcopy(self.decoder).apply(weight_reset)
        self.decoder_std[-1] = nn.Sequential(nn.Softplus(), AdditiveRegularizer())

        self.decoder = NormalSigmoidResample(self.decoder_mu, self.decoder_std)

    def training_step(self, batch, batch_idx=0):
        output_dict = super().training_step(batch, batch_idx)
        std = self.decoder._std_out
        if std is not None:
            self.log("training_std", std.mean(), prog_bar=True)
            output_dict["loss"] += 100.0 / std.mean()
            self.log("adjusted_training_loss", output_dict["loss"], prog_bar=True)
            # plot std images
            if batch_idx == 0:
                self.logger.experiment.log({"std": wandb.Image(std[:8].detach())}, commit=False)

        return output_dict

    def training_epoch_end(self, outputs):
        self.training_epoch_end_plotter(outputs, mc_samples=self.hparams.mc_samples)
