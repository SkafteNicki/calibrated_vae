import bnn
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch import distributions as D
from torch import nn

import wandb
from models.layers import AdditiveRegularizer, MCDropout, MCDropout2d, Reshape
from models.vae import VAE


class BVAE(VAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        bnn.bayesianize_(self, inference="inducing", inducing_rows=64, inducing_cols=64)

    def training_step(self, batch, batch_idx=0):
        output = super().training_step(batch, batch_idx)
        loss = output["loss"]
        kl_bayes = sum(m.kl_divergence() for m in self.modules() if hasattr(m, "kl_divergence"))
        loss += kl_bayes / 50000

    def training_epoch_end(self, outputs):
        self.training_epoch_end_plotter(outputs, mmc_samples=50)
