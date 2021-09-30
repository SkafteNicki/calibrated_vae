from copy import deepcopy
from itertools import chain

import matplotlib.pyplot as plt
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch import Tensor
from torch import distributions as D
from torch import nn

import wandb
from models.ensample import EVAE
from models.layers import EnsembleList, weight_reset


class DEVAE(EVAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.encoder2 = EnsembleList(
            [deepcopy(self.encoder).apply(weight_reset) for _ in range(self.hparams.n_ensemble)]
        )
        self.encoder_mu2 = EnsembleList(
            [deepcopy(self.encoder_mu).apply(weight_reset) for _ in range(self.hparams.n_ensemble)]
        )
        self.encoder_std2 = EnsembleList(
            [deepcopy(self.encoder_std).apply(weight_reset) for _ in range(self.hparams.n_ensemble)]
        )

        self.switch = True
        self.stop_counter = 0
        self.automatic_optimization = False

    def encode(self, x: Tensor) -> Tensor:
        if self.switch:
            h = self.encoder(x)
            z_mu = self.encoder_mu(h)
            z_std = self.encoder_std(h)
        else:
            h = self.encoder2(x)
            z_mu = self.encoder_mu2(h)
            z_std = self.encoder_std2(h)
        return z_mu, z_std

    def training_step(self, batch, batch_idx=0):
        opt1, opt2 = self.optimizers()
        optimizer = opt1 if self.switch else opt2
        optimizer.zero_grad()

        output_dict = super().training_step(batch, batch_idx)
        loss = output_dict["loss"]

        self.manual_backward(loss)
        optimizer.step()

        return output_dict

    def configure_optimizers(self):
        optimizer1 = torch.optim.Adam(
            chain(
                self.encoder.parameters(),
                self.encoder_mu.parameters(),
                self.encoder_std.parameters(),
                self.decoder.parameters(),
            ),
            lr=self.hparams.learning_rate,
        )
        optimizer2 = torch.optim.Adam(
            chain(
                self.encoder2.parameters(),
                self.encoder_mu2.parameters(),
                self.encoder_std2.parameters(),
            ),
            lr=self.hparams.learning_rate,
        )
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer1, mode="min", patience=int(self.hparams.patience / 2)
        )
        scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer2, mode="min", patience=int(self.hparams.patience / 2)
        )
        return [optimizer1, optimizer2], [scheduler1, scheduler2]

    def on_validation_epoch_end(self) -> None:
        sch = self.lr_schedulers()
        scheduler = sch[0] if self.switch else sch[1]
        scheduler.step(self.trainer.callback_metrics["val_loss"])

    def on_train_epoch_end(self):
        if self.trainer.should_stop and self.stop_counter < 1:  # early stopping got triggered
            self.print(f"Changing to phase 2 after epoch {self.current_epoch}")
            self.switch = False  # activate second phase

            # prevent stopping
            self.trainer.should_stop = False

            # reset early stopping
            for cb in self.trainer.callbacks:
                if isinstance(cb, EarlyStopping):
                    cb.__init__(monitor="val_loss", mode="min", patience=self.hparams.patience)
                    cb.on_init_end(self.trainer)

            self.stop_counter += 1
