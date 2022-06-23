import os
from copy import deepcopy
from itertools import chain
from random import randint
from typing import Tuple

import torch
from pytorch_lightning import LightningModule, Trainer, callbacks, loggers
from torch import Tensor
from torch import distributions as D
from torch import nn
from tqdm import tqdm

from scr.layers import AddBound, EnsembleLayer, Reshape


class VAE(LightningModule):
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
            ],
            "max_epochs": 50,
        }
        return config

    def __init__(self, **kwargs):
        super().__init__()

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
            nn.Linear(2, 128),
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

        self.encoder_mu = nn.Linear(512, 2)
        self.encoder_std = nn.Sequential(
            nn.Linear(512, 2),
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
        if self._prior is None or (
            self._prior is not None and self._prior.base_dist.loc.device != self.device
        ):
            self._prior = D.Independent(
                D.Normal(
                    torch.zeros(1, 2, device=self.device),
                    torch.ones(1, 2, device=self.device),
                ),
                1,
            )
        return self._prior

    @property
    def beta(self):
        if self.training:
            self.warmup_steps += 1
            return min([float(self.warmup_steps / 1000), 1.0])
        else:
            return 1.0

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
        kl = D.kl_divergence(q_dist, self.prior)
        return z_mu, z_std, z, x_hat, kl

    def training_step(self, batch, batch_idx):
        x, y = batch
        z_mu, z_std, z, x_hat, kl = self.encode_decode(x)
        recon = self.recon(x_hat, x).sum(dim=[1, 2, 3])
        loss = (recon + kl).mean()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z_mu, z_std, z, x_hat, kl = self.encode_decode(x)
        recon = self.recon(x_hat, x).sum(dim=[1, 2, 3])
        loss = (recon + kl).mean()
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    @classmethod
    def fit(cls, n_ensemble, train_dataloader, val_dataloader=None):
        model = cls()
        config = cls.trainer_config
        trainer = Trainer(**config)
        trainer.fit(
            model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader
        )
        model.eval()
        return model

    @staticmethod
    def save_checkpoint(model, path):
        torch.save(model.state_dict(), path)

    @classmethod
    def load_checkpoint(cls, path, n_ensemble=None):
        model = cls()
        model.load_state_dict(torch.load(path))
        return model

    @staticmethod
    def refit_encoder(model, dataloader):
        model.configure_optimizer = lambda: torch.optim.Adam(
            chain(
                model.encoder.parameters(),
                model.encoder_mu.parameters(),
                model.encoder_std.parameters(),
            ),
            lr=1e-4
        )
        trainer = Trainer(
            logger=False, 
            max_epochs=2,
            accelerator="auto",
            devices=1,
            enable_model_summary=False,
        )   
        trainer.fit(model, dataloader)

    @staticmethod
    @torch.inference_mode()
    def calc_score(model, score_method, dataloader):
        if score_method == "logprob":
            score = []
            for batch in tqdm(dataloader):
                x, _ = batch
                _, _, _, x_hat, _ = model.encode_decode(x.to(model.device))
                dist = D.Independent(D.Bernoulli(probs=x_hat, validate_args=False), 3)
                score.append(dist.log_prob(x.to(model.device)))
            return torch.cat(score, dim=0)
        else:
            raise ValueError("Unknown scoring method")


class EnsembleVAE(VAE):
    @classmethod
    def fit(cls, n_ensemble, train_dataloader, val_dataloader=None):
        models = []
        for _ in range(n_ensemble):
            model = cls()
            config = cls.trainer_config
            trainer = Trainer(**config)
            trainer.fit(
                model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader
            )
            model.eval()
            models.append(deepcopy(model))
        return models

    @staticmethod
    def save_checkpoint(model, path):
        torch.save([m.state_dict() for m in model], path)

    @classmethod
    def load_checkpoint(cls, path, n_ensemble=None):
        model = cls()
        states = torch.load(path)
        models = []
        for s in states:
            temp = deepcopy(model)
            temp.load_state_dict(s)
            temp.eval()
            models.append(temp)
        return models

    @staticmethod
    def refit_encoder(model, dataloader):
        for m in model:
            super(EnsembleVAE, EnsembleVAE).refit_encoder(m, dataloader)

    @staticmethod
    @torch.inference_mode()
    def calc_score(model, score_method, dataloader):
        score = []
        for batch in tqdm(dataloader):
            x, _ = batch
            s = []
            for m in model:
                _, _, _, x_hat, _ = m.encode_decode(x.to(m.device))
                dist = D.Independent(D.Bernoulli(probs=x_hat, validate_args=False), 3)
                if score_method != "entropy":
                    s.append(dist.log_prob(x.to(m.device)))
                else:
                    s.append(dist.entropy())
            s = torch.stack(s, dim=0)
            if score_method == "logprob":
                score.append(s.mean(dim=0))
            elif score_method == "waic":
                score.append(s.mean(dim=0) - s.var(dim=0))
            elif score_method == "entropy":
                score.append(s.var(dim=0))
            else:
                raise ValueError("Unknown scoring method")    
        return torch.cat(score, dim=0)


class MixVAE(VAE):
    def __init__(self, n_ensemble):
        super().__init__()
        self.n_ensemble = n_ensemble
        self.encoder = EnsembleLayer(self.encoder, n_ensemble)
        self.decoder = EnsembleLayer(self.decoder, n_ensemble)
        self.encoder_mu = EnsembleLayer(self.encoder_mu, n_ensemble)
        self.encoder_std = EnsembleLayer(self.encoder_std, n_ensemble)

    def encode(self, x: Tensor) -> Tensor:
        idx = randint(0, self.n_ensemble - 1)
        h = self.encoder[idx](x)
        z_mu = self.encoder_mu[idx](h)
        z_std = self.encoder_std[idx](h)
        return z_mu, z_std

    @classmethod
    def fit(cls, n_ensemble, train_dataloader, val_dataloader=None):
        model = cls(n_ensemble)
        config = cls.trainer_config
        trainer = Trainer(**config)
        trainer.fit(
            model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader
        )
        model.eval()
        return model

    @staticmethod
    @torch.inference_mode()
    def calc_score(model, score_method, dataloader):
        score = []
        for batch in tqdm(dataloader):
            x, _ = batch
            s = []
            for _ in range(25):  # number of samples
                _, _, _, x_hat, _ = model.encode_decode(x.to(model.device))
                dist = D.Independent(D.Bernoulli(probs=x_hat, validate_args=False), 3)
                if score_method != "entropy":
                    s.append(dist.log_prob(x.to(model.device)))
                else:
                    s.append(dist.entropy())
            s = torch.stack(s, dim=0)
            if score_method == "logprob":
                score.append(s.mean(dim=0))
            elif score_method == "waic":
                score.append(s.mean(dim=0) - s.var(dim=0))
            elif score_method == "entropy":
                score.append(s.var(dim=0))
            else:
                raise ValueError("Unknown scoring method")    
        return torch.cat(score, dim=0)

    @classmethod
    def load_checkpoint(cls, path, n_ensemble=None):
        model = cls(n_ensemble)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model


def get_model(name: str) -> LightningModule:
    if name == "vae":
        model_class = VAE
    elif name == "ensemblevae":
        model_class = EnsembleVAE
    elif name == "mixvae":
        model_class = MixVAE
    else:
        raise ValueError("Unknown model")
    return model_class


def get_model_from_file(path: str):
    if "ensemblevae" in path:
        model_class = EnsembleVAE
    elif "mixvae" in path:
        model_class = MixVAE
    elif "vae" in path:
        model_class = VAE
    else:
        raise ValueError("Unknown model")
    model = model_class.load_checkpoint(path, n_ensemble=5)
    return model, model_class
