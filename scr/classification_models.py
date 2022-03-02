from copy import deepcopy
import os

import torch
import torchvision
from pytorch_lightning import LightningModule, Trainer, callbacks, loggers
from torch import nn
from torchmetrics import Accuracy
from torchmetrics.functional import accuracy
from tqdm import tqdm
import wandb

from scr.layers import create_mixensamble
from scr.utils import brierscore


class DeepEnsembles(LightningModule):
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
                callbacks.RichProgressBar(leave=True),
            ],
            "min_epochs": 20,
        }
        return config

    @staticmethod
    def update_logger_config(config):
        if "ENABLE_LOGGING" in os.environ:
            config.pop("logger")
            config.pop("callbacks")
            wandb.config.update(config)

    def __init__(self):
        super().__init__()
        self.base = torchvision.models.resnet18(pretrained=False)
        self.base.fc = nn.Linear(512, 10)
        self.loss_fn = nn.CrossEntropyLoss()

        self.val_acc = Accuracy(num_classes=10)

    def forward(self, x):
        return self.base(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = accuracy(preds, y, num_classes=10)
        self.log("loss", loss)
        self.log("acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        val_loss = self.loss_fn(preds, y)
        self.val_acc.update(preds, y)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)

    @staticmethod
    def get_predictions(model, x):
        return torch.stack([m(x) for m in model]).mean(dim=0)

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
            cls.update_logger_config(config)
            model.eval()
            models.append(deepcopy(model))
        return models

    @classmethod
    def ensample_predict(cls, model, test_dataloader):
        with torch.no_grad():
            acc, nll, brier = 0.0, 0.0, 0.0
            for batch in tqdm(test_dataloader, desc="Evaluating test set"):
                x, y = batch
                pred = cls.get_predictions(model, x)
                acc += accuracy(pred, y, num_classes=10).item()
                nll += torch.nn.functional.nll_loss(pred, y).item()
                brier += brierscore(pred.softmax(dim=-1), y).item()
            acc = acc / len(test_dataloader)
            nll = nll / len(test_dataloader)
            brier = brier / len(test_dataloader)
            return acc, nll, brier

    @staticmethod
    def save_checkpoint(model, path):
        torch.save([m.state_dict() for m in model], path)

    @classmethod
    def load_checkpoint(cls, path, n_ensemble=None):
        model = cls()
        states = torch.load(path)
        return [deepcopy(model.load_state_dict(s)).eval() for s in states]


class MixLayerEnsembles(DeepEnsembles):
    level = "layer"

    def __init__(self, n_ensemble):
        super().__init__()
        create_mixensamble(self, n_ensemble, level=self.level)

    @staticmethod
    def get_predictions(model, x):
        return torch.stack([model(x) for _ in range(25)]).mean(dim=0)

    @classmethod
    def fit(cls, n_ensemble, train_dataloader, val_dataloader=None):
        model = cls(n_ensemble)
        config = cls.trainer_config
        trainer = Trainer(**config)
        trainer.fit(
            model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader
        )
        cls.update_logger_config(config)
        model.eval()
        return model

    @staticmethod
    def save_checkpoint(model, path):
        torch.save(model.state_dict(), path)

    @classmethod
    def load_checkpoint(cls, path, n_ensemble=None):
        model = cls(n_ensemble)
        state = torch.load(path)
        return (model.load_state_dict(state)).eval()


class MixBlockEnsembles(MixLayerEnsembles):
    level = "block"


class DeepMixLayerEnsembles(MixLayerEnsembles):
    @staticmethod
    def get_predictions(model, x):
        return torch.stack([m(x) for m in model for _ in range(25)]).mean(dim=0)

    @staticmethod
    def save_checkpoint(model, path):
        torch.save([m.state_dict() for m in model], path)

    @classmethod
    def fit(cls, n_ensemble, train_dataloader, val_dataloader=None):
        models = []
        for _ in range(n_ensemble):
            model = cls(n_ensemble)
            config = cls.trainer_config
            trainer = Trainer(**config)
            trainer.fit(
                model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader
            )
            if "ENABLE_LOGGING" in os.environ:
                wandb.config.update(**config)
            model.eval()
            models.append(deepcopy(model))
        return models

    @staticmethod
    def save_checkpoint(model, path):
        torch.save([m.state_dict() for m in model], path)

    @classmethod
    def load_checkpoint(cls, path, n_ensemble=None):
        model = cls(n_ensemble)
        states = torch.load(path)
        return [deepcopy(model.load_state_dict(s)).eval() for s in states]


def get_model(model_name):
    return {
        "DeepEnsembles": DeepEnsembles,
        "MixLayerEnsembles": MixLayerEnsembles,
        "MixBlockEnsembles": MixBlockEnsembles,
        "DeepMixLayerEnsembles": DeepMixLayerEnsembles,
    }[model_name]


def get_classification_model_from_file(path):
    if "DeepMixLayer" in path:
        model_class = DeepMixLayerEnsembles
    elif "MixLayer" in path:
        model_class = MixLayerEnsembles
    elif "MixBlock" in path:
        model_class = MixBlockEnsembles
    elif "DeepEnsemble" in path:
        model_class = DeepEnsembles

    n_ensemble = int(path[:-3].split("_")[-1])
    return model_class.load_checkpoint(path, n_ensemble)
