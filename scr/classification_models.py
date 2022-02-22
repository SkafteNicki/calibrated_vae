from copy import deepcopy

import torch
import torchvision
from pytorch_lightning import LightningModule, Trainer, callbacks
from torch import nn
from torchmetrics import Accuracy
from torchmetrics.functional import accuracy
from tqdm import tqdm

from scr.layers import create_mixensamble
from scr.utils import brierscore


class DeepEnsembles(LightningModule):
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
        self.log("acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        self.val_acc.update(preds, y)
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
            trainer = Trainer(
                logger=False,
                accelerator="auto",
                devices=1,
                callbacks=[
                    callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=5)
                ],
                max_epochs=1,
            )
            trainer.fit(
                model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader
            )
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
                brier += brierscore(pred, y).item()
            acc = acc / len(test_dataloader)
            nll = nll / len(test_dataloader)
            brier = brier / len(test_dataloader)
            return acc, nll, brier


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
        trainer = Trainer(
            logger=False,
            accelerator="auto",
            devices=1,
            callbacks=[
                callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=5)
            ],
            max_epochs=1,
        )
        trainer.fit(
            model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader
        )
        model.eval()
        return model


class MixBlockEnsembles(MixLayerEnsembles):
    level = "block"
