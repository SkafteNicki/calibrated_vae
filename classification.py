import functools
import time
from copy import deepcopy
from gc import callbacks
from multiprocessing.sharedctypes import Value

import numpy as np
import torch
import torchvision
from pytorch_lightning import LightningModule, Trainer, callbacks
from torch import nn
from torchmetrics import Accuracy
from torchmetrics.functional import accuracy

svhn = torchvision.datasets.SVHN(
    root="svhn/", download=True, split="train", transform=torchvision.transforms.ToTensor()
)
train, val, test = torch.utils.data.random_split(svhn, [65931, 3663, 3663])
train_dataloader = torch.utils.data.DataLoader(train, batch_size=512)
val_dataloader = torch.utils.data.DataLoader(val, batch_size=512)
test_dataloader = torch.utils.data.DataLoader(test, batch_size=512)

base_resnet = torchvision.models.resnet18(pretrained=False)


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


class EnsampleLayer(nn.ModuleList):
    def __init__(self, submodule, size=5):
        super().__init__()
        for _ in range(size):
            self.append(deepcopy(submodule))

    def forward(self, *args, **kwargs):
        idx = np.random.randint(len(self))
        return self[idx](*args, **kwargs)


def brier_multi(probs, targets):
    n_class = probs.shape[1]
    return (probs - torch.nn.functional.one_hot(targets, n_class)).pow(2.0).sum(dim=-1).mean()


def create_mixensamble(module, n_ensemble, level="block"):
    if level == "block":
        attr_list = [
            "layer1.0",
            "layer1.1",
            "layer2.0",
            "layer2.1",
            "layer3.0",
            "layer3.1",
            "layer4.0",
            "layer4.1",
        ]
    elif level == "layer":
        attr_list = ["layer1", "layer2", "layer3", "layer4"]
    else:
        raise ValueError()

    base = getattr(module, "base")
    for attr in attr_list:
        rsetattr(base, attr, EnsampleLayer(rgetattr(base, attr), n_ensemble))


class DeepEnsembles(LightningModule):
    def __init__(self):
        super().__init__()
        self.base = base_resnet
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
                callbacks=[callbacks.EarlyStopping(monitor="val_acc", mode="max")],
                max_epochs=2,
            )
            trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
            model.eval()
            models.append(model)
        return models

    @classmethod
    def ensample_predict(cls, model, test_dataloader):
        with torch.no_grad():
            acc, nll, brier = 0.0, 0.0, 0.0
            for batch in test_dataloader:
                x, y = batch
                pred = cls.get_predictions(model, x)
                acc += accuracy(pred, y, num_classes=10).item()
                nll += torch.nn.functional.nll_loss(pred, y).item()
                brier += brier_multi(pred, y).item()
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
        return torch.stack([model(x) for _ in range(100)]).mean(dim=0)

    @classmethod
    def fit(cls, n_ensemble, train_dataloader, val_dataloader=None):
        model = cls(n_ensemble)
        trainer = Trainer(
            logger=False,
            accelerator="auto",
            devices=1,
            callbacks=[callbacks.EarlyStopping(monitor="val_acc", mode="max")],
            max_epochs=2,
        )
        trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
        model.eval()
        return model


class MixBlockEnsembles(DeepEnsembles):
    level = "block"


if __name__ == "__main__":
    with open("classification_scores.txt", "w") as file:
        file.write("model_class, n_ensemble, train_time, acc, nll, brier \n")

    for model_class in [DeepEnsembles, MixLayerEnsembles, MixBlockEnsembles]:
        for n_ensemble in range(1, 2):
            print(
                "======================================================== \n"
                f"Model={model_class.__name__}, n_ensemble={n_ensemble}   \n"
                "======================================================== \n"
            )

            start = time.time()
            model = model_class.fit(n_ensemble, train_dataloader, val_dataloader)
            end = time.time()

            acc, nll, brier = model_class.ensample_predict(model, test_dataloader)

            with open("classification_scores.txt", "a") as file:
                file.write(
                    f"{model_class.__name__}, {n_ensemble}, {end-start}, {acc}, {nll}, {brier} \n"
                )