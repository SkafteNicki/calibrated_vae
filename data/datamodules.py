import csv

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class MoonsDatamodule(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        Xtrain = np.zeros((500, 4))
        with open("data/data_v2.csv") as csvfile:
            reader = csv.reader(csvfile, delimiter=" ")
            for i, row in enumerate(reader):
                if i != 0:
                    Xtrain[i - 1] = [float(r) for r in row[1:]]
        Xtest = np.zeros((500, 4))
        with open("data/data_v2_eval.csv") as csvfile:
            reader = csv.reader(csvfile, delimiter=" ")
            for i, row in enumerate(reader):
                if i != 0:
                    Xtest[i - 1] = [float(r) for r in row[1:]]
        Xtrain = Xtrain.astype("float32")
        test = Xtest.astype("float32")

        train, val = train_test_split(Xtrain, test_size=0.05)
        self.train_data = TensorDataset(train)
        self.val_data = TensorDataset(val)
        self.test_data = TensorDataset(test)

    def train_dataloader(self):
        return DataLoader(self.train_data)

    def val_dataloader(self):
        return DataLoader(self.val_data)

    def test_dataloader(self):
        return DataLoader(self.test_data)


class MnistDatamodule(LightningDataModule):
    def __init__(
        self,
        name: str = "mnist",
        data_dir: str = "",
        labels_to_use=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        batch_size: int = 64,
        **kwargs
    ):
        super().__init__()
        self.name = name
        self.data_dir = data_dir
        self.labels_to_use = labels_to_use
        self.batch_size = batch_size

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage in (None, "fit"):
            data = MNIST(
                self.data_dir,
                train=True,
                download=True,
                transform=transforms.ToTensor(),
            )
            data = self._get_label_set(data)
            self.n_train = int(0.95 * data.targets.shape[0])
            self.n_val = data.targets.shape[0] - self.n_train
            self.train, self.val = random_split(data, [self.n_train, self.n_val])

        if stage in (None, "test"):
            data = MNIST(
                self.data_dir,
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            )
            self.test = self._get_label_set(data)
            self.n_test = self.test.targets.shape[0]

    def _get_label_set(self, data):
        idx = torch.stack([data.targets == val for val in self.labels_to_use]).sum(0).bool()
        data.data = data.data[idx]
        data.targets = data.targets[idx]
        return data

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class SmallMnistDatamodule(MnistDatamodule):
    def __init__(
        self,
        name: str = "mnist",
        data_dir: str = "",
        batch_size: int = 64,
        **kwargs
    ):
        super().__init__(
            name=name,
            data_dir=data_dir,
            labels_to_use=[0, 1],
            batch_size=batch_size
        )


if __name__ == "__main__":
    datamodule = MnistDatamodule(labels_to_use=[0, 1])
    datamodule.prepare_data()
    datamodule.setup()
    print(datamodule.n_train)
    print(datamodule.n_val)
    print(datamodule.n_test)
