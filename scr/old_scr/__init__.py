from functools import partial

from pl_bolts.datamodules import CIFAR10DataModule, FashionMNISTDataModule

from .datamodules import (SVHN, Mnist01Datamodule, Mnist23DataModule,
                          MnistDatamodule, MoonsDatamodule)


def get_data(name: str):
    return {
        "moons": MoonsDatamodule,
        "mnist": MnistDatamodule,
        "mnist01": Mnist01Datamodule,
        "mnist23": Mnist23DataModule,
        "cifar10": CIFAR10DataModule,
        "fmnist": FashionMNISTDataModule,
        "svhn": SVHN
    }[name]
