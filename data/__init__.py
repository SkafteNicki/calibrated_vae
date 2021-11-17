from .datamodules import MnistDatamodule, MoonsDatamodule, Mnist01Datamodule, Mnist23DataModule, SVHN
from pl_bolts.datamodules import CIFAR10DataModule, FashionMNISTDataModule

from functools import partial

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
