from .datamodules import MnistDatamodule, MoonsDatamodule, Mnist01Datamodule, Mnist23DataModule
from pl_bolts.datamodules import CIFAR10DataModule, FashionMNISTDataModule

def get_data(name: str):
    return {
        "moons": MoonsDatamodule,
        "mnist": MnistDatamodule,
        "mnist01": Mnist01Datamodule,
        "mnist23": Mnist23DataModule,
        "cifar": CIFAR10DataModule,
        "fmnist": FashionMNISTDataModule,
    }[name]
