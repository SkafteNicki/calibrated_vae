from .datamodules import MnistDatamodule, MoonsDatamodule, Mnist01Datamodule, Mnist23DataModule


def get_data(name: str):
    return {
        "moons": MoonsDatamodule,
        "mnist": MnistDatamodule,
        "mnist01": Mnist01Datamodule,
        "mnist23": Mnist23DataModule
    }[name]
