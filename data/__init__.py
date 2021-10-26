from .datamodules import MnistDatamodule, MoonsDatamodule, SmallMnistDatamodule


def get_data(name: str):
    return {
        "moons": MoonsDatamodule,
        "mnist": MnistDatamodule,
        "small_mnist": SmallMnistDatamodule,
    }[name]
