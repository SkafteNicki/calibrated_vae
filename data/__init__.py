from .datamodules import MnistDatamodule, MoonsDatamodule


def get_data(name: str):
    return {"moons": MoonsDatamodule, "mnist": MnistDatamodule}[name]
