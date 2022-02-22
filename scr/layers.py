from copy import deepcopy

import numpy as np
from torch import nn

from scr.utils import rgetattr, rsetattr


class EnsampleLayer(nn.ModuleList):
    def __init__(self, submodule, size=5):
        super().__init__()
        for _ in range(size):
            self.append(deepcopy(submodule))
        self.index = None

    def forward(self, *args, **kwargs):
        if 'index' in kwargs:
            idx = kwargs.pop("index")
        elif self.index is not None:
            idx = self.index
        else:
            idx = np.random.randint(len(self))
        return self[idx](*args, **kwargs)


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