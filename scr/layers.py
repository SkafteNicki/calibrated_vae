from copy import deepcopy
from random import randint
import numpy as np
from torch import nn

from scr.utils import rgetattr, rsetattr


class EnsampleLayer(nn.ModuleList):
    def __init__(self, submodule, size=5):
        super().__init__()
        for _ in range(size):
            self.append(deepcopy(submodule))
        self.size = size

    def forward(self, *args, **kwargs):
        idx = randint(0, self.size-1)
        return self[idx](*args, **kwargs)


class FixedEnsempleLayer(EnsampleLayer):
    def __init__(self, submodule, size=5):
        pass
    
    def forward(self, *args, **kwargs):
        idx = randint(0, self.size-1)
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
