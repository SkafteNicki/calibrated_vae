from copy import deepcopy

from torch import nn
import numpy as np


class EnsampleLayer(nn.ModuleList):
    def __init__(self, submodule, size=5):
        super().__init__()
        for _ in range(size):
            self.append(deepcopy(submodule))

    def forward(self, *args, **kwargs):
        idx = np.random.randint(len(self))
        return self[idx](*args, **kwargs)