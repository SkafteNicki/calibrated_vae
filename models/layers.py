import torch
from torch import Tensor
from torch import distributions as D
from torch import nn
from torch.nn.modules.activation import MultiheadAttention


class AdditiveRegularizer(nn.Module):
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        return x + self.epsilon


class Reshape(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], *self.dims)


class MCDropout(nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__(p, inplace)
        self.training = True

    def train(self, mode):
        # disable switching into eval mode
        return self


class MCDropout2d(nn.Dropout2d):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__(p, inplace)
        self.training = True

    def train(self, mode):
        # disable switching into eval mode
        return self


class EnsembleList(nn.ModuleList):
    def forward(self, x: Tensor) -> Tensor:
        # If input as
        if x.shape[0] == len(self):
            return torch.stack([m(xx) for xx, m in zip(x, self)])
        else:
            return torch.stack([m(x) for m in self])


class NormalSigmoidResample(nn.Module):
    def __init__(self, mean_module, std_module):
        super().__init__()
        self.mean_module = mean_module
        self.std_module = std_module
        self._std_out = None

    def forward(self, x: Tensor) -> Tensor:
        mean = self.mean_module(x)
        std = self.std_module(x)
        self._std_out = std
        d = D.Normal(mean, std)
        return torch.sigmoid(d.rsample())


def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()
