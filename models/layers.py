import torch
from torch import nn, Tensor

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
            return torch.stack([m(xx) for xx,m in zip(x, self)])
        else:
            return torch.stack([m(x) for m in self])