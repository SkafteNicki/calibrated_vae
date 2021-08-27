import torch
from torch import nn, Tensor

class AdditiveRegulizer(nn.Module):
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
        
    def forward(self, x: Tensor) -> Tensor:
        return x + self.epsilon


class Reshape(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], *self.dims)
