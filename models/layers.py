import torch
from torch import nn, Tensor

class Reshape(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], *self.dims)
