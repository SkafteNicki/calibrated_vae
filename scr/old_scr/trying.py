from torch import nn


def _conv(channel_size, kernel_num):
    return nn.Sequential(
        nn.Conv2d(
            channel_size, kernel_num,
            kernel_size=4, stride=2, padding=1,
        ),
        nn.BatchNorm2d(kernel_num),
        nn.ReLU(),
    )

def _deconv(channel_num, kernel_num):
    return nn.Sequential(
        nn.ConvTranspose2d(
            channel_num, kernel_num,
            kernel_size=4, stride=2, padding=1,
        ),
        nn.BatchNorm2d(kernel_num),
        nn.ReLU(),
    )

def _linear(in_size, out_size, relu=True):
    return nn.Sequential(
        nn.Linear(in_size, out_size),
        nn.ReLU(),
    ) if relu else nn.Linear(in_size, out_size)

channel_num = 3
kernel_num = 128
image_size = 28
z_size = 2

encoder = nn.Sequential(
    _conv(channel_num, kernel_num // 4),
    _conv(kernel_num // 4, kernel_num // 2),
    _conv(kernel_num // 2, kernel_num),
)

# encoded feature's size and volume
feature_size = image_size // 8
feature_volume = kernel_num * (feature_size ** 2)

# q
q_mean = _linear(feature_volume, z_size, relu=False)
q_logvar = _linear(feature_volume, z_size, relu=False)

# projection
project = _linear(z_size, feature_volume, relu=False)

# decoder
decoder = nn.Sequential(
    _deconv(kernel_num, kernel_num // 2),
    _deconv(kernel_num // 2, kernel_num // 4),
    _deconv(kernel_num // 4, channel_num),
    nn.Sigmoid()
)

import torch

h = project(torch.randn(10, 2))
h = h.view(-1, kernel_num, feature_size, feature_size)
o = decoder(h)