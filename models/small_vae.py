from torch import nn, Tensor
from utils import AdditiveRegulizer


class Encoder(nn.Module):
    def __init__(self, latent_size: int = 2, n_neurons: int = 50):
        self.encoder = nn.Sequential(
            nn.Linear(4, n_neurons),
            nn.LeakyReLU(),
        )
        self.encoder_mu = nn.Linear(n_neurons, latent_size)
        
        self.encoder_std = nn.Sequential(
            nn.Linear(n_neurons, latent_size),
            nn.Softplus(),
            AdditiveRegulizer()
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.encoder(x)
        return self.encoder_mu(h), self.encoder_std(h)


class DecoderNormal(nn.Module):
    def __init__(self, latent_size: int = 2, n_neurons: int = 50):
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, n_neurons),
            nn.ReLU(),
        )
        self.decoder_mu = nn.Linear(n_neurons, 4)
        self.decoder_std = nn.Sequential(
            nn.Linear(n_neurons, 4),
            nn.Softplus(),
            AdditiveRegulizer()
        )
        
    def forward(self, z: Tensor) -> Tensor:
        h = self.decoder(z)
        return self.decoder_mu(h), self.decoder_std(h)
        
    

class VAE(nn.Module):
    pass

