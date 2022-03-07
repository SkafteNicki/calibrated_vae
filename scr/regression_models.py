from functools import partial

import torch
from torch import nn

from scr.layers import EnsampleLayer


class Ensamble(nn.Module):
    def __init__(self, input_dim, hidden_dim, activate_fn, ensemble_size = 5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), activate_fn, nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

    def loss(self, x, y):
        return (self(x) - y).pow(2.0).mean()

    @staticmethod
    def ensample_predict(model_list, x, scaler=None):
        with torch.no_grad():
            ypred = [model(x) for model in model_list]
            if scaler is not None:
                ypred = [scaler.inverse_transform(yp) for yp in ypred]
            ypred = torch.tensor(ypred)
            return ypred.mean(dim=0), ypred.var(dim=0)
        

class EnsambleNLL(nn.Module):
    def __init__(self, input_dim, hidden_dim, activate_fn, ensemble_size = 5):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activate_fn,
        )
        self.head1 = nn.Linear(hidden_dim, 1)
        self.head2 = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())

    def forward(self, x):
        base = self.base(x)
        mean, var = self.head1(base), self.head2(base) + 1e-6
        return [mean, var]

    def loss(self, x, y):
        mean, var = self(x)
        return (var.log() / 2 + (mean - y) ** 2 / (2 * var)).mean()

    @staticmethod
    def ensample_predict(model_list, x, scaler=None):
        with torch.no_grad():
            means = [model(x)[0] for model in model_list]
            vars = [model(x)[1] for model in model_list]
            if scaler is not None:
                means = [scaler.inverse_transform(m) for m in means]
                vars = [v * scaler.var_ for v in vars]
            means = torch.tensor(means)
            vars = torch.stack(vars)
            mean = means.mean(dim=0)
            var = (vars + means ** 2).mean(dim=0) - mean ** 2
            return mean, var


class MixEnsemble(nn.Module):
    def __init__(self, input_dim, hidden_dim, activate_fn, ensemble_size = 5):
        super().__init__()
        ensemble_layer = partial(EnsampleLayer, size=ensemble_size)
        self.model = nn.Sequential(
            ensemble_layer(nn.Linear(input_dim, hidden_dim)),
            activate_fn,
            ensemble_layer(nn.Linear(hidden_dim, 1)),
        )

    def forward(self, x):
        return self.model(x)

    def loss(self, x, y):
        return (self(x) - y).pow(2.0).mean()

    @staticmethod
    def ensample_predict(model_list, x, scaler=None):
        with torch.no_grad():
            ypred = [model_list[0](x) for _ in range(25)]
            if scaler is not None:
                ypred = [scaler.inverse_transform(yp) for yp in ypred]
            ypred = torch.tensor(ypred)
            return ypred.mean(dim=0), ypred.var(dim=0)


class MixEnsembleNLL(nn.Module):
    def __init__(self, input_dim, hidden_dim, activate_fn, ensemble_size = 5):
        super().__init__()
        ensemble_layer = partial(EnsampleLayer, size=ensemble_size)
        self.base = nn.Sequential(
            ensemble_layer(nn.Linear(input_dim, hidden_dim)),
            activate_fn,
        )
        self.head1 = ensemble_layer(nn.Linear(hidden_dim, 1))
        self.head2 = nn.Sequential(
            ensemble_layer(nn.Linear(hidden_dim, 1)), nn.Softplus()
        )

    def forward(self, x):
        base = self.base(x)
        mean, var = self.head1(base), self.head2(base) + 1e-6
        return [mean, var]

    def loss(self, x, y):
        mean, var = self(x)
        return (var.log() / 2 + (mean - y) ** 2 / (2 * var)).mean()

    @staticmethod
    def ensample_predict(model_list, x, scaler=None):
        with torch.no_grad():
            output = [model_list[0](x) for _ in range(25)]
            means = [out[0] for out in output]
            vars = [out[1] for out in output]
            if scaler is not None:
                means = [scaler.inverse_transform(m) for m in means]
                vars = [v * scaler.var_ for v in vars]
            means = torch.tensor(means)
            vars = torch.stack(vars)
            mean = means.mean(dim=0)
            var = (vars + means ** 2).mean(dim=0) - mean ** 2
            return mean, var
