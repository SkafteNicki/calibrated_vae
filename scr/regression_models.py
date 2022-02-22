import torch
from torch import nn

from scr.layers import EnsampleLayer


class Ensamble(nn.Module):
    def __init__(self, input_dim, hidden_dim, activate_fn):
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
            ypred = torch.stack([model(x) for model in model_list])
            ypred = ypred if scaler is None else scaler.inverse_transform(ypred)
            return ypred.mean(dim=0), ypred.var(dim=0)


class EnsambleNLL(nn.Module):
    def __init__(self, input_dim, hidden_dim, activate_fn):
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
            means = torch.stack([model(x)[0] for model in model_list])
            vars = torch.stack([model(x)[1] for model in model_list])
            means = means if scaler is None else scaler.inverse_transform(means)
            vars = vars if scaler is None else vars * scaler.vars_
            mean = means.mean(dim=0)
            var = (vars + means ** 2).mean(dim=0) - mean ** 2
            return mean, var


class MixEnsemble(nn.Module):
    def __init__(self, input_dim, hidden_dim, activate_fn):
        super().__init__()
        self.model = nn.Sequential(
            EnsampleLayer(nn.Linear(input_dim, hidden_dim)),
            activate_fn,
            EnsampleLayer(nn.Linear(hidden_dim, 1)),
        )

    def forward(self, x):
        return self.model(x)

    def loss(self, x, y):
        return (self(x) - y).pow(2.0).mean()

    @staticmethod
    def ensample_predict(model_list, x, scaler=None):
        with torch.no_grad():
            ypred = torch.stack([model_list[0](x) for _ in range(10)])
            ypred = ypred if scaler is None else scaler.inverse_transform(ypred)
            return ypred.mean(dim=0), ypred.var(dim=0)


class MixEnsembleNLL(nn.Module):
    def __init__(self, input_dim, hidden_dim, activate_fn):
        super().__init__()
        self.base = nn.Sequential(
            EnsampleLayer(nn.Linear(input_dim, hidden_dim)),
            activate_fn,
        )
        self.head1 = EnsampleLayer(nn.Linear(hidden_dim, 1))
        self.head2 = nn.Sequential(
            EnsampleLayer(nn.Linear(hidden_dim, 1)), nn.Softplus()
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
            output = [model_list[0](x) for _ in range(10)]
            means = torch.stack([out[0] for out in output])
            vars = torch.stack([out[1] for out in output])
            means = means if scaler is None else scaler.inverse_transform(means)
            vars = vars if scaler is None else vars * scaler.vars_
            mean = means.mean(dim=0)
            var = (vars + means ** 2).mean(dim=0) - mean ** 2
            return mean, var
