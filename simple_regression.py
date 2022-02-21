import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as D
from torch import nn


def rmse(x, mean):
    return (x - mean).abs().pow(2.0).mean().sqrt().item()


def ll(x, mean, var):
    d = D.Normal(mean, var.sqrt())
    return -d.log_prob(x).mean().item()


class EnsampleLayer(nn.ModuleList):
    def __init__(self, submodule, size=5):
        super().__init__()
        for _ in range(size):
            self.append(deepcopy(submodule))

    def forward(self, *args, **kwargs):
        idx = np.random.randint(len(self))
        return self[idx](*args, **kwargs)


class BaseModel(nn.Module):
    def __init__(self, activate_fn):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(1, 100), activate_fn, nn.Linear(100, 1))

    def forward(self, x):
        return self.model(x)

    def loss(self, x, y):
        return (self(x) - y).pow(2.0).mean()

    @classmethod
    def ensample_predict(self, model_list, x):
        with torch.no_grad():
            ypred = torch.stack([model(x) for model in model_list])
            return ypred.mean(dim=0), ypred.var(dim=0)


class NNLModel(nn.Module):
    def __init__(self, activate_fn):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(1, 100),
            activate_fn,
        )
        self.head1 = nn.Linear(100, 1)
        self.head2 = nn.Sequential(nn.Linear(100, 1), nn.Softplus())

    def forward(self, x):
        base = self.base(x)
        mean, var = self.head1(base), self.head2(base) + 1e-6
        return [mean, var]

    def loss(self, x, y):
        mean, var = self(x)
        return (var.log() / 2 + (mean - y) ** 2 / (2 * var)).mean()

    @classmethod
    def ensample_predict(self, model_list, x):
        with torch.no_grad():
            means = torch.stack([model(x)[0] for model in model_list])
            vars = torch.stack([model(x)[1] for model in model_list])
            mean = means.mean(dim=0)
            var = (vars + means ** 2).mean(dim=0) - mean ** 2
            return mean, var


class MixEnsemble(nn.Module):
    def __init__(self, activate_fn):
        super().__init__()
        self.model = nn.Sequential(
            EnsampleLayer(nn.Linear(1, 100)), activate_fn, EnsampleLayer(nn.Linear(100, 1))
        )

    def forward(self, x):
        return self.model(x)

    def loss(self, x, y):
        return (self(x) - y).pow(2.0).mean()

    @classmethod
    def ensample_predict(self, model_list, x):
        with torch.no_grad():
            ypred = torch.stack([model_list[0](x) for _ in range(10)])
            return ypred.mean(dim=0), ypred.var(dim=0)


class MixEnsembleNLL(nn.Module):
    def __init__(self, activate_fn):
        super().__init__()
        self.base = nn.Sequential(
            EnsampleLayer(nn.Linear(1, 100)),
            activate_fn,
        )
        self.head1 = EnsampleLayer(nn.Linear(100, 1))
        self.head2 = nn.Sequential(EnsampleLayer(nn.Linear(100, 1)), nn.Softplus())

    def forward(self, x):
        base = self.base(x)
        mean, var = self.head1(base), self.head2(base) + 1e-6
        return [mean, var]

    def loss(self, x, y):
        mean, var = self(x)
        return (var.log() / 2 + (mean - y) ** 2 / (2 * var)).mean()

    @classmethod
    def ensample_predict(self, model_list, x):
        with torch.no_grad():
            output = [model_list[0](x) for _ in range(10)]
            means = torch.stack([out[0] for out in output])
            vars = torch.stack([out[1] for out in output])
            mean = means.mean(dim=0)
            var = (vars + means ** 2).mean(dim=0) - mean ** 2
            return mean, var


if __name__ == "__main__":
    for problem in ["homoscedasticity", "heteroscedastic"]:
        if problem == "homoscedasticity":
            f = lambda x: x ** 3
            N = 20
            axis = (-6, 6, -200, 200)
            activate_fn = nn.ReLU()
            epochs = 200

            x = (8 * torch.rand(N) - 4).reshape(-1, 1)
            y = f(x) + torch.randn(N, 1) * 3

            _x = torch.linspace(*axis[:2], 100).reshape(-1, 1)
            _y = f(_x)

            assert x.shape == y.shape
            assert _x.shape == _y.shape
        elif problem == "heteroscedastic":
            f = lambda x: x * torch.sin(x)
            N = 1000
            axis = (-6, 16, -20, 20)
            activate_fn = nn.Sigmoid()
            epochs = 2000

            x = (10 * torch.randn(N)).reshape(-1, 1)
            y = f(x) + 0.3 * torch.randn(N, 1) + 0.3 * x * torch.randn(N, 1)

            _x = torch.linspace(*axis[:2], 100).reshape(-1, 1)
            _y = f(_x)

        assert x.shape == y.shape
        assert _x.shape == _y.shape

        models = []
        for model_class in [BaseModel, NNLModel]:
            models.append([])
            for _ in range(5):
                model = model_class(activate_fn)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
                for i in range(epochs):
                    optimizer.zero_grad()
                    loss = model.loss(x, y)
                    loss.backward()
                    optimizer.step()
                    if i % 39 == 0:
                        print(f"Epoch {i}, Loss {loss.item()}")

                models[-1].append(model)

        for model_class in [MixEnsemble, MixEnsembleNLL]:
            model = model_class(activate_fn)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            for i in range(5 * epochs):
                optimizer.zero_grad()
                loss = model.loss(x, y)
                loss.backward()
                optimizer.step()
                if i % 39 == 0:
                    print(f"Epoch {i}, Loss {loss.item()}")
            models.append([])
            models[-1].append(model)

        fig, ax = plt.subplots(nrows=1, ncols=4)

        for i, (model_class, color) in enumerate(
            zip([BaseModel, NNLModel, MixEnsemble, MixEnsembleNLL], ["b", "g", "r", "m"])
        ):
            ax[i].plot(x, y, ".c")
            ax[i].plot(_x, _y, "-k")

            mean, var = model_class.ensample_predict(models[i], _x)
            std3 = 3 * var.sqrt()
            ax[i].plot(_x, mean, f"--{color}", label=model_class.__name__)
            ax[i].fill_between(
                _x.flatten(),
                (mean - std3).flatten(),
                (mean + std3).flatten(),
                alpha=0.1,
                color=color,
            )

            mean, var = model_class.ensample_predict(models[i], x)
            ax[i].title.set_text(
                f"{model_class.__name__}\n"
                f"RMSE={rmse(y, mean):.2f}\n"
                f"LL={ll(y, mean, var):.2f}"
            )

            ax[i].axis(axis)

        os.makedirs("figures/simple_regression/", exist_ok=True)
        fig.savefig(f"figures/simple_regression/{problem}.png", bbox_inches="tight")
        fig.savefig(f"figures/simple_regression/{problem}.svg", bbox_inches="tight")

    plt.show()
