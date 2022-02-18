import torch
from torch import nn
import torch.distributions as D
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np


N = 20
x = (8*torch.rand(N)-4).reshape(-1,1)
y = x**3 + torch.randn(N,1)*3

_x = torch.linspace(-6,6,100).reshape(-1,1)
_y = _x**3

assert x.shape == y.shape
assert _x.shape == _y.shape

class EnsampleLayer(nn.ModuleList):
    def __init__(self, submodule, size=5):
        super().__init__()
        for _ in range(size):
            self.append(deepcopy(submodule))

    def forward(self, *args, **kwargs):
        idx = np.random.randint(len(self))
        return self[idx](*args, **kwargs)


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

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
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU(),
        )
        self.head1 = nn.Linear(100, 1)
        self.head2 = nn.Sequential(
            nn.Linear(100, 1),
            nn.Softplus()
        )

    def forward(self, x):
        base = self.base(x)
        mean, var = self.head1(base), self.head2(base)+1e-6
        return [mean, var]

    def loss(self, x, y):
        mean, var = self(x)
        return (var.log() / 2 + (mean - y)**2 / (2*var)).mean()

    @classmethod
    def ensample_predict(self, model_list, x):
        with torch.no_grad():
            means = torch.stack([model(x)[0] for model in model_list])
            vars = torch.stack([model(x)[1] for model in model_list])
            mean = means.mean(dim=0)
            var = (vars + means**2).mean(dim=0) - mean**2
            return mean, var

class MixEnsemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            EnsampleLayer(nn.Linear(1, 100)),
            nn.ReLU(),
            EnsampleLayer(nn.Linear(100, 1))
        )

    def forward(self, x):
        return self.model(x)

    def loss(self, x, y):
        return (self(x) - y).pow(2.0).mean()

    @classmethod
    def ensample_predict(self, model_list, x):
        with torch.no_grad():
            ypred = torch.stack([model_list[0](x) for _ in range(25)])
            return ypred.mean(dim=0), ypred.var(dim=0)


class MixEnsembleNLL(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            EnsampleLayer(nn.Linear(1, 100)),
            nn.ReLU(),
        )
        self.head1 = EnsampleLayer(nn.Linear(100, 1))
        self.head2 = nn.Sequential(
            EnsampleLayer(nn.Linear(100, 1)),
            nn.Softplus()
        )

    def forward(self, x):
        base = self.base(x)
        mean, var = self.head1(base), self.head2(base)+1e-6
        return [mean, var]

    def loss(self, x, y):
        mean, var = self(x)
        return (var.log() / 2 + (mean - y)**2 / (2*var)).mean()

    @classmethod
    def ensample_predict(self, model_list, x):
        with torch.no_grad():
            output = [model_list[0](x) for _ in range(25)]
            means = torch.stack([out[0] for out in output])
            vars = torch.stack([out[1] for out in output])
            mean = means.mean(dim=0)
            var = (vars + means**2).mean(dim=0) - mean**2
            return mean, var


if __name__ == "__main__":

    models = [ ]
    for model_class in [BaseModel, NNLModel]:
        models.append([ ])
        for _ in range(5):
            model = model_class()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            for i in range(40):
                optimizer.zero_grad()
                loss = model.loss(x, y)
                loss.backward()
                optimizer.step()
                if i % 39 == 0:
                    print(f"Epoch {i}, Loss {loss.item()}")

            models[-1].append(model)

    for model_class in [MixEnsemble, MixEnsembleNLL]:
        model = model_class()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        for i in range(1000):
            optimizer.zero_grad()
            loss = model.loss(x, y)
            loss.backward()
            optimizer.step()
            if i % 39 == 0:
                print(f"Epoch {i}, Loss {loss.item()}")
        models.append([])
        models[-1].append(model)

    fig, ax = plt.subplots(nrows=1, ncols=4)

    for i, (model_class, color) in enumerate(zip(
        [BaseModel, NNLModel, MixEnsemble, MixEnsembleNLL], ['b', 'g', 'r', 'm'])
    ):
        ax[i].plot(x,y,'.c')
        ax[i].plot(_x, _y, '-k')

        mean, var = model_class.ensample_predict(models[i], _x)
        std3 = 3*var.sqrt()
        ax[i].plot(_x, mean, f'--{color}', label=model_class.__name__)

        ax[i].fill_between(
            _x.flatten(), (mean-std3).flatten(), (mean+std3).flatten(), 
            alpha=0.1, color=color
        )
        ax[i].axis([-6,6,-200,200])
    plt.show()
    






