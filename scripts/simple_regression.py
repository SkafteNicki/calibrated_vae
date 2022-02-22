import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as D
from torch import nn

from scr.regression_models import Ensamble, EnsambleNLL, MixEnsemble, MixEnsembleNLL
from scr.utils import ll, rmse

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

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
        for model_class in [Ensamble, EnsambleNLL]:
            models.append([])
            for _ in range(5):
                model = model_class(1, 100, activate_fn)
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
            model = model_class(1, 100, activate_fn)
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
            zip(
                [Ensamble, EnsambleNLL, MixEnsemble, MixEnsembleNLL],
                ["b", "g", "r", "m"],
            )
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
