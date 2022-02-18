from pickletools import optimize
import torch
from torch import nn
from torch import distributions as D
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import math
import time
from copy import deepcopy

SEED=42
torch.manual_seed(SEED)
np.random.seed(SEED)

class EnsampleLayer(nn.ModuleList):
    def __init__(self, submodule, size=5):
        super().__init__()
        for _ in range(size):
            self.append(deepcopy(submodule))

    def forward(self, *args, **kwargs):
        idx = np.random.randint(len(self))
        return self[idx](*args, **kwargs)

with open('boston.pkl', 'rb') as file:
    data, target = pickle.load(file)
target = target.reshape(-1, 1)


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(13, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
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
            nn.Linear(13, 50),
            nn.ReLU(),
        )
        self.head1 = nn.Linear(50, 1)
        self.head2 = nn.Sequential(
            nn.Linear(50, 1),
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
            EnsampleLayer(nn.Linear(13, 50)),
            nn.ReLU(),
            EnsampleLayer(nn.Linear(50, 1))
        )

    def forward(self, x):
        return self.model(x)

    def loss(self, x, y):
        return (self(x) - y).pow(2.0).mean()

    @classmethod
    def ensample_predict(self, model_list, x):
        with torch.no_grad():
            ypred = torch.stack([model_list[0](x) for _ in range(5)])
            return ypred.mean(dim=0), ypred.var(dim=0)


class MixEnsembleNLL(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            EnsampleLayer(nn.Linear(13, 50)),
            nn.ReLU(),
        )
        self.head1 = EnsampleLayer(nn.Linear(50, 1))
        self.head2 = nn.Sequential(
            nn.Linear(50, 1),
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
            output = [model_list[0](x) for _ in range(5)]
            means = torch.stack([out[0] for out in output])
            vars = torch.stack([out[1] for out in output])
            mean = means.mean(dim=0)
            var = (vars + means**2).mean(dim=0) - mean**2
            return mean, var

def rmse(x, mean):
    return (x-mean).abs().pow(2.0).mean().sqrt()

def log_likelihood(x, mean, var):
    d = D.Normal(mean, var.sqrt())
    return -d.log_prob(x).mean()

for model_class in [BaseModel, NNLModel, MixEnsemble, MixEnsembleNLL]:
    scores = {'rmse': [], 'nll': [], 'time': []}

    for rep in range(20):
        models = [ ]
        train_index, test_index = train_test_split(np.arange(len(data)), test_size=0.1, random_state=(rep+1)*SEED)
        train_data, train_target = data[train_index], target[train_index]
        test_data, test_target = data[test_index], target[test_index]
        
        data_scaler = StandardScaler()
        data_scaler.fit(train_data)
        train_data = data_scaler.transform(train_data)
        test_data = data_scaler.transform(test_data)

        target_scaler = StandardScaler()
        target_scaler.fit(train_target)
        train_target = target_scaler.transform(train_target)

        train = DataLoader(
            TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_target, dtype=torch.float32)), 
            batch_size=100
        )
        
        if 'mix' in model_class.__name__.lower():
            reps = 1
            n_epochs = 500
        else:
            reps = 5
            n_epochs = 40

        start = time.time()
        for _ in range(reps):
            model = model_class()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

            for epoch in range(n_epochs):
                for batch in train:
                    optimizer.zero_grad()
                    loss = model.loss(*batch)
                    loss.backward()
                    optimizer.step()

            models.append(model)
        end = time.time()

        x, y = torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_target, dtype=torch.float32)
        mean, var = model_class.ensample_predict(models, x)
        mean = mean + math.sqrt(target_scaler.var_) + target_scaler.mean_
        var = var * target_scaler.var_

        scores['rmse'].append(rmse(y, mean).item())
        scores['nll'].append(log_likelihood(y, mean, var).item())
        scores['time'].append(end - start)
    
    print(
        f"Model {model_class.__name__}. \n"
        f"RMSE: {np.mean(scores['rmse'])}+-{np.std(scores['rmse'])} \n"
        f"NLL : {np.mean(scores['nll'])}+-{np.std(scores['nll'])} \n"
        f"Time: {np.mean(scores['time'])}+-{np.std(scores['time'])} \n"
    )