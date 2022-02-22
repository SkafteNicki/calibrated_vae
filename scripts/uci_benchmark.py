import math
import pickle
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import distributions as D
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scr.regression_models import Ensamble, EnsambleNLL, MixEnsemble, MixEnsembleNLL
from scr.utils import ll, rmse

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


with open("data/boston.pkl", "rb") as file:
    data, target = pickle.load(file)
target = target.reshape(-1, 1)


for model_class in [Ensamble, EnsambleNLL, MixEnsemble, MixEnsembleNLL]:
    scores = {"rmse": [], "nll": [], "time": []}

    for rep in range(20):
        models = []
        train_index, test_index = train_test_split(
            np.arange(len(data)), test_size=0.1, random_state=(rep + 1) * SEED
        )
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
            TensorDataset(
                torch.tensor(train_data, dtype=torch.float32),
                torch.tensor(train_target, dtype=torch.float32),
            ),
            batch_size=100,
        )

        if "mix" in model_class.__name__.lower():
            reps = 1
            n_epochs = 500
        else:
            reps = 5
            n_epochs = 40

        start = time.time()
        for _ in range(reps):
            model = model_class(13, 100, nn.ReLU())
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

            for epoch in range(n_epochs):
                for batch in train:
                    optimizer.zero_grad()
                    loss = model.loss(*batch)
                    loss.backward()
                    optimizer.step()

            models.append(model)
        end = time.time()

        x, y = torch.tensor(test_data, dtype=torch.float32), torch.tensor(
            test_target, dtype=torch.float32
        )
        mean, var = model_class.ensample_predict(models, x, scaler=target_scaler)

        scores["rmse"].append(rmse(y, mean))
        scores["nll"].append(ll(y, mean, var))
        scores["time"].append(end - start)

    print(
        f"Model {model_class.__name__}. \n"
        f"RMSE: {np.mean(scores['rmse'])}+-{np.std(scores['rmse'])} \n"
        f"NLL : {np.mean(scores['nll'])}+-{np.std(scores['nll'])} \n"
        f"Time: {np.mean(scores['time'])}+-{np.std(scores['time'])} \n"
    )
