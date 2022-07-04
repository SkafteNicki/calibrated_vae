import argparse
import os
import time
import traceback

import matplotlib.pyplot as plt
import torch
from pytorch_lightning.utilities.seed import seed_everything

from scr.data import get_dataset
from scr.generative_models import get_model
from scr.notify import post_message

if __name__ == "__main__":
    seed_everything(42)

    n_ensemble = 5
    train, val, _, _ = get_dataset("mnist")

    def process_data(dataset):
        data, labels = dataset.dataset.tensors
        data = data[labels==3]
        labels = labels[labels==3]
        data = data.round()
        return torch.utils.data.TensorDataset(data, labels)

    train = process_data(train)
    val = process_data(val)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=64)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=64)

    model_class = get_model("vae")
    model = model_class.fit(n_ensemble, 1, train_dataloader, val_dataloader)
    model.eval()

    classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 100),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.1),
        torch.nn.Linear(100, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 1)
    )

    train_img = train.tensors[0]
    reg_score = train_img.sum(dim=[1,2,3])
    classifier_data = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_img, reg_score),
        batch_size=64
    )
    data_loader = iter(classifier_data)

    opt = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    count = 0
    while count < 10000:
        opt.zero_grad()

        try:
            batch = next(data_loader) 
        except StopIteration:
            data_iter = iter(classifier_data)
            batch = next(data_iter) 

        img, score = batch
        pred = classifier(img).squeeze()
        loss = (pred - score).pow(2.0).sum()
        if count % 100 == 0:
            print(loss)
        loss.backward()
        opt.step()
        count += 1

    classifier.eval()

    latent_coords = torch.nn.Parameter(torch.randn(1, 2))
    opt = torch.optim.SGD([latent_coords], lr=1e-2)

    count = 0
    while count < 10000:
        opt.zero_grad()
        recon = model.decoder(latent_coords)
        pred = classifier(recon)
        loss = -pred.sum()
        loss.backward()
        if count % 100 == 0:
            print(loss, latent_coords)
        opt.step()
        count += 1

    from bayes_opt import BayesianOptimization

    def g(x, y):
        return model.decoder(torch.tensor([[x, y]], dtype=torch.float)).detach()

    def f(x, y):
        recon = g(x, y)
        pred = classifier(recon)
        return pred.item()

    pbounds = {'x': (-5, 5), 'y': (-5, 5)}

    optimizer = BayesianOptimization(
        f=f,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.maximize()


    grid = torch.stack(
        torch.meshgrid(
            [torch.linspace(-10, 10, 50) for _ in range(2)], indexing='ij'
        )
    ).reshape(2, -1).T
    recons = torch.stack([g(z[0], z[1]) for z in grid], 0).squeeze().reshape(50, 50, 28, 28)
    bigimg = torch.zeros(28*50, 28*50)
    for i in range(50):
        for j in range(50):
            bigimg[28*i:28*(i+1), 28*j:28*(j+1)] = recons[i, j]




    #             train_end = time.time()

    #             os.makedirs("models/generative_models/", exist_ok=True)
    #             model_class.save_checkpoint(
    #                 model, f"models/generative_models/{model_name}_{dataset_name}.pt"
    #             )
    #         except Exception as e:
    #             print(f"Exception happened:")
    #             traceback.print_exc()
    #             post_message(
    #                 "the following exception happended: \n" f"{traceback.format_exc()}"
    #             )