import argparse

import matplotlib.pyplot as plt
import torch
from torch import distributions as D
import tqdm

from scr.data import get_dataset
from scr.generative_models import get_model_from_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weight_file")
    args = parser.parse_args()

    print("Running script with arguments:")
    print(args)

    model, model_class = get_model_from_file(args.weight_file)
    dataset_name = args.weight_file.split("_")[-1][:-3]
    dataset = torch.utils.data.DataLoader(get_dataset(dataset_name)[0], batch_size=64)

    latent_grid = torch.meshgrid(
        [torch.linspace(-4, 4, 100) for _ in range(2)]
    )
    latent_grid = torch.stack(latent_grid, dim=2).reshape(-1, 2)
    grid_dataset = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(latent_grid),
        batch_size=64
    )

    with torch.inference_mode():
        embeddings, labels = [ ], [ ]
        for batch in tqdm.tqdm(dataset):
            x, y = batch
            e = [ ]
            for i in range(10):
                z_mu, _ = model.encode(x)
                e.append(z_mu)
                
            embeddings.append(torch.stack(e, dim=0).mean(dim=0))
            labels.append(y)
        
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)

        e_mean, e_std = [ ], [ ]
        for batch in tqdm.tqdm(grid_dataset):
            z, = batch
            r = [ ]
            for i in range(10):
                r.append(model(z))
            dist = D.Independent(D.Bernoulli(probs=torch.stack(r, dim=0)), 3)
            entropy = dist.entropy()

            e_mean.append(entropy.mean(dim=0))
            e_std.append(entropy.std(dim=0))


        e_mean = torch.cat(e_mean, dim=0).reshape(100, 100)
        e_std = torch.cat(e_std, dim=0).reshape(100, 100)


    fig = plt.figure()
    for label in torch.unique(y):
        plt.plot(
            embeddings[label == labels, 0],
            embeddings[label == labels, 1],
            '.'
        )

    plt.contourf(
        latent_grid[:,0].reshape(100, 100),
        latent_grid[:,1].reshape(100, 100),
        e_std,
    )
    
    plt.colorbar()