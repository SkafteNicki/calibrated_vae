import torch
from torch import distributions as D
from tqdm import tqdm
from models import get_model
from data import get_data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro, probplot, normaltest

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":   
    mevae_class = get_model("mevae")
    #mevae = mevae_class.load_from_checkpoint("wandb/run-20211130_113209-14h3b46e/files/checkpoints/best_model.ckpt")
    mevae = mevae_class.load_from_checkpoint("wandb/run-20211201_154735-23ukp74a/files/checkpoints/best_model.ckpt")
    
    with torch.no_grad():
        weights = dict()
        for name in ['encoder', 'encoder_mu', 'encoder_std', 'decoder']:
            subnet = getattr(mevae, name)
            keys = dict(subnet[0].named_parameters()).keys()
            for k in keys:
                weights[f'{name}.{k}'] = []
                for i in range(len(subnet)):
                    subnet_i = subnet[i]
                    subnet_i_weights = dict(subnet_i.named_parameters())
                    weights[f'{name}.{k}'].append(subnet_i_weights[k])
                weights[f'{name}.{k}'] = torch.stack(weights[f'{name}.{k}'], dim=-1)
                weights[f'{name}.{k}'] = weights[f'{name}.{k}'].reshape(-1, len(subnet))

    stattest = []
    not_passing = []
    for values in weights.values():
        for v in tqdm(values):
            stat1, p1 = shapiro(v)
            stat2, p2 = normaltest(v)
            normal = p1 > 0.05 and p2 > 0.05
            stattest.append(1 if normal else 0)
            if not normal:
                not_passing.append(v)

    idx = np.random.choice(a=len(not_passing), size=5, replace=False)
    for i in idx:
        sns.distplot(not_passing[i])
        #probplot(not_passing[i], dist='norm', plot=plt)

    print(f'{sum(stattest)/len(stattest)}')


    eval_size = 100
    mevae = mevae.to(device)
    z = torch.stack(torch.meshgrid([
        torch.linspace(-5, 5, eval_size),
        torch.linspace(-5, 5, eval_size)
    ])).reshape(2, -1).T

    batch_size = 16
    entropy = [ ]
    for n in tqdm(range(len(z) // batch_size)):
        with torch.no_grad():
            decoded_z = mevae(z[n*batch_size:(n+1)*batch_size].to(device), use_all=True)
            dist = D.Independent(D.Bernoulli(probs=decoded_z), 3)
            entropy.append(dist.entropy())
    entropy = torch.cat(entropy, dim=1).cpu() # [ensemble size, grid size**2]
    
    entropy_mean = entropy.mean(dim=0)
    entropy_std = entropy.std(dim=0)

    
    plt.contourf(
        z[:, 0].reshape(eval_size, eval_size),
        z[:, 1].reshape(eval_size, eval_size),
        entropy_std.reshape(eval_size, eval_size),
        levels=50,
        zorder=0,
    )






