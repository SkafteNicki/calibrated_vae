import torch
from torch import distributions as D
from tqdm import tqdm
from models import get_model
from data import get_data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    vae_class = get_model("vae")
    vae = vae_class.load_from_checkpoint("wandb/run-20211101_141118-109z7s1w/files/checkpoints/best_model.ckpt")
    vae.eval()
    vae.to(device)
    
    mevae_class = get_model("mevae")
    mevae = mevae_class.load_from_checkpoint("wandb/run-20211102_151231-2htnoygr/files/checkpoints/best_model.ckpt")
    mevae.to(device)

    mnist01 = get_data("mnist01")()
    mnist01.setup()
    mnist23 = get_data("mnist23")()
    mnist23.setup()

    def agreement(model, dataloader):
        with torch.no_grad():
            entropys, log_probs = [ ], [ ]
            for batch in tqdm(dataloader):
                entropy, log_prob = [], []
                for _ in range(100):
                    x, y = batch
                    z_mu, z_std, x_hat, kl = model.encode_decode(x.to(device))
                    dist = D.Independent(D.Bernoulli(probs=x_hat), 3)
                    entropy.append(dist.entropy())
                    log_prob.append(dist.log_prob(x.to(device)))
                entropy = torch.stack(entropy, dim=0)
                log_prob = torch.stack(log_prob, dim=0)
                entropys.append(entropy.std(dim=0).cpu())
                log_probs.append(log_prob.std(dim=0).cpu())
            return torch.cat(entropys, dim=0), torch.cat(log_probs, dim=0)

    entropy1, log_probs1 = agreement(mevae, mnist01.test_dataloader())
    entropy2, log_probs2 = agreement(mevae, mnist23.test_dataloader())
    n1, n2 = len(entropy1), len(entropy2)
    n = n1 + n2

    dataframe = pd.DataFrame()
    dataframe['entropy'] = torch.cat([entropy1, entropy2]).numpy()
    dataframe['log_probs'] = torch.cat([log_probs1, log_probs2]).numpy()
    dataframe['dataset'] = n1 * ['mnist01'] + n2 * ['mnist23']
    dataframe['dataset_class'] = n1 * [0] + n2 * [1]

    plt.figure()
    sns.histplot(data=dataframe, x='entropy', hue='dataset')

    plt.figure()
    sns.histplot(data=dataframe, x='log_probs', hue='dataset')
    
    plt.show()

    acc1, acc2 = [ ], [ ]
    thredshols = np.linspace(0, 10, 1000)
    for t in thredshols:
        classifier = dataframe['entropy'] > t
        acc1.append(
            (classifier.to_numpy().astype('int') == dataframe['dataset_class'].to_numpy()).mean()
        )
        classifier = dataframe['log_probs'] > t
        acc2.append(
            (classifier.to_numpy().astype('int') == dataframe['dataset_class'].to_numpy()).mean()
        )
    acc1, acc2 = np.array(acc1), np.array(acc2)
    print(f"best threshold for entropy {thredshols[np.argmax(acc1)]} with acc {np.max(acc1)}")
    print(f"best threshold for entropy {thredshols[np.argmax(acc2)]} with acc {np.max(acc2)}")
