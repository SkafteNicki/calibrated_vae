import enum
from protein_vae import TOKEN_SIZE, MixVAE
import pytorch_lightning as pl
import torch
from scr.data import get_dataset, important_organisms
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse
import torch.distributions as D
from torch import nn
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score


train1, val1, test1, _ = get_dataset('protein_split1')
train2, val2, test2, _ = get_dataset('protein_split2')

model = MixVAE()
model.load_state_dict(torch.load("models/generative_models/MixVAE_True.pt"))

trainer = pl.Trainer(logger=False)
predictions = trainer.predict(model, dataloaders=torch.utils.data.DataLoader(train1, batch_size=64))
embeddings1 = torch.cat([p['encoder_mean'] for p in predictions], dim=0)
labels1 = torch.cat([p['label'] for p in predictions], dim=0)

predictions = trainer.predict(model, dataloaders=torch.utils.data.DataLoader(train2, batch_size=64))
embeddings2 = torch.cat([p['encoder_mean'] for p in predictions], dim=0)
labels2 = torch.cat([p['label'] for p in predictions], dim=0)

in_organisms = {
    "Acidobacteria": 0,
    "Actinobacteria": 1,
    "Bacteroidetes": 2,
    "Chloroflexi": 3,
    "Cyanobacteria": 4
}

out_organisms = {
    "Deinococcus-Thermus": 5,
    "Firmicutes": 6,
    "Fusobacteria": 7,
    "Proteobacteria": 8,
}


fig = plt.figure()
def plot_embeddings(ax=None, alpha=1.0, plot_in=True, plot_out=False):
    if ax is None:
        ax = plt
    if plot_in:
        for name in in_organisms:
            current_label = in_organisms[name]
            ax.plot(
                embeddings1[labels1 == current_label, 0],
                embeddings1[labels1 == current_label, 1],
                '.',
                alpha=alpha,
            )
    if plot_out:
        for name in out_organisms:
            current_label = out_organisms[name]
            ax.plot(
                embeddings2[labels2 == current_label, 0],
                embeddings2[labels2 == current_label, 1],
                '.',
                alpha=alpha,
            )

plot_embeddings(plot_in=True, plot_out=False, alpha=0.3)
plot_embeddings(plot_in=False, plot_out=True)

def log_prob(model, data):
    with torch.inference_mode():
        loader = torch.utils.data.DataLoader(data, batch_size=64)
        score = [ ]
        for batch in loader:
            x = batch[0].long()
            recon, _ = model(x, idx=0)
            dist = D.Independent(D.Categorical(recon.permute(0, 2, 1).softmax(dim=-1)), 1)
            score.append(dist.log_prob(x))
        return torch.cat(score)

log_probs_in = log_prob(model, train1)
log_probs_out = log_prob(model, train2)

auroc = roc_auc_score(
    torch.cat([torch.ones_like(log_probs_in), torch.zeros_like(log_probs_out)]),
    torch.cat([log_probs_in, log_probs_out]),
)

plt.hist(log_probs_in.numpy(), label='in', bins=100, alpha=0.5)
plt.hist(log_probs_out.numpy(), label='out', bins=100, alpha=0.5)
plt.legend()


def disagreement(model, data):
    with torch.inference_mode():
        loader = torch.utils.data.DataLoader(data, batch_size=64)
        score = [ ]
        for batch in loader:
            x = batch[0].long()
            recons = [ ]
            for i in range(5):
                z_mu, _ = model.encode(x, idx=i)
                for j in range(5):
                    recon = model.decode(z_mu, idx=j).argmax(dim=1)
                    recons.append(recon)
            recons = torch.stack(recons)
            disagreement = [(recons[i] != recons[j]).float().sum(1) for i in range(25) for j in range(25)]
            score.append(torch.stack(disagreement, 0).mean(0))
        return torch.cat(score)

disagreement_in = disagreement(model, train1)
disagreement_out = disagreement(model, train2)

auroc = roc_auc_score(
    torch.cat([torch.ones_like(disagreement_out), torch.zeros_like(disagreement_in)]),
    torch.cat([disagreement_out, disagreement_in]),
)

plt.hist(disagreement_in.numpy(), label='in', bins=100, alpha=0.5)
plt.hist(disagreement_out.numpy(), label='out', bins=100, alpha=0.5)
plt.legend()
