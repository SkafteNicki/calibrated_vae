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


train1, val1, test1, _ = get_dataset('protein_split1')
train2, val2, test2, _ = get_dataset('protein_split2')

model = MixVAE()
model.load_state_dict(torch.load("models/generate_models/MixVAE_False.pt"))

trainer = pl.Trainer(logger=False)
predictions = trainer.predict(model, dataloaders=torch.utils.data.DataLoader(train1, batch_size=64))
embeddings = torch.cat([p['encoder_mean'] for p in predictions], dim=0)
labels = torch.cat([p['label'] for p in predictions], dim=0)

important_organisms = {
    "Acidobacteria": 0,
    "Actinobacteria": 1,
    "Bacteroidetes": 2,
    "Chloroflexi": 3,
    "Cyanobacteria": 4
}

fig = plt.figure()
def plot_embeddings(ax=None, alpha=1.0):
    if ax is None:
        ax = plt
    for name in important_organisms:
        current_label = important_organisms[name]
        ax.plot(
            embeddings[labels == current_label, 0],
            embeddings[labels == current_label, 1],
            '.',
            alpha=alpha,
        )
plot_embeddings()