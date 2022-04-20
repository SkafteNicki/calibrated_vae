from torch import nn, distributions as D
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
from random import randint
from scr.layers import EnsembleLayer
from scr.data import get_dataset, important_organisms, aa1_to_index
import os

SEQ_LEN = 2592
TOKEN_SIZE = len(aa1_to_index)

class VAE(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(SEQ_LEN*TOKEN_SIZE, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU()
        )

        self.encoder_mu = nn.Linear(100, 2)
        self.encoder_scale = nn.Sequential(nn.Linear(100, 2), nn.Softplus())

        self.decoder = nn.Sequential(
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.Linear(100, 500),
            nn.ReLU(),
            nn.Linear(500, SEQ_LEN*TOKEN_SIZE)
        )

        self.loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=22)
        self._prior = None
        self._step_counter = 0

    @property
    def beta(self):
        if self.training:
            self._step_counter += 1
            return min(1.0, float(self._step_counter / 20000))
        return 1.0

    @property
    def prior(self):
        if self._prior is None:
            self._prior =  D.Independent(D.Normal(torch.zeros(2).to(self.device),
                                                  torch.ones(2).to(self.device)), 1)
        return self._prior

    def encode(self, x):
        x = nn.functional.one_hot(x, TOKEN_SIZE)
        h = self.encoder(x.float().reshape(x.shape[0], -1))
        return self.encoder_mu(h), self.encoder_scale(h)

    def decode(self, z):
        recon = self.decoder(z).reshape(*z.shape[:-1], TOKEN_SIZE, -1)
        return recon

    def forward(self, x):
        encoder_mu, encoder_std = self.encode(x)
        q_dist = D.Independent(D.Normal(encoder_mu, encoder_std + 1e-4), 1)
        z = q_dist.rsample()
        recon = self.decode(z)
        return recon, q_dist

    def _step(self, batch, beta=1.0):
        x = batch[0].long()
        recon, q_dist = self(x)
        recon_loss = self.loss_fn(recon, x).sum(dim=-1).mean()
        kl_loss = D.kl_divergence(q_dist, self.prior).mean()
        loss = recon_loss + beta * kl_loss
        acc = (recon.argmax(dim=1) == x)[x!=22].float().mean()
        return loss, recon_loss, kl_loss, acc

    def training_step(self, batch, batch_idx):
        loss, recon_loss, kl_loss, acc = self._step(batch, beta=self.beta)

        self.log_dict({'train_loss': loss,
                       'train_recon': recon_loss,
                       'train_kl': kl_loss,
                       'train_acc': acc},
                      prog_bar=True,
                      logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon_loss, kl_loss, acc = self._step(batch)
        self.log_dict(
            {
                'val_loss': loss,
                'val_recon': recon_loss,
                'val_kl': kl_loss,
                'val_acc': acc
            },
            logger=True
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y =  batch
        encoder_mu, encoder_std = self.encode(x)
        return {
            'encoder_mean': encoder_mu,
            'encoder_std': encoder_std,
            'label': y
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


class MixVAE(VAE):
    def __init__(self, ensemble_size=5):
        super().__init__()

        self.encoder = EnsembleLayer(
            nn.Sequential(
                nn.Linear(SEQ_LEN*TOKEN_SIZE, 500),
                nn.ReLU(),
                nn.Linear(500, 100),
                nn.ReLU()
            ),
            ensemble_size
        )

        self.encoder_mu = EnsembleLayer(nn.Linear(100, 2), ensemble_size)
        self.encoder_scale = EnsembleLayer(nn.Sequential(nn.Linear(100, 2), nn.Softplus()), ensemble_size)

        self.decoder = EnsembleLayer(
            nn.Sequential(
                nn.Linear(2, 100),
                nn.ReLU(),
                nn.Linear(100, 500),
                nn.ReLU(),
                nn.Linear(500, SEQ_LEN*TOKEN_SIZE)
            ),
            ensemble_size
        )
        self.ensemble_size = ensemble_size
        self.loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=22)
        self._prior = None


    def encode(self, x):
        idx = randint(0, self.ensemble_size-1)
        x = nn.functional.one_hot(x, TOKEN_SIZE)
        h = self.encoder[idx](x.float().reshape(x.shape[0], -1))
        return self.encoder_mu[idx](h), self.encoder_scale[idx](h)


if __name__ == "__main__":
    train, val, test = get_dataset('protein')

    train_dl = torch.utils.data.DataLoader(train, batch_size=32)
    val_dl = torch.utils.data.DataLoader(val, batch_size=32)

    model = MixVAE()

    trainer = pl.Trainer(
        logger=pl.loggers.WandbLogger(),
        accelerator="auto",
        num_sanity_val_steps=0,
        devices=1,
        max_epochs=100
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    
    os.makedirs("models/generate_models/", exist_ok=True)
    torch.save(model.state_dict(), f"models/generate_models/{model.__class__.__name__}.pt",)
    
    predictions = trainer.predict(model, dataloaders=train_dl)
    embeddings = torch.cat([p['encoder_mean'] for p in predictions], dim=0)
    labels = torch.cat([p['label'] for p in predictions], dim=0)
    
    for name in important_organisms:
        current_label = important_organisms[name]
        plt.plot(
            embeddings[labels == current_label, 0],
            embeddings[labels == current_label, 1],
            '.'
        )
    
    plt.show()
    
    with torch.inference_mode():
        plot_bound = 7.0
        n_points = 50
        linspaces = [torch.linspace(-plot_bound, plot_bound, n_points) for _ in range(2)]
        meshgrid = torch.meshgrid(linspaces)
        z_sample = torch.stack(meshgrid).reshape(2, -1).T
        
        samples = [ ]
        for _ in range(5):
            x_out = model.decode(z_sample)
            x_var = D.Categorical(probs=x_out.softmax(dim=1).permute(0, 2, 1)).entropy().sum(dim=-1)
            samples.append(x_var)

        x_var = torch.stack(samples).mean(dim=0)
        x_var_std = torch.stack(samples).std(dim=0)
        x_var_std[torch.isnan(x_var_std)] = 0.0

        plt.contourf(
            z_sample[:, 0].reshape(n_points, n_points),
            z_sample[:, 1].reshape(n_points, n_points),
            x_var.reshape(n_points, n_points).detach().cpu(),
            levels=50,
            zorder=0,
        )
        
        plt.contourf(
            z_sample[:, 0].reshape(n_points, n_points),
            z_sample[:, 1].reshape(n_points, n_points),
            x_var_std.reshape(n_points, n_points).detach().cpu(),
            levels=50,
            zorder=0,
        )
