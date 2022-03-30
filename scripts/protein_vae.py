from torch import nn, distributions as D
import pytorch_lightning as pl
import torch

from scr.data import get_dataset

class VAE(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(62208, 1500),
            nn.ReLU(),
            nn.Linear(1500, 1500),
            nn.ReLU()
        )

        self.encoder_mu = nn.Linear(1500, 2)
        self.encoder_scale = nn.Sequential(nn.Linear(1500, 2), nn.Softplus())

        self.decoder = nn.Sequential(
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.Linear(100, 500),
            nn.ReLU(),
            nn.Linear(500, 62208)
        )

        self.loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=22)
        self._prior = None

    @property
    def prior(self):
        if self._prior is None:
            self._prior =  D.Independent(D.Normal(torch.zeros(2).to(self.device),
                                                  torch.ones(2).to(self.device)), 1)
        return self._prior

    def decode(self, z):
        recon = self.decoder(z).reshape(*z.shape[:-1], 24, -1)
        return recon

    def forward(self, x):
        x = nn.functional.one_hot(x, 24)
        h = self.encoder(x.float().reshape(x.shape[0], -1))

        q_dist = D.Independent(
            D.Normal(
                self.encoder_mu(h),
                self.encoder_scale(h) + 1e-4
            ), 
            1
        )
        z = q_dist.rsample()

        recon = self.decode(z)
        return recon, q_dist

    def _step(self, batch, batch_idx):
        x = batch[0].long()
        recon, q_dist = self(x)
        recon_loss = self.loss_fn(recon, x).sum(dim=-1).mean()
        kl_loss = D.kl_divergence(q_dist, self.prior).mean()
        loss = recon_loss + kl_loss
        acc = (recon.argmax(dim=1) == x)[x!=22].float().mean()
        return loss, recon_loss, kl_loss, acc

    def training_step(self, batch, batch_idx):
        loss, recon_loss, kl_loss, acc = self._step(batch, batch_idx)

        self.log_dict({'train_loss': loss,
                       'train_recon': recon_loss,
                       'train_kl': kl_loss,
                       'train_acc': acc},
                      prog_bar=True,
                      logger=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

if __name__ == "__main__":
    train, val, test = get_dataset('protein')

    train = torch.utils.data.DataLoader(train, batch_size=16)

    model = VAE()

    trainer = pl.Trainer(
        logger=pl.loggers.WandbLogger(),
        accelerator="auto",
        num_sanity_val_steps=0,
        devices=1,
        max_epochs=20
    )
    trainer.fit(model, train_dataloaders=train)