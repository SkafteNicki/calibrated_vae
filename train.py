import pytorch_lightning as pl
from pytorch_lightning.core import datamodule
from models import VAE, MCVAE, EVAE, NVAE
from data.datamodules import MnistDatamodule
from argparse import ArgumentParser


if __name__ == "__main__":
    # Argument parsing
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--labels_to_use", nargs='+', type=int, default=[0,1,2,3,4,5,6,7,8,9])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--kl_warmup_steps", type=int, default=1)
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--prob", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--n_ensemble", type=int, default=5)
    parser.add_argument("--ensemble_only_decoder", type=bool, default=False)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if args.model == 'VAE':
        model_class = VAE
    elif args.model == 'MCVAE':
        model_class = MCVAE
    elif args.model == 'EVAE':
        model_class = EVAE
    elif args.model == 'NVAE':
        model_class = NVAE

    datamodule = MnistDatamodule(args.data_dir, args.labels_to_use)

    checkpointer = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/", monitor="val_loss", mode="min"
    )
    stopper = pl.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=args.patience,
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=pl.loggers.WandbLogger(project='calibrated_vae'),
        callbacks=[
            checkpointer, 
            stopper, 
            pl.callbacks.LearningRateMonitor(),
        ],
    )

    model = model_class(**vars(args))

    trainer.fit(model, datamodule=datamodule)