import os
from argparse import ArgumentParser

import pytorch_lightning as pl

from data import get_data
from models import get_model


def train(args):

    # Initialize model
    model_class = get_model(args.model)
    model = model_class(**vars(args))

    # Initialize data
    datamodule_class = get_data(args.dataset)
    datamodule = datamodule_class(**vars(args))


    # Initialize callbacks
    callbacks = [ ]
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath="checkpoints", monitor="val_loss", mode="min"
        )
    )
    callbacks.append(
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=args.patience,
            verbose=True
        )
    )
    callbacks.append(pl.callbacks.LearningRateMonitor())
    if args.gpus:
        callbacks.append(pl.callbacks.GPUStatsMonitor())

    # Initialize trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=pl.loggers.WandbLogger(project="calibrated_vae"),
        callbacks=callbacks
    )

    # Train and test!
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    # Common arguments
    parser = ArgumentParser()
    parser.add_argument("model", type=str, default="")
    parser.add_argument("--dataset", type=str, default="small_mnist")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--kl_warmup_steps", type=int, default=1000)
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=20)

    # Model specific arguments
    parser.add_argument("--prob", type=float, default=0.05)
    parser.add_argument("--mc_samples", type=int, default=10)
    parser.add_argument("--only_decoder_mc", type=bool, default=False)

    parser.add_argument("--n_ensemble", type=int, default=5)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    train(args)
