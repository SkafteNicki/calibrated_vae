from argparse import ArgumentParser

import pytorch_lightning as pl
from data import get_data
from models import get_model
from pytorch_lightning.core import datamodule
import os
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import logging
log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train")
def train(config: DictConfig):
    log.info(f"Working directory : {os.getcwd()}")
    log.info(f"Config {config}")

    # Initialize model
    model_class = get_model(config.model.name)
    model = model_class(**config.model)

    # Initialize data
    config.dataset.data_dir = get_original_cwd()
    datamodule_class = get_data(config.dataset.name)
    datamodule = datamodule_class(**config.dataset)
    
    checkpointer = pl.callbacks.ModelCheckpoint(
        dirpath=f"{os.getcwd()}/checkpoints/", monitor="val_loss", mode="min"
    )

    stopper = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=config.early_stopping_patience,
    )

    trainer = pl.Trainer(
        gpus=config.gpus,
        logger=pl.loggers.WandbLogger(project="calibrated_vae"),
        callbacks=[
                checkpointer,
                stopper,
                pl.callbacks.LearningRateMonitor(),            
        ] + [pl.callbacks.GPUStatsMonitor()] if config.gpus else [],
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)



if __name__ == "__main__":
    train()
