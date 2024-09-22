import argparse
import collections

import numpy as np
import pandas as pd
import torch
import wandb
# from parse_config import ConfigParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data_loader.data_loaders import TextDataLoader
from model.model import STSModel
from utils.preprocessing import preprocessing
from utils.tokenizer import get_tokenizer
from utils.util import WandbCheckpointCallback, set_seed


def main(args):
    ## initialize wandb
    run = wandb.init()
    ## call configuration from wandb
    config = wandb.config

    ## parameters
    EPOCHS = config["EPOCHS"]
    BATCH_SIZE = config["BATCH_SIZE"]
    LEARNING_RATE = config["LEARNING_RATE"]
    MAX_LEN = config["MAX_LEN"]

    PATH = args.path
    MODEL_NAME = args.model_name
    SEED = args.seed

    ## seed setting
    set_seed(SEED)

    ## data
    train = pd.read_csv("data/train.csv", dtype={'label': np.float32})
    dev = pd.read_csv("data/dev.csv", dtype={'label': np.float32})

    train = preprocessing(train)
    dev = preprocessing(dev)

    tokenizer = get_tokenizer(MODEL_NAME)
    dataloader = TextDataLoader(
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        train_data=train,
        dev_data=dev,
        truncation=True,
        batch_size=BATCH_SIZE,
    )
    model = STSModel(config)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="saved",
        filename=f'{epoch:02d}-{int(val_pearson_corr*1000):04d}',
        save_top_k=3,
        monitor="val_pearson_corr",
        mode="min",
    )

    wandb_checkpoint_callback = WandbCheckpointCallback(top_k=3)

    run_name = f'{MODEL_NAME}_{LEARNING_RATE}'
    wandb_logger = WandbLogger(name=run_name, project="Level1-STS")

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=EPOCHS,
        val_check_interval=1,
        callbacks=[
            early_stop_callback,
            checkpoint_callback,
            wandb_checkpoint_callback
        ],
        logger = wandb_logger
    )

    trainer.fit(model, datamodule=dataloader)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-p",
        "--path",
        default="./data",
        type=str,
        help="config data path (default: ./data)",
    )
    args.add_argument(
        "-m",
        "--model_name",
        default=None,
        type=str,
        help="what models to call (default: None)",
    )
    args.add_argument(
        "-s",
        "--seed",
        default=12345,
        type=int,
        help="give seed number for experiment (default: 12345)"
    )

    main(args)
