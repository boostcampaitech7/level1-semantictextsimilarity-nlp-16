import os

import numpy as np
import pandas as pd
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data_loader.data_loaders import TextDataLoader
from model.model import STSModel
from utils.clean import clean_texts
from utils.tokenizer import get_tokenizer
from utils.util import set_seed


def main():
    ## initialize wandb
    run = wandb.init(project="Level1_STS", entity="kangjun205")
    ## call configuration from wandb
    config = wandb.config

    ## parameters
    EPOCHS = config["EPOCHS"]
    BATCH_SIZE = config["BATCH_SIZE"]
    LEARNING_RATE = config["LEARNING_RATE"]
    MAX_LEN = config["MAX_LEN"]
    LORA_RANK = config['LORA_RANK']
    MODEL_NAME = config["MODEL_NAME"]
    MODULE_NAMES = config["MODULE_NAMES"]

    ## seed setting
    SEED = config["SEED"]
    set_seed(SEED)

    ## data
    data_dir = config['DATA_DIR']
    train_dir = os.path.join(data_dir, 'train_augmented.csv')
    dev_dir = os.path.join(data_dir, 'dev.csv')

    train = pd.read_csv(train_dir, dtype={'label': np.float32})
    dev = pd.read_csv(dev_dir, dtype={'label': np.float32})

    ## preprocessing
    train['sentence_1'] = clean_texts(train['sentence_1'])
    train['sentence_2'] = clean_texts(train['sentence_2'])

    dev['sentence_1'] = clean_texts(dev['sentence_1'])
    dev['sentence_2'] = clean_texts(dev['sentence_2'])

    tokenizer = get_tokenizer(MODEL_NAME)
    dataloader = TextDataLoader(
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        train_data=train,
        dev_data=dev,
        truncation=True,
        batch_size=BATCH_SIZE,
    )
    model = STSModel(
        {
            'MODEL_NAME': MODEL_NAME,
            'LEARNING_RATE': LEARNING_RATE,
            'MAX_LEN': MAX_LEN,
            'LORA_RANK': LORA_RANK,
            'MODULE_NAMES': MODULE_NAMES,
            'SEED': SEED
        }
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="saved",
        filename='{epoch:02d}-{val_pearson_corr:.3f}',
        save_top_k=3,
        monitor="val_pearson_corr",
        mode="max",
    )

    run_name = f'{MODEL_NAME}_{LEARNING_RATE}'
    wandb_logger = WandbLogger(
        name=run_name,
        project="Level1_STS",
        log_model='best'
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=EPOCHS,
        val_check_interval=1.0,
        callbacks=[
            early_stop_callback,
            checkpoint_callback
        ],
        logger = wandb_logger
    )

    trainer.fit(model, datamodule=dataloader)


if __name__ == "__main__":
    main()
