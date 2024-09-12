import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device


import datetime

import numpy as np
import pandas as pd
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utils.tokenizer import get_tokenizer
from data_loader.data_loaders import TextDataLoader
from utils.util import set_seed
from model.model import STSModel
from utils.util import WandbCheckpointCallback

def main(config):
    
    ## data
    train = pd.read_csv('data/train.csv')
    dev = pd.read_csv('data/dev.csv')

    tokenizer = get_tokenizer(config['MODEL_NAME'])
    dataloader = TextDataLoader(
        tokenizer=tokenizer,
        max_len=config['MAX_LEN'],
        train_data=train,
        dev_data=dev,
        truncation=True,
        batch_size=config['BATCH_SIZE']
    )
    model = STSModel(config)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath='saved',
        filename=f'best-model-{datetime.datetime.now().strftime('%d%H%M')}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )

    wandb_checkpoint_callback = WandbCheckpointCallback(top_k=3)

    run_name = f'{config['MODEL_NAME']}-{datetime.datetime.now().strftime('%d%H%M')}'
    wandb_logger = WandbLogger(name = run_name, project="Level1-STS")

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config['EPOCH'],
        log_every_n_steps=1,
        callbacks=[early_stop_callback, checkpoint_callback, wandb_checkpoint_callback],
        logger = wandb_logger
        )
    
    trainer.fit(model, dataloader)
    trainer.validate(model, dataloader)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    SEED = 42
    set_seed(SEED)

    main(wandb.config)
