import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer, AutoModel

import wandb
from data_loader.data_loaders import TextDataLoader
from model.model import STSModel
# from utils.preprocessing import preprocessing
from utils.util import set_seed


def main():
    ## call configuration from wandb
    ## initialize wandb
    wandb_logger = WandbLogger(
        log_model='all',
        reinit=True
    )
    config = wandb_logger.experiment.config

    ## parameters
    EPOCHS = config["EPOCHS"]
    BATCH_SIZE = config["BATCH_SIZE"]
    LEARNING_RATE = config["LEARNING_RATE"]
    MAX_LEN = config["MAX_LEN"]
    LORA_RANK = config["LORA_RANK"]
    MODEL_NAME = config["MODEL_NAME"]
    MODULE_NAMES = config["MODULE_NAMES"]

    ## seed setting
    SEED = config["SEED"]
    set_seed(SEED)

    ## load, preprocess data
    data_dir = config["DATA_DIR"]
    train_dir = os.path.join(data_dir, "train.csv")
    dev_dir = os.path.join(data_dir, "dev.csv")

    train = pd.read_csv(train_dir, dtype={'label': np.float32})
    dev = pd.read_csv(dev_dir, dtype={'label': np.float32})

    # preprocessed_train_dir = os.path.join(data_dir, "preprocessed_train.csv")
    # preprocessed_dev_dir = os.path.join(data_dir, "preprocessed_dev.csv")

    # if os.path.exists(preprocessed_train_dir) and os.path.exists(preprocessed_dev_dir):
    #     print("Loading preprocessed files...")
    #     train = pd.read_csv(preprocessed_train_dir, dtype={"label": np.float32})
    #     dev = pd.read_csv(preprocessed_dev_dir, dtype={"label": np.float32})
    # else:
    #     train = pd.read_csv(train_dir, dtype={"label": np.float32})
    #     dev = pd.read_csv(dev_dir, dtype={"label": np.float32})

    #     print("Preprocessing train data...")
    #     train = preprocessing(train)
    #     print(f"Saving preprocessed train data to {preprocessed_train_dir}")
    #     train.to_csv(preprocessed_train_dir, index=False)
    #     print("Preprocessing dev data...")
    #     dev = preprocessing(dev)
    #     print(f"Saving preprocessed dev data to {preprocessed_dev_dir}")
    #     dev.to_csv(preprocessed_dev_dir, index=False)

    ## 학습 세팅
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    tokens = '<PERSON>'
    tokenizer.add_tokens(tokens)
    model.resize_token_embeddings(len(tokenizer))

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
            "MODEL_NAME": MODEL_NAME,
            "LEARNING_RATE": LEARNING_RATE,
            "MAX_LEN": MAX_LEN,
            "LORA_RANK": LORA_RANK,
            "MODULE_NAMES": MODULE_NAMES,
            "SEED": SEED,
        },
        model
    )

    ## 매 에포크마다 모델 체크포인트를 로컬에 저장
    current_datetime = datetime.now().strftime("%y%m%d_%H%M%S")
    checkpoint_callback = ModelCheckpoint(
        dirpath="saved",
        filename="{epoch:02d}-{val_pearson_corr:.3f}",
        save_top_k=3,
        monitor="val_pearson_corr",
        mode="max",
    )

    ## 얼리스탑 설정
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )

    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=EPOCHS,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        val_check_interval=1.0,
    )

    trainer.fit(model, datamodule=dataloader)

    # artifact = wandb_logger.experiment.wandb.Artifact(
    #     name=f"model-{wandb_logger.experiment.id}", 
    #     type="model"
    # )
    # artifact.add_file(checkpoint_callback.best_model_path)
    # wandb_logger.experiment.log_artifact(artifact)


if __name__ == "__main__":
    main()
