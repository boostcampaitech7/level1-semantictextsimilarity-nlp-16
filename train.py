import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoModel, AutoTokenizer

import wandb
from data_loader.data_loaders import TextDataLoader
from model.model import STSModel
from utils.preprocessing import preprocessing
from utils.augmentation import augment_data
from utils.util import set_seed


def main():
    ## initialize wandb
    wandb_logger = WandbLogger(
        reinit=True
    )
    ## call configuration from wandb
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
    preprocess = False  # 전처리 데이터 적용시 True로 변경
    data_dir = config["DATA_DIR"]
    train_dir = os.path.join(data_dir, "train.csv")
    dev_dir = os.path.join(data_dir, "dev.csv")

    preprocessed_train_dir = os.path.join(data_dir, "preprocessed_train.csv")
    preprocessed_dev_dir = os.path.join(data_dir, "preprocessed_dev.csv")
    if preprocess == True:
        if os.path.exists(preprocessed_train_dir) and os.path.exists(
            preprocessed_dev_dir
        ):
            print("Loading preprocessed data...")
            train = pd.read_csv(preprocessed_train_dir, dtype={"label": np.float32})
            dev = pd.read_csv(preprocessed_dev_dir, dtype={"label": np.float32})
        else:
            train = pd.read_csv(train_dir, dtype={"label": np.float32})
            dev = pd.read_csv(dev_dir, dtype={"label": np.float32})
            print("Preprocessing train data...")
            train = preprocessing(train)
            print(f"Saving preprocessed train data to {train_dir}")
            train.to_csv(preprocessed_train_dir, index=False)
            print("Preprocessing dev data...")
            dev = preprocessing(dev)
            print(f"Saving preprocessed dev data to {dev_dir}")
            dev.to_csv(preprocessed_dev_dir, index=False)
    else:
        print("Loading data...")
        train = pd.read_csv(train_dir, dtype={"label": np.float32})
        dev = pd.read_csv(dev_dir, dtype={"label": np.float32})

    ## 데이터 증강, 전처리와 동시 적용하려면 증강만 된 데이터를 지우고 전처리를 True로 설정 후 적용
    augment = False # 증강 적용시 True로 설정
    augmented_train_dir = os.path.join(data_dir, "augmented_train.csv")
    augmented_dev_dir = os.path.join(data_dir, "augmented_dev.csv")
    if augment:
        if os.path.exists(augmented_train_dir) and os.path.exists(augmented_dev_dir):
            print("Loading augmented data...")
            train = pd.read_csv(augmented_train_dir, dtype={"label": np.float32})
            dev = pd.read_csv(augmented_dev_dir, dtype={"label": np.float32})
        else:
            print("Augmenting train data...")
            train = augment_data(train)
            print(f"Saving augmented train data to {augmented_train_dir}")
            train.to_csv(augmented_train_dir, index=False)
            print("Augmenting dev data...")
            dev = augment_data(dev)
            print(f"Saving augmented dev data to {augmented_dev_dir}")
            dev.to_csv(augmented_dev_dir, index=False)

    ## 학습 세팅
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    tokens = "<PERSON>"
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
        model,
    )

    ## 매 에포크마다 모델 체크포인트를 로컬에 저장
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{MODEL_NAME}/{current_time}_{wandb.run.id}",
        filename="{epoch:02d}-{val_pearson_corr:.4f}",
        save_top_k=1,
        monitor="val_pearson_corr",
        mode="max",
    )

    ## 얼리스탑 설정
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=EPOCHS,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        val_check_interval=1.0,
    )

    # 학습
    trainer.fit(model, datamodule=dataloader)

    config_dict = dict(config)
    with open("config.json", "w") as f:
        json.dump(config_dict, f)

    artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="model")
    artifact.add_file(checkpoint_callback.best_model_path)
    artifact.add_file("config.json")
    wandb.log_artifact(artifact)


if __name__ == "__main__":
    main()
