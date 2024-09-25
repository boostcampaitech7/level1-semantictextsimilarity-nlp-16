import json
import os
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
from utils.augmentation import augment_data
from utils.preprocessing import preprocess_data
from utils.util import set_seed


def main():
    ## initialize wandb
    wandb_logger = WandbLogger(reinit=True)
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

    ## load data
    data_dir = config["DATA_DIR"]
    train_dir = os.path.join(data_dir, "train.csv")
    dev_dir = os.path.join(data_dir, "dev.csv")
    train = pd.read_csv(train_dir, dtype={"label": np.float32})
    dev = pd.read_csv(dev_dir, dtype={"label": np.float32})

    ## 이상치 행 삭제
    filtered_ids = [757, 847, 924, 925, 926, 938, 952, 975, 1031, 1066, 
		1117, 1119, 1122, 1163, 1180, 1191, 1193, 1203, 1204, 1223, 1233,
		1254, 1275, 1301, 1308, 1312, 1321, 1359, 1376, 1379, 1388, 1417,
		1421, 1458, 1494, 1522, 1533, 1544, 1548, 1610, 1615, 1613, 1618,
		1627, 1628, 1649, 1674, 1694, 1711, 1726, 1731, 1741, 1764, 1791,
		1805, 1809, 1818, 1871, 1872, 1881, 1913, 1919, 1920, 1923, 1946,
		1949, 1957, 1960, 1969, 1975, 1997, 2871, 2585, 3044, 3147, 4713,
		4811, 4818, 4868, 4873, 4923, 4946, 4958, 5006, 5041, 5073, 5113,
		5120, 5190, 5236, 5257, 5293, 5297, 5354, 5355, 5360, 5365, 5404,
		5405, 5408, 5465, 5495, 5501, 5524, 5538, 5587, 5591, 5594, 5643,
		5689, 5691, 5700, 5753, 5762, 5829, 5862, 5894, 5896, 5897, 5941,
		5960, 5971, 5986, 6059, 6145, 6232, 6255, 6559, 6581, 6622, 6701,
		6898, 6907, 7231, 7378, 7396, 8981, 9318, 9026, 9112, 9122, 8140,
		8138]
    train.drop(index=filtered_ids, inplace=True)
    train.reset_index(drop=True, inplace=True)

    ## 데이터 전처리
    preprocess = False  # 전처리 데이터 적용시 True로 변경
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
            print("Preprocessing train data...")
            train = preprocess_data(train)
            print(f"Saving preprocessed train data to {preprocessed_train_dir}")
            train.to_csv(preprocessed_train_dir, index=False)
            print("Preprocessing dev data...")
            dev = preprocess_data(dev)
            print(f"Saving preprocessed dev data to {preprocessed_dev_dir}")
            dev.to_csv(preprocessed_dev_dir, index=False)
    else:
        print("Loading raw data...")
        train = pd.read_csv(train_dir, dtype={"label": np.float32})
        dev = pd.read_csv(dev_dir, dtype={"label": np.float32})

    ## 데이터 증강과 전처리를 동시 적용할 경우, augmented 데이터 삭제 후 preprocess를 True로 설정 후 적용
    augment = False  # 증강 적용시 True로 설정
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
