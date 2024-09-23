import os
import argparse

import pandas as pd
import torch
import wandb
from pytorch_lightning import Trainer

from data_loader.data_loaders import TextDataLoader
from model.model import STSModel
from utils.preprocessing import preprocessing
from utils.tokenizer import get_tokenizer
from utils.util import model_load
from utils.clean import clean_texts


def main(model, config):

    ## data reading
    test = pd.read_csv("data/test.csv") ## data_dir arg 설정
    submission = pd.read_csv('data/sample_submission.csv') ## data_dir

    ## model/config loading
    wandb.login()

    api = wandb.Api()
    run = api.run("kangjun205/Level1_STS/dlyeghmc") ## run path argument로 설정

    artifact = api.artifact('kangjun205/Level1_STS/model-q4x8581k:v14')
    model_dir = artifact.download()
    config = run.config

    model = STSModel.load_from_checkpoint(f'{model_dir}/model.ckpt')

    ## processing
    test['sentence_1'] = clean_texts(test['sentence_1'])
    test['sentence_2'] = clean_texts(test['sentence_2'])

    tokenizer = get_tokenizer(config['MODEL_NAME'])
    dataloader = TextDataLoader(
        tokenizer=tokenizer,
        max_len=config['MAX_LEN'],
        test_data=test,
        truncation=True
    )
        
    trainer = Trainer(
        accelerator="gpu",
        devices=1
    )

    preds = trainer.predict(model, dataloader)
    all_pred = [val for pred in preds for val in pred]

    submission['target'] = all_pred
    print(submission.head())

    submission.to_csv('data/submission.csv', index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    model, config = model_load("run_name", "model_path")
    main(model, config)
