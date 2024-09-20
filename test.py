import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

import pandas as pd
from pytorch_lightning import Trainer
from data_loader.data_loaders import TextDataLoader
from utils.tokenzier import get_tokenizer
from utils.util import model_load


def main(model, config):
    test = pd.read_csv("data/test.csv")
    tokenizer = get_tokenizer(config["MODEL_NAME"])
    dataloader = TextDataLoader(
        tokenizer=tokenizer,
        max_len=config["MAX_LEN"],
        test_data=test,
        truncation=True,
        batch_size=config["BATCH_SIZE"],
    )

    trainer = Trainer(accelerator="gpu", devices=1)

    trainer.test(model, dataloader)


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
