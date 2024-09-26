import argparse
import json
import os

import pandas as pd
import torch
from pytorch_lightning import Trainer
from transformers import AutoModel, AutoTokenizer

import wandb
from data_loader.data_loaders import TextDataLoader
from model.model import STSModel
from utils.preprocessing import preprocess_data


def main(arg):
    ## data reading
    test_dir = os.path.join(arg.data_dir, "test.csv")
    submission_dir = os.path.join(arg.data_dir, "sample_submission.csv")

    test = pd.read_csv(test_dir)
    submission = pd.read_csv(submission_dir)

    ## model/config loading
    wandb.login()

    run = wandb.init()
    artifact = run.use_artifact(arg.model_path)
    model_dir = artifact.download()

    with open(f"{model_dir}/config.json", "r") as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config["MODEL_NAME"])
    model = AutoModel.from_pretrained(config["MODEL_NAME"])

    tokens = "<PERSON>"
    tokenizer.add_tokens(tokens)
    model.resize_token_embeddings(len(tokenizer))

    model = STSModel(config, model)
    model.load_state_dict(
        torch.load(f"{model_dir}/{arg.model_name}.ckpt")["state_dict"]
    )

    ## processing
    preprocess = False
    if preprocess == True:
        test = preprocess_data(test)

    test = test.dropna(subset=["sentence_1", "sentence_2"])
    test = test.reset_index(drop=True)

    dataloader = TextDataLoader(
        tokenizer=tokenizer, max_len=config["MAX_LEN"], predict_data=test, truncation=True
    )

    trainer = Trainer(accelerator="gpu")

    preds = trainer.predict(model, dataloader)
    all_pred = [val for pred in preds for val in pred]
    submission["target"] = all_pred

    print(submission.head())
    submission.to_csv("data/submission.csv", index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-d",
        "--data_dir",
        default=None,
        type=str,
        help="directory path for data (default: None)",
    )
    args.add_argument(
        "-m",
        "--model_path",
        default=None,
        type=str,
        help="artifact path for a model (default: all)",
    )
    args.add_argument(
        "-n",
        "--model_name",
        default=None,
        type=str,
        help="name of the model to call (default: all)",
    )

    arg = args.parse_args()
    main(arg)
